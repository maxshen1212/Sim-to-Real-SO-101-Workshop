# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================
# 雙臂任務：左右各取自己半邊的試管 → 放進中央共用試管架（Phase 4）
#
# 繼承雙臂任務視覺層（so101_dual_task_env_cfg.py，已含三相機+燈箱+墊子），加上：
#   - 4 支試管：左 2（y>0，左臂負責）、右 2（y<0，右臂負責）
#   - 1 個中央試管架 rack_center（y≈0，兩臂共用，4 槽）
#   - 兩個夾爪接觸感測器：contact_grasp_left / _right（各 filter 自己半邊的試管）
#   - reset：把試管放回左右兩側、架子放中央
#   - 觀測：左右各自的「抓取 / 放置」subtask
#
# 註：grasp/placed 判定函式已重構成 per-sensor 狀態（terms.py），左右臂不互相干擾。
#     termination（eval 成功判定）涉及雙臂共享狀態，留待 Phase 6 處理；
#     此檔為遙操作錄製用的 base（無 terminations）。
#
# 佈局：架子 body 置中於墊子中心 (world 0.22, 0)；試管放墊子左右半邊（在墊子上即可達）。
# 墊子範圍 world x[0.068,0.372] y[-0.229,0.229]。DR 隨機量之後再討論。
# ============================================================

import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import (
    SO101_DUAL_LEFT_CONTACT_CFG,
    SO101_DUAL_RIGHT_CONTACT_CFG,
)
from sim_to_real_so101.mdp import (
    reset_vials_rack,
    any_vial_grasped,
    vial_placed_on_rack,
)

from .so101_dual_task_env_cfg import (
    SO101DualTaskSceneCfg,
    SO101DualTaskEnvCfg,
    DualTaskEventCfg,
    DualTaskObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

VIAL_SPAWN_Z = 0.05

# 試管模板（透明玻璃，輕、高角阻尼避免亂滾）
_vial = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Vial",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{assets_path}/usd/Vial_opaque.usda",
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, VIAL_SPAWN_Z)),
)

# 中央試管架模板
_rack = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Rack",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{assets_path}/usd/Vial_rack_simple.usda",
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.0, 0.06)),
)

# 直立朝向（USD 預設橫躺，繞 Y 轉 90° 站起來）
_UPRIGHT = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)


def _vial_at(prim_name: str, x: float, y: float) -> RigidObjectCfg:
    v = _vial.replace()
    v.prim_path = f"{{ENV_REGEX_NS}}/{prim_name}"
    v.init_state.pos = (x, y, VIAL_SPAWN_Z)
    v.init_state.rot = _UPRIGHT
    return v


# 左右半邊試管名稱（給 contact sensor / obs 用）
VIALS_LEFT = ["Vial_Left_1", "Vial_Left_2"]
VIALS_RIGHT = ["Vial_Right_1", "Vial_Right_2"]


# ============================================================
# 場景：雙臂(帶接觸感測) + 4 試管 + 中央架 + 兩接觸感測器
# ============================================================
@configclass
class SO101DualVialsSceneCfg(SO101DualTaskSceneCfg):

    # 換成帶接觸感測的雙臂
    robot_left: ArticulationCfg = SO101_DUAL_LEFT_CONTACT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_Left"
    )
    robot_right: ArticulationCfg = SO101_DUAL_RIGHT_CONTACT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_Right"
    )

    # ── 左半邊 2 支（左臂負責，y>0）──
    # x=0.18：比墊子中線(0.22)略靠近機器人，落在墊子近半邊，且比架子(0.22)近，
    # 形成「前面拿試管 → 放後面中央架」的動線；避免和架子擠在同一條深度線上。
    vial_left_1 = _vial_at("Vial_Left_1", 0.18, 0.10)
    vial_left_2 = _vial_at("Vial_Left_2", 0.18, 0.18)
    # ── 右半邊 2 支（右臂負責，y<0）──
    vial_right_1 = _vial_at("Vial_Right_1", 0.18, -0.10)
    vial_right_2 = _vial_at("Vial_Right_2", 0.18, -0.18)

    # ── 中央試管架（兩臂共用，4 槽）──
    # 架子原點在角落、body 中心在本地 (0.06, 0.06)；放原點 (0.16, -0.06) 讓 body 中心
    # 正對墊子中心 (0.22, 0)。4 槽會橫跨 world x[0.19,0.25] y[-0.03,0.03]，置中於墊子。
    rack_center = _rack.replace()
    rack_center.prim_path = "{ENV_REGEX_NS}/Rack_Center"
    rack_center.init_state.pos = (0.16, -0.06, 0.06)

    # ── 接觸感測器：各臂 jaw，filter 自己半邊的試管 ──
    contact_grasp_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Left/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Vial_Left_1",
            "{ENV_REGEX_NS}/Vial_Left_2",
        ],
    )
    contact_grasp_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Right/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Vial_Right_1",
            "{ENV_REGEX_NS}/Vial_Right_2",
        ],
    )


# ============================================================
# 事件：reset 時把 4 試管放回左右、架子放中央
# ============================================================
@configclass
class SO101DualVialsEventCfg(DualTaskEventCfg):

    reset_vials_setup = EventTerm(
        func=reset_vials_rack,
        mode="reset",
        params={
            "vials": ["vial_left_1", "vial_left_2", "vial_right_1", "vial_right_2"],
            "rack": "rack_center",
            "rack_pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.01, 0.01),
                "yaw": (-0.2, 0.2),
            },
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.01, 0.01),
                "roll": (-0.3, 0.3),
                "yaw": (0.0, 0.0),
            },
            "fixed_vial_z": VIAL_SPAWN_Z,
            "rack_placement_prob": 0.0,  # 雙臂分側，先不要預先擺一支在架上
        },
    )


# ============================================================
# 觀測：左右各自的 抓取 / 放置 subtask
# ============================================================
@configclass
class SO101DualVialsObservationsCfg(DualTaskObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        # ── 左臂 ──
        vial_grasped_left = ObsTerm(
            func=any_vial_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_left"),
                "vials": ["vial_left_1", "vial_left_2"],
                "min_height": 0.055,
                "warmup_steps": 30,
                "force_threshold": 2,
            },
        )
        vial_placed_left = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_left"),
                "vials": ["vial_left_1", "vial_left_2"],
                "rack_name": "rack_center",
                "warmup_steps": 30,
                "grasp_history_window": 20,
                "force_threshold": 2,
                "rack_local_x_min": 0.0,
                "rack_local_x_max": 0.12,
                "rack_local_y_min": 0.0,
                "rack_local_y_max": 0.12,
                "rack_local_z_max": 0.1,
                "vertical_threshold": 0.7,
            },
        )
        # ── 右臂 ──
        vial_grasped_right = ObsTerm(
            func=any_vial_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_right"),
                "vials": ["vial_right_1", "vial_right_2"],
                "min_height": 0.055,
                "warmup_steps": 30,
                "force_threshold": 2,
            },
        )
        vial_placed_right = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_right"),
                "vials": ["vial_right_1", "vial_right_2"],
                "rack_name": "rack_center",
                "warmup_steps": 30,
                "grasp_history_window": 20,
                "force_threshold": 2,
                "rack_local_x_min": 0.0,
                "rack_local_x_max": 0.12,
                "rack_local_y_min": 0.0,
                "rack_local_y_max": 0.12,
                "rack_local_z_max": 0.1,
                "vertical_threshold": 0.7,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


# ============================================================
# 完整環境（遙操作錄製用 base，無 terminations）
# ============================================================
@configclass
class SO101DualVialsEnvCfg(SO101DualTaskEnvCfg):
    scene: SO101DualVialsSceneCfg = SO101DualVialsSceneCfg()
    events: SO101DualVialsEventCfg = SO101DualVialsEventCfg()
    observations: SO101DualVialsObservationsCfg = SO101DualVialsObservationsCfg()
