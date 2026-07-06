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
# 協作式雙臂任務：任一臂拿任一支試管 → 放進中央共用試管架（Phase 4A）
#
# 繼承雙臂任務視覺層（so101_dual_task_env_cfg.py，已含三相機+燈箱+墊子），加上：
#   - 4 支試管：散佈整個墊子（不綁左右分邊，任一臂都可拿任一支）
#   - 1 個中央試管架 rack_center（y≈0，兩臂共用，4 槽）
#   - 兩個夾爪接觸感測器：contact_grasp_left / _right（各自 filter 全部 4 支）
#   - reset：把 4 支試管打散在墊子近側、架子放中央
#   - 觀測：左右臂各自的「抓取 / 放置」subtask，皆對全部 4 支試管判定
#
# 註：grasp/placed 判定函式已重構成 per-sensor 狀態（terms.py），左右臂不互相干擾。
#     兩個 obs 的 vials 順序必須和對應 contact sensor 的 filter_prim_paths_expr
#     順序一致（terms.py 以 index 對齊 filter 與 vial）。
#     termination（eval 成功判定）涉及雙臂共享狀態，留待 Phase 6 處理；
#     此檔為遙操作錄製用的 base（無 terminations）。
#
# 佈局：架子 body 置中於墊子中心 (world 0.22, 0)；試管散在墊子近側（x<架子，在墊子上即可達）。
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


# 全部 4 支試管（協作式：兩個 contact sensor 與兩個 obs 都用全部 4 支）。
# 兩份清單順序必須一致：contact filter 用 prim 名、obs/event 用 config 屬性名，
# terms.py 以 index 對齊 filter 與 vial，順序錯位會導致抓取判定張冠李戴。
VIAL_PRIMS_ALL = ["Vial_Left_1", "Vial_Left_2", "Vial_Right_1", "Vial_Right_2"]
VIALS_ALL = ["vial_left_1", "vial_left_2", "vial_right_1", "vial_right_2"]


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

    # ── 4 支試管散佈整個墊子近側（不綁左右分邊）──
    # 名稱沿用 *_Left_*/*_Right_* 只是內部識別，實際位置打散；任一臂可拿任一支。
    # 架子 body 佔 world x[0.16,0.28] y[-0.06,0.06]，避免試管(含 reset jitter ±0.03)壓到架子：
    #   - 中間兩支(|y|≈0.07)放低 x=0.12(max 0.15 < 0.16)→ 永遠在架子前方。
    #   - 外側兩支(|y|≈0.17)可放高 x=0.16，因為 |y| 遠離架子 y 帶。
    # 形成「前面撿散落試管 → 放後面中央架」的動線，避免和架子擠在同一條深度線上。
    vial_left_1 = _vial_at("Vial_Left_1", 0.16, 0.17)
    vial_left_2 = _vial_at("Vial_Left_2", 0.12, 0.07)
    vial_right_1 = _vial_at("Vial_Right_1", 0.12, -0.07)
    vial_right_2 = _vial_at("Vial_Right_2", 0.16, -0.17)

    # ── 中央試管架（兩臂共用，4 槽）──
    # 架子原點在角落、body 中心在本地 (0.06, 0.06)；放原點 (0.16, -0.06) 讓 body 中心
    # 正對墊子中心 (0.22, 0)。4 槽會橫跨 world x[0.19,0.25] y[-0.03,0.03]，置中於墊子。
    rack_center = _rack.replace()
    rack_center.prim_path = "{ENV_REGEX_NS}/Rack_Center"
    rack_center.init_state.pos = (0.16, -0.06, 0.06)

    # ── 接觸感測器：各臂 jaw，兩邊都 filter 全部 4 支（協作式：任一臂可碰任一支）──
    # filter 順序 = VIAL_PRIMS_ALL，須與 obs 傳入的 VIALS_ALL 順序一致。
    contact_grasp_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Left/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/" + p for p in VIAL_PRIMS_ALL
        ],
    )
    contact_grasp_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Right/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/" + p for p in VIAL_PRIMS_ALL
        ],
    )


# ============================================================
# 事件：reset 時把 4 試管打散在墊子近側、架子放中央
# ============================================================
@configclass
class SO101DualVialsEventCfg(DualTaskEventCfg):

    reset_vials_setup = EventTerm(
        func=reset_vials_rack,
        mode="reset",
        params={
            "vials": VIALS_ALL,
            "rack": "rack_center",
            "rack_pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.01, 0.01),
                "yaw": (-0.2, 0.2),
            },
            # 加大 jitter 讓 4 支散佈墊子近側（不再固定成一條線）。
            # y ±0.03 仍保持基準間距 ≥0.10 → 相鄰試管最小間隙 ~0.04，不會互穿。
            # x ±0.03：中間兩支 base 0.12→max 0.15（<架子 0.16），外側兩支 |y| 夠遠，皆不壓架子。
            # 旋轉沿用單臂驗證過的 roll 微擾；yaw 維持 0（試管已繞 Y 立起，大 yaw delta 會變成傾倒而非自轉）。
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "roll": (-0.3, 0.3),
                "yaw": (0.0, 0.0),
            },
            "fixed_vial_z": VIAL_SPAWN_Z,
            "rack_placement_prob": 0.0,  # 遙操作錄製：先不要預先擺一支在架上
        },
    )


# ============================================================
# 觀測：左右各自的 抓取 / 放置 subtask
# ============================================================
@configclass
class SO101DualVialsObservationsCfg(DualTaskObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        # 協作式：左右兩臂的 subtask 都對「全部 4 支」判定（vials 順序 = 對應
        # contact sensor 的 filter 順序 = VIAL_PRIMS_ALL）。
        # ── 左臂 ──
        vial_grasped_left = ObsTerm(
            func=any_vial_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_left"),
                "vials": VIALS_ALL,
                "min_height": 0.055,
                "warmup_steps": 30,
                "force_threshold": 2,
            },
        )
        vial_placed_left = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_left"),
                "vials": VIALS_ALL,
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
                "vials": VIALS_ALL,
                "min_height": 0.055,
                "warmup_steps": 30,
                "force_threshold": 2,
            },
        )
        vial_placed_right = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp_right"),
                "vials": VIALS_ALL,
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
