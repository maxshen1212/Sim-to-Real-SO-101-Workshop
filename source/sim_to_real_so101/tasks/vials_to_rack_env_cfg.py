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
# 第三層環境設定（操作任務層）── "Vials to Rack"
#
# 繼承第二層（task_env_cfg.py），在帶相機的視覺場景上新增：
#   - 3 支試管（Vial）的 RigidObject
#   - 1 個試管架（Rack）的 RigidObject
#   - 夾爪接觸感測器（ContactSensor）用於抓取偵測
#   - reset 事件：隨機化試管和架子的擺放位置
#   - 觀測：subtask 進度（已抓取？已放置？）
#   - 終止條件：放置成功 / 時間超限
#
# 此外衍生出四個變體：
#   VialsToRackEnvCfg       — 基礎版（遙操作錄製用）
#   VialsToRackDREnvCfg     — DR 版（額外隨機化機器人顏色、天空光、墊子旋轉）
#   VialsToRackEvalEnvCfg   — 評估版（加上終止條件，episode 限 7.5 分鐘）
#   VialsToRackEvalDREnvCfg — DR 評估版（DR + 終止條件）
# ============================================================

import os
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    reset_vials_rack,
    randomize_sky_light,
    ROBOT_COLORS,
    randomize_mat_rotation,
    randomize_robot_color,
    any_vial_grasped,
    vial_placed_on_rack,
    vial_placed_on_rack_termination,
    time_out,
)

from .so101_env_cfg import EventCfg
from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
)

# assets 資料夾的絕對路徑，用來拼接 USD 路徑
assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# ============================================================
# 操作物件模板（試管 & 架子）
# RigidObjectCfg 定義「可被物理引擎模擬的剛體」
# 先定義共用基底，再用 .replace() 複製並覆寫差異
# ============================================================
manipulation_object_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ManipulationObject",  # 佔位，後面會覆寫
    spawn=sim_utils.UsdFileCfg(usd_path=""),        # USD 路徑也會被覆寫
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),  # 預設高度 6 cm（離地面略高，避免初始碰撞）
    ),
)

# ── 試管設定 ──
vial = manipulation_object_base.replace()
vial.spawn.usd_path = f"{assets_path}/usd/Vial_opaque.usda"
vial.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.02)          # 20 g
vial.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0)  # 高角阻尼，讓試管不會滾來滾去

# ── 架子設定 ──
rack = manipulation_object_base.replace()
rack.prim_path = "{ENV_REGEX_NS}/VialRack"
rack.spawn.usd_path = f"{assets_path}/usd/Vial_rack_simple.usda"
rack.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.2)           # 200 g，比試管重很多，不容易被碰飛

# 試管的 mass/rigid_props 需要再設一次（因為 .replace() 只做淺拷貝，上面的 rack 操作不影響 vial）
vial.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.02)
vial.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0)

# 試管 reset 時的 Z 高度（公尺）；設為 0.05 讓試管站立在桌面上
VIAL_SPAWN_Z = 0.05


# ============================================================
# 場景設定（Scene）── 第三層
# 在任務視覺場景上加入操作物件
# ============================================================
@configclass
class VialsToRackSceneCfg(SO101TaskSceneCfg):

    # 換成帶接觸感測器的機器人版本（S0101_CONTACT_GRASP_CFG），
    # 這樣才能偵測夾爪碰到試管的接觸力
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── 三支試管 ──
    # 各自指定不同的初始位置，呈一排擺在機器人前方
    # rot = [0, 90, 0]（度）讓試管直立（預設 USD 是橫躺的）
    vial_1 = vial.replace()
    vial_1.prim_path = "{ENV_REGEX_NS}/Vial_1"
    vial_1.init_state.pos = (0.23, -0.08, VIAL_SPAWN_Z)
    vial_1.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    vial_2 = vial.replace()
    vial_2.prim_path = "{ENV_REGEX_NS}/Vial_2"
    vial_2.init_state.pos = (0.23, 0, VIAL_SPAWN_Z)
    vial_2.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    vial_3 = vial.replace()
    vial_3.prim_path = "{ENV_REGEX_NS}/Vial_3"
    vial_3.init_state.pos = (0.23, -0.16, VIAL_SPAWN_Z)
    vial_3.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    # ── 試管架 ──
    rack_left = rack.replace()
    rack_left.prim_path = "{ENV_REGEX_NS}/Rack_Left"
    rack_left.init_state.pos = (0.18, 0.08, 0.06)  # 擺在試管左側

    # ── 夾爪接觸感測器 ──
    # 偵測 jaw（夾爪末端）對三支試管的接觸力
    # filter_prim_paths_expr 只過濾「jaw 接觸到試管」的事件，排除其他碰撞
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,      # 每步都更新
        history_length=1,       # 只保留最近 1 步的接觸歷史
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Vial_1",
            "{ENV_REGEX_NS}/Vial_2",
            "{ENV_REGEX_NS}/Vial_3",
        ],
    )


# ============================================================
# DR 版場景 ── 額外加入可隨機化的天空光（HDRI）
# ============================================================
@configclass
class VialsToRackDRSceneCfg(VialsToRackSceneCfg):

    # DomeLightCfg 是全方向環境光（天空光），用 HDRI 貼圖模擬不同時段的光線
    # DR 時會在 reset 事件中替換 HDRI 貼圖並隨機調整色溫 / 亮度
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",   # 注意：天空光掛在 /World 下（全域），不在 ENV_REGEX_NS 內
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",  # 預設 HDRI（reset 時會被換掉）
            visible_in_primary_ray=False,   # 不讓天空光本身出現在相機畫面中
            enable_color_temperature=True,
            color_temperature=6500.0,       # 預設 6500K（日光）
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()


# ============================================================
# 事件設定（Events）── 第三層
# 繼承視覺 DR 事件，新增試管 & 架子位置的 reset 邏輯
# ============================================================
@configclass
class VialsToRackEventCfg(TaskEventCfg):
    """Configuration for events."""

    # ── 試管和架子的 reset 事件 ──
    # 每次 reset 時把試管和架子放回初始位置，並套用小幅隨機偏移
    # 這樣 policy 就必須學會在不同擺放位置下都能完成任務
    reset_vials_setup = EventTerm(
        func=reset_vials_rack,
        mode="reset",
        params={
            "vials": ["vial_1", "vial_2", "vial_3"],   # 要 reset 的試管名稱列表
            "rack": "rack_left",                         # 要 reset 的架子名稱
            # 架子位置的隨機範圍（相對於 init_state）
            "rack_pose_range": {
                "x":   (-0.04, 0.04),   # ±4 cm 前後
                "y":   (-0.01, 0.01),   # ±1 cm 左右
                "yaw": (-0.5, 0.5),     # ±0.5 rad ≈ ±28.6° 旋轉
            },
            # 試管位置的隨機範圍
            "pose_range": {
                "x":    (-0.04, 0.04),
                "y":    (-0.01, 0.01),
                "roll": (-0.3, 0.3),    # 小幅傾斜（試管站立不完全垂直）
                "yaw":  (0.0, 0.0),     # yaw 不隨機（保持朝向一致）
            },
            "fixed_vial_z": 0.05,       # 固定試管 Z 高度，確保不穿地板
        },
    )


# ============================================================
# DR 版事件 ── 比基礎版多了機器人顏色、天空光、墊子旋轉的隨機化
# ============================================================
@configclass
class VialsToRackEventDRCfg(VialsToRackEventCfg):

    # 隨機化機器人外觀顏色（從 ROBOT_COLORS 調色盤隨機挑一個）
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": list(ROBOT_COLORS.keys()),  # 覆寫基礎層的固定橘色
        },
    )

    # 隨機化天空光的 HDRI 貼圖、曝光值、色溫
    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range":     (-4.0, 3.0),       # 很大的曝光範圍，模擬室內到室外
            "temperature_range":  (2500.0, 9500.0),  # 從暖黃光到冷藍光
            "textures_root":      f"{assets_path}/hdri",  # HDRI 貼圖資料夾
            "asset_cfg":          SceneEntityCfg("sky_light"),
        },
    )

    # 覆寫基礎層的 mat 旋轉範圍（DR 版允許更大的旋轉）
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.3, 0.3),  # ±17°，比基礎層的 ±5.7° 更大
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )


# ============================================================
# 觀測設定（Observations）── 第三層
# 繼承視覺觀測，新增 subtask 進度觀測
# ============================================================
@configclass
class VialsToRackObservationsCfg(TaskObservationsCfg):
    """Configuration for observations."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        # ── 抓取狀態觀測 ──
        # 判斷夾爪是否已經抓起任意一支試管
        # 條件：接觸力 > force_threshold AND 試管高度 > min_height
        # warmup_steps：前 30 步忽略（物理穩定後才開始偵測）
        vial_grasped = ObsTerm(
            func=any_vial_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "vials":              ["vial_1", "vial_2", "vial_3"],
                "min_height":         0.055,  # 5.5 cm，確認試管已被提起（不是只碰到）
                "warmup_steps":       30,
                "force_threshold":    2,      # N，最小有效接觸力
            },
        )

        # ── 放置狀態觀測 ──
        # 判斷是否有試管已被放入架子
        # 條件：試管在架子的 XY 範圍內 AND 在 Z_max 以下 AND 試管保持直立
        # grasp_history_window：需要曾經抓過（避免試管自己滾進去算成功）
        vial_placed = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg":  SceneEntityCfg("contact_grasp"),
                "vials":               ["vial_1", "vial_2", "vial_3"],
                "rack_name":           "rack_left",
                "warmup_steps":        30,
                "grasp_history_window": 20,  # 過去 20 步內曾抓取才算有效放置
                "force_threshold":     2,    # N
                # 以下是 Vial_rack_simple.usda 的本地座標範圍（從 USD extent 量出來的）
                "rack_local_x_min":   0.0,
                "rack_local_x_max":   0.12,  # 架子深度 12 cm
                "rack_local_y_min":   0.0,
                "rack_local_y_max":   0.12,  # 架子寬度 12 cm
                "rack_local_z_max":   0.1,   # 插槽入口高度 10 cm
                "vertical_threshold": 0.7,   # 試管 up 向量的 Z 分量必須 > 0.7（接近直立）
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False   # subtask 觀測不加雜訊
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


# ============================================================
# 終止條件設定（Terminations）── 只有 Eval 版本才用
# ============================================================
@configclass
class VialsToRackTerminationsCfg:
    """Termination terms for the vials to tray evaluation task."""

    # 時間超限：episode 超過設定長度就強制結束
    time_out = DoneTerm(
        func=time_out,
        time_out=True,
    )

    # 成功終止：試管放置成功即結束 episode（讓 eval 快速進入下一輪）
    # 參數與 vial_placed 觀測一致，確保判斷邏輯相同
    success = DoneTerm(
        func=vial_placed_on_rack_termination,
        time_out=False,
        params={
            "contact_sensor_cfg":  SceneEntityCfg("contact_grasp"),
            "vials":               ["vial_1", "vial_2", "vial_3"],
            "rack_name":           "rack_left",
            "warmup_steps":        30,
            "grasp_history_window": 20,
            "force_threshold":     2,
            "rack_local_x_min":   0.0,
            "rack_local_x_max":   0.12,
            "rack_local_y_min":   0.0,
            "rack_local_y_max":   0.12,
            "rack_local_z_max":   0.1,
            "vertical_threshold": 0.7,
        },
    )


# ============================================================
# 完整環境設定（四個變體）
# ============================================================

@configclass
class VialsToRackEnvCfg(SO101TaskEnvCfg):
    """
    基礎版：遙操作錄製用，無終止條件
    """
    scene: VialsToRackSceneCfg = VialsToRackSceneCfg()
    events: VialsToRackEventCfg = VialsToRackEventCfg()
    observations: VialsToRackObservationsCfg = VialsToRackObservationsCfg()


@configclass
class VialsToRackDREnvCfg(VialsToRackEnvCfg):
    """
    DR 版：訓練時使用，額外隨機化機器人顏色、天空光、墊子旋轉
    """
    scene: VialsToRackDRSceneCfg = VialsToRackDRSceneCfg()
    events: VialsToRackEventDRCfg = VialsToRackEventDRCfg()


@configclass
class VialsToRackEvalEnvCfg(VialsToRackEnvCfg):
    """
    Eval 版：評估 policy 用，加上成功/超時終止條件
    episode 最長 7.5 分鐘（450 秒 / 60 = 7.5 分鐘）
    """
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0  # 7.5 分鐘


@configclass
class VialsToRackEvalDREnvCfg(VialsToRackDREnvCfg):
    """
    DR Eval 版：同時啟用 Domain Randomization 和終止條件（最嚴格的評估環境）
    """
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0  # 7.5 分鐘
