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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import (
    SO101_DUAL_LEFT_CONTACT_CFG,
    SO101_DUAL_RIGHT_CONTACT_CFG,
)
from sim_to_real_so101.mdp import (
    reset_joints_to_pose,
    reset_vials_rack,
    hide_random_vials,
    any_vial_grasped,
    vial_placed_on_rack,
    all_active_vials_placed_termination,
    time_out,
    ROBOT_COLORS,
    RACK_COLORS,
    randomize_robot_color,
    randomize_rack_color,
    randomize_vial_transparency,
    randomize_sky_light,
    randomize_mat_rotation,
    randomize_camera_focal_length,
    randomize_camera_pose,
)

from .so101_dual_env_cfg import SO101_JOINTS
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

# 試管擺放朝向（沿用你錄 74 集的穩定基準）。
# 註：此 [0,90,0] 其實會把沿 Z 細長的試管「放倒」成沿世界 X（診斷 quat
# =[0.707,0,0.707,0] 已證實），與變數名/註解宣稱的「直立」相反；若日後要真的
# 站直，改成 identity (1,0,0,0) 並把 reset 的 roll 抖動調小（避免站著被 roll 弄倒）。
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
    #
    # ⚠️ 這裡的 y 值是「保證不重疊」解出來的，不能隨手改，理由如下：
    # 試管在 USD 裡沿 local +Z 細長，_UPRIGHT 把它放倒成沿世界 +X，所以它在桌面上
    # 的佔地是 0.1165 m (X) × 0.0341 m (Y) 的長條（Vial_opaque.usda collider extent：
    # x,y ±0.017029、z [-0.017, 0.0995]），不是 3.4 cm 的圓。
    #
    # 由此推論：**X 方向永遠分不開任何東西**——
    #   - 兩支試管要靠 X 錯開需 |Δx| > 0.1165，但墊子只有 0.304 m 深，抖不出來。
    #   - 要靠 X 閃過架子需原點 x ≤ 0.005，但墊子從 x=0.068 才開始。
    # 所以間隔全部押在 Y 上，而 Y 的預算是這樣配的（單位 m）：
    #   架子頂邊最遠 y = 0.0914（-0.06 + jitter 0.01 + 角落轉 yaw 0.2 rad 後的 0.1414）
    #   內側試管 |y| = 0.126  ← 0.0914 + 0.005 餘裕 + 0.017 半寬 + 0.012 jitter
    #   外側試管 |y| = 0.190  ← 0.126 + 0.012 + 0.017 + 0.005 + 0.017 + 0.012
    #   最外緣觸及 |y| = 0.219 < 墊子 0.229 ✓
    # 改 y 值、y jitter 或架子 yaw 範圍都會動到這條不等式鏈，改完要重算。
    vial_left_1 = _vial_at("Vial_Left_1", 0.16, 0.190)
    vial_left_2 = _vial_at("Vial_Left_2", 0.12, 0.126)
    vial_right_1 = _vial_at("Vial_Right_1", 0.12, -0.126)
    vial_right_2 = _vial_at("Vial_Right_2", 0.16, -0.190)

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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + p for p in VIAL_PRIMS_ALL],
    )
    contact_grasp_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Right/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + p for p in VIAL_PRIMS_ALL],
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
            # x ±0.03：試管沿 X 躺著，前後抖不會撞到任何東西，只受墊子邊界限制
            #          （最前 0.12-0.03-0.017=0.073 > 墊子 0.068；最後 0.16+0.03+0.0995=0.2895 < 0.372）。
            # y ±0.012：**這個上限是算出來的，不是隨手填的**。四支試管彼此、以及跟架子的分離
            #          完全靠 Y（理由見上面 vial_left_1 那段），一側要塞兩支 3.41 cm 寬的試管，
            #          y jitter 放到 ±0.015 外側試管就會超出墊子邊緣 0.6 mm，±0.03 則超出 60 mm。
            #          要放大 y jitter 只有兩條路：把架子 yaw 範圍縮小騰出空間，或改成一側只放一支。
            # 旋轉沿用單臂驗證過的 roll 微擾；yaw 維持 0（試管已繞 Y 立起，大 yaw delta 會變成傾倒而非自轉）。
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.012, 0.012),
                "roll": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "fixed_vial_z": VIAL_SPAWN_Z,
            "rack_placement_prob": 0.0,  # 遙操作錄製：先不要預先擺一支在架上
        },
    )

    # ── 隨機出現數量：每次 reset 隨機保留 1~4 支，其餘移到場景外（相機看不到、搆不到）──
    # 排在 reset_vials_setup 之後：先照常 2/2 擺放，再把沒選中的試管停到遠處。
    reset_hide_vials = EventTerm(
        func=hide_random_vials,
        mode="reset",
        params={
            "vials": VIALS_ALL,
            "min_count": 1,
            "max_count": 4,
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
    observations: SO101DualVialsObservationsCfg = (
        SO101DualVialsObservationsCfg()
    )


# ============================================================
# DR 版場景（Phase 5）── 加入可隨機化的天空光（HDRI DomeLight）
# ============================================================
@configclass
class SO101DualVialsDRSceneCfg(SO101DualVialsSceneCfg):

    # 天空光掛在 /World（全域，不在 ENV_REGEX_NS 內）；reset 時換 HDRI + 調曝光/色溫。
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",  # 預設，reset 會被換掉
            visible_in_primary_ray=False,  # 不讓天空光本身出現在相機畫面
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )

    # DR 專用：把 4 支試管換成「可控透明度」的 OmniPBR 版（enable_opacity），
    # 讓 reset 時的 randomize_vial_transparency 能驅動 opacity_constant。
    # base 錄製場景維持不透明 Vial_opaque.usda，僅 DR 訓練場景改用透明變體。
    def __post_init__(self) -> None:
        super().__post_init__()
        for vial_name in VIALS_ALL:
            getattr(self, vial_name).spawn.usd_path = (
                f"{assets_path}/usd/Vial_transparent.usda"
            )


# ============================================================
# DR 版事件（Phase 5）── 在 base reset 之上加外觀隨機化
#
# 沿用單臂驗證過的做法（vials_to_rack_env_cfg.py）：
#   - 機器人顏色：左右兩臂各自獨立隨機（randomize_robot_color robot_names）
#   - 天空光：換 HDRI + 曝光 + 色溫
#   - 墊子旋轉：覆寫 base 的 ±0.1 → 加大到 ±0.3
#   - 相機：兩腕相機焦距抖動 + 中央 ego 相機位姿抖動
#
# 相機刻意沿用單臂的分工（焦距抖腕、位姿抖中央）：
#   - 中央 ego 相機「不」抖焦距 —— 它的 FOV(HFOV≈69°=D435)已量測對齊真機，
#     抖焦距會把 FOV 推離真值、破壞 sim-real 對齊，只小幅抖位姿模擬安裝誤差。
#   - 腕相機只抖焦距、不抖位姿 —— 沿用單臂;腕部 mount xform 是否支援位姿抖動待
#     Isaac 內確認後再加。
# ============================================================
@configclass
class SO101DualVialsEventDRCfg(SO101DualVialsEventCfg):

    # ── 機器人顏色（左右兩臂）──
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": list(ROBOT_COLORS.keys()),
            "robot_names": ("robot_left", "robot_right"),
        },
    )

    # ── 天空光 HDRI + 曝光 + 色溫 ──
    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),  # 室內到室外
            "temperature_range": (2500.0, 9500.0),  # 暖黃到冷藍
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )

    # ── 墊子旋轉（覆寫 base 的 ±0.1 → ±0.3 ≈ ±17°）──
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.3, 0.3),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )

    # ── 左腕相機焦距抖動（基準 13.5mm）──
    reset_camera_wrist_left_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),
            "asset_cfg": SceneEntityCfg("camera_wrist_left"),
        },
    )

    # ── 右腕相機焦距抖動 ──
    reset_camera_wrist_right_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),
            "asset_cfg": SceneEntityCfg("camera_wrist_right"),
        },
    )

    # ── 中央 ego 相機位姿抖動（模擬安裝不精確；不抖焦距以保 FOV 對齊）──
    reset_camera_center_pose = EventTerm(
        func=randomize_camera_pose,
        mode="reset",
        params={
            "prim_path_pattern": "{ENV_REGEX_NS}/LightStudio/LightBox/ego_cam",
            "pos_range": {
                "x": (-0.02, 0.02),  # ±2 cm
                "y": (-0.02, 0.02),
                "z": (-0.01, 0.01),
            },
            "rot_range": {
                "roll": (-0.05, 0.05),  # ±約 3°
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    # ── 中央試管架顏色（從 RACK_COLORS 挑，偏暗含黑色，貼近真實硬體）──
    reset_rack_color = EventTerm(
        func=randomize_rack_color,
        mode="reset",
        params={
            "color_names": list(RACK_COLORS.keys()),
            "rack_names": ("rack_center",),
        },
    )

    # ── 試管透明度（see-through 玻璃 DR；每支獨立，opacity 0=全透明、1=不透明）──
    # 需搭配 DR scene 把試管換成 Vial_transparent.usda（OmniPBR enable_opacity）。
    reset_vial_transparency = EventTerm(
        func=randomize_vial_transparency,
        mode="reset",
        params={
            "opacity_range": (0.08, 0.5),
            "vials": VIALS_ALL,
        },
    )


# ============================================================
# DR 版完整環境（訓練用；仍無 terminations，錄製/rollout 皆可）
# ============================================================
@configclass
class SO101DualVialsDREnvCfg(SO101DualVialsEnvCfg):
    scene: SO101DualVialsDRSceneCfg = SO101DualVialsDRSceneCfg()
    events: SO101DualVialsEventDRCfg = SO101DualVialsEventDRCfg()


# ============================================================
# Eval（Phase 6）── 加上終止條件，評估 GR00T policy 用
#
# 錄製用 base 沒有 terminations（episode 不會自己結束）。eval 版加：
#   - time_out：episode 超時強制結束（回下一輪）
#   - success：dual-safe 成功判定（所有啟用中試管都上架，跨左右夾爪）
# 幾何/力參數與 SubtaskCfg 的 vial_placed 觀測一致，確保成功標準相同。
# ============================================================
@configclass
class SO101DualVialsTerminationsCfg:
    """雙臂 vials-to-rack 評估用終止條件。"""

    time_out = DoneTerm(func=time_out, time_out=True)

    success = DoneTerm(
        func=all_active_vials_placed_termination,
        time_out=False,
        params={
            # 左右兩個夾爪 contact sensor 一起看（協作式：任一臂拿任一支）
            "contact_sensor_cfgs": (
                SceneEntityCfg("contact_grasp_left"),
                SceneEntityCfg("contact_grasp_right"),
            ),
            "vials": VIALS_ALL,
            "rack_name": "rack_center",
            "force_threshold": 2.0,
            "rack_local_x_min": 0.0,
            "rack_local_x_max": 0.12,
            "rack_local_y_min": 0.0,
            "rack_local_y_max": 0.12,
            "rack_local_z_max": 0.1,
            "vertical_threshold": 0.7,
            "active_radius": 1.0,
            "confirm_steps": 25,
        },
    )


# ── eval env：關掉「隨機藏試管」，固定 4 支全在場 ──
# 錄製/訓練用 reset_hide_vials 隨機留 1~4 支(數量魯棒性);但 eval 若也隨機藏,
# 「成功 = 所有 active 試管上架」在只留 1 支的 episode 就會「放 1 支即成功」,
# 看起來像提早中止、難解讀。eval 固定 4 支 → 成功 = 4 支都放進架子,指標清楚。
# 用 __post_init__ 把實例上的 event 設 None(Isaac Lab 官方 disable term 做法),
# 保證生效、不依賴 configclass 對「無標註覆寫成 None」的細節。
# (要量「數量魯棒性」把下面那行 reset_hide_vials = None 註解掉即可。)
# episode 超時上限。Isaac Lab 的欄位單位是「秒」，而 max_episode_length(步) =
# episode_length_s / (sim.dt * decimation) = episode_length_s * 60。真正該想的單位是
# env step，所以這裡先定步數再換算成秒，不要再寫成裸的 `x / 60` 讓人誤讀成分鐘。
#
# 但預算真正的單位是「policy 動作數」，不是 env step：eval 會把每個預測動作撐住
# --steps_per_action 個 env step（見 scripts/lerobot_eval_dual.py，理由是訓練資料是
# 降採樣過的 10 fps，一個 action 不等於一個 env step），所以
#     policy 動作數 = EVAL_EPISODE_STEPS / steps_per_action
# sim 示範平均 332 格/集（86,442 frames / 260 episodes，= 996 env steps）。預設
# steps_per_action=3 時 5000 步 = 1666 個動作 ≈ 示範的 5 倍餘裕（放不完就走 time_out）。
# ⚠️ 改 --steps_per_action 就要一起改這裡，否則餘裕倍數會跟著跑掉。
# 想更寬鬆就調大、想 eval 快就調小（失敗的 episode 會一直跑到用完才結束）。
EVAL_EPISODE_STEPS = 1700
EVAL_EPISODE_LENGTH_S = EVAL_EPISODE_STEPS / 60


# eval 起始姿態（弧度，SO101_JOINTS 順序：Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw）。
# 取自錄製資料集 datasets/bimanual-so101-pickvials 240 集第一格的中位姿態，換算成 sim 弧度。
# 目的：讓 eval reset 的開機姿態落在訓練分布內，policy 第一步才不會在分布外亂動。
# （warmup 也用同一組值，見 scripts/lerobot_eval_dual.py，一處定義兩處共用。）
DUAL_LEFT_START_POSE = [-0.0425, -1.7256, 1.5708, -1.6437, -1.6648, -0.1573]
DUAL_RIGHT_START_POSE = [-0.0009, -1.7274, 1.5702, -1.6126, -1.6469, -0.1348]


def _use_dataset_start_pose(events) -> None:
    """把左右臂的 reset 從『USD 預設姿態』換成『資料集起始姿態』（絕對重設）。

    只在 eval cfg 呼叫，錄製/訓練用的 base 環境不受影響。
    preserve_order=True 確保 joint_ids 與 target_pos 同序（見 reset_joints_to_pose 說明）。
    """
    events.reset_robot_left.func = reset_joints_to_pose
    events.reset_robot_left.params = {
        "asset_cfg": SceneEntityCfg("robot_left", joint_names=SO101_JOINTS, preserve_order=True),
        "target_pos": DUAL_LEFT_START_POSE,
    }
    events.reset_robot_right.func = reset_joints_to_pose
    events.reset_robot_right.params = {
        "asset_cfg": SceneEntityCfg("robot_right", joint_names=SO101_JOINTS, preserve_order=True),
        "target_pos": DUAL_RIGHT_START_POSE,
    }


@configclass
class SO101DualVialsEvalEnvCfg(SO101DualVialsEnvCfg):
    """乾淨場景 + 終止條件（評估純 sim policy）。固定 4 支、episode 上限見 EVAL_EPISODE_STEPS。"""

    terminations: SO101DualVialsTerminationsCfg = (
        SO101DualVialsTerminationsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = EVAL_EPISODE_LENGTH_S  # 秒；success 會提前結束
        self.events.reset_hide_vials = None  # eval 固定 4 支，不隨機藏
        _use_dataset_start_pose(self.events)  # 開機姿態對齊訓練資料


@configclass
class SO101DualVialsEvalDREnvCfg(SO101DualVialsDREnvCfg):
    """DR 場景 + 終止條件（最嚴格：外觀隨機 + 評估）。固定 4 支、episode 上限見 EVAL_EPISODE_STEPS。"""

    terminations: SO101DualVialsTerminationsCfg = (
        SO101DualVialsTerminationsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = EVAL_EPISODE_LENGTH_S  # 秒；success 會提前結束
        self.events.reset_hide_vials = None  # eval 固定 4 支，不隨機藏
        _use_dataset_start_pose(self.events)  # 開機姿態對齊訓練資料
