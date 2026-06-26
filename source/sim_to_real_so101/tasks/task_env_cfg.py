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
# 第二層環境設定（任務視覺層）
#
# 繼承第一層（so101_env_cfg.py），在基礎機器人場景上新增：
#   - lightbox（打光燈箱 USD）
#   - mat（桌墊 USD）
#   - 兩個相機：wrist cam（夾爪視角）、外部 D455 固定相機
#   - 對應的 DR 事件：燈光曝光、相機焦距、相機位姿、墊子旋轉隨機化
#   - RGB / Depth / Instance Segmentation 觀測
#
# 繼承關係：
#   so101_env_cfg.py（基礎層）
#       ↓ 本檔繼承並擴充
#   task_env_cfg.py（視覺任務層）
#       ↓ 被 vials_to_rack_env_cfg.py 繼承，再加入操作物件
# ============================================================

import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    randomize_light_exposure,
    randomize_sky_light,
    randomize_mat_rotation,
    randomize_camera_focal_length,
    randomize_camera_pose,
    image,
    image_raw,
)

from .so101_env_cfg import (
    SO101TeleopEnvCfg,
    LerobotSo101BaseSceneCfg,
    EventCfg,
    ObservationsCfg,
)

# assets 資料夾的絕對路徑，用來拼接 USD 路徑
assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# ============================================================
# 相機模板
# TiledCameraCfg 是 Isaac Lab 的批次渲染相機，支援多環境並行
# 這裡定義共用的相機參數，後面再用 .replace() 複製並覆寫 prim_path
# ============================================================
camera_object = TiledCameraCfg(
    prim_path="",       # 由各相機實例覆寫
    update_period=0.0,  # 0 = 每個物理 step 都更新畫面
    height=480,
    width=640,
    # 同時取得 RGB、深度圖、instance segmentation（用於 debug 與 DR 標記）
    data_types=["rgb", "depth", "instance_id_segmentation_fast"],
    colorize_instance_segmentation=True,  # 把 instance ID 轉成彩色圖方便目視
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,          # 光圈設很大（景深幾乎無限遠），避免模糊干擾訓練
        focal_length=13.5,   # 焦距（mm），對應真實 D435i 的 1/10 縮放（Isaac Sim 單位慣例）
        focus_distance=0.05, # 對焦距離 5 cm（wrist cam 近距離）
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",  # Isaac Sim 相機座標用 OpenGL 慣例（Z 朝後、Y 朝上）
    ),
)


# ============================================================
# 場景設定（Scene）── 第二層
# 繼承基礎場景，加入燈箱、墊子、兩個相機
# ============================================================
@configclass
class SO101TaskSceneCfg(LerobotSo101BaseSceneCfg):

    # ── 燈箱 USD ──
    # lightbox-simple.usd 是一個預先做好的「攝影棚」模型，
    # 裡面包含牆壁、面板燈（RectLight）和外部相機的安裝座（camera_mount）。
    # 把整個 USD 載入後，我們可以直接引用其內部的子 prim（如 LightBox/RectLight）
    lightstudio = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/lightbox-simple.usd",
        ),
        # 稍微往後移動並墊高，讓機器人位於燈箱正中央
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0, 0.0257)),
    )

    # ── 燈箱內的矩形燈 ──
    # spawn=None 表示不生成新物件，只是「引用」USD 裡既有的 prim，
    # 讓 Isaac Lab 的 EventManager 能控制它（例如隨機化亮度）
    lightbox_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio/LightBox/RectLight"
    )

    # ── 天空光（HDRI 環境光）── 目前關閉，DR 版本才啟用
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/sky_light",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=1000.0,
    #         texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
    #         visible_in_primary_ray=False,
    #         enable_color_temperature=True,
    #         color_temperature=6500.0
    #     )
    # )

    # ── 桌墊 USD ──
    # 桌墊提供視覺紋理，DR 時會旋轉它來增加外觀多樣性
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0, 0.032),   # 擺在機器人前方的桌面上
            rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),  # 旋轉 90° 對齊
        ),
    )

    # ── Wrist Camera（夾爪視角相機）──
    # 掛在機器人夾爪（gripper）prim 下，跟著手臂移動
    # offset 定義相機在 gripper 座標系下的偏移位置和朝向角度
    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos = (-0.005, 0.06, -0.062)   # 相機中心位置（公尺）
    camera_ego.offset.rot = euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True)  # 俯角 45°

    # ── 外部固定相機（D455 模擬）──
    # 掛在燈箱 USD 內的 camera_mount 節點下，位置由燈箱 USD 決定
    # spawn=None 表示相機 prim 已經在 USD 裡定義好了，直接引用即可
    camera_external_D455 = camera_object.replace()
    camera_external_D455.prim_path = "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount/rsd455/RSD455/Camera_OmniVision_OV9782_Right"
    camera_external_D455.spawn = None  # USD 裡已有此相機 prim，不需要再生成


# ============================================================
# 事件設定（Events）── 第二層
# 繼承基礎 EventCfg，新增視覺 Domain Randomization 事件
# ============================================================
@configclass
class TaskEventCfg(EventCfg):
    """Configuration for events."""

    # ── 燈光亮度隨機化 ──
    # 每次 reset 時隨機改變燈箱 RectLight 的曝光值（EV）
    # 範圍 -3~+1 EV，模擬真實場景中的不同打光條件
    reset_lightbox_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("lightbox_light"),
        },
    )

    # ── 桌墊旋轉隨機化 ──
    # 小幅度隨機旋轉桌墊（±0.1 rad ≈ ±5.7°），模擬真實擺放時的歪斜
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )

    # ── Wrist Camera 焦距隨機化 ──
    # 每次 reset 隨機改變焦距（±約10%），模擬相機個體差異
    reset_camera_ego_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),  # mm，基準值 13.5mm
            "asset_cfg": SceneEntityCfg("camera_ego"),
        },
    )

    # ── 外部相機位姿隨機化 ──
    # 小幅度隨機移動並旋轉外部相機的安裝座（camera_mount），
    # 模擬真實世界中相機固定不精確的情況
    # 注意：這裡用 prim_path_pattern 而非 asset_cfg，因為安裝座不是獨立 Asset
    reset_camera_external_pose = EventTerm(
        func=randomize_camera_pose,
        mode="reset",
        params={
            "prim_path_pattern": "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount",
            "pos_range": {
                "x": (-0.02, 0.02),  # ±2 cm
                "y": (-0.02, 0.02),
                "z": (-0.01, 0.01),
            },
            "rot_range": {
                "roll":  (-0.05, 0.05),  # ±約 3°
                "pitch": (-0.05, 0.05),
                "yaw":   (-0.05, 0.05),
            },
        },
    )


# ============================================================
# 觀測設定（Observations）── 第二層
# 繼承基礎觀測，新增相機影像觀測群組
# ============================================================
@configclass
class TaskObservationsCfg(ObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):
        # ── Wrist Camera 影像 ──

        # RGB 畫面（normalize=False 表示保持 0~255 整數，不縮放到 0~1）
        rgb_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        # 深度圖（每個 pixel 是公尺距離）
        depth_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "depth",
            },
        )

        # Instance Segmentation（用 image_raw 取原始 ID tensor，不做後處理）
        instance_id_seg_ego = ObsTerm(
            func=image_raw,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "instance_id_segmentation_fast",
            },
        )

        # ── 外部固定相機（D455）影像 ──

        rgb_external_D455 = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external_D455"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        depth_external_D455 = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external_D455"),
                "data_type": "depth",
            },
        )

        instance_id_seg_external_D455 = ObsTerm(
            func=image_raw,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external_D455"),
                "data_type": "instance_id_segmentation_fast",
            },
        )

        def __post_init__(self) -> None:
            # 影像觀測不加雜訊（雜訊加在影像上沒有意義）
            self.enable_corruption = False
            self.concatenate_terms = False

    # 將視覺觀測群組掛到 visual 這個 key
    visual: VisualCfg = VisualCfg()


# ============================================================
# 完整環境設定（Environment）── 第二層
# 組合視覺場景、任務事件、視覺觀測
# ============================================================
@configclass
class SO101TaskEnvCfg(SO101TeleopEnvCfg):
    """Configuration for the task environment."""

    scene: SO101TaskSceneCfg = SO101TaskSceneCfg()
    events: TaskEventCfg = TaskEventCfg()
    observations: TaskObservationsCfg = TaskObservationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # 任務層開啟半透明渲染（試管需要它才能正確顯示玻璃材質）
        self.sim.render.enable_translucency = True
        # 追加光線追蹤進階設定，讓試管玻璃、反射看起來更真實
        carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,       # 每次折射都計算反射
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,        # 最多 5 次折射
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.sim.render.carb_settings = carb_settings
