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
# 雙臂任務視覺層（Phase 3）
#
# 繼承雙臂 base（so101_dual_env_cfg.py），加上：
#   - lightbox（燈箱，已含中央 ego 相機 ego_cam）+ mat（桌墊）
#   - 三台 D435i：camera_wrist_left（左腕）、camera_wrist_right（右腕）、
#                 camera_center（燈箱開口置中俯視）
#   - 三相機的 RGB + Depth 觀測
#
# 相機都用 spawn=None 直接指向 USD 內已烤好的 Camera prim
# （腕部 = 各臂 d435i USD；中央 = 燈箱 ego_cam）。
# 任務物件（vials/rack）與雙臂 handoff 邏輯是 Phase 4。
# ============================================================

import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    randomize_light_exposure,
    randomize_mat_rotation,
    image,
)

from .so101_dual_env_cfg import (
    LerobotSo101DualBaseSceneCfg,
    SO101DualTeleopEnvCfg,
    DualEventCfg,
    DualObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# d435i 相機在各 USD 內的 Camera prim 相對路徑（gripper / ego_cam 之後都一樣）
_CAM_SUFFIX = "camera_link/camera_color_frame/camera_color_optical_frame/Camera"

# 共用相機模板（spawn=None 時 intrinsics 取自 USD Camera prim，這裡的 spawn 參數會被忽略）
camera_object = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb", "depth", "instance_id_segmentation_fast"],
    colorize_instance_segmentation=True,
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,
        focal_length=13.5,
        focus_distance=0.05,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)


# ============================================================
# 場景：雙臂 base + 燈箱 + 墊子 + 三相機
# ============================================================
@configclass
class SO101DualTaskSceneCfg(LerobotSo101DualBaseSceneCfg):

    # ── 燈箱（已含中央 ego 相機）──
    lightstudio = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/lightbox-egocam.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0, 0.0257)),
    )
    lightbox_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio/LightBox/RectLight"
    )

    # ── 桌墊（置中 y=0）──
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0, 0.032),
            rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
        ),
    )

    # ── 左腕 D435i ──
    camera_wrist_left = camera_object.replace()
    camera_wrist_left.prim_path = (
        f"{{ENV_REGEX_NS}}/Robot_Left/gripper/realsense_d435i/{_CAM_SUFFIX}"
    )
    camera_wrist_left.spawn = None

    # ── 右腕 D435i ──
    camera_wrist_right = camera_object.replace()
    camera_wrist_right.prim_path = (
        f"{{ENV_REGEX_NS}}/Robot_Right/gripper/realsense_d435i/{_CAM_SUFFIX}"
    )
    camera_wrist_right.spawn = None

    # ── 中央 ego D435i（燈箱開口俯視）──
    camera_center = camera_object.replace()
    camera_center.prim_path = (
        f"{{ENV_REGEX_NS}}/LightStudio/LightBox/ego_cam/{_CAM_SUFFIX}"
    )
    camera_center.spawn = None


# ============================================================
# 事件：雙臂 reset + 燈光/墊子小幅隨機（完整 DR 在 Phase 5）
# ============================================================
@configclass
class DualTaskEventCfg(DualEventCfg):

    reset_lightbox_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("lightbox_light"),
        },
    )

    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )


# ============================================================
# 觀測：雙臂關節 + 三相機 RGB/Depth
# ============================================================
@configclass
class DualTaskObservationsCfg(DualObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):
        # 左腕
        rgb_wrist_left = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist_left"),
                    "data_type": "rgb", "normalize": False},
        )
        depth_wrist_left = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist_left"),
                    "data_type": "depth"},
        )
        # 右腕
        rgb_wrist_right = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist_right"),
                    "data_type": "rgb", "normalize": False},
        )
        depth_wrist_right = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist_right"),
                    "data_type": "depth"},
        )
        # 中央 ego
        rgb_center = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_center"),
                    "data_type": "rgb", "normalize": False},
        )
        depth_center = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_center"),
                    "data_type": "depth"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


# ============================================================
# 完整環境
# ============================================================
@configclass
class SO101DualTaskEnvCfg(SO101DualTeleopEnvCfg):
    scene: SO101DualTaskSceneCfg = SO101DualTaskSceneCfg()
    events: DualTaskEventCfg = DualTaskEventCfg()
    observations: DualTaskObservationsCfg = DualTaskObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 玻璃/反射用的渲染設定（之後 Phase 4 放 vials 會用到）
        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
