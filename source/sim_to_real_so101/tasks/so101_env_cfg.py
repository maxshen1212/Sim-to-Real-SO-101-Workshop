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
# 第一層環境設定（基礎層）
#
# 這個檔案定義了最基礎的 SO-101 機器人手臂環境。
# 只包含：機器人本體、末端執行器 (end-effector) 位置追蹤、
# 關節動作、關節狀態觀測，以及 reset 時的基礎事件。
#
# 繼承關係（由下往上疊加）：
#   so101_env_cfg.py（本檔，基礎層）
#       ↓ 被 task_env_cfg.py 繼承，加入相機、燈光、墊子
#           ↓ 被 vials_to_rack_env_cfg.py 繼承，加入試管、架子、抓取邏輯
# ============================================================

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

# import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from sim_to_real_so101.assets.so101 import SO101_CFG

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    JointPositionActionCfg,
    joint_pos,
    reset_joints_by_offset,
    randomize_robot_color,
    ee_frame_state,
    joint_pos_rel,
)

# assets 資料夾的絕對路徑，用來拼接 USD 檔案路徑
assets_path = os.path.dirname(os.path.abspath(assets.__file__))


# ============================================================
# 場景設定（Scene）
# InteractiveSceneCfg 定義「舞台上有哪些物件」
# ============================================================
@configclass
class LerobotSo101BaseSceneCfg(InteractiveSceneCfg):

    # 每個環境實例之間的間距（公尺）；多環境並行訓練時才有意義
    env_spacing = 4.0
    # 遙操作模式下固定只開一個環境
    num_envs = 1

    # ── 機器人 ──
    # SO101_CFG 定義了機器人的 USD 路徑、初始關節角度、各關節的 ImplicitActuator
    # prim_path 使用 {ENV_REGEX_NS} 佔位符，Isaac Lab 會自動展開成每個環境的路徑
    # 例如：/World/envs/env_0/Robot、/World/envs/env_1/Robot ...
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── 末端執行器座標追蹤器 ──
    # FrameTransformerCfg 讓我們在每個 step 取得 gripper 在機器人 base 座標系下的位姿
    # prim_path 指向「參考原點」（base），target_frames 指向「我們要追蹤的目標」（gripper）
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"
            ),  # 不加 offset，保持原始位姿供 IK 計算使用
        ],
    )


# ============================================================
# 動作設定（Actions）
# 定義 policy 輸出什麼、怎麼傳給機器人
# ============================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # JointPositionAction：policy 輸出「目標關節角度」，直接寫入模擬器
    # joint_names 的順序必須與 USD articulation 定義的順序一致
    # scale=1 表示 policy 輸出的單位直接是弧度（不縮放）
    # use_default_offset=False 表示輸出值就是絕對角度，不加上預設姿勢偏移
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1,
        use_default_offset=False,
    )


# ============================================================
# 觀測設定（Observations）
# 定義 policy 能看到哪些資訊
# ============================================================
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 各關節的當前絕對角度（弧度）
        joint_pos_obs = ObsTerm(func=joint_pos)

        # 各關節相對於預設姿勢的偏移量（相對角度）
        joint_pos_rel = ObsTerm(func=joint_pos_rel)

        # gripper 末端在機器人 base 座標系下的位姿（位置 + 四元數）
        ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),    # 上面定義的追蹤器
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            # 允許對觀測值加入雜訊（模擬感測器誤差，增加 sim-to-real 遷移性）
            self.enable_corruption = True
            # False = 各觀測項目保持獨立 tensor，不拼接成單一向量
            self.concatenate_terms = False

    # 將上面定義的 PolicyCfg 掛到 policy 這個 observation group
    policy: PolicyCfg = PolicyCfg()


# ============================================================
# 事件設定（Events）
# 定義在 reset、step 等時間點會觸發哪些事件
# ============================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # ── Reset 事件：重設機器人關節角度 ──
    # mode="reset" 表示每次 episode 開始（reset）時觸發
    # position_range=(0,0) 表示不加隨機偏移，每次都回到相同的預設姿勢
    # velocity_range=(0,0) 表示初始速度為零
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Rotation",
                    "Pitch",
                    "Elbow",
                    "Wrist_Pitch",
                    "Wrist_Roll",
                    "Jaw",
                ],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    # ── Reset 事件：隨機化機器人外觀顏色 ──
    # 固定使用橘色（遙操作 demo 模式）；DR 版本會隨機從調色盤挑色
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": ["orange"],
        },
    )


# ============================================================
# 完整環境設定（Environment）
# 將上面所有組件組合成一個 ManagerBasedRLEnv
# ============================================================
@configclass
class SO101TeleopEnvCfg(ManagerBasedRLEnvCfg):
    # 場景中有哪些物件
    scene: LerobotSo101BaseSceneCfg = LerobotSo101BaseSceneCfg()

    # policy 看到什麼
    observations: ObservationsCfg = ObservationsCfg()
    # policy 輸出什麼
    actions: ActionsCfg = ActionsCfg()
    # reset/step 時發生什麼事
    events: EventCfg = EventCfg()

    # 遙操作模式不需要獎勵函數和終止條件
    rewards = None
    terminations = None

    def __post_init__(self) -> None:
        """Post initialization."""
        # decimation=2：物理模擬每跑 2 步，policy 才輸出一次動作
        # 等效控制頻率 = 120 Hz / 2 = 60 Hz
        self.decimation = 2
        # 每個 episode 最長 5 秒（遙操作時無上限意義，但需要設定）
        self.episode_length_s = 5

        # 遙操作固定單一環境
        self.scene.num_envs = 1
        # 視角設定（Isaac Sim 預設相機位置和朝向）
        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)
        # 物理模擬步長：1/120 秒 = 120 Hz
        self.sim.dt = 1 / 120
        # 渲染間隔與 decimation 對齊
        self.sim.render_interval = self.decimation

        # 使用高品質渲染模式（光線追蹤）；半透明先關掉（基礎層不需要）
        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False
