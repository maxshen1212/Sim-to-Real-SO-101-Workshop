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
# 雙臂基礎環境（Phase 2）
#
# 把單臂 base（so101_env_cfg.py）擴成雙臂：
#   - robot_left / robot_right 兩個獨立 articulation（各自的 base 位置）
#   - 兩個 ee_frame 追蹤器
#   - 兩組 6 維 JointPositionAction → 合計 12 維動作
#   - 左右各一份關節狀態 + ee 位姿觀測
#   - reset 事件分別重設左右臂
#
# 先做能站住的 base（無相機、無任務物件），用 zero_agent 驗證兩臂不互撞爆掉，
# 再往上疊相機（Phase 3）與任務（Phase 4）。
# ============================================================

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg

from sim_to_real_so101.assets.so101 import SO101_DUAL_LEFT_CFG, SO101_DUAL_RIGHT_CFG
from sim_to_real_so101.mdp import (
    JointPositionActionCfg,
    joint_pos,
    joint_pos_rel,
    reset_joints_by_offset,
    ee_frame_state,
)

# SO-101 的關節順序（必須與 USD articulation 一致）
SO101_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _ee_frame(prefix: str) -> FrameTransformerCfg:
    """產生一隻手臂的 ee 追蹤器：gripper 相對 base 的位姿。"""
    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{prefix}/gripper", name="gripper"
            ),
        ],
    )


# ============================================================
# 場景：兩隻手臂 + 兩個 ee_frame
# ============================================================
@configclass
class LerobotSo101DualBaseSceneCfg(InteractiveSceneCfg):

    env_spacing = 4.0
    num_envs = 1

    # 兩隻手臂（base 位置：左 +y、右 −y，皆面向 +x）
    robot_left: ArticulationCfg = SO101_DUAL_LEFT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_Left"
    )
    robot_right: ArticulationCfg = SO101_DUAL_RIGHT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_Right"
    )

    ee_frame_left: FrameTransformerCfg = _ee_frame("Robot_Left")
    ee_frame_right: FrameTransformerCfg = _ee_frame("Robot_Right")


# ============================================================
# 動作：左右各 6 維 → 合計 12 維
# ============================================================
@configclass
class DualActionsCfg:
    """Action specifications for the dual-arm MDP."""

    joint_positions_left = JointPositionActionCfg(
        asset_name="robot_left",
        joint_names=SO101_JOINTS,
        scale=1,
        use_default_offset=False,
    )
    joint_positions_right = JointPositionActionCfg(
        asset_name="robot_right",
        joint_names=SO101_JOINTS,
        scale=1,
        use_default_offset=False,
    )


# ============================================================
# 觀測：左右各一份關節狀態 + ee 位姿
# ============================================================
@configclass
class DualObservationsCfg:
    """Observation specifications for the dual-arm MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ── 左臂 ──
        joint_pos_left = ObsTerm(
            func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot_left")}
        )
        joint_pos_rel_left = ObsTerm(
            func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_left")}
        )
        ee_frame_left = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame_left"),
                "robot_cfg": SceneEntityCfg("robot_left"),
            },
        )
        # ── 右臂 ──
        joint_pos_right = ObsTerm(
            func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot_right")}
        )
        joint_pos_rel_right = ObsTerm(
            func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_right")}
        )
        ee_frame_right = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame_right"),
                "robot_cfg": SceneEntityCfg("robot_right"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# 事件：reset 時分別把左右臂歸位
# ============================================================
@configclass
class DualEventCfg:
    """Configuration for events."""

    reset_robot_left = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_left", joint_names=SO101_JOINTS),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )
    reset_robot_right = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right", joint_names=SO101_JOINTS),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )
    # 註：顏色 DR（randomize_robot_color）目前寫死 env.scene["robot"]，雙臂不適用，
    #     等 Phase 5 處理雙臂 DR 時再擴成可指定 asset。


# ============================================================
# 完整環境
# ============================================================
@configclass
class SO101DualTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: LerobotSo101DualBaseSceneCfg = LerobotSo101DualBaseSceneCfg()

    observations: DualObservationsCfg = DualObservationsCfg()
    actions: DualActionsCfg = DualActionsCfg()
    events: DualEventCfg = DualEventCfg()

    rewards = None
    terminations = None

    def __post_init__(self) -> None:
        """Post initialization."""
        # decimation=4：物理跑 4 步，policy 才輸出一次動作。
        # 等效控制頻率 = 120 Hz / 4 = 30 Hz —— 這個 30 是刻意的，不是隨手選的：
        # 錄製時每個 env step 推一格資料（scripts/lerobot_agent_dual.py），所以
        # 「env 控制率」就是 sim 資料集的真實 fps。設 30 讓它等於真機 RealSense 的
        # 30 fps，sim/real co-train 才會在同一個時間基準上。
        # （單臂那邊仍是 decimation=2 → 60 Hz，沒有動。）
        # ⚠️ 改這個值會連動 so101_dual_vials_env_cfg.EVAL_EPISODE_LENGTH_S 的換算。
        self.decimation = 4
        self.episode_length_s = 5

        self.scene.num_envs = 1
        # 視角：拉遠一點看得到兩臂（兩臂在 y=±0.15）
        self.viewer.eye = (-0.5, 0.0, 0.4)
        self.viewer.lookat = (0.2, 0.0, 0.1)
        # 模擬設定
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False
