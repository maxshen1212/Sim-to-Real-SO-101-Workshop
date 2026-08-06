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
"""雙臂 SO-101 eval client（連遠端 GR00T inference server，在 sim 裡跑 rollout）。

單臂版 lerobot_eval.py 的雙臂變體：12 維 action、左右各一個 LeRobotSO101Interface、
state/action 依 SO101_bimanual modality 分成 left_arm/left_gripper/right_arm/right_gripper。
"""
import argparse
import random
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Isaac Lab 雙臂 SO-101 Eval Client (remote GR00T inference server)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Lerobot-So101-Dual-Vials-To-Rack-Eval", help="Name of the task."
)
parser.add_argument("--seed", type=int, default=1984, help="Environment seed")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--policy_host", type=str, default="localhost", help="GR00T policy server host")
parser.add_argument("--policy_port", type=int, default=5555, help="GR00T policy server port")
parser.add_argument(
    "--action_horizon", type=int, default=16, help="Number of action steps to execute per server query"
)
parser.add_argument(
    "--steps_per_action",
    type=int,
    default=3,
    help=(
        "把每個預測動作撐住幾個 env step。env 的控制率是 60 Hz(sim.dt=1/120 x decimation=2)，"
        "但 checkpoint 吃的是降採樣過的 10 fps 資料集，一個 action 不等於一個 env step。"
        "要設成建立 sim-10fps 資料集時用的降採樣倍率：sim 錄製是每個 env step 推一格，"
        "所以一格訓練資料 = 降採樣倍率個 env step。設 1 會讓手臂快 3 倍——"
        "就是真機那邊抓到的同一個錯。"
        "預設 3 已驗證：ChihHanShen/bimanual-so101-pickvials-sim 是 260 集 / 259,079 格，"
        "-sim-10fps 是同樣 260 集 / 86,442 格，比值 2.997 = 每 3 格留 1 格。"
        "重建資料集換了倍率就要改這裡，並一起調 EVAL_EPISODE_STEPS。"
    ),
)
parser.add_argument(
    "--lang_instruction",
    type=str,
    default="Pick up the vials and place them into the rack",
    help="Language instruction for the policy",
)
parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Record camera_center (俯視) 每個 episode 一支 mp4，供 demo 用。",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="./eval_videos",
    help="錄影輸出資料夾（--record_video 時生效）。",
)
parser.add_argument(
    "--video_fps", type=int, default=30, help="錄影 mp4 的 fps。"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import os

import gymnasium as gym
import numpy as np
import torch


def _to_uint8_frame(cam_obs: torch.Tensor) -> np.ndarray:
    """把單張相機觀測 (num_envs, H, W, 3) 轉成 env 0 的 HWC uint8 numpy frame。"""
    img = cam_obs[0].detach().to("cpu")
    if img.dtype != torch.uint8:
        # normalize=False 時是 0~255 的 float；保險起見再 clip 一次
        img = img.clamp(0, 255).round().to(torch.uint8)
    return img.numpy()

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.so101_dual_vials_env_cfg import (
    DUAL_LEFT_START_POSE,
    DUAL_RIGHT_START_POSE,
)
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import (
    LeRobotSO101Interface,
    GR00TDualRemotePolicy,
)


def main():
    keyboard_control = KeyboardControl()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    # Seed all RNGs for reproducible episode resets
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 這支腳本只支援單一 env：觀測一律取 [0]、policy 算出的 action 會廣播到所有 env，
    # 成功率也是用 .any() 統計 —— num_envs > 1 會靜靜地算錯，不如直接擋掉。
    if env.unwrapped.num_envs != 1:
        env.close()
        raise ValueError(
            f"lerobot_eval_dual 只支援 --num_envs 1（目前 {env.unwrapped.num_envs}）："
            "obs 只取 env 0、action 會廣播到全部 env，成功率統計不正確。"
        )

    if args_cli.steps_per_action < 1:
        env.close()
        raise ValueError(f"--steps_per_action 必須 >= 1（目前 {args_cli.steps_per_action}）。")

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")

    # 把控制頻率印出來對帳。要對齊的是「錄製時一格資料等於幾個 env step」，不是資料集標的
    # fps：sim 錄製是每個 env step 推一格（見 lerobot_agent_dual.py），標籤卻寫 30 fps，
    # 所以 sim 資料集的 fps 標籤本來就不等於 sim 時間。steps_per_action 要設成建立
    # bimanual-so101-pickvials-sim-10fps 時用的降採樣倍率。
    # 附帶結果：倍率 3 → policy 在 sim time 是 20 Hz（不是 10 Hz），這是正常的。
    control_hz = 1.0 / env.unwrapped.step_dt
    policy_hz = control_hz / args_cli.steps_per_action
    max_actions = env.unwrapped.max_episode_length // args_cli.steps_per_action
    print(
        f"[INFO]: env {control_hz:.1f} Hz x {args_cli.steps_per_action} steps/action "
        f"-> policy {policy_hz:.1f} Hz (sim time)。steps/action 必須等於 sim 資料集的降採樣倍率。"
    )
    print(
        f"[INFO]: Episode 上限 {env.unwrapped.max_episode_length} env steps "
        f"= {max_actions} 個 policy 動作 = {env.unwrapped.max_episode_length_s:.1f} s sim time"
    )

    # cameras（自動發現 camera_ 開頭的 scene 物件 → wrist_left / wrist_right / center）
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print(f"[Info]: No cameras found - videos will not be recorded")

    # 左右各一個 interface（只用其 joint mapping；sim 內不接硬體，port=None）
    left_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="eval_left",
        cameras=cameras, fps=30, kind="follower",
    )
    right_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="eval_right",
        cameras=cameras, fps=30, kind="follower",
    )
    left_iface.init_device(visualize=args_cli.rerun)
    right_iface.init_device(visualize=False)

    # 雙臂 remote policy
    policy = GR00TDualRemotePolicy(
        left_iface=left_iface,
        right_iface=right_iface,
        camera_names=list(cameras.keys()),
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        lang_instruction=args_cli.lang_instruction,
    )
    policy.connect()

    # 錄影設定：只錄 camera_center 俯視（headless demo 用）。
    record_video = args_cli.record_video
    video_cam_key = "rgb_center"
    if record_video:
        os.makedirs(args_cli.video_dir, exist_ok=True)
        print(f"[INFO]: Recording camera_center to {os.path.abspath(args_cli.video_dir)}")
    frames = []  # 累積當前 episode 的 frame

    def _write_video(ep_idx: int, success: bool) -> None:
        if not record_video or len(frames) == 0:
            return
        import imageio.v2 as imageio
        tag = "success" if success else "fail"
        path = os.path.join(args_cli.video_dir, f"ep{ep_idx:03d}_{tag}.mp4")
        imageio.mimwrite(path, frames, fps=args_cli.video_fps, macro_block_size=1)
        print(f"[INFO]: Saved {path} ({len(frames)} frames)")

    # reset environment
    obs, _ = env.reset()
    policy.reset()

    # 12 維 action（左 0:6、右 6:12）。warmup 命令的姿態要跟 env reset 的起始姿態一致，
    # 否則手臂在 warmup 就開始移動、第一格 obs 又偏離訓練分布。這兩組常數就是 reset 用的
    # 資料集起始姿態（見 tasks/so101_dual_vials_env_cfg.py），左右各一組。
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    left_rest = torch.tensor(DUAL_LEFT_START_POSE, device=env.unwrapped.device)
    right_rest = torch.tensor(DUAL_RIGHT_START_POSE, device=env.unwrapped.device)
    initial_action = torch.cat([left_rest, right_rest], dim=0)  # (12,)

    # WARMUP_STEPS 的單位是 env step（不是 policy 動作），跟 `step` 一致。
    WARMUP_STEPS = 10
    steps_per_action = args_cli.steps_per_action

    step = 0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0

    pbar = None

    while simulation_app.is_running():
        with torch.inference_mode():

            if step == 0:
                pbar = tqdm(
                    total=env.unwrapped.max_episode_length,
                    desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)",
                    unit="step",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            if step < WARMUP_STEPS:  # warmup
                actions[:] = initial_action
            else:
                joint_pos_left = obs["policy"]["joint_pos_left"][0].clone()
                joint_pos_right = obs["policy"]["joint_pos_right"][0].clone()
                actions[:] = policy.get_action(
                    joint_pos_left, joint_pos_right, obs["visual"], log=True
                )

            # 同一個動作撐 steps_per_action 個 env step。sim 裡沒有真機 client 的 sleep 可用，
            # 這就是「control rate = 訓練資料集 fps」在 sim 的等價做法：env 是 60 Hz，資料是
            # 10 fps，一個 action 送一步會讓手臂比訓練資料快 steps_per_action 倍。
            is_terminated = False
            is_truncated = False
            for _ in range(steps_per_action):
                obs, rewards, terminated, truncated, info = env.step(actions)

                if record_video and video_cam_key in obs["visual"]:
                    frames.append(_to_uint8_frame(obs["visual"][video_cam_key]))

                step += 1

                if pbar is not None:
                    pbar.update(1)

                is_terminated = (
                    terminated.item() if terminated.numel() == 1 else terminated.any().item()
                )
                is_truncated = (
                    truncated.item() if truncated.numel() == 1 else truncated.any().item()
                )
                if is_terminated or is_truncated:
                    break

            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None

                num_episodes += 1
                episode_success = is_terminated and not is_truncated
                if episode_success:
                    num_successes += 1
                success_rate = (num_successes / num_episodes) * 100

                _write_video(num_episodes, episode_success)
                frames.clear()

                # 收工判斷放在 reset 之前：擺在迴圈尾端會多開一集、多跑一步、
                # 多印一條假的進度條才退出。
                if num_episodes >= args_cli.num_episodes:
                    break

                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            # Manual reset with 'R' key
            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[MANUAL RESET] Episode interrupted at step {step}")
                frames.clear()  # 手動中斷的 episode 不存
                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

    if pbar is not None:
        pbar.close()
    print(f"[INFO]: Evaluated {num_episodes} episodes")
    if num_episodes > 0:
        print(f"[INFO]: Success Rate: {num_successes}/{num_episodes} ({success_rate:.1f}%)")
    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
