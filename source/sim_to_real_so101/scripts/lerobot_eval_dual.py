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
    "--lang_instruction",
    type=str,
    default="pick up the vials and place them into the rack",
    help="Language instruction for the policy",
)
parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")

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


import gymnasium as gym
import numpy as np
import torch

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

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")

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

            if step < 10:  # warmup
                actions[:] = initial_action
            else:
                joint_pos_left = obs["policy"]["joint_pos_left"][0].clone()
                joint_pos_right = obs["policy"]["joint_pos_right"][0].clone()
                actions[:] = policy.get_action(
                    joint_pos_left, joint_pos_right, obs["visual"], log=True
                )

            obs, rewards, terminated, truncated, info = env.step(actions)

            step += 1

            if pbar is not None:
                pbar.update(1)

            is_terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
            is_truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()

            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None

                num_episodes += 1
                if is_terminated and not is_truncated:
                    num_successes += 1
                success_rate = (num_successes / num_episodes) * 100

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
                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            if num_episodes >= args_cli.num_episodes:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes")
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)")
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":

    main()

    while True:
        simulation_app.update()

    simulation_app.close()
