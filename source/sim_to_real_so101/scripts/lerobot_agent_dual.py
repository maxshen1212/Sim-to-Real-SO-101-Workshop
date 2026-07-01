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
#
# 雙臂遙操作「驅動」版（Phase 4）：兩支 SO-101 leader → 12 維動作 → 雙臂 sim。
# 目前只驅動、不錄製（先用來驗證抓/放與雙臂操作；dataset 錄製之後再加）。
#
# port 每次插拔會變，用 `lerobot-find-port` 找出後 export：
#   export TELEOP_PORT_LEFT=/dev/ttyACM0   TELEOP_ID_LEFT=<左臂校正id>
#   export TELEOP_PORT_RIGHT=/dev/ttyACM1  TELEOP_ID_RIGHT=<右臂校正id>
#   sudo chmod 666 $TELEOP_PORT_LEFT $TELEOP_PORT_RIGHT
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab dual-arm SO-101 teleop (drive only).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Dual-Vials-To-Rack")
# 左臂 leader
parser.add_argument("--port_left", type=str, default=os.getenv("TELEOP_PORT_LEFT", "/dev/ttyACM0"))
parser.add_argument("--id_left", type=str, default=os.getenv("TELEOP_ID_LEFT", "leader_left"))
# 右臂 leader
parser.add_argument("--port_right", type=str, default=os.getenv("TELEOP_PORT_RIGHT", "/dev/ttyACM1"))
parser.add_argument("--id_right", type=str, default=os.getenv("TELEOP_ID_RIGHT", "leader_right"))
parser.add_argument("--seed", type=int, default=101)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 雖然不錄製，仍開相機讓三相機正常初始化
args_cli.enable_cameras = True

# Launch Isaac Sim Simulator first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface


def _make_leader(device, port, robot_id):
    iface = LeRobotSO101Interface(
        device=device,
        port=port,
        id=robot_id,
        cameras={},          # 驅動不需相機（leader config 不吃 cameras）
        fps=30,
        kind="leader",
    )
    iface.init_device()
    iface.connect()
    return iface


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")
    env.reset()

    # 兩支 leader
    print(f"[INFO]: Connecting LEFT leader  port={args_cli.port_left}  id={args_cli.id_left}")
    iface_left = _make_leader(env.unwrapped.device, args_cli.port_left, args_cli.id_left)
    print(f"[INFO]: Connecting RIGHT leader port={args_cli.port_right} id={args_cli.id_right}")
    iface_right = _make_leader(env.unwrapped.device, args_cli.port_right, args_cli.id_right)

    # 動作 tensor：12 維（左 0:6、右 6:12，順序同 DualActionsCfg 定義）
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            # 讀兩支 leader → 各自映射成 sim 弧度
            _, mapped_left = iface_left.real_to_sim_obs_processor(
                iface_left.robot.get_action()
            )
            _, mapped_right = iface_right.real_to_sim_obs_processor(
                iface_right.robot.get_action()
            )
            actions[..., 0:6] = mapped_left
            actions[..., 6:12] = mapped_right

            env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()
    simulation_app.close()
