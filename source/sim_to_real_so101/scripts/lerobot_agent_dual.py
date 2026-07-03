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
# 雙臂遙操作 + 錄製（Phase 4）：兩支 SO-101 leader → 12 維動作 → 雙臂 sim，
# 按 S 起停錄製，寫成 12 維(left_*/right_*) state/action + 3 相機的 LeRobotDataset。
#
# port 每次插拔會變，用 `lerobot-find-port` 找出後 export：
#   export TELEOP_PORT_LEFT=/dev/ttyACM0   TELEOP_ID_LEFT=<左臂校正id>
#   export TELEOP_PORT_RIGHT=/dev/ttyACM1  TELEOP_ID_RIGHT=<右臂校正id>
#   sudo chmod 666 $TELEOP_PORT_LEFT $TELEOP_PORT_RIGHT
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab dual-arm SO-101 teleop + record.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Dual-Vials-To-Rack")
# 左臂 leader
parser.add_argument("--port_left", type=str, default=os.getenv("TELEOP_PORT_LEFT", "/dev/ttyACM0"))
parser.add_argument("--id_left", type=str, default=os.getenv("TELEOP_ID_LEFT", "leader_left"))
# 右臂 leader
parser.add_argument("--port_right", type=str, default=os.getenv("TELEOP_PORT_RIGHT", "/dev/ttyACM1"))
parser.add_argument("--id_right", type=str, default=os.getenv("TELEOP_ID_RIGHT", "leader_right"))
# 錄製（三個都給才會啟用）
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--depth", action="store_true", default=False, help="也存 depth 進 buffer")
parser.add_argument("--seed", type=int, default=101)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder


def _make_leader(device, port, robot_id):
    iface = LeRobotSO101Interface(
        device=device, port=port, id=robot_id, cameras={}, fps=30, kind="leader",
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
    print(f"[INFO]: 'R' reset world | 'S' start/stop recording")
    env.reset()

    # 發現三相機（camera_ 前綴），供 recorder 用
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cam_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": cam_cfg.height, "width": cam_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")

    # 兩支 leader
    print(f"[INFO]: LEFT  leader port={args_cli.port_left}  id={args_cli.id_left}")
    iface_left = _make_leader(env.unwrapped.device, args_cli.port_left, args_cli.id_left)
    print(f"[INFO]: RIGHT leader port={args_cli.port_right} id={args_cli.id_right}")
    iface_right = _make_leader(env.unwrapped.device, args_cli.port_right, args_cli.id_right)

    # 12 維關節命名：left_* 然後 right_*（順序同 DualActionsCfg：左 0:6、右 6:12）
    dual_joint_names = (
        [f"left_{n}" for n in iface_left.SO101_JOINT_ORDER]
        + [f"right_{n}" for n in iface_right.SO101_JOINT_ORDER]
    )

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # 錄製設定
    recording_mode = all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name])
    recorder = None
    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=False,
            depth=args_cli.depth,
            instance_id_seg=False,   # 雙臂任務目前沒收 instance seg
            joint_names=dual_joint_names,
        )
        try:
            recorder.init_dataset()
        except ValueError:
            print("[ERROR]: dataset folder already exists")
            env.close()
            simulation_app.close()

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # 讀兩支 leader：real_* = 原始正規化讀值(給 dataset action)，mapped_* = sim 弧度(給 env.step)
                real_left, mapped_left = iface_left.real_to_sim_obs_processor(
                    iface_left.robot.get_action()
                )
                real_right, mapped_right = iface_right.real_to_sim_obs_processor(
                    iface_right.robot.get_action()
                )
                actions[..., 0:6] = mapped_left
                actions[..., 6:12] = mapped_right

                obs, _, _, _, _ = env.step(actions)

                if keyboard_control.reset_world:
                    keyboard_control.reset_world = False
                    env.reset()
                    continue

                if recording_mode and keyboard_control.recording:
                    visual_obs = obs.get("visual", None)
                    if visual_obs is None:
                        print("[WARNING]: no 'visual' obs group — recording needs cameras")
                        keyboard_control.recording = False
                        continue

                    # 12 維 state：兩臂 sim 關節弧度 → 正規化
                    jl = obs["policy"]["joint_pos_left"][0]
                    jr = obs["policy"]["joint_pos_right"][0]
                    real_obs = torch.cat([
                        iface_left.get_raw_actions_from_radians(jl),
                        iface_right.get_raw_actions_from_radians(jr),
                    ])
                    # 12 維 action：兩支 leader 原始讀值
                    action_rec = torch.cat([real_left, real_right])

                    # 三相機 rgb（recorder 的相機 key = 去掉 camera_ 前綴；obs key = rgb_<key>）
                    rgb_buffers = {c: visual_obs[f"rgb_{c}"][0] for c in cameras}
                    depth_buffers = {}
                    if args_cli.depth:
                        depth_buffers = {c: visual_obs[f"depth_{c}"][0] for c in cameras}

                    recorder.push_frame_to_buffer(
                        action_rec, real_obs, rgb_buffers, depth_buffers, {}
                    )
    except KeyboardInterrupt:
        print("[INFO]: Ctrl+C received - finishing pending episode saves before exit...")
    finally:
        # Flush any queued/encoding episodes before tearing down the sim, otherwise
        # the last episode's parquet/mp4 can be truncated and corrupt the dataset.
        if recording_mode:
            recorder.close()
        env.close()


if __name__ == "__main__":
    main()
    
    simulation_app.close()
