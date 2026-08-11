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
#
# port 走 udev 固定名稱（`graphen-setup-udev --apply` 一次設定，之後插拔都不會變），
# 校正檔與真機共用 lerobot/calibration/bimanual_leader/ 底下同樣那兩個 JSON。
# 平常不需要設任何環境變數；要覆寫才用 TELEOP_PORT_LEFT / TELEOP_ID_LEFT / ...
import argparse
import os
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab dual-arm SO-101 teleop + record.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Dual-Vials-To-Rack")
# 左臂 leader
parser.add_argument("--port_left", type=str, default=os.getenv("TELEOP_PORT_LEFT", "/dev/ttyLeaderLeft"))
parser.add_argument("--id_left", type=str,
                    default=os.getenv("TELEOP_ID_LEFT", "bimanual_so101_leader_left"))
# 右臂 leader
parser.add_argument("--port_right", type=str, default=os.getenv("TELEOP_PORT_RIGHT", "/dev/ttyLeaderRight"))
parser.add_argument("--id_right", type=str,
                    default=os.getenv("TELEOP_ID_RIGHT", "bimanual_so101_leader_right"))
# 校正檔目錄 —— 與真機共用同一份，不要讓它掉回 LeRobot 的 HF cache
parser.add_argument("--calibration_dir", type=str,
                    default=os.getenv("TELEOP_CALIBRATION_DIR",
                                      "/home/graphen/sim2real/lerobot/calibration/bimanual_leader"))
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


def _check_calibration(robot_id, calibration_dir):
    """校正檔不在就直接停,不要讓 LeRobot 靜靜地跳進「重新校準」流程。

    最常見的原因是 shell 裡還留著舊的 TELEOP_ID_* / TELEOP_CALIBRATION_DIR export
    —— 它們會蓋掉本檔的預設值,於是去找一個不存在的檔名。
    """
    if not calibration_dir:
        return
    fpath = Path(calibration_dir) / f"{robot_id}.json"
    if fpath.is_file():
        print(f"[INFO]: calibration {fpath}")
        return
    raise SystemExit(
        f"\n[ERROR]: 找不到校正檔 {fpath}\n"
        f"         id={robot_id}  calibration_dir={calibration_dir}\n"
        f"         這兩個值可被環境變數覆寫 —— 先檢查 `env | grep TELEOP`,\n"
        f"         有殘留就 `unset TELEOP_PORT_LEFT TELEOP_ID_LEFT TELEOP_PORT_RIGHT "
        f"TELEOP_ID_RIGHT TELEOP_CALIBRATION_DIR`。\n"
        f"         校正指令見 run_cheatsheet.md 第 2 步。\n"
    )


def _make_leader(device, port, robot_id, calibration_dir=None):
    _check_calibration(robot_id, calibration_dir)
    iface = LeRobotSO101Interface(
        device=device, port=port, id=robot_id, cameras={}, fps=30, kind="leader",
        calibration_dir=calibration_dir,
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
    iface_left = _make_leader(env.unwrapped.device, args_cli.port_left, args_cli.id_left,
                              args_cli.calibration_dir)
    print(f"[INFO]: RIGHT leader port={args_cli.port_right} id={args_cli.id_right}")
    iface_right = _make_leader(env.unwrapped.device, args_cli.port_right, args_cli.id_right,
                               args_cli.calibration_dir)

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
        # fps 從 env 推導，不要寫死。錄製迴圈是「每個 env step 推一格」，所以資料集的
        # fps 標籤只有等於 env 控制率才是對的；寫死 30 而 env 跑 60 Hz 的話，時間戳會
        # 把 9.5 秒的模擬標成 19 秒，之後 GR00T eval 照資料集 fps 送 action 就會差一倍。
        # 目前 = 1/(sim.dt 1/120 x decimation 4) = 30 Hz，和真機 RealSense 的 30 fps 一致。
        fps = round(1.0 / env.unwrapped.step_dt)
        if abs(fps * env.unwrapped.step_dt - 1.0) > 1e-6:
            raise ValueError(
                f"env 控制率 {1.0 / env.unwrapped.step_dt:.3f} Hz 不是整數，無法當資料集 fps。"
                "請調整 sim.dt / decimation 讓 1/(sim.dt * decimation) 為整數。"
            )
        print(f"[INFO]: recording at {fps} fps (= env control rate)")
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=fps,
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
