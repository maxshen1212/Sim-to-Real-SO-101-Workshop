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
import torch


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def _to_torch(x: "torch.Tensor") -> torch.Tensor:
    """Materialize Isaac Lab 6.x warp-backed ``ProxyArray`` fields to a real ``torch.Tensor``.

    Many ``asset.data.*`` fields now return a ``ProxyArray`` (warp-backed, lazy). It behaves like a
    tensor for eager torch ops but is rejected by torchscript math utils (``quat_inv`` etc.), which
    strictly require ``torch.Tensor``. ``ProxyArray`` exposes a ``.torch`` property that materializes
    (and caches) the tensor; real tensors have no such attribute and pass through unchanged.
    """
    return x.torch if hasattr(x, "torch") else x


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.
    """
    robot = env.scene[robot_cfg.name]
    robot_root_pos = _to_torch(robot.data.root_pos_w)
    robot_root_quat = _to_torch(robot.data.root_quat_w)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = _to_torch(ee_frame.data.target_pos_w[:, 0, :])
    ee_frame_quat = _to_torch(ee_frame.data.target_quat_w[:, 0, :])
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    ee_frame_state = torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)

    return ee_frame_state


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:

    sensor = env.scene[sensor_cfg.name]
    images = sensor.data.output[data_type]

    return images.clone()