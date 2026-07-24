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
import math
import torch
import glob
import os
import yaml

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import get_current_stage
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
# Isaac Sim 6.0 replaced the old importable `isaacsim.core.prims.XFormPrim` with
# `isaacsim.core.experimental.prims.XformPrim` (note the casing), which lives in an extension that
# is NOT enabled by default, so a plain import raises ModuleNotFoundError. Enable it first. This is
# safe here because mdp modules are only imported after the simulator app is launched (see the
# scripts' AppLauncher-first convention), so the Kit extension manager is already available.
# API deltas handled at the call sites below:
#   - constructor takes `paths` positionally (not `prim_paths_expr=`)
#   - get_local_poses() returns warp arrays (converted to torch here)
#   - set_local_poses(orientations=...) wants list/np.ndarray/wp.array (torch is converted to numpy)
import omni.kit.app as _omni_kit_app

_omni_kit_app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.core.experimental.prims", True
)
from isaacsim.core.experimental.prims import XformPrim as XFormPrim  # noqa: E402




# Robot color palette based on WowRobo and Seeed Studio offerings
ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def randomize_robot_color(env,
    env_ids: torch.Tensor | None,
    color_names: list[str] = list(ROBOT_COLORS.keys()),
    robot_names: tuple[str, ...] = ("robot",),
):
    """Randomly set robot color from predefined palette on each reset.

    Each robot in ``robot_names`` is drawn an independent color, so the same
    term serves single-arm (default ``("robot",)``) and dual-arm
    (``("robot_left", "robot_right")``) scenes. Colour is appearance-only DR,
    so the two arms deliberately need not match.
    """
    with Sdf.ChangeBlock():
        for robot_name in robot_names:
            idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
            selected_color = ROBOT_COLORS[color_names[idx]]
            robot = env.scene[robot_name]
            material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
            material_prim = sim_utils.find_matching_prims(material_prim_path)[0]
            material_prim.GetAttribute("inputs:diffuse_color_constant").Set(selected_color)


# Vial-rack color palette for appearance DR. Includes the asset's original
# yellow plus saturated primaries and neutral/dark tones (incl. black) so the
# policy sees the rack in many colors.
RACK_COLORS = {
    "yellow": (0.93822396, 0.58649296, 0.04346985),  # original Vial_rack_simple.usda color
    "red": (0.8, 0.12, 0.1),
    "blue": (0.1, 0.22, 0.8),
    "green": (0.12, 0.6, 0.2),
    "orange": (0.876, 0.317, 0.132),
    "black": (0.02, 0.02, 0.02),
    "gray": (0.45, 0.45, 0.45),
    "white": (0.9, 0.9, 0.9),
}


def randomize_rack_color(env,
    env_ids: torch.Tensor | None,
    color_names: list[str] = list(RACK_COLORS.keys()),
    rack_names: tuple[str, ...] = ("rack_left",),
):
    """Randomly set the vial-rack color from a predefined palette on each reset.

    Mirrors :func:`randomize_robot_color`: appearance-only DR that edits the
    rack's OmniPBR shader ``diffuse_color_constant`` via ``Sdf.ChangeBlock``.
    Each rack in ``rack_names`` is drawn an independent color.
    """
    with Sdf.ChangeBlock():
        for rack_name in rack_names:
            idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
            selected_color = RACK_COLORS[color_names[idx]]
            rack = env.scene[rack_name]
            shader_prim_path = rack.cfg.prim_path + "/Looks/OmniPBR/Shader"
            shader_prim = sim_utils.find_matching_prims(shader_prim_path)[0]
            shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(selected_color)


def randomize_vial_transparency(env,
    env_ids: torch.Tensor | None,
    opacity_range: tuple[float, float] = (0.08, 0.5),
    vials: tuple[str, ...] = ("vial_1", "vial_2", "vial_3"),
):
    """Randomize per-vial body opacity on each reset (see-through glass DR).

    The vial body material is an OmniPBR shader with ``enable_opacity`` on (see
    ``Vial_opaque.usda``); this term samples a fresh ``opacity_constant`` in
    ``opacity_range`` (0 = fully transparent, 1 = opaque) so the policy sees
    vials ranging from nearly clear glass to translucent. Each vial is drawn an
    independent opacity.
    """
    with Sdf.ChangeBlock():
        for vial_name in vials:
            opacity = math_utils.sample_uniform(
                opacity_range[0], opacity_range[1], (1,), device="cpu"
            ).item()
            vial = env.scene[vial_name]
            shader_prim_path = vial.cfg.prim_path + "/Looks/Vial_plastic/Shader"
            shader_prim = sim_utils.find_matching_prims(shader_prim_path)[0]
            shader_prim.GetAttribute("inputs:opacity_constant").Set(opacity)


def randomize_mat_rotation(
    env,
    env_ids: torch.Tensor | None,
    yaw_range: tuple[float, float] = (-0.3, 0.3),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize mat yaw rotation on reset.
    
    Args:
        yaw_range: Range of yaw rotation in radians (default ±0.3 rad ≈ ±17°).
    """
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]
    
    # Sample random yaw for each environment
    yaw = math_utils.sample_uniform(
        yaw_range[0], yaw_range[1], (len(env_ids),), device=env.unwrapped.device
    )
    # Keep roll and pitch at 0, add 90° base yaw (mat's default orientation)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    base_yaw = torch.full_like(yaw, math.pi / 2)  # 90° in radians
    
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw)
    
    asset_xform = XFormPrim(asset_prim_path)

    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations.detach().cpu().numpy())


def randomize_camera_focal_length(
    env,
    env_ids: torch.Tensor | None,
    focal_length_range: tuple[float, float] = (12.0, 15.0),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize camera focal length on reset (affects field of view).
    
    Args:
        focal_length_range: Range of focal lengths in mm (default 12-15mm).
            Lower = wider FOV, higher = narrower FOV.
    """
    stage = get_current_stage()
    camera = env.scene[asset_cfg.name]
    camera_prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    
    focal_length = math_utils.sample_uniform(
        focal_length_range[0], focal_length_range[1], (1,), device="cpu"
    ).item()
    
    camera_prims = sim_utils.find_matching_prims(camera_prim_path)
    
    with Sdf.ChangeBlock():
        for prim in camera_prims:
            if prim.IsValid():
                focal_attr = prim.GetAttribute("focalLength")
                if focal_attr.IsValid():
                    focal_attr.Set(focal_length)


# Cache for storing default poses read from USD
_default_poses_cache = {}


def randomize_camera_pose(
    env,
    env_ids: torch.Tensor | None,
    prim_path_pattern: str,
    pos_range: dict[str, tuple[float, float]] = None,
    rot_range: dict[str, tuple[float, float]] = None,
):
    """Randomize camera mount position and orientation relative to its USD default.
    
    Args:
        prim_path_pattern: Prim path pattern for the camera mount xform.
        pos_range: Dict with x, y, z keys and (min, max) tuple values in meters.
        rot_range: Dict with roll, pitch, yaw keys and (min, max) tuple values in radians.
    """
    if pos_range is None:
        pos_range = {}
    if rot_range is None:
        rot_range = {}
    
    stage = get_current_stage()
    prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_path)
    
    if not prims:
        return
    
    # Read and cache default pose from first prim on first call
    if prim_path_pattern not in _default_poses_cache:
        prim = prims[0]
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        
        default_pos = (0.0, 0.0, 0.0)
        default_quat = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
        
        if translate_attr.IsValid():
            val = translate_attr.Get()
            if val is not None:
                default_pos = (val[0], val[1], val[2])
        
        if orient_attr.IsValid():
            val = orient_attr.Get()
            if val is not None:
                default_quat = (val.GetReal(), val.GetImaginary()[0], val.GetImaginary()[1], val.GetImaginary()[2])
        
        _default_poses_cache[prim_path_pattern] = {"pos": default_pos, "quat": default_quat}
    
    base_pos = _default_poses_cache[prim_path_pattern]["pos"]
    base_quat = _default_poses_cache[prim_path_pattern]["quat"]
    
    # Sample random offsets
    x = base_pos[0] + math_utils.sample_uniform(*pos_range.get("x", (0, 0)), (1,), device="cpu").item()
    y = base_pos[1] + math_utils.sample_uniform(*pos_range.get("y", (0, 0)), (1,), device="cpu").item()
    z = base_pos[2] + math_utils.sample_uniform(*pos_range.get("z", (0, 0)), (1,), device="cpu").item()
    
    # Sample rotation offsets and combine with base quaternion
    roll = math_utils.sample_uniform(*rot_range.get("roll", (0, 0)), (1,), device="cpu").item()
    pitch = math_utils.sample_uniform(*rot_range.get("pitch", (0, 0)), (1,), device="cpu").item()
    yaw = math_utils.sample_uniform(*rot_range.get("yaw", (0, 0)), (1,), device="cpu").item()
    
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw])
    )[0]
    
    # Combine base quaternion with delta
    base_quat_tensor = torch.tensor([base_quat])
    final_quat = math_utils.quat_mul(base_quat_tensor, delta_quat.unsqueeze(0))[0]
    
    with Sdf.ChangeBlock():
        for prim in prims:
            if prim.IsValid():
                # Set translation
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(x, y, z))
                # Set orientation
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid():
                    orient_attr.Set(Gf.Quatd(final_quat[0].item(), final_quat[1].item(), final_quat[2].item(), final_quat[3].item()))


def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)


def randomize_sky_light(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    temperature_range: tuple[float, float],
    textures_root: str,
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    temperature = math_utils.sample_uniform(*temperature_range, (1,), device="cpu").item()

    # list all .exr files in the textures_root
    textures = glob.glob(os.path.join(textures_root, "*.exr"))
    # get a random texture (using torch)

    if not textures:
        print("[WARNING] No textures found in the textures_root")
        return

    texture = torch.randint(0, len(textures), (1,), device="cpu").item()
    texture = textures[texture]

    yaw_mapping_path = os.path.join(textures_root, "yaw_mapping.yaml")
    with open(yaw_mapping_path, 'r') as f:
        yaw_mapping = yaml.safe_load(f)

    yaw_mapping = yaw_mapping.get(os.path.basename(texture), None)

    range_list = [
        (0.0, 0.0),
        (0.0, 0.0),
        (
            yaw_mapping[0] if yaw_mapping else 0.0,
            yaw_mapping[1] if yaw_mapping else 0.0,
        ),
    ]
    ranges = torch.tensor(range_list, device=env.unwrapped.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.unwrapped.device
    )
    orientations = math_utils.quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    asset_xform = XFormPrim(asset_prim_path)

    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations.detach().cpu().numpy())
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)
            prim.GetAttribute("inputs:colorTemperature").Set(temperature)
            texture = Sdf.AssetPath(texture)
            prim.GetAttribute("inputs:texture:file").Set(texture)


def random_asset_pose(
        env, 
        env_ids, 
        asset, 
        pose_range, 
        pos_offset
):

    root_states = asset.data.default_root_state.torch[env_ids].clone()
    pos_offset_list = [pos_offset.get(key, 0.0) for key in ["x", "y", "z"]]
    pos_offset = torch.tensor(pos_offset_list, device=asset.device)
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + rand_samples[:, 0:3]
        + pos_offset
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return positions, orientations


def reset_vials_rack(
        env,
        env_ids: torch.Tensor,
        vials: list[str],
        rack: str,
        rack_pose_range: dict[str, tuple[float, float]],
        pose_range: dict[str, tuple[float, float]],
        fixed_vial_z: float,
        rack_placement_prob: float = 0.33,
):

    vial_objects: list[RigidObject | Articulation] = [
        env.scene[asset_name] for asset_name in vials
    ]

    rack = env.scene[rack]
    # NOTE: experimental XformPrim treats the path as a *regex* (not a glob like the old API), so the
    # slot prims top_01..top_04 must be matched with `top_.*`, not the glob `top_*`.
    slots_xform_view = XFormPrim(f"{rack.cfg.prim_path}/Body1/Mesh/top_.*")
    total_slots = len(slots_xform_view)

    # randomize rack pose
    new_rack_positions, new_rack_orientations = random_asset_pose(env, env_ids, rack, rack_pose_range, {})
    # Clear velocities immediately after positioning
    zero_velocity = torch.zeros((len(env_ids), 6), device=rack.device)
    rack.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    placed_on_rack_indices = []
    if torch.rand(1, device=env.unwrapped.device).item() < rack_placement_prob:
        # Pick a random vial to place on the rack
        vial_idx = torch.randint(0, len(vial_objects), (1,), device=env.unwrapped.device).item()
        placed_on_rack_indices.append(vial_idx)
    
    # experimental XformPrim.get_local_poses() returns warp arrays; downstream code needs torch
    # tensors on the env device (uses .unsqueeze/.repeat and math_utils.combine_frame_transforms).
    _slot_pos_wp, _slot_ori_wp = slots_xform_view.get_local_poses()
    slot_positions_local = torch.as_tensor(_slot_pos_wp.numpy(), device=env.unwrapped.device)
    slot_orientations_local = torch.as_tensor(_slot_ori_wp.numpy(), device=env.unwrapped.device)
    # Place selected vials on rack
    for vial_idx in placed_on_rack_indices:
        vial = vial_objects[vial_idx]
        
        # Select a random slot for this vial
        slot_idx = torch.randint(0, total_slots, (1,), device=env.unwrapped.device).item()
        
        # Thank you Sonnet lol
        # IMPORTANT: Cannot use get_world_poses() here because write_root_pose_to_sim() doesn't
        # update USD until sim.step(). Instead, manually compute slot positions from local transforms.
        # Transform slot positions from rack local frame to world frame using the NEW rack pose
        # For each environment, we need to transform the slot position
        slot_position_local = slot_positions_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        slot_orientation_local = slot_orientations_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        
        # Combine transforms: world_pose = rack_pose ⊕ slot_local_pose
        slot_position, slot_orientation = math_utils.combine_frame_transforms(
            new_rack_positions, new_rack_orientations, slot_position_local, slot_orientation_local
        )
        # slot_position and slot_orientation are already per-env tensors (shape: len(env_ids), 3/4)
        slot_pose = torch.cat([slot_position, slot_orientation], dim=-1)
        vial.write_root_pose_to_sim(slot_pose, env_ids=env_ids)

        zero_velocity = torch.zeros((len(env_ids), 6), device=vial.device)
        vial.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    pose_range_z_fixed = {**pose_range, "z": (0.0, 0.0)}
    for i, v in enumerate(vial_objects):
        if i not in placed_on_rack_indices:
            default_z = v.data.default_root_state.torch[env_ids[0], 2].item()
            pos_offset = {"z": fixed_vial_z - default_z}
            _, _ = random_asset_pose(env, env_ids, v, pose_range_z_fixed, pos_offset)
            zero_velocity = torch.zeros((len(env_ids), 6), device=v.device)
            v.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def hide_random_vials(
        env,
        env_ids: torch.Tensor,
        vials: list[str],
        min_count: int = 1,
        max_count: int = 4,
        hide_xy: tuple[float, float] = (2.0, 2.0),
        hide_z: float = 0.05,
):
    """Keep a random number of vials on the mat, park the rest off-scene.

    Meant to run *after* the normal placement (:func:`reset_vials_rack`): the
    placement leaves all vials on the mat, then this term picks a per-env random
    count in ``[min_count, max_count]`` to keep and teleports the remaining vials
    far away (``hide_xy``, ~2 m out) so they are out of every camera's view and
    unreachable. Each hidden vial gets a distinct spot and zeroed velocity.

    The grasp/placement observations consider all vials, but a parked vial is
    never contacted or placed, so effectively only the on-mat vials count.
    """
    vial_objects: list[RigidObject | Articulation] = [env.scene[name] for name in vials]
    n_vials = len(vial_objects)
    E = len(env_ids)
    device = env.unwrapped.device

    # per-env: mark which vials are hidden this reset (mask on env_ids' device
    # so the boolean indexing below is valid)
    hidden_mask = torch.zeros((n_vials, E), dtype=torch.bool, device=env_ids.device)
    for e in range(E):
        n_keep = torch.randint(min_count, max_count + 1, (1,)).item()
        for vi in torch.randperm(n_vials).tolist()[n_keep:]:
            hidden_mask[vi, e] = True

    for vi, v in enumerate(vial_objects):
        env_subset = env_ids[hidden_mask[vi]]
        if len(env_subset) == 0:
            continue
        k = len(env_subset)
        pos = torch.tensor(
            [hide_xy[0] + 0.1 * vi, hide_xy[1], hide_z], device=device
        ).unsqueeze(0).repeat(k, 1) + env.scene.env_origins[env_subset]
        quat = v.data.default_root_state.torch[env_subset][:, 3:7]
        pose = torch.cat([pos, quat], dim=-1)
        v.write_root_pose_to_sim(pose, env_ids=env_subset)
        v.write_root_velocity_to_sim(torch.zeros((k, 6), device=v.device), env_ids=env_subset)


def reset_joints_to_pose(
    env,
    env_ids: torch.Tensor,
    target_pos: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
):
    """把選定關節「絕對」重設到 target_pos（弧度），不參照 USD 的預設姿態。

    對照 isaaclab 內建的 reset_joints_by_offset：那個是「USD 預設姿態 + 偏移」，起始姿態
    會被 USD 的 init pose 綁死。這個版本直接寫入絕對目標姿態，讓起始姿態由呼叫端（task cfg）
    決定，與 USD init pose 脫鉤——用於讓 eval 的開機姿態對齊訓練資料集的起始姿態。

    target_pos 依 asset_cfg.joint_names 的順序給值。asset_cfg 請設 preserve_order=True，
    這樣 joint_ids 會維持 joint_names 的順序、與 target_pos 一一對應（否則 joint_ids 會被
    排序，導致對位錯亂）。

    position_range / velocity_range 為選用：在目標值附近加均勻噪聲，可複刻錄製時的自然變異；
    預設 (0, 0) 即固定姿態。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # target_pos 展開成 (欲重設的環境數, 關節數)
    target = torch.tensor(target_pos, dtype=torch.float32, device=asset.device)
    joint_pos = target.unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)

    # 選用：在目標姿態附近加均勻噪聲
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # 夾到關節上下限，避免噪聲把姿態推出可動範圍
    joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

    # 寫進物理引擎
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
