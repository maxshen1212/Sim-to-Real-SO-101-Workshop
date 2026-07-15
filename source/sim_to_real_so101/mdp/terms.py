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
import os

import torch

from pxr import Gf, Sdf


import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg



def any_vial_grasped(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    vials: list[str],
    min_height: float = 0.01,
    warmup_steps: int = 30,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if any vial is currently grasped by the gripper.
    
    A vial is considered grasped if:
    1. The gripper has contact with it (force above threshold)
    2. The vial is above the minimum height threshold (only checked for initial grasp)
    3. We are past the warmup period
    
    Once grasped, stays grasped as long as contact is maintained (height no longer matters).
    Released only when contact is lost.
    
    Args:
        env: The environment instance.
        contact_sensor_cfg: Configuration for the contact sensor.
        vials: List of vial asset names to check.
        min_height: Minimum Z height (meters) for vial to be considered lifted. Default 1cm.
        warmup_steps: Number of initial steps to ignore (warmup period). Default 30.
        force_threshold: Minimum contact force (Newtons) to detect contact.
    
    Returns:
        Boolean tensor of shape (num_envs, 1) indicating grasp status per environment.
    """
    num_envs = env.num_envs
    device = env.device
    
    # Per-contact-sensor state, keyed by sensor name, so multiple grippers
    # (e.g. dual-arm left/right) don't clobber each other's grasp state.
    key = contact_sensor_cfg.name
    if not hasattr(any_vial_grasped, "_state"):
        any_vial_grasped._state = {}
    if key not in any_vial_grasped._state:
        any_vial_grasped._state[key] = {
            "prev_grasped": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "is_holding": torch.zeros(num_envs, dtype=torch.bool, device=device),
        }
    st = any_vial_grasped._state[key]
    
    # Check warmup: no vial can be grasped in the first N steps
    current_step = env.episode_length_buf
    in_warmup = current_step < warmup_steps
    
    # Reset holding state for environments that just reset (step 0 or 1)
    just_reset = current_step <= 1
    st["is_holding"][just_reset] = False
    st["prev_grasped"][just_reset] = False
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    
    # Get contact forces - shape: (num_envs, num_bodies, num_filters, 3)
    contact_forces = contact_sensor.data.force_matrix_w
    
    # Calculate force magnitude per filter (each filter corresponds to a vial)
    # Shape: (num_envs, num_bodies, num_filters)
    contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    
    # Sum over bodies to get per-env, per-filter contact detection
    # Shape: (num_envs, num_filters)
    contact_per_filter = contact_force_norm.sum(dim=1)
    
    # Track contact and lift status across all vials
    any_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    new_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    for vial_idx, vial_name in enumerate(vials):
        vial: RigidObject = env.scene[vial_name]
        vial_z = vial.data.root_pos_w[:, 2]  # Z position per environment
        
        has_contact_with_vial = contact_per_filter[:, vial_idx] > force_threshold
        vial_is_lifted = vial_z > min_height
        
        # Track if we have contact with any vial
        any_contact = any_contact | has_contact_with_vial
        
        # New grasp: contact + lifted + not already holding
        new_grasp = new_grasp | (has_contact_with_vial & vial_is_lifted & (~st["is_holding"]))
    
    # Update holding state with hysteresis:
    # - Start holding: new grasp detected (contact + lifted)
    # - Keep holding: was holding AND still have contact
    # - Stop holding: was holding AND lost contact
    was_holding = st["is_holding"].clone()
    st["is_holding"] = (was_holding & any_contact) | new_grasp
    
    # Apply warmup mask
    is_grasped = st["is_holding"] & (~in_warmup)
    
    # Debug: print state transitions
    prev = st["prev_grasped"]
    just_grasped = is_grasped & (~prev)
    just_released = (~is_grasped) & prev
    
    if just_grasped.any():
        env_ids = torch.where(just_grasped)[0].tolist()
        print(f"[GRASP] Vial grasped in env(s): {env_ids}")
    
    if just_released.any():
        env_ids = torch.where(just_released)[0].tolist()
        print(f"[RELEASE] Vial released in env(s): {env_ids}")
    
    # Update previous state
    st["prev_grasped"] = is_grasped.clone()
    
    # Return as float tensor with shape (num_envs, 1) for observation
    return is_grasped.float().unsqueeze(-1)




def vial_placed_on_rack(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    vials: list[str],
    rack_name: str,
    warmup_steps: int = 30,
    grasp_history_window: int = 20,
    force_threshold: float = 2.0,
    # Rack local dimensions (from Vial_rack_simple.usda extent: 0→0.12 x, 0→0.12 y)
    rack_local_x_min: float = 0.0,
    rack_local_x_max: float = 0.12,
    rack_local_y_min: float = 0.0,
    rack_local_y_max: float = 0.12,
    # Slot entries are at local z=0.1; rack body top at z=0.073
    rack_local_z_max: float = 0.1,
    # Orientation: abs(vial_up_z) must exceed this to count as "vertical" (in rack)
    vertical_threshold: float = 0.7,
) -> torch.Tensor:
    """Check if a vial has been placed into the rack.

    A vial is considered placed in the rack if:
    1. We are past the warmup period
    2. The vial is approximately vertical — its local Z axis is roughly aligned
       with world Z (either up or down).  This distinguishes a vial sitting in a
       slot from one lying on its side on the rack surface.
    3. The vial's position in rack-local coordinates is within the rack's XY bounding box
    4. The vial's rack-local Z is below the slot entry level (rack_local_z_max)
    5. THIS SPECIFIC vial was grasped at some point in the last N steps
    6. THIS SPECIFIC vial is no longer grasped

    Args:
        env: The environment instance.
        contact_sensor_cfg: Configuration for the contact sensor.
        vials: List of vial asset names to check.
        rack_name: Name of the rack asset in the scene.
        warmup_steps: Number of initial steps to ignore. Default 30.
        grasp_history_window: Number of steps to track grasp history. Default 20.
        force_threshold: Minimum contact force (N) to detect grasp.
        rack_local_x_min: Rack local X minimum bound.
        rack_local_x_max: Rack local X maximum bound.
        rack_local_y_min: Rack local Y minimum bound.
        rack_local_y_max: Rack local Y maximum bound.
        rack_local_z_max: Maximum rack-local Z for vial center (slot entry level).
        vertical_threshold: The vial's up-vector projected onto world-Z must
            have abs value above this to be considered "vertical" (in the slot).
            Default 0.7 (~45° from vertical).

    Returns:
        Float tensor of shape (num_envs, 1) indicating placement status per environment.
    """
    num_envs = env.num_envs
    device = env.device
    num_vials = len(vials)

    # Per-contact-sensor state, keyed by sensor name, so dual-arm left/right
    # don't share grasp-history / placed-flags.
    key = contact_sensor_cfg.name
    if not hasattr(vial_placed_on_rack, "_state"):
        vial_placed_on_rack._state = {}
    if key not in vial_placed_on_rack._state:
        vial_placed_on_rack._state[key] = {
            "grasp_history": torch.zeros(
                num_envs, num_vials, grasp_history_window, dtype=torch.bool, device=device
            ),
            "history_idx": 0,
            "prev_placed": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "vial_placed_flags": torch.zeros(
                num_envs, num_vials, dtype=torch.bool, device=device
            ),
        }
    st = vial_placed_on_rack._state[key]

    current_step = env.episode_length_buf
    in_warmup = current_step < warmup_steps

    just_reset = current_step <= 1
    if just_reset.any():
        st["grasp_history"][just_reset] = False
        st["prev_placed"][just_reset] = False
        st["vial_placed_flags"][just_reset] = False

    # Get contact sensor for grasp detection
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.force_matrix_w
    contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    contact_per_filter = contact_force_norm.sum(dim=1)  # (num_envs, num_filters)

    # Get rack pose in world frame
    rack_obj: RigidObject = env.scene[rack_name]
    rack_pos_w = rack_obj.data.root_pos_w       # (num_envs, 3)
    rack_quat_w = rack_obj.data.root_quat_w     # (num_envs, 4)
    rack_quat_inv = math_utils.quat_inv(rack_quat_w)

    any_vial_newly_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Unit Z vector used for the vertical orientation check
    unit_z = torch.zeros(num_envs, 3, device=device)
    unit_z[:, 2] = 1.0

    for vial_idx, vial_name in enumerate(vials):
        vial: RigidObject = env.scene[vial_name]
        vial_pos_w = vial.data.root_pos_w       # (num_envs, 3)
        vial_quat_w = vial.data.root_quat_w     # (num_envs, 4)

        # --- Grasp detection ---
        vial_grasped_now = contact_per_filter[:, vial_idx] > force_threshold
        st["grasp_history"][:, vial_idx, st["history_idx"]] = vial_grasped_now
        vial_was_grasped_recently = st["grasp_history"][:, vial_idx, :].any(dim=1)

        # --- Vertical orientation check ---
        # Transform vial's local Z axis into world frame.  A vial sitting in a
        # rack slot will be roughly vertical (abs(z) close to 1), while one lying
        # on its side will have abs(z) close to 0.
        vial_up_world = math_utils.quat_apply(vial_quat_w, unit_z)
        is_vertical = torch.abs(vial_up_world[:, 2]) > vertical_threshold

        # --- Position check in rack-local coordinates ---
        vial_pos_relative = vial_pos_w - rack_pos_w
        vial_pos_local = math_utils.quat_apply(rack_quat_inv, vial_pos_relative)

        vial_local_x = vial_pos_local[:, 0]
        vial_local_y = vial_pos_local[:, 1]
        vial_local_z = vial_pos_local[:, 2]

        x_in_bounds = (vial_local_x >= rack_local_x_min) & (vial_local_x <= rack_local_x_max)
        y_in_bounds = (vial_local_y >= rack_local_y_min) & (vial_local_y <= rack_local_y_max)
        z_below_top = vial_local_z < rack_local_z_max

        position_ok = x_in_bounds & y_in_bounds & z_below_top

        # --- Combine all conditions ---
        vial_is_placed = (
            is_vertical
            & position_ok
            & vial_was_grasped_recently
            & (~vial_grasped_now)
            & (~in_warmup)
            & (~st["vial_placed_flags"][:, vial_idx])
        )

        newly_placed = vial_is_placed
        if newly_placed.any():
            env_ids = torch.where(newly_placed)[0].tolist()
            print(f"[RACK] {vial_name} placed in rack in env(s): {env_ids}")
            st["vial_placed_flags"][:, vial_idx] = (
                st["vial_placed_flags"][:, vial_idx] | newly_placed
            )

        any_vial_newly_placed = any_vial_newly_placed | newly_placed

    # Advance history ring buffer index
    st["history_idx"] = (st["history_idx"] + 1) % grasp_history_window

    any_placed = st["vial_placed_flags"].any(dim=1) & (~in_warmup)

    prev = st["prev_placed"]
    st["prev_placed"] = any_placed.clone()

    return any_placed.float().unsqueeze(-1)


def vial_placed_on_rack_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    vials: list[str],
    rack_name: str,
    warmup_steps: int = 30,
    grasp_history_window: int = 20,
    force_threshold: float = 2.0,
    rack_local_x_min: float = 0.0,
    rack_local_x_max: float = 0.12,
    rack_local_y_min: float = 0.0,
    rack_local_y_max: float = 0.12,
    rack_local_z_max: float = 0.1,
    vertical_threshold: float = 0.7,
    confirm_steps: int = 25,
) -> torch.Tensor:
    """Termination term for vial placed in rack.

    Calls the observation to detect the initial placement event, then
    re-evaluates live physics conditions (vertical, in-bounds, released)
    for ``confirm_steps`` consecutive steps before reporting termination.
    If any condition fails during confirmation the counter resets.

    Returns:
        Boolean tensor of shape (num_envs,) for termination.
    """
    num_envs = env.num_envs
    device = env.device

    result = vial_placed_on_rack(
        env=env,
        contact_sensor_cfg=contact_sensor_cfg,
        vials=vials,
        rack_name=rack_name,
        warmup_steps=warmup_steps,
        grasp_history_window=grasp_history_window,
        force_threshold=force_threshold,
        rack_local_x_min=rack_local_x_min,
        rack_local_x_max=rack_local_x_max,
        rack_local_y_min=rack_local_y_min,
        rack_local_y_max=rack_local_y_max,
        rack_local_z_max=rack_local_z_max,
        vertical_threshold=vertical_threshold,
    )
    trigger = result.squeeze(-1).bool()

    if not hasattr(env, "_rack_success_counter"):
        env._rack_success_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        env._rack_confirm_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    env._rack_success_counter[env.episode_length_buf <= 1] = 0
    env._rack_confirm_active[env.episode_length_buf <= 1] = False

    newly_triggered = trigger & (~env._rack_confirm_active)
    if newly_triggered.any():
        env._rack_confirm_active[newly_triggered] = True
        env._rack_success_counter[newly_triggered] = 0

    still_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if env._rack_confirm_active.any():
        contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
        contact_forces = contact_sensor.data.force_matrix_w
        contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
        contact_per_filter = contact_force_norm.sum(dim=1)

        rack_obj: RigidObject = env.scene[rack_name]
        rack_pos_w = rack_obj.data.root_pos_w
        rack_quat_w = rack_obj.data.root_quat_w
        rack_quat_inv = math_utils.quat_inv(rack_quat_w)

        unit_z = torch.zeros(num_envs, 3, device=device)
        unit_z[:, 2] = 1.0

        for vial_idx, vial_name in enumerate(vials):
            if not hasattr(vial_placed_on_rack, "_vial_placed_flags"):
                break
            was_placed = vial_placed_on_rack._vial_placed_flags[:, vial_idx]
            if not was_placed.any():
                continue

            vial_obj: RigidObject = env.scene[vial_name]
            vial_pos_w = vial_obj.data.root_pos_w
            vial_quat_w = vial_obj.data.root_quat_w

            vial_grasped_now = contact_per_filter[:, vial_idx] > force_threshold
            vial_up_world = math_utils.quat_apply(vial_quat_w, unit_z)
            is_vertical = torch.abs(vial_up_world[:, 2]) > vertical_threshold

            vial_pos_local = math_utils.quat_apply(rack_quat_inv, vial_pos_w - rack_pos_w)
            x_ok = (vial_pos_local[:, 0] >= rack_local_x_min) & (vial_pos_local[:, 0] <= rack_local_x_max)
            y_ok = (vial_pos_local[:, 1] >= rack_local_y_min) & (vial_pos_local[:, 1] <= rack_local_y_max)
            z_ok = vial_pos_local[:, 2] < rack_local_z_max

            vial_ok = was_placed & is_vertical & x_ok & y_ok & z_ok & (~vial_grasped_now)
            still_valid = still_valid | vial_ok

    env._rack_success_counter = torch.where(
        env._rack_confirm_active & still_valid,
        env._rack_success_counter + 1,
        torch.zeros_like(env._rack_success_counter),
    )
    env._rack_confirm_active[env._rack_success_counter == 0] = False

    confirmed = env._rack_success_counter >= confirm_steps
    # if (env._rack_success_counter > 0).any():
    #     print(f"[RACK CONFIRM] counter={env._rack_success_counter.tolist()} / {confirm_steps}")
    # if confirmed.any():
    #     print(f"[RACK CONFIRM] success confirmed in env(s): {torch.where(confirmed)[0].tolist()}")

    return confirmed


def all_active_vials_placed_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfgs,
    vials: list[str],
    rack_name: str,
    force_threshold: float = 2.0,
    rack_local_x_min: float = 0.0,
    rack_local_x_max: float = 0.12,
    rack_local_y_min: float = 0.0,
    rack_local_y_max: float = 0.12,
    rack_local_z_max: float = 0.1,
    vertical_threshold: float = 0.7,
    active_radius: float = 1.0,
    confirm_steps: int = 25,
) -> torch.Tensor:
    """雙臂協作式成功判定：所有「啟用中」的試管都同時放進共用架。

    與單臂 :func:`vial_placed_on_rack_termination` 的差異：

    * **跨左右兩個 contact sensor** 判斷「試管是否正被夾住」——任一夾爪對該試管
      有接觸力就算被夾（協作式任務任一臂都可能拿任一支）。
    * 成功條件是 **「所有啟用中的試管都在架上」**（而非單臂的「任一支放上去」）。
      「啟用中」= 沒被 :func:`hide_random_vials` 搬到場外的試管
      （場外約 2 m，用離 env 原點的水平距離判定）→ 天然支援 Phase 4B 的
      隨機 1~4 支：被藏起來的試管不計入成功條件。
    * 幾何準則（直立、在架子 local 範圍內、已放開）與 ``vial_placed_on_rack``
      觀測一致，確保 eval 成功判定與錄製時的 subtask 訊號同標準。
    * 需連續 ``confirm_steps`` 步都成立才回報成功；中途任一試管被撞出/被夾起
      就把計數歸零，避免「路過架子上方」誤判。

    Args:
        contact_sensor_cfgs: 左右夾爪 contact sensor 的 SceneEntityCfg（tuple/list，
            兩者的 filter 順序都須等於 ``vials``）。
        vials: 場景中全部試管名稱（順序對齊 contact sensor 的 filter）。
        rack_name: 共用試管架的 scene 物件名。
        active_radius: 離 env 原點水平距離小於此值才算「啟用中」（用來排除被藏到
            場外的試管）。

    Returns:
        Boolean tensor，shape ``(num_envs,)``，成功即 True（作為 terminated）。
    """
    num_envs = env.num_envs
    device = env.device

    # --- 每支試管是否正被任一夾爪夾住（跨左右 sensor 取 OR）---
    grasped_now = torch.zeros(num_envs, len(vials), dtype=torch.bool, device=device)
    for cfg in contact_sensor_cfgs:
        contact_sensor: ContactSensor = env.scene[cfg.name]
        force_norm = torch.linalg.vector_norm(contact_sensor.data.force_matrix_w, dim=-1)
        per_filter = force_norm.sum(dim=1)  # (num_envs, n_filter)，filter 順序 = vials
        grasped_now |= per_filter > force_threshold

    # --- 架子座標系 ---
    rack_obj: RigidObject = env.scene[rack_name]
    rack_pos_w = rack_obj.data.root_pos_w
    rack_quat_inv = math_utils.quat_inv(rack_obj.data.root_quat_w)
    unit_z = torch.zeros(num_envs, 3, device=device)
    unit_z[:, 2] = 1.0
    env_origins = env.scene.env_origins

    active = torch.zeros(num_envs, len(vials), dtype=torch.bool, device=device)
    on_rack = torch.zeros(num_envs, len(vials), dtype=torch.bool, device=device)
    for i, vial_name in enumerate(vials):
        vial_obj: RigidObject = env.scene[vial_name]
        vial_pos_w = vial_obj.data.root_pos_w
        vial_quat_w = vial_obj.data.root_quat_w

        # 啟用中 = 沒被搬到場外（hide_random_vials 把藏起來的停在離原點 ~2 m）
        rel = vial_pos_w - env_origins
        active[:, i] = (torch.abs(rel[:, 0]) < active_radius) & (
            torch.abs(rel[:, 1]) < active_radius
        )

        # 在架上判定（與 vial_placed_on_rack 觀測同準則）
        vial_up_world = math_utils.quat_apply(vial_quat_w, unit_z)
        is_vertical = torch.abs(vial_up_world[:, 2]) > vertical_threshold
        local = math_utils.quat_apply(rack_quat_inv, vial_pos_w - rack_pos_w)
        x_ok = (local[:, 0] >= rack_local_x_min) & (local[:, 0] <= rack_local_x_max)
        y_ok = (local[:, 1] >= rack_local_y_min) & (local[:, 1] <= rack_local_y_max)
        z_ok = local[:, 2] < rack_local_z_max
        on_rack[:, i] = is_vertical & x_ok & y_ok & z_ok & (~grasped_now[:, i])

    any_active = active.any(dim=1)
    # 成功 = 至少有一支啟用中，且沒有任何「啟用中卻不在架上」的試管
    all_placed = any_active & (~(active & ~on_rack)).all(dim=1)

    # --- 連續確認 confirm_steps 步（reset 當下歸零）---
    if not hasattr(env, "_dual_rack_success_counter"):
        env._dual_rack_success_counter = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
    env._dual_rack_success_counter[env.episode_length_buf <= 1] = 0
    env._dual_rack_success_counter = torch.where(
        all_placed,
        env._dual_rack_success_counter + 1,
        torch.zeros_like(env._dual_rack_success_counter),
    )

    # --- opt-in 診斷（DUAL_EVAL_DEBUG=1）：印 env0 的 per-vial active/on_rack ---
    if os.environ.get("DUAL_EVAL_DEBUG"):
        step = int(env.episode_length_buf[0].item())
        placed_change = bool(on_rack[0].any()) or bool(all_placed[0])
        if step % 30 == 0 or placed_change:
            rel0 = (env.scene[vials[0]].data.root_pos_w - env_origins)[0]  # noqa: F841
            act = active[0].tolist()
            onr = on_rack[0].tolist()
            grp = grasped_now[0].tolist()
            n_active = int(active[0].sum().item())
            print(
                f"[DUAL SUCC] step={step} n_active={n_active} "
                f"active={act} on_rack={onr} grasped={grp} "
                f"all_placed={bool(all_placed[0])} "
                f"counter={int(env._dual_rack_success_counter[0].item())}/{confirm_steps}"
            )
            # 各試管相對 env 原點的 xy（看誰被 hide 到 ~2m、誰在墊上）
            for vi, vn in enumerate(vials):
                p = (env.scene[vn].data.root_pos_w - env_origins)[0]
                lp = math_utils.quat_apply(
                    rack_quat_inv, env.scene[vn].data.root_pos_w - rack_pos_w
                )[0]
                print(
                    f"    {vn}: world_rel_xy=({p[0]:.2f},{p[1]:.2f}) "
                    f"rack_local=({lp[0]:.3f},{lp[1]:.3f},{lp[2]:.3f}) "
                    f"active={act[vi]} on_rack={onr[vi]}"
                )

    return env._dual_rack_success_counter >= confirm_steps

