# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Workshop codebase for sim-to-real transfer of the SO-101 robot arm using NVIDIA Isaac Sim / Isaac Lab. The pipeline is: teleoperate a simulated (or real) SO-101 to record LeRobot datasets → train a GR00T policy → evaluate the policy in sim against a remote GR00T inference server → deploy on the real robot. The flagship task is "vials to rack" (pick a vial, place it in a yellow rack).

There is an active fork goal: converting this single-arm setup into a **bimanual handoff** task for GR00T sim-to-real. See [ROADMAP.md](ROADMAP.md) (written in Traditional Chinese) for the staged plan and locked-in design decisions before making structural changes to the scene/USD/env configs.

## Running

This package is **not runnable standalone** — it must be installed into an Isaac Sim + Isaac Lab Python environment (Python 3.11). Two supported setups:

- **Docker** (per [README.md](README.md) / [docker/README.md](docker/README.md)): build `teleop-docker` (`docker/sim/Dockerfile`) for sim/teleop, and `real-robot` (`docker/real/build.sh <ada|blackwell>`) for the real robot + GR00T inference server.
- **Native** (per [cheatsheet.md](cheatsheet.md)): a `uv` venv at `~/env_isaaclab` with `isaacsim`, Isaac Lab installed `-e` from `~/IsaacLab`, LeRobot pinned to commit `e670ac5d`, then `uv pip install -e source/sim_to_real_so101/`. Activate with `source ~/env_isaaclab/bin/activate`.

The package exposes console-script entry points (defined in `pyproject.toml`); after install they are on PATH:

```bash
list_envs                                              # list registered gym envs
zero_agent   --task Lerobot-So101-Teleop-Base          # debug, zero actions, no hardware
random_agent --task Lerobot-So101-Teleop-Base          # debug, random actions
lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack \
    --repo_id <hf_id/dataset> --repo_root ./datasets --task_name vials_to_rack   # teleop + record
lerobot_eval  --task Lerobot-So101-Teleop-Vials-To-Rack-Eval                     # eval vs GR00T server
lerobot_push_dataset --repo_id <hf_id/dataset>
```

Lint: `flake8` (config in `.flake8`, max line length 120). There is no test suite — validation is done by running the agents in Isaac Sim.

## Registered environments

Gym IDs are registered in [source/sim_to_real_so101/tasks/__init__.py](source/sim_to_real_so101/tasks/__init__.py). Debug envs: `-Base`, `-Wristcam`, `-Task`. Task envs: `-Vials-To-Rack` and `-Vials-To-Rack-DR` (domain randomized). Eval variants append `-Eval` / `-DR-Eval`.

## Architecture

**Env config inheritance (the central pattern).** Environments are Isaac Lab `ManagerBasedRLEnv`s built by composing `@configclass` configs across three files, each layer adding to the previous:

1. [tasks/so101_env_cfg.py](source/sim_to_real_so101/tasks/so101_env_cfg.py) — base scene: the robot articulation, `JointPositionAction`, ee-frame transformer, joint-state observations, base events.
2. [tasks/task_env_cfg.py](source/sim_to_real_so101/tasks/task_env_cfg.py) — adds `TiledCamera`(s), lightbox, mat, image observations, and the domain-randomization event/observation groups (lighting, sky, mat rotation, camera focal length / pose).
3. [tasks/vials_to_rack_env_cfg.py](source/sim_to_real_so101/tasks/vials_to_rack_env_cfg.py) — adds the manipulation objects (vials, rack), contact sensor, the grasp/placement reset and termination terms, and the `-DR` / `-Eval` permutations.

When changing a task, find which layer owns the thing you're editing rather than duplicating config.

**MDP terms** ([mdp/](source/sim_to_real_so101/mdp/)) are flat-imported via `from .resets import *` etc. and re-export Isaac Lab's `mdp` too:
- `terms.py` — task logic: `any_vial_grasped` (contact-force + height based, sticky once grasped), `vial_placed_on_rack`, termination terms.
- `obs.py` — `ee_frame_state` (ee pose in robot frame), `image` / `image_raw` camera observations.
- `resets.py` — reset + domain-randomization events: `randomize_robot_color` (ROBOT_COLORS palette, edits USD material via `Sdf.ChangeBlock`), `reset_vials_rack`, light/sky/mat/camera randomizers.

**Robot asset.** [assets/so101.py](source/sim_to_real_so101/assets/so101.py) defines `SO101_CFG` (the `ArticulationCfg`, USD path, init joint pose, per-joint `ImplicitActuator`s) and `S0101_CONTACT_GRASP_CFG` (variant with contact sensing). USD/USDA assets, HDRIs for lighting DR, and the camera-module USDs live under `assets/usd/` and `assets/hdri/`.

**LeRobot bridge** ([utils/](source/sim_to_real_so101/utils/)):
- `lerobot_interface.py` — `LeRobotSO101Interface` connects sim to physical SO-101 leader/follower hardware. Critically, it holds `SO101_USD_MAPPING`: the USD articulation joint ranges differ from LeRobot's normalized ranges, so joint values are remapped both directions. Joint *order* matters and follows the USD articulation order.
- `lerobot_recorder.py` — `LeRobotRecorder` writes a `LeRobotDataset` during teleop (RGB always; depth / instance-segmentation captured but **not** stored in the dataset).

**GR00T inference client** ([gr00t_client/](source/sim_to_real_so101/gr00t_client/)) — `BasePolicy` interface plus a ZMQ `server_client` that talks to a remote GR00T server (default `localhost:5555`). Used by `lerobot_eval` to run policy rollouts in sim; the real-robot side runs `run_gr00t_server.py` inside the `real-robot` container.

## Critical convention: Isaac Sim launch ordering

Every runnable script (`scripts/*.py`) **must launch the simulator before importing any `isaaclab.*` / task modules.** The pattern is: parse args → `AppLauncher(args_cli)` → `simulation_app = app_launcher.app` ("Launch Isaac Sim Simulator first") → only then import env configs and run ("Rest everything follows"). Importing Isaac Lab modules before the app launches will fail. Preserve this ordering when adding scripts.

## Real-robot operations

Hardware control, calibration, teleoperation, and the GR00T eval rollout commands are documented in [docker/README.md](docker/README.md) and [script.md](script.md) (port/camera env vars, `lerobot-calibrate`, `lerobot-teleoperate`, calibration-quality checks). These require physical hardware and run inside the Docker containers.
