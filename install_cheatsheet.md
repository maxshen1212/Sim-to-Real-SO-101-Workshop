# Sim-to-Real SO-101 Workshop — Local Environment Cheat Sheet

Native (no Docker) install on Ubuntu + NVIDIA Blackwell. Docker users follow the README instead.

| Item      | Spec                                                          |
| --------- | ------------------------------------------------------------- |
| GPU       | NVIDIA Blackwell (RTX 5060 Pro), CUDA 13.0                     |
| OS        | Ubuntu 22.04+                                                  |
| Isaac     | Isaac Sim 5.1.0.0 / Isaac Lab 0.54.4                           |
| Python    | 3.11 only — `isaacsim 5.1` is `Requires-Python: ==3.11.*`      |

**Already have a working env?** Skip to [Restore](#restore).

---

## Step 1 — Fix IOMMU (native only)

Without this Isaac Sim's CUDA init fails with `error 999`.

```bash
sudo nano /etc/default/grub     # append iommu=pt to GRUB_CMDLINE_LINUX_DEFAULT
sudo update-grub && sudo reboot
```

## Step 2 — Isaac Sim + Isaac Lab

```bash
uv venv ~/env_isaaclab --python 3.11

uv pip install --python ~/env_isaaclab/bin/python \
    isaacsim==5.1.0.0 --extra-index-url https://pypi.nvidia.com

cd ~/IsaacLab
uv pip install --python ~/env_isaaclab/bin/python \
    -e source/isaaclab -e source/isaaclab_tasks -e source/isaaclab_rl -e source/isaaclab_assets
```

## Step 3 — LeRobot, pinned to `e670ac5d` (v0.4.3)

Newer LeRobot needs Python 3.12 (PEP 695 syntax), so the sim env stays on v0.4.3. Real robot keeps
the newer one — datasets (`v3.0`) and calibration files are compatible both ways.

```bash
# worktree off the existing fork — shares the object store, ~11 MB
cd ~/sim2real/lerobot
git worktree add -b sim-pinned ~/lerobot-pinned e670ac5daf9b76

uv pip install --python ~/env_isaaclab/bin/python --no-deps -e ~/lerobot-pinned
```

No fork checked out? `git clone https://github.com/maxshen1212/lerobot.git ~/lerobot-pinned && cd ~/lerobot-pinned && git checkout -b sim-pinned e670ac5daf9b76`

- `--no-deps` keeps LeRobot from moving Isaac's `torch` / `numpy`.
- Editable install → don't move or delete `~/lerobot-pinned`. Remove with `git worktree remove`, not `rm -rf`.

## Step 4 — Dependencies

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    "numpy==1.26.0" \
    "torch==2.7.0" \
    "datasets>=4.0.0,<4.2.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "pynput>=1.7.7,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "feetech-servo-sdk>=1.0.0,<2.0.0"

# rerun-sdk wants numpy>=2 — install without deps
uv pip install --python ~/env_isaaclab/bin/python --no-deps "rerun-sdk>=0.24.0,<0.27.0"

# ZMQ client for the GR00T server
uv pip install --python ~/env_isaaclab/bin/python pyzmq
```

- `numpy` / `torch` pinned so the resolver can't swap out Isaac's versions.
- `feetech-servo-sdk` is a LeRobot *extra*, so Step 3's `--no-deps` skips it — but `so101_leader` /
  `so101_follower` need it.

## Step 5 — Workshop package

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    -e ~/sim2real/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
```

---

## Verify

Stop at the first failure.

```bash
source ~/env_isaaclab/bin/activate

# numpy MUST still be 1.26.0
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"

# LeRobot import paths match utils/lerobot_interface.py
python -c "
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.processor import make_default_processors
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
print('lerobot OK')"

python -c "from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface; print('bridge OK')"

list_envs                                          # 12 envs; 6 Teleop- (single) + 6 Dual- (bimanual)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack # scene loads, no hardware needed
```

---

## Restore

[requirements-isaac.lock](requirements-isaac.lock) pins the whole env (290 packages) and replaces
Steps 3–5:

```bash
uv venv ~/env_isaaclab --python 3.11        # only if the venv is gone
uv pip sync --python ~/env_isaaclab/bin/python \
    ~/sim2real/Sim-to-Real-SO-101-Workshop/requirements-isaac.lock
```

It carries its own `--extra-index-url` lines for `pypi.nvidia.com` and `download.pytorch.org/whl/cu128`.
Not covered: the venv must be Python 3.11, and `~/IsaacLab` / `~/lerobot-pinned` / this repo must exist.

Regenerate after intentional changes with `uv pip freeze --python ~/env_isaaclab/bin/python`, then
re-add the two `--extra-index-url` lines.

---

## Commands

```bash
source ~/env_isaaclab/bin/activate

list_envs
zero_agent   --task Lerobot-So101-Dual-Vials-To-Rack     # zero actions
random_agent --task Lerobot-So101-Dual-Base              # random actions

# bimanual teleop + record (two SO-101 leaders)
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack \
    --repo_id <hf_id/dataset> \
    --repo_root ~/sim2real/Sim-to-Real-SO-101-Workshop/datasets/<dataset> \
    --task_name "Pick up the vials and place them into the rack"

# eval vs GR00T server on localhost:5555
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval --num_episodes 10

lerobot_push_dataset --repo-id <hf_id/dataset> --root <path>   # note: --repo-id, hyphenated
```

`--headless` runs without a window. Full bimanual workflow (ports, calibration, recording keys,
dataset cleanup, GR00T eval): [run_cheatsheet.md](run_cheatsheet.md).

---

## Paths

| Item                  | Path                                                             |
| --------------------- | ---------------------------------------------------------------- |
| venv                  | `~/env_isaaclab` (Python 3.11, Isaac Sim installed inside)        |
| Isaac Lab             | `~/IsaacLab`                                                      |
| LeRobot — sim         | `~/lerobot-pinned` — worktree, branch `sim-pinned` @ `e670ac5d`   |
| LeRobot — real        | `~/sim2real/lerobot` — branch `graphen`                           |
| Workshop source       | `~/sim2real/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101` |
| Lock file             | `~/sim2real/Sim-to-Real-SO-101-Workshop/requirements-isaac.lock`  |
| Datasets              | `~/sim2real/Sim-to-Real-SO-101-Workshop/datasets`                 |
| Captured images       | `~/sim2real/Sim-to-Real-SO-101-Workshop/outputs/captured_images`  |
