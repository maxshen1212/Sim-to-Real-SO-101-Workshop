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

## Step 3 — LeRobot, branch `n1.7-graphen` (v0.4.3, base `e670ac5d`)

**One checkout, one venv — sim and real robot share both.** v0.4.3 requires only Python 3.10+, so
it runs in Isaac Sim's 3.11 env; and its `use_degrees=False` default (±100) matches what NVIDIA's
GR00T reference pipeline pins. The old `~/lerobot-pinned` worktree is gone — do not recreate it.

```bash
cd ~/sim2real/lerobot
git checkout n1.7-graphen

uv pip install --python ~/env_isaaclab/bin/python --no-deps -e ~/sim2real/lerobot
```

No fork checked out? `git clone https://github.com/maxshen1212/lerobot.git ~/sim2real/lerobot && cd ~/sim2real/lerobot && git checkout n1.7-graphen`

- `--no-deps` keeps LeRobot from moving Isaac's `torch` / `numpy`.
- Editable install → don't move or delete `~/sim2real/lerobot`.
- **Do not create a `.venv` inside `~/sim2real/lerobot`** (i.e. no `uv run` in there) — the real-robot
  commands are meant to run in `~/env_isaaclab` too. See [ROADMAP.md](ROADMAP.md) (technical debt).

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

# RealSense backend — needed for the REAL robot (lerobot-record / lerobot-teleoperate with cameras)
uv pip install --python ~/env_isaaclab/bin/python pyrealsense2
```

- `numpy` / `torch` pinned so the resolver can't swap out Isaac's versions.
- `feetech-servo-sdk` is a LeRobot *extra*, so Step 3's `--no-deps` skips it — but `so101_leader` /
  `so101_follower` need it.
- `pyrealsense2` has no dependencies of its own, so it can't disturb Isaac's pins.

## Step 5 — Workshop package

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    -e ~/sim2real/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
```

## Step 6 — GR00T client slice (for real-robot eval)

The real-robot eval **client** (`eval_so101_dual.py`) needs `lerobot` *and* a thin slice of
`gr00t` (`PolicyClient`, which only imports numpy/msgpack/zmq — never torch). The **server** stays
in Isaac-GR00T's own 3.12 `.venv`; only the client lives here.

```bash
# uv refuses this (gr00t declares requires-python >=3.12,<3.13; this venv is 3.11),
# so use pip, which has --ignore-requires-python. uv has no equivalent flag.
~/env_isaaclab/bin/python -m pip install \
    --no-deps --ignore-requires-python --no-build-isolation \
    -e ~/sim2real/Isaac-GR00T
```

- `--no-deps` is essential: gr00t pins `torch==2.9.0` / `transformers==4.57.3` and would wreck
  Isaac's pins. The client half needs none of it.
- The 3.12 requirement covers gr00t's training/model code, not the ZMQ client. Verified working
  on 3.11 — but treat anything else in the package as unsupported here.
- Undo with `~/env_isaaclab/bin/python -m pip uninstall -y gr00t`.

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

# real-robot side lives in this same venv now — these must import too
python -c "
import pyrealsense2
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.scripts.lerobot_record import RecordConfig, DatasetRecordConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig
print('real-robot OK')"

lerobot-calibrate --help >/dev/null && echo "console scripts OK"

list_envs                                          # 12 envs; 6 Teleop- (single) + 6 Dual- (bimanual)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack # scene loads, no hardware needed
```
