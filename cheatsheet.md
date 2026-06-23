# Sim-to-Real SO-101 Workshop — Local Environment Cheat Sheet

> This guide covers running the workshop **natively (without Docker)** on Ubuntu with an NVIDIA Blackwell GPU.
> Docker users can skip Steps 1–5 and follow the README directly.

---

## System Requirements

| Item            | Spec                                  |
| --------------- | ------------------------------------- |
| GPU             | NVIDIA Blackwell (RTX 5060 Pro)       |
| CUDA            | 13.0                                  |
| OS              | Ubuntu 22.04+                         |
| Python          | 3.11 (via uv)                         |
| Package manager | [uv](https://github.com/astral-sh/uv) |

---

## Step 1 — Fix IOMMU (native only, not needed for Docker)

Docker runs with `--privileged` which bypasses IOMMU automatically.
On bare metal, IOMMU causes Isaac Sim's CUDA initialisation to fail with `error 999`.

```bash
sudo nano /etc/default/grub
```

Find `GRUB_CMDLINE_LINUX_DEFAULT` and append `iommu=pt`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"
```

Apply and reboot:

```bash
sudo update-grub
sudo reboot
```

---

## Step 2 — Install Isaac Sim + Isaac Lab

```bash
# Create uv virtual environment
uv venv ~/env_isaaclab --python 3.11

# Install Isaac Sim from NVIDIA pip index
uv pip install --python ~/env_isaaclab/bin/python \
    isaacsim==5.1.0.0 \
    --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab from source
cd ~/IsaacLab
uv pip install --python ~/env_isaaclab/bin/python \
    -e source/isaaclab \
    -e source/isaaclab_tasks \
    -e source/isaaclab_rl \
    -e source/isaaclab_assets
```

---

## Step 3 — Install LeRobot (pinned commit)

The workshop requires a specific LeRobot commit (`e670ac5d`).

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot && git checkout e670ac5daf9b76

uv pip install --python ~/env_isaaclab/bin/python --no-deps -e .
```

---

## Step 4 — Install Workshop Dependencies

Mirrors the package list in `docker/sim/Dockerfile`:

```bash
uv pip install --python ~/env_isaaclab/bin/python \
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

# rerun-sdk must be installed without deps (numpy version conflict)
uv pip install --python ~/env_isaaclab/bin/python \
    --no-deps "rerun-sdk>=0.24.0,<0.27.0"

uv pip install --python ~/env_isaaclab/bin/python pyzmq
```

---

## Step 5 — Install the Workshop Package

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    -e ~/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
```

---

## Verify Installation

```bash
source ~/env_isaaclab/bin/activate

list_envs
```

Expected output — all 6 environments should appear:

```
Lerobot-So101-Teleop-Base
Lerobot-So101-Teleop-Task
Lerobot-So101-Teleop-Vials-To-Rack
Lerobot-So101-Teleop-Vials-To-Rack-DR
Lerobot-So101-Teleop-Vials-To-Rack-Eval
Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval
```

---

## Common Commands

```bash
source ~/env_isaaclab/bin/activate

# List available environments
list_envs

# Debug: run with zero actions (no robot needed)
zero_agent --task Lerobot-So101-Teleop-Base

# Debug: run with random actions
random_agent --task Lerobot-So101-Teleop-Base

# Teleoperation + dataset recording
lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack \
    --repo_id <your_hf_id/dataset_name> \
    --repo_root ~/Sim-to-Real-SO-101-Workshop/datasets \
    --task_name vials_to_rack

# Evaluate a trained model in simulation
lerobot_eval --task Lerobot-So101-Teleop-Vials-To-Rack-Eval

# Push dataset to HuggingFace Hub
lerobot_push_dataset --repo_id <your_hf_id/dataset_name>
```

---

## Key Paths

| Item                | Path                                                     |
| ------------------- | -------------------------------------------------------- |
| Virtual environment | `~/env_isaaclab`                                         |
| Isaac Sim           | installed inside `~/env_isaaclab` via pip                |
| Isaac Lab           | `~/IsaacLab`                                             |
| LeRobot             | `~/lerobot`                                              |
| Workshop source     | `~/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101` |
| Datasets output     | `~/Sim-to-Real-SO-101-Workshop/datasets`                 |
| Captured images     | `~/Sim-to-Real-SO-101-Workshop/outputs/captured_images`  |

```

```
