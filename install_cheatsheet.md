# Sim-to-Real SO-101 Workshop — Local Environment Cheat Sheet

## Step 0 — System prerequisites

```bash
sudo nano /etc/default/grub
sudo update-grub && sudo reboot
```

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    ffmpeg libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 \
    libxkbcommon0 libxkbcommon-x11-0
```

## Step 1 — Isaac Sim + Isaac Lab

```bash
uv venv ~/env_isaaclab --python 3.11 --seed
source ~/env_isaaclab/bin/activate

uv pip install "isaacsim[all,extscache]==5.1.0.0" \
    --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout b0542fe2d
./isaaclab.sh --install
```

```bash
# Step 1 必須帶進這三個,Step 2 的 lerobot 是 --no-deps 裝的,不會補:
#   einops / transformers==4.57.6      ← isaaclab（source/isaaclab/setup.py）
#   opencv-python-headless==4.11.0.86  ← isaacsim-core
# 抓不到就是 Step 1 沒裝完,回去重跑,不要手動補套件。
python -c "import einops, transformers, cv2; print('step 1 deps OK')"
```

```bash
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc && source ~/.bashrc
```

## Step 2 — LeRobot, branch `n1.7-graphen`

```bash
git clone https://github.com/maxshen1212/lerobot.git ~/sim2real/lerobot
cd ~/sim2real/lerobot
git checkout n1.7-graphen

uv pip install --no-deps -e ~/sim2real/lerobot
```

```bash
printf '%s\n' \
    "packaging==23.0" \
    "numpy==1.26.0" \
    "lxml==4.9.4" \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "imageio==2.37.0" \
    > /tmp/lerobot-constraints.txt

uv pip install -c /tmp/lerobot-constraints.txt \
    "datasets>=4.0.0,<4.2.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "cmake>=3.29.0.1,<4.2.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "pynput>=1.7.7,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "wandb>=0.20.0,<0.22.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "feetech-servo-sdk>=1.0.0,<2.0.0"

uv pip install --no-deps "rerun-sdk>=0.24.0,<0.27.0"
uv pip install pyzmq
uv pip install "pyrealsense2>=2.55.1.6486,<2.57.0"
```

## Step 3 — Workshop package

```bash
uv pip install -e ~/sim2real/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
```

## Step 4 — Isaac-GR00T client slice

```bash
git clone git@github.com:maxshen1212/Isaac-GR00T.git ~/sim2real/Isaac-GR00T
cd ~/sim2real/Isaac-GR00T
git checkout n1.7-graphen
```

```bash
# 這一步是全篇唯一的裸 pip,而裸 pip 在沒啟 venv 時會安靜地掉到系統 Python。
# 症狀:出現 "Defaulting to user installation" 後,--no-build-isolation 撿到系統
# 的 setuptools 59.6(Ubuntu 22.04),它沒有 PEP 660 的 build_editable hook →
# "uses a build backend that is missing the 'build_editable' hook"。
source ~/env_isaaclab/bin/activate
which pip    # 必須是 ~/env_isaaclab/bin/pip

# 不能改成 uv pip:gr00t 的 requires-python 是 >=3.12,<3.13,本 venv 是 3.11,
# 只有 pip 有 --ignore-requires-python 可以略過這個檢查。
pip install \
    --no-deps --ignore-requires-python --no-build-isolation \
    -e ~/sim2real/Isaac-GR00T
```

## Step 5 — Verify

```bash
source ~/env_isaaclab/bin/activate

python -c "
import numpy, torch
print(numpy.__version__, torch.__version__)
assert numpy.__version__ == '1.26.0', numpy.__version__
assert torch.__version__.endswith('+cu128'), 'torch is not the cu128 build — rerun ./isaaclab.sh --install'
assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
print('cuda available:', torch.cuda.is_available())"

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

python -c "
import torchcodec
from torchcodec.decoders import VideoDecoder
print('torchcodec OK')"

python -c "from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface; print('bridge OK')"

python -c "
import pyrealsense2
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.scripts.lerobot_record import RecordConfig, DatasetRecordConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig
print('real-robot OK')"

python -c "
from gr00t.eval._horizon_contract import *
from gr00t.policy.server_client import PolicyClient
print('gr00t client OK')"

lerobot-calibrate --help >/dev/null && echo "console scripts OK"

list_envs

zero_agent --task Lerobot-So101-Dual-Vials-To-Rack --headless
```
