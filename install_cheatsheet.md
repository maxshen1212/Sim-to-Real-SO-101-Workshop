# Sim-to-Real SO-101 Workshop — LeRobot Install Cheat Sheet

Adds the in-project LeRobot fork (`/home/graphen/sim2real/lerobot`, v0.6.1) to the existing
Isaac Sim / Isaac Lab venv (`~/env_isaaclab`, Python 3.12) **without disturbing Isaac's package stack**.

Isaac and LeRobot want different versions of shared packages (numpy, huggingface-hub, packaging,
transformers). So install everything `--no-deps`, then add only the genuinely-missing runtime
deps under a constraints file that freezes Isaac's pinned versions.

## Step 1 — Install the LeRobot fork

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    --no-deps -e /home/graphen/sim2real/lerobot
```

## Step 2 — Add missing runtime deps (frozen against Isaac)

```bash
cat > /tmp/isaac_constraints.txt <<'EOF'
numpy==2.4.4
torch==2.10.0
torchvision==0.25.0
huggingface-hub==0.36.2
packaging==26.0
setuptools==81.0.0
einops==0.9.0.dev0
lxml==7.0.0a3
imageio==2.37.2
transformers==4.57.6
EOF

uv pip install --python ~/env_isaaclab/bin/python -c /tmp/isaac_constraints.txt \
    "draccus==0.10.0" \
    "datasets>=4.7.0,<5.0.0" \
    "pandas>=2.0.0,<3.0.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "pynput>=1.7.8,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "pyzmq>=26.2.1,<28.0.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "accelerate>=1.14.0,<2.0.0" \
    "feetech-servo-sdk>=1.0.0,<2.0.0"
```

> Bounds are from `lerobot/pyproject.toml`. If an `import` later hits `ModuleNotFoundError` /
> `AttributeError` for some dep, check its bound there and install that range with
> `-c /tmp/isaac_constraints.txt`.

**transformers stays pinned to 4.57.6** (Isaac Lab hard pin) even though LeRobot's VLA policies want
`>=5.4` — `utils/lerobot_interface.py` lazily imports `lerobot.policies.*` so this doesn't break the
bimanual record/GR00T-eval workflow. Don't "fix" this by upgrading transformers.

**torchcodec is deliberately skipped** (ABI-locked, would mismatch torch 2.10) — LeRobot falls back to
PyAV (`av`, installed above) for video read/write.

## Step 3 — Install the workshop package

```bash
uv pip install --python ~/env_isaaclab/bin/python \
    --no-deps -e /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
```

## Verify

```bash
source ~/env_isaaclab/bin/activate

python -c "import sim_to_real_so101.utils.lerobot_interface; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('lerobot bridge imports OK')"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__)"   # expect 2.4.4 | 2.10.0+cu128

list_envs
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack
```

Next (teleop / record / eval commands) → [run_cheatsheet.md](run_cheatsheet.md).

## If you rebuild the Isaac venv later

Re-detect the pins before trusting Step 2's constraints file:

```bash
~/env_isaaclab/bin/python - <<'PY'
import importlib.metadata as m
for p in ["numpy","torch","torchvision","packaging","huggingface-hub","lxml","imageio","setuptools","einops","isaacsim","isaaclab"]:
    try: print(f"{p:16} {m.version(p)}")
    except Exception: print(f"{p:16} -- missing")
PY
```

Then update `/tmp/isaac_constraints.txt` and re-run Steps 1–3.
