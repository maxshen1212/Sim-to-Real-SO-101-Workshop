# 雙臂 SO-101 run cheatsheet(指令速查)

**這份只有指令。** 每一步的原理、陷阱、出錯怎麼辦 → **[run_notes.md](run_notes.md)**;
校準理論與維修後比對 → **[CALIBRATION.md](CALIBRATION.md)**。

所有指令都是寫死的絕對路徑,單獨複製貼上就能跑,不依賴任何變數或當前目錄。
唯一的前提是 **§0 的環境啟動**。

> 📌 路徑全部寫死成 `/home/graphen/...`。換人或換電腦時,把全檔的 `/home/graphen`
> 取代成該台機器的家目錄即可:
> `sed -i 's|/home/graphen|/home/<你的帳號>|g' run_cheatsheet.md`

| § | 做什麼 | 多久做一次 | 詳解 |
| --- | --- | --- | --- |
| [**0**](#0-every-session) | 環境啟動 + 兩個開工檢查 | 每次開工 | [run_notes 0.5](run_notes.md) |
| [**1**](#1-one-time-setup) | udev 綁 USB、抓相機序號 | 一次性 / 換電腦 | [run_notes 0.1–0.2](run_notes.md) |
| [**2**](#2-calibration) | 四支手臂逐支校準 + commit | 只在必要時 | [CALIBRATION.md](CALIBRATION.md) |
| [**3**](#3-collect-data--sim) | Sim 遙操作收資料 | 收資料期間 | [run_notes A](run_notes.md) |
| [**4**](#4-collect-data--real) | 真機遙操作收資料 | 收資料期間 | [run_notes B](run_notes.md) |
| [**5**](#5-clean-data) | 檢查、刪集、上傳下載 | 收完資料 | [run_notes C](run_notes.md) |
| [**6**](#6-evaluation--sim) | Sim rollout 成功率 | 訓練完 | [run_notes D1–D2](run_notes.md) |
| [**7**](#7-evaluation--real) | 真機 rollout 拍板 | 訓練完 | [run_notes D3](run_notes.md) |

---

## §0 Every Session

```bash
# 0-1  啟動環境(sim 與真機共用這一個 venv)
source /home/graphen/env_isaaclab/bin/activate
cd /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop

# 0-2  USB symlink 都在、序號正確
graphen-setup-udev

# 0-3  校正檔沒有漂移(應該是乾淨的)
git -C /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop status --short calibration/
```

**0-3 出現 `M`** = 有人重跑過校準,資料集與 checkpoint 的輸入分布已經對不上。
先確認是不是故意的,不是的話還原:

```bash
git -C /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop checkout calibration/
```

路徑對照(參考用,不用執行):

| 東西 | 路徑 |
| --- | --- |
| 本 repo | `/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop` |
| 四份 yaml(calibrate / teleoperate / record) | `.../calibration/config` |
| follower 校正檔 ×2 | `.../calibration/bimanual_follower` |
| leader 校正檔 ×2 | `.../calibration/bimanual_leader` |
| sim 資料集 | `.../datasets/bimanual-so101-pickvials-sim` |
| real 資料集 | `.../datasets/bimanual-so101-pickvials-real` |

---

## §1 One-Time Setup

```bash
# 1-1  USB 綁序號(綁定後插拔、重開機都不會換名字)
graphen-setup-udev --apply       # 第一次 / 換電腦(需 sudo)
graphen-setup-udev --identify    # 只有換控制板、或把板子換到另一支手臂才要(會請你逐支拔插)
graphen-setup-udev               # 純檢查,不改動任何東西

# 1-2  取得三台 RealSense 序號
lerobot-find-cameras realsense
```

**1-2 之後**把序號貼進
`/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/config/bimanual_so101_record_config.yaml`。

---

## §2 Calibration

⚠️ **不要在收資料與 eval 之間重跑校準** —— 資料集存的是正規化後的關節值,
重跑會靜默改變 policy 的輸入分布。動手前先讀 [CALIBRATION.md](CALIBRATION.md)。

```bash
# 2-1  四支逐支跑(不要用雙臂 config 一次跑四支)
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/ttyFollowerLeft \
  --robot.id=bimanual_so101_follower_left \
  --robot.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_follower

lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/ttyFollowerRight \
  --robot.id=bimanual_so101_follower_right \
  --robot.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_follower

lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyLeaderLeft \
  --teleop.id=bimanual_so101_leader_left \
  --teleop.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_leader

lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyLeaderRight \
  --teleop.id=bimanual_so101_leader_right \
  --teleop.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_leader

# 2-2  立刻 commit,message 一定要寫這批校準對應的 dataset 名稱
git -C /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop add calibration/
git -C /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop commit -m "calib: baseline for <dataset_name> dataset"
```

**掃描時**:每個關節推到**真正的機械硬限位**;
但 **`wrist_roll` 不要多轉** —— 盯著 live 表格,MIN 接近 `0` 或 MAX 接近 `4095` 就停手重來。
(±100 正規化下,沒掃到真硬限位會同時改掉零點**和**增益,不只是偏移。)

---

## §3 Collect Data — Sim

```bash
# 3-1  只驗場景載入(零動作,不需硬體)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack

# 3-2  兩支 leader 驅動 sim、不錄(確認左右對、抓放偵測正常)
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR

# 3-3  錄製(三個 repo 參數都給才會啟用)
lerobot_agent_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-DR \
  --repo_id ChihHanShen/bimanual-so101-pickvials-sim \
  --repo_root /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-sim \
  --task_name "Pick up the vial and place it in the rack"
```

**開錄前確認這行**:`[INFO]: recording at 30 fps (= env control rate)`,**不是 30 就停下來**。

**按鍵**(焦點要在 Isaac Sim 視窗):

| 鍵 | 作用 |
| --- | --- |
| **S** | 起 / 停錄製(停 = 存檔一集) |
| **C** | 取消當前這段(不存) |
| **R** | reset 場景(會先存檔) |

看到 `Episode N saved.` 才算真的寫完。

---

## §4 Collect Data — Real

```bash
# 4-1  純遙操作,不錄
lerobot-teleoperate \
  --config_path=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/config/bimanual_so101_teleoperate_config.yaml

# 4-2  這批資料集的「第一次」錄製 —— 必須加 --resume=false(建立資料集)
lerobot-record \
  --config_path=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/config/bimanual_so101_record_config.yaml \
  --resume=false

# 4-3  之後每次接續錄(append 進既有資料集)
lerobot-record \
  --config_path=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/config/bimanual_so101_record_config.yaml
```

```bash
# 4-4  在真機重播某一集(驗證校正 / 接線)—— real 資料集,30 fps
lerobot-replay \
  --robot.type=bi_so101_follower \
  --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_follower \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-real \
  --dataset.episode=0 \
  --dataset.fps=30

# 4-5  同上,但重播 sim 資料集 —— 注意 fps 要跟著資料集改成 10
lerobot-replay \
  --robot.type=bi_so101_follower \
  --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/calibration/bimanual_follower \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-sim-10fps \
  --dataset.root=/home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-sim-10fps \
  --dataset.episode=0 \
  --dataset.fps=10
```

**按鍵**(pynput 全域監聽,**跟 sim 完全不一樣**):

| 鍵 | 作用 |
| --- | --- |
| **→** | 結束這集 / 略過等待 |
| **←** | 丟棄並重錄上一集 |
| **Esc** | 停止並存檔 |

---

## §5 Clean Data

sim 與 real 共用。`tools/*.py` 是相對路徑,要在 repo 根目錄執行(§0 已經 `cd` 過了)。

```bash
# 5-1  視覺化(影片 + 12 維曲線同步播放,Rerun 視窗)
lerobot-dataset-viz \
  --repo-id ChihHanShen/bimanual-so101-pickvials-sim \
  --root /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-sim \
  --episode-index 47

# 5-2  列出每一集(檔號 → episode_index 對照)
python tools/list_episodes.py                          # sim 資料集(預設)
python tools/list_episodes.py \
  --repo ChihHanShen/bimanual-so101-pickvials-real \
  --root datasets/bimanual-so101-pickvials-real        # 改看 real

# 5-3  刪掉品質不好的 episode(用檔號,不是 episode_index)
python tools/delete_episodes.py 47 30                  # 刪 file-047、file-030
```

```bash
# 5-4  上傳 / 下載
hf auth login    # 第一次;寫入需 write token

lerobot_push_dataset \
  --repo-id ChihHanShen/bimanual-so101-pickvials-real \
  --root /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-real
lerobot_push_dataset \
  --repo-id ChihHanShen/bimanual-so101-pickvials-sim \
  --root /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-sim

hf download ChihHanShen/bimanual-so101-pickvials-real \
  --repo-type dataset \
  --local-dir /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-real
hf download ChihHanShen/bimanual-so101-pickvials-sim \
  --repo-type dataset \
  --local-dir /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials-sim
```

⚠️ **不要用 `lerobot-edit-dataset`** —— 0.4.3 版對本專案的扁平 layout 是壞的,一律用上面的 `tools/`。
刪過集之後要讓 Hub 精確鏡像本機(清孤兒 mp4)的寫法在 [run_notes.md](run_notes.md) §C3。

---

## §6 Evaluation — Sim

需要兩個終端機,**兩個 venv 不能混用**。

```bash
# 6-1  終端機 A:起 GR00T server(sim 與真機共用同一個 server)
cd /home/graphen/sim2real/Isaac-GR00T
source /home/graphen/sim2real/Isaac-GR00T/.venv/bin/activate

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-cotrain/pickvials-n1p7-r3-cotrain/checkpoint-10000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
    --device cuda:0
# 看到 `Server is ready and listening on tcp://...:5555` 才算起好
```

```bash
# 6-2  終端機 B:跑 rollout
cd /home/graphen/sim2real/Sim-to-Real-SO-101-Workshop
source /home/graphen/env_isaaclab/bin/activate

lerobot_eval_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 \
  --policy_host localhost \
  --policy_port 5555 \
  --steps_per_action 3 \
  --action_horizon 8

# 換 -DR-Eval 是同一個 policy 打隨機化場景
lerobot_eval_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-DR-Eval \
  --num_episodes 10 --rerun

# 6-3  錄 demo 影片(每集一支 mp4,headless 也能錄)
lerobot_eval_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 \
  --record_video \
  --headless
# 自訂輸出:--video_dir ./demo_out --video_fps 30
```

跑完會印 `Success Rate: N/M (%)`。

⚠️ **換機器人要換 modality** —— `--modality-config-path` 指的那支 python 檔要跟著改。
⚠️ **`--steps_per_action`** 必須等於「訓練資料集相對錄製資料集的降採樣倍率」:
10 fps 訓練 = **3**,直接用 30 fps 資料訓練 = **1**。設錯手臂會等比例變快或變慢。
⚠️ **`--action_horizon`** 是執行長度,不是模型預測長度(chunk 長度由 checkpoint 決定,N1.7 = 16)。
這裡設 8 就要跟 §7 真機的 `--execution_horizon 8` 一致,否則 sim 排名不代表真機表現。

---

## §7 Evaluation — Real

```bash
# 7-1  終端機 A:起 server —— 指令與 §6-1 完全相同,不用改
cd /home/graphen/sim2real/Isaac-GR00T
source /home/graphen/sim2real/Isaac-GR00T/.venv/bin/activate

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-cotrain/pickvials-n1p7-r3-cotrain/checkpoint-10000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
    --device cuda:0
# 看到 `Server is ready and listening on tcp://...:5555` 才算起好
```

```bash
# 7-2  終端機 B:一次執行 = 一集
source /home/graphen/env_isaaclab/bin/activate
cd /home/graphen/sim2real/Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual

# 第一次帶電:限位設到最小,手放電源開關,只確認方向對、Ctrl-C 收得乾淨
python eval_so101_dual.py \
  --execution_horizon 8 \
  --max_relative_target '{"shoulder_pan":0.5,"shoulder_lift":0.5,"elbow_flex":0.5,"wrist_flex":0.5,"wrist_roll":0.5,"gripper":1.0}'

# 確認沒問題後全速跑
python eval_so101_dual.py --execution_horizon 8
```

成功或卡住就 Ctrl-C、人工 reset、再跑一次,成功率自己記。硬體參數其餘都用 script 預設值。

⚠️ **`--fps`**(預設 10)必須等於**訓練資料集的 fps** —— 新資料若不降採樣就要改成 **30**。
⚠️ **Ctrl-C 會讓手臂軟掉**(torque 斷),夾著試管時避免中途按。
⚠️ **`--execution_horizon`** 要跟 §6-2 的 `--action_horizon` 一致。
