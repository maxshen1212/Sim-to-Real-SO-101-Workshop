# 雙臂 SO-101 run cheatsheet(指令速查)

**只有指令。** 每一步的原理、陷阱、出錯怎麼辦 → **[run_notes.md](run_notes.md)**;
校準理論與維修後比對 → **[CALIBRATION.md](CALIBRATION.md)**;專案進度與設計決策 → **[ROADMAP.md](ROADMAP.md)**。

| § | 範圍 |
| --- | --- |
| **0** | 每次開工(環境 + 路徑變數 + 兩個檢查) |
| **1** | 設定(一次性):udev、相機序號 |
| **2** | 校準:四支逐支 + commit |
| **3** | 收資料 — Sim |
| **4** | 收資料 — 真機 |
| **5** | 資料集整理(sim/real 共用) |
| **6** | Eval — Sim |
| **7** | Eval — 真機 |

---

## §0 每次開工

```bash
source ~/env_isaaclab/bin/activate          # sim 與真機共用這一個 venv
cd ~/sim2real/Sim-to-Real-SO-101-Workshop

# 以下各節的指令都用這幾個變數
WORKSHOP=~/sim2real/Sim-to-Real-SO-101-Workshop         # 本 repo:校正檔、資料集、tools、sim 進入點
CONFIG_DIR=$WORKSHOP/calibration/config                 # 四份 yaml(calibrate/teleoperate/record)
CALIB_FOLLOWER=$WORKSHOP/calibration/bimanual_follower  # 兩支 follower 的校正檔
CALIB_LEADER=$WORKSHOP/calibration/bimanual_leader      # 兩支 leader 的校正檔
DATASET_SIM=$WORKSHOP/datasets/bimanual-so101-pickvials-sim
DATASET_REAL=$WORKSHOP/datasets/bimanual-so101-pickvials-real
```

開工前兩個檢查:

```bash
graphen-setup-udev                              # symlink 都在、序號正確
git -C $WORKSHOP status --short calibration/     # 應該乾淨;有 M → git -C $WORKSHOP checkout calibration/
```

---

## §1 設定(一次性)

```bash
# 1-1  USB 綁序號(插拔後 /dev/tty{Leader,Follower}{Left,Right} 不變)
graphen-setup-udev --apply                  # 第一次 / 換電腦(需 sudo)
graphen-setup-udev --identify               # 只有換過 USB 轉板才要
graphen-setup-udev                          # 純檢查

# 1-2  取得三台 RealSense 序號 → 貼進 $CONFIG_DIR/bimanual_so101_record_config.yaml
lerobot-find-cameras realsense
```

---

## §2 校準

```bash
# 2-1  四支逐支跑(不要用雙臂 config 一次跑四支)
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyFollowerLeft \
  --robot.id=bimanual_so101_follower_left   --robot.calibration_dir=$CALIB_FOLLOWER
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyFollowerRight \
  --robot.id=bimanual_so101_follower_right  --robot.calibration_dir=$CALIB_FOLLOWER
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyLeaderLeft \
  --teleop.id=bimanual_so101_leader_left    --teleop.calibration_dir=$CALIB_LEADER
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyLeaderRight \
  --teleop.id=bimanual_so101_leader_right   --teleop.calibration_dir=$CALIB_LEADER

# 2-2  立刻 commit,message 一定要寫 dataset 名稱
git -C $WORKSHOP add calibration/ && git -C $WORKSHOP commit -m "calib: baseline for <dataset name>"
```

掃描時:每個關節推到**真正的機械硬限位**;但 **`wrist_roll` 不要多轉**,盯 live 表格,
MIN 接近 `0` 或 MAX 接近 `4095` 就停手重來。

---

## §3 收資料 — Sim

```bash
# 3-1  只驗場景載入(零動作,不需硬體)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack

# 3-2  兩支 leader 驅動 sim、不錄(確認左右對、抓放偵測正常)
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR

# 3-3  錄製(三個 repo 參數都給才會啟用)
lerobot_agent_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-DR \
  --repo_id  ChihHanShen/bimanual-so101-pickvials-sim \
  --repo_root $DATASET_SIM \
  --task_name "Pick up the vial and place it in the rack"
```

開錄前確認這行:`[INFO]: recording at 30 fps (= env control rate)`,**不是 30 就停下來**。

按鍵(焦點在 Isaac Sim 視窗):**S** 起/停錄製(停 = 存檔一集)·
**C** 取消當前這段(不存)· **R** reset 場景(會先存檔)。看到 `Episode N saved.` 才算寫完。

---

## §4 收資料 — 真機

```bash
# 4-1  純遙操作,不錄
lerobot-teleoperate --config_path=$CONFIG_DIR/bimanual_so101_teleoperate_config.yaml

# 4-2  這批資料集的「第一次」錄製 —— 必須加 --resume=false(建立資料集)
lerobot-record --config_path=$CONFIG_DIR/bimanual_so101_record_config.yaml --resume=false

# 4-3  之後每次接續錄(append 進既有資料集)
lerobot-record --config_path=$CONFIG_DIR/bimanual_so101_record_config.yaml

# 4-4  在真機重播某一集(驗證校正 / 接線)
lerobot-replay \
  --robot.type=bi_so101_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=$CALIB_FOLLOWER \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=$DATASET_REAL \
  --dataset.episode=0 --dataset.fps=30
```

按鍵(pynput 全域監聽,**跟 sim 不一樣**):**→** 結束這集 / 略過等待 ·
**←** 丟棄並重錄上一集 · **Esc** 停止並存檔。

---

## §5 資料集整理(sim/real 共用)

```bash
# 5-1  單批自檢 + sim↔real 規格與值域比對(co-train 前的閘門)
python tools/check_dataset_parity.py --sim $DATASET_SIM --real $DATASET_REAL
python tools/check_dataset_parity.py --real $DATASET_REAL        # 只收好一批也能跑
#   --quick 不讀 parquet | --strict WARN 也算失敗 | --iou-warn 調重疊門檻

# 5-2  視覺化(影片 + 12 維曲線同步播放,Rerun 視窗)
lerobot-dataset-viz --repo-id ChihHanShen/bimanual-so101-pickvials-sim \
  --root $DATASET_SIM --episode-index 47

# 5-3  刪掉品質不好的 episode(先改 tools/*.py 最上面兩行 REPO / ROOT)
python tools/list_episodes.py                # 列出「檔號 → episode_index」對照
python tools/delete_episodes.py 47 30        # 刪 file-047、file-030(用檔號,不是 episode_index)

# 5-4  上傳 / 下載
hf auth login                                # 第一次;寫入需 write token
hf download ChihHanShen/bimanual-so101-pickvials-real \
  --repo-type dataset --local-dir $DATASET_REAL
lerobot_push_dataset --repo-id ChihHanShen/bimanual-so101-pickvials-sim --root $DATASET_SIM
```

⚠️ **不要用 `lerobot-edit-dataset`** —— 在 0.4.3 對本專案的扁平 layout 是壞的,用上面的 `tools/`。
刪過集之後要讓 Hub 精確鏡像本機(清孤兒 mp4)的寫法在 [run_notes.md](run_notes.md) §C3。

檢查器本身的測試(合成資料集 + 注入十種故障,不需硬體):

```bash
python tools/test_check_dataset_parity.py    # 15 個情境,約 1 分鐘
```

---

## §6 Eval — Sim

```bash
# 6-1  終端機 A:起 GR00T server(sim 與真機共用同一個 server)
cd ~/sim2real/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-sim/pickvials-n1p7-run2/checkpoint-50000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
    --device cuda:0
# 看到 `Server is ready and listening on tcp://...:5555` 才算起好
# checkpoint 還沒下載的話:
#   uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
#     --include "pickvials-n1p7-run2/checkpoint-50000/*" --exclude "*global_step*" \
#     --local-dir ~/models/bimanual-pickvials-sim

# 6-2  終端機 B:跑 rollout
cd $WORKSHOP
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 --policy_host localhost --policy_port 5555
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR-Eval --num_episodes 10 --rerun

# 6-3  錄 demo 影片(每集一支 mp4,headless 也能錄)
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 --record_video --headless   # 自訂:--video_dir ./demo_out --video_fps 30
```

跑完印 `Success Rate: N/M (%)`。

⚠️ `--steps_per_action`(預設 3)必須等於**訓練資料集相對錄製資料集的降採樣倍率** ——
直接用 30 fps 資料訓練就要改成 **1**。

本機兩個 checkpoint:

| | 路徑 |
| --- | --- |
| 純 sim | `~/models/bimanual-pickvials-sim/pickvials-n1p7-run2/checkpoint-50000` |
| real + sim co-train | `~/models/bimanual-pickvials-cotrain/pickvials-n1p7-run3/checkpoint-25000` |

---

## §7 Eval — 真機

```bash
# 7-1  終端機 A:起 server —— 與 §6-1 完全相同

# 7-2  終端機 B:一次執行 = 一集
source ~/env_isaaclab/bin/activate
cd ~/sim2real/Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual
python eval_so101_dual.py
```

成功或卡住就 Ctrl-C、人工 reset、再跑一次,成功率自己記。硬體參數都是 script 預設值。

⚠️ `fps`(預設 10)必須等於**訓練資料集的 fps** —— 新資料若不降採樣就要改成 **30**。
三個會弄壞真機的點見 [ROADMAP.md](ROADMAP.md) Phase 8。
