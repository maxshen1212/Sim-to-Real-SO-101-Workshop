# 雙臂 SO-101 run cheatsheet(sim + 真機)

一條 pipeline:**校準 → 收資料(sim / 真機)→ 整理資料集 → 訓練 → eval**。
校準、資料集工具、GR00T server 兩邊**共用**,所以只寫一次。

> **環境:`source ~/env_isaaclab/bin/activate`,指令直接呼叫、不要加 `uv run`。**
> sim 與真機共用這一個 venv(`lerobot` 是 `n1.7-graphen` = v0.4.3 的 editable 安裝),
> `~/sim2real/lerobot` 底下**沒有自己的 `.venv`** —— 在那裡跑 `uv run` 會就地生一個出來。
>
> **校正檔統一放 `~/sim2real/lerobot/calibration/`**(git 追蹤),sim 與真機讀同一份。
> 關節正規化是 `RANGE_M100_100`(±100),`use_degrees` 一律不設。

---

# 0. 一次性設定

### 0.1 USB(udev,綁序號,插拔不變)

```bash
graphen-setup-udev              # 檢查 /dev/tty{Leader,Follower}{Left,Right}
graphen-setup-udev --apply      # 第一次 / 換電腦(需 sudo)
graphen-setup-udev --identify   # 只有換過 USB 轉板才要
```

規則帶 `MODE="0666"`,所以**不用再 `chmod`**。

### 0.2 相機序號

```bash
lerobot-find-cameras realsense  # 貼進 record config(省略參數則掃全部)
```

### 0.3 校準四支

**逐支跑單臂型別**,不要用雙臂 config 一次跑四支 —— 雙臂類別是序列校準(左臂跑完才輪到右臂),
中間右臂的 port 開著空等好幾分鐘,USB 一抖動 fd 就失效,會在切換到右臂時炸 `termios.error EIO`。
單臂逐支跑把這個暴露窗口從幾分鐘縮到幾十秒。

```bash
CF=~/sim2real/lerobot/calibration/bimanual_follower
CL=~/sim2real/lerobot/calibration/bimanual_leader

lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyFollowerLeft \
  --robot.id=bimanual_so101_follower_left   --robot.calibration_dir=$CF
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyFollowerRight \
  --robot.id=bimanual_so101_follower_right  --robot.calibration_dir=$CF

lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyLeaderLeft \
  --teleop.id=bimanual_so101_leader_left    --teleop.calibration_dir=$CL
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyLeaderRight \
  --teleop.id=bimanual_so101_leader_right   --teleop.calibration_dir=$CL
```

**每個關節都要推到真正的機械硬限位。** ±100 下 span 直接就是尺度,掃不到底整條軸的 gain 就錯了。
判讀、維修後怎麼比對、跑掉了怎麼救,全在 **[CALIBRATION.md](CALIBRATION.md)**。

> `id` 寫成子臂名字,寫出來就是雙臂類別之後會讀的同一個檔(檔名只由 `id` 決定)。
> 之後 `lerobot-record` / `lerobot-teleoperate` 照樣用雙臂 config,不受影響。
> **一定要用 `so101_*` / `bi_so101_*`,不要用 `so100_*` / `bi_so100_*`** —— SO100 那條路徑
> 會把 `wrist_roll` 寫死成 0–4095。

### 0.4 立刻 commit

```bash
cd ~/sim2real/lerobot && git add calibration/ && git commit -m "calib: baseline for <dataset name>"
```

**message 一定要寫 dataset 名稱** —— dataset 不存 calibration,對應關係只在這行訊息裡。

## 0.5 每次開工前

```bash
graphen-setup-udev                                   # symlink 都在、序號正確
git -C ~/sim2real/lerobot status --short calibration/ # 應該乾淨;有 M → git checkout calibration/
```

維修過手臂要另外比對硬限位,見 [CALIBRATION.md](CALIBRATION.md) §4。

---

# A. 收資料 —— Sim

```bash
# 只驗場景載入(零動作,不需硬體)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack

# 兩支 leader 驅動 sim、不錄(確認左右對、抓放偵測正常)
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR

# 錄製(三個 repo 參數都給才會啟用;-DR 版本每集隨機化外觀+相機+試管數)
lerobot_agent_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack \
  --repo_id  ChihHanShen/bimanual-so101-pickvials \
  --repo_root $(pwd)/datasets/bimanual-so101-pickvials \
  --task_name "Pick up the vials and place them into the rack"
```

port / id / 校正目錄都已是預設值,**不用設環境變數**。要覆寫才用 `--port_left` 等旗標
(或 `TELEOP_PORT_LEFT` / `TELEOP_ID_LEFT` / `TELEOP_CALIBRATION_DIR`)。
啟動時會印出實際載入的校正檔路徑;檔案不在會直接停,不會靜靜跳進重新校準。

錄製鍵盤(焦點要在 Isaac Sim 視窗):

| 鍵 | 動作 |
| --- | --- |
| **S** | 起 / 停錄製。再按一次 **停止並存檔**(一段 = 一集) |
| **C** | **取消**當前這段(丟棄不存);只有錄製中有效 |
| **R** | reset 場景。**R 會先 stop_recording** → 正在錄的話會先存檔再 reset |

- 看到 log `Episode N saved.` 才代表真的寫完。
- 丟掉壞的 + 換佈局 → 先 **C** 再 **R**(直接按 R 會把壞的存進去)。
- 先錄 3~5 集 → 跳到 C1 檢查 schema,再放大到 50~100 集。
- **DR 版本相機也會抖 → 每集視角不同**,寫到另一個資料集,不要混進乾淨的那批。

---

# B. 收資料 —— 真機

```bash
# 純遙操作,不錄
lerobot-teleoperate --config_path=~/sim2real/lerobot/calibration/config/bimanual_so101_teleoperate_config.yaml

# 錄製
lerobot-record --config_path=~/sim2real/lerobot/calibration/config/bimanual_so101_record_config.yaml
```

錄製鍵盤(pynput 全域監聽,需有畫面;無頭環境停用)—— **跟 sim 的按鍵不一樣**:

| 鍵 | 錄製中 | 重置等待中 |
| --- | --- | --- |
| → | 停止這一集、進下一步 | 略過等待、直接開始下一集 |
| ← | 丟棄並重錄上一集 | — |
| Esc | 完全停止並存檔 | 完全停止並存檔 |

> **第一次錄要盯 fps 穩不穩。** 0.4.x 是 writer thread 寫 PNG、再於 `save_episode()` 編碼
> (config 內已設 `num_image_writer_processes/threads_per_camera/video_encoding_batch_size` = `0 / 4 / 1`)。
> 不穩就把 `num_image_writer_processes` 調到 1 以上。

**Replay(在真機重播某一集,驗證校正/接線)**:

```bash
lerobot-replay \
  --robot.type=bi_so101_follower --robot.id=bimanual_so101_follower \
  --robot.calibration_dir=~/sim2real/lerobot/calibration/bimanual_follower \
  --robot.left_arm_port=/dev/ttyFollowerLeft \
  --robot.right_arm_port=/dev/ttyFollowerRight \
  --dataset.repo_id=ChihHanShen/bimanual-so101-pickvials-real \
  --dataset.root=~/sim2real/lerobot/datasets/bimanual-so101-pickvials-real \
  --dataset.episode=0 --dataset.fps=30
```

> `--dataset.fps` 要對上**該資料集自己的** fps(看 `meta/info.json`),不是照抄 30。

---

# C. 資料集(sim / 真機共用)

### C1 檢查 schema

```bash
DS=$(pwd)/datasets/bimanual-so101-pickvials
ls $DS && cat $DS/meta/info.json | head

python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="ChihHanShen/bimanual-so101-pickvials", root="$DS")
print("episodes:", ds.meta.total_episodes, "frames:", ds.meta.total_frames, "fps:", ds.meta.fps)
for k, v in ds.meta.features.items():
    print(k, v.get("shape"), v.get("names"))
PY
```

驗三點:`observation.state` / `action` 形狀 **(12,)** 且 names 是 6 個 `left_*` + 6 個 `right_*`;
有 **3 個** `observation.images.*`(`wrist_left`/`wrist_right`/`center`,480×640);`fps` 與集數正確。

視覺化(影片 + 12 維曲線同步播放,Rerun 視窗):

```bash
lerobot-dataset-viz --repo-id <id> --root <path> --episode-index 47
```

### C2 刪掉品質不好的 episode

```bash
# 先改 tools/*.py 最上面兩行 REPO / ROOT 指向目標資料集
python tools/list_episodes.py           # 列出「檔號 → episode_index」對照
python tools/delete_episodes.py 47 30   # 刪 file-047、file-030
```

- **用檔號,不是 episode_index。** 檔號刪掉後永遠空著、不會重用;`episode_index` 每刪一次就重排。
  編號不連續是正常的,不影響訓練或上傳。
- mp4 檔名是「檔號 file-XXX」而且一個 mp4 串了多集 → 要查表,不能直接點 mp4 猜。
- 安全網:先寫暫存目錄並驗證能載入才替換,失敗時原始資料留在 `..._bak`。

> ⚠️ **不要用 `lerobot-edit-dataset`。** 在 0.4.3 上它對本專案的扁平 layout 是壞的:
> `--new_root` **這個旗標不存在**(0.6.1 才加的),而 `get_output_path()` 把輸出算成 `root / <repo_id>`
> —— 載入時 `root` 當成資料集本身、輸出時又拿它去拼 repo_id,兩邊慣例不一致,
> 結果寫進 `datasets/<name>/ChihHanShen/<name>/` 這種巢狀位置,不會替換到你的資料集。

### C3 Hugging Face 上傳 / 下載

沒有專門的 CLI:下載用 `hf download`,上傳用 `lerobot_push_dataset`(sim)或 `push_to_hub()`。
本機資料夾名(底線)與 HF 名(連字號)不同沒關係,`info.json` 不存 repo_id。

```bash
hf auth login       # 第一次;寫入需 write token

# 下載 —— --repo-type dataset 必填(預設是 model);--local-dir 才會落地成平舖資料夾
hf download ChihHanShen/bimanual-so101-pickvials-real \
  --repo-type dataset --local-dir ~/sim2real/lerobot/datasets/bimanual-so101-pickvials-real

# 上傳(注意是連字號旗標;--root 一定要帶,否則會去 HF cache 找)
lerobot_push_dataset --repo-id ChihHanShen/bimanual-so101-pickvials \
  --root $(pwd)/datasets/bimanual-so101-pickvials       # 私有加 --private
```

`push_to_hub()` 是「覆蓋 + 新增」,**不會刪掉 Hub 上本機已無的檔**。只加集時完全正確;
**刪過集之後**單純重推會留下孤兒 mp4,要用 `delete_patterns` 讓 Hub 精確鏡像本機:

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
ds = LeRobotDataset("ChihHanShen/bimanual-so101-pickvials-real",
                    root="/home/graphen/sim2real/lerobot/datasets/bimanual-so101-pickvials-real")
HfApi().upload_folder(
    repo_id=ds.repo_id, repo_type="dataset", folder_path=str(ds.root),
    ignore_patterns=["images/"],
    delete_patterns=["data/*", "videos/*", "meta/*"],   # 刪掉本機已無的舊 shard
)
PY
```

---

# D. Eval

sim 與真機**共用同一個 GR00T server**,只有 client 不同 —— 兩邊成功率才可比。

### D1 起 server(終端機 A)

⚠️ **`--model-path` 要指到 checkpoint 那一層**(有 `config.json` 的目錄),不是 HF repo 根。
本專案的 HF repo 把 checkpoint **巢狀**放在 `pickvials-n1p7-runN/checkpoint-XXXXX/` 底下,
直接給 repo id 會報 `Unrecognized model ... should have a model_type key in its config.json`。

```bash
cd ~/sim2real/Isaac-GR00T

# 下載(約 6GB)。--local-dir 之後就是 --model-path 的前綴,兩者要對得起來
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  --include "pickvials-n1p7-run2/checkpoint-50000/*" --exclude "*global_step*" \
  --local-dir ~/models/bimanual-pickvials-sim

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-sim/pickvials-n1p7-run2/checkpoint-50000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
    --device cuda:0
```

本機現有兩個 checkpoint:

| | 路徑 |
| --- | --- |
| 純 sim(Phase 6 基準線 50%) | `~/models/bimanual-pickvials-sim/pickvials-n1p7-run2/checkpoint-50000` |
| real + sim co-training | `~/models/bimanual-pickvials-cotrain/pickvials-n1p7-run3/checkpoint-25000` |

- server 必須跑在 Isaac-GR00T 自己的 `.venv`(pin `torch==2.9.0`/`transformers==4.57.3`),
  借用別的 venv 會讓前處理跟訓練時不同。
- `--embodiment-tag` 要對上訓練設定(大小寫不拘,`EmbodimentTag.resolve()` 會處理)。
- `--modality-config-path` 讓 server 知道 12 維怎麼切、3 相機怎麼對。
- 看到 `Server is ready and listening on tcp://...:5555` 才算起好。
- 單 GPU 要注意 VRAM(server 與 Isaac Sim 搶記憶體),多卡可 `--device cuda:1`。
  port 被占 → server 加 `--port 5556`,client 也要跟著帶。

### D2 sim eval(終端機 B)

```bash
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 --policy_host localhost --policy_port 5555

lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR-Eval --num_episodes 10 --rerun
```

**成功判定(dual-safe)**:所有「啟用中」試管(沒被 `hide_random_vials` 藏掉的)同時放進中央架、
連續 25 步維持,才算成功並提前結束;超過 7.5 分鐘 time_out。跑完印 `Success Rate: N/M (%)`。

> sim eval **不讀校正檔** —— 那兩個 interface 只用硬編碼的 `SO101_USD_MAPPING` 做 joint mapping,
> `port=None`、從不 connect。會讀校正檔的只有 `lerobot_agent_dual`。

錄 demo 影片(中央俯視,每集一支 mp4,headless 也能錄):

```bash
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 --record_video --headless        # 自訂:--video_dir ./demo_out --video_fps 30
```

檔名帶結果(`ep001_success.mp4`);按 `R` 中斷的那集不存;每集結束會同步編碼(sim 短暫暫停屬正常)。

### D3 真機 eval(終端機 B)

```bash
source ~/env_isaaclab/bin/activate
cd ~/sim2real/Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual && python eval_so101_dual.py
```

**一次執行 = 一集**,成功或卡住就 Ctrl-C、人工 reset、再跑一次,成功率自己記。
硬體參數都是 script 預設值。三個會弄壞真機的點(`use_relative_action`、10 Hz 控制頻率、
不要用 `docker/real/scripts/so101_eval.py` 那支單臂版)見 [ROADMAP.md](ROADMAP.md) Phase 8。

---

# 附錄:0.6.1 → 0.4.3 的差異

| | v0.6.1 | 現在(v0.4.3) |
| --- | --- | --- |
| 型別 | `bi_so_follower` / `bi_so_leader` | **`bi_so101_follower` / `bi_so101_leader`**(本 repo 新增) |
| 每臂欄位 | `--robot.left_arm_config.port` | **`--robot.left_arm_port`**(扁平) |
| 正規化 | `use_degrees=true`(度) | **不設**(預設 `False` = ±100) |
| 跑 policy | `lerobot-rollout` | **不存在** → `lerobot-record --policy.path=...` |
| diffusion 裁切 | `resize_shape` + `crop_ratio` | `resize_shape` **不存在**;改 `--policy.crop_shape=[H,W]`<br>⚠️ 預設 `(84,84)` 是 PushT 用的,對 480×640 會裁爛且不報錯 |
| 串流編碼 | `streaming_encoding` 等 | **不存在** → `num_image_writer_*` / `video_encoding_batch_size` |

**遷移前收的真機資料集是「度」錄的,和現在的 ±100 不可混用**,要重收。
