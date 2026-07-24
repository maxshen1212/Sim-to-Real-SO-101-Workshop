# Bimanual(雙臂:設定 port → 校正 → 錄製 → 上傳 → 檢查)

雙臂用**兩支 SO-101 leader** 驅動 sim,收 12 維(`left_*`/`right_*`)state/action + 3 相機。
所有錄製只寫**本機**,上傳是最後手動一步。

## 0. 進環境

```bash
source ~/env_isaaclab/bin/activate
```

## 1. 找 port + 授權(每次插拔都會變,要重跑)

```bash
lerobot-find-port          # 拔插左臂 → 記下它的 /dev/ttyACM?
lerobot-find-port          # 拔插右臂 → 記下它的 /dev/ttyACM?

export TELEOP_PORT_LEFT=/dev/ttyACM0   TELEOP_ID_LEFT=leader_left
export TELEOP_PORT_RIGHT=/dev/ttyACM1  TELEOcP_ID_RIGHT=leader_right

echo "LEFT  $TELEOP_PORT_LEFT  id=$TELEOP_ID_LEFT"
echo "RIGHT $TELEOP_PORT_RIGHT id=$TELEOP_ID_RIGHT"

sudo chmod 666 $TELEOP_PORT_LEFT $TELEOP_PORT_RIGHT
sudo chmod -R 777 ~/.cache/huggingface/lerobot/     # 校正檔寫得進去
```

## 2. 校正兩支 leader(id 必須不同,只需做一次;校正檔會存起來)

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$TELEOP_PORT_LEFT  --teleop.id=$TELEOP_ID_LEFT
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$TELEOP_PORT_RIGHT --teleop.id=$TELEOP_ID_RIGHT
```

## 3.(選用)先驗場景 / 驅動再錄

```bash
# 只看場景載入正確(零動作,不需硬體)
zero_agent --task Lerobot-So101-Dual-Vials-To-Rack

# 兩支 leader 驅動 sim、但不錄(確認左右對、抓放偵測正常)
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR
```

## 4. 錄製(三個 repo 參數都給才會啟用錄製;要 depth 加 --depth)

```bash
lerobot_agent_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack \
  --repo_id  ChihHanShen/bimanual-so101-pickvials \
  --repo_root $(pwd)/datasets/bimanual-so101-pickvials \
  --task_name "Pick up the vials and place them into the rack"
```

錄製時鍵盤(焦點要在 Isaac Sim 視窗):

| 鍵    | 動作                                                                            |
| ----- | ------------------------------------------------------------------------------- |
| **S** | 起 / 停錄製(切換)。按 S 開始收 frame,再按 S **停止並存檔**(一段 = 一個 episode) |
| **C** | **取消**當前這段(丟棄不存,錄壞用這個);只有錄製中有效                            |
| **R** | reset 場景(換佈局)。**R 會先 stop_recording** → 正在錄的話會**先存檔再 reset**  |

- 一個 episode = `S開 → 操作 → S停`。看到 log `Episode N saved.` 才代表真的寫完。
- 丟掉壞的 + 換佈局 → 先 **C** 再 **R**(直接按 R 會把壞的存進去)。
- 先錄 3~5 集 → 跳到第 6 步檢查 schema 正確,再放大到 50~100 集。

## 4b. DR 錄製(全套 domain randomization)

跟第 4 步一樣,只是 task 換成 `-DR`,每集會隨機化 架子顏色/機器人顏色/光/天空/墊子/**相機 pose+焦距**/試管透明/試管數(1~4)。**注意相機也會抖 → 每集視角不同。**寫到另一個資料集,避免混進乾淨的 74 集:

```bash
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR \
  --repo_id ChihHanShen/bimanual-so101-pickvials \
  --repo_root $(pwd)/datasets/bimanual-so101-pickvials \
  --task_name "Pick up the vials and place them into the rack"
```

鍵盤操作、schema 檢查、上傳都跟第 4~6 步相同。

## 5. 上傳到 HuggingFace(手動、獨立步驟)

```bash
hf auth login                       # 第一次才要

# 注意旗標:--repo-id / --root(連字號),--root 要對上錄製的 $DATASET_ROOT
lerobot_push_dataset --repo-id ChihHanShen/bimanual-so101-pickvials --root $(pwd)/datasets/bimanual-so101-pickvials
```
# 私有資料集加 --private
```

> ⚠️ `--root` 不給會去預設 cache `~/.cache/huggingface/lerobot/<repo_id>` 找,
> 但你錄到的是 `$DATASET_ROOT`,所以**一定要帶 --root**。

## 6. 檢查資料集(確認 schema 正確)

```bash
# 資料夾長相
ls $(pwd)/datasets/bimanual-so101-pickvials              # 應有 meta/ data/ videos/
cat $(pwd)/datasets/bimanual-so101-pickvials/meta/info.json | head

# 用 LeRobot 讀出來驗特徵
python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="ChihHanShen/bimanual-so101-pickvials", root="$(pwd)/datasets/bimanual-so101-pickvials")
print("episodes:", ds.meta.total_episodes, "frames:", ds.meta.total_frames, "fps:", ds.meta.fps)
for k, v in ds.meta.features.items():
    print(k, v.get("shape"), v.get("names"))
PY
```

驗這幾點:

- `observation.state` 與 `action` 形狀 **(12,)**、`names` 為 6 個 `left_*` + 6 個 `right_*`。
- 有 **3 個** `observation.images.*`(`wrist_left` / `wrist_right` / `center`),480×640。
- `fps=30`、`episodes` = 你錄的集數。

## 7. 刪除品質不好的 episode(用「檔號 file-XXX」刪)

**先列出目前每一集**(檔號、長度、可開來看的影片路徑):

```bash
python tools/list_episodes.py
```

**開影片確認哪幾集不好,記下它的檔號,再用檔號刪**(可多個):

```bash
python tools/delete_episodes.py 47 30      # 刪掉 file-047、file-030 那兩集
```

- **用「檔號」刪,不是 episode_index。** 檔號穩定(刪掉後永遠空著、不會被重用);
  episode_index 每次刪都會重排,容易搞錯。工具會在刪之前印出「檔號 → episode_index」對照讓你確認。
- 檔號打錯或已刪 → 直接報錯、不動資料。
- 純刪除,不會自動上傳;要更新 Hub 就刪完再跑第 5 步的 `lerobot_push_dataset`。
- 內建安全網:先寫到暫存目錄並驗證能載入,才替換正式資料;失敗時原始資料留在 `..._bak`。

> ⚠️ **不要直接用官方 CLI `lerobot-edit-dataset`。** 它假設資料集在 `<root>/<repo_id>`
> (org/名稱 巢狀結構),但本專案是扁平的 `datasets/bimanual-so101-pickvials`,
> CLI 會把結果寫進 `datasets/bimanual-so101-pickvials/ChihHanShen/...` 這種巢狀錯位置、
> 不會替換到你的資料集。wrapper 用明確輸出路徑繞過這個問題。

> 檔名本來就會有空號(刪過的檔號空著、不連續),屬正常、不影響訓練/上傳。
> 最後全部收集完、要整理成漂亮的連續編號時,再跟我說跑一次「重新切分」即可。

---

# Eval(Phase 6:純 sim 雙臂 GR00T 在 sim 裡評估)

**全本機、無 Docker**、兩段式、兩個**獨立環境**,透過 ZMQ `localhost:5555` 溝通:
- **終端機 A**:GR00T server 跑在 `~/Isaac-GR00T` 的 **uv 環境**(`uv run python ...`,首次自動建)。
- **終端機 B**:`lerobot_eval_dual` 跑在 **`~/env_isaaclab`**,用專案自己的 `gr00t_client` 連 server(跟 GR00T 的 uv 環境無關)。

state/action 依 `examples/SO101_bimanual/modality.json` 分成
`left_arm`(0:5)/`left_gripper`(5:6)/`right_arm`(6:11)/`right_gripper`(11:12);
相機 `center`/`wrist_left`/`wrist_right` 已對上,不用 rename。

## A. 起 GR00T server(終端機 A:`~/Isaac-GR00T`,uv 環境)

⚠️ **`--model-path` 要指到 checkpoint 那一層(有 config.json 的目錄),不是 HF repo 根。**
本專案的 HF repo `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials` 把 checkpoint **巢狀**放在
`pickvials-n1p7-run1/checkpoint-XXXXX/` 底下,repo 根**沒有 config.json** → 直接給 repo id 會報
`Unrecognized model ... should have a model_type key in its config.json`。

先下載選定的 checkpoint(`*` 會遞迴含 experiment_cfg/ 的 stats),再指本機路徑:

```bash
cd ~/Isaac-GR00T

# 1) 下載 checkpoint-50000(最新;約 6GB。要比較就換成 15000/10000/5000)
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  --include "pickvials-n1p7-run2/checkpoint-20000/*" \
  --exclude "*global_step*" \
  --local-dir ~/models/bimanual-pickvials

# 2) 起 server,--model-path 指到那個 checkpoint 目錄
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials/pickvials-n1p7-run2/checkpoint-50000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
    --device cuda:0
```

- `--embodiment-tag new_embodiment` = 訓練時用的 tag(checkpoint 的 embodiment_id.json 確認 `new_embodiment: 10`)。
- `--modality-config-path` 指向 bimanual config(.py),讓 server 知道 12 維怎麼切、3 相機怎麼對。
  **省略且該 tag 沒內建 modality config → server 會報錯要你補這個。**
- 看到 `Server is ready and listening on tcp://...:5555` 才算起好,再開終端機 B。

> ⚠️ **單 GPU 要注意 VRAM**:server(policy 推論)和 Isaac Sim(渲染+物理)同一張卡會搶記憶體。
> 多卡可用 `--device cuda:1` 把 server 丟到另一張。
> port 被占(`Address already in use`)→ server 加 `--port 5556`,client 也帶 `--policy_port 5556`。

## B. 跑 eval(終端機 B:`source ~/env_isaaclab/bin/activate`)

```bash
# 乾淨場景評估(10 集)
lerobot_eval_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 \
  --policy_host localhost --policy_port 5555

# DR 場景評估(外觀隨機,較嚴格)
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-DR-Eval --num_episodes 10 --rerun

# 加 --rerun 開 Rerun 視覺化(看 policy 收到的畫面 + 動作)
```

**成功判定(dual-safe)**:所有「啟用中」試管(沒被 `hide_random_vials` 藏掉的)都同時放進中央架、
且連續 25 步維持(直立、在架子範圍內、已放開),episode 才算成功並提前結束;超過 7.5 分鐘則 time_out。
被藏起來的試管不計入 → 天然支援隨機 1~4 支。

**跑完會印**:`Success Rate: <成功數>/<總集數> (<%>)`。

> ⚠️ server 的 modality/embodiment 一定要對上訓練設定,否則維度或語意對不上、動作會亂。
> 若 rollout 看起來完全不動或亂飛,先確認 server 這端的 `--modality-config-path` 與 `--embodiment-tag`。

## C. 錄 eval 過程當 demo 影片

加 `--record_video`,eval 時把 **`camera_center` 俯視**每一集存成一支 mp4(headless 也能錄,不用開視窗)。

```bash
# 錄 demo:每集一支 mp4，落在 ./eval_videos/
lerobot_eval_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 10 \
  --record_video --headless

# 自訂輸出資料夾 / fps
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --record_video --video_dir ./demo_out --video_fps 30 --num_episodes 10 --headless
```

- 檔名帶結果:`ep001_success.mp4` / `ep002_fail.mp4`(成功=所有啟用試管放進架子並提前結束)。
- 錄的是**中央俯視**(看得到桌面/試管/架子,看不到手臂側面)。錄的畫面 = policy 的中央相機輸入,零額外 render 開銷。
- 按 `R` 手動中斷的那一集**不存**(丟棄半截)。
- 每集結束會**同步編碼**,sim 會短暫暫停一兩秒(正常)。
- 480×640、libx264;相機解析度目前是偶數所以沒問題。要斜角全景 demo 機位要另外加(目前場景只有三台任務相機)。
