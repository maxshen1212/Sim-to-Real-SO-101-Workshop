# Single Arm

```bash

lerobot-find-port

export TELEOP_PORT=/dev/ttyACM0
export TELEOP_ID=orange_teleop

export ROBOT_PORT=/dev/ttyACM2
export ROBOT_ID=orange_robot

echo "Teleop port is $TELEOP_PORT with id $TELEOP_ID"
echo "Robot port is $ROBOT_PORT with id $ROBOT_ID"

sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM2

sudo chmod -R 777 /home/max/.cache/huggingface/lerobot/
sudo chmod -R 777 /home/air-420/.cache/huggingface/lerobot/

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$TELEOP_PORT \
    --teleop.id=$TELEOP_ID

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID

python docker/real/scripts/so101_check_calibration.py

lerobot-find-cameras opencv

export CAMERA_GRIPPER=6
export CAMERA_EXTERNAL=12

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=$ROBOT_PORT \
  --robot.id=$ROBOT_ID \
  --teleop.type=so101_leader \
  --teleop.port=$TELEOP_PORT \
  --teleop.id=$TELEOP_ID \
  --display_data=true \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": '"$CAMERA_GRIPPER"',
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "front": {
      "type": "opencv",
      "index_or_path": '"$CAMERA_EXTERNAL"',
      "width": 640,
      "height": 480,
      "fps": 30
    }
  }'

lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack-DR

lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack-DR \
    --repo_id ${HF_USER}/so101_teleop_vials \
    --repo_root $(pwd)/datasets/so101_teleop_vials \
    --task_name "Pick up the vial and place it in the rack"
```

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
export TELEOP_PORT_RIGHT=/dev/ttyACM1  TELEOP_ID_RIGHT=leader_right

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
lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack
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
