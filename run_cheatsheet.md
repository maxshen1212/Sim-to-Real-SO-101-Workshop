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

# Bimanual

```bash
source ~/env_isaaclab/bin/activate

lerobot-find-port          # 找左臂,假設 /dev/ttyACM0
lerobot-find-port          # 找右臂,假設 /dev/ttyACM1

export TELEOP_PORT_LEFT=/dev/ttyACM0   TELEOP_ID_LEFT=leader_left
export TELEOP_PORT_RIGHT=/dev/ttyACM1  TELEOP_ID_RIGHT=leader_right
sudo chmod 666 $TELEOP_PORT_LEFT $TELEOP_PORT_RIGHT

# 各校正一次(id 要不同)
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$TELEOP_PORT_LEFT  --teleop.id=$TELEOP_ID_LEFT
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$TELEOP_PORT_RIGHT --teleop.id=$TELEOP_ID_RIGHT

lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack

zero_agent --task Lerobot-So101-Dual-Vials-To-Rack

lerobot_agent_dual \
  --task Lerobot-So101-Dual-Vials-To-Rack \
  --repo_id <hf_id/dual_vials> \
  --repo_root ./datasets/dual_vials \
  --task_name "left arm picks left vials, right arm picks right vials, both into center rack"

# S	開始 / 停止錄製(切換)	第一次按 → 開始收 frame;再按一次 → 停止並存檔(這一段成為一個 episode,丟給背景 thread 寫進 dataset)
# C	取消當前這段	丟棄正在錄的 buffer、不存檔(錄壞了就用這個)。只有錄製中才有效
# R	reset 場景	把試管/架子重新擺放。注意:R 會同時 stop_recording → 如果你正在錄,R 會先存檔再 reset
```
