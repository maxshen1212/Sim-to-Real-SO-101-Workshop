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
