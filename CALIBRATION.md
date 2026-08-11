# SO-101 雙臂 Calibration

目的：**讓收資料時和 eval 時的 calibration 完全一致**，以及維修過後能判斷有沒有跑掉。

| 路徑 | |
| --- | --- |
| [calibration/bimanual_follower/](../lerobot/calibration/bimanual_follower/)、[calibration/bimanual_leader/](../lerobot/calibration/bimanual_leader/) | 四個 calibration JSON（會被 `lerobot-calibrate` 覆寫） |
| [calibration/config/](../lerobot/calibration/config/) | 四個 YAML + `arm_serials.json`（USB 序號 → 手臂） |

---

## 0. 三條規則

**① 每個關節都要推到真正的機械硬限位。**
六個關節都是「兩端點決定一切」的正規化（身體關節 `RANGE_M100_100`，gripper `RANGE_0_100`）：

```
norm = (raw - range_min) / (range_max - range_min) * 200 - 100
```

`range_min`/`range_max` 同時決定**零點和尺度** —— span 掃少 5%，那條軸的值就全部歪 5%。
（舊的 v0.6.1 走 DEGREES，尺度固定，掃不準只會平移零點；現在不是了。`use_degrees` 一律不設。）

好消息：只要兩次校準都掃到同樣的硬限位，`homing_offset` 會自動抵消 ——
**認真重校本來就會回到原本的正規化**，不需要保存或搬移任何數字。

**② 一定要用 `bi_so101_*`，不要用 `bi_so100_*`。**
`bi_so100_*` 包的是 SO100 單臂類別，會把 `wrist_roll` 當 full-turn 馬達、
跳過掃描直接寫死 `0–4095` → 那條軸的 gain 直接錯掉。四個 YAML 已經指定好了。

> `bi_so101_follower` / `bi_so101_leader` 是本 repo 自己加的
> （[robots/bi_so101_follower/](../lerobot/src/lerobot/robots/bi_so101_follower/)、
> [teleoperators/bi_so101_leader/](../lerobot/src/lerobot/teleoperators/bi_so101_leader/)）——
> 照抄 `bi_so100_*` 並把內部的單臂類別換成 SO101。`id` 沒變 → 讀寫的還是原本那四個 JSON。

**③ 基準就是 git。**
凍結＝`commit`，檢查＝`git status`，還原＝`git checkout`。不需要別的備份機制。

---

## 1. 建立基準（開始收一批新資料前做一次）

**逐支跑單臂型別。** 檔名只由 `id` 決定，而雙臂類別給子臂的 `id` 就是 `<id>_left` / `<id>_right`，
所以單臂寫出來的就是雙臂類別之後會讀的同一個檔 —— 不用改任何程式碼。

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

git -C ~/sim2real/lerobot add calibration/ && git -C ~/sim2real/lerobot commit -m "calib: baseline for <dataset name>"
```

> **為什麼不用雙臂 config 一次跑完。** `bi_so101_*` 是**序列**校準：`connect()` 時兩臂的 port
> 都先開好，但左臂掃完（可能好幾分鐘）才輪到右臂。這段期間右臂的 fd 一直開著空等，
> USB 只要抖一下重新列舉，那個 fd 就永久失效，切到右臂時炸
> `termios.error: (5, 'Input/output error')` —— 而且左臂已經存檔了，變成半新半舊。
> 逐支跑把暴露窗口從幾分鐘縮到幾十秒。
>
> 雙臂 config 仍然可以用來**重用**既有校準（提示時按 ENTER 寫回馬達）：
> `lerobot-calibrate --config_path=~/sim2real/lerobot/calibration/config/bimanual_so101_follower_config.yaml`

已有校準檔時會先問：**ENTER = 把舊檔寫回馬達（不重掃）**，**`c` = 重新校準**。

**commit message 一定要寫 dataset 名稱** —— dataset 本身不存 calibration，
「這批資料是用哪組校準收的」只存在於這行訊息裡（之後靠 `git log calibration/` 回溯）。

校準當下要注意：

- **「擺到行程中間」要盡量準。** `homing_offset` 上限是 `±2047`，擺太偏會讓**下次**校準直接 `ValueError`。
- **夾爪的「開到底 / 閉到底」四支要用同一套定義**，否則左右夾爪尺度不同。
- **校準時不要拉扯線束** —— 四支共用同一個 USB hub，搬動一支很容易讓旁邊那支掉線。

---

## 2. 每次開工前

```bash
graphen-setup-udev              # USB symlink 是否都在、序號是否正確
git -C ~/sim2real/lerobot status calibration/         # 應該乾淨，沒有任何 M
```

出現 `M calibration/...` = 有人重跑過 `lerobot-calibrate`。**不要接受**：

```bash
git -C ~/sim2real/lerobot diff calibration/ && git -C ~/sim2real/lerobot checkout calibration/
```

**正常開工時不該看到 `Press ENTER to use provided calibration file...` 這個提示。**
它只在「檔案 vs 馬達 EEPROM」不一致時才跳。看到就先停下來查；
確定檔案是對的再按 ENTER（寫回馬達，正確動作）。**千萬不要按 `c`**，那會覆蓋基準。

> 這兩個檢查**驗不出機構位移**。那要看 §4。

---

## 3. 什麼時候要重新校準

STS3215 是 12-bit 絕對式磁編碼器，**斷電／重開機／重插 USB 都不會掉**。

| 情況 | |
| --- | --- |
| 關機／重開機／重插 USB／換 USB 孔 | **不用** |
| 換外殼、連桿，但沒動舵機與喇叭盤 | **不用**（不放心就走 §4） |
| 喇叭盤（horn）從花鍵拆下再裝回 | **要**（約 24 齒，差一齒 ≈ 15°） |
| 換一顆舵機 | **要** |
| 舵機 factory reset／刷韌體／改 motor ID | **要** |
| 覺得「怪怪的」 | 先走 §4 找證據，不要憑感覺 |

原則：沒有具體理由就不要重跑；真的要跑，就用 §4 驗證結果。

---

## 4. 維修過後：重校 + 比對

**4.1** 用 §1 末尾的單臂指令只跑動過的那一支，提示時輸入 `c`。**務必推到真正的硬限位。**

**4.2** 比對兩個端點：

```bash
git -C ~/sim2real/lerobot diff calibration/
```

對每個關節算 `E_min = homing_offset + range_min`、`E_max = homing_offset + range_max`，
看新舊的差。**兩端都要看** —— 只比中點會漏掉 gain 誤差。

```
誤差角度 = ΔE × 360 / 4095          （ΔE = 30 counts ≈ 2.6°）
```

| 情況 | 處理 |
| --- | --- |
| `ΔE_min`、`ΔE_max` 都在 ±30 counts 內 | 沒跑掉 → `git -C ~/sim2real/lerobot checkout calibration/` 還原，保持基準乾淨 |
| 兩端同方向平移、span 幾乎沒變 | 機構真的位移了 → 重掃確認 → §5 |
| span 明顯變小、只有一端在動 | 那一端沒推到底 → **重掃** |
| 每次數字都不一樣 | 掃描不穩 → 重掃，檢查關節有沒有卡滯 |
| range 剛好是 `0–4095` | 根本沒被掃（規則②）→ 用 `so101_*` 單臂型別重校 |

> **不要把舊數字搬到新檔案。** 掃到的 range 已經扣掉 `homing_offset`，兩者要一起看才有意義；
> 只貼舊 range 卻留新 `homing_offset`，整條軸會偏掉一整個 offset 差。
> 搬「舊 span」也一樣沒用 —— span 只是症狀，誤差本體在端點上，錨錯邊會讓誤差加倍。
> 要嘛整份舊檔（`git checkout`），要嘛整份新檔，不要混。

---

## 5. 真的跑掉了

**新檔是對的**（硬限位的物理角度沒變，重新掃描已經吸收掉位移）。
**錯的是舊 dataset** —— 那些資料是用舊端點錄的。所以：

1. commit 新的 calibration，message 寫清楚是哪次維修造成的。
2. 換算 `ΔE`：幾度以內通常吞得掉；一顆花鍵齒 ≈ **170 counts ≈ 15°** 就不能當沒事。
   span 也變的話再看尺度誤差（span 差 1% ≈ 該軸端點附近差 2 個 ±100 單位）。
3. 二選一：**重收該手臂的資料**（推薦），或接受 domain gap 並在 eval 報告註明 `ΔE`。

**不要把舊數字硬湊回新機構上** —— 那會讓手臂靜靜地跑到錯誤的物理姿態。

---

## 6. USB 連線（udev）

四支控制板用同一顆晶片（`1a86:55d3` WCH CH343），`/dev/ttyACM*` 的編號會隨插拔改變。
udev 依 **USB 序號**綁成固定名稱 `/dev/tty{Follower,Leader}{Left,Right}`，
規則檔在 `/etc/udev/rules.d/99-robot.rules`，序號表在
[calibration/config/arm_serials.json](../lerobot/calibration/config/arm_serials.json)（跟著 repo 走）。

```bash
graphen-setup-udev              # 檢查目前連線（預設，不改動任何東西）
graphen-setup-udev --apply      # 依序號寫入規則（需 sudo，會先備份；冪等）
graphen-setup-udev --identify   # 只有換過 USB 轉板才需要：重新辨識序號後自動套用
```

- **必須綁 `ATTRS{serial}`**，不能綁 `devpath` / `ID_PATH` —— 綁路徑換孔就會錯，而且錯得很安靜。
- **必須有 `MODE="0666"`**，否則每次插拔都要手動 `chmod`。
- 序號屬於**控制板**，不屬於手臂。換外殼／連桿／舵機都不用動 udev；
  **把控制板換到另一支手臂上，身分就跟著板子走** → 要重跑 `--identify`。

### 三個沒有任何機制會擋的錯

port 和 calibration 檔之間**只靠 YAML 寫得一致**，沒有交叉檢查：

```
USB 序號 ──udev──▶ /dev/ttyFollowerLeft ──▶ YAML: left_arm_port
                                                 ⇕  只靠 YAML 寫得一致
YAML: id + "_left" ──▶ calibration_dir/<id>_left.json
```

1. **YAML 的 port 寫錯 / symlink 指錯** → 右臂載入左臂的 calibration，不報錯。
2. 唯一的半套安全網是 §2 那個提示，但**在那裡按 ENTER 會把錯的 calibration 寫進馬達**，
   蓋掉原本正確的 —— 看起來最像「預設繼續」的動作正是最糟的。
3. **把左右兩支手臂連同控制板整組對調** → 電氣上全對、所有檢查都過，但物理角色反了。
   只能靠實體標籤 + teleop 時目視確認。
