# 雙臂 SO-101 Sim-to-Real Roadmap

**目標**:單臂模擬環境改雙臂 + RealSense D435i ×3,雙臂遙操作收資料 → 訓練 GR00T →
DR + 真實資料 co-training → sim-to-real。

**現在位置**:Phase 8。sim eval(50%,基準線)、真機 50 集、co-training checkpoint 都已完成,
**唯一還沒打通的是真機雙臂 real eval**。

最後更新:2026-08-10

> 這份文件只記**決策與狀態**。指令在:[run_cheatsheet.md](run_cheatsheet.md)(sim)、
> [install_cheatsheet.md](install_cheatsheet.md)(安裝)、
> [lerobot/run_cheatsheet.md](../lerobot/run_cheatsheet.md) 與
> [lerobot/CALIBRATION.md](../lerobot/CALIBRATION.md)(真機)。

---

## 已定案的設定

| 項目 | 決定 |
| --- | --- |
| **任務** | 協作式雙臂 vials-to-rack:**任一臂可拿任一支**,全部放進**中央共用架**(不分左右邊) |
| **試管** | 場景固定 4 支,`hide_random_vials` 每集隨機藏幾支 → 等效 1~4 支 |
| **佈局** | 兩臂並排面向 +x,base (-0.05, ±0.15, 0),間距 0.30 m ✅ 真機已驗證 |
| **架 / 墊** | 中央共用架置於墊子中心 world (0.22, 0);mat x[0.068, 0.372] × y[-0.229, 0.229] |
| **相機** | 3 台(左腕/右腕/ego 俯視),**只收 RGB** |
| **ego 相機**(已量測) | world ≈ (-0.23, 0.03, 0.53),俯視約 45°;focal 15.245mm、aperture 20.955×15.716 → **HFOV≈69°**(= D435 color)。相對 LightBox:translate (0, 0.4, 0.5)、euler (45°,0,-90°) |
| **語言指令** | 單一 embodiment-agnostic 句:`"Pick up the vials and place them into the rack"` |
| **dataset schema** | 12 維 state/action(`left_*`×6 + `right_*`×6)+ 3 相機(`wrist_left`/`wrist_right`/`center`,480×640)+ fps=30 |
| **關節正規化** | `RANGE_M100_100`(±100),sim / 真機同一套。`use_degrees` 一律不設 |
| **校正檔** | 統一放 `lerobot/calibration/`(git 追蹤),sim 與真機**讀同一份**,不用 HF cache |

**兩個核心觀念**:① 真實場景**幾何要**對齊 sim(尺寸、相機位姿/FOV、佈局),**外觀不要**(交給 DR)。
② GR00T 只吃 RGB + 關節狀態,**沒有內建物件計數器** → 要對「試管數量」魯棒就得在資料裡涵蓋不同數量。

---

## 進度總覽

```
Phase 1-3   單臂 USD / 雙臂 Isaac Lab 環境 / 三相機        ✅
Phase 4A/B  雙臂遙操作收資料 + 數量隨機                    ✅  74 集,已上 HF Hub
Phase 5     Domain Randomization                           ✅  桌面紋理暫由 HDRI 補
Phase 6     GR00T 訓練(純 sim)+ sim eval                 ✅  成功率 50%(基準線)
Phase 7     真機對齊 + 收真實資料                          ✅  50 集
Phase 8     Co-training + 真機 eval                        ← 現在
Phase 9     進階 sim-to-real(Cosmos / SAGE+GapONet)       尚未探索

技術債      收斂成單一 LeRobot repo(n1.7-graphen, v0.4.3)  程式碼✅ / 真機驗證未跑
```

---

## Phase 1–7 現況速查

**場景**:`SO-ARM101-USD-d435i-physics.usd`(相機/支架含 physics,重量影響動力學)、`RSD435i.usd`、
`Wrist_cam_mount_D435_clean.usd`、`lightbox-egocam.usd`。三層 config 繼承:
`so101_dual_env_cfg`(12 維 action,左 0:6 / 右 6:12)→ `so101_dual_task_env_cfg`(三相機 + DR 群組)
→ `so101_dual_vials_env_cfg`(試管、架、contact sensor)。
Env:`Lerobot-So101-Dual-{Base,Vials-To-Rack,Vials-To-Rack-DR,-Eval,-DR-Eval}`。
三相機皆 `spawn=None` 指向 USD 內烤好的 prim,**由 script 掃 `camera_` 前綴自動發現**。單臂鏈沒動、照樣可用。

**任務邏輯(mdp/)**:`any_vial_grasped`/`vial_placed_on_rack` 用 per-sensor 狀態,左右互不干擾;
`all_active_vials_placed_termination` 跨兩個 contact sensor,成功 = 所有**啟用中**試管同時在架上連續 25 步。

**DR**:燈箱曝光 + 25 張 HDRI、雙臂各自隨機顏色、架 8 色、試管透明度、桌面旋轉 ±0.3。
相機抖動分工:**兩腕抖焦距、ego 只抖位姿**(ego 的 HFOV 已量測對齊真機,抖焦距會破壞對齊)。
〰️ 桌面**紋理**未做,暫由 HDRI 補。

**資料與模型**:sim 74 集 → `ChihHanShen/bimanual-so101-pickvials`;真機 50 集;
純 sim 與 real+sim co-training 兩個 checkpoint 都已訓好。
遙操作用兩支 leader,port 走 udev 固定名稱、**校正檔與真機共用** `lerobot/calibration/bimanual_leader/`
(`lerobot_agent_dual` 的預設值已指過去,不用設環境變數)。
刪壞集用 `tools/delete_episodes.py <檔號>`(**檔號、非 episode_index**)。

**sim eval 基準線**:`run_gr00t_server.py`(ZMQ 5555)+ `lerobot_eval_dual` 跑 rollout。
**純 sim checkpoint 在乾淨場景與 DR 場景成功率都是 50%** → 沒被外觀 DR 拖垮,這是真機 eval 的對照基準線。

---

## Phase 8 — Co-training + 真機 eval ← 現在

- [x] co-training finetune;`eval_so101_dual.py` 寫完;server 通聯驗證通過
- [ ] **雙臂真機 eval** —— 目前唯一還沒打通的環節
- [ ] 量真機成功率,迭代調 DR / 補資料 / 修相機對齊

✅ **完成標準**:真機雙臂成功率達標,sim-to-real gap 收斂。

**做法**:Isaac-GR00T 官方 server/client。client 是以官方 `SO100/eval_so100.py` 改寫的
`Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual/eval_so101_dual.py`(235 行,官方 291 行),
server 沿用 sim eval 那顆。理由:**推論路徑與訓練同源**(normalization、relative-action、
影像 crop/resize 全由 checkpoint 的 `processor_config.json` 決定),且 sim/real 共用同一個 server,
兩邊成功率才可比。

### ⚠️ 四個會直接弄壞真機的點

1. **`use_relative_action` = `true`。** checkpoint 有兩份值不同的 config,**生效的是
   `processor_config.json`**(`Gr00tPolicy` 走 `AutoProcessor.from_pretrained`),不是
   `experiment_cfg/final_model_config.json`。所以 `left_arm`/`right_arm` 輸出是**相對當前 state 的 delta**、
   兩個 gripper 是絕對值。走 Isaac-GR00T 路線會自動讀,設不錯;**把 delta 當絕對關節角送進真機會暴衝**。
2. **控制頻率 10 Hz,不是 30 Hz。** 官方一致做法是 **control rate = 訓練資料集 fps**(DROID 寫死 15、
   SO100 用 1/30),不內插。本專案 checkpoint 吃 10fps 資料集(已查 `meta/info.json`)。
   錄製 config 的 30 fps 是**擷取**頻率。**用 30 Hz 會快 3 倍。**
   → 16 步 chunk 覆蓋 1.6 秒;實測推論延遲 0.070 s,23 倍餘裕,不會 stop-and-go。
3. **server 必須跑在 Isaac-GR00T 自己的 `.venv`**(pin `torch==2.9.0`/`transformers==4.57.3`)。
   借用別的 venv = 影像前處理/tokenizer 跟訓練時不同,「推論路徑同源」的選型理由就沒了。
4. **不要用 workshop 的 `docker/real/scripts/so101_eval.py`** —— 那是**單臂**版(6 維
   `single_arm`+`gripper`、相機 `front`/`wrist`、寫死 30 Hz、passive_mode 走單臂才有的 `robot.bus`),
   三個地方都對不上雙臂 checkpoint。

### 部署(two-terminal,與 sim eval 同構)

```bash
# A:server —— Isaac-GR00T 自己的 3.12 .venv
cd ~/sim2real/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-cotrain/pickvials-n1p7-run3/checkpoint-25000 \
    --embodiment-tag NEW_EMBODIMENT --port 5555

# B:client —— env_isaaclab,一次執行 = 一集
source ~/env_isaaclab/bin/activate
cd ~/sim2real/Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual && python eval_so101_dual.py
```

硬體參數(port / 校正目錄 / 相機序號 / task 字串)都是 script 預設值。換 checkpoint 只重啟 A。

> **client 沒有第三個 venv。** 它同時需要 lerobot(硬體)與 `gr00t`(只有 ZMQ 薄片,不碰 torch),
> 兩者都在 `env_isaaclab`。`gr00t` 宣告 `requires-python >=3.12` 而這個 venv 是 3.11,
> 用 `--ignore-requires-python` 裝進去即可(見 install_cheatsheet Step 6),已實測可用。

### 成功率協定

- **二元判定**:90 秒內 4 支全部坐進架子才算成功(對齊 sim 的 all-or-nothing);
  每集額外記 `vials_placed` (0-4) 當第二指標。
- `f` = 判斷已無法挽回時提前中止(相對 sim truncation 的刻意偏離,要寫進報告);
  `r` = 只用於操作者/硬體失誤,不進分母。
- **N=20**(N=10 時 50% 的 95% CI 約 ±31 個百分點,分不出 30% 和 70%)。

**⚠️ 跟 sim 的 50% 對照時三個不對等要註明**:① sim 跑 60 Hz,16 步 chunk 0.27 秒跑完,
比資料代表的 1.6 秒快 6 倍(physics-time vs data-time 的產物,不該轉移);
② sim 是 `confirm_steps=25` 自動判定、真機是人判;③ sim 22.5 s vs 真機 90 秒。
→ **50% 是參考點,不是嚴格 baseline。**

### 剩下的驗證順序

- [x] checkpoint 契約吻合、server 通聯實測(延遲 0.070 s)、12 顆馬達正常、相機 key 無前綴
- [x] client 環境就緒(`gr00t` 已裝進 `env_isaaclab`,`--help` 實跑通過)
- [ ] **首次帶電**:1 集、`max_relative_target` 全設 0.5、手放電源開關
- [ ] **全速單集**,跟 `lerobot-replay` 的真人 demo 並排比對速度(驗 10 Hz)
- [ ] **20 集正式跑**,再換純 sim checkpoint 重跑一輪做對照

---

## Phase 9 — 進階 sim-to-real(尚未探索)

官方 Strategy 3 / 4,文件都只寫**單臂**流程,套到雙臂前要先想清楚怎麼擴。

- **[Cosmos 資料增強](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html)**
  —— world foundation model 做 video-to-video,生成「外觀不同、任務結構不變」的變體,與 DR 互補。
  **雙臂待解**:官方是單相機,3 相機要先確認多視角一致性,否則變體會對不上。
- **[SAGE + GapONet](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)**
  —— 處理 DR/co-training/Cosmos 都沒碰到的**致動落差**(backlash、接觸動力學、URDF 誤差)。
  [sage](https://github.com/isaac-sim2real/sage) 收 sim/真機 paired 資料算每關節 gap,
  [gaponet](https://github.com/jiemingcui/gaponet) 訓「指令 → 實際動作」反過來補償。
  可①補進 sim 訓練或②只在部署時補償。官方範例不是 SO-101 原生任務,要另外確認怎麼接。

**定位**:等真機 eval 跑起來、看到具體 gap 表現後再決定投入哪一個。

---

## 技術債 — 收斂成單一 LeRobot repo

**程式碼與環境已完成**:`~/sim2real/lerobot` @ `n1.7-graphen`(v0.4.3),editable 裝進 `~/env_isaaclab`,
sim + 真機共用。`~/lerobot-pinned` worktree 與 `lerobot/.venv` 都已移除。

<details>
<summary>遷移細節 / 為什麼是「真機端往下收斂」</summary>

原本的方案是升 Isaac Sim 才能把 sim 端拉到 0.6.1;改成把真機端拉回 0.4.3,理由:
① NVIDIA 的 GR00T 參考 pipeline 本來就 pin 0.4.x(`c75455a6` = v0.4.1),我們的 `e670ac5d` 是 v0.4.3,
同一條線而且更新;② 0.4.x 預設 `use_degrees=False`(±100),與 NVIDIA 一致 —— 0.6.1 把預設翻成度,
正是 sim/真機慣例分歧的來源;③ 0.4.x 要 Python ≥3.10,`env_isaaclab`(3.11)直接可用,
**完全不用動 Isaac Sim**;④ sim 端一行都不用改。

**做了什麼**:四個 YAML 改成 0.4.x 扁平欄位;新增 `bi_so101_follower`/`bi_so101_leader`
(上游 `bi_so100_*` 包的 SO100 類別會把 `wrist_roll` 寫死成 0–4095,見
[CALIBRATION.md](../lerobot/CALIBRATION.md) §0 規則②),順帶修掉 `lerobot_calibrate.py`
從未 import 任何雙臂類別的 bug;`eval_so101_dual.py` 改用新型別、補修相機 teardown。

**0.4.3 缺的東西**:沒有 `lerobot-rollout`(→ `lerobot-record --policy.path=...`);
`--policy.resize_shape` 不存在、`--policy.crop_ratio` → `--policy.crop_shape=[H,W]`
(⚠️ 預設 `(84,84)` 是給 PushT 用的,對 480×640 會裁成一小塊而且不報錯)。

</details>

**尚未完成 —— 下面六步一步都還沒跑**(需要四支手臂都接上):

```
1. lerobot-calibrate 按 ENTER 重用既有校準檔,確認能寫回馬達(四支)
2. 補校 leader_right 的 wrist_roll(單臂 so101_leader 型別,見 CALIBRATION.md §1 末尾)
   → 校完 span 應落在 3800-3950,不再是 4095
3. lerobot-teleoperate → 每個關節推到兩端硬限位,確認 leader/follower 都讀到 ≈ ±100
4. lerobot-record 錄 1 集 → stats.json 值域在 ±100 內、camera key 無前綴、fps 穩不穩
5. sim 端不動,確認 lerobot_agent_dual / lerobot_eval_dual 仍正常
6. 正式重收真機資料
```

---

## 風險預警

- **⚠️ 舊真機資料集是「度」錄的,要重收。** 慣例分歧已在收斂中消掉(兩邊都是 ±100),但遷移前的
  真機資料是 DEGREES(`...-real-15fps` 的 `wrist_roll` stats 到 **-163.9**,超出 ±100 就是證據)。
  換算比例**隨每個關節的 span 不同**(`wrist_roll` ×1.717 差最多),同一個物理姿態新資料記 `+80`、
  舊資料記 `+137`。**不要混在同一個 `embodiment_tag` 下訓練** —— `_merge_statistics()` 只在同 tag 內
  取包絡,數值大的那份決定包絡、另一份被壓縮,而且只印 warning 不報錯。

  > ±100 本身沒問題:GR00T 對 state/action 做 min-max 到 [-1,1],統計量**以 `embodiment_tag` 為 key**、
  > 由自己的 dataset 算出,affine 差會完全抵消。鐵律只有一條:**部署要跟收資料時同一套單位**。

- **⚠️ `leader_right` 的 `wrist_roll` 校準檔還是舊的**(寫死 `0–4095`,另三支是掃出來的 3873–3910)。
  成因已修(改用 `bi_so101_*`),但那支要補校一次(技術債步驟 2)。在 ±100 下 span 就是尺度,
  寫死 4095 = 該軸 gain 錯約 5%,實際後果是兩端各約 5% 行程 teleop 碰不到。

- **真機 eval 是自訂工程**;**收 demo 難度高**(雙手同時操作要練習,初期廢片多);
  **雙 leader teleop** 的 port 每次插拔會變;**多台 RealSense** 可能撞 USB 頻寬 → 分 controller 或降 fps;
  **藏試管**要完全不入鏡;**試管 `_UPRIGHT` 其實是橫躺**(既有誤名),reset yaw 保持 0。

---

## 進度日誌

- **2026-06-23** 建立 roadmap;定案間距 0.30m、ego 相機置中
- **2026-06-29** 任務定為**協作式**(任一臂拿任一支 → 中央共用架);P1~P3 完成
- **2026-07-01** P4A 錄製程式建好;量測確認 ego 相機(HFOV≈69°)
- **2026-07-06** P4A 協作式改造完成;P5 DR 大致完成
- **2026-07-13** P4A 收滿 74 集上 Hub;P4B/P5 完成;**純 sim 雙臂 GR00T 模型訓好**;P6 基礎設施建好
- **2026-07-24** **P6 實跑:乾淨與 DR 場景成功率都是 50%** → 定為真機基準線。
  同時確認 P7 已完成(50 集 + co-training checkpoint);Cosmos/SAGE 記為 Phase 9
- **2026-07-27** 真機 eval 做法定案。查證釐清 **`use_relative_action` 以 `processor_config.json`
  為準 = `true`**,弄錯會暴衝;全面移除 depth
- **2026-07-28** `eval_so101_dual.py` 寫完(235 行;一度膨脹到 1100 行後砍回)。
  **查出會讓手臂快 3 倍的問題:控制頻率是 10 Hz 不是 30 Hz**,據此決定馬達寫入不做內插。
  前置驗證完成:契約吻合、server 延遲 0.070 s、12 顆馬達正常
- **2026-08-10** **真機端 0.6.1 → 0.4.3,收斂成單一 checkout + 單一 venv**。
  新增 `bi_so101_*` 解決 `wrist_roll` 被寫死,順帶修掉 `lerobot_calibrate.py` 的上游 bug。
  慣例統一成 ±100 → sim/真機分歧消失,但**舊真機資料要重收**。
  `gr00t` client 薄片裝進 `env_isaaclab`,eval 不再需要第三個 venv。
  文件精簡:`CALIBRATION.md` 365→187、`run_cheatsheet.md` 213→165、本檔 440→249 行
