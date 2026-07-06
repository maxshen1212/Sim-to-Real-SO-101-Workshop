# 雙臂 SO-101 Sim-to-Real Roadmap

**目標**:把單臂模擬環境改成雙臂,換上自製支架 + RealSense D435i,用 3 相機收 RGB+depth,
雙臂遙操作收資料 → 訓練 GR00T → Domain Randomization + 真實資料 co-training → sim-to-real。

最後更新:2026-07-06

---

## 已定案的設定

| 項目 | 決定 |
| --- | --- |
| **任務** | **協作式雙臂 vials-to-rack**:試管隨機散在墊子上,**任一隻手臂都可拿任一支**試管,目標是把**所有試管放進中央共用試管架**。(不是 handoff、不綁定左右分邊) |
| **試管數量** | 目前**固定 4 支**、位置隨機(選項 1);之後改成**每集數量隨機 1~4 支**(選項 2,見 Phase 4B)以增加數量魯棒性 |
| **試管架** | **1 個中央共用架**(body 中心置中於墊子中心 world (0.22, 0)),兩臂都搆得到 |
| **佈局** | 兩臂並排、皆面向 +x;左臂 base (-0.05, +0.15, 0)、右臂 (-0.05, -0.15, 0) |
| **間距** | 0.30 m(左臂 y=+0.15、右臂 y=−0.15)✅ 真機已驗證 |
| **相機** | 3 台:左腕、右腕、ego(燈箱開口置中俯視),收 RGB + depth |
| **ego 相機(已量測確認 2026-07-01)** | world 位置 ≈ **(-0.23, 0.03, 0.53)**(離地高 ~0.53m),朝墊子中心 (0.22,0) 俯視約 45°;intrinsics focal 15.245mm、aperture 20.955×15.716 → **HFOV≈69°**(= D435 color)。ego_cam 相對 LightBox 的 local:translate (0, 0.4, 0.5)、euler (45°,0,-90°)。畫面已在 `camera_center` 目視確認**置中正確** |
| **depth** | 只存檔備用,**不進 policy** |
| **工作墊** | mat.usda,world x[0.068, 0.372](前後 0.3048m)× y[-0.229, 0.229](左右 0.4572m),中心 (0.22, 0) |
| **語言指令** | **沿用單臂那句**(embodiment-agnostic):`"pick up the vials and place them into the rack"`。不寫「左手做什麼右手做什麼」,雙臂差異靠 embodiment tag + 12 維 state 表達 |
| **dataset schema** | 12 維 state/action(命名 `left_*` 6 個 + `right_*` 6 個)+ 3 相機(wrist_left/wrist_right/center,480×640)+ fps=30。**選項 1↔2 不改此 schema,可同一 dataset 無縫 append** |

---

## 兩個核心觀念

1. **真實場景要和 sim 一樣嗎?** → 幾何要、外觀不要。
   - **要對齊**:機器人尺寸、相機安裝位置/角度、相機 FOV/解析度、工作區佈局。
   - **不用對齊(交給 DR)**:背景、桌面紋理、光照、顏色。
2. **GR00T 吃什麼?** → 相機 RGB + 關節狀態,不需要物體的 3D 世界座標。
   GR00T 是「看畫面反應」的策略,**沒有內建物件計數器** → 想讓它對「試管數量」魯棒,
   就要在收資料時涵蓋不同數量(這就是選項 2 存在的理由)。

---

## 階段總覽

```
Phase 1  單臂 USD(D435i 支架+相機+physics)         ✅ 完成
Phase 2  雙臂 Isaac Lab 環境                         ✅ 完成
Phase 3  三相機 + RGB/Depth                          ✅ 完成
Phase 4A 雙臂遙操作 + 收資料(固定 4 支,選項 1)     ← 現在(錄製程式已建好,待實跑驗證)
Phase 4B 數量隨機(選項 2,1~4 支)                   ← 之後補,增加數量魯棒性
Phase 5  Domain Randomization                        🔶 大致完成(紋理待補),待 sim 驗證
Phase 6  GR00T 訓練(純 sim)+ dual-safe 成功判定
Phase 7  真機對齊 + 收真實資料
Phase 8  Co-training + 部署
```

---

## Phase 1 — 單臂 USD ✅ 完成

換上自製支架 + RealSense D435i,**含 physics(質量 + 碰撞)**,因為是 sim-to-real,
實體相機重量會影響任務動力學。相機/支架用 Approach A(質量加在 collision shape、不另建 rigid body,
避免自由落體),掛在 gripper body 上。

- ✅ 自製自封裝 D435i(`RSD435i.usd`,Camera prim [translate,orient,scale] 皆 double3)
- ✅ 支架 `Wrist_cam_mount_D435_clean.usd`
- ✅ 手臂 USD `SO-ARM101-USD-d435i-physics.usd`(assets/so101.py 的 `SO101_CFG.spawn.usd_path`)
- ✅ 燈箱 `lightbox-egocam.usd`:拔掉牆上舊 D455,改中央 ego D435i(`ego_cam`)俯視

---

## Phase 2 — 雙臂 Isaac Lab 環境 ✅ 完成

- ✅ 資產:`SO101_DUAL_LEFT/RIGHT_CFG`(+ `_CONTACT_CFG` 變體),base 左 (-0.05, +0.15, 0)/右 (-0.05, -0.15, 0)
- ✅ `so101_dual_env_cfg.py`:`robot_left/right`、`ee_frame_left/right`、12 維 action(左 0:6、右 6:12)、左右關節觀測
- ✅ 註冊 `Lerobot-So101-Dual-Base`
- ✅ 每一層是**獨立新檔**,單臂鏈完全沒動、照樣可用

---

## Phase 3 — 三相機 + RGB/Depth ✅ 完成

- ✅ `so101_dual_task_env_cfg.py`:`camera_wrist_left/right`(掛各臂 gripper 內 d435i)+ `camera_center`(燈箱 ego_cam),皆 `spawn=None` 指向 USD 內已烤好的 Camera prim
- ✅ 三相機 ×(rgb+depth)觀測群組
- ✅ 相機是**自動發現**的:script 掃 `camera_` 開頭物件,命名對了 recorder 就自動收
- ✅ 註冊 `Lerobot-So101-Dual-Task`

---

## Phase 4A — 雙臂遙操作 + 收資料(固定 4 支,選項 1)← 現在

**任務環境**(`so101_dual_vials_env_cfg.py`,註冊 `Lerobot-So101-Dual-Vials-To-Rack`):
- ✅ 4 支試管 + 中央共用架 `rack_center`
- ✅ 兩個夾爪接觸感測器 `contact_grasp_left/right`
- ✅ per-sensor 狀態重構(mdp/terms.py):`any_vial_grasped`/`vial_placed_on_rack` 改用 `func._state[sensor_name]`,左右臂不互相干擾

**遙操作 + 錄製**(`scripts/lerobot_agent_dual.py`,console `lerobot_agent_dual`):
- ✅ 兩支 leader(env 變數 `TELEOP_PORT_LEFT/RIGHT` + `TELEOP_ID_LEFT/RIGHT`,每次插拔要重設)
- ✅ 12 維動作(左 0:6、右 6:12);鍵盤 R reset、S 起停錄製
- ✅ `lerobot_recorder.py` 參數化 `joint_names`(單臂預設 6;雙臂傳 12 個 `left_*/right_*`)
- ✅ 驅動路徑已驗:兩臂各自抓取、`[RACK] ... placed` 正確獨立觸發

**協作式任務改造(把「左右分邊」改成「任一臂拿任一支」)** — ✅ 完成(2026-07-06,待 sim 目視驗證):
> 註:下面 1、2 只影響**除錯/成功訊號**,**不影響錄下的 dataset**(subtask obs 不寫入 dataset);
> 3 影響實際擺放,對「人看到什麼、去夾什麼」才重要。
- [x] **接觸感測器 filter** → `contact_grasp_left` 與 `_right` **各自 filter 全部 4 支**(共用 `VIAL_PRIMS_ALL`)
- [x] **subtask 觀測** → `vial_grasped_left/right`、`vial_placed_left/right` 的 `vials` 改成**全部 4 支**(`VIALS_ALL`,順序對齊 filter)
- [x] **reset 隨機化** → 試管**散佈整個墊子近側**(基準位置打散 + y jitter ±0.01→±0.03),不再固定左 y>0 / 右 y<0;中間兩支放低 x 避開架子 footprint

**收資料驗證** — 待做:
- [ ] 重裝註冊 console script:`uv pip install -e source/sim_to_real_so101/`
- [ ] 跑 `lerobot_agent_dual --task Lerobot-So101-Dual-Vials-To-Rack --repo_id <id> --repo_root ./datasets/dual_vials --task_name "pick up the vials and place them into the rack"`
- [ ] 先錄 3~5 集,驗 dataset:`observation.state`/`action` 為 12 維、`names` 為 `left_*/right_*`、有 3 個 `observation.images.*`
- [ ] 確認無誤再放大(50~100 集)

✅ **完成標準**:雙手 leader 同步驅動 sim 雙臂,試管散在墊子上任一臂都能撿,按 S 錄製,dataset 正確生成(12 維 + 3 相機)。

---

## Phase 4B — 數量隨機(選項 2,1~4 支)← 之後補

目的:讓 policy 對「推論時試管數量 ≠ 訓練時」魯棒(例:訓練常見 4 支,deploy 只有 3 支也要能做完就停)。

**為什麼要**:GR00T 沒有計數器,只對當下畫面反應。若訓練資料每集都固定 4 支,
換成 3 支時常見壞法是「放完 3 支還去戳空位找第 4 支」,因為它沒學過「沒有第 4 支」的畫面。
要它學會「畫面上有幾支就處理幾支、沒了就停」,就得在**收資料時就讓數量變**。

**實作方式**(Isaac Lab 場景物件數固定,做不到真的生成不同數量):
- [ ] 固定 4 支,reset 時**隨機把其中幾支「藏到相機看不到的地方」**(桌面下夠深 / 場外很遠),等效 1~4 支
- [ ] subtask 成功判定**只算「啟用中(沒被藏)」的試管**
- [ ] 藏的位置要確定不入任何相機(含 ego 俯視),避免半截入鏡污染影像

**相容性(已確認,可放心)**:選項 2 只改 reset 擺放 + 成功判定,
**不動 dataset schema**(12 維、3 相機、指令字串、fps 都不變)→ **可以接續選項 1 的同一個 dataset 繼續 append**,
舊 episode 不受影響。混著收(部分固定 4 支、部分數量隨機)對訓練反而更好。

**建議節奏**:先用選項 1 打通「錄製→訓練→推論」整條管線,確認沒問題,再上選項 2 補數量魯棒性
(一次只動一個變因好 debug)。

---

## Phase 5 — Domain Randomization

複用單臂 `vials_to_rack_env_cfg.py` 的 DR event。實作於 `so101_dual_vials_env_cfg.py`
的 `SO101DualVialsDRSceneCfg` / `SO101DualVialsEventDRCfg` / `SO101DualVialsDREnvCfg`。
(2026-07-06 大致完成,待 sim 目視驗證)

- [x] 光照:燈箱曝光(base 層 `reset_lightbox_light_exposure` 已含)+ DR 加 `randomize_sky_light`(換 HDRI + 曝光 + 色溫,25 張 .exr)
- [x] 機器人顏色(左右臂都要):`randomize_robot_color` 已擴成多機器人,DR 傳 `("robot_left","robot_right")` 各自獨立隨機
- [~] 三相機 pose + focal length 抖動:**兩腕相機抖焦距** + **中央 ego 相機抖位姿**。
      刻意沿用單臂分工:中央 ego「不」抖焦距(FOV≈69° 已量測對齊真機,抖焦距會破壞 sim-real 對齊);
      腕相機「不」抖位姿(mount xform 是否支援待 Isaac 內確認)。→ 若確認 mount 可動再補腕位姿。
- [~] 桌面紋理 + **旋轉**:旋轉已加大(base ±0.1 → DR ±0.3);**紋理未做**(目前無 mat 紋理隨機器,
      單臂也只旋轉;要補需另建 texture-swap 函式 + 多紋理 mat 資產,外觀多樣性暫由 HDRI 天空光補)
- [x] 註冊 `Lerobot-So101-Dual-Vials-To-Rack-DR`

✅ **完成標準**:每次 reset 外觀明顯不同,但任務幾何不變。
⚠️ **驗證重點**:DR reset 後左右臂顏色會變、天空光/HDRI 會換、腕相機 FOV 微變、中央相機小幅位移;
   但中央相機 FOV 與所有試管/架子幾何維持不變。

---

## Phase 6 — GR00T 訓練(純 sim)+ dual-safe 成功判定

- [ ] **dual-safe 成功/termination**:`vial_placed_on_rack_termination` 仍用共享 `env._rack_success_counter` + 單 sensor,**尚未雙臂安全**。協作式任務的成功條件 = 「所有(啟用中)試管都在架上」,需重寫成不綁左右、能數多支
- [ ] embodiment tag:`gr00t_client/embodiment_tags.py` 目前沒有雙臂 SO-101 tag,要加
- [ ] dataset push 到 HF(`lerobot_push_dataset`)
- [ ] Isaac-GR00T finetune(雙臂 embodiment)→ checkpoint
- [ ] sim 內評估:`lerobot_eval --task Lerobot-So101-Dual-Vials-To-Rack-Eval`

✅ **完成標準**:純 sim policy 在 sim eval 有合理成功率。

---

## Phase 7 — 真機對齊 + 收真實資料

對齊 geometry,外觀差異交給 DR。

- [ ] 真實雙臂 + 雙 leader,`lerobot-calibrate` 全關節 → [so101_check_calibration.py](docker/real/scripts/so101_check_calibration.py) 驗證
- [ ] **實體安裝**支架 + RealSense D435i,extrinsics 盡量貼近 sim 的 offset
- [ ] 真機工作墊做 18"×12"、雙臂間距 30cm、中央架
- [ ] `lerobot-find-cameras` 找三台 RealSense index
- [ ] 真機遙操作收真實 demo(RGB+depth),schema 與 sim 一致

✅ **完成標準**:真機相機視角/FOV/解析度對得上 sim,dataset 同 schema。

---

## Phase 8 — Co-training + 部署

- [ ] sim(DR)+ real 混合 co-training finetune
- [ ] 部署:`run_gr00t_server.py` 起 server + `so101_eval.py` rollout([docker/README.md](docker/README.md))
- [ ] 量真機成功率,迭代調 DR / 補資料 / 修相機對齊

✅ **完成標準**:真機雙臂任務成功率達標,sim-to-real gap 收斂。

---

## 風險預警
- **收 demo 難度高**:雙手同時操作要練習,初期廢片多 → 預留時間
- **雙 leader teleop** 是自訂程式,非改設定;port 每次插拔會變,要重設 env 變數
- **多台 RealSense** 同接 USB 可能撞頻寬 → 分 USB controller 或降 fps
- **sim depth 太乾淨**,真機 depth 有噪 → 考慮對 sim depth 加噪(但 depth 不進 policy,影響小)
- **12 維雙臂 embodiment** 需確認 Isaac-GR00T 是否原生支援
- **選項 2 藏試管** 要確定完全不入鏡,否則半截入鏡會污染訓練影像
- **dual-safe 成功判定**(Phase 6)是協作式任務的自訂邏輯,比單臂複雜

---

## 進度日誌
- 2026-06-23 — 建立 roadmap,釐清 USD 結構;定案間距 0.30m(真機驗證)、ego 置中、depth 只存檔
- 2026-06-29 — **任務改為協作式**(原 handoff 作廢):任一臂拿任一支 → 中央共用架;P1~P3 完成
- 2026-07-01 — P4A 錄製程式建好(12 維 + 3 相機,`left_*/right_*` 命名);釐清任務語意(不綁左右分邊、數量不限);
  規劃選項 2(數量隨機 1~4 支)為 Phase 4B;確認選項 1↔2 dataset schema 相容可 append;
  試管初始位置往近側調(x 0.23→0.18)
- 2026-07-01 — 調整並目視確認 ego 中央相機:world (-0.23,0.03,0.53)、高 ~0.53m、俯視墊子中心、HFOV≈69°(D435);
  `camera_center` 畫面置中正確,參數記入上表
- 2026-07-06 — 完成 P4A 協作式任務改造(so101_dual_vials_env_cfg.py):兩 contact sensor + 四個 subtask obs 都改吃全部 4 支
  (`VIAL_PRIMS_ALL`/`VIALS_ALL`,順序對齊避免抓取判定張冠李戴);試管基準位置打散、y jitter 加大到 ±0.03,
  中間兩支放低 x=0.12 避開架子 footprint(world x[0.16,0.28] y[-0.06,0.06]);yaw 保持 0(避免立起的試管被大 yaw delta 弄倒)。
  待 sim 目視驗證:reset 後 4 支散落不壓架子、任一臂抓任一支能觸發正確 subtask 訊號
- 2026-07-06 — P5 DR 大致完成:`randomize_robot_color` 擴成多機器人(左右臂各自隨機);新增 DR 場景(sky_light DomeLight)+
  DR 事件(機器人顏色、天空光 HDRI/曝光/色溫、墊子旋轉加大到 ±0.3、兩腕相機焦距抖動、中央 ego 相機位姿抖動);
  註冊 `Lerobot-So101-Dual-Vials-To-Rack-DR`。中央相機刻意不抖焦距(保 HFOV≈69° 對齊)、腕相機不抖位姿(mount 待確認);
  桌面「紋理」隨機化未做(無此函式,暫靠 HDRI 補外觀多樣)。待 sim 目視驗證外觀變、幾何不變
