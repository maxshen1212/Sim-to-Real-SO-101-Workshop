# 雙臂 SO-101 Sim-to-Real Roadmap

**目標**:把單臂模擬環境改成雙臂,換上自製支架 + RealSense D435i,用 3 相機收 RGB+depth,
雙臂遙操作收資料 → 訓練 GR00T → Domain Randomization + 真實資料 co-training → sim-to-real。

**現在位置**:Phase 6 —— 純 sim 雙臂 GR00T 模型**已訓練完成**,下一步是**在 sim 裡建 eval 基礎設施**評估它。

最後更新:2026-07-13

---

## 已定案的設定

| 項目 | 決定 |
| --- | --- |
| **任務** | **協作式雙臂 vials-to-rack**:試管散在墊子上,**任一隻手臂都可拿任一支**試管,目標是把**所有試管放進中央共用試管架**。(不是 handoff、不綁定左右分邊) |
| **試管數量** | 場景固定 4 支;reset 時 `hide_random_vials` 隨機藏掉幾支 → 等效**每集 1~4 支**(見 Phase 4B,已完成)。擺放維持穩定 2 左 / 2 右(全散佈版已還原,以求 reset 穩定) |
| **試管架** | **1 個中央共用架**(body 中心置中於墊子中心 world (0.22, 0)),兩臂都搆得到 |
| **佈局** | 兩臂並排、皆面向 +x;左臂 base (-0.05, +0.15, 0)、右臂 (-0.05, -0.15, 0) |
| **間距** | 0.30 m(左臂 y=+0.15、右臂 y=−0.15)✅ 真機已驗證 |
| **相機** | 3 台:左腕、右腕、ego(燈箱開口置中俯視),收 RGB + depth |
| **ego 相機(已量測確認 2026-07-01)** | world 位置 ≈ **(-0.23, 0.03, 0.53)**(離地高 ~0.53m),朝墊子中心 (0.22,0) 俯視約 45°;intrinsics focal 15.245mm、aperture 20.955×15.716 → **HFOV≈69°**(= D435 color)。ego_cam 相對 LightBox 的 local:translate (0, 0.4, 0.5)、euler (45°,0,-90°)。畫面已在 `camera_center` 目視確認**置中正確** |
| **depth** | 只存檔備用,**不進 policy** |
| **工作墊** | mat.usda,world x[0.068, 0.372](前後 0.3048m)× y[-0.229, 0.229](左右 0.4572m),中心 (0.22, 0) |
| **語言指令** | **沿用單臂那句**(embodiment-agnostic):`"pick up the vials and place them into the rack"`。不寫「左手做什麼右手做什麼」,雙臂差異靠 embodiment tag + 12 維 state 表達 |
| **dataset schema** | 12 維 state/action(命名 `left_*` 6 個 + `right_*` 6 個)+ 3 相機(wrist_left/wrist_right/center,480×640)+ fps=30 |

---

## 兩個核心觀念

1. **真實場景要和 sim 一樣嗎?** → 幾何要、外觀不要。
   - **要對齊**:機器人尺寸、相機安裝位置/角度、相機 FOV/解析度、工作區佈局。
   - **不用對齊(交給 DR)**:背景、桌面紋理、光照、顏色。
2. **GR00T 吃什麼?** → 相機 RGB + 關節狀態,不需要物體的 3D 世界座標。
   GR00T 是「看畫面反應」的策略,**沒有內建物件計數器** → 想讓它對「試管數量」魯棒,
   就要在收資料時涵蓋不同數量(這就是 Phase 4B 存在的理由)。

---

## 階段總覽

```
Phase 1  單臂 USD(D435i 支架+相機+physics)         ✅ 完成
Phase 2  雙臂 Isaac Lab 環境                         ✅ 完成
Phase 3  三相機 + RGB/Depth                          ✅ 完成
Phase 4A 雙臂遙操作 + 收資料(固定 4 支)             ✅ 完成(74 集乾淨、已上 HF Hub)
Phase 4B 數量隨機(1~4 支,hide_random_vials)        ✅ 完成
Phase 5  Domain Randomization                        ✅ 完成(桌面紋理暫由 HDRI 補)
Phase 6  GR00T 訓練(純 sim)+ sim eval              ← 現在:模型已訓練,eval 基礎設施待建
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

## Phase 4A — 雙臂遙操作 + 收資料(固定 4 支)✅ 完成

**任務環境**(`so101_dual_vials_env_cfg.py`,註冊 `Lerobot-So101-Dual-Vials-To-Rack`):
- ✅ 4 支試管 + 中央共用架 `rack_center`
- ✅ 兩個夾爪接觸感測器 `contact_grasp_left/right`,各自 filter 全部 4 支(`VIAL_PRIMS_ALL`)
- ✅ per-sensor 狀態重構(mdp/terms.py):`any_vial_grasped`/`vial_placed_on_rack` 用 `func._state[sensor_name]`,左右臂不互相干擾
- ✅ subtask 觀測 `vial_grasped_left/right`、`vial_placed_left/right` 都吃全部 4 支(`VIALS_ALL`,順序對齊 filter)

**遙操作 + 錄製**(`scripts/lerobot_agent_dual.py`,console `lerobot_agent_dual`):
- ✅ 兩支 leader(env 變數 `TELEOP_PORT_LEFT/RIGHT` + `TELEOP_ID_LEFT/RIGHT`,每次插拔要重設)
- ✅ 12 維動作;鍵盤 S 起停錄製、C 取消當段、R reset(R 會先 stop_recording)
- ✅ `lerobot_recorder.py` 參數化 `joint_names`(雙臂傳 12 個 `left_*/right_*`)

**收資料成果**:
- ✅ 已錄 **74 集乾淨資料**(一集一檔)、schema 驗證正確(12 維 + 3 相機 480×640 + fps=30)、**已上 HF Hub**(`ChihHanShen/bimanual-so101-pickvials`)
- 錄製/清理工作流見 [run_cheatsheet.md](run_cheatsheet.md);刪壞集用 `tools/delete_episodes.py <檔號>`(用檔號、非 episode_index)

---

## Phase 4B — 數量隨機(1~4 支)✅ 完成

目的:讓 policy 對「推論時試管數量 ≠ 訓練時」魯棒(例:訓練常見 4 支,deploy 只有 3 支也要能做完就停)。
GR00T 沒有計數器、只對當下畫面反應,所以要在收資料時就讓數量變。

- ✅ `hide_random_vials`(mdp/resets.py):reset 時隨機把幾支藏到相機看不到的地方,等效 1~4 支;藏的位置確定不入任何相機(含 ego 俯視)
- ✅ **eval 安全**:成功判定只算「啟用中(沒被藏)」的試管
- ✅ 不動 dataset schema → 可接續同一 dataset append,混著收(部分 4 支、部分隨機)對訓練更好

---

## Phase 5 — Domain Randomization ✅ 完成

複用單臂 `vials_to_rack_env_cfg.py` 的 DR event,實作於 `so101_dual_vials_env_cfg.py`
的 `SO101DualVialsDRSceneCfg` / `SO101DualVialsEventDRCfg` / `SO101DualVialsDREnvCfg`,
註冊 `Lerobot-So101-Dual-Vials-To-Rack-DR`。

- ✅ 光照:燈箱曝光(base)+ `randomize_sky_light`(換 HDRI + 曝光 + 色溫,25 張 .exr)
- ✅ 機器人顏色:`randomize_robot_color` 擴成多機器人,左右臂各自獨立隨機
- ✅ 試管架顏色(8 色)+ 試管透明度隨機化
- ✅ 相機抖動(刻意沿用單臂分工):**兩腕相機抖焦距** + **中央 ego 相機抖位姿**。
      中央 ego「不」抖焦距(HFOV≈69° 已量測對齊真機,抖焦距會破壞 sim-real 對齊);
      腕相機「不」抖位姿(mount xform 可動性待 Isaac 內確認,可動再補)
- ✅ 桌面**旋轉**加大(base ±0.1 → DR ±0.3)
- 〰️ 桌面**紋理**:暫未做(需另建 texture-swap 函式 + 多紋理 mat 資產),外觀多樣性暫由 HDRI 天空光補

✅ **完成標準**:每次 reset 外觀明顯不同(左右臂顏色/天空光/架子色/試管透明/腕相機 FOV/中央相機小幅位移),
   但任務幾何不變(中央相機 FOV、所有試管/架子幾何維持不變)。

---

## Phase 6 — GR00T 訓練(純 sim)+ sim eval ← 現在

**訓練** — ✅ 純 sim 雙臂 GR00T 模型已 finetune 完成(checkpoint 就緒)。

**sim eval 基礎設施** — ✅ **已實作**(2026-07-13,對齊 `examples/SO101_bimanual/modality.json`;
待真跑一輪 sim rollout 驗證 + 調參)。client-server 兩段式:起 GR00T server 吃 checkpoint(port 5555)
→ 跑 `lerobot_eval_dual` 當 client 做 rollout 算成功率。指令見 [run_cheatsheet.md](run_cheatsheet.md) 的「Eval」段。

- [x] **dual eval task**:`SO101DualVialsEvalEnvCfg` / `SO101DualVialsEvalDREnvCfg`(terminations + 7.5 分鐘上限);
      註冊 `Lerobot-So101-Dual-Vials-To-Rack-Eval` / `-DR-Eval`
- [x] **dual-safe 成功判定**:`all_active_vials_placed_termination`(mdp/terms.py)——跨左右兩個 contact sensor、
      成功 = 所有「啟用中(沒被 `hide_random_vials` 藏掉)」試管都同時在架上、連續 25 步;天然支援隨機 1~4 支
- [x] **dual client**:`GR00TDualRemotePolicy`(lerobot_interface.py)——左右各一 `LeRobotSO101Interface`、
      state/action 分成 `left_arm`(0:5)/`left_gripper`(5:6)/`right_arm`(6:11)/`right_gripper`(11:12)、12 維
- [x] **dual eval script**:`scripts/lerobot_eval_dual.py`(console `lerobot_eval_dual`)——12 維 action、雙 interface、
      讀 `joint_pos_left`+`joint_pos_right` 與 `rgb_wrist_left/right/center`
- [x] ~~embodiment tag~~:**不用做**——訓練用 `NEW_EMBODIMENT`,client/server 皆已存在
- [x] 相機 key `center/wrist_left/wrist_right` 已對上 modality.json → **不用 rename**
- [] **待真跑**:起 server(`--modality-config-path examples/SO101_bimanual/so101_bimanual_config.py`
      `--embodiment-tag new_embodiment`)+ `lerobot_eval_dual` 一輪,看成功率;必要時調 `active_radius`/
      成功判定 bounds / `action_horizon`

⚠️ **關鍵依賴**:server 的 `--modality-config-path` / `--embodiment-tag` 必須對上訓練設定,否則維度或語意對不上。

✅ **完成標準**:純 sim policy 在 sim eval 跑得動、有合理成功率(dual-safe 成功判定正確計數多支試管)。

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
- **sim eval 是自訂工程**:dual-safe 成功判定 + 12 維 client 改寫比單臂複雜,且 modality key 要對上訓練設定
- **收 demo 難度高**:雙手同時操作要練習,初期廢片多 → 預留時間
- **雙 leader teleop** 是自訂程式,非改設定;port 每次插拔會變,要重設 env 變數
- **多台 RealSense** 同接 USB 可能撞頻寬 → 分 USB controller 或降 fps
- **sim depth 太乾淨**,真機 depth 有噪 → 考慮對 sim depth 加噪(但 depth 不進 policy,影響小)
- **12 維雙臂 embodiment** 需確認 Isaac-GR00T 是否原生支援
- **藏試管**(Phase 4B)要確定完全不入鏡,否則半截入鏡會污染訓練影像
- **試管 orientation**:`_UPRIGHT` 命名其實是橫躺(既有 bug/誤名),reset yaw 保持 0 避免立起的試管被大 yaw delta 弄倒

---

## 進度日誌
- 2026-06-23 — 建立 roadmap,釐清 USD 結構;定案間距 0.30m(真機驗證)、ego 置中、depth 只存檔
- 2026-06-29 — **任務改為協作式**(原 handoff 作廢):任一臂拿任一支 → 中央共用架;P1~P3 完成
- 2026-07-01 — P4A 錄製程式建好(12 維 + 3 相機,`left_*/right_*` 命名);規劃 Phase 4B(數量隨機);
  調整並目視確認 ego 中央相機:world (-0.23,0.03,0.53)、高 ~0.53m、俯視墊子中心、HFOV≈69°(D435)
- 2026-07-06 — 完成 P4A 協作式任務改造(兩 contact sensor + 四個 subtask obs 都吃全部 4 支);P5 DR 大致完成
  (機器人顏色多機化、天空光、墊子旋轉加大、腕相機焦距 + 中央相機位姿抖動)
- 2026-07-13 — **P4A 收滿 74 集乾淨資料並上 HF Hub;P4B 數量隨機(`hide_random_vials`,eval 安全)完成;
  P5 DR 補齊(架子 8 色 + 試管透明度);純 sim 雙臂 GR00T 模型訓練完成**。
- 2026-07-13 — **P6 sim eval 基礎設施建好**:dual-safe 成功 termination(`all_active_vials_placed_termination`)、
  `SO101DualVialsEvalEnvCfg`/`-DR-Eval` 並註冊、`GR00TDualRemotePolicy`(對齊 SO101_bimanual modality:
  left_arm/left_gripper/right_arm/right_gripper 12 維)、`lerobot_eval_dual` script + entry point。
  確認訓練用 `NEW_EMBODIMENT`(不需新 tag)、相機 key 已對上(不需 rename)。`list_envs` 驗證兩個 eval env 註冊成功、
  import 乾淨。**待真跑一輪 sim rollout 驗證成功率 + 調參**(server 記得帶 `--modality-config-path` + `--embodiment-tag new_embodiment`)
