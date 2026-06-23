# 雙臂 SO-101 Sim-to-Real Roadmap

**目標**:把單臂模擬環境改成雙臂,換上自製支架 + RealSense,用 3 相機收 RGB+depth,
雙臂遙操作收資料 → 訓練 GR00T → Domain Randomization + 真實資料 co-training → sim-to-real。

最後更新:2026-06-23

---

## 已定案的設定

| 項目 | 決定 |
| --- | --- |
| **任務** | 雙臂 handoff:右臂取料 → 中間交接 → 左臂放入 rack(沿用現有 vial/rack 資產) |
| **佈局** | 兩臂並排、皆面向 +x;vials 在右側 (y<0)、rack 在左側 (y>0)、交接在中間 (y=0) |
| **間距** | 0.30 m(左臂 y=+0.15、右臂 y=−0.15)✅ 真機已驗證 handoff 可行 |
| **相機** | 3 台:左腕、右腕、ego(置中對準交接點),收 RGB + depth |
| **depth** | 只存檔備用,**不進 policy** |
| **工作墊** | 0.4572 m(左右)× 0.3048 m(前後),中心 y=0 |

---

## 兩個核心觀念

1. **真實場景要和 sim 一樣嗎?** → 幾何要、外觀不要。
   - **要對齊**:機器人尺寸、相機安裝位置/角度、相機 FOV/解析度、工作區佈局。
   - **不用對齊(交給 DR)**:背景、桌面紋理、光照、顏色。
2. **GR00T 吃什麼?** → 相機 RGB + 關節狀態,不需要物體的 3D 世界座標。
   只要「視覺夠像 + 動作空間一致」就能遷移。

---

## 階段總覽

```
Phase 1  單臂 USD(換上你的支架+相機)   ← 起點
Phase 2  雙臂 Isaac Lab 環境
Phase 3  三相機 + RGB/Depth
Phase 4  雙臂遙操作 + 收資料           ← 最大工作量
Phase 5  Domain Randomization
Phase 6  GR00T 訓練(純 sim)
Phase 7  真機對齊 + 收真實資料
Phase 8  Co-training + 部署
```

---

## Phase 1 — 單臂 USD(換上你的支架 + 相機)

先把「一隻手臂 + 你的支架 + RealSense」做對,雙臂只是複製兩份。
目標結構：

```
SO-ARM101-wristcam.usd          ← 最終手臂 USD
└── /gripper
    └── cam_mount  (Xform)
        ├── (ref) Wrist_cam_mount_RealSense_D435.usd   ← 支架外觀
        └── (ref) d435i_real2sim.usd                   ← 相機(含精確 optical frame)
                  └── override: 移除 PhysicsArticulationRootAPI
```

**Phase 1A — 做出 `camera_module.usd`(支架 + 相機)**

1. Isaac Sim `File > New` 開空 stage
2. 建 Xform,命名 `CameraModule`(右鍵 `Create > Xform`)
3. 選中 `CameraModule` → 右鍵 `Add > Reference` → 選 `Wrist_cam_mount_RealSense_D435.usd`
4. 選中 `CameraModule` → 右鍵 `Add > Reference` → 選 `realsense_d435i_portable/usd/d435i_real2sim.usd`
5. 選中 reference 進來的 `D435i` root prim → Property 面板移除 `PhysicsArticulationRootAPI`
   (只寫進 camera_module.usd 的 override，原始檔不動)
6. 調整 `D435i` 的 transform，把相機對準支架的實體安裝位置(螺孔對齊)
7. `File > Save As` → `Robot_usd/camera_module.usd`

✅ **1A 完成標準**：`camera_module.usd` 單獨開起來，支架 + 相機正確疊在一起，無 ArticulationRoot 警告。

**Phase 1B — 把模組掛到手臂 gripper 上**

1. `File > Open` → `SO-ARM101-USD-NO-CAMERA.usd`
2. Stage 面板找到 `gripper` prim
3. 選中 `gripper` → 右鍵 `Create > Xform`，命名 `cam_mount`
4. 選中 `cam_mount` → 右鍵 `Add > Reference` → 選 `camera_module.usd`
5. 調整 `cam_mount` 的 transform，讓支架貼合 gripper 的實體安裝點
6. `File > Save As` → `Robot_usd/SO-ARM101-wristcam.usd`

✅ **1B 完成標準**：轉動 Wrist 關節，支架 + 相機跟著 gripper 一起動；無嵌套 ArticulationRoot 警告。

**Phase 1 完成後 Phase 3 的 `prim_path` 就直接指向 optical frame：**
```python
camera_wrist_right.prim_path = "{ENV_REGEX_NS}/Robot_Right/gripper/cam_mount/D435i/camera_color_optical_frame"
camera_wrist_right.spawn = None  # 用已存在的 prim，不另建
```
不需要手動量測光學中心位置，optical frame 的 offset 是 Intel 官方規格值。

---

## Phase 2 — 雙臂 Isaac Lab 環境

改 [so101.py](source/sim_to_real_so101/assets/so101.py) 與 [so101_env_cfg.py](source/sim_to_real_so101/tasks/so101_env_cfg.py)。

- [ ] **資產**:新增 `SO101_DUAL_LEFT_CFG` / `_RIGHT_CFG`(複製 `SO101_REALSENSE_CFG`,
      改 usd_path,base 位置左 `(-0.05, +0.15, 0)` / 右 `(-0.05, -0.15, 0)`)
- [ ] **Scene**:`robot` → `robot_left` + `robot_right`,各自一個 `ee_frame`
- [ ] **Action**:兩組 `JointPositionActionCfg`,各 6 joints → 共 12 維
- [ ] **Observation**:左右兩份關節狀態
- [ ] **註冊**:[tasks/__init__.py](source/sim_to_real_so101/tasks/__init__.py) 加 `Lerobot-So101-Dual-*`
- [ ] **臂間碰撞**:確認兩臂交接時不穿模、也不會讓 solver 爆掉

✅ **完成標準**:`zero_agent --task Lerobot-So101-Dual-Base` 能開,兩臂不互撞爆掉。

---

## Phase 3 — 三相機 + RGB/Depth

相機是**自動發現**的:[lerobot_agent.py:122](source/sim_to_real_so101/scripts/lerobot_agent.py#L122) 掃描所有 `camera_` 開頭物件,
命名對了 recorder 就自動收。參考 [task_env_cfg.py:105-114](source/sim_to_real_so101/tasks/task_env_cfg.py#L105-L114)。

- [ ] 加三個相機:`camera_wrist_left`、`camera_wrist_right`(各掛在對應 gripper)、`camera_ego`(置中)
- [ ] 腕部相機的 `prim_path` 指向 Phase 1 組好的 `optical_frame`，`spawn=None`
      ```python
      camera_wrist_left.prim_path  = ".../Robot_Left/gripper/cam_mount/D435i/camera_color_optical_frame"
      camera_wrist_right.prim_path = ".../Robot_Right/gripper/cam_mount/D435i/camera_color_optical_frame"
      camera_wrist_left.spawn  = None  # 用 USD 裡已有的 prim
      camera_wrist_right.spawn = None
      ```
- [ ] **對齊 intrinsics**:`PinholeCameraCfg` 的 focal_length/FOV 對到 RealSense D435,解析度 640×480
      (現值 13.5mm 是「真實 1/10」的縮放,改前先看 [task_env_cfg.py:56-61](source/sim_to_real_so101/tasks/task_env_cfg.py#L56-L61) 註解)
- [ ] **觀測群組**:[VisualCfg](source/sim_to_real_so101/tasks/task_env_cfg.py#L166-L228) 改成三相機 ×(rgb+depth)
- [ ] 收集時加 `--depth`;depth 只存檔,訓練時不餵進 policy

✅ **完成標準**:`random_agent` 能在 viewport/rerun 看到三路 RGB + 三路 depth。

---

## Phase 4 — 雙臂遙操作 + 收資料(最大工作量)

現有 [lerobot_interface.py](source/sim_to_real_so101/utils/lerobot_interface.py) 只支援單 leader,要自訂擴充。

**handoff 任務邏輯**
- [ ] **不對稱 reset**:改 `reset_vials_rack`,vials 生右側、rack 放左側
      ([vials_to_rack_env_cfg.py:140-159](source/sim_to_real_so101/tasks/vials_to_rack_env_cfg.py#L140-L159))
- [ ] **多階段成功判定**:擴 [mdp/terms.py](source/sim_to_real_so101/mdp/terms.py) 成狀態機
      `右臂抓起 → 交接(右放、左抓同一 vial)→ 左臂放入 rack`,需區分左右臂 contact sensor

**遙操作**
- [ ] 兩支 leader 接兩個 USB,各自 `lerobot-calibrate`(右=取料、左=放置)
- [ ] 改 [lerobot_agent.py](source/sim_to_real_so101/scripts/lerobot_agent.py):兩個 `LeRobotSO101Interface`,
      兩個 6 維動作 concat 成 12 維
- [ ] dataset 寫入:12 維 state/action + 3 相機(擴 `sim_to_real_dataset_processor`)
- [ ] embodiment tag 用雙臂 SO-101([gr00t_client/embodiment_tags.py](source/sim_to_real_so101/gr00t_client/embodiment_tags.py))

**收集**
- [ ] `lerobot_agent --task ... --repo_id ... --repo_root ./datasets --task_name ... --depth`
- [ ] 先收 50–100 條測 pipeline,再放大

✅ **完成標準**:雙手 leader 同時驅動 sim 雙臂,按 `S` 錄製,dataset 正確生成(12 維 + 3 相機)。

---

## Phase 5 — Domain Randomization

直接複用 [vials_to_rack_env_cfg.py](source/sim_to_real_so101/tasks/vials_to_rack_env_cfg.py) 的 DR event。

- [ ] 光照(`randomize_light_exposure` / `randomize_sky_light`,多張 HDRI)
- [ ] 機器人顏色(左右臂都要)
- [ ] 三相機 pose + focal length 抖動
- [ ] 桌面紋理 + 旋轉
- [ ] 註冊 `Lerobot-So101-Dual-*-DR`

✅ **完成標準**:每次 reset 外觀明顯不同,但任務幾何不變。

---

## Phase 6 — GR00T 訓練(純 sim)

- [ ] dataset push 到 HF(`lerobot_push_dataset`)
- [ ] Isaac-GR00T finetune(雙臂 embodiment)→ checkpoint
- [ ] sim 內評估:`lerobot_eval --task Lerobot-So101-Dual-*-Eval`

✅ **完成標準**:純 sim policy 在 sim eval 有合理成功率。

---

## Phase 7 — 真機對齊 + 收真實資料

對齊 geometry,外觀差異交給 DR。

- [ ] 真實雙臂 + 雙 leader,`lerobot-calibrate` 全關節 → [so101_check_calibration.py](docker/real/scripts/so101_check_calibration.py) 驗證
- [ ] **實體安裝**支架 + RealSense,extrinsics 盡量貼近 sim 的 offset
- [ ] 真機工作墊做 18"×12"、雙臂間距 30cm
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
- **多階段成功判定**是自訂狀態機,比單臂複雜 → Phase 4 主要工作量
- **收 demo 難度高**:雙手同步交接要練習,初期廢片多 → 預留時間
- **雙 leader teleop** 是自訂程式,非改設定
- **多台 RealSense** 同接 USB 可能撞頻寬 → 分 USB controller 或降 fps
- **sim depth 太乾淨**,真機 depth 有噪 → 考慮對 sim depth 加噪
- **12 維雙臂 embodiment** 需確認 Isaac-GR00T 是否原生支援

---

## 進度日誌
- 2026-06-23 — 建立 roadmap,釐清 USD 結構
- 2026-06-23 — 定案:任務 handoff(右取→中交→左放)、間距 0.30m(真機驗證)、ego 置中、depth 只存檔
