# 雙臂 SO-101 Sim-to-Real Roadmap

**目標**:把單臂模擬環境改成雙臂,換上自製支架 + RealSense D435i ×3,雙臂遙操作收資料 →
訓練 GR00T → Domain Randomization + 真實資料 co-training → sim-to-real。

**現在位置**:Phase 8。sim eval(成功率 50%,作為基準線)、真機 50 集資料、real + sim
co-training checkpoint 都已完成,**唯一還沒打通的是真機雙臂 real eval**。

最後更新:2026-07-27

---

## 已定案的設定

| 項目 | 決定 |
| --- | --- |
| **任務** | **協作式雙臂 vials-to-rack**:試管散在墊子上,**任一隻手臂都可拿任一支**試管,目標是把**所有試管放進中央共用試管架**(不綁定左右分邊) |
| **試管數量** | 場景固定 4 支;reset 時 `hide_random_vials` 隨機藏掉幾支 → 等效**每集 1~4 支**。擺放維持穩定 2 左 / 2 右 |
| **試管架** | **1 個中央共用架**(body 中心置中於墊子中心 world (0.22, 0)),兩臂都搆得到 |
| **佈局** | 兩臂並排、皆面向 +x;左臂 base (-0.05, +0.15, 0)、右臂 (-0.05, -0.15, 0) |
| **間距** | 0.30 m(左臂 y=+0.15、右臂 y=−0.15)✅ 真機已驗證 |
| **相機** | 3 台:左腕、右腕、ego(燈箱開口置中俯視),**只收 RGB** |
| **ego 相機(已量測確認)** | world 位置 ≈ **(-0.23, 0.03, 0.53)**(離地高 ~0.53m),朝墊子中心 (0.22,0) 俯視約 45°;intrinsics focal 15.245mm、aperture 20.955×15.716 → **HFOV≈69°**(= D435 color)。ego_cam 相對 LightBox 的 local:translate (0, 0.4, 0.5)、euler (45°,0,-90°) |
| **工作墊** | mat.usda,world x[0.068, 0.372](前後 0.3048m)× y[-0.229, 0.229](左右 0.4572m),中心 (0.22, 0) |
| **語言指令** | embodiment-agnostic 的一句:`"Pick up the vials and place them into the rack"`。不寫「左手做什麼右手做什麼」,雙臂差異靠 embodiment tag + 12 維 state 表達 |
| **dataset schema** | 12 維 state/action(`left_*` 6 個 + `right_*` 6 個)+ 3 相機(`wrist_left`/`wrist_right`/`center`,480×640)+ fps=30 |

---

## 兩個核心觀念

1. **真實場景要和 sim 一樣嗎?** → 幾何要、外觀不要。
   - **要對齊**:機器人尺寸、相機安裝位置/角度、相機 FOV/解析度、工作區佈局。
   - **不用對齊(交給 DR)**:背景、桌面紋理、光照、顏色。
2. **GR00T 吃什麼?** → 相機 RGB + 關節狀態,不需要物體的 3D 世界座標。
   GR00T 是「看畫面反應」的策略,**沒有內建物件計數器** → 想讓它對「試管數量」魯棒,
   就要在收資料時涵蓋不同數量(這就是 Phase 4B 存在的理由)。

---

## 進度總覽

```
Phase 1   單臂 USD(D435i 支架 + 相機 + physics)      ✅
Phase 2   雙臂 Isaac Lab 環境                          ✅
Phase 3   三相機                                       ✅
Phase 4A  雙臂遙操作 + 收資料                          ✅  74 集,已上 HF Hub
Phase 4B  數量隨機(1~4 支)                            ✅
Phase 5   Domain Randomization                         ✅  桌面紋理暫由 HDRI 補
Phase 6   GR00T 訓練(純 sim)+ sim eval               ✅  成功率 50%(基準線)
Phase 7   真機對齊 + 收真實資料                        ✅  50 集
Phase 8   Co-training + 真機 eval                      ← 現在
Phase 9   進階 sim-to-real(Cosmos / SAGE+GapONet)     尚未探索

技術債    真機端遷回 0.4.x → 收斂成單一 LeRobot repo    進行中  分支 n1.7-graphen
          (升級 Isaac Sim 6.0 降級為非阻塞的獨立任務)   未開始
```

---

## Phase 1–7 已完成 — 現況速查

### 場景與環境

- **資產**:`SO-ARM101-USD-d435i-physics.usd`(手臂,相機/支架含 physics——實體相機重量會影響
  任務動力學)、`RSD435i.usd`、`Wrist_cam_mount_D435_clean.usd`、`lightbox-egocam.usd`
  (中央 ego D435i 俯視)
- **三層 config 繼承**:`so101_dual_env_cfg.py`(雙臂 articulation + 12 維 action,左 0:6 / 右 6:12)
  → `so101_dual_task_env_cfg.py`(三相機 + DR event/observation 群組)
  → `so101_dual_vials_env_cfg.py`(試管、共用架、contact sensor、DR / eval 變體)
- **註冊 env**:`Lerobot-So101-Dual-Base`、`-Vials-To-Rack`、`-Vials-To-Rack-DR`、`-Eval`、`-DR-Eval`
- 三相機 `camera_wrist_left/right` + `camera_center` 皆 `spawn=None`,指向 USD 內已烤好的 Camera prim。
  相機是**自動發現**的:script 掃 `camera_` 開頭物件,命名對了 recorder 就自動收
- 單臂鏈完全沒動、照樣可用;目前所有實驗聚焦雙臂,單臂流程僅供參考架構

### 任務邏輯(mdp/)

- `hide_random_vials`(resets.py):reset 時隨機把幾支藏到相機看不到的地方 → 等效每集 1~4 支
- per-sensor 狀態重構(terms.py):`any_vial_grasped` / `vial_placed_on_rack` 用
  `func._state[sensor_name]`,左右臂不互相干擾;subtask 觀測四個都吃全部 4 支(順序對齊 filter)
- `all_active_vials_placed_termination`:dual-safe 成功判定——跨左右兩個 contact sensor,
  成功 = 所有「啟用中(沒被藏掉)」試管同時在架上、連續 25 步。天然支援隨機 1~4 支

### Domain Randomization(Phase 5)

- 光照:燈箱曝光 + `randomize_sky_light`(換 HDRI + 曝光 + 色溫,25 張 .exr)
- 外觀:左右臂各自獨立隨機顏色、試管架 8 色、試管透明度
- 相機抖動分工:**兩腕相機抖焦距** + **中央 ego 抖位姿**。ego「不」抖焦距
  (HFOV≈69° 已量測對齊真機,抖焦距會破壞 sim-real 對齊)
- 桌面旋轉加大(base ±0.1 → DR ±0.3)
- 〰️ 桌面**紋理**未做(需 texture-swap 函式 + 多紋理 mat 資產),外觀多樣性暫由 HDRI 天空光補

### 資料與模型

- **sim**:74 集乾淨資料 → `ChihHanShen/bimanual-so101-pickvials`(HF Hub)
- **真機**:50 集 demo;雙臂 + 雙 leader + 3 台 D435i 已安裝校正,收資料 pipeline 跑通
- **checkpoint**:純 sim 版、real + sim co-training 版都已訓好
- 遙操作用兩支 leader(env 變數 `TELEOP_PORT_LEFT/RIGHT` + `TELEOP_ID_LEFT/RIGHT`,每次插拔要重設);
  鍵盤 S 起停錄製、C 取消當段、R reset
- 錄製 / 清理工作流見 [run_cheatsheet.md](run_cheatsheet.md);刪壞集用
  `tools/delete_episodes.py <檔號>`(用檔號、非 episode_index)

### sim eval 基準線(Phase 6)

兩段式:起 `run_gr00t_server.py` 吃 checkpoint(ZMQ port 5555)→ 跑 `lerobot_eval_dual` 當 client
做 rollout 算成功率。指令見 [run_cheatsheet.md](run_cheatsheet.md) 的「Eval」段。

- `GR00TDualRemotePolicy`(lerobot_interface.py):左右各一 `LeRobotSO101Interface`,state/action
  分成 `left_arm`(0:5)/`left_gripper`(5:6)/`right_arm`(6:11)/`right_gripper`(11:12)
- **實跑結果(純 sim checkpoint)**:乾淨場景 `-Eval` 成功率 **50%**、DR 場景 `-DR-Eval`
  成功率**同樣 50%** → policy 沒有被 Phase 5 的外觀 DR 拖垮。**這是真機 eval 的對照基準線**
- ⚠️ server 的 `--modality-config-path` / `--embodiment-tag new_embodiment` 必須對上訓練設定,
  否則維度或語意對不上

---

## Phase 8 — Co-training + 真機 eval ← 現在

- [x] sim(DR)+ real 混合 co-training finetune
- [ ] **雙臂真機 eval**(見下)——**目前唯一還沒打通的環節**
- [ ] 量真機成功率,迭代調 DR / 補資料 / 修相機對齊

✅ **完成標準**:真機雙臂任務成功率達標,sim-to-real gap 收斂。

### 做法:Isaac-GR00T 官方 server/client

以 `Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so100.py` 為模板改寫雙臂 client
`eval_so101_dual.py`,server 沿用 sim eval 那顆 `run_gr00t_server.py`,robot 端用 lerobot fork
(`maxshen1212/lerobot@graphen`)原生的 `bi_so_follower`。

那支官方 script 本來就是給人改寫的**參考實作**——docstring 自己寫「SO100 **/ SO101**」,
`examples/SO100/README.md` 的 Closed-Loop 段也註明它是「how to write deployment code using
Policy API」。官方 [12-real-evaluation] 用的 `so101_eval.py`(workshop docker 版,多一個
`--rerun`)就是同一支的變體。

選它的理由:**推論路徑與訓練同源**——normalization、relative-action 編碼、影像 crop/resize 全部
由 checkpoint 自己的 `processor_config.json` 決定,不需要人工對齊。現階段要量的是 sim-to-real
gap,推論路徑上不能再多一個未驗證的變因;而且 sim / real 共用同一個 server,兩邊成功率才可比。

### ⚠️ `use_relative_action` = **true**

checkpoint 裡有兩份值不同的 config,**推論時生效的是 `processor_config.json`(= `true`)**,
不是 `experiment_cfg/final_model_config.json`(= `false`)——`Gr00tPolicy` 走
`AutoProcessor.from_pretrained(processor_dir)`(`Isaac-GR00T/gr00t/policy/gr00t_policy.py:124`)。

搭配 `examples/SO101_bimanual/so101_bimanual_config.py` 的 per-key 設定,模型實際輸出是:
`left_arm`/`right_arm` = **相對當前 state 的 delta**,兩個 gripper = 絕對值。

走 Isaac-GR00T 路線這是自動從 checkpoint 讀的,**不用設也設不錯**;但任何把這些 delta 當成
絕對關節角送進真機的做法(例如在別的 runtime 上手動把這個 flag 設成 false)**都會讓手臂暴衝**。

### ⚠️ 控制頻率是 10 Hz,不是 30 Hz

**官方的一致做法是「control rate = 訓練資料集的採集 fps」**,不是相機 fps、也不做內插:
DROID client 寫死 `DROID_CONTROL_FREQUENCY = 15`(15 fps 資料),註解就是 "Sleep to match DROID
data collection frequency";SO100 client sleep 到 1/30 因為那個資料集是 30fps 錄的。

本專案的 co-train checkpoint 吃的是 `bimanual-so101-pickvials-{real,sim}-10fps`
(`Isaac-GR00T/CHEATSHEET.md:12-13`),**已實查 `meta/info.json` → `fps: 10`**
(22,157 frames / 49 episodes = 452 frames/集)。`CHEATSHEET.md:241` 也明寫換 10fps 就是為了把
開環 chunk 數從 ~140 降到 21/28。

→ **一個 action = 0.1 秒,16 步的 chunk 覆蓋 1.6 秒。** 錄製 config 裡的 30 fps 是**擷取**頻率,
資料進訓練前已降採樣。用 30 Hz 跑會讓手臂快 3 倍。

順帶一提:官方文件擔心的 stop-and-go(latency 要 < 33ms 才跟得上 30 Hz)**在我們這裡不存在**——
3B 模型 0.2-0.3 秒的 round-trip 遠小於 1.6 秒的重規劃窗口。

### 實作:`eval_so101_dual.py` ✅ 已完成

位置:`Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual/`(**不動上游的 `SO100/`**——它 pin 的
lerobot commit 早於 `so_follower`/`bi_so_follower` 重構,無法沿用)。指令見 CHEATSHEET 4c。

**刻意貼著官方 `eval_so100.py` 的結構寫**(235 行,官方 291 行):同樣是 connect → `while True`,
只差兩件事——12 維 + 3 相機的 adapter、10 Hz 控制頻率。其餘一律不加。

- **12 維 state 分組**:`left_arm`(0:5) / `left_gripper`(5:6) / `right_arm`(6:11) /
  `right_gripper`(11:12);`bi_so_follower.get_observation()` 吐的 `left_*.pos`/`right_*.pos`
  天然對上,`decode_action_chunk` 反向 concat 回 12 個 `.pos`。
- **`np.float32` 是關鍵**:lerobot 回 Python float,裸 `np.array([...])` 會給 float64,
  server 直接拒絕(`gr00t_policy.py:320`)。
- **三台相機全放 top-level**,任何一台掉進 `right_arm_config.cameras` 會變成 `right_wrist_right`。
- **跑法就是官方的 `while True`**:一次執行 = 一集,成功或卡住就 Ctrl-C、人工 reset、再跑一次,
  成功率自己記。因此**不需要** episode 迴圈、outcome 提問、鍵盤監聽、每集回起始姿態。
- 加的只有 `max_relative_target` 軟限位(純 config 傳遞)。
- 一度寫成 1100 行(smoke/dryrun/capture-home 模式、mp4/jsonl 記錄、契約與觀測驗證、
  連線重試、延遲統計、episode 迴圈與回位),已砍回 **235 行**——比官方的 291 行還短。

### 真機成功率協定

- **頭條數字用二元判定**:90 秒內 4 支全部坐進架子才算成功(對齊 sim 的 all-or-nothing),
  每集**額外記 `vials_placed` (0-4)** 當低雜訊的第二指標。
- `f` = 判斷已無法挽回時提前中止(相對 sim truncation 的刻意偏離,要寫進報告);
  `r` = 只用於操作者/硬體失誤,不進分母。
- **N=20**。N=10 時 50% 的 95% CI 約 ±31 個百分點,分不出 30% 和 70%。

### ⚠️ 跟 sim 的 50% 對照時,三個不對等必須註明

1. **時間基準**:sim `decimation=2` / `sim.dt=1/120` → 60 Hz,16 步 chunk 在 0.27 秒內跑完,
   比資料代表的 1.6 秒快 6 倍。那是 physics-time vs data-time 的產物,不會也不該轉移。
2. **成功偵測**:sim 是 `confirm_steps=25` 的自動判定,真機是人判。
3. **時間預算**:sim `EVAL_EPISODE_LENGTH_S = 22.5s` sim 時間,真機是 90 秒(900 步 @ 10 Hz)。

50% 當參考點,不是嚴格 baseline。

### 兩個環境(不可混用)

| 角色 | 環境 | 理由 |
| --- | --- | --- |
| **Server** | Isaac-GR00T 自己的 `.venv`(`uv sync`) | pin `torch==2.9.0` / `transformers==4.57.3` |
| **Client** | **沿用 lerobot 的 `.venv`** + `msgpack`/`msgpack-numpy` + `uv pip install --no-deps -e <Isaac-GR00T>` | 這個 venv 本來就在驅動這套硬體(資料就是它錄的) |

⚠️ **server 不能借用 lerobot venv 跑**:lerobot 是 `transformers 5.5.4` / `torch 2.11.0`,
跟 Isaac-GR00T 差一個主版本。借用等於讓影像前處理/tokenizer 跟訓練時不同,
「推論路徑與訓練同源」這個選型理由就沒了。

> 本機首次 `uv sync` 會卡在 `scripts/deployment/dgpu/wheels/torchcodec-...aarch64.whl` ——
> 那是沒 smudge 的 Git LFS pointer(雖然本機是 x86_64,uv 仍會讀它的 metadata)。
> 解法:`git lfs install --local && git lfs pull --include="scripts/deployment/dgpu/wheels/*.whl"`。

### 部署方式(two-terminal,與 sim eval 同構)

```bash
# 終端機 A(~/Isaac-GR00T)——與 sim eval 完全相同的 server
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-cotrain/pickvials-n1p7-run3/checkpoint-25000 \
    --embodiment-tag NEW_EMBODIMENT --port 5555
# 註:有 --model-path 時 --modality-config-path 會被靜默忽略(那段在 dataset_path 分支裡),
#     modality config 來自 checkpoint 自己的 processor。帶了無害,但不是必要。

# 終端機 B(Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual/)——一次執行 = 一集
/home/graphen/sim2real/lerobot/.venv/bin/python eval_so101_dual.py
```

硬體參數(port / 校正目錄 / 三台相機序號 / task 字串)都已是 script 的預設值,平常不用帶。
換 checkpoint 只需重啟 server,client 完全不動。

### 待辦 / 驗證順序

- [x] 確認 co-training checkpoint 的 `processor_config.json` → **`use_relative_action: true`**,
      action delta 16、四組 key 與三台相機 key 全部吻合、reps = RELATIVE/ABSOLUTE/RELATIVE/ABSOLUTE
- [x] 確認訓練 fps → **`meta/info.json` fps = 10**(22,157 frames / 49 集 = 452 frames/集)
- [x] 寫 `eval_so101_dual.py`;client 環境就緒(lerobot venv + msgpack + gr00t no-deps)
- [x] **server 通聯驗證通過**(2026-07-28):契約由 server 實際回報並吻合
      (`action_horizon=16`、`video_delta=(0,)`、`state_delta=(0,)`)、chunk shape/dtype 全對。
      **穩態推論延遲 0.070 s vs 1.60 s 重規劃窗口 = 23 倍餘裕**(warmup 0.594 s)
- [x] 12 顆馬達實測全部正常回應(11.8-11.9 V);相機 obs key 確認為不帶前綴的
      `center`/`wrist_left`/`wrist_right`
- [ ] **首次帶電**:1 集、`max_relative_target` 全設 0.5、手放電源開關
- [ ] **全速單集**,跟 `lerobot-replay` 的真人 demo 並排比對速度(驗 10 Hz 是否正確)
- [ ] **20 集正式跑**,再換純 sim checkpoint 重跑一輪做對照

[12-real-evaluation]: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/12-real-evaluation.html

---

## Phase 9 — 進階 sim-to-real 策略(尚未探索)

官方教學後半段的 Strategy 3 / 4,**還沒碰過**,先記錄路徑。兩者官方文件都只寫**單臂**流程,
套到雙臂前要先想清楚怎麼擴。

**Cosmos 資料增強**(官方 [14-strategy3-cosmos](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html))——
用 world foundation model 把既有 demo 做 video-to-video,生成「光照/背景/物件位置不同、任務結構
不變」的變體,跟 DR 互補而非取代。**雙臂待解**:官方是單相機單臂,3 相機要嘛三路各跑一次、
要嘛先確認 Cosmos 支不支援多視角一致性,否則生成的變體會對不上。

**SAGE + GapONet**(官方 [15-strategy4-sage](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html))——
處理 DR / co-training / Cosmos 都沒碰到的**致動落差**(伺服 backlash、摩擦與接觸動力學建模不準、
URDF 轉換誤差),即「同一個關節指令,sim 和真機實際動到的位置不一樣」。
[sage](https://github.com/isaac-sim2real/sage) 跑同一組 motion file 在 sim / 真機各收一份 paired
資料算出每個關節的 gap;[gaponet](https://github.com/jiemingcui/gaponet) 拿那份資料訓「指令 → 實際
動作」的網路,推論時反過來補償。兩種套用模式:① 補進 sim 訓練、② 只在部署時對送出動作做補償。
官方範例用的是 Isaac Lab `Isaac-Humanoid-Operator-Delta-Action`,不是 SO-101 原生任務,要另外
確認怎麼接到雙臂 SO-101 config。

**定位**:等 Phase 8 真機 eval 跑起來、看到具體的 gap 表現後,再決定投入 Cosmos(補視覺多樣性)
或 SAGE+GapONet(補動力學落差)。

---

## 技術債 — 往 0.4.x 收斂成單一 LeRobot repo

**現況**:同一個 fork 拆成兩個 worktree,兩邊都要維護。

| Checkout | Branch | LeRobot | requires-python | 用途 |
| --- | --- | --- | --- | --- |
| `~/sim2real/lerobot` | `graphen` | 0.6.1 | >=3.12 | 真機 |
| `~/lerobot-pinned` | `sim-pinned` @ `e670ac5d` | 0.4.3 | >=3.10 | **workshop 實際 import 的** |

**代價**:任何工具/修正都要做兩次,sim 端也拿不到新版 LeRobot 的東西。

**方向已改為往下收斂(不再等 Isaac Sim 6.0)。** 原本的計畫是升 Isaac Sim 6.0
把 sim 端拉上 0.6.1;現在改成把真機端拉回 0.4.3,理由:

1. **NVIDIA 的 GR00T 參考 pipeline 本來就是 0.4.x。**
   `Isaac-GR00T/gr00t/eval/real_robot/SO100/pyproject.toml:17` pin 的是
   `lerobot@c75455a6`(v0.4.1,2025-10-23)。我們的 `e670ac5d` 是 v0.4.3
   (2026-01-05)——**同一條 0.4.x 線,而且更新**。
2. **0.4.x 預設就是 `use_degrees=False`(±100)**,與 NVIDIA 的慣例一致;
   0.6.1 把預設翻成 `True`(度),才造成 sim/真機的資料慣例分歧(見風險預警)。
3. **不需要動 Isaac Sim。** 0.4.x 的 `requires-python` 是 `>=3.10`,
   現有的 `~/env_isaaclab`(Python 3.11)直接可用。
4. sim 端**完全不用改**——它已經在跑 0.4.3。

**目標**:真機端遷到 0.4.3,退掉 `~/lerobot-pinned` worktree,兩邊共用一個 checkout。

**已建立分支**:`n1.7-graphen`(基底 `e670ac5d`,位於 `~/sim2real/lerobot`)。

**遷移清單與逐項討論**:見 [MIGRATION.md](MIGRATION.md)。

**Isaac Sim 6.0 升級**仍然值得做(新功能、效能),但**不再是收斂成單一 repo 的前提**,
降級為獨立的、非阻塞的任務。升級時仍要注意:USD/物理行為跨大版本可能有差異,
Phase 6 的 sim eval 基準線(50%)要重跑確認,不能直接沿用。

---

## 風險預警

- **⚠️ sim 與真機資料集用了不同的正規化慣例(Phase 8 co-training 直接相關)**

  | | 用的 LeRobot | `use_degrees` 預設 | 5 個身體關節 | gripper |
  | --- | --- | --- | --- | --- |
  | sim | `~/lerobot-pinned` v0.4.3 | **`False`** | `RANGE_M100_100`(±100,會 clamp) | `RANGE_0_100` |
  | 真機 | `~/sim2real/lerobot` v0.6.1 | **`True`** | `DEGREES`(度,**不 clamp**) | `RANGE_0_100` |

  workshop 建 `SO101LeaderConfig(port=..., id=...)` 時沒傳 `use_degrees`,
  `utils/lerobot_interface.py` 也寫死 `(raw+100)/200` 的假設 → sim 端確定是 ±100。
  真機資料集 `bimanual-so101-pickvials-real-15fps` 的 stats 實測 `wrist_roll`
  跑到 **-163.9**,超出 ±100 → 確定是 DEGREES(RANGE_M100_100 會被 clamp 在 ±100)。

  兩者的換算是 `度 = v × span × 360 / (2×4095×100)`,**比例隨每個關節的 span 不同**:

  ```
  shoulder_pan   span 2788  x1.225      elbow_flex   span 2237  x0.983
  shoulder_lift  span 2436  x1.071      wrist_flex   span 2380  x1.046
  wrist_roll     span 3906  x1.717  ← 差最多      gripper 兩邊相同 ✅
  ```

  也就是同一個物理姿態,sim 記 `+80`、真機記 `+137`(wrist_roll)。
  gripper 兩邊都是 `RANGE_0_100`,沒問題。

  **單獨 fine-tune 沒問題,co-training 才有問題。** 已查證 GR00T 的行為:

  - `StateActionProcessor` 對 state/action 做 **min-max 正規化到 [-1,1]**,
    統計量(`min`/`max`,或 `use_percentiles=True` 時的 `q01`/`q99`)
    **以 `embodiment_tag` 為 key**,由你自己的 dataset 算出來
    (`gr00t/data/stats.py`、`gr00t/data/state_action/state_action_processor.py`)。
  - ±100 與「度」之間是**純線性(affine)換算**,min-max 正規化會**完全抵消**它。
  - 唯一對單位敏感的是 `sin_cos_embedding_keys`(官方 `data_config.md`:
    「Best for dimensions that are in radians... If not specified, min-max
    normalization is used」)。`examples/SO101_bimanual/so101_bimanual_config.py`
    **沒有**啟用,`apply_sincos_state_encoding` 預設也是 `False`。
  - 已驗證真機資料**全部落在校準範圍內**(最接近的是 elbow_flex 96.6 vs 上限 98.3),
    所以換成 ±100 編碼也不會被 clamp → 沒有資訊損失。

  → **所以用 ±100 資料 fine-tune 本身不會有問題。**
  真正的鐵律是**部署要跟收資料時同一套單位** —— `eval_so101_dual.py:180-183`
  已經明寫 `use_degrees=True`(註解:「to match how the dataset was recorded」),
  真機這條迴路是自洽的。

  順帶:GR00T 全套程式碼與文件**從沒要求過 degrees**。唯一提到單位的地方都指向
  **radians**(`policy.md:174`「joint positions in radians」;`data_config.md:109`
  sin/cos 編碼「Best for dimensions that are in radians」),而唯一啟用
  `sin_cos_embedding_keys` 的 embodiment 是 `robocasa_gr1_tabletop`。
  LIBERO 的 action 更是全部 `Box(low=-1, high=1)` 的 delta EEF,連物理單位都不是。

  **真正的風險**:sim 與真機兩份 dataset 若掛在**同一個 `embodiment_tag`**
  (目前兩邊都是 `NEW_EMBODIMENT`),它們會**共用同一組統計量**,
  affine 差就不再抵消 —— 同一個物理動作在 wrist_roll 上會差 1.717 倍,
  在 [-1,1] 空間裡落到不同區段。GR00T 原始碼對這種情況只會印一行
  warning(`Statistics for embodiment ... already present; new stats DISCARDED`),
  不會報錯。

  **已查證:GR00T 的 mixture dataset 支援一次訓練掛多個 tag。**
  `ShardedMixtureDataset._merge_statistics()` 先用 `embodiment_tag` 分組
  (`all_stats_by_emb`),**只在同組內**合併統計量。合併方式是取包絡:
  `min=np.min(min_list)`、`max=np.max(max_list)`、`q01=np.min(q01_list)`、
  `q99=np.max(q99_list)` —— 所以同 tag 混單位時,數值範圍大的那份(度)會決定包絡,
  ±100 那份會被壓縮到中間一小段。不同 tag 則各自獨立,不受影響。

  **co-training 前三選一**:① 兩份資料用不同的 `embodiment_tag`(不用重收);
  ② sim 端改 `use_degrees=True`;③ 真機端改 `use_degrees: false`。
  **若兩邊資料都要重收,建議 ②**(理由見下)。

- **若採用 ②(統一成 degrees),要注意單位對齊 ≠ 物理對齊**

  兩種慣例的**物理錨點不同**:sim 的 ±100 錨在 `SO101_USD_MAPPING` 的 USD joint
  limit,真機的度錨在**校準硬限位的中點**。把 sim 改成度之後,
  `utils/lerobot_interface.py` 的 `(raw+100)/200 → [joint_min, joint_max]` 映射
  要重新設計成「度 → USD 關節角」,而這只有在**校準中點恰好等於 USD 零位**時才成立。
  這是 Phase 7 的真機對齊工作,不是單純換單位 —— 沒做的話只是把一種錯配換成另一種。

  好消息:要改的兩處(`lerobot_interface.py:114` 的 `SO101LeaderConfig`、
  以及 `:140-167` 的兩個映射函式)**都在 workshop repo**,不在 `~/lerobot-pinned`;
  且 `use_degrees` 在 pinned v0.4.3 本來就是支援的欄位(只是預設 `False`),
  所以不需要動 pinned repo。

- **真機 eval 是自訂工程**:12 維 adapter + episode/成功率統計 + 安全機制都要自己補
- **收 demo 難度高**:雙手同時操作要練習,初期廢片多 → 預留時間
- **雙 leader teleop** 是自訂程式,非改設定;port 每次插拔會變,要重設 env 變數
- **多台 RealSense** 同接 USB 可能撞頻寬 → 分 USB controller 或降 fps
- **藏試管**(Phase 4B)要確定完全不入鏡,否則半截入鏡會污染訓練影像
- **試管 orientation**:`_UPRIGHT` 命名其實是橫躺(既有 bug/誤名),reset yaw 保持 0,
  避免立起的試管被大 yaw delta 弄倒

---

## 進度日誌

- 2026-06-23 — 建立 roadmap,釐清 USD 結構;定案間距 0.30m(真機驗證)、ego 相機置中
- 2026-06-29 — **任務定為協作式**:任一臂拿任一支 → 中央共用架;P1~P3 完成
- 2026-07-01 — P4A 錄製程式建好(12 維 + 3 相機);量測並目視確認 ego 中央相機
  (world (-0.23,0.03,0.53)、俯視墊子中心、HFOV≈69°)
- 2026-07-06 — P4A 協作式任務改造完成(兩 contact sensor + 四個 subtask obs 都吃全部 4 支);
  P5 DR 大致完成
- 2026-07-13 — **P4A 收滿 74 集並上 HF Hub;P4B(`hide_random_vials`)完成;P5 DR 補齊;
  純 sim 雙臂 GR00T 模型訓練完成**;P6 sim eval 基礎設施建好(dual-safe termination、
  eval env、`GR00TDualRemotePolicy`、`lerobot_eval_dual`)
- 2026-07-24 — **P6 真跑一輪**:純 sim checkpoint 在乾淨場景與 DR 場景成功率**都是 50%**,
  P6 完成、當作真機評估基準線。同時確認 **P7 其實已完成**(50 集真機 demo,並已與 sim 資料
  混訓出 real + sim co-training checkpoint);Cosmos 與 SAGE+GapONet 記為 Phase 9
- 2026-07-27 — **真機 eval 做法定案**:Isaac-GR00T 官方 server/client,以 `eval_so100.py` 為模板
  寫雙臂版 `eval_so101_dual.py`,server 沿用 sim eval 那顆(理由:推論路徑與訓練同源、
  sim/real 成功率可比)。查證釐清 **`use_relative_action` 應以 `processor_config.json` 為準
  = `true`**(手臂輸出是 delta、gripper 是絕對值),弄錯會讓真機暴衝;確認相機命名
  `wrist_left`/`center`/`wrist_right` 已對上、不需 rename。同日精簡 roadmap:移除已被取代的
  舊方案敘述,並**全面移除 depth**(專案已不再使用 depth)
- 2026-07-28 — **`eval_so101_dual.py` 寫完**(`Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual/`,
  新目錄,不動上游 `SO100/`)。**235 行,官方 `eval_so100.py` 是 291 行**——同樣是
  connect → `while True`,只差 12 維 + 3 相機 adapter 與 10 Hz 控制頻率兩件事。
  跑法也照官方:一次執行 = 一集,人工判定與 reset 後再跑,成功率自己記。
  (過程中一度膨脹到 1100 行——smoke/dryrun/capture-home 模式、mp4/jsonl 記錄、契約與觀測驗證、
  連線重試、延遲統計、episode 迴圈與每集回位——經指正後全部砍掉。)
  **查出一個會讓手臂快 3 倍的問題:控制頻率是 10 Hz 不是 30 Hz。** 官方 client 的一致做法是
  「control rate = 訓練資料集的採集 fps」(DROID 寫死 15、SO100 用 1/30),而本專案 co-train
  checkpoint 吃的是 10fps 資料集——已實查 `meta/info.json` `fps: 10`。錄製 config 的 30 fps 是
  擷取頻率,進訓練前已降採樣。一個 16 步 chunk 因此覆蓋 1.6 秒,順帶讓官方擔心的 stop-and-go
  完全不成問題。**依此決定馬達寫入不做內插**(照官方 reference code 的語意)。
  前置驗證完成:co-train checkpoint `use_relative_action: true`、modality 契約全部吻合、
  server 通聯實測穩態延遲 0.070 s、12 顆馬達全部正常。
  另確認 **server 不能借用 lerobot venv**(transformers 5.5.4 vs Isaac-GR00T pin 的 4.57.3,
  差一個主版本),client 則沿用 lerobot venv 沒問題。修掉本機 `uv sync` 卡住的
  Git LFS pointer wheel。下一步是首次帶電。
