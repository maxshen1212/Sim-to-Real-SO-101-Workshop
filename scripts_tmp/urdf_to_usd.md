# RealSense URDF → USD 轉換流程

把 RealSense 相機(以 **D435i** 為例)從官方 `realsense-ros` 的 URDF/xacro 匯入成 Isaac Sim 可用的 USD。
換別款相機(D455、D405…)只要把第 2 步的 xacro 目標檔換掉即可。

---

## 概觀

```
realsense-ros (git)
  └─ realsense2_description/urdf/*.xacro
         │  ① clone repo + meshes
         ▼
  ② xacro 展開（use_nominal_extrinsics:=true）──► 扁平 URDF
         │  ③ 修 mesh 路徑 package:// → 絕對路徑
         ▼
     /tmp/d435i.urdf
         │  ④ Isaac Sim GUI: File > Import
         ▼
  intel_realsense_d435i.usd   （匯入器會塞一堆 physics）
         │  ⑤ strip physics（pxr 腳本）
         ▼
  intel_realsense_d435i_clean.usd   （純視覺 + optical frames，可用）
```

---

## 前置需求

- ROS 2（本機是 Humble，`/opt/ros/humble`）— 提供 `xacro`
- Isaac Sim（GUI，用內建的 URDF Importer）
- 離線編輯 USD 用的 `pxr`：見文末「pxr headless 設定」

---

## ① 取得 repo + meshes

```bash
git clone -b ros2-master https://github.com/realsenseai/realsense-ros.git
# 之後用到的路徑：
#   $HOME/realsense-ros/realsense2_description/urdf/   ← xacro 巨集
#   $HOME/realsense-ros/realsense2_description/meshes/ ← d435.dae 等
```

URDF 那條鏈：`test_d435i_camera.urdf.xacro` → `_d435i.urdf.xacro` → `_d435.urdf.xacro`（定義 mesh + optical frames）+ `_d435i_imu_modules.urdf.xacro`。

---

## ② xacro 展開成扁平 URDF

xacro 不能直接匯入，要先展開。**關鍵參數 `use_nominal_extrinsics:=true`** — 沒有它就拿不到 `camera_*_optical_frame`（Intel 原廠規格的相機光學中心）。

直接用腳本（已寫好，含所有踩坑修正）：

```bash
bash scripts_tmp/gen_d435i_urdf.sh
# 產出 /tmp/d435i.urdf（15 links / 14 joints / 含 color_optical_frame）
```

腳本做的事 = 下面 ②③ 兩步，重點在三個踩坑修正：

### 坑 1：`$(find realsense2_description)` 找不到 package
xacro 裡用 `$(find realsense2_description)` include 同目錄的其他 xacro，但你只 clone 沒 build，ament 索引裡沒有它 → `PackageNotFoundError`。
**解法：做一個「假的 ament prefix」指向 clone 的原始碼**（不用 sudo、不用 build）：

```bash
PREFIX=/tmp/rs_prefix
mkdir -p "$PREFIX/share/ament_index/resource_index/packages"
touch "$PREFIX/share/ament_index/resource_index/packages/realsense2_description"
ln -sfn "$HOME/realsense-ros/realsense2_description" "$PREFIX/share/realsense2_description"
export AMENT_PREFIX_PATH="$PREFIX:$AMENT_PREFIX_PATH"
source /opt/ros/humble/setup.bash
xacro "$HOME/realsense-ros/realsense2_description/urdf/test_d435i_camera.urdf.xacro" \
      use_nominal_extrinsics:=true > /tmp/d435i.urdf
```

### 坑 2：空檔陷阱
`xacro ... > /tmp/d435i.urdf` 的 `>` 會**先把檔案清空、才執行 xacro**。若 xacro 失敗就留下一個 0 byte 空檔，拿去 Isaac 匯入會報 `number of links is zero`。
**解法：先寫暫存檔，驗證成功（合法 XML 且 link 數 > 0）才 `mv` 覆蓋目標檔。**（腳本已內建）

### 坑 3：不要用 `set -u`
ROS 的 `source setup.bash` 在 `set -u`（未定義變數即報錯）下會壞掉，導致 xacro 沒進 PATH。腳本用 `set -eo pipefail`，**不加 `-u`**。

---

## ③ 修 mesh 路徑（package:// → 絕對路徑）

URDF 裡 mesh 寫的是 `package://realsense2_description/meshes/d435.dae`，Isaac 匯入器不一定解析得了 `package://`。

```bash
MESHDIR="$HOME/realsense-ros/realsense2_description/meshes"
sed -i "s#package://realsense2_description/meshes/#${MESHDIR}/#g" /tmp/d435i.urdf
```

（這步腳本也已包含。）

---

## ④ Isaac Sim GUI 匯入

`File > Import` → 選 `/tmp/d435i.urdf`，匯入器設定：

| 選項 | 設定 | 原因 |
|------|------|------|
| **Fix Base Link** | ✅ ON | 相機要剛性固定，不是浮動 |
| **Merge Fixed Joints** | ✅ ON | 14 個 fixed joint 合併，optical frame 變乾淨 Xform 階層 |
| **Create Physics Scene** | ❌ OFF | 只要相機本身 |
| Output | `intel_realsense_d435i.usd` | |

---

## ⑤ strip physics（必做）

即使關掉上面選項，URDF 匯入器還是會留 `PhysicsArticulationRootAPI` + RigidBody + Collision + FixedJoint。
直接 reference 進手臂會造成**巢狀 ArticulationRoot**（手臂壞掉）或**自由落體**。要用 pxr 腳本掃掉，只留視覺 + 座標框：

```python
# strip_physics.py  —  用法: python strip_physics.py in.usd out.usd
import sys
from pxr import Usd, Sdf
src, dst = sys.argv[1], sys.argv[2]
stage = Usd.Stage.Open(src)

PHYS_API = ("PhysicsArticulationRootAPI","PhysxArticulationAPI","PhysicsRigidBodyAPI",
            "PhysxRigidBodyAPI","PhysicsMassAPI","PhysicsCollisionAPI","PhysicsMeshCollisionAPI",
            "PhysxCollisionAPI","PhysxConvexHullCollisionAPI","PhysxSDFMeshCollisionAPI")
PHYS_TYPES = ("PhysicsFixedJoint","PhysicsRevoluteJoint","PhysicsPrismaticJoint",
              "PhysicsJoint","PhysicsCollisionGroup")

for p in stage.Traverse():
    for s in list(p.GetAppliedSchemas()):
        if any(k in s for k in PHYS_API):
            p.RemoveAppliedSchema(s)

to_del = [p.GetPath() for p in stage.Traverse() if p.GetTypeName() in PHYS_TYPES]
for top in ("/realsense2_camera/colliders","/colliders","/realsense2_camera/joints"):
    if stage.GetPrimAtPath(top).IsValid(): to_del.append(Sdf.Path(top))
for p in list(stage.Traverse()):
    if p.GetName() == "collisions": to_del.append(p.GetPath())
for path in sorted(set(to_del), key=lambda x: len(str(x)), reverse=True):
    stage.RemovePrim(path)

stage.GetRootLayer().Export(dst)
print("done:", dst)
```

```bash
PYTHONPATH=<usd-core dir> python strip_physics.py \
    intel_realsense_d435i.usd intel_realsense_d435i_clean.usd
```

完成標準：單獨開 `*_clean.usd` 無 ArticulationRoot 警告、d435 外殼正常、比例正確（公尺）。

---

## 關鍵觀念

### optical frame 的座標慣例（決定相機朝向對不對）
- **ROS optical frame**：+Z 朝前、+X 朝右、+Y 朝下，rpy = `(-π/2, 0, -π/2)`
- **Isaac/USD Camera**：看 **−Z**、+Y 朝上
- 匯入後 optical frame 只是 **Xform 座標框，不是 Camera prim**。兩種接法：
  1. env config 把 `TiledCameraCfg` 掛到 `camera_color_optical_frame`，設 `offset.convention="ros"` → Isaac Lab 自動轉，免手算
  2. 在 USD 裡的 optical frame 底下放 `UsdGeom.Camera`，繞 X 軸轉 180°（quat `(0,1,0,0)`）讓它沿 +Z 朝前（像內建 D455）

### nominal extrinsics
`use_nominal_extrinsics:=true` 帶入 Intel 原廠規格的相機間距/偏移（IR 基線 50mm、color 相對 camera_link +15mm Y 等），不用自己量。

---

## 後續處理（非 URDF→USD 本身，但通常會接著做）

- **封裝成自包含（像 D455）**：Isaac 匯入器把 mesh 放在頂層 `/visuals` `/meshes` Scope，用 `instanceable` 內部參考指過去。直接 reference 進手臂時**只帶 defaultPrim 子樹 → mesh 消失**。
  解法：`SetInstanceable(False)` 全部 → `stage.Flatten()`（內聯）→ 刪頂層 `/visuals` `/meshes` `/Render`。**務必測試「reference 進測試 stage 後 Mesh>0 且 Camera>0」**。
- **加 physics（要影響任務時）**：相機/腳架的 collider + 質量放在 gripper link 底下、**不要**給它們自己的 RigidBodyAPI（會自由落體）。每個 shape 給質量、移除 gripper body 層級的 mass override → PhysX 自動算合成質心。

---

## pxr headless 設定（離線跑 USD 腳本）

Isaac 內建 pxr 直接 import 會缺 `libpython3.11.so.1.0`。改用獨立的 usd-core：

```bash
TGT=/tmp/usdpkg
~/env_isaaclab/bin/pip install --target "$TGT" usd-core
PYTHONPATH="$TGT" ~/env_isaaclab/bin/python your_script.py
```

---

## 檔案產出對照

| 檔案 | 來源步驟 | 內容 |
|------|---------|------|
| `/tmp/d435i.urdf` | ②③ | 扁平 URDF，絕對 mesh 路徑 |
| `intel_realsense_d435i.usd` | ④ | GUI 匯入原檔（含 physics 雜物） |
| `intel_realsense_d435i_clean.usd` | ⑤ | strip 後純視覺 + frames |
| `RSD435i.usd` | 後續封裝 | 自包含、含 Camera prim，可安全 reference |
