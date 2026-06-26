#!/usr/bin/env bash
# 產生扁平化、可被 Isaac Sim 匯入的 d435i URDF。
# 用法: bash gen_d435i_urdf.sh
# 重點: 先寫到暫存檔、xacro 成功且 XML 合法後才覆蓋目標檔，
#       避免「xacro 失敗時 > 重導向留下空檔」的陷阱。
# 注意: 不要用 set -u，會弄壞 ROS 的 source setup.bash。
set -eo pipefail

RS_PKG="$HOME/realsense-ros/realsense2_description"
OUT="/tmp/d435i.urdf"
TMP="$(mktemp /tmp/d435i_XXXX.urdf)"

# 1) 假 ament prefix，讓 $(find realsense2_description) 指向 clone 的原始碼
PREFIX=/tmp/rs_prefix
rm -rf "$PREFIX"
mkdir -p "$PREFIX/share/ament_index/resource_index/packages"
touch "$PREFIX/share/ament_index/resource_index/packages/realsense2_description"
ln -sfn "$RS_PKG" "$PREFIX/share/realsense2_description"
export AMENT_PREFIX_PATH="$PREFIX:${AMENT_PREFIX_PATH}"

# 2) ROS 環境 + xacro 展開 → 先進暫存檔
source /opt/ros/humble/setup.bash 2>/dev/null
xacro "$RS_PKG/urdf/test_d435i_camera.urdf.xacro" use_nominal_extrinsics:=true > "$TMP"

# 3) mesh 路徑 package:// → 絕對路徑
sed -i "s#package://realsense2_description/meshes/#${RS_PKG}/meshes/#g" "$TMP"

# 4) 驗證：合法 XML 且 link 數 > 0，才覆蓋目標檔
python3 -c "import xml.dom.minidom as m; m.parse('$TMP')"
LINKS=$(grep -c '<link ' "$TMP")
[ "$LINKS" -gt 0 ] || { echo "❌ link 數為 0，放棄覆蓋"; rm -f "$TMP"; exit 1; }

mv "$TMP" "$OUT"
echo "✅ 產生 $OUT — link=$LINKS joint=$(grep -c '<joint ' "$OUT") mesh=$(grep -c '<mesh ' "$OUT")"
