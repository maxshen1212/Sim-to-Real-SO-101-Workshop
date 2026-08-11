#!/usr/bin/env python3
"""SO-101 四支手臂的固定 /dev 名稱設定與檢查。

四支手臂的控制板用的都是同一顆 USB-serial 晶片 (1a86:55d3 = WCH CH343)，
`/dev/ttyACM*` 的編號會隨插拔與開機順序改變，所以用 udev 規則把「USB 序號」
綁到固定名稱。綁序號（不是 USB 路徑）代表插在哪個實體孔都不影響。
注意序號屬於控制板，不屬於手臂 —— 換控制板要重跑 --identify。

序號存在本 repo 的 calibration/config/arm_serials.json，跟著 repo 走，所以套用規則不需要任何互動：

    graphen-setup-udev              # 檢查目前連線（預設，不會改動任何東西）
    graphen-setup-udev --apply      # 依序號寫入 udev 規則（需要 sudo）
    graphen-setup-udev --identify   # 換過 USB 轉板時，重新辨識序號後自動套用

序號表的位置可以用環境變數 GRAPHEN_ARM_SERIALS 覆寫。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BY_ID_DIR = Path("/dev/serial/by-id")
RULES_PATH = Path("/etc/udev/rules.d/99-robot.rules")
SERIALS_RELPATH = Path("calibration/config/arm_serials.json")
VENDOR_ID = "1a86"
PRODUCT_ID = "55d3"

# role -> (/dev 下的固定名稱, 說明)。名稱必須和 calibration/config/*.yaml 的 port 一致。
ARMS = {
    "follower_left": ("ttyFollowerLeft", "Follower 左臂"),
    "follower_right": ("ttyFollowerRight", "Follower 右臂"),
    "leader_left": ("ttyLeaderLeft", "Leader   左臂"),
    "leader_right": ("ttyLeaderRight", "Leader   右臂"),
}


def connected_serials() -> dict[str, str]:
    """回傳目前連線的 {序號: ttyACMx}。"""
    found = {}
    if not BY_ID_DIR.exists():
        return found
    for link in BY_ID_DIR.iterdir():
        if "USB_Single_Serial_" not in link.name:
            continue
        serial = link.name.split("USB_Single_Serial_")[1].split("-")[0]
        found[serial] = link.resolve().name
    return found


def repo_serials() -> Path:
    """序號表的正式位置：本 repo 的 calibration/config/arm_serials.json。

    source/sim_to_real_so101/scripts/graphen_setup_udev.py -> parents[3] = repo 根目錄。
    """
    return Path(__file__).resolve().parents[3] / SERIALS_RELPATH


def serials_candidates() -> list[Path]:
    """序號表的搜尋順序：環境變數 → 本 repo → 目前工作目錄。"""
    candidates = []
    if env := os.environ.get("GRAPHEN_ARM_SERIALS"):
        candidates.append(Path(env).expanduser())
    candidates.append(repo_serials())
    candidates.append(Path.cwd() / SERIALS_RELPATH)
    return candidates


def serials_path() -> Path:
    for path in serials_candidates():
        if path.exists():
            return path
    print("[錯誤] 找不到序號表，已嘗試：")
    for path in serials_candidates():
        print(f"  {path}")
    print("請從本 repo 根目錄執行，或設定 GRAPHEN_ARM_SERIALS，或執行 `graphen-setup-udev --identify`。")
    sys.exit(1)


def load_serials() -> dict[str, str]:
    path = serials_path()
    data = json.loads(path.read_text())
    missing = [role for role in ARMS if role not in data]
    if missing:
        print(f"[錯誤] {path} 缺少：{', '.join(missing)}")
        sys.exit(1)
    return {role: data[role] for role in ARMS}


def save_serials(serials: dict[str, str]) -> None:
    # 都不存在時（全新 checkout 跑 --identify）寫到正式位置(本 repo)，不要落在 cwd。
    path = next((p for p in serials_candidates() if p.exists()), repo_serials())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_comment": (
            "SO-101 四支手臂的 USB 序號。由 `graphen-setup-udev --identify` 產生，"
            "也可以手動編輯。換 USB 轉板時才需要更新。"
        ),
        **serials,
    }
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    print(f"序號已寫入 {path}（記得 commit）")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def check() -> int:
    """檢查四個固定名稱是否都指向正確的裝置。不會改動任何東西。"""
    print("=" * 68)
    print("  SO-101 USB 連線檢查")
    print("=" * 68)
    print()

    serials = load_serials()
    live = connected_serials()
    ok = True
    claimed = set()

    for role, (link, desc) in ARMS.items():
        want = serials[role]
        path = Path("/dev") / link
        if not path.exists():
            reason = "裝置未連接" if want not in live else "udev 規則未套用 → --apply"
            print(f"  {desc}  /dev/{link:<17} ✗ 不存在（{reason}）")
            ok = False
            continue

        tty = path.resolve().name
        claimed.add(tty)
        got = next((s for s, t in live.items() if t == tty), None)
        if got == want:
            print(f"  {desc}  /dev/{link:<17} → /dev/{tty:<8} 序號 {got}  ✓")
        else:
            print(f"  {desc}  /dev/{link:<17} → /dev/{tty:<8} 序號 {got}  ✗ 應為 {want}")
            ok = False

    for tty in sorted(set(live.values()) - claimed):
        serial = next(s for s, t in live.items() if t == tty)
        known = next((r for r, s in serials.items() if s == serial), None)
        note = f"（{known} 的序號，但 symlink 沒建立）" if known else "（不在序號表中）"
        print(f"\n  [警告] /dev/{tty} 序號 {serial} 沒有對應的固定名稱 {note}")
        ok = False

    print()
    print(f"  序號表：{serials_path()}")
    if RULES_PATH.exists():
        print(f"  規則檔：{RULES_PATH}")
    else:
        print(f"  [警告] 找不到 {RULES_PATH} → 執行 `graphen-setup-udev --apply`")
        ok = False

    others = sorted(
        p.name for p in RULES_PATH.parent.glob("*.rules") if "lerobot" in p.name or "robot" in p.name
    )
    if len(others) > 1:
        print(f"  [警告] 偵測到多份手臂規則檔：{', '.join(others)}，請只保留 {RULES_PATH.name}")
        ok = False

    print()
    print("  ✓ 全部正常" if ok else "  ✗ 有問題，見上方")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def apply() -> int:
    """依序號表產生並套用 udev 規則。不需要互動。"""
    serials = load_serials()

    dupes = {s for s in serials.values() if list(serials.values()).count(s) > 1}
    if dupes:
        print(f"[錯誤] 序號表中有重複的序號：{', '.join(sorted(dupes))}")
        return 1

    lines = [
        "# SO-101 固定 /dev 名稱規則",
        f"# 由 graphen-setup-udev --apply 產生，序號來源：{serials_path()}",
        "# 綁定依據是 USB 序號，與插在哪個實體 USB 孔無關。",
        "",
    ]
    for role, (link, _) in ARMS.items():
        lines.append(
            f'SUBSYSTEM=="tty", ATTRS{{idVendor}}=="{VENDOR_ID}", '
            f'ATTRS{{idProduct}}=="{PRODUCT_ID}", ATTRS{{serial}}=="{serials[role]}", '
            f'SYMLINK+="{link}", MODE="0666"'
        )

    print("即將寫入的規則：\n")
    for line in lines[4:]:
        print(f"  {line}")
    print()

    if RULES_PATH.exists():
        backup = RULES_PATH.with_name(f"{RULES_PATH.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"備份既有規則 → {backup}")
        if subprocess.run(["sudo", "cp", str(RULES_PATH), str(backup)]).returncode != 0:
            return 1

    print(f"寫入 {RULES_PATH}（需要 sudo）...")
    written = subprocess.run(
        ["sudo", "tee", str(RULES_PATH)],
        input=("\n".join(lines) + "\n").encode(),
        capture_output=True,
    )
    if written.returncode != 0:
        print(f"[錯誤] 寫入失敗：{written.stderr.decode()}")
        return 1

    print("套用規則...")
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["sudo", "udevadm", "trigger"], check=True)
    time.sleep(1)
    print()

    rc = check()
    print()
    # 0.4.x 的每臂欄位是扁平的（left_arm_port），不是 0.6.x 的 left_arm_config.port。
    print("這些名稱和 calibration/config/*.yaml 的 port 一致，例如：")
    print("  --robot.left_arm_port=/dev/ttyFollowerLeft")
    print("  --teleop.left_arm_port=/dev/ttyLeaderLeft")
    return rc


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


def identify_one(desc: str) -> str | None:
    """請使用者拔掉一支手臂以辨識其序號，再插回。回傳序號。"""
    while True:
        before = connected_serials()
        input(f"  請「拔掉」{desc} 的 USB，然後按 Enter...")
        gone = set(before) - set(connected_serials())
        if len(gone) == 1:
            serial = gone.pop()
            print(f"  → 序號 {serial}")
            input("  請插回 USB，然後按 Enter...")
            if serial not in connected_serials():
                print("  [警告] 尚未偵測到裝置插回，請確認後再繼續。")
            return serial
        if not gone:
            print("  [警告] 沒有偵測到任何裝置被拔掉，請再試一次。")
        else:
            print(f"  [警告] 一次偵測到 {len(gone)} 個裝置消失，請一次只拔一支。")


def identify() -> int:
    """重新辨識四支手臂的序號並更新序號表，接著自動套用。"""
    print("=" * 68)
    print("  重新辨識 SO-101 序號")
    print("=" * 68)
    print()
    print("請先確認四支手臂都已接上電腦。接著會逐一請你拔插一支手臂來辨識。")
    print()

    live = connected_serials()
    if not live:
        print("[錯誤] 找不到任何 USB 序列裝置。")
        return 1
    print(f"目前連線 {len(live)} 個裝置：")
    for serial, tty in sorted(live.items(), key=lambda kv: kv[1]):
        print(f"  {serial}  →  /dev/{tty}")
    if len(live) != len(ARMS):
        print(f"\n[警告] 預期 {len(ARMS)} 個裝置。")
    print()

    serials: dict[str, str] = {}
    for role, (_, desc) in ARMS.items():
        print(f"─── {desc} ───")
        serial = identify_one(desc)
        if serial in serials.values():
            dup = next(r for r, s in serials.items() if s == serial)
            print(f"  [錯誤] 序號 {serial} 已經指派給 {dup}，請重新執行。")
            return 1
        serials[role] = serial
        print()

    save_serials(serials)
    print()
    return apply()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="依序號表寫入 udev 規則（需要 sudo）")
    group.add_argument("--identify", action="store_true", help="重新辨識序號後自動套用（換 USB 轉板時用）")
    group.add_argument("--check", action="store_true", help="檢查目前連線（預設行為）")
    args = parser.parse_args()

    if args.identify:
        return identify()
    if args.apply:
        return apply()
    return check()


if __name__ == "__main__":
    sys.exit(main())
