#!/usr/bin/env python3
"""check_dataset_parity.py 的自我測試。

建一組乾淨的合成 sim/real 資料集(schema 與真實 pipeline 完全相同),先確認乾淨的
會過;然後逐一注入九種真實會發生的故障,確認每一種都被抓到、而且是被「對的那一條
檢查」抓到。沒有這層,檢查器本身壞掉是無聲的。

執行(不需要硬體、不需要 Isaac Sim):
    source ~/env_isaaclab/bin/activate
    python tools/test_check_dataset_parity.py

需要 ffmpeg(影片編碼)。第一次跑約 30~60 秒,之後的情境都是複製 + 改 meta,很快。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_dataset_parity.py"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES = [f"left_{j}.pos" for j in JOINTS] + [f"right_{j}.pos" for j in JOINTS]
CAMS = ["wrist_left", "center", "wrist_right"]
TASK = "Pick up the vial and place it in the rack"
ARM_IDX = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER_IDX = [5, 11]

# 兩批共用的「工作空間中心」,讓 sim 與 real 的值域自然重疊 —— 真實情況就是這樣:
# 同一個任務、同一套 ±100 正規化,只是軌跡不同。
HOME = np.array([-5.0, -60.0, 55.0, -20.0, -10.0, 30.0,
                 5.0, -58.0, 57.0, -18.0, -12.0, 30.0], dtype=np.float64)


# ── 建資料集 ──────────────────────────────────────────────────────────────
def build(root: Path, repo_id: str, seed: int, fps: int = 30, n_ep: int = 2, n_frames: int = 24,
          names=NAMES, cams=CAMS, task=TASK, amp: float = 1.0) -> None:
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(seed)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": list(names)},
        "action": {"dtype": "float32", "shape": (len(names),), "names": list(names)},
    }
    for c in cams:
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root,
                               robot_type="bi_so101_follower", use_videos=True)
    dim = len(names)
    for _ in range(n_ep):
        phase = rng.uniform(0, 2 * np.pi, size=dim)
        span = rng.uniform(18, 30, size=dim) * amp
        for t in range(n_frames):
            s = HOME[:dim] + span * np.sin(phase + 2 * np.pi * t / n_frames)
            for gi in GRIPPER_IDX:
                if gi < dim:
                    s[gi] = 30.0 + 25.0 * np.sin(phase[gi] + 2 * np.pi * t / n_frames)
            s = np.clip(s, -100, 100)
            for gi in GRIPPER_IDX:
                if gi < dim:
                    s[gi] = np.clip(s[gi], 0, 100)
            a = np.clip(s + rng.normal(0, 0.3, size=dim), -100, 100)
            for gi in GRIPPER_IDX:
                if gi < dim:
                    a[gi] = np.clip(a[gi], 0, 100)
            frame = {"observation.state": s.astype(np.float32),
                     "action": a.astype(np.float32), "task": task}
            for c in cams:
                frame[f"observation.images.{c}"] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            ds.add_frame(frame)
        ds.save_episode()


# ── 改造工具(注入故障用) ────────────────────────────────────────────────
def copy_of(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def patch_info(root: Path, fn) -> None:
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    fn(info)
    p.write_text(json.dumps(info, indent=4), encoding="utf-8")


def patch_values(root: Path, fn) -> None:
    """改 data/*.parquet 裡的 state/action,並同步更新 meta/stats.json 的 min/max。"""
    files = sorted((root / "data").rglob("*.parquet"))
    allv = {"observation.state": [], "action": []}
    for f in files:
        df = pd.read_parquet(f)
        for key in ("observation.state", "action"):
            m = np.stack(df[key].to_numpy()).astype(np.float64)
            m = fn(m, key)
            df[key] = [row.astype(np.float32) for row in m]
            allv[key].append(m)
        df.to_parquet(f, index=False)
    sp = root / "meta" / "stats.json"
    if sp.is_file():
        stats = json.loads(sp.read_text(encoding="utf-8"))
        for key, mats in allv.items():
            if key in stats and mats:
                m = np.concatenate(mats, axis=0)
                stats[key]["min"] = m.min(axis=0).tolist()
                stats[key]["max"] = m.max(axis=0).tolist()
                stats[key]["mean"] = m.mean(axis=0).tolist()
                stats[key]["std"] = m.std(axis=0).tolist()
        sp.write_text(json.dumps(stats, indent=4), encoding="utf-8")


# ── 執行檢查器 ────────────────────────────────────────────────────────────
def run(sim: Path | None = None, real: Path | None = None, *extra: str):
    cmd = [sys.executable, str(CHECKER)]
    if sim:
        cmd += ["--sim", str(sim)]
    if real:
        cmd += ["--real", str(real)]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "NO_COLOR": "1"})
    return r.returncode, r.stdout + r.stderr


class Results:
    def __init__(self):
        self.passed = 0
        self.failed: list[str] = []

    def expect(self, name: str, code: int, out: str, *, want_code: int,
               contains: list[str] = (), not_contains: list[str] = (),
               no_findings: bool = False) -> None:
        problems = []
        if code != want_code:
            problems.append(f"離開碼 {code},預期 {want_code}")
        for c in contains:
            if c not in out:
                problems.append(f"輸出少了預期字串:{c!r}")
        for c in not_contains:
            if c in out:
                problems.append(f"輸出出現了不該有的字串:{c!r}")
        # 只看「以 FAIL/WARN 開頭的結果行」,不要誤中說明文字裡的同名字眼
        if no_findings and (found := marker_lines(out)):
            problems.append(f"預期零 FAIL 零 WARN,實際有 {len(found)} 條:{found[:3]}")
        if problems:
            self.failed.append(name)
            print(f"  ✘ {name}")
            for p in problems:
                print(f"      {p}")
            tail = "\n".join(line for line in out.splitlines() if "FAIL" in line or "WARN" in line)
            if tail:
                print("      實際抓到的問題:\n        " + tail.replace("\n", "\n        "))
        else:
            self.passed += 1
            print(f"  ✔ {name}")


def marker_lines(out: str, markers=("FAIL", "WARN")) -> list[str]:
    """只抓結果行(以 FAIL/WARN 開頭),忽略說明文字裡出現的同名字眼。"""
    return [ln.strip() for ln in out.splitlines() if ln.strip().startswith(markers)]


# ── 主測試 ────────────────────────────────────────────────────────────────
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="parity-test-"))
    print(f"工作目錄:{tmp}\n建立乾淨的合成資料集(含影片編碼,約 30~60 秒)…")
    clean_sim, clean_real = tmp / "clean-sim", tmp / "clean-real"
    build(clean_sim, "test/bimanual-sim", seed=11)
    build(clean_real, "test/bimanual-real", seed=22)
    print("完成。\n")

    R = Results()

    # ── 0. 乾淨的兩批必須完全通過 ──
    print("【基準】乾淨資料")
    code, out = run(clean_sim, clean_real)
    R.expect("乾淨的 sim+real 完全通過,零 FAIL 零 WARN", code, out,
             want_code=0, contains=["✔ 全部通過"], no_findings=True)

    code, out = run(clean_sim, None)
    R.expect("只給 --sim 時仍跑完自檢,並跳過比對", code, out,
             want_code=0, contains=["只給了一批資料集,跳過比對"], no_findings=True)

    code, out = run(clean_sim, clean_real, "--quick")
    R.expect("--quick 走 stats.json 也通過", code, out,
             want_code=0, contains=["meta/stats.json"], no_findings=True)

    # ── 1. 真機錄成 DEGREES(2026-08 真的發生過) ──
    print("\n【故障 1】真機錄成 DEGREES,值域衝出 ±100")
    d = copy_of(clean_real, tmp / "f1-degrees")
    patch_values(d, lambda m, k: m * 1.9)
    code, out = run(clean_sim, d)
    R.expect("抓到手臂關節超出 ±100,並提示是 DEGREES", code, out, want_code=1,
             contains=["手臂關節都在 ±100 內", "DEGREES"])

    # ── 2. state/action 欄位名稱少了 .pos 後綴 ──
    print("\n【故障 2】sim 的 names 少了 .pos 後綴(sim/real 命名分岔)")
    d = copy_of(clean_sim, tmp / "f2-names")
    bad = [n.replace(".pos", "") for n in NAMES]
    patch_info(d, lambda i: [i["features"][k].__setitem__("names", bad) for k in ("observation.state", "action")])
    code, out = run(d, clean_real)
    R.expect("抓到 names 不符規格,且比對區也報名稱不同", code, out, want_code=1,
             contains=["names 不符合規格", "names 與順序完全相同"])

    # ── 3. fps 不一致(decimation 改壞的等價後果) ──
    print("\n【故障 3】sim fps 標成 60、真機 30")
    d = copy_of(clean_sim, tmp / "f3-fps")
    patch_info(d, lambda i: i.__setitem__("fps", 60))
    code, out = run(d, clean_real)
    R.expect("抓到 fps != 30 且兩批 fps 不同", code, out, want_code=1,
             contains=["fps == 30", "fps 相同"])

    # ── 4. 少一台相機 ──
    print("\n【故障 4】真機只錄了兩台相機")
    d = copy_of(clean_real, tmp / "f4-cams")
    patch_info(d, lambda i: i["features"].pop("observation.images.center"))
    code, out = run(clean_sim, d)
    R.expect("抓到相機集合不符,且 modality 需要的 video 欄位缺失", code, out, want_code=1,
             contains=["相機正好是", "modality 需要的所有 video 欄位"])

    # ── 5. 影像解析度不是 480×640 ──
    print("\n【故障 5】sim 相機解析度被改成 240×320")
    d = copy_of(clean_sim, tmp / "f5-res")

    def shrink(i):
        i["features"]["observation.images.center"]["shape"] = [240, 320, 3]
    patch_info(d, shrink)
    code, out = run(d, clean_real)
    R.expect("抓到解析度不是 480×640,且兩批 shape 不同", code, out, want_code=1,
             contains=["shape == 480×640×3", "shape 相同"])

    # ── 6. 右臂整批沒動(上次測試集真的長這樣) ──
    print("\n【故障 6】右臂六個關節整批常數(leader 沒接上)")
    d = copy_of(clean_sim, tmp / "f6-frozen")

    def freeze(m, k):
        m[:, 6:12] = m[0, 6:12]
        return m
    patch_values(d, freeze)
    code, out = run(d, clean_real)
    R.expect("以 WARN 報出凍結的關節(預設不擋)", code, out, want_code=0,
             contains=["整批完全沒動", "right_shoulder_pan.pos"])
    code, out = run(d, clean_real, "--strict")
    R.expect("--strict 下同一個情況會擋下來", code, out, want_code=1)

    # ── 7. 時間基準差一倍 ──
    print("\n【故障 7】sim 每格位移只有真機的一半(fps 標籤差一倍的指紋)")
    d = copy_of(clean_sim, tmp / "f7-timebase")
    build(d, "test/bimanual-sim", seed=11, n_frames=48, amp=1.0)  # 同樣振幅、兩倍格數 => 每格位移減半
    code, out = run(d, clean_real)
    R.expect("以 WARN 報出每格位移量差約一倍", code, out, want_code=0,
             contains=["每格位移量差", "fps 標籤"])

    # ── 8. 任務字串不一致 ──
    print("\n【故障 8】真機的 language 字串是舊的")
    d = tmp / "f8-task"
    build(d, "test/bimanual-real", seed=22, task="pick up the vial")
    code, out = run(clean_sim, d)
    R.expect("抓到任務字串既不符規格、兩批也不同", code, out, want_code=1,
             contains=["任務字串不符", "任務字串相同"])

    # ── 9. 夾爪跑到負值(誤用 ±100 正規化) ──
    print("\n【故障 9】夾爪用了 ±100 而不是 0~100")
    d = copy_of(clean_real, tmp / "f9-gripper")

    def shift_gripper(m, k):
        m[:, GRIPPER_IDX] -= 60.0
        return m
    patch_values(d, shift_gripper)
    code, out = run(clean_sim, d)
    R.expect("抓到夾爪超出 0~100", code, out, want_code=1,
             contains=["夾爪都在 0~100 內", "RANGE_0_100"])

    # ── 10. 值域完全不相交(左右對調 / 校準檔錯配) ──
    print("\n【故障 10】sim 的左右臂被對調")
    d = copy_of(clean_sim, tmp / "f10-swap")

    def swap(m, k):
        out_ = m.copy()
        out_[:, 0:6], out_[:, 6:12] = m[:, 6:12], m[:, 0:6]
        return out_ + np.array([80.0] * 6 + [-80.0] * 6)  # 推開到不相交
    patch_values(d, swap)
    code, out = run(d, clean_real)
    R.expect("抓到值域完全不相交", code, out, want_code=1,
             contains=["沒有完全不相交的關節", "不在同一個座標系"])

    # ── 11. 檢查器對壞路徑的行為 ──
    print("\n【邊界】不存在的路徑")
    code, out = run(tmp / "does-not-exist", clean_real)
    R.expect("路徑不存在時乾淨報錯,不 traceback", code, out, want_code=1,
             contains=["讀不到資料集"], not_contains=["Traceback"])

    print("\n" + "=" * 72)
    total = R.passed + len(R.failed)
    if R.failed:
        print(f"✘ {len(R.failed)}/{total} 個情境不如預期:{R.failed}")
        print(f"  工作目錄保留供除錯:{tmp}")
        return 1
    print(f"✔ {R.passed}/{total} 個情境全部符合預期。")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
