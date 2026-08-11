#!/usr/bin/env python3
"""驗證 sim 與真機資料集的規格一致、且 state/action 值域對得上。

為什麼需要這支:雙臂 sim-to-real 要 co-train,兩批資料必須「長得一模一樣」——
schema 差一個欄位名、fps 差一倍、單位差一個 scale,GR00T 都不會報錯,只會學壞。
2026-08 那次就是踩到:真機錄成 DEGREES、sim 的 fps 標籤差一倍,兩個都是靜默的。

用法:
    # 兩批都收完後,完整比對
    python tools/check_dataset_parity.py \\
        --sim  ~/sim2real/lerobot/datasets/bimanual-so101-pickvials-sim \\
        --real ~/sim2real/lerobot/datasets/bimanual-so101-pickvials-real

    # 只收了一批 -> 仍會跑完該批的自檢(Section 1),自動跳過比對
    python tools/check_dataset_parity.py --sim ~/sim2real/lerobot/datasets/...-sim

    --quick    不讀 parquet(略過全量值域與速度比對,只信 meta/stats.json)
    --strict   把 WARN 也當失敗(適合放進 CI)

離開碼:0 = 全過;1 = 有 FAIL(--strict 下 WARN 也算)。

四個區塊:
  1. 每批各自的自檢     —— fps / schema / 值域 / 任務字串,單批就能跑
  2. sim vs real 規格比對 —— 硬性相等:欄位、名稱順序、相機、fps
  3. state/action 值域比對 —— 兩批的分佈要重疊;完全不相交 = 單位或校準對不上
  4. GR00T modality.json  —— 12 維切法與 video key 對得上 checkpoint 的設定
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas ships with lerobot
    pd = None

# ── 這個專案的規格常數(改這裡就等於改驗收標準) ────────────────────────────
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CANONICAL_NAMES = [f"left_{j}.pos" for j in JOINTS] + [f"right_{j}.pos" for j in JOINTS]
CANONICAL_CAMERAS = {"center", "wrist_left", "wrist_right"}
CANONICAL_SHAPE = [480, 640, 3]
EXPECTED_FPS = 30
EXPECTED_TASK = "Pick up the vial and place it in the rack"

# RANGE_M100_100 的手臂關節 vs RANGE_0_100 的夾爪(見 so101_follower.py 的 Motor 定義)
ARM_IDX = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER_IDX = [5, 11]
ARM_RANGE = (-100.0, 100.0)
GRIPPER_RANGE = (0.0, 100.0)

DEFAULT_MODALITY = Path.home() / "sim2real/Isaac-GR00T/examples/SO101_bimanual/modality.json"

VALUE_KEYS = ["observation.state", "action"]


# ── 報告 ──────────────────────────────────────────────────────────────────
class Report:
    def __init__(self, strict: bool = False, color: bool | None = None):
        self.fails = 0
        self.warns = 0
        self.strict = strict
        self.color = sys.stdout.isatty() if color is None else color

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def section(self, title: str) -> None:
        print(f"\n{self._c('1', '━━ ' + title + ' ' + '━' * max(0, 66 - len(title)))}")

    def ok(self, msg: str) -> None:
        print(f"  {self._c('32', 'PASS')}  {msg}")

    def fail(self, msg: str, hint: str = "") -> None:
        self.fails += 1
        print(f"  {self._c('31', 'FAIL')}  {msg}")
        if hint:
            print(f"        {self._c('31', '↳ ' + hint)}")

    def warn(self, msg: str, hint: str = "") -> None:
        self.warns += 1
        print(f"  {self._c('33', 'WARN')}  {msg}")
        if hint:
            print(f"        {self._c('33', '↳ ' + hint)}")

    def info(self, msg: str) -> None:
        print(f"        {msg}")

    def check(self, cond: bool, msg: str, hint: str = "") -> bool:
        self.ok(msg) if cond else self.fail(msg, hint)
        return cond

    def finish(self) -> int:
        bad = self.fails + (self.warns if self.strict else 0)
        print()
        if bad == 0 and self.warns == 0:
            print(self._c("32", "✔ 全部通過。"))
        elif bad == 0:
            print(self._c("33", f"✔ 沒有 FAIL,但有 {self.warns} 個 WARN —— 上面每一條都要自己看過再決定放不放行。"))
        else:
            print(self._c("31", f"✘ {self.fails} 個 FAIL、{self.warns} 個 WARN。這兩批資料還不能一起訓練。"))
        return 1 if bad else 0


# ── 載入 ──────────────────────────────────────────────────────────────────
class Dataset:
    """只讀 meta/ 與 data/ 的 parquet,不碰影片(所以很快)。"""

    def __init__(self, root: str | os.PathLike, label: str, quick: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.label = label
        if not (self.root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"{self.root} 不像 LeRobot 資料集(找不到 meta/info.json)")
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        stats_path = self.root / "meta" / "stats.json"
        self.stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        self.features = self.info.get("features", {})
        self.fps = self.info.get("fps")
        self.tasks = self._load_tasks()
        self.frames = None if quick else self._load_frames()

    def _load_tasks(self) -> list[str]:
        p = self.root / "meta" / "tasks.parquet"
        if pd is None or not p.is_file():
            return []
        df = pd.read_parquet(p)
        # 0.4.3 把任務字串放在 index、task_index 當唯一欄位
        return [str(x) for x in (df.index if df.index.name != "task_index" else df["task"])]

    def _load_frames(self):
        if pd is None:
            return None
        files = sorted(glob.glob(str(self.root / "data" / "**" / "*.parquet"), recursive=True))
        if not files:
            return None
        cols = VALUE_KEYS + ["episode_index", "frame_index", "timestamp"]
        parts = []
        for f in files:
            df = pd.read_parquet(f)
            parts.append(df[[c for c in cols if c in df.columns]])
        return pd.concat(parts, ignore_index=True)

    def matrix(self, key: str) -> np.ndarray | None:
        """(N, 12) 的實際數值;--quick 或缺欄位時回 None。"""
        if self.frames is None or key not in self.frames.columns:
            return None
        return np.stack(self.frames[key].to_numpy()).astype(np.float64)

    def bounds(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        """優先用實際資料,退回 meta/stats.json。"""
        m = self.matrix(key)
        if m is not None:
            return m.min(axis=0), m.max(axis=0)
        s = self.stats.get(key)
        if s and "min" in s and "max" in s:
            return np.asarray(s["min"], dtype=np.float64), np.asarray(s["max"], dtype=np.float64)
        return None

    def cameras(self) -> dict[str, dict]:
        return {k[len("observation.images."):]: v
                for k, v in self.features.items() if k.startswith("observation.images.")}

    def step_deltas(self, key: str) -> np.ndarray | None:
        """每格之間的 |Δ|,只在同一集內計算。用來抓時間基準差一倍的問題。"""
        m = self.matrix(key)
        if m is None or "episode_index" not in self.frames.columns:
            return None
        ep = self.frames["episode_index"].to_numpy()
        d = np.abs(np.diff(m, axis=0))
        return d[ep[1:] == ep[:-1]]


# ── Section 1:單批自檢 ───────────────────────────────────────────────────
def check_single(ds: Dataset, rep: Report, expect_fps: int, expect_task: str | None) -> None:
    rep.section(f"1. 單批自檢 — {ds.label}")
    rep.info(f"{ds.root}")
    rep.info(f"episodes={ds.info.get('total_episodes')} frames={ds.info.get('total_frames')} "
             f"fps={ds.fps} robot_type={ds.info.get('robot_type')} codebase={ds.info.get('codebase_version')}")

    rep.check(ds.fps == expect_fps, f"fps == {expect_fps}",
              f"實際 {ds.fps}。sim 端 fps 由 env.step_dt 推導(decimation),真機端由相機 fps 決定;"
              "兩邊都必須是 30,部署速度直接綁這個值")

    # 1a. state / action schema
    for key in VALUE_KEYS:
        f = ds.features.get(key)
        if not rep.check(f is not None, f"有 `{key}` 欄位"):
            continue
        rep.check(list(f.get("shape", [])) == [12], f"`{key}` shape == (12,)",
                  f"實際 {f.get('shape')}。雙臂是 6+6")
        rep.check(f.get("dtype") == "float32", f"`{key}` dtype == float32", f"實際 {f.get('dtype')}")
        names = list(f.get("names") or [])
        if names == CANONICAL_NAMES:
            rep.ok(f"`{key}` names 與順序正確(left_*×6 → right_*×6,含 .pos 後綴)")
        else:
            missing = [n for n in CANONICAL_NAMES if n not in names]
            extra = [n for n in names if n not in CANONICAL_NAMES]
            hint = f"缺少 {missing}" if missing else (f"多出 {extra}" if extra else "名稱相同但順序不同")
            rep.fail(f"`{key}` names 不符合規格", hint + f";實際 = {names}")

    # 1b. 相機
    cams = ds.cameras()
    rep.check(set(cams) == CANONICAL_CAMERAS, f"相機正好是 {sorted(CANONICAL_CAMERAS)}",
              f"實際 {sorted(cams)}")
    for name, f in sorted(cams.items()):
        rep.check(list(f.get("shape", [])) == CANONICAL_SHAPE, f"相機 `{name}` shape == 480×640×3",
                  f"實際 {f.get('shape')}")
        rep.check(f.get("dtype") == "video", f"相機 `{name}` 存成 video(非 image)",
                  f"實際 {f.get('dtype')};存成 image 會讓資料集大很多")
        vfps = (f.get("info") or {}).get("video.fps")
        if vfps is not None:
            rep.check(abs(float(vfps) - float(ds.fps)) < 1e-6,
                      f"相機 `{name}` 的 video.fps == 資料集 fps",
                      f"video.fps={vfps} 但資料集 fps={ds.fps} —— 影片與關節曲線會對不齊")

    # 1c. 任務字串
    if expect_task is not None:
        if len(ds.tasks) == 1 and ds.tasks[0] == expect_task:
            rep.ok(f'任務字串唯一且正確:"{expect_task}"')
        elif not ds.tasks:
            rep.warn("讀不到 meta/tasks.parquet,跳過任務字串檢查")
        else:
            rep.fail(f"任務字串不符(共 {len(ds.tasks)} 個)",
                     f"實際 {ds.tasks};GR00T 的 language 要單一句、sim/real 一致")

    # 1d. 值域 —— 這是 2026-08 那次真機錄成 DEGREES 的偵測點
    for key in VALUE_KEYS:
        b = ds.bounds(key)
        if b is None:
            rep.warn(f"`{key}` 沒有可用的值域資料(--quick 且無 stats.json)")
            continue
        lo, hi = b
        src = "實際資料" if ds.matrix(key) is not None else "meta/stats.json"
        if len(lo) != 12:
            rep.fail(f"`{key}` 值域長度 {len(lo)} != 12")
            continue
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            rep.fail(f"`{key}` 有 NaN/Inf")
            continue

        bad_arm = [(CANONICAL_NAMES[i], lo[i], hi[i]) for i in ARM_IDX
                   if lo[i] < ARM_RANGE[0] - 1e-3 or hi[i] > ARM_RANGE[1] + 1e-3]
        arm_ok = rep.check(not bad_arm, f"`{key}` 手臂關節都在 ±100 內({src})",
                           "; ".join(f"{n} [{a:.1f}, {b_:.1f}]" for n, a, b_ in bad_arm[:4]))
        if not arm_ok and max(abs(hi[ARM_IDX]).max(), abs(lo[ARM_IDX]).max()) > 150:
            rep.info("↳ 超出這麼多,幾乎確定是錄成 DEGREES(use_degrees 被設成 true)。"
                     "換算比例每個關節不同,不能事後轉,只能重收")

        bad_grip = [(CANONICAL_NAMES[i], lo[i], hi[i]) for i in GRIPPER_IDX
                    if lo[i] < GRIPPER_RANGE[0] - 1e-3 or hi[i] > GRIPPER_RANGE[1] + 1e-3]
        rep.check(not bad_grip, f"`{key}` 夾爪都在 0~100 內({src})",
                  "; ".join(f"{n} [{a:.1f}, {b_:.1f}]" for n, a, b_ in bad_grip)
                  + " —— 夾爪是 RANGE_0_100,不是 ±100")

        # 沒動的關節:min == max 表示整批資料那個關節從頭到尾是常數
        frozen = [CANONICAL_NAMES[i] for i in range(12) if abs(hi[i] - lo[i]) < 1e-6]
        if frozen:
            rep.warn(f"`{key}` 有 {len(frozen)} 個關節整批完全沒動:{frozen}",
                     "leader 沒接上 / port 對錯 / 那隻手臂整批沒操作。模型會學成「這關節永遠不動」")

    # 1e. 時間戳自洽
    if ds.frames is not None and "timestamp" in ds.frames.columns and ds.fps:
        ts = ds.frames["timestamp"].to_numpy(dtype=np.float64)
        fi = ds.frames["frame_index"].to_numpy(dtype=np.float64)
        err = float(np.abs(ts - fi / ds.fps).max())
        rep.check(err < 1e-3, "timestamp == frame_index / fps", f"最大偏差 {err:.4f} s")
        secs = ds.info.get("total_frames", 0) / ds.fps if ds.fps else 0
        rep.info(f"總長 = {ds.info.get('total_frames')} 格 ÷ {ds.fps} fps = {secs:.1f} s"
                 f"({secs / 60:.1f} 分);請自行對照實際錄製時間,差一倍就是 fps 標籤錯了")


# ── Section 2:規格比對 ───────────────────────────────────────────────────
def check_parity(sim: Dataset, real: Dataset, rep: Report) -> None:
    rep.section("2. sim vs real 規格比對(硬性相等)")

    rep.check(sim.fps == real.fps, "fps 相同",
              f"sim={sim.fps} real={real.fps} —— co-train 兩批時間基準必須一致")

    sim_keys = {k for k in sim.features if not k.startswith(("timestamp", "frame_index",
                                                             "episode_index", "index", "task_index"))}
    real_keys = {k for k in real.features if not k.startswith(("timestamp", "frame_index",
                                                               "episode_index", "index", "task_index"))}
    rep.check(sim_keys == real_keys, "資料欄位集合相同",
              f"只在 sim: {sorted(sim_keys - real_keys)};只在 real: {sorted(real_keys - sim_keys)}")

    for key in VALUE_KEYS:
        fs, fr = sim.features.get(key, {}), real.features.get(key, {})
        rep.check(list(fs.get("shape", [])) == list(fr.get("shape", [])), f"`{key}` shape 相同",
                  f"sim={fs.get('shape')} real={fr.get('shape')}")
        ns, nr = list(fs.get("names") or []), list(fr.get("names") or [])
        rep.check(ns == nr, f"`{key}` names 與順序完全相同",
                  f"第一個不同處:{next((f'idx {i}: sim={a} real={b}' for i, (a, b) in enumerate(zip(ns, nr)) if a != b), '長度不同')}")
        rep.check(fs.get("dtype") == fr.get("dtype"), f"`{key}` dtype 相同",
                  f"sim={fs.get('dtype')} real={fr.get('dtype')}")

    cs, cr = sim.cameras(), real.cameras()
    rep.check(set(cs) == set(cr), "相機 key 集合相同",
              f"只在 sim: {sorted(set(cs) - set(cr))};只在 real: {sorted(set(cr) - set(cs))}")
    for name in sorted(set(cs) & set(cr)):
        rep.check(list(cs[name].get("shape", [])) == list(cr[name].get("shape", [])),
                  f"相機 `{name}` shape 相同",
                  f"sim={cs[name].get('shape')} real={cr[name].get('shape')}")

    if sim.tasks and real.tasks:
        rep.check(sim.tasks == real.tasks, "任務字串相同",
                  f"sim={sim.tasks} real={real.tasks}")

    # 這些不相等是正常的,只報不判
    if sim.info.get("robot_type") != real.info.get("robot_type"):
        rep.info(f"robot_type 不同(正常):sim={sim.info.get('robot_type')} real={real.info.get('robot_type')}")
    if sim.info.get("codebase_version") != real.info.get("codebase_version"):
        rep.warn(f"codebase_version 不同:sim={sim.info.get('codebase_version')} "
                 f"real={real.info.get('codebase_version')}", "同一個 LeRobot checkout 錄的話不該不同")


# ── Section 3:值域比對 ───────────────────────────────────────────────────
def _overlap(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> float:
    """兩個區間的 IoU;0 = 完全不相交。

    退化情況要另外處理:某個關節整批沒動時區間寬度是 0,IoU 恆為 0,會被誤判成
    「不相交」而蓋掉真正的問題(那是「沒動」,由別條檢查負責報)。這裡改問
    「那個點有沒有落在對方區間內」。
    """
    span_a, span_b = a[1] - a[0], b[1] - b[0]
    if span_a <= tol or span_b <= tol:
        pt, iv = (a[0], b) if span_a <= tol else (b[0], a)
        return 1.0 if iv[0] - tol <= pt <= iv[1] + tol else 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > tol else 1.0


def check_ranges(sim: Dataset, real: Dataset, rep: Report, iou_warn: float) -> None:
    rep.section("3. state / action 值域比對")
    rep.info("兩批是不同的軌跡,值域本來就不會一模一樣。這裡要抓的是「對不上」:")
    rep.info("完全不相交 = 單位或校準錯了(FAIL);重疊太少 = 涵蓋的工作空間差太多(WARN)。")

    for key in VALUE_KEYS:
        bs, br = sim.bounds(key), real.bounds(key)
        if bs is None or br is None:
            rep.warn(f"`{key}` 缺值域資料,跳過")
            continue
        rep.info("")
        rep.info(f"── {key} " + "─" * 52)
        rep.info(f"{'關節':<22}{'sim min':>9}{'sim max':>9}{'real min':>10}{'real max':>10}"
                 f"{'重疊':>7}{'span比':>8}")
        disjoint, thin = [], []
        for i, name in enumerate(CANONICAL_NAMES):
            a = (float(bs[0][i]), float(bs[1][i]))
            b = (float(br[0][i]), float(br[1][i]))
            iou = _overlap(a, b)
            span_s, span_r = a[1] - a[0], b[1] - b[0]
            ratio = span_s / span_r if span_r > 1e-6 else math.inf
            flag = "  ✘" if iou <= 0 else ("  ⚠" if iou < iou_warn else "")
            rep.info(f"{name:<22}{a[0]:>9.1f}{a[1]:>9.1f}{b[0]:>10.1f}{b[1]:>10.1f}"
                     f"{iou:>7.2f}{ratio:>8.2f}{flag}")
            if iou <= 0:
                disjoint.append(name)
            elif iou < iou_warn:
                thin.append(name)

        rep.info("")
        rep.check(not disjoint, f"`{key}` 沒有完全不相交的關節",
                  f"{disjoint} —— 這代表 sim 和真機在這些關節上根本不在同一個座標系。"
                  "先查單位(DEGREES vs ±100)、再查左右是否對調、再查校準檔")
        if thin:
            rep.warn(f"`{key}` 有 {len(thin)} 個關節重疊 < {iou_warn}:{thin}",
                     "多半是兩邊示範涵蓋的動作範圍不同(例如真機不敢做大動作)。"
                     "不一定是 bug,但 co-train 時這些關節的分佈會被拉開,值得先看一集 replay")
        elif not disjoint:
            rep.ok(f"`{key}` 12 個關節的值域重疊都 ≥ {iou_warn}")

    # 每格位移量:抓時間基準差一倍
    ds_, dr_ = sim.step_deltas("observation.state"), real.step_deltas("observation.state")
    if ds_ is not None and dr_ is not None and len(ds_) and len(dr_):
        ms, mr = np.median(ds_[:, ARM_IDX]), np.median(dr_[:, ARM_IDX])
        rep.info("")
        if mr > 1e-6:
            r = ms / mr
            rep.info(f"每格位移中位數(手臂關節):sim={ms:.3f} real={mr:.3f} → 比值 {r:.2f}")
            # 窗口刻意設得夠緊,能抓到 0.5 / 2.0 這種「時間基準差一倍」的指紋。
            # 代價是操作速度差很多時也會叫 —— 所以是 WARN,不是 FAIL。
            if r < 0.6 or r > 1.7:
                rep.warn(f"兩批的每格位移量差 {r:.2f} 倍",
                         "若接近 0.5 或 2.0,很可能是某一邊的 fps 標籤與真實時間差一倍"
                         "(sim 端就是 decimation 與 recorder fps 不同步)。"
                         "也可能只是操作速度不同 —— 用 replay 目視確認哪一個")
            else:
                rep.ok("兩批每格位移量同一個量級(時間基準看起來一致)")


# ── Section 4:GR00T modality.json ────────────────────────────────────────
def check_modality(datasets: list[Dataset], modality_path: Path, rep: Report) -> None:
    rep.section("4. GR00T modality.json 對齊")
    if not modality_path.is_file():
        rep.warn(f"找不到 {modality_path},跳過")
        return
    mod = json.loads(modality_path.read_text(encoding="utf-8"))
    rep.info(str(modality_path))

    expected = {"left_arm": (0, 5), "left_gripper": (5, 6), "right_arm": (6, 11), "right_gripper": (11, 12)}
    for block in ("state", "action"):
        got = {k: (v["start"], v["end"]) for k, v in mod.get(block, {}).items()}
        rep.check(got == expected, f"`{block}` 切法 = left_arm 0:5 / left_gripper 5:6 / "
                                   f"right_arm 6:11 / right_gripper 11:12", f"實際 {got}")

    covered = sorted({i for v in mod.get("state", {}).values() for i in range(v["start"], v["end"])})
    rep.check(covered == list(range(12)), "`state` 切法剛好覆蓋 0~11 不重不漏", f"實際覆蓋 {covered}")

    keys = {k: v.get("original_key") for k, v in mod.get("video", {}).items()}
    rep.check(set(keys) == CANONICAL_CAMERAS, f"video 三台相機 = {sorted(CANONICAL_CAMERAS)}",
              f"實際 {sorted(keys)}")
    for ds in datasets:
        missing = [ok for ok in keys.values() if ok not in ds.features]
        rep.check(not missing, f"{ds.label} 有 modality 需要的所有 video 欄位", f"缺 {missing}")


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="驗證 sim / real 雙臂資料集規格一致且值域對得上",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--sim", help="sim 資料集 root")
    p.add_argument("--real", help="真機資料集 root")
    p.add_argument("--modality", default=str(DEFAULT_MODALITY), help="GR00T modality.json 路徑")
    p.add_argument("--expect-fps", type=int, default=EXPECTED_FPS)
    p.add_argument("--task", default=EXPECTED_TASK, help="預期的任務字串('' = 不檢查)")
    p.add_argument("--iou-warn", type=float, default=0.5, help="值域重疊低於此值就 WARN(預設 0.5)")
    p.add_argument("--quick", action="store_true", help="不讀 parquet,只信 meta/stats.json")
    p.add_argument("--strict", action="store_true", help="WARN 也算失敗")
    args = p.parse_args()

    if not args.sim and not args.real:
        p.error("至少要給 --sim 或 --real 其中一個")

    rep = Report(strict=args.strict)
    datasets: list[Dataset] = []
    for flag, label in ((args.sim, "sim"), (args.real, "real")):
        if not flag:
            continue
        try:
            datasets.append(Dataset(flag, label, quick=args.quick))
        except (FileNotFoundError, ValueError) as e:
            rep.section(f"1. 單批自檢 — {label}")
            rep.fail(f"讀不到資料集:{e}")

    for ds in datasets:
        check_single(ds, rep, args.expect_fps, args.task or None)

    sim = next((d for d in datasets if d.label == "sim"), None)
    real = next((d for d in datasets if d.label == "real"), None)
    if sim and real:
        check_parity(sim, real, rep)
        check_ranges(sim, real, rep, args.iou_warn)
    else:
        rep.section("2-3. sim vs real 比對")
        rep.info("只給了一批資料集,跳過比對。兩批都收完後再跑一次完整版。")

    if datasets:
        check_modality(datasets, Path(args.modality).expanduser(), rep)

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
