#!/usr/bin/env python3
"""列出資料集目前每一集:檔號、episode_index、幀數、長度、可開來看的影片路徑。

用途:分批錄製/刪除時,先跑這個看清楚每一集是哪一集,再用「檔號」去
tools/delete_episodes.py 刪。刪除是用檔號(file-XXX)指定的,所以這裡把檔號放第一欄。

用法(在專案根目錄、isaaclab venv 裡執行):

    python tools/list_episodes.py                                    # sim 資料集(預設)
    python tools/list_episodes.py \
        --repo ChihHanShen/bimanual-so101-pickvials-real \
        --root datasets/bimanual-so101-pickvials-real                # 改看 real
"""
import argparse
import glob
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 預設看 sim 資料集;要看 real 就下 --root/--repo(不必再改這個檔案)。
DEFAULT_REPO = "ChihHanShen/bimanual-so101-pickvials-sim"
DEFAULT_ROOT = "datasets/bimanual-so101-pickvials-sim"

# 用哪顆相機的影片路徑來顯示(方便你直接開來看)
PREVIEW_CAM = "observation.images.center"


def main():
    p = argparse.ArgumentParser(description="列出資料集每一集的檔號 / episode_index / 長度")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"repo_id(預設 {DEFAULT_REPO})")
    p.add_argument("--root", default=DEFAULT_ROOT, help=f"資料集路徑(預設 {DEFAULT_ROOT})")
    p.add_argument("--cam", default=PREVIEW_CAM, help="預覽影片用哪顆相機")
    args = p.parse_args()

    root = Path(args.root).resolve()
    ds = LeRobotDataset(args.repo, root=root)
    fps = ds.meta.fps
    print(f"[INFO] {ds.meta.total_episodes} 集,共 {ds.meta.total_frames} 幀,fps={fps}\n")

    # 從 meta/episodes 讀每集的:檔號(data/影片共用)、episode_index、長度
    rows = []
    for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet")):
        t = pq.read_table(f)
        for i in range(t.num_rows):
            ep = t.column("episode_index").to_pylist()[i]
            length = t.column("length").to_pylist()[i]
            fnum = t.column("data/file_index").to_pylist()[i]
            rows.append((fnum, ep, length))
    rows.sort()  # 依檔號排序

    # 表頭:檔號放第一欄(刪除時就是給這個號)
    print(f"{'檔號(刪這個)':>12} | {'index':>5} | {'幀數':>6} | {'長度':>7} | 影片(可開來看)")
    print("-" * 78)
    for fnum, ep, length in rows:
        sec = length / fps
        vpath = f"{root}/videos/{args.cam}/chunk-000/file-{fnum:03d}.mp4"
        print(f"{'file-%03d' % fnum:>12} | {ep:>5} | {length:>6} | {sec:>6.1f}s | {vpath}")


if __name__ == "__main__":
    main()
