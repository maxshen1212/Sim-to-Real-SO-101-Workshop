#!/usr/bin/env python3
"""用「檔號 file-XXX」刪除 episode。

你在資料夾/清單看到的是檔號(file-047),但 LeRobot 官方刪除函式要的是它內部的
episode_index。這支工具讓你直接給檔號,內部自動換算成 episode_index 再刪。

為什麼用檔號:檔號是穩定的 —— 某集刪掉後,它的檔號永遠空著、不會被重新分配;
而 episode_index 每次刪都會重新連號。所以用檔號指定要刪哪集,比較不會搞錯。

用法(在專案根目錄、isaaclab venv 裡執行):

    python tools/delete_episodes.py 47 30      # sim 資料集(預設),刪 file-047 / file-030
    python tools/delete_episodes.py 12 \
        --repo ChihHanShen/bimanual-so101-pickvials-real \
        --root datasets/bimanual-so101-pickvials-real     # 改編 real

（用 tools/list_episodes.py 可以列出目前每一集的檔號。）

安全性:先把刪除後的結果寫到 <root>_edit_tmp 並驗證集數,才把原本的搬到 <root>_bak、
新的搬進正式位置,再驗證一次才刪備份。中途出錯原始資料都還在 _bak。
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

# 預設編輯 sim 資料集;要編 real 就下 --root/--repo(不必再改這個檔案)。
DEFAULT_REPO = "ChihHanShen/bimanual-so101-pickvials-sim"
DEFAULT_ROOT = "datasets/bimanual-so101-pickvials-sim"


def main():
    p = argparse.ArgumentParser(
        description="用檔號(file-XXX)刪除 episode",
        epilog="例:python tools/delete_episodes.py 47 30",
    )
    p.add_argument("file_nums", type=int, nargs="+", metavar="檔號",
                   help="要刪除的檔號(file-XXX 的 XXX),可給多個")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"repo_id(預設 {DEFAULT_REPO})")
    p.add_argument("--root", default=DEFAULT_ROOT, help=f"資料集路徑(預設 {DEFAULT_ROOT})")
    args = p.parse_args()

    repo = args.repo
    file_nums = sorted(set(args.file_nums))
    root = Path(args.root).resolve()
    print(f"[INFO] 資料集 {repo}\n       {root}")

    # 載入現有資料集。若資料集損壞/無法載入,會在這裡直接報錯。
    ds = LeRobotDataset(repo, root=root)
    n = ds.meta.total_episodes

    # 建立「檔號 -> episode_index」對照表(從 meta/episodes 讀,data 檔號=影片檔號)
    file_to_ep = {}
    for f in glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet"):
        t = pq.read_table(f, columns=["episode_index", "data/file_index"])
        for i in range(t.num_rows):
            file_to_ep[t.column("data/file_index").to_pylist()[i]] = t.column("episode_index").to_pylist()[i]

    # 檢查你給的檔號都真的存在(可能已刪或打錯)
    missing = [fn for fn in file_nums if fn not in file_to_ep]
    if missing:
        sys.exit(f"[ERROR] 這些檔號不存在(可能已刪或打錯): {['file-%03d' % m for m in missing]}")

    # 換算成 episode_index,並印出對照讓你確認刪對了
    episodes = sorted(file_to_ep[fn] for fn in file_nums)
    print(f"[INFO] 目前 {n} 集。將刪除:")
    for fn in file_nums:
        print(f"    file-{fn:03d}  ->  episode_index {file_to_ep[fn]}")
    print(f"[INFO] 刪掉這 {len(episodes)} 集 → 預計剩 {n - len(episodes)} 集")

    # 暫存輸出(tmp)與備份(bak)目錄,同一層、同磁碟 → 搬移是瞬間完成
    tmp_out = root.parent / (root.name + "_edit_tmp")
    bak = root.parent / (root.name + "_bak")
    for d in (tmp_out, bak):
        if d.exists():
            shutil.rmtree(d)

    # 1) 呼叫官方 delete_episodes:把「刪掉指定集之後」的結果寫成全新資料集到 tmp_out
    #    (不會動到原本的資料集;一集一檔時保留的檔案是直接複製、不重編碼)
    delete_episodes(ds, episode_indices=episodes, output_dir=tmp_out, repo_id=repo)

    # 2) 先確認新資料集能正常載入、集數正確,才敢動原本的
    chk = LeRobotDataset(repo, root=tmp_out)
    assert chk.meta.total_episodes == n - len(episodes), "刪除後集數不符,中止"

    # 3) 安全替換:原本的搬去 _bak → 新的搬進正式位置 → 再驗證 → 確認無誤才刪備份
    #    (中途任何一步出錯,原始資料都還在 _bak,可還原)
    shutil.move(str(root), str(bak))
    shutil.move(str(tmp_out), str(root))
    shutil.rmtree(root / "images", ignore_errors=True)  # 清掉 LeRobot 可能留下的空暫存資料夾
    final = LeRobotDataset(repo, root=root)
    print(f"[OK] 完成,現在 {final.meta.total_episodes} 集")
    shutil.rmtree(bak)


if __name__ == "__main__":
    main()
