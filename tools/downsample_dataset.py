#!/usr/bin/env python3
"""把資料集從原始 fps 降採樣到 target fps(丟幀,不內插)。

背景:sim 與真機現在都用 RealSense 對齊的 30 fps 錄製,但 GR00T 部署時
「控制頻率 = 訓練資料集 fps」(見 ROADMAP.md「四個會直接弄壞真機的點」第 2 點)。
上一輪拍板用 10 fps 訓練,所以錄完的 30 fps 原始資料集要先跑這支腳本降到 10 fps
的獨立新資料集,再送進 Isaac-GR00T 訓練管線。

做法:不手改 parquet/影片檔,而是借用 LeRobotDataset 自己的寫入路徑
(跟 lerobot_agent_dual 錄製時同一條 add_frame/save_episode),對每個 episode
每 stride 幀取 1 幀重新餵進去,由它自己重新編碼影片、算 stats、寫 info.json。
好處:影片實際編碼用的 fps 一定跟 info.json 頂層的 fps 一致,不會出現
「decimation 改了但某個角落還寫死舊數字」那種標籤對不上資料的坑
(ROADMAP.md 記錄過一次:decimation 從 2 改 4,recorder fps 卻沒跟著從
env.step_dt 推導,標成 30 但實際是 60 Hz 的教訓)。

有一個地方 LeRobotDatasetMetadata.create() 不會幫你做:傳進去的 features
字典裡,state/action 自己的 "fps" 欄位、以及每支攝影機 info 裡的 "video.fps",
都是從來源資料集原封不動複製,不會因為 create(fps=10) 就自動改成 10。
不修就會出現「info.json 頂層寫 10,個別 feature 卻還留著 30」的不一致,
下面 _retag_feature_fps() 就是在補這一塊。

輸出是全新資料夾,不會動到原始 30 fps 資料集;這支腳本本身也**不會**上傳到
Hub —— 跟這專案其他工具一樣,上傳一律是跑完之後自己按 run_cheatsheet.md 的
lerobot_push_dataset 指令,方便上傳前先目視檢查一下資料對不對。

用法(在專案根目錄、isaaclab venv 裡執行):

    python tools/downsample_dataset.py \
        --repo ChihHanShen/bimanual-so101-pickvials-sim \
        --root datasets/bimanual-so101-pickvials-sim \
        --dst-repo ChihHanShen/bimanual-so101-pickvials-sim-10fps \
        --dst-root datasets/bimanual-so101-pickvials-sim-10fps

    python tools/downsample_dataset.py \
        --repo ChihHanShen/bimanual-so101-pickvials-real \
        --root datasets/bimanual-so101-pickvials-real \
        --dst-repo ChihHanShen/bimanual-so101-pickvials-real-10fps \
        --dst-root datasets/bimanual-so101-pickvials-real-10fps

跑完後上傳(兩個資料集都要):

    lerobot_push_dataset --repo-id ChihHanShen/bimanual-so101-pickvials-sim-10fps \
        --root datasets/bimanual-so101-pickvials-sim-10fps
    lerobot_push_dataset --repo-id ChihHanShen/bimanual-so101-pickvials-real-10fps \
        --root datasets/bimanual-so101-pickvials-real-10fps
"""
import argparse
import copy
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _to_hwc_uint8(image):
    return (image * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _retag_feature_fps(features: dict, target_fps: int) -> dict:
    """深拷貝 features,把裡面殘留的舊 fps 標籤(state/action 的 "fps"、
    影片 feature 的 "info"/"video.fps")都改成 target_fps。"""
    features = copy.deepcopy(features)
    for spec in features.values():
        if "fps" in spec:
            spec["fps"] = target_fps
        if spec.get("dtype") == "video" and "video.fps" in spec.get("info", {}):
            spec["info"]["video.fps"] = target_fps
    return features


def downsample(src: LeRobotDataset, dst_repo: str, dst_root: Path, target_fps: int) -> LeRobotDataset:
    src_fps = int(src.fps)
    if src_fps % target_fps != 0:
        raise ValueError(
            f"來源 fps={src_fps} 不能整除 target_fps={target_fps},"
            "這支腳本只做整數倍丟幀,不做內插"
        )
    stride = src_fps // target_fps
    print(f"[INFO] {src_fps} fps -> {target_fps} fps,stride={stride}(每 {stride} 幀留 1 幀)")

    features = _retag_feature_fps(src.meta.features, target_fps)
    dst = LeRobotDataset.create(
        repo_id=dst_repo,
        fps=target_fps,
        features=features,
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
    )

    n_eps = src.meta.total_episodes
    for ep_idx in range(n_eps):
        ep = src.meta.episodes[ep_idx]
        ep_start, ep_end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        kept_indices = range(ep_start, ep_end, stride)

        for idx in kept_indices:
            item = src[idx]
            # 讀出來是 CHW float32 [0,1](解碼後的格式),但寫入端跟錄製時一樣要 HWC uint8。
            frame = {key: _to_hwc_uint8(item[key]) for key in src.meta.camera_keys}
            frame["observation.state"] = item["observation.state"]
            frame["action"] = item["action"]
            frame["task"] = item["task"]
            dst.add_frame(frame)

        # parallel_encoding=False:預設會開 ProcessPoolExecutor 讓三支相機並行編碼,
        # 但那條路徑有 race —— 曾在 real 資料集第 7 集炸出 FileNotFoundError
        # (某個 worker 的 PNG 在 glob 完、還沒 Image.open 之前就被砍掉),而且不是每次都發生,
        # 同一集重跑就過。序列編碼走的是同一個 _encode_video_worker,輸出完全一樣,
        # 而且 LeRobot 自己註明 ffmpeg 內部已經多執行緒,並行不會比較快。
        dst.save_episode(parallel_encoding=False)
        print(
            f"[INFO] episode {ep_idx + 1}/{n_eps}: "
            f"{ep_end - ep_start} 幀 -> {len(kept_indices)} 幀"
        )

    # 一定要 finalize:episode metadata 是每 10 集才 flush 一次的,parquet footer
    # 也只在關閉 writer 時才寫。少了這一步,資料集載不回來(最後幾集的 metadata 會掉)。
    dst.finalize()
    return dst


def main():
    p = argparse.ArgumentParser(description="把 LeRobot 資料集降採樣到 target fps(整數倍丟幀)")
    p.add_argument("--repo", required=True, help="來源 repo_id")
    p.add_argument("--root", required=True, help="來源資料集路徑")
    p.add_argument("--dst-repo", required=True, help="輸出 repo_id(建議 <repo>-<fps>fps)")
    p.add_argument("--dst-root", required=True, help="輸出資料集路徑(必須是新資料夾)")
    p.add_argument("--target-fps", type=int, default=10, help="目標 fps(預設 10)")
    args = p.parse_args()

    src_root = Path(args.root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if dst_root == src_root:
        raise SystemExit("[ERROR] --dst-root 不能跟 --root 相同")
    if dst_root.exists():
        raise SystemExit(f"[ERROR] {dst_root} 已存在,先清掉或換個路徑(避免覆蓋舊輸出)")

    print(f"[INFO] 讀取來源資料集 {args.repo}\n       {src_root}")
    src = LeRobotDataset(args.repo, root=src_root)
    print(f"[INFO] {src.meta.total_episodes} episodes, {src.meta.total_frames} frames @ {src.fps} fps")

    dst = downsample(src, args.dst_repo, dst_root, args.target_fps)

    print(f"[OK] 完成:{dst.meta.total_episodes} episodes, {dst.meta.total_frames} frames @ {dst.meta.fps} fps")
    print(f"[OK] 寫到 {dst_root}")
    print(f"[NEXT] 檢查沒問題後上傳:lerobot_push_dataset --repo-id {args.dst_repo} --root {dst_root}")


if __name__ == "__main__":
    main()
