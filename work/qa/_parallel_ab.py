# -*- coding: utf-8 -*-
"""临时验收：_run_job 并行 vs 串行（JINGYIN_WORKERS=1）。

只统计像素与字节，不看页面内容：
1. 两遍产出的每个成品文件 md5 必须完全一致；
2. job["results"] 顺序必须等于任务原始顺序；
3. 失败项要独立标记 error、不影响同批其他文件，且计数/状态/进度口径不变；
4. 视频任务与图片任务混跑时，视频独占一条串行通道，进度按帧推进。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
QA = ROOT / "work" / "qa"
import server  # noqa: E402

REGIONS = [{"x": 0.02, "y": 0.02, "w": 0.95, "h": 0.2}]

PAGE_SIZES = [
    (1400, 1000), (900, 1300), (1600, 900), (1200, 1200),
    (800, 600), (2000, 1400), (1000, 750),
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def make_pages() -> list[Path]:
    """七份自合成的「正文 + 顶部斜向半透明蓝章」页面。

    旧版直接读 work/uploads/*/*/source.png（用户上传的真页），而 cleanup_old_files()
    按 72 小时整批删会话目录——语料清空后这里直接 IndexError。判据要的是「图片引擎真的
    跑了、且逐文件字节可复现」，不需要用户的作品，所以尺寸随机序列写死、内容自己画。
    """
    src_dir = QA / "_par_src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)
    pages: list[Path] = []
    for index, (w, h) in enumerate(PAGE_SIZES):
        rng = random.Random(1000 + index)
        page = Image.new("RGB", (w, h), (247, 246, 242))
        draw = ImageDraw.Draw(page)
        y = int(h * 0.24)
        while y < h - 30:
            x = int(w * 0.06)
            limit = int(w * 0.94) - rng.randint(0, int(w * 0.15))
            while x < limit:
                seg = rng.randint(max(4, int(w * 0.02)), max(8, int(w * 0.06)))
                draw.line([x, y, min(x + seg, limit), y], fill=(32, 32, 34), width=2)
                x += seg + rng.randint(6, 14)
            y += rng.randint(18, 26)
        band = int(h * 0.22)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        stamp = ImageDraw.Draw(overlay, "RGBA")
        for offset in range(-band, w + band, 90):
            stamp.line(
                [(offset, 0), (offset + band, band)],
                fill=(30, 90, 205, 120),
                width=18,
            )
        out = src_dir / f"par{index}.png"
        Image.alpha_composite(page.convert("RGBA"), overlay).convert("RGB").save(out)
        pages.append(out)
    return pages


def make_clip(base_page: Path, seconds: int = 24) -> Path:
    """合成一段 720x540、右下角带半透明色块的 mp4（只为验证视频通路）。"""
    with Image.open(base_page) as opened:
        base = opened.convert("RGB").resize((720, 540), Image.Resampling.BILINEAR)
    frames_dir = QA / "_par_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for i in range(seconds):
        frame = base.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        offset = (i * 3) % 40
        draw.rectangle([380 + offset, 430, 700, 510], fill=(60, 120, 220, 130))
        frame.save(frames_dir / f"{i:04d}.png")
    clip = QA / "_par_clip.mp4"
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", "12",
         "-i", str(frames_dir / "%04d.png"), "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    shutil.rmtree(frames_dir)
    return clip


def build_tasks(output_dir: Path, clip: Path, pages: list[Path]) -> list[dict]:
    tasks = []
    for index, source in enumerate(pages):
        with Image.open(source) as opened:
            width, height = opened.size
        tasks.append(
            {
                "id": f"t{index}",
                "item_id": source.stem,
                "name": source.name,
                "kind": "image",
                "source_path": source,
                "width": width,
                "height": height,
                "suffix": ".png",
                "regions": REGIONS,
                "method": "smart",
                "status": "pending",
                "progress": 0,
                "error": None,
                "output_name": f"图{index}_已去水印.png",
            }
        )
    tasks.append(
        {
            "id": "tv",
            "item_id": "clip",
            "name": "clip.mp4",
            "kind": "video",
            "source_path": clip,
            "width": 720,
            "height": 540,
            "suffix": ".mp4",
            "regions": [{"x": 0.5, "y": 0.78, "w": 0.46, "h": 0.16}],
            "method": "smart",
            "status": "pending",
            "progress": 0,
            "error": None,
            "output_name": "视频_已去水印.mp4",
        }
    )
    tasks.append(
        {
            "id": "tbad",
            "item_id": "missing",
            "name": "缺失.png",
            "kind": "image",
            "source_path": QA / "_par_not_exist.png",
            "width": 100,
            "height": 100,
            "suffix": ".png",
            "regions": REGIONS,
            "method": "smart",
            "status": "pending",
            "progress": 0,
            "error": None,
            "output_name": "坏件_已去水印.png",
        }
    )
    return tasks


def poll_job(job_id: str, stop: threading.Event) -> list[tuple[float, int]]:
    """处理过程中按前端节奏轮询 public_job（它自己会抢 INDEX_LOCK），检查口径是否自洽。"""
    snaps: list[tuple[float, int]] = []
    while not stop.is_set():
        job = server.JOBS.get(job_id)
        if job:
            view = server.public_job(job)
            snaps.append((float(view["progress"]), len(view["results"])))
        stop.wait(0.02)
    return snaps


def check_snapshots(snaps: list[tuple[float, int]]) -> bool:
    if len(snaps) < 5:
        print("[FAIL] 采样太少", len(snaps))
        return False
    progress = [s[0] for s in snaps]
    counts = [s[1] for s in snaps]
    mono_p = all(b - a >= -1e-9 for a, b in zip(progress, progress[1:]))
    mono_c = all(b >= a for a, b in zip(counts, counts[1:]))
    bounded = min(progress) >= 0.0 and max(progress) <= 1.0
    print(f"[{'OK' if mono_p else 'FAIL'}] 轮询到的总进度单调不减 = {mono_p} "
          f"范围 {min(progress):.3f}~{max(progress):.3f}")
    print(f"[{'OK' if mono_c else 'FAIL'}] 轮询到的成品数只增不减 = {mono_c} "
          f"{min(counts)}~{max(counts)} 个（采样 {len(snaps)} 次）")
    print(f"[{'OK' if bounded else 'FAIL'}] 进度不越界 = {bounded}")
    return mono_p and mono_c and bounded


def run_once(
    tag: str, clip: Path, workers: str | None, pages: list[Path]
) -> tuple[dict, Path, float, list]:
    output_dir = QA / f"_par_out_{tag}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    job_id = f"job_{tag}"
    tasks = build_tasks(output_dir, clip, pages)
    server.JOBS[job_id] = {
        "id": job_id,
        "session_id": "s",
        "status": "queued",
        "progress": 0,
        "message": "",
        "tasks": tasks,
        "results": [],
        "success_count": 0,
        "failed_count": 0,
        "output_dir": output_dir,
        "created_at": time.time(),
    }
    if workers:
        os.environ["JINGYIN_WORKERS"] = workers
    else:
        os.environ.pop("JINGYIN_WORKERS", None)
    started = time.perf_counter()
    stop = threading.Event()
    poller_out: list[list] = []
    poller = threading.Thread(
        target=lambda: poller_out.append(poll_job(job_id, stop)), daemon=True
    )
    poller.start()
    server._run_job(job_id)
    stop.set()
    poller.join(timeout=5)
    elapsed = time.perf_counter() - started
    job = server.JOBS[job_id]
    return (
        json.loads(json.dumps(server.public_job(job), default=str)),
        output_dir,
        elapsed,
        poller_out[0] if poller_out else [],
    )


def main() -> None:
    pages = make_pages()
    print(f"图片夹具= {len(pages)} 份自合成页面（不再读 work/uploads 的真语料）")
    clip = make_clip(pages[0])
    print("视频夹具=", clip, clip.stat().st_size, "B")
    serial_job, serial_dir, serial_s, serial_snaps = run_once("serial", clip, "1", pages)
    par_job, par_dir, par_s, par_snaps = run_once("parallel", clip, None, pages)
    print(f"串行 {serial_s:.2f}s   并行 {par_s:.2f}s   加速 {serial_s / par_s:.2f}x  任务数=9")
    print("-- 串行时的轮询口径 --")
    ok_serial = check_snapshots(serial_snaps)
    print("-- 并行时的轮询口径 --")
    ok_par = check_snapshots(par_snaps)

    ok = ok_serial and ok_par
    # 1. 逐文件字节一致（预览缓存文件名带随机 result_id，两遍必然不同，只数个数）
    names = sorted(p.name for p in serial_dir.iterdir() if not p.name.startswith("."))
    par_names = sorted(p.name for p in par_dir.iterdir() if not p.name.startswith("."))
    if names != par_names:
        ok = False
        print("[FAIL] 产出文件清单不同", set(names) ^ set(par_names))
    diff = []
    for name in names:
        a, b = md5(serial_dir / name), md5(par_dir / name)
        if a != b:
            diff.append(name)
    print(f"[{'OK' if not diff else 'FAIL'}] 成品逐文件 md5 一致 = {not diff} 差异文件={diff}")
    ok = ok and not diff

    # 2. 结果顺序 == 任务原始顺序
    for tag, job in (("串行", serial_job), ("并行", par_job)):
        got = [r["task_id"] for r in job["results"]]
        expect = [t["id"] for t in par_job["tasks"] if t["status"] == "done"]
        order_ok = got == expect
        ok = ok and order_ok
        print(f"[{'OK' if order_ok else 'FAIL'}] {tag} results 顺序按任务原序 = {order_ok}")
        print(f"     状态={job['status']} 成功={job['success_count']} 失败={job['failed_count']} "
              f"进度={job['progress']} 消息={job['message']}")
        bad = [t for t in job["tasks"] if t["id"] == "tbad"][0]
        print(f"     坏件 status={bad['status']} error={bad['error']}")
        ok = ok and bad["status"] == "error" and bool(bad["error"])
        snake = {"success_count", "failed_count", "output_dir", "zip_ready", "completed"} <= set(job)
        ok = ok and snake
        print(f"     snake_case 口径不变 = {snake} / zip_ready={job['zip_ready']} "
              f"completed={job['completed']}/{job['total']}")
        kinds = {r["kind"] for r in job["results"]}
        sizes = {r["name"]: r["size"] for r in job["results"]}
        print(f"     成品类型={sorted(kinds)} 视频成品字节={sizes.get('视频_已去水印.mp4')}")

    # 3. 预览文件：图片有 .preview_，视频预览就是成品
    for d in (serial_dir, par_dir):
        prev = [p.name for p in d.iterdir() if p.name.startswith(".preview_")]
        print(f"     {d.name} 预览缓存 {len(prev)} 个（图片 7 张应有 7 个）")
        ok = ok and len(prev) == 7

    print("结论=", "全部通过" if ok else "有不合格项")
    for leftover in (serial_dir, par_dir, clip, QA / "_par_src"):
        if leftover.is_dir():
            shutil.rmtree(leftover)
        elif leftover.is_file():
            leftover.unlink()


if __name__ == "__main__":
    main()
