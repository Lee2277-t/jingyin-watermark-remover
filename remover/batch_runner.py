# -*- coding: utf-8 -*-
"""批量并行执行：图片按内存预算开多路，视频固定单路（同一时刻只跑一个 ffmpeg）。

worker 数由批内最大图的尺寸决定：实测引擎峰值内存约 130MB/百万像素，
一次解码 + 色度平面 + 反演结果会同时在场，所以大图必须少开甚至退回单路。
`JINGYIN_WORKERS` 可强制指定路数（诊断用）。

单份素材失败只写在它自己的 BatchItem 上，不影响同批其他文件。

`should_stop` 是「停止」闸门：置位后还没开工的素材直接记 cancelled（不再烧 CPU），
正在跑的视频由引擎在检查点抛 CancelRequested；已完成的成品一律保留。
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from remover import options
from remover.cancel import CancelRequested
from remover.image_ops import process_image_file
from remover.media import make_preview
from remover.video_ops import StopCheck, process_video_file

# 实测：处理 1.4MP / 5.8MP / 13MP 页面时进程峰值 RSS 增量 190 / 752 / 1181 MB
PEAK_GB_PER_MP = 0.13
MEM_BUDGET_GB = 5.0


def cpu_cap() -> int:
    # 8 路是实测拐点（16 核，8 页真卷：串行 9.24s → 8 路 2.26s，再高收益很小）
    return max(2, min(8, (os.cpu_count() or 2) // 2))


def plan_workers(megapixels: float) -> int:
    forced = os.environ.get("JINGYIN_WORKERS", "").strip()
    if forced.isdigit():
        return max(1, int(forced))
    per_worker = PEAK_GB_PER_MP * max(megapixels, 0.5)
    return max(1, min(cpu_cap(), int(MEM_BUDGET_GB / per_worker)))


@dataclass
class BatchItem:
    item_id: str
    kind: str  # "image" | "video"
    input_path: Path
    output_path: Path
    regions: list[dict]
    method: str = options.DEFAULT_METHOD
    video_mode: str = options.DEFAULT_VIDEO_MODE  # "quality"＝贴边走逐帧；"fast"＝贴边内缩走 delogo
    motion: bool = False  # 视频：逐帧跟踪会移动的水印（强制走逐帧内容填充）
    width: int = 0
    height: int = 0
    preview_path: Path | None = None
    result: dict | None = None
    error: Exception | None = None
    trace: str = ""
    cancelled: bool = False  # 用户按了停止：这一份没出成品，但不算失败


def _run_one(item: BatchItem, on_progress: Callable[[float], None], should_stop: StopCheck | None) -> dict:
    if item.kind == "image":
        detail = process_image_file(
            item.input_path, item.output_path, item.regions, item.method
        )
        if item.preview_path is not None:
            make_preview(item.output_path, item.preview_path)
    else:
        detail = process_video_file(
            item.input_path,
            item.output_path,
            item.regions,
            item.width,
            item.height,
            method=item.method,
            on_progress=on_progress,
            video_mode=item.video_mode,
            motion=item.motion,
            should_stop=should_stop,
        )
    if not item.output_path.exists() or item.output_path.stat().st_size == 0:
        raise RuntimeError("没有生成有效结果文件")
    return detail


def _execute(
    item: BatchItem,
    on_event: Callable[[BatchItem, str, float], None] | None,
    should_stop: StopCheck | None = None,
) -> BatchItem:
    def fire(phase: str, value: float) -> None:
        if on_event:
            on_event(item, phase, value)

    # 停止之后排队的素材不再开工：已完成的结果保留，未开工的记 cancelled。
    # 已经开始的那一份由引擎在检查点抛 CancelRequested（图片逐像素路线没有中断点，
    # 单份中位不到 1s，等它跑完）。
    if should_stop is not None and should_stop():
        item.cancelled = True
        fire("done", 1.0)
        return item

    fire("start", 0.0)
    try:
        item.result = _run_one(item, lambda value: fire("tick", float(value)), should_stop)
    except CancelRequested:
        item.cancelled = True
    except Exception as exc:  # 单份失败不拖垮整批
        item.error = exc
        item.trace = traceback.format_exc()
    fire("done", 1.0)
    return item


def run_batch(
    items: list[BatchItem],
    workers: int | None = None,
    on_event: Callable[[BatchItem, str, float], None] | None = None,
    should_stop: StopCheck | None = None,
) -> int:
    """并行执行一批素材：图片走线程池，视频共用一条串行通道，返回实际用到的并行路数。"""
    images = [item for item in items if item.kind == "image"]
    videos = [item for item in items if item.kind != "image"]
    biggest = max((item.width * item.height / 1e6 for item in images), default=1.0)
    pool_size = workers or plan_workers(biggest)

    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="jingyin-image") as image_pool:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jingyin-video") as video_pool:
            futures = [image_pool.submit(_execute, item, on_event, should_stop) for item in images]
            futures += [video_pool.submit(_execute, item, on_event, should_stop) for item in videos]
            for future in as_completed(futures):
                future.result()
    return pool_size
