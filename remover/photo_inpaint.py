# -*- coding: utf-8 -*-
"""照片 / 插画专用的本地重绘路线（LaMa ONNX），只补选框区域，不碰框外像素。

为什么单独一条路线：文档类素材必须走 `image_ops` 的逐像素反演（保字，实测字存活率 77.4%、
零发红）；神经网络重绘会把选框里的印刷字一起抹掉（实测字存活率 4.6%），所以**照片模式
只在用户显式选择时启用，默认仍是智能纹理修复**。它解决的是另一类素材：logo 压在人脸、
墙面、天空等连续纹理上，反演和贴片都补不干净，重绘才有解。

实现要点：
- 模型：`models/lama_fp32.onnx`（LaMa，输入 image[1,3,H,W] 0..1、mask[1,1,H,W] 0/1）。
- 每个选框单独裁一块带上下文的 ROI（长边上限 `MAX_ROI`），等比缩到模型固定的 512×512
  并补边后送推理，结果缩回原尺寸，只在掩膜内羽化回贴——**框外像素逐像素不变**。
- ROI 之间不合并：多个小框各跑一次，比整页缩放到 1024 再放大的质量好得多。
- 模型懒加载（首次照片任务 1–2 秒）；没装 onnxruntime 或模型文件缺失时 `available()`
  为 False，界面上的「照片重绘」卡片直接禁用。
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import cv2
import numpy as np

from .image_ops import Box

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lama_fp32.onnx"
INPUT_SIDE = 512  # 该导出是固定 512×512：ROI 等比缩放到长边 512，短边补边，结果再缩回原尺寸
MAX_ROI = 1024  # 单个 ROI 的长边上限（再大也只是被缩到 512，控制内存）
CONTEXT_RATIO = 0.6
CONTEXT_MIN = 24
FEATHER = 5
THREADS = 8  # 实测 8 线程：单次 512×512 推理约 2.3s；4 线程 5.6s、12 线程 2.0s
# ROI 长边超过这个值就改走分块推理：整块缩到 512 再放大，大 logo 会明显糊。
# 分块时每块不超过 512（不缩放、只补边），只保留掩膜内像素按「离块边多远」加权融合。
TILE_TRIGGER = 640
TILE_OVERLAP = 128
TILE_CONTEXT = 96

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_session = None
_session_error: str | None = None
_lock = threading.Lock()


class PhotoUnavailable(RuntimeError):
    pass


def available() -> bool:
    return _HAS_ORT and MODEL_PATH.exists()


def unavailable_reason() -> str:
    if not _HAS_ORT:
        return "缺少 onnxruntime，照片重绘不可用"
    if not MODEL_PATH.exists():
        return "缺少本地模型 models/lama_fp32.onnx，照片重绘不可用"
    return ""


def warmup() -> None:
    """后台预加载模型：首次加载要 20–30 秒，用户点开「照片重绘」时就开始准备，别等提交任务。"""
    if not available():
        return

    def run() -> None:
        try:
            _load_session()
        except PhotoUnavailable:
            pass

    threading.Thread(target=run, name="jingyin-photo-warmup", daemon=True).start()


def _load_session():
    global _session, _session_error
    if _session is not None:
        return _session
    with _lock:
        if _session is not None:
            return _session
        if not available():
            raise PhotoUnavailable(unavailable_reason())
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = THREADS
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # 模型损坏 / 版本不匹配都要让界面看到原因
            _session_error = f"照片修复模型加载失败：{exc}"
            raise PhotoUnavailable(_session_error) from exc
        return _session


def _run_model(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """把任意尺寸 ROI 等比缩到 512 内并补边，送模型后缩回 ROI 原尺寸。"""
    session = _load_session()
    height, width = roi_mask.shape[:2]
    scale = INPUT_SIDE / max(height, width)
    work_w = max(8, int(round(width * scale)))
    work_h = max(8, int(round(height * scale)))
    if (work_w, work_h) != (width, height):
        roi_rgb = cv2.resize(roi_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        roi_mask = cv2.resize(roi_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    pad_h = INPUT_SIDE - work_h
    pad_w = INPUT_SIDE - work_w
    if pad_h > 0 or pad_w > 0:
        roi_rgb = cv2.copyMakeBorder(roi_rgb, 0, max(0, pad_h), 0, max(0, pad_w), cv2.BORDER_REPLICATE)
        roi_mask = cv2.copyMakeBorder(
            roi_mask, 0, max(0, pad_h), 0, max(0, pad_w), cv2.BORDER_CONSTANT, value=0
        )
    image_input = roi_rgb[:INPUT_SIDE, :INPUT_SIDE].astype(np.float32) / 255.0
    image_input = np.transpose(image_input, (2, 0, 1))[None, ...]
    mask_input = (roi_mask[:INPUT_SIDE, :INPUT_SIDE] > 0).astype(np.float32)[None, None, ...]
    outputs = session.run(None, {"image": image_input, "mask": mask_input})
    result = np.asarray(outputs[0])
    if result.ndim == 4:
        result = result[0]
    result = np.transpose(result, (1, 2, 0))
    if result.max() <= 1.5:
        result = result * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = result[:work_h, :work_w]
    if (work_w, work_h) != (width, height):
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    return result


def _tile_starts(start: int, end: int, tile: int, stride: int, limit: int) -> list[int]:
    """在 [start, end) 上排布边长 tile 的窗口起点，保证最后一块贴住 end。"""
    last = end - tile
    if last <= start:
        return [min(max(0, start), max(0, limit - tile))]
    starts = list(range(start, last, stride))
    starts.append(last)
    return [min(max(0, value), max(0, limit - tile)) for value in starts]


def _tile_weight(height: int, width: int) -> np.ndarray:
    """离块边越远权重越高：重叠区由「周围上下文更多」的那一块说了算。

    下限 0.05 保证贴画面边的像素也一定被写满（边缘块的权重正好是 0）。
    """
    ramp = max(1, TILE_OVERLAP)

    def axis(length: int) -> np.ndarray:
        positions = np.arange(length, dtype=np.float32)
        distance = np.minimum(positions, length - 1 - positions)
        return np.clip(distance / ramp, 0.0, 1.0)

    weight = axis(height)[:, None] * axis(width)[None, :]
    return (0.05 + 0.95 * weight)[..., None]


def _run_tiled(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """大 ROI 分块推理：每块按原生分辨率送进 512 模型，重叠区按权重融合。"""
    height, width = roi_mask.shape[:2]
    rows, cols = np.nonzero(roi_mask > 0)
    if rows.size == 0:
        return roi_rgb
    mx0, mx1 = int(cols.min()), int(cols.max()) + 1
    my0, my1 = int(rows.min()), int(rows.max()) + 1
    area_x0, area_x1 = max(0, mx0 - TILE_CONTEXT), min(width, mx1 + TILE_CONTEXT)
    area_y0, area_y1 = max(0, my0 - TILE_CONTEXT), min(height, my1 + TILE_CONTEXT)
    stride = INPUT_SIDE - TILE_OVERLAP
    tiled = roi_rgb.astype(np.float32)
    accumulated = np.zeros_like(tiled)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    for y0 in _tile_starts(area_y0, area_y1, INPUT_SIDE, stride, height):
        for x0 in _tile_starts(area_x0, area_x1, INPUT_SIDE, stride, width):
            y1 = min(height, y0 + INPUT_SIDE)
            x1 = min(width, x0 + INPUT_SIDE)
            tile_mask = roi_mask[y0:y1, x0:x1]
            if tile_mask.max() == 0:
                continue
            filled = _run_model(roi_rgb[y0:y1, x0:x1], tile_mask).astype(np.float32)
            weight = _tile_weight(y1 - y0, x1 - x0) * (tile_mask > 0)[..., None]
            accumulated[y0:y1, x0:x1] += filled * weight
            weights[y0:y1, x0:x1] += weight
    covered = weights[..., 0] > 0
    if covered.any():
        colours = accumulated[covered] / weights[covered]
        tiled[covered] = np.clip(colours, 0, 255)
    return tiled.astype(np.uint8)


def _roi_for(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    margin = max(CONTEXT_MIN, int(max(box.width, box.height) * CONTEXT_RATIO))
    x0 = max(0, box.x0 - margin)
    y0 = max(0, box.y0 - margin)
    x1 = min(width, box.x1 + margin)
    y1 = min(height, box.y1 + margin)
    longest = max(x1 - x0, y1 - y0)
    if longest > MAX_ROI:
        scale = MAX_ROI / longest
        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
        half_w = int(round((x1 - x0) * scale / 2))
        half_h = int(round((y1 - y0) * scale / 2))
        x0 = max(0, center_x - half_w)
        y0 = max(0, center_y - half_h)
        x1 = min(width, center_x + half_w)
        y1 = min(height, center_y + half_h)
    return x0, y0, x1, y1


def inpaint_boxes(rgb: np.ndarray, boxes: list[Box]) -> dict:
    """按框重绘并只回贴掩膜区域，返回与图片引擎同口径的统计。"""
    if not boxes:
        return {"fallback_regions": 0, "directions": [], "masked_regions": 0, "unblend_regions": 0}
    _load_session()
    height, width = rgb.shape[:2]
    repaired = 0
    tiled_regions = 0
    for box in boxes:
        x0, y0, x1, y1 = _roi_for(box, width, height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        roi = rgb[y0:y1, x0:x1, :]
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        bx0 = max(0, box.x0 - x0)
        by0 = max(0, box.y0 - y0)
        bx1 = min(x1 - x0, box.x1 - x0)
        by1 = min(y1 - y0, box.y1 - y0)
        mask[by0:by1, bx0:bx1] = 255
        if mask.max() == 0:
            continue
        if max(mask.shape[:2]) > TILE_TRIGGER:
            filled = _run_tiled(roi, mask)
            tiled_regions += 1
        else:
            filled = _run_model(roi, mask)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (FEATHER * 2 + 1, FEATHER * 2 + 1), 0)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        blended = filled.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
        roi[...] = np.clip(blended, 0, 255).astype(np.uint8)
        repaired += 1
    return {
        "fallback_regions": 0,
        "directions": [],
        "masked_regions": repaired,
        "unblend_regions": 0,
        "photo_regions": repaired,
        "tiled_regions": tiled_regions,
    }
