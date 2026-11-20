from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from remover.detection_lowcontrast import low_contrast_regions

log = logging.getLogger(__name__)

CORNER_LABELS = {
    "br": "右下角",
    "bl": "左下角",
    "tr": "右上角",
    "tl": "左上角",
}

SUGGESTION_LABELS = {
    "tiled": "整页平铺",
    "stamp": "较暗印章",
    "corner-dark": "角落深色标",
}
MAX_SUGGESTIONS = 6
SUGGESTION_OVERLAP = 0.35


@dataclass
class Candidate:
    x0: int
    y0: int
    x1: int
    y1: int
    corner: str
    boxes: int
    pixels: int
    score: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _corner_roi(width: int, height: int, corner: str) -> tuple[int, int, int, int]:
    roi_width = max(60, int(width * 0.38))
    roi_height = max(45, int(height * 0.24))
    if corner in ("br", "tr"):
        x0 = width - roi_width
    else:
        x0 = 0
    if corner in ("br", "bl"):
        y0 = height - roi_height
    else:
        y0 = 0
    return x0, y0, x0 + roi_width, y0 + roi_height


def _near_corner(box: tuple[int, int, int, int], width: int, height: int, corner: str) -> bool:
    x0, y0, x1, y1 = box
    margin_x = max(18, int(width * 0.13))
    margin_y = max(14, int(height * 0.12))
    if corner == "br":
        return x1 >= width - margin_x and y1 >= height - margin_y
    if corner == "bl":
        return x0 <= margin_x and y1 >= height - margin_y
    if corner == "tr":
        return x1 >= width - margin_x and y0 <= margin_y
    return x0 <= margin_x and y0 <= margin_y


def _cluster_boxes(boxes: list[tuple[int, int, int, int]], gap: int) -> list[list[int]]:
    clusters: list[list[int]] = []
    for x0, y0, x1, y1 in boxes:
        merged = [x0, y0, x1, y1]
        remaining: list[list[int]] = []
        for cluster in clusters:
            if not (
                merged[2] + gap < cluster[0]
                or cluster[2] + gap < merged[0]
                or merged[3] + gap < cluster[1]
                or cluster[3] + gap < merged[1]
            ):
                merged[0] = min(merged[0], cluster[0])
                merged[1] = min(merged[1], cluster[1])
                merged[2] = max(merged[2], cluster[2])
                merged[3] = max(merged[3], cluster[3])
            else:
                remaining.append(cluster)
        remaining.append(merged)
        clusters = remaining
    return clusters


def _score_cluster(
    bbox: tuple[int, int, int, int],
    pixel_count: int,
    component_count: int,
    width: int,
    height: int,
    corner: str,
) -> float:
    x0, y0, x1, y1 = bbox
    if corner == "br":
        gap_x = width - x1
        gap_y = height - y1
    elif corner == "bl":
        gap_x = x0
        gap_y = height - y1
    elif corner == "tr":
        gap_x = width - x1
        gap_y = y0
    else:
        gap_x = x0
        gap_y = y0
    proximity = max(0.0, 1.0 - (gap_x / max(1.0, width * 0.13) + gap_y / max(1.0, height * 0.12)) / 2.0)
    density = pixel_count / max(1, (x1 - x0) * (y1 - y0))
    return pixel_count * 0.012 + component_count * 2.8 + proximity * 18.0 + density * 12.0


def _dominant_hue(hsv: np.ndarray, eligible: np.ndarray) -> tuple[int, float]:
    values = hsv[..., 0][eligible].astype(np.int32)
    if values.size == 0:
        return -1, 0.0
    hist = np.bincount(values, minlength=180).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return -1, 0.0
    extended = np.concatenate([hist[-9:], hist, hist[:9]])
    smoothed = np.convolve(extended, np.ones(19), mode="same")[9:-9]
    peak = int(np.argmax(smoothed))
    return peak, float(smoothed[peak] / total)


def _chromatic_regions(rgb: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """全画面搜索有色半透明水印（蓝字、红章、彩色 logo 等），返回 (x0,y0,x1,y1,像素数)。"""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    f = rgb.astype(np.float32)
    lum = f.mean(axis=2)
    chroma_mag = np.abs(f - lum[..., None]).max(axis=2)
    saturated = (sat >= 45) & (val >= 38) & (chroma_mag >= 12)
    total = int(saturated.sum())
    if total < max(260, int(width * height * 0.00004)):
        return []
    if total > width * height * 0.45:
        return []
    hue, share = _dominant_hue(hsv, saturated)
    if hue < 0 or share < 0.24:
        return []
    dist = np.abs(hsv[..., 0] - hue)
    dist = np.minimum(dist, 180.0 - dist)
    mask = (saturated & (dist <= 16)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    merge = max(3, int(min(width, height) * 0.01)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((merge, merge), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(14, int(width * height * 0.000012))
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        if w > width * 0.9 or h > height * 0.9:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    if not boxes:
        return []
    gap = max(8, int(min(width, height) * 0.012))
    regions: list[tuple[int, int, int, int, int]] = []
    for cluster in _cluster_boxes(boxes, gap):
        cx0, cy0, cx1, cy1 = cluster
        cx0 = max(0, cx0)
        cy0 = max(0, cy0)
        cx1 = min(width, cx1)
        cy1 = min(height, cy1)
        box_area = max(1, (cx1 - cx0) * (cy1 - cy0))
        pixels = int(np.count_nonzero(mask[cy0:cy1, cx0:cx1]))
        density = pixels / box_area
        if pixels < 150:
            continue
        if not 0.012 <= density <= 0.62:
            continue
        if box_area > width * height * 0.55:
            continue
        regions.append((cx0, cy0, cx1, cy1, pixels))
    regions.sort(key=lambda item: -item[4])
    return regions[:10]


def _corner_regions(
    rgb: np.ndarray,
    only: str | None = None,
    single_min_width_share: float = 0.035,
    require_lift: float | None = None,
) -> list[Candidate]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    kernel_size = int(min(width, height) * 0.018)
    kernel_size = max(11, min(31, kernel_size | 1))
    background = cv2.medianBlur(gray, kernel_size)
    difference = gray.astype(np.int16) - background.astype(np.int16)

    bright_text = (
        (difference >= 5)
        & (value >= 125)
        & (saturation <= 72)
    ).astype(np.uint8) * 255
    bright_text = cv2.morphologyEx(bright_text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bright_text = cv2.morphologyEx(
        bright_text, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=1
    )

    components: list[tuple[int, int, int, int]] = []
    candidates: list[Candidate] = []

    corners = ("br", "bl", "tr", "tl") if only is None else (only,)
    for corner in corners:
        rx0, ry0, rx1, ry1 = _corner_roi(width, height, corner)
        mask = bright_text[ry0:ry1, rx0:rx1]
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        roi_components: list[tuple[int, int, int, int]] = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if w < max(7, int(width * 0.004)) or h < max(4, int(height * 0.006)):
                continue
            if w > width * 0.34 or h > height * 0.16:
                continue
            if area < 8 or area / max(1, w * h) < 0.06:
                continue
            gx0, gy0 = rx0 + int(x), ry0 + int(y)
            gx1, gy1 = gx0 + int(w), gy0 + int(h)
            global_box = (gx0, gy0, gx1, gy1)
            roi_components.append(global_box)
            components.append(global_box)

        if not roi_components:
            continue
        gap = max(8, int(min(width, height) * 0.018))
        for cluster in _cluster_boxes(roi_components, gap):
            cx0, cy0, cx1, cy1 = cluster
            box_width = cx1 - cx0
            box_height = cy1 - cy0
            if box_width < max(12, int(width * 0.012)) or box_height < max(7, int(height * 0.01)):
                continue
            if box_width > width * 0.4 or box_height > height * 0.2:
                continue
            if box_width / max(1, box_height) < 1.05:
                continue
            if not _near_corner((cx0, cy0, cx1, cy1), width, height, corner):
                continue
            mask_cluster = bright_text[cy0:cy1, cx0:cx1]
            pixel_count = int(np.count_nonzero(mask_cluster))
            if pixel_count < 24:
                continue
            density = pixel_count / max(1, box_width * box_height)
            if density < 0.035:
                continue
            component_count = sum(
                1
                for bx0, by0, bx1, by1 in roi_components
                if bx0 >= cx0 - 2 and by0 >= cy0 - 2 and bx1 <= cx1 + 2 and by1 <= cy1 + 2
            )
            if component_count < 2 and (box_width < width * single_min_width_share or density < 0.14):
                continue
            if require_lift is not None:
                # 虚焦高光的边缘也过得了上面所有尺寸门槛，但它只比背景亮 10 级左右（实测 320×176
                # 放大 6 倍的正片左下角出过 1 框），真台标是实打实的白字（实测亮 140 级）。
                lift = difference[cy0:cy1, cx0:cx1][bright_text[cy0:cy1, cx0:cx1] > 0]
                if lift.size == 0 or float(lift.mean()) < require_lift:
                    continue
            score = _score_cluster(
                (cx0, cy0, cx1, cy1), pixel_count, component_count, width, height, corner
            )
            candidates.append(
                Candidate(cx0, cy0, cx1, cy1, corner, component_count, pixel_count, score)
            )

    if not candidates:
        return []
    best = max(candidates, key=lambda item: item.score)
    same_corner = [item for item in candidates if item.corner == best.corner]
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = best.x0, best.y0, best.x1, best.y1
    merge_x = int(width * 0.025)
    merge_y = int(height * 0.035)
    for item in same_corner:
        if item is best:
            continue
        if not (
            item.x1 + merge_x < bbox_x0
            or bbox_x1 + merge_x < item.x0
            or item.y1 + merge_y < bbox_y0
            or bbox_y1 + merge_y < item.y0
        ):
            bbox_x0 = min(bbox_x0, item.x0)
            bbox_y0 = min(bbox_y0, item.y0)
            bbox_x1 = max(bbox_x1, item.x1)
            bbox_y1 = max(bbox_y1, item.y1)
    best.x0, best.y0, best.x1, best.y1 = bbox_x0, bbox_y0, bbox_x1, bbox_y1
    return [best]


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return False
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return ix * iy > smaller * 0.55


def _overlap_share(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return ix * iy / max(1, smaller)


def _suggestion_boxes(
    rgb: np.ndarray, accepted: list[dict]
) -> list[dict]:
    """低对比候选：只提示、不当修复框；与已确定的修复框和彼此之间去重。"""
    try:
        candidates = low_contrast_regions(rgb)
    except Exception:
        log.warning("low contrast detection failed", exc_info=True)
        return []
    if not candidates:
        return []
    height, width = rgb.shape[:2]
    taken = [
        (
            int(round(item["x"] * width)),
            int(round(item["y"] * height)),
            int(round((item["x"] + item["w"]) * width)),
            int(round((item["y"] + item["h"]) * height)),
        )
        for item in accepted
    ]
    kept: list[dict] = []
    for cand in candidates:
        box = (
            int(round(cand["x"] * width)),
            int(round(cand["y"] * height)),
            int(round((cand["x"] + cand["w"]) * width)),
            int(round((cand["y"] + cand["h"]) * height)),
        )
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            continue
        if any(_overlap_share(box, other) > SUGGESTION_OVERLAP for other in taken):
            continue
        kept.append(cand)
        taken.append(box)
        if len(kept) >= MAX_SUGGESTIONS:
            break
    return [
        {
            "x": cand["x"],
            "y": cand["y"],
            "w": cand["w"],
            "h": cand["h"],
            "kind": cand["kind"],
            "label": SUGGESTION_LABELS.get(cand["kind"], "低对比水印"),
            "score": cand["score"],
            "low_confidence": True,
        }
        for cand in kept
    ]


def _pad(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    pad_x = max(8, int(width * 0.009))
    pad_y = max(7, int(height * 0.011))
    x0, y0, x1, y1 = box
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)


NATIVE_CORNER_MIN_LIFT = 22.0


def corner_regions_native(rgb: np.ndarray) -> list[dict]:
    """按原生分辨率逐个角 ROI 找角标，返回归一化修复框。

    高分辨率视频会先把长边压到 1280 再检测：4K 上 1.5–3% 画面高的台标压完只剩 8–16px，
    角标那几道尺寸门槛全过不去（实测 4K 24px 台标在 1280 口径 0 框，原生角裁剪 1 框）。
    这里只在四个角 38%×24% 的 ROI 里按原尺寸再找一遍，给 `_detect_video` 补漏用。
    """
    height, width = rgb.shape[:2]
    found: list[dict] = []
    for corner in CORNER_LABELS:
        x0, y0, x1, y1 = _corner_roi(width, height, corner)
        crop = rgb[y0:y1, x0:x1]
        # 只查这块裁剪自己的那个角：`_corner_regions` 默认会扫传入图像的四个角，
        # 那个「内角」落在整帧中间，会把画面中央的正常内容当角标（实测 bbb 上出过 1 框）。
        # 放宽单连通域宽度（0.035→0.02）是因为补漏的对象本来就小：4K 上 24px 高的「AI生成」
        # 四个字经闭运算会连成一个 46px 宽的块，只有 3.15% 的裁剪宽，按原门槛永远补不回来。
        for item in _corner_regions(
            crop,
            only=corner,
            single_min_width_share=0.02,
            require_lift=NATIVE_CORNER_MIN_LIFT,
        ):
            box = _pad((x0 + item.x0, y0 + item.y0, x0 + item.x1, y0 + item.y1), width, height)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            found.append(
                {
                    "x": round(box[0] / width, 5),
                    "y": round(box[1] / height, 5),
                    "w": round((box[2] - box[0]) / width, 5),
                    "h": round((box[3] - box[1]) / height, 5),
                }
            )
    return found


def detect_watermark(rgb: np.ndarray) -> dict:
    height, width = rgb.shape[:2]
    if width < 60 or height < 60:
        return {
            "found": False,
            "regions": [],
            "suggestions": [],
            "message": "图片太小，请手动框选水印",
        }

    found: list[tuple[int, int, int, int, str]] = []
    for cx0, cy0, cx1, cy1, _pixels in _chromatic_regions(rgb):
        found.append((cx0, cy0, cx1, cy1, "彩色"))
    for item in _corner_regions(rgb):
        box = (item.x0, item.y0, item.x1, item.y1)
        if not any(_overlaps(box, (a, b, c, d)) for a, b, c, d, _ in found):
            found.append((item.x0, item.y0, item.x1, item.y1, CORNER_LABELS[item.corner]))

    regions = []
    labels = []
    for x0, y0, x1, y1, label in found[:12]:
        px0, py0, px1, py1 = _pad((x0, y0, x1, y1), width, height)
        if (px1 - px0) < 4 or (py1 - py0) < 4:
            continue
        regions.append(
            {
                "x": round(px0 / width, 5),
                "y": round(py0 / height, 5),
                "w": round((px1 - px0) / width, 5),
                "h": round((py1 - py0) / height, 5),
            }
        )
        labels.append(label)

    suggestions = _suggestion_boxes(rgb, regions)
    extra = f"；另有 {len(suggestions)} 处低对比疑似位置，请核对虚线框" if suggestions else ""

    if not regions:
        if suggestions:
            message = f"未自动找到明显水印，另有 {len(suggestions)} 处低对比疑似位置（虚线框），请核对"
        else:
            message = "未自动找到明显水印，请直接拖动鼠标框选（可框多个）"
        return {
            "found": False,
            "regions": [],
            "suggestions": suggestions,
            "message": message,
        }
    if len(regions) == 1:
        message = f"在{labels[0]}找到疑似水印，请核对选框位置"
    elif labels.count("彩色") >= 3:
        message = f"自动识别到 {len(regions)} 处彩色水印，请核对选框位置"
    else:
        message = f"自动识别到 {len(regions)} 处疑似水印，请核对选框位置"
    return {
        "found": True,
        "regions": regions,
        "suggestions": suggestions,
        "message": message + extra,
    }
