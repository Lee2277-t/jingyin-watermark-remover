# -*- coding: utf-8 -*-
"""低对比度水印候选检测（第十二轮已接入 detect_watermark，只出疑似的虚线框）。

背景：现有检测（detection.py）只覆盖两类——彩色半透明水印（HSV 主色相）
与四角亮白字（medianBlur 差值）。漏检场景：与背景同色相的低对比平铺字样、
灰色半透明大章、深色 logo。本模块补三条「低对比」候选路线，**只做候选提示**
（low_confidence=True，让用户手动确认），不自动决定修复框，避免在彩色立绘上
误框（上次误框的教训）。

三条路线（按证据强度排序）：
- tiled：整页斜向平铺。中值估背景后取窄偏差带，灰度开运算滤掉 1px 的
  正文/JPEG 彩边尖峰（2px 以上的平铺笔画保留），FFT 自相关取四个象限的最强
  非轴向平移——正文行距/字距只有轴向周期，被非轴向条件拦掉；四个象限的向量
  逐个试算周期支撑掩膜（±平移都要有像素），取能长出最多平铺块的那个。支撑块
  即候选（最多 8 个）。
- stamp：灰色半透明大章。整页纸面亮度（灰度 90 分位）为基准，找「成片、均匀」
  地比纸暗 8–45 级的区域——大章比任何局部中值核都大，中值估背景在章内部恒为
  0、只剩一圈 2px 的翻转条纹（核 29/61 实测都测不到）。开运算去字边、5×5 闭
  运算桥接正文线造成的断口；整页带内占比过高或合格块过多（纹理/渐变/噪声）
  直接放弃。
- corner-dark：深色角落 logo。现有四角路线只找「比背景亮」，这条镜像找
  「比背景暗」，其余判定与亮路线一致。

已知边界：漫画格线、砖墙等本身就有二维周期的规律纹理可能误报，所以本模块
永远只出 low_confidence 候选，不自动加入修复框。

接线方式（第十二轮）：
1. `detection._suggestion_boxes(rgb, regions)` 调 `low_contrast_regions(rgb)`，转成归一化
   `{x,y,w,h,kind,label,score,low_confidence:True}`；与正式修复框按「交叠÷较小者面积
   > 0.35」去重，候选之间同样去重，最多留 6 条。
2. `detect_watermark` 把它们放在 `suggestions` 字段单独下发，**不进 `regions`**：
   前端画琥珀色虚线框 + 「疑似」徽标，用户点「采纳」才变成正式选框（受 16 框上限约束），
   且不复用 localStorage，刷新即消失。
3. 阈值用合成夹具标定（六类夹具：平铺/灰章/深角标命中，纯文字页/渐变图/噪声图零误报），
   契约回归在 `work/qa/_test_lowcontrast.py`。**仍未拿用户的真实低对比素材复核过**——
   上线后第一次遇到这类素材应回头核一遍阈值。
"""

from __future__ import annotations

import cv2
import numpy as np

# —— 偏差带（亮/暗两侧共用）——
DEV_LO = 3.0          # 偏差下限：低于此视为纸面纹理/噪声
DEV_HI = 26.0         # 偏差上限：高于此是深色正文或高对比水印，归现有路线管

# —— 平铺周期 ——
LATTICE_MIN_FRAC = 0.05     # 平移向量每轴至少 5% 边长（拦文字行距谐波 dx≈4%·宽）
LATTICE_MAX_FRAC = 0.45     # 平移向量每轴至多 45% 边长
TILE_MIN_COVERAGE = 0.22    # 平移向量覆盖的掩膜占比下限（A(v)/mass）
TILE_COVERAGE_RATIO = 2.5   # 峰值须 ≥ 2.5 × 随机互相关期望（掩膜占比 f）
TILE_MIN_COMPONENTS = 2     # 至少两个周期支撑连通块才算平铺
TILE_MIN_AREA_FRAC = 0.00015
TILE_MAX_BBOX_FRAC = 0.06   # 单块包围盒面积上限（拦长对角线/格线）
TILE_MAX_ASPECT = 8.0

# —— 灰色半透明大章 ——
STAMP_SIDE = 320            # 独立工作分辨率（大章分块与纸面亮度统计都在这里做）
STAMP_PAPER_PCT = 90.0      # 纸面亮度取灰度 90 分位（纸占多数的文档才成立）
STAMP_BAND_LO = 8.0
STAMP_BAND_HI = 45.0
STAMP_MIN_AREA_FRAC = 0.0008
STAMP_MAX_AREA_FRAC = 0.15
STAMP_MIN_DENSITY = 0.35    # 成片的章填充包围盒；细长文字/格线密度低
STAMP_MAX_STD = 9.0         # 带内亮度标准差：半透明章均匀，噪声/渐变起伏大
STAMP_MAX_PAGE_BAND = 0.15  # 整页带内像素占比上限（超过＝整页纹理/渐变，不是章）
STAMP_MAX_OBJECTS = 6       # 合格块数上限（超过＝纹理碎片，不是章）
STAMP_BORDER_FRAC = 0.02    # 连通块距页面边至少 2% 边长（排除页边阴影/黑边）

# —— 深色角落 logo ——
CORNER_DEV = -6.0           # 比中值背景暗 6 级以上
CORNER_VALUE_MAX = 175
CORNER_SAT_MAX = 110

MAX_SIDE = 960              # 内部处理分辨率上限（FFT/中值成本封顶）
MAX_CANDIDATES = 8


def _bright_deviation(small: np.ndarray) -> np.ndarray:
    """亮侧偏差图，灰度开运算滤掉 1px 尖峰（字边、JPEG 彩边）。"""
    sh, sw = small.shape[:2]
    kernel = max(9, min(31, (min(sh, sw) // 25) | 1))
    background = cv2.medianBlur(small, kernel).astype(np.float32)
    diff = small.astype(np.float32) - background
    bright = np.abs(diff).max(axis=2)
    return cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _linear_autocorr(band: np.ndarray) -> np.ndarray:
    """零填充到 2 倍尺寸的线性自相关：ac[ty,tx] = 平移 (dy,dx) 的重合像素对数。"""
    sh, sw = band.shape
    f = np.fft.rfft2(band.astype(np.float32), s=(sh * 2, sw * 2))
    ac = np.fft.irfft2(f * np.conj(f), s=(sh * 2, sw * 2)).real
    np.maximum(ac, 0.0, out=ac)
    return ac


def _quadrant_peaks(ac: np.ndarray, sh: int, sw: int) -> list[tuple[int, int, float]]:
    """四个象限各取最强峰，返回 (dy, dx, 重合对数)；只搜非轴向平移。"""
    dy_lo = max(2, int(sh * LATTICE_MIN_FRAC))
    dy_hi = min(sh - 1, int(sh * LATTICE_MAX_FRAC))
    dx_lo = max(2, int(sw * LATTICE_MIN_FRAC))
    dx_hi = min(sw - 1, int(sw * LATTICE_MAX_FRAC))
    quads = [
        (slice(dy_lo, dy_hi), slice(dx_lo, dx_hi)),
        (slice(dy_lo, dy_hi), slice(2 * sw - dx_hi, 2 * sw - dx_lo)),
        (slice(2 * sh - dy_hi, 2 * sh - dy_lo), slice(dx_lo, dx_hi)),
        (slice(2 * sh - dy_hi, 2 * sh - dy_lo), slice(2 * sw - dx_hi, 2 * sw - dx_lo)),
    ]
    peaks: list[tuple[int, int, float]] = []
    for ys, xs in quads:
        if ys.start >= ys.stop or xs.start >= xs.stop:
            continue
        sub = ac[ys, xs]
        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
        ty, tx = ys.start + iy, xs.start + ix
        dy = ty - 2 * sh if ty >= sh else ty
        dx = tx - 2 * sw if tx >= sw else tx
        peaks.append((dy, dx, float(ac[ty, tx])))
    return peaks


def _intersect_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """mask 与其平移 (dy,dx) 副本的交集，落在「两端」两片区域上。"""
    sh, sw = mask.shape
    out = np.zeros_like(mask)
    h, w = sh - abs(dy), sw - abs(dx)
    if h <= 0 or w <= 0:
        return out
    y0a, y0b = max(0, -dy), max(0, dy)
    x0a, x0b = max(0, -dx), max(0, dx)
    overlap = mask[y0a:y0a + h, x0a:x0a + w] & mask[y0b:y0b + h, x0b:x0b + w]
    out[y0a:y0a + h, x0a:x0a + w] = overlap
    out[y0b:y0b + h, x0b:x0b + w] |= overlap
    return out


def _periodic_blocks(periodic: np.ndarray, page_area: int) -> list[tuple[int, int, int, int]]:
    """周期支撑掩膜的合格平铺块（面积 / 包围盒 / 长宽比三重过滤）。"""
    count, _, stats, _ = cv2.connectedComponentsWithStats(periodic * 255, 8)
    blocks: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area < TILE_MIN_AREA_FRAC * page_area:
            continue
        if bw * bh > TILE_MAX_BBOX_FRAC * page_area:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > TILE_MAX_ASPECT:
            continue
        blocks.append((int(x), int(y), int(x + bw), int(y + bh)))
    return blocks


def _tiled_candidates(dev: np.ndarray) -> list[dict]:
    sh, sw = dev.shape
    if sh < 120 or sw < 120:
        return []
    band = ((dev >= DEV_LO) & (dev <= DEV_HI)).astype(np.uint8)
    mass = int(band.sum())
    if mass < 200:
        return []
    frac = mass / (sh * sw)
    if frac > 0.30:
        return []
    ac = _linear_autocorr(band)
    peaks = _quadrant_peaks(ac, sh, sw)
    if not peaks:
        return []
    peaks.sort(key=lambda p: -p[2])
    page_area = sh * sw
    # 四个象限的峰里可能有文字行距谐波压过真周期，逐个试算、取能长出最多块者
    tried: set[tuple[int, int]] = set()
    best_blocks: list[tuple[int, int, int, int]] = []
    for cand_dy, cand_dx, cand_value in peaks:
        if (cand_dy, cand_dx) in tried or (-cand_dy, -cand_dx) in tried:
            continue
        tried.add((cand_dy, cand_dx))
        norm_v = cand_value / mass
        if norm_v < TILE_MIN_COVERAGE or norm_v < TILE_COVERAGE_RATIO * frac:
            continue
        periodic = _intersect_shift(band, cand_dy, cand_dx)
        periodic = _intersect_shift(periodic, -cand_dy, -cand_dx)
        blocks = _periodic_blocks(periodic, page_area)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
    if len(best_blocks) < TILE_MIN_COMPONENTS:
        return []
    best_blocks.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    candidates: list[dict] = []
    for bx0, by0, bx1, by1 in best_blocks[:MAX_CANDIDATES]:
        candidates.append(
            {
                "x": round(bx0 / sw, 5),
                "y": round(by0 / sh, 5),
                "w": round((bx1 - bx0) / sw, 5),
                "h": round((by1 - by0) / sh, 5),
                "low_confidence": True,
                "kind": "tiled",
                "score": round(((bx1 - bx0) * (by1 - by0)) / page_area, 5),
            }
        )
    return candidates


def _stamp_candidates(rgb: np.ndarray) -> list[dict]:
    """灰色半透明大章：以纸面亮度为基准找「成片均匀变暗」的区域。"""
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, STAMP_SIDE / max(h0, w0))
    if scale < 1.0:
        small = cv2.resize(
            rgb, (max(1, int(w0 * scale)), max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        small = rgb
    sh, sw = small.shape[:2]
    if sh < 80 or sw < 80:
        return []
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    paper = float(np.percentile(gray, STAMP_PAPER_PCT))
    solid = paper - gray.astype(np.float32)
    band0 = ((solid >= STAMP_BAND_LO) & (solid <= STAMP_BAND_HI)).astype(np.uint8)
    page_area = sh * sw
    if float(band0.mean()) > STAMP_MAX_PAGE_BAND:
        return []
    band = cv2.morphologyEx(band0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 正文线穿过章时带被打断，5×5 闭运算桥接 ≤4px 的缺口
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    border = max(2, int(min(sh, sw) * STAMP_BORDER_FRAC))
    band[:border, :] = 0
    band[-border:, :] = 0
    band[:, :border] = 0
    band[:, -border:] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(band * 255, 8)
    candidates: list[dict] = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        area_frac = area / page_area
        if not STAMP_MIN_AREA_FRAC <= area_frac <= STAMP_MAX_AREA_FRAC:
            continue
        if bw > sw * 0.5 or bh > sh * 0.5:
            continue
        if area / max(1, bw * bh) < STAMP_MIN_DENSITY:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 6:
            continue
        values = solid[y:y + bh, x:x + bw][band0[y:y + bh, x:x + bw] > 0]
        if values.size < 60 or float(values.std()) > STAMP_MAX_STD:
            continue
        candidates.append(
            {
                "x": round(x / sw, 5),
                "y": round(y / sh, 5),
                "w": round(bw / sw, 5),
                "h": round(bh / sh, 5),
                "low_confidence": True,
                "kind": "stamp",
                "score": round(area_frac, 5),
            }
        )
    if len(candidates) > STAMP_MAX_OBJECTS:
        return []
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:4]


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


def _dark_corner_candidates(small: np.ndarray) -> list[dict]:
    height, width = small.shape[:2]
    if width < 120 or height < 120:
        return []
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    kernel_size = int(min(width, height) * 0.018)
    kernel_size = max(11, min(31, kernel_size | 1))
    background = cv2.medianBlur(gray, kernel_size)
    difference = gray.astype(np.int16) - background.astype(np.int16)
    dark_text = (
        (difference <= CORNER_DEV) & (value <= CORNER_VALUE_MAX) & (saturation <= CORNER_SAT_MAX)
    ).astype(np.uint8) * 255
    dark_text = cv2.morphologyEx(dark_text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark_text = cv2.morphologyEx(dark_text, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=1)

    candidates: list[tuple[int, int, int, int, str, float]] = []
    for corner in ("br", "bl", "tr", "tl"):
        rx0, ry0, rx1, ry1 = _corner_roi(width, height, corner)
        mask = dark_text[ry0:ry1, rx0:rx1]
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
            roi_components.append((rx0 + int(x), ry0 + int(y), rx0 + int(x) + int(w), ry0 + int(y) + int(h)))
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
            pixel_count = int(np.count_nonzero(dark_text[cy0:cy1, cx0:cx1]))
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
            if component_count < 2 and (box_width < width * 0.035 or density < 0.14):
                continue
            score = _score_cluster(
                (cx0, cy0, cx1, cy1), pixel_count, component_count, width, height, corner
            )
            candidates.append((cx0, cy0, cx1, cy1, corner, score))
    if not candidates:
        return []
    best = max(candidates, key=lambda c: c[5])
    same_corner = [c for c in candidates if c[4] == best[4]]
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = best[0], best[1], best[2], best[3]
    merge_x = int(width * 0.025)
    merge_y = int(height * 0.035)
    for item in same_corner:
        if item is best:
            continue
        if not (
            item[2] + merge_x < bbox_x0
            or bbox_x1 + merge_x < item[0]
            or item[3] + merge_y < bbox_y0
            or bbox_y1 + merge_y < item[1]
        ):
            bbox_x0 = min(bbox_x0, item[0])
            bbox_y0 = min(bbox_y0, item[1])
            bbox_x1 = max(bbox_x1, item[2])
            bbox_y1 = max(bbox_y1, item[3])
    return [
        {
            "x": round(bbox_x0 / width, 5),
            "y": round(bbox_y0 / height, 5),
            "w": round((bbox_x1 - bbox_x0) / width, 5),
            "h": round((bbox_y1 - bbox_y0) / height, 5),
            "low_confidence": True,
            "kind": "corner-dark",
            "score": round(best[5], 3),
        }
    ]


def low_contrast_regions(rgb: np.ndarray) -> list[dict]:
    """找出低对比水印候选，返回归一化候选框列表。

    返回 [{"x","y","w","h","low_confidence":True,"kind","score"}]，
    kind ∈ {"tiled","stamp","corner-dark"}；只做提示，不自动加入修复框。
    """
    height, width = rgb.shape[:2]
    if width < 160 or height < 160:
        return []
    scale = min(1.0, MAX_SIDE / max(width, height))
    if scale < 1.0:
        small = cv2.resize(
            rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        small = rgb
    bright = _bright_deviation(small)
    candidates: list[dict] = []
    candidates.extend(_tiled_candidates(bright))
    candidates.extend(_stamp_candidates(rgb))
    candidates.extend(_dark_corner_candidates(small))
    return candidates[:MAX_CANDIDATES]


if __name__ == "__main__":
    # 调试入口：给一张图路径，打印候选框
    import sys

    for path in sys.argv[1:]:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        print(path, low_contrast_regions(image))
