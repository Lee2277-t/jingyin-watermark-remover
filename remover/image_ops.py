from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import options


class ImageProcessError(Exception):
    pass


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _intersects(a: Box, b: Box, margin: int = 0) -> bool:
    return not (
        a.x1 + margin <= b.x0
        or b.x1 + margin <= a.x0
        or a.y1 + margin <= b.y0
        or b.y1 + margin <= a.y0
    )


def regions_to_boxes(regions: list[dict], width: int, height: int, padding: int) -> list[Box]:
    boxes: list[Box] = []
    for region in regions:
        try:
            rx = float(region["x"])
            ry = float(region["y"])
            rw = float(region["w"])
            rh = float(region["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (rx, ry, rw, rh)):
            continue
        rx = min(max(rx, 0.0), 1.0)
        ry = min(max(ry, 0.0), 1.0)
        rw = min(max(rw, 0.0), 1.0 - rx)
        rh = min(max(rh, 0.0), 1.0 - ry)
        if rw < 0.002 or rh < 0.002:
            continue
        x0 = max(0, math.floor(rx * width) - padding)
        y0 = max(0, math.floor(ry * height) - padding)
        x1 = min(width, math.ceil((rx + rw) * width) + padding)
        y1 = min(height, math.ceil((ry + rh) * height) + padding)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            boxes.append(Box(x0, y0, x1, y1))

    merged: list[Box] = []
    for box in sorted(boxes, key=lambda item: (item.y0, item.x0)):
        for index, existing in enumerate(merged):
            if _intersects(box, existing, margin=2):
                merged[index] = Box(
                    min(box.x0, existing.x0),
                    min(box.y0, existing.y0),
                    max(box.x1, existing.x1),
                    max(box.y1, existing.y1),
                )
                break
        else:
            merged.append(box)
    return merged


def _feather_alpha(height: int, width: int, box: Box, full_height: int, full_width: int) -> np.ndarray:
    feather = min(7, max(2, min(box.width, box.height) // 6))
    alpha_y = np.ones(height, dtype=np.float32)
    alpha_x = np.ones(width, dtype=np.float32)
    if box.y0 > 0:
        alpha_y[:feather] = np.linspace(0.0, 1.0, feather, endpoint=False)
    if box.y1 < full_height:
        alpha_y[-feather:] = np.linspace(1.0, 0.0, feather, endpoint=False)
    if box.x0 > 0:
        alpha_x[:feather] = np.linspace(0.0, 1.0, feather, endpoint=False)
    if box.x1 < full_width:
        alpha_x[-feather:] = np.linspace(1.0, 0.0, feather, endpoint=False)
    return np.outer(alpha_y, alpha_x)[..., None]


# 水印覆盖度判定带（相对值）：低于 LOW 视为纸面噪声，高于 HIGH 视为实心水印直接反解
WEIGHT_LOW = 0.10
WEIGHT_HIGH = 0.34
# 「这一带是白纸」的判据：中性暗像素（蓝水印本体够暗但偏蓝，先排除）膨胀成安全边距后，
# 再看窗口内还剩多少。用选区单独拟合时，边缘一圈按「不知道」处理
INK_DARK = 0.72
INK_CHROMA = 34.0
INK_DILATE = 13
INK_WIN = 21
INK_LO = 0.0
INK_HI = 0.02
INK_MARGIN = (INK_DILATE + INK_WIN) // 2
# 白纸上允许把反演补满的最低覆盖度
COV_LO = 0.02
COV_HI = 0.10
# 检测框之间的空隙里可能还压着同一支水印（对角印章会被拆成两段），只补框外的高覆盖核心。
# 正文的 JPEG 蓝边单像素覆盖率也能到 0.9，但都是 1–2px 细线，先做 3×3 开运算只会留下实心水印点
CORE_SWEEP = 0.30
CORE_SWEEP_GATE = 0.20
CORE_OPEN = 3
# 空白纸面上残留的同款水印点：左边距那些 4×18 的细竖条过不了 3×3 开运算，反演又只按
# 拟合覆盖度（0.38–0.55）洗掉一半，留下看得见的蓝点。这条不看形状，看「底下是不是空白纸」：
# 用最严的 on_paper（周边 10px 内没有任何中性正文）再加 31px 邻域的白占比，
# 两条都成立时底下的原色必然就是纸，直接归位到纸色不会连字一起洗
PAPER_DOT_ON_PAPER = 0.95
PAPER_DOT_LOCAL = 0.80
PAPER_DOT_WIN = 31
PAPER_DOT_LUM = 0.88
NOMINAL_COVERAGE = 0.90
# 水印盖住的字被染成蓝色，现有的「中性暗像素」判据认不出它，于是被当成白纸提亮，
# 表现是笔画变细、只剩橙色 JPEG 边。用「水印模型预期亮度 − 实际亮度」找这些被压暗的字：
# 预期亮度 = 纸面按覆盖度衰减后的亮度，观测明显更暗的地方就是字
TEXT_DEFICIT_LOW = 30.0
TEXT_DEFICIT_HIGH = 60.0
# 只在真正的水印足迹内动字：中值 7px 滤掉 1–2px 的字边（覆盖率同样很高），保留 6–10px 的笔画
TEXT_FOOT_MED = 7
TEXT_FOOT_LEVEL = 60
TEXT_FOOT_DILATE = 9
# 文字核心外 3px 内的暖色边（R>B）一并中性化，否则蓝边去掉后只剩橙边，看起来发红
TEXT_HALO_GROW = 3
TEXT_WARM_KNEE = 4.0
# 上面这条暖色晕要乘在水印足迹（7px 中值）上；平铺细水印的足迹会被中值滤成 0，
# 于是框内只洗掉了蓝、暖边留在字的另一侧，这片的字就比整页别处的字偏红。
# 这条不看足迹：从「已经被明显中和」的像素向外扩一圈，把那里的暖边一并归中性
WARM_HALO_GROW = 3
# 角标白字（比背景更亮的半透明白字，如「AI生成」）：按「比开运算估计的背景亮多少」找出来抬回背景。
# 背景必须用开运算估（中值会被密集笔画抬高，掩膜只剩 1px 骨架）；没有这条路线，
# 白字会落到邻域贴片路线，把周围的深色内容糊进水印区
BRIGHT_DEV_LOW = 6.0
BRIGHT_DEV_HIGH = 16.0
# 单个连通域不成章：真章是文字、至少两个笔画块；孤立实心块多半是画面里的彩色物件
# （立绘上的蓝花、眼睛等），反演会把物件本体洗成纸色
MIN_SPOTS = 2


def _smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (x >= high).astype(np.float32)
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _dominant_hue(hsv: np.ndarray, eligible: np.ndarray) -> tuple[int, float]:
    values = hsv[..., 0][eligible].astype(np.int32)
    if values.size == 0:
        return -1, 0.0
    hist = np.bincount(values, minlength=180).astype(np.float64)
    total = float(hist.sum())
    extended = np.concatenate([hist[-9:], hist, hist[:9]])
    smoothed = np.convolve(extended, np.ones(19), mode="same")[9:-9]
    if total <= 0:
        return -1, 0.0
    peak = int(np.argmax(smoothed))
    return peak, float(smoothed[peak] / total)


@dataclass
class ChromaPlan:
    coverage: np.ndarray
    weight: np.ndarray
    colour: np.ndarray
    paper: np.ndarray
    spots: int
    on_paper: np.ndarray


def _chroma_plan(sub: np.ndarray, margin: int = 0) -> ChromaPlan | None:
    """彩色半透明水印：逐像素解出水印覆盖比例，并拟合出水印本色。

    margin 用于「只拿到一块选区」的调用：选区边缘外有没有正文无从得知，
    就把这一圈标成「不是白纸」，免得把跨过边界的笔画抬进白纸。
    """
    hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    bright = float(np.percentile(val, 92))
    paper = (sat < 34) & (val > max(150.0, bright * 0.86))
    if float(paper.mean()) < 0.34:
        return None
    f = sub.astype(np.float32)
    lum = f.mean(axis=2)
    chroma = f - lum[..., None]
    chroma_mag = np.abs(chroma).max(axis=2)
    # 半透明彩色水印压在白纸上必然偏亮；不设亮度下限，扫描件的蓝黑正文
    # 会因为 HSV 饱和度高而被当成水印本体，把 colour 拟合到墨色上
    saturated = (sat >= 45) & (val >= max(110.0, bright * 0.5)) & (chroma_mag >= 12)
    frac = float(saturated.mean())
    if not 0.003 <= frac <= 0.42:
        return None
    hue, share = _dominant_hue(hsv, saturated)
    if hue < 0 or share < 0.2:
        return None
    dist = np.abs(hsv[..., 0] - hue)
    dist = np.minimum(dist, 180.0 - dist)
    core = saturated & (dist <= 16)
    if int(core.sum()) < 55:
        return None
    tint = np.median(f[core], axis=0)
    background = np.median(f[paper], axis=0)
    direction = (tint - tint.mean()) - (background - background.mean())
    denom = float(direction @ direction)
    if denom < 320.0:
        return None
    # 把色偏在水印方向上的投影当作覆盖比例：纸面像素应为 0，水印本体约为名义不透明度。
    # 必须用零均值色度做投影，不能用「离纸面多远」——中性黑正文在任意方向上的分量都很大，
    # 一旦改用纸色锚定的投影，笔画会被算成高覆盖水印，整页扫下来字被抬亮一大片
    raw = chroma.dot(direction) / denom - float(
        np.median(chroma[paper].dot(direction) / denom)
    )
    top = raw[core] >= float(np.percentile(raw[core], 90))
    anchor = float(np.mean(raw[core][top]))
    if int(np.count_nonzero(top)) < 20 or not 0.30 <= anchor <= 2.6:
        return None
    level = np.clip(raw / anchor, 0.0, 1.0)
    # 水印本色取核心云团的第一主方向，而不是逐通道中位数或高分位数：射线必须穿过云团，
    # 否则真水印像素的垂距会和正文一样大，「离射线多远」这个判据就废了
    cloud = f[core] - background
    axis = np.linalg.svd(cloud, full_matrices=False)[2][0]
    if float(axis @ cloud.mean(axis=0)) < 0.0:
        axis = -axis
    spread = float(np.percentile(cloud @ axis, 97))
    if not 24.0 <= spread <= 620.0:
        return None
    tint = background + axis * spread
    # 名义不透明度取固定值：覆盖拉满的像素据此反解恰好回到纸色
    nominal = NOMINAL_COVERAGE
    weight = _smoothstep(level, WEIGHT_LOW, WEIGHT_HIGH)
    if not 0.0004 <= float(weight.mean()) <= 0.55:
        return None
    colour = background + (tint - background) / nominal
    hits = (weight > 0.5).astype(np.uint8)
    spots = max(0, int(cv2.connectedComponentsWithStats(cv2.dilate(hits, np.ones((5, 5), np.uint8)), 8)[0]) - 1)
    if spots < MIN_SPOTS:
        return None
    # 「这一带确实是白纸」：中性暗像素（必须排除蓝水印本体，它自己就够暗）膨胀成安全边距后，
    # 窗口内一个都不剩。这是唯一能区分「水印压白纸」和「水印压正文」的判据——
    # 偏色方向、离纸面多远都不行，中性黑正文对着这条水印射线的夹角余弦高达 0.86，比真水印还像
    paper_lum = float(background.mean())
    inkish = (lum < paper_lum * INK_DARK) & (chroma_mag < INK_CHROMA)
    padded = cv2.dilate(inkish.astype(np.uint8), np.ones((INK_DILATE, INK_DILATE), np.uint8))
    on_paper = 1.0 - _smoothstep(cv2.boxFilter(padded, -1, (INK_WIN, INK_WIN), normalize=True), INK_LO, INK_HI)
    if margin:
        on_paper[:, :margin] = 0.0
        on_paper[:, -margin:] = 0.0
        on_paper[:margin, :] = 0.0
        on_paper[-margin:, :] = 0.0
    return ChromaPlan(
        np.clip(level * nominal, 0.0, nominal).astype(np.float32),
        weight.astype(np.float32),
        colour.astype(np.float32),
        background.astype(np.float32),
        spots,
        on_paper.astype(np.float32),
    )


def _blank_paper(crop: np.ndarray, plan: ChromaPlan) -> np.ndarray:
    """「这一点和它 31px 邻域确实是空白纸面」——比 on_paper 更硬的一层证据。

    on_paper 只看「中性暗像素」，被章染蓝的正文不在其内；再要求邻域亮度普遍接近纸面，
    才能断定这里的原色就是纸，允许直接归位而不是按拟合覆盖度反演。
    """
    lum = crop.astype(np.float32).mean(axis=2)
    paperish = (lum > float(plan.paper.mean()) * PAPER_DOT_LUM).astype(np.float32)
    local = cv2.boxFilter(paperish, -1, (PAPER_DOT_WIN, PAPER_DOT_WIN), normalize=True)
    return (plan.on_paper >= PAPER_DOT_ON_PAPER) & (local >= PAPER_DOT_LOCAL)


def _chroma_repair(rgb: np.ndarray, box: Box, plan: ChromaPlan) -> int:
    """按 观测 = 覆盖·水印色 + (1-覆盖)·原色 反解，逐像素还原，不搬运任何邻域内容。

    无水印处权重为 0，所以整页一起扫也安全：同一支水印在别处的副本会被一并清掉。
    """
    target = rgb[box.y0 : box.y1, box.x0 : box.x1, :]
    crop = target.astype(np.float32)
    coverage = plan.coverage[..., None]
    weight = plan.weight[..., None]
    solved = (crop - coverage * plan.colour[None, None, :]) / np.maximum(1.0 - coverage, 0.08)
    lum = crop.mean(axis=2, keepdims=True)
    solved_lum = solved.mean(axis=2, keepdims=True)
    paper_lum = float(plan.paper.mean())
    colour_lum = float(plan.colour.mean())
    # 反演的强度本来靠 weight 逐级放开，为的是不敢碰笔画；但水印压在真空白上的那部分
    # 因此只被抬起一半，留下一层灰影（就是「空白处长出字」的成因）。
    # 判据已经由 _chroma_plan 用空间信息给出，白纸就允许补满
    boost = plan.on_paper[..., None] * _smoothstep(coverage, COV_LO, COV_HI)
    alpha = np.maximum(weight, boost)
    # 覆盖顶到上限时反解是奇异的（分母→0），唯一稳的答案就是纸色本身
    lift = np.maximum(solved_lum - lum, 0.0)
    blank = _blank_paper(target, plan)
    snap = np.maximum(
        boost * _smoothstep(coverage, 0.55, 0.80),
        (blank & (coverage[..., 0] >= COV_HI)).astype(np.float32)[..., None],
    )
    # 反解不许凭空造出更黑的笔画，也不许把纸面像素抬过白纸
    base_lum = np.maximum(lum + np.maximum(lift, (paper_lum - lum) * snap), lum)
    base_lum = np.minimum(base_lum, paper_lum + 6.0)
    # 水印底下的字被染成蓝色，on_paper 认不出它、会把它提亮成纸色，只剩橙色边。
    # 用「按覆盖度衰减后的纸面亮度」减去观测亮度：明显更暗的就是被压住或紧邻的字。
    # 只在水印足迹内生效（中值 7px 把 1–2px 字边滤掉，只留 6–10px 的水印笔画）。
    band = np.zeros_like(lum)
    if colour_lum < paper_lum - 6.0:
        deficit = paper_lum - coverage * (paper_lum - colour_lum) - lum
        text = _smoothstep(deficit, TEXT_DEFICIT_LOW, TEXT_DEFICIT_HIGH)
        cov8 = np.clip(coverage[..., 0] / NOMINAL_COVERAGE, 0.0, 1.0) * 255.0
        med = cv2.medianBlur(cov8.astype(np.uint8), TEXT_FOOT_MED)
        foot = cv2.dilate(
            (med > TEXT_FOOT_LEVEL).astype(np.uint8),
            np.ones((TEXT_FOOT_DILATE, TEXT_FOOT_DILATE), np.uint8),
        ).astype(np.float32)
        halo = cv2.dilate(
            (text > 0.5).astype(np.uint8),
            np.ones((TEXT_HALO_GROW, TEXT_HALO_GROW), np.uint8),
        ).astype(np.float32)
        warm = np.clip((crop[..., 0:1] - crop[..., 2:3] - TEXT_WARM_KNEE) / 20.0, 0.0, 1.0)
        band = np.maximum(text, halo[..., None] * warm) * foot[..., None]
        alpha = np.maximum(alpha, band)
        # 文字保留观测亮度（水印压暗了多少就保多少），色度归中性；这会把笔画从「细橙边」拉回黑字
        base_lum = base_lum * (1.0 - band) + lum * band
    warm_halo = (
        cv2.dilate(
            (alpha > 0.35).astype(np.uint8),
            np.ones((WARM_HALO_GROW, WARM_HALO_GROW), np.uint8),
        ).astype(np.float32)[..., None]
        * np.clip((crop[..., 0:1] - crop[..., 2:3] - TEXT_WARM_KNEE) / 16.0, 0.0, 1.0)
        * (lum < paper_lum * INK_DARK)
    )
    alpha = np.maximum(alpha, warm_halo)
    base_lum = base_lum * (1.0 - warm_halo) + lum * warm_halo
    base = base_lum + (plan.paper - paper_lum)[None, None, :]
    blended = crop * (1.0 - alpha) + np.clip(base, 0.0, 255.0) * alpha
    target[...] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return int(np.count_nonzero(alpha > 0.02))


def _neutral_mask(sub: np.ndarray) -> np.ndarray | None:
    """浅色文档上的灰白水印：只取「比纸面略暗、且四周都是纸面」的像素。"""
    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    local = cv2.medianBlur(gray, 31).astype(np.float32)
    bright = float(np.percentile(gray, 90))
    if bright < 170:
        return None
    paper = (sat < 30) & (gray.astype(np.float32) > bright * 0.86)
    if float(paper.mean()) < 0.5:
        return None
    paper_local = cv2.boxFilter(paper.astype(np.float32), -1, (17, 17), normalize=True)
    deviation = local - gray.astype(np.float32)
    band = (deviation > 11) & (deviation < bright * 0.34) & (sat < 46)
    ink = cv2.dilate((gray.astype(np.float32) < bright * 0.62).astype(np.uint8), np.ones((7, 7), np.uint8))
    band &= paper_local > 0.5
    band &= ink == 0
    mask = _smoothstep(deviation, 11.0, 22.0) * band.astype(np.float32)
    if not 0.002 <= float(mask.mean()) <= 0.3:
        return None
    return mask


def _bright_kernel(sub: np.ndarray) -> int:
    kernel = int(min(sub.shape[0], sub.shape[1]) * 0.3) | 1
    return max(15, min(41, kernel))


def _open_background(sub: np.ndarray) -> np.ndarray:
    """开运算估计「无字背景」：腐蚀去掉比核细的亮笔画，膨胀恢复背景。"""
    kernel = _bright_kernel(sub)
    return cv2.morphologyEx(
        sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    )


def _bright_mask(sub: np.ndarray) -> np.ndarray | None:
    """浅色背景上的亮色水印（角标白字）：只取比开运算背景更亮且低饱和的像素。
    背景不能用中值滤波估——密集笔画会把中值抬高，dev 只剩 1px 骨架，修复不干净。"""
    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1]
    bright = float(np.percentile(gray, 90))
    if bright < 150:
        return None
    paper = (sat < 60) & (gray > bright * 0.80)
    if float(paper.mean()) < 0.5:
        return None
    background = _open_background(sub)
    deviation = gray - cv2.cvtColor(background, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = (deviation > 0) & (sat < 40)
    mask = _smoothstep(deviation, BRIGHT_DEV_LOW, BRIGHT_DEV_HIGH) * band.astype(np.float32)
    if not 0.001 <= float(mask.mean()) <= 0.45:
        return None
    spots = cv2.connectedComponentsWithStats((mask >= 0.5).astype(np.uint8), 8)[0] - 1
    if spots < MIN_SPOTS:
        return None
    return mask


def _bright_repair(rgb: np.ndarray, box: Box, mask: np.ndarray) -> None:
    """把亮色水印像素抬回开运算背景：alpha 到 1 时等于无字背景本身。"""
    crop = rgb[box.y0 : box.y1, box.x0 : box.x1, :]
    background = _open_background(crop).astype(np.float32)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    blended = crop.astype(np.float32) * (1.0 - alpha) + background * alpha
    rgb[box.y0 : box.y1, box.x0 : box.x1, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _masked_repair(rgb: np.ndarray, box: Box, mask: np.ndarray) -> None:
    height, width = rgb.shape[:2]
    margin = min(28, max(6, min(box.width, box.height) // 3))
    cx0 = max(0, box.x0 - margin)
    cy0 = max(0, box.y0 - margin)
    cx1 = min(width, box.x1 + margin)
    cy1 = min(height, box.y1 + margin)
    inner = np.zeros((cy1 - cy0, cx1 - cx0), dtype=np.float32)
    inner[box.y0 - cy0 : box.y1 - cy0, box.x0 - cx0 : box.x1 - cx0] = mask
    crop = rgb[cy0:cy1, cx0:cx1, :]
    lum = crop.astype(np.float32).mean(axis=2)
    chroma_mag = np.abs(crop.astype(np.float32) - lum[..., None]).max(axis=2)
    seed = (inner >= 0.05).astype(np.uint8) * 255
    if int(np.count_nonzero(seed)) < 8:
        return
    # 明显比纸面暗、且本身不偏色的像素是正文笔画：扩边与羽化一律不许碰它
    protect = (
        (lum < float(np.percentile(lum, 82)) * 0.78) & (chroma_mag < 20) & (seed == 0)
    ).astype(np.uint8)
    grown = cv2.dilate(seed, np.ones((5, 5), np.uint8))
    grown[protect > 0] = 0
    if int(np.count_nonzero(grown)) < 8:
        return
    filled = cv2.inpaint(crop, grown, 3, cv2.INPAINT_TELEA)
    sigma = float(min(1.8, max(0.8, min(box.width, box.height) * 0.004)))
    alpha = np.clip(cv2.GaussianBlur(grown.astype(np.float32) / 255.0, (0, 0), sigma) * 1.25, 0.0, 1.0)
    alpha[protect > 0] = 0.0
    blended = crop.astype(np.float32) * (1.0 - alpha[..., None]) + filled.astype(np.float32) * alpha[..., None]
    crop[...] = np.clip(blended, 0, 255).astype(np.uint8)


def _source_box(box: Box, direction: str, gap: int) -> Box | None:
    if direction == "left":
        return Box(box.x0 - box.width - gap, box.y0, box.x0 - gap, box.y1)
    if direction == "right":
        return Box(box.x1 + gap, box.y0, box.x1 + box.width + gap, box.y1)
    if direction == "up":
        return Box(box.x0, box.y0 - box.height - gap, box.x1, box.y0 - gap)
    if direction == "down":
        return Box(box.x0, box.y1 + gap, box.x1, box.y1 + box.height + gap)
    return None


def _inside(box: Box, width: int, height: int) -> bool:
    return box.x0 >= 0 and box.y0 >= 0 and box.x1 <= width and box.y1 <= height


def _edge_score(rgb: np.ndarray, target: Box, source: Box, direction: str) -> float:
    height, width = rgb.shape[:2]
    scores: list[float] = []
    if direction in ("left", "right"):
        if target.x0 > 0:
            near = rgb[target.y0 : target.y1, target.x0 - 1, :].astype(np.float32)
            edge = rgb[source.y0 : source.y1, source.x0, :].astype(np.float32)
            scores.append(float(np.mean(np.abs(near - edge))))
        if target.x1 < width:
            near = rgb[target.y0 : target.y1, target.x1, :].astype(np.float32)
            edge = rgb[source.y0 : source.y1, source.x1 - 1, :].astype(np.float32)
            scores.append(float(np.mean(np.abs(near - edge))))
    else:
        if target.y0 > 0:
            near = rgb[target.y0 - 1, target.x0 : target.x1, :].astype(np.float32)
            edge = rgb[source.y0, source.x0 : source.x1, :].astype(np.float32)
            scores.append(float(np.mean(np.abs(near - edge))))
        if target.y1 < height:
            near = rgb[target.y1, target.x0 : target.x1, :].astype(np.float32)
            edge = rgb[source.y1 - 1, source.x0 : source.x1, :].astype(np.float32)
            scores.append(float(np.mean(np.abs(near - edge))))
    return float(np.mean(scores)) if scores else 255.0


def _smart_fill(
    rgb: np.ndarray,
    box: Box,
    avoid: list[Box],
    preferred_direction: str = "auto",
) -> str | None:
    height, width = rgb.shape[:2]
    gap = max(4, int(min(width, height) * 0.006))
    directions = ["left", "right", "up", "down"] if preferred_direction == "auto" else [preferred_direction]
    candidates: list[tuple[float, str, Box]] = []

    for direction in directions:
        source = _source_box(box, direction, gap)
        if source is None or not _inside(source, width, height):
            continue
        if any(_intersects(source, blocked, margin=2) for blocked in avoid):
            continue
        score = _edge_score(rgb, box, source, direction)
        candidates.append((score, direction, source))

    if not candidates:
        return None

    _, direction, source = min(candidates, key=lambda item: item[0])
    patch = rgb[source.y0 : source.y1, source.x0 : source.x1, :].copy()
    if patch.shape[:2] != (box.height, box.width):
        return None
    alpha = _feather_alpha(box.height, box.width, box, height, width)
    target = rgb[box.y0 : box.y1, box.x0 : box.x1, :]
    blended = target.astype(np.float32) * (1.0 - alpha) + patch.astype(np.float32) * alpha
    target[...] = np.clip(blended, 0, 255).astype(np.uint8)
    return direction


def _inpaint_fill(rgb: np.ndarray, box: Box) -> None:
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[box.y0 : box.y1, box.x0 : box.x1] = 255
    filled = cv2.inpaint(rgb, mask, 4, cv2.INPAINT_TELEA)
    alpha = _feather_alpha(box.height, box.width, box, height, width)
    target = rgb[box.y0 : box.y1, box.x0 : box.x1, :]
    patch = filled[box.y0 : box.y1, box.x0 : box.x1, :]
    blended = target.astype(np.float32) * (1.0 - alpha) + patch.astype(np.float32) * alpha
    target[...] = np.clip(blended, 0, 255).astype(np.uint8)


def _save_image(image: Image.Image, output_path: Path, source_info: dict) -> None:
    suffix = output_path.suffix.lower()
    format_name = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }.get(suffix, "PNG")

    save_image = image
    if format_name == "JPEG" and "A" in image.getbands():
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        save_image = background
    elif format_name in ("BMP",) and image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        save_image = background

    save_params: dict = {}
    icc_profile = source_info.get("icc_profile")
    if icc_profile:
        save_params["icc_profile"] = icc_profile
    if format_name == "JPEG":
        save_params.update(quality=95, subsampling=0, optimize=True, progressive=True)
    elif format_name == "PNG":
        save_params.update(optimize=True)
    elif format_name == "WEBP":
        save_params.update(quality=96, method=4)
    elif format_name == "TIFF":
        save_params.update(compression="tiff_lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写 .part 再改名：中途崩溃/断电不会留下「存在但被截断」的成品被当成成功
    part_path = output_path.with_name(output_path.name + ".part")
    try:
        save_image.save(part_path, format=format_name, **save_params)
        os.replace(part_path, output_path)
    finally:
        if part_path.exists():
            os.remove(part_path)


def _sweep_stamp_cores(rgb: np.ndarray, boxes: list[Box], plan: ChromaPlan) -> int:
    """补掉检测框之间空隙里的同款水印核心。

    覆盖度单看像素会被正文的 JPEG 蓝边骗到（细线也能到 0.9），先开运算把细线滤掉，
    只留下成块的实心水印，再向外扩一圈按低门限补齐边缘。
    """
    height, width = rgb.shape[:2]
    covered = np.zeros((height, width), dtype=bool)
    for box in boxes:
        covered[box.y0 : box.y1, box.x0 : box.x1] = True
    core = (plan.coverage >= CORE_SWEEP) & ~covered
    if not core.any():
        return 0
    seed = cv2.morphologyEx(
        core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((CORE_OPEN, CORE_OPEN), np.uint8)
    ).astype(bool)
    # 过不了开运算的细条水印（左边距等距重复的小竖条）改由「空白纸面」这条证据进种子
    seed |= _blank_paper(rgb, plan) & core
    if not seed.any():
        return 0
    region = cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    region &= (plan.coverage >= CORE_SWEEP_GATE) & ~covered
    if not region.any():
        return 0
    repaired = rgb.copy()
    _chroma_repair(repaired, Box(0, 0, width, height), plan)
    rgb[region] = repaired[region]
    return int(region.sum())


def remove_boxes_inplace(
    rgb: np.ndarray, boxes: list[Box], method: str = options.DEFAULT_METHOD
) -> dict:
    fallback_count = 0
    unblend_count = 0
    masked_count = 0
    directions: list[str] = []
    height, width = rgb.shape[:2]
    page_plan = _chroma_plan(rgb) if method == options.SMART and boxes else None
    for box in boxes:
        if method == options.SMART:
            crop = rgb[box.y0 : box.y1, box.x0 : box.x1, :]
            plan = None
            if page_plan is not None:
                plan = ChromaPlan(
                    page_plan.coverage[box.y0 : box.y1, box.x0 : box.x1],
                    page_plan.weight[box.y0 : box.y1, box.x0 : box.x1],
                    page_plan.colour,
                    page_plan.paper,
                    page_plan.spots,
                    page_plan.on_paper[box.y0 : box.y1, box.x0 : box.x1],
                )
                if float(plan.weight.mean()) < 0.0004:
                    plan = None
            if plan is None:
                plan = _chroma_plan(crop, INK_MARGIN)
            if plan is not None:
                _chroma_repair(rgb, box, plan)
                unblend_count += 1
                continue
            mask = _neutral_mask(crop)
            if mask is not None:
                mask = cv2.morphologyEx(
                    (mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                ).astype(np.float32) / 255.0
                _masked_repair(rgb, box, mask)
                masked_count += 1
                continue
            mask = _bright_mask(crop)
            if mask is not None:
                _bright_repair(rgb, box, mask)
                masked_count += 1
                continue
        if method == options.SMART:
            direction = _smart_fill(rgb, box, boxes)
            if direction is None:
                _inpaint_fill(rgb, box)
                fallback_count += 1
            else:
                directions.append(direction)
        else:
            _inpaint_fill(rgb, box)
    swept = 0
    if method == options.SMART and page_plan is not None and unblend_count:
        swept = _sweep_stamp_cores(rgb, boxes, page_plan)
    return {
        "fallback_regions": fallback_count,
        "directions": directions,
        "masked_regions": masked_count,
        "unblend_regions": unblend_count,
        "page_scope": False,
        "swept_pixels": swept,
    }


def method_note(method: str, boxes: list[Box], width: int, height: int, detail: dict) -> str:
    masked_count = detail.get("masked_regions", 0)
    unblend_count = detail.get("unblend_regions", 0)
    if method == options.SMART and unblend_count + masked_count == len(boxes) and boxes:
        note = "已逐像素识别水印本体并修补，未触碰其余文字与空白"
        if detail.get("swept_pixels"):
            note += "；框外同款水印已一并清除"
    elif method == options.SMART:
        note = "已用周边纹理智能修复"
        directions = detail.get("directions", [])
        if directions:
            direction_names = {"left": "左侧", "right": "右侧", "up": "上方", "down": "下方"}
            used = "、".join(dict.fromkeys(direction_names.get(item, item) for item in directions))
            note += f"（取样方向：{used}）"
        fallback_count = detail.get("fallback_regions", 0)
        if fallback_count:
            note += f"；{fallback_count} 处周边空间不足，已改用内容填充"
        if masked_count:
            note += f"；{masked_count} 处按水印像素精准修补"
    else:
        note = "已用内容填充修复"
    if method == options.SMART and unblend_count and note.startswith("已用周边"):
        note += f"；{unblend_count} 处按水印颜色逐像素还原"
    area_ratio = sum(box.width * box.height for box in boxes) / max(1, width * height)
    if area_ratio > 0.18:
        note += "；选区较大，建议适当缩小后复查"
    return note


def process_image_file(
    input_path: Path,
    output_path: Path,
    regions: list[dict],
    method: str = options.DEFAULT_METHOD,
) -> dict:
    try:
        with Image.open(input_path) as opened:
            source_info = dict(opened.info)
            image = ImageOps.exif_transpose(opened)
            has_alpha = "A" in image.getbands() or (
                image.mode == "P" and "transparency" in image.info
            )
            image = image.convert("RGBA" if has_alpha else "RGB")
    except FileNotFoundError as exc:
        raise ImageProcessError("原文件已不在素材库，请重新添加这个文件") from exc
    except Exception as exc:
        raise ImageProcessError(f"无法打开图片：{exc}") from exc

    if not regions:
        raise ImageProcessError("没有设置水印区域")
    array = np.array(image)
    height, width = array.shape[:2]
    padding = max(4, round(min(width, height) * 0.006))
    boxes = regions_to_boxes(regions, width, height, padding)
    if not boxes:
        raise ImageProcessError("水印区域太小或无效")

    rgb = array[..., :3].copy()
    if method == options.PHOTO:
        # 照片模式：本地 LaMa 重绘，只回贴选框内的像素；文档请继续用 smart。
        from . import photo_inpaint

        try:
            detail = photo_inpaint.inpaint_boxes(rgb, boxes)
        except photo_inpaint.PhotoUnavailable as exc:
            raise ImageProcessError(str(exc)) from exc
        note = "已用本地照片重绘模型修补选框区域，适合照片与插画；文字页面请改用智能纹理修复"
        missed = len(boxes) - detail.get("photo_regions", 0)
        if missed:
            note += f"；{missed} 处选框过小未能重绘"
        array[..., :3] = rgb
        result = Image.fromarray(array)
        _save_image(result, output_path, source_info)
        return {
            "method": method,
            "regions": len(boxes),
            "fallback_regions": 0,
            "note": note,
            "width": width,
            "height": height,
        }

    detail = remove_boxes_inplace(rgb, boxes, method)
    array[..., :3] = rgb
    result = Image.fromarray(array)
    _save_image(result, output_path, source_info)

    return {
        "method": method,
        "regions": len(boxes),
        "fallback_regions": detail["fallback_regions"],
        "note": method_note(method, boxes, width, height, detail),
        "width": width,
        "height": height,
    }
