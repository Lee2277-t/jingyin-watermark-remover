"""图片修复引擎的自合成语料回归（第二十三轮）。

**为什么现在必须有它**：`_check.py` 唯一的语料是 `work/uploads/*/*/source.png`，而
`cleanup_old_files()`（server.py:1209）按**会话目录 mtime 满 72 小时**整目录 `rmtree`。
2026-10-26 量到盘上七个会话目录全在 68.5–71.4 小时之间——再过一两小时，全项目最敏感那块
（§9 图片修复引擎）唯一的回归语料归零，而且永远回不来：素材是用户自己的作品，到期真删是产品行为，
不能为了测试留档。§11.5 记的份数轨迹 143→118→104→90→76→71 就是这个蒸发过程。2026-10-27
复核：它们已经 **73.2–76.1 小时**、仍然在盘上，只是还没人重启过服务——`cleanup_old_files()` 只在
启动时跑，所以「什么时候归零」取决于用户下一次重启，而不是 72 小时那一刻。这正是这份语料必须
自己造出来的原因：判据不能挂在一个随时会消失、且消失后无法重建的目录上。

**顺带把验收口径升级一格**。`_check.py` 只能量「有没有伤正文」（erased / darkened / 框外改动），
因为它没有「水印底下原本长什么样」的答案；合成夹具知道，所以这里同时量两半：

- 没伤正文：`erased` / `darkened` / `ink_moved`（真值正文像素被改动）/ `stray`（框外且不在笔画
  影响圈里的改动）——全按真值算，不再靠人工判读「只允许冒号点与空白纸面水印点」；
- 修得对不对：`residual`（水印笔画像素上 |成品−真值| 均值）、`missed`（**框外**笔画像素修完仍然
  偏蓝＝该清清不掉）、`ripple`（成品高频能量 ÷ 真值高频能量，>1 就是自己造了振铃——§8 视频那条
  老规矩搬到图片上）；
- §9.2 那九条结论各有一条夹具守着：`--canaries` 把对应常量逐个猴子补丁（结论 1 没有常量可改，
  它禁的是一种写法，所以那条改的是 `_chroma_plan` 的函数体），断言**该变红的那条必须变红**——
  永远不会失败的测试不如没有。

**夹具自己必须先像真页**。同一批 71 份真语料逐份标定（`work/qa/_r23_calib.txt`，只统计像素、
不看内容）量到的形状：中性暗字占页面 p50 **0.0084**（p75 0.0116 / p90 0.0155 / max 0.0252）、
暗字亮度 **p50 0.0876×纸面**、字外一圈 1px 镶边占 **p50 0.0149**（max 0.0355）、**镶边亮度
p50 0.8444×纸面**（是「浅而偏色」，不是「深而偏色」）、镶边色度 p50 30.3、其中 HSV 饱和（sat≥45）
占比 p50 0.530。第一版夹具把镶边画成**暗**的（0.44×纸面）且正文密了近十倍，`_chroma_plan` 于是把
整页彩边当成低覆盖水印反演，`erased` 直接 8125——那是夹具失真，不是产品缺陷。所以下面每个夹具先过
`realism()`，量到的形状不在真语料带里就直接判失败，再谈引擎。

夹具全部脚本内合成，**不读用户素材**；`--keep` 才落盘 `_ie_*.png`，`--probe` 打印改动落点分解
（调阈值时用，不进基线）。
"""

import argparse
import inspect
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from remover import image_ops  # noqa: E402
from remover.image_ops import Box, remove_boxes_inplace  # noqa: E402

QA = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(QA, "_image_engine_baseline.json")

INK_GUARD = 47  # 离水印笔画 23px 以内的正文允许留一丝色晕（§9.3 的实测取舍），再远必须逐像素不动
STRAY_GUARD = 13  # 框外只允许紧贴笔画 ±6px 的过渡带被补扫碰到

INK_LUM = 22.7  # 0.089×纸面亮度，对齐真语料的暗字 p50 0.0876
# 章的**观测**本体色必须是真页那一个，不是「纯水印色」。`work/qa/_r23_calib_bands.txt` 第三节量到 18 份
# 文本页的章色度方向单位向量都是 [−0.714, +0.013, +0.701]（G 恰好为 0：真章本体 (7,105,203)
# 的绿正好落在自己的均值上），而 0.85×(30,92,200) 落在纸上得到的是 (62,115,206)、色度模只有
# 103，方向歪成 [−0.624,−0.145,0.768]。覆盖度拟合的分母是 `direction·direction`
# （image_ops.py:244），所以 `coverage ≈ 0.9×(彩边偏离·章偏离)/|章偏离|²` 对同一个彩边是
# **反比于章的色度模**的——章画淡了，正文彩边的覆盖度就整体虚高（实测 0.272 vs 真页 0.177～0.199），
# 反演强度 `_smoothstep` 跟着从 0.32 跳到 0.80，把夹具自己的蓝边剥到 ±25 计数门之下，
# `暖/蓝` 那条判据就变成一张不存在的素材在自评。所以章必须按**观测色**对齐：真章本体 (7,105,203)
# 配 §9.3 记下的真实不透明度 0.96，反解出纯章色 (0,99,202)——叠在纸上得 (10,105,203)。
# 代价要写在前面：0.96 > `NOMINAL_COVERAGE=0.90`，反演在这点上接近奇异（§9.3「保字留晕」），
# 所以忠实之后 `stamp`/`dots`/`tiled` 的残差封顶必须重新量，不能假装修得回来。
STAMP_BLUE = (0, 99, 202)
STAMP_ALPHA = 0.96  # 三个彩色夹具共用：让「叠在纸上的观测色」而不是「纯章色」对上真页
# ⑦ 专用：章从浓到淡的 α 梯度两端。**上端必须是 `STAMP_ALPHA`**——`_chroma_plan` 的
# `anchor` 取核心云团 raw 投影的前 10% 均值（image_ops.py:227-228），方向与前 10% 由最浓那一段
# 定住，正文彩边的 coverage 才不会整体虚高，`cast_split` 那条真页带才照旧成立。
# 下端 0.62 是「结论 6 那条带必须真能点火」的下界：压在字上时
# `deficit = paper_lum − cov·(paper_lum − colour_lum) − lum ≈ 219.3 − 210.7α`
# （cov≈0.9375α、L_ink=22.7），要越过 `TEXT_DEFICIT_LOW=30` 就得 α<0.898，
# 要饱和到 `TEXT_DEFICIT_HIGH=60` 就得 α<0.86。所以均匀 0.96 的章**根本走不到这一支**。
WASH_ALPHA_LO = 0.62
WASH_ALPHA_HI = STAMP_ALPHA

# 真语料标定带（`work/qa/_r23_calib.txt`，71 份，只统计像素不看内容）。下界取 p25，上界取观测到的 max。
CORPUS_INK_SHARE = (0.004, 0.026)
CORPUS_INK_LUM = (0.06, 0.13)
CORPUS_RING_SHARE = (0.008, 0.036)
CORPUS_RING_LUM = (0.75, 0.95)
CORPUS_RING_CHROMA = (10.0, 49.0)  # max|通道−亮度| 的中位：p25 14.0 / p50 30.3 / max 48.3
CORPUS_RING_SAT = (0.30, 0.75)  # 镶边里 HSV 饱和（sat>=45）的占比：p50 0.528 / p90 0.615
CORPUS_DARK_WARM_OVER_BLUE = (0.4, 1.6)  # 真页「暗字本体就带彩边」，p50 0.713

# 覆盖度门槛换算到 `_chroma_plan` 的输出口径：plan 里没有 `level`，只有
# `coverage = clip(level*NOMINAL_COVERAGE, 0, NOMINAL_COVERAGE)`（image_ops.py:265），
# 所以 `WEIGHT_LOW/HIGH` 这两道 weight 门槛要乘回同一个系数才能与 coverage 比（:244）。
COV_LOW = image_ops.WEIGHT_LOW * image_ops.NOMINAL_COVERAGE  # 0.090
COV_HIGH = image_ops.WEIGHT_HIGH * image_ops.NOMINAL_COVERAGE  # 0.306

# 真页「正文自己的偏色」被覆盖度拟合吃掉的程度（`work/qa/_r23_cast_amp.txt`，18 份文本页，
# 只统计像素不看内容）。这是**夹具保真**判据，不是画质判据：夹具若画不出真页那种
# 「蓝边被反演、暖边原样留下」的不对称，后面 暖/蓝 两项量的就是一张不存在的素材。
CORPUS_CAST_BLUE_P50 = (0.16, 0.225)  # 蓝偏正文 coverage p50：真页 18 份 0.177–0.199（`_r23_calib_bands.txt` 第三节）
# ↑ 上界没有贴着真页 max 0.199 收，留到 0.225：夹具的章是合成纯色（没有 JPEG 噪声摊薄
#   `direction·direction` 那个分母），同一个彩边量到的 coverage 必然略高，实测 0.202–0.203。
#   这条带的两端各拦一头——上界拦「章画淡了 → coverage 虚高」（第一版 `STAMP_ALPHA` 太轻时
#   量到 **0.272**，见上面 STAMP_BLUE 的推导），下界拦「章画浓了 → 正文彩边被拟合得没反演强度」。
#   两端都必须比 §9.1 的 `WEIGHT_HIGH=0.34×0.90=0.306`（`COV_HIGH`）低：真页的彩边本来就不该被
#   当成实心水印，否则夹具在替一种真页上不存在的「整页低覆盖水印」做反演。
CORPUS_CAST_BLUE_SHARE = (0.95, 1.0)  # 其中 coverage≥COV_LOW（真会进反演）的占比：真页 0.99–1.00
CORPUS_CAST_WARM_SHARE = (0.0, 0.02)  # 暖偏正文同一占比：真页 18/18 份都是 0.000

# 同一族像素的**振幅**（`work/qa/_r23_cast_band.txt`，量的是框外——引擎从不碰那一片，
# 所以它就是「正文彩边原样长什么样」的真值）。口径与 `cast_populations()` 同一对门限。
# 这一条专门拦「把彩边写成刚好越过 ±25 计数门槛的常数」：常数振幅让 `blue_in` 在部分剥离后
# 恰好归零，暖/蓝 于是飞成夹具的刀口（真页修后 p50 1.48 / max 4.57，夹具第一版量到 31–310）。
CORPUS_CAST_AMP = {"p10": (24.0, 36.0), "p50": (34.0, 50.0), "p90": (44.0, 62.0)}
CORPUS_CAST_SPREAD_MIN = 12.0  # p90−p10：真页 26；写死常数时是 0
CORPUS_CAST_MIN_N = 400  # 低于这个样本量，分位数没有意义
CORPUS_CAST_BLUE_POP_SHARE = (0.42, 0.62)  # 蓝偏条数 ÷（蓝+暖）：真页 29,728/56,576 = 0.525

results = []
fails = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    if not ok:
        fails.append(f"{name} {detail}")
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# --------------------------------------------------------------------------
# 画布：纸、正文（暗核 + 浅色偏色镶边）、水印 overlay
# --------------------------------------------------------------------------


def canvas(seed, width=1100, height=1400, base=(246, 244, 238), sigma=2.2):
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), np.float32)
    img[:] = np.array(base, np.float32)
    img += rng.normal(0.0, sigma, (height, width, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _shift(mask, dy, dx):
    """把 mask 平移 (dy,dx)，出界截断——JPEG ringing 是单边的，不是四边对称。"""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    tys = slice(max(0, -dy), h + min(0, -dy))
    txs = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = mask[tys, txs]
    return out


def draw_text(img, region, rng, line_h=34, ring_chroma=28.0):
    """画一块正文，形状照着真语料的统计带（docstring）。

    五个要点都是量出来的、不是猜的：
    ① 暗核只有 **1px** 笔画——真页暗字占比 p50 0.0084，只有 1px 笔画加得起来；画成实心块会密近十倍；
    ② 镶边**浅而偏色**（平均 0.84×纸面、色度 ~28）；画成整片暗的不属于这份语料，而且会把 `_chroma_plan`
       骗成「整页都是低覆盖水印」；
    ③ 镶边的彩度是**单边**的（偏蓝就是 B 抬高、R/G 各降一半），不是 R/B 对称外推。第一版画成
       对称 ±dev，同一份 `max|通道−亮度|` 下 HSV 饱和从 1.5x 变成 2x，镶边 sat>=45 的占比
       量到 **0.87**（真语料 p50 0.528 / p90 0.615）——等于夹具自带一圈真页上不存在的浓彩边，
       而 `_sweep_stamp_cores` 正好把它当框外实心水印，353px 框外改动全来自这里；
    ④ **暗字本体自己就带彩边**（真页暗像素 暖/蓝 p50 0.713，中性核算不进这个比值）。这一条是
       「修完不许只剩暖边」能拿真值当判据的前提：核里没有彩边时，框内暖/蓝 只能在 0/0 与飞掉
       之间跳，断言无从成立。
    ⑤ 彩边的**振幅必须有分布**，不能是一个刚好越过 ±25 计数门槛的常数——见下面两处注释；
       这条由 `cast_amplitude()` 守着，写回常数它就先变红。
    """
    x0, y0, x1, y1 = region
    mask = np.zeros(img.shape[:2], np.uint8)
    y = y0
    while y < y1 - 6:
        x = x0
        while x < x1 - 26:
            for _ in range(1 + int(rng.random() < 0.45)):
                if rng.random() < 0.5:
                    gy = int(y + rng.integers(0, 20))
                    cv2.line(mask, (x, gy), (x + int(rng.integers(9, 21)), gy), 255, 1)
                else:
                    gx = int(x + rng.integers(0, 20))
                    cv2.line(mask, (gx, y), (gx, y + int(rng.integers(9, 21))), 255, 1)
            x += int(rng.integers(30, 46))
        y += line_h
    core = mask > 0
    edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool) & ~core

    f = img.astype(np.float32)
    lum = f.mean(axis=2)
    bg = float(np.percentile(lum, 80))
    cy, cx = np.where(core)
    base = np.full(len(cy), INK_LUM, np.float32)
    # 暖 10% / 蓝 14.3% -> 比值 0.70，其余保持中性（真页暗像素 暖/蓝 p50 0.713）
    # 振幅**逐像素取**，不许写成一个常数：真页这一族像素的 |B−R| 是 28/42/54（p10/p50/p90，
    # `work/qa/_r23_cast_band.txt`），第一版在这里写死 ±32 是为了过 ±25 那道计数门槛——
    # 结果整批像素正好摆在门槛上，反演削掉 7 级就一起掉出统计，量到的 暖/蓝 是夹具的刀口。
    # 核只有 22.7 级亮，所以两侧对称偏移是唯一既保住亮度、又保住色度的画法：
    # R=base+A/2、B=base−A/2 ⇒ 亮度仍是 base、色度=A/2（A=44 → 22，仍低于 INK_CHROMA=34），
    # 而 A 再往 51 以上就要把一只通道推出 0 以下，那一部分交给镶边去承担（见下面 bimodal）。
    kind = rng.random(len(cy))
    amp = np.clip(rng.normal(33.0, 6.0, len(cy)), 26.0, 44.0).astype(np.float32)
    off = np.where(kind < 0.10, amp, np.where(kind < 0.243, -amp, 0.0)).astype(np.float32)
    f[cy, cx, 0] = base + off / 2.0
    f[cy, cx, 1] = base
    f[cy, cx, 2] = base - off / 2.0
    if edge.any():
        # 上/左一侧偏蓝、下/右一侧偏暖：同一道边在真页上是两种符号（实测暖/蓝 p50 0.70）
        blue_side = _shift(core, -1, 0) | _shift(core, 0, -1)
        ys, xs = np.where(edge)
        # 亮度是**双峰**的，不是清一色的浅边：真页的一道笔画边既有「笔画与纸面之间的过渡像素」
        # （比纸面暗、落在 INK_DARK=0.72 以下）也有 JPEG 冲过头的浅边。第一版只画了浅的那一半
        # （N(0.84,0.09) 只有 9% 掉进门槛以下），于是「暗字本体带彩边」这一族在夹具里近乎不存在，
        # 暖/蓝 的计数只能靠核上那个刀口振幅——这是本轮第一版把夹具画歪的地方。
        # 两峰配比按 `lum[edge].mean()` 反推：0.25×0.62 + 0.75×0.90 = 0.837，对齐真语料 p50 0.8444。
        trans = rng.random(len(ys)) < 0.25
        levels = (
            np.where(
                trans,
                np.clip(rng.normal(0.62, 0.05, len(ys)), 0.52, 0.70),
                np.clip(rng.normal(0.90, 0.06, len(ys)), 0.74, 0.99),
            )
            * bg
        ).astype(np.float32)
        # 这个模型里 通道偏离 = (mag/2, mag/2, mag) ⇒ 色度恰为 mag、|B−R| 恰为 1.5×mag。
        # mag 取 N(28,6) ⇒ |B−R| = 30/42/53（p10/p50/p90），落在真页带 28/42/54 上。
        mag = np.clip(rng.normal(ring_chroma, 6.0, len(ys)), 1.0, 50.0).astype(np.float32)
        cast = np.where(blue_side[ys, xs], 2, 0).astype(np.int64)  # 2=B 偏蓝，0=R 偏暖
        f[ys, xs, :] = (levels - mag / 2.0)[:, None]
        f[ys, xs, cast] = levels + mag
    img[...] = np.clip(f, 0, 255).astype(np.uint8)
    return core, edge


def overlay(img, mask, colour, alpha):
    """把半透明水印压上去；返回压之前的干净真值。"""
    clean = img.copy()
    f = img.astype(np.float32)
    c = np.array(colour, np.float32)
    f[mask] = alpha * c + (1.0 - alpha) * f[mask]
    img[...] = np.clip(f, 0, 255).astype(np.uint8)
    return clean


def overlay_faded(img, mask, colour, lo, hi, seed):
    """按「印泥不匀」压一枚章：α 从左上到右下渐淡，另加一层低频起伏。

    只为一件事服务：结论 6 那条「亮度亏损」判据在均匀 0.96 的章上**永远点不着火**
    （`deficit ≈ 219.3 − 210.7α`，见 `WASH_ALPHA_LO` 那段推导）。真章本来就不是平的，
    而 §9.2 结论 6 那三件套（亮度亏损 + 足迹门 + 暖色晕）正是为真页上「章压字但章比较淡」
    那一段写的——夹具不画出淡的那一段，那一支代码就没有测试守着。
    """
    clean = img.copy()
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # 左上浓、右下淡，再叠一层 σ24px 的低频起伏（真章的边缘就是这个样子）
    tilt = np.clip((xs - 390.0) / 380.0 + (ys - 470.0) / 380.0, 0.0, 1.0) * 0.5
    ripple = cv2.GaussianBlur(
        rng.standard_normal((h, w)).astype(np.float32), (0, 0), 24.0
    )
    ripple = ripple / float(np.percentile(np.abs(ripple), 98))
    alpha = hi - (hi - lo) * np.clip(tilt + 0.16 * ripple, 0.0, 1.0)
    f = img.astype(np.float32)
    c = np.array(colour, np.float32)
    a = alpha[mask][..., None]
    f[mask] = a * c + (1.0 - a) * f[mask]
    img[...] = np.clip(f, 0, 255).astype(np.uint8)
    return clean


def stamp_ring(shape, center, radius, thickness, extra_bar=True):
    """一枚章：外圈 + 内横条（两个连通域，正好喂 MIN_SPOTS=2 这道闸）。"""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, center, radius, 255, thickness)
    if extra_bar:
        cx, cy = center
        cv2.rectangle(mask, (cx - radius + 8, cy - 5), (cx + radius - 8, cy + 5), 255, -1)
    return mask > 0


def dark_patch(img, center, axes, keep=0.45):
    """一块深色插图内容：角标白字如果没有路线，就落到邻域贴片把这一块糊进水印区（结论 7）。"""
    patch = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(patch, center, axes, 0, 0, 360, 255, -1)
    p = patch > 0
    f = img.astype(np.float32)
    f[p] = f[p] * keep + np.array((36, 30, 44), np.float32) * (1.0 - keep)
    img[...] = np.clip(f, 0, 255).astype(np.uint8)
    return p


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def _cast(img):
    """B−R（int32）。真语的「偏色」在图片引擎里就是这一族量。"""
    f = img.astype(np.int32)
    return f[..., 2] - f[..., 0]


def _delta(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16))


def _dilate(mask, k):
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def dlum_delta(clean, after):
    """带符号的亮度变化（正=被抬亮，负=被压暗）。

    必须带符号：`erased`（抬亮）与 `paper_dark`（没归位、仍偏暗）是两种相反的坏法，
    取绝对值就分不开，而 §9.3 那条「保字留晕」的取舍恰恰只在「抬亮」这一侧成立。
    """
    return after.astype(np.float32).mean(axis=2) - clean.astype(np.float32).mean(axis=2)


def _stats(dlum, sel):
    """返回 (|Δ亮度| 均值, |Δ亮度| p99)，空选择回 (0.0, 0.0)。"""
    if not sel.any():
        return (0.0, 0.0)
    v = np.abs(dlum[sel])
    return (round(float(v.mean()), 2), round(float(np.percentile(v, 99)), 1))


def ink_of(clean):
    """真值里的「正文」：中性暗像素，与 `_check.py` 同口径（chroma<34 把彩色水印本体排除在外）。"""
    f = clean.astype(np.float32)
    lum = f.mean(axis=2)
    bg = float(np.percentile(lum, 80))
    chroma = np.abs(f - lum[..., None]).max(axis=2)
    return (lum < bg * 0.6) & (chroma < 34.0), bg


def ink_damage(before, after):
    lb = before.astype(np.float32).mean(axis=2)
    la = after.astype(np.float32).mean(axis=2)
    bg = float(np.percentile(lb, 80))
    chroma = np.abs(before.astype(np.float32) - lb[..., None]).max(axis=2)
    real_ink = (lb < bg * 0.6) & (chroma < 34.0)
    erased = int(np.count_nonzero(real_ink & (la - lb > 26)))
    darkened = int(np.count_nonzero(real_ink & (lb - la > 26)))
    return erased, darkened


def warm_blue(img, region, bg=None):
    f = img.astype(np.float32)
    lum = f.mean(axis=2)
    ref = bg if bg is not None else float(np.percentile(lum, 80))
    dark = region & (lum < ref * image_ops.INK_DARK)
    r, b = f[..., 0], f[..., 2]
    warm = int(np.count_nonzero(dark & (r - b > 25)))
    blue = int(np.count_nonzero(dark & (b - r > 25)))
    return warm, blue


def boxes_mask(boxes, shape):
    m = np.zeros(shape[:2], bool)
    for bx in boxes:
        m[bx.y0 : bx.y1, bx.x0 : bx.x1] = True
    return m


def residual(clean, after, stroke):
    d = _delta(clean, after).mean(axis=2)
    if not stroke.any():
        return 0.0, 0.0
    return float(d[stroke].mean()), float(d[stroke].max())


def ripple(clean, after, region):
    def energy(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        return float(np.mean(lap[region] ** 2)) if region.any() else 0.0

    ec, eo = energy(clean), energy(after)
    return eo / ec if ec > 1e-6 else float("inf")


def realism(name, clean, core, edge, chromatic=True):
    """夹具自己像不像真页——不像的话，后面所有指标都在量一张不存在的素材。

    `ring饱和占比` 与 `ring色度` 用的是标定真语料时（`work/qa/_r23_calib_bands.txt`）的同两个公式
    （sat = (max−min)/max×255 判 45；色度 = max|通道−亮度|），所以这两项和
    `work/qa/_r23_calib.txt` 是可以直接对照的，不存在「夹具与语料各量一种饱和度」。
    """
    f = clean.astype(np.float32)
    lum = f.mean(axis=2)
    bg = float(np.percentile(lum, 80))
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0) * 255.0
    chroma = np.abs(f - lum[..., None]).max(axis=2)
    measured = {
        "ink占比": (float(core.mean()), CORPUS_INK_SHARE),
        "ink亮度": (float(lum[core].mean() / bg) if core.any() else 0.0, CORPUS_INK_LUM),
        "ring占比": (float(edge.mean()), CORPUS_RING_SHARE),
        "ring亮度": (float(lum[edge].mean() / bg) if edge.any() else 0.0, CORPUS_RING_LUM),
    }
    if chromatic and edge.any():
        measured["ring饱和占比"] = (float((sat[edge] >= 45.0).mean()), CORPUS_RING_SAT)
        measured["ring色度"] = (float(np.median(chroma[edge])), CORPUS_RING_CHROMA)
        # 判据对象与标定口令（`_r23_calib_bands.txt`）一致：整页「比纸面暗」的像素里暖与蓝的条数比
        dark = lum < bg * image_ops.INK_DARK
        r, b = f[..., 0], f[..., 2]
        warm = int(np.count_nonzero(dark & (r - b > 25)))
        blue = int(np.count_nonzero(dark & (b - r > 25)))
        measured["暗字暖/蓝"] = ((warm + 1) / (blue + 1), CORPUS_DARK_WARM_OVER_BLUE)
    bad = [
        f"{key} {val[0]:.4f} 不在真语料带 {val[1]}"
        for key, val in measured.items()
        if not val[1][0] <= val[0] <= val[1][1]
    ]
    shape = " ".join(f"{key}={val[0]:.4f}" for key, val in measured.items())
    print(f"  真值形状 {name}: {shape}")
    check(f"{name} 夹具形状在真语料带内", not bad, "；".join(bad))
    return not bad


def cast_populations(img):
    """按「正文自己带的偏色方向」把暗字分成蓝偏 / 暖偏 / 中性三堆。

    口径与 `work/qa/_r23_cast_band.txt` 完全一致：比纸面暗、且色度模 `<INK_CHROMA`
    （同一对常量，`_chroma_plan` 用它划 `inkish` 在 image_ops.py:256），再按 `B−R` 的符号分堆。
    纸面亮度这里取整页 p80（与 `realism()` 标定真语料时同一个取法），而 `_chroma_plan`
    内部取的是 `paper` 掩膜的中位数亮度——两处只在「谁是纸面」这一句上不同，
    分堆用的门限是相对纸面的亮度比，所以按 p80 划堆与按 plan 自己的 `on_paper` 划堆
    在纸面页上同向；**但别把这两个数当成同一个**。
    """
    f = img.astype(np.float32)
    lum = f.mean(axis=2)
    bg = float(np.percentile(lum, 80))
    chroma_mag = np.abs(f - lum[..., None]).max(axis=2)
    cast = f[..., 2] - f[..., 0]
    neutral = (lum < bg * image_ops.INK_DARK) & (chroma_mag < image_ops.INK_CHROMA)
    return {
        "蓝": neutral & (cast > 25),
        "暖": neutral & (cast < -25),
        "中性": neutral & (np.abs(cast) <= 25),
    }


def cast_split(name, marked):
    """夹具必须复现真页那条不对称：蓝偏正文被拟合出覆盖度、暖偏正文一点都没进反演。

    第二十三轮 `--probe` 量到框内暗字 暖/蓝 = 31–320（`blue_in` 恰为 0），第一反应是
    「夹具把彩边画错了」。拿同一口径回头量用户真页（只统计像素，不看内容），18 份文本页
    全是同一个形状：蓝偏正文 coverage p50 **0.186–0.196**、≥`COV_LOW` 的占比 99–100%，
    暖偏正文 ≥`COV_LOW` 的占比 **18/18 份都是 0.000**。也就是说这是产品的既有取舍
    （α 反演分不开「字自己的蓝边」与「字上的水印」，只能剥一边），已写进 §9.3 并交给用户定。
    本函数把那次确认固化成闸门：以后谁把正文彩边画成双边对称、或把章的方向画歪，这里先变红。
    """
    pops = cast_populations(marked)
    plan = image_ops._chroma_plan(marked.copy())
    if plan is None:
        print(f"  偏色不对称 {name}: 无彩章计划（这条夹具不进反演）")
        return True
    bad = []
    shown = []
    for key, band, need in (
        ("蓝", CORPUS_CAST_BLUE_P50, CORPUS_CAST_BLUE_SHARE),
        ("暖", None, CORPUS_CAST_WARM_SHARE),
    ):
        cov = plan.coverage[pops[key]]
        if cov.size == 0:
            if key == "蓝":
                bad.append("蓝偏正文 0px，这张夹具没有可判的正文彩边")
            shown.append(f"{key}=n0")
            continue
        share = float((cov >= COV_LOW).mean())
        p50 = float(np.percentile(cov, 50))
        shown.append(f"{key} n={cov.size} p50={p50:.3f} ≥LOW={share:.3f}")
        if key == "蓝":
            if cov.size < 200:
                bad.append(f"蓝偏正文只有 {cov.size}px，样本太少不足以定形状")
            elif not band[0] <= p50 <= band[1]:
                bad.append(f"蓝偏正文覆盖度 p50 {p50:.3f} 不在真页带 {band}")
            elif not need[0] <= share <= need[1]:
                bad.append(
                    f"蓝偏正文仅 {share:.3f} 进反演（真页 0.99–1.00）——"
                    "章的方向与正文彩边离得太远，夹具失真"
                )
        elif not need[0] <= share <= need[1]:
            bad.append(
                f"暖偏正文有 {share:.3f} 进了反演（真页 0.000）——"
                "夹具把暖边也画到章方向上了，双边对称的彩边不是真页的样子"
            )
    print("  偏色不对称 " + " ".join(shown) + f"  ← {name}")
    check(f"{name} 夹具复现真页的偏色不对称", not bad, "；".join(bad))
    return not bad


def cast_amplitude(name, clean):
    """夹具正文彩边的**振幅分布**必须铺在真页带上，不能挤成一个常数。

    第一版在这里写死 `off = ±32`，为的是让彩边正好越过 `cast_populations()` 的 ±25 计数门。
    后果是整族像素一起摆在门槛上：反演削掉几级，`blue_in` 恰好归零，暖/蓝 于是飞成夹具自己的
    刀口数 31–310（真页修后是 p50 **1.48** / max **4.57**，`work/qa/_r23_cast_amp.txt`）。
    真页框外这一族像素的振幅是 p10/p50/p90 = **28/42/54**、蓝条数占比 **0.525**
    （`work/qa/_r23_cast_band.txt`，18 份文本页合并；框外＝引擎从不碰的那一片，所以它就是真值）。
    要的是**分布**，不是某一个数，所以这里判三个分位点 + 跨度 + 蓝占比。

    **不判 p99**：夹具的镶边模型是「色度 = ⅔·|B−R|」（见 `draw_text` 里那段通道账），
    于是 |B−R| 被 `INK_CHROMA=34` 那道色度门截在 ~51，而真页 p99 到 63——顶那一截在真页上是
    「R 降、B 升、G 跟着亮度走」（|B−R| ≈ 2×色度），夹具画不出来。这是**已知的模型失真**，
    写在这里供复核，不是可以拿去放宽判据掩盖的事。以后谁把彩边改回常数，这里先变红。
    """
    pops = cast_populations(clean)
    both = pops["蓝"] | pops["暖"]
    f = clean.astype(np.float32)
    amp = np.abs(f[..., 2] - f[..., 0])[both]
    n = int(amp.size)
    if n < CORPUS_CAST_MIN_N:
        print(f"  彩边振幅 n={n}（门槛 {CORPUS_CAST_MIN_N}） ← {name}")
        check(f"{name} 夹具彩边样本量足以定分位数", False, f"只有 {n}px，分位数没有意义")
        return False
    q = {key: float(np.percentile(amp, int(key[1:]))) for key in CORPUS_CAST_AMP}
    spread = q["p90"] - q["p10"]
    share = int(pops["蓝"].sum()) / n
    bad = []
    for key, band in CORPUS_CAST_AMP.items():
        if not band[0] <= q[key] <= band[1]:
            bad.append(f"{key} {q[key]:.1f} 不在真页带 {band}")
    if spread < CORPUS_CAST_SPREAD_MIN:
        bad.append(f"p90−p10 只有 {spread:.1f}（真页 26；彩边写成常数时这里是 0）")
    if not CORPUS_CAST_BLUE_POP_SHARE[0] <= share <= CORPUS_CAST_BLUE_POP_SHARE[1]:
        bad.append(f"蓝偏条数占比 {share:.3f} 不在真页带 {CORPUS_CAST_BLUE_POP_SHARE}")
    print(
        f"  彩边振幅 n={n} p10/p50/p90={q['p10']:.1f}/{q['p50']:.1f}/{q['p90']:.1f}"
        f" 跨度={spread:.1f} 蓝占比={share:.3f} ← {name}"
    )
    check(f"{name} 夹具彩边振幅在真页带内", not bad, "；".join(bad))
    return not bad


# --------------------------------------------------------------------------
# 七个夹具：每个都写清它守的是 §9.2 / §9 的哪一条
# --------------------------------------------------------------------------


def fixture_blue_stamp():
    """① 彩色半透明章压正文 + 压在空白纸上 → 逐像素 α 反演（结论 1/2/3/6）。

    三个「按实测放宽」的封顶都在这里说明原因（契约里那句「松了多少要写在 docstring」）。
    实测数取自 `work/qa/_r23_probe6.txt`（章按 `STAMP_ALPHA=0.96` 的观测色对齐真页之后）：

    - `max_ink_moved=260`（实量 **237 px / 最大 35 级**）。这些是离笔画 47px 以外、却被
      `weight` 平滑场（21px 窗口）碰到的正文浅洗。真页同族量是结论 6 记的「该带在 14 页只改
      8,651 px」≈ **618 px/页**，夹具比它小一个量级；配套的 `max_ink_moved_max=40` 才是关键——
      真页镶边色度 p50 30.0 / max 48.3（`_r23_calib.txt`），35 级就是一根彩边的振幅量级，
      超过 40 级说明动的不是边而是笔画本体。
    - `max_lift=560`（实量 **529 px**，口径是「比真值亮 >26 级」）。这是 §9.3 认掉的那笔账：
      章的真实不透明度 0.96 高于 `NOMINAL_COVERAGE=0.90`，压字那部分反解接近奇异，
      「清干净」与「不伤字」互斥、项目选择保字。**要说清这不是从真页抄来的数**：真页同一群像素
      的平均亮度是**降**的（章下字 78.4→76.6，结论 6），夹具这里是个别像素被抬向纸色。
      所以它是**上限**而不是目标值，任何上涨都会被基线逐份抓住。
    - `erased`/`darkened`/`stray`/`stray_ink`/`missed`/`paper_dark` **一律保持默认 0**：
      这几项在真卷上有对应的 0（§11.5「标题等正文区域 0 改动」、`erased` 13 页 0 / 1 页 1），
      没有理由在夹具上放宽。
    """
    rng = np.random.default_rng(11)
    img = canvas(11)
    core, edge = draw_text(img, (70, 90, 1030, 1180), rng)
    clean = img.copy()
    stroke = stamp_ring(img.shape, (560, 640), 150, 22)
    overlay(img, stroke, STAMP_BLUE, STAMP_ALPHA)
    return {
        "id": "stamp",
        "guard": "结论 1/3/8：绝对投影把字算成水印、覆盖顶到 1.0、压在字上的暖边",
        "clean": clean,
        "marked": img,
        "boxes": [Box(390, 470, 730, 810)],
        "stroke": stroke,
        "core": core,
        "edge": edge,
        "ink": core,
        "expect_unblend": 1,
        "expect_masked": 0,
        "expect_dirs": 0,
        "expect_fallback": 0,
        "max_residual": 26.0,
        "max_ripple": 1.60,
        "max_missed": 0,
        "max_ink_moved": 260,
        "max_ink_moved_max": 40,
        "max_lift": 560,
        "max_stray": 0,
    }


def fixture_margin_dots():
    """② 框内一枚实心章 + 左边距等距重复的 4px／2px 小竖条（结论 5/9）。

    竖条一律在框外，只有 `_sweep_stamp_cores` 能清；其中 2px 的过不了 `CORE_OPEN=3` 开运算，
    只能由「这一点和它的邻域确实是空白纸」这条证据（`PAPER_DOT_*`）归位。实心章放在框内，
    为的是让**整页**的 `_chroma_plan` 先成立——补扫的入口条件就是 `page_plan is not None`。

    放宽的两道封顶同样要写原因（实测取自 `work/qa/_r23_probe6.txt`）：
    `max_ink_moved=70`（实量 **61 px / 最大 32 级**）与 `max_lift=175`（实量 **158 px**）
    是 `stamp` 那两条的同一件事、只是这张图框内正文更少，所以数更小、不是判据更松；
    真页同族量 618 px/页（结论 6）仍是上界。`missed`/`paper_dark`/`stray_ink` 保持默认 0——
    4px 竖条必须被补扫清掉、2px 竖条必须被空白纸面证据归位，这正是本夹具要守的两条。
    """
    rng = np.random.default_rng(23)
    img = canvas(23)
    core, edge = draw_text(img, (150, 90, 1030, 1180), rng)
    clean = img.copy()
    stamp = stamp_ring(img.shape, (170, 250), 90, 24)
    bars = np.zeros(img.shape[:2], bool)
    for y in range(430, 1300, 48):
        bars[y : y + 20, 58 : 62] = True  # 4px
        bars[y + 24 : y + 44, 92 : 94] = True  # 2px：开运算滤掉，靠空白纸面证据
    stroke = stamp | bars
    overlay(img, stroke, STAMP_BLUE, STAMP_ALPHA)
    return {
        "id": "dots",
        "guard": "结论 2/4/9：白纸判据的色度条件、补扫门槛与空白纸面归位",
        "clean": clean,
        "marked": img,
        "boxes": [Box(40, 120, 300, 380)],
        "stroke": stroke,
        "core": core,
        "edge": edge,
        "ink": core,
        "expect_unblend": 1,
        "expect_masked": 0,
        "expect_dirs": 0,
        "expect_fallback": 0,
        "max_residual": 22.0,
        "max_ripple": 1.60,
        "max_missed": 0,
        "max_ink_moved": 70,
        "max_ink_moved_max": 38,
        "max_lift": 175,
        "max_stray": 0,
    }


def fixture_gray_mark():
    """③ 浅色文档上的灰白半透明大章（§9 第二条路线 `_neutral_mask`）。

    这条夹具的前提是「整页没有彩色章」，所以正文彩边压到 `sat<45` 门槛以下（`ring_chroma=6`）——
    否则 `_chroma_plan` 会从彩边里拟合出一个假章，`_neutral_mask` 永远走不到。彩边占比与亮度仍按
    真语料带断言，只有色度这一项按本夹具的前提让路（`chromatic=False`）。

    **横杠两端必须缩在圆环里面（470–650），不许碰到圆**：第一版横杠写成 430–690，正好穿过
    `cv2.circle` 最左/最右那两点，于是接头处长出两坨 82 与 86 px 的实心块（行 294–307、
    列 429–440 / 680–692，`--probe` 逐像素定位到的）。那两坨被 `_neutral_mask` 自己的
    `paper_local > 0.5` 前置条件排除（17×17 邻域里同时有圆和杠，量到 p50 0.44），所以它们
    一直是没修的：`空白纸面笔画残差 p99` 恰好 39.0（= 0.34×128+0.66×245≈205.6 与真值 245 之差）、
    `paper_dark` 168 px。**这不是引擎缺陷，是夹具画出了一条引擎按承诺不该修的几何**——
    而这条路线的承诺正是「只取四周都是纸面的像素」（§9 第二条）。把横杠收到离圆 35px，
    杠端的 17×17 邻域非纸占比降到 ~19%（`paper_local`≈0.81），两坨接头消失，
    `max_paper_dark`/`max_resid_paper` 两道硬门就能保持默认的 0 与 12.0，
    不需要为了给夹具让路而放松「空白纸面必须归位」这条判据。
    """
    rng = np.random.default_rng(31)
    img = canvas(31)
    core, edge = draw_text(img, (70, 640, 1030, 1180), rng, ring_chroma=6.0)
    clean = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mark, (560, 300), 130, 255, 5)
    cv2.rectangle(mark, (470, 298), (650, 303), 255, -1)
    stroke = mark > 0
    overlay(img, stroke, (128, 128, 128), 0.34)
    return {
        "id": "gray",
        "guard": "§9 灰白路线：只取比纸面略暗、且四周都是纸面的像素",
        "clean": clean,
        "marked": img,
        "boxes": [Box(400, 140, 720, 460)],
        "stroke": stroke,
        "core": core,
        "edge": edge,
        "ink": core,
        "chromatic": False,
        "expect_unblend": 0,
        "expect_masked": 1,
        "expect_dirs": 0,
        "expect_fallback": 0,
        "max_residual": 16.0,
        "max_ripple": 1.30,
        "max_missed": 0,
        "max_ink_moved": 0,
        "max_stray": 0,
    }


def fixture_corner_white():
    """④ 粉彩插图上的角标白字（§9 第三条路线 `_bright_mask`，结论 7：背景必须开运算估）。

    底色 (225,175,215) 的 sat≈57，正好靠 `paper` 门放宽到 sat<60 才认得出纸面；旁边一块深色插图
    内容用来抓「白字没有路线 → 落到邻域贴片 → 把深色糊进水印区」。
    """
    rng = np.random.default_rng(41)
    img = canvas(41, base=(225, 175, 215), sigma=3.0)
    dark = dark_patch(img, (280, 1150), (150, 110))
    clean = img.copy()
    strokes = np.zeros(img.shape[:2], np.uint8)
    for x in range(760, 1020, 66):
        cv2.rectangle(strokes, (x, 1100), (x + 16, 1240), 255, -1)
        cv2.rectangle(strokes, (x + 6, 1150), (x + 52, 1166), 255, -1)
    core = strokes > 0
    glow = cv2.dilate(strokes, np.ones((7, 7), np.uint8)) > 0
    gf = img.astype(np.float32)
    gf[glow & ~core] = gf[glow & ~core] * 0.55 + 255.0 * 0.45
    img[...] = np.clip(gf, 0, 255).astype(np.uint8)
    stroke = core | (glow & ~core)
    return {
        "id": "corner",
        "guard": "结论 7：亮色路线的背景用开运算估；哨兵把 BRIGHT_DEV_HIGH 抬到 60，alpha 就压回 1px 骨架、辉光整圈留下",
        "clean": clean,
        "marked": img,
        "boxes": [Box(740, 1080, 1060, 1270)],
        "stroke": stroke,
        "core": np.zeros(img.shape[:2], bool),
        "edge": np.zeros(img.shape[:2], bool),
        "no_realism": "这张是插图页，没有正文笔画，realism 带的对象是文字页",
        "ink": dark,
        "expect_unblend": 0,
        "expect_masked": 1,
        "expect_dirs": 0,
        "expect_fallback": 0,
        "max_residual": 24.0,
        "max_ripple": 1.45,
        "max_missed": 0,
        "max_ink_moved": 0,
        "max_stray": 0,
    }


def fixture_tiled_thin():
    """⑤ 斜向平铺细字样压住正文（结论 8：只要动色度就必须两边一起动）。

    七张里唯一**选框外也大面积有水印**的一张（框只盖到 y=620，平铺线一路铺到 1180），
    所以它同时验「框外补扫」与「暖色晕中性化」两条在框外生效的路线。实测 `_r23_probe7.txt`：
    框外笔画 56,710px、补扫 swept=57,582、`stray`/`stray_ink`/`blank_blue`/`missed` 全 0，
    `warm_wiped`=975（字形自带的暖边被中性化洗掉，逐像素分类：975 全在压字处、空白纸上 0；
    第一版把它按「比真值蓝 25 级」误报成「水印没清掉 768px」，拆开定义的经过写在 metrics 那段注释里）。

    只有两条上限比 `contract()` 的默认松，各有实测理由：
    - `max_lift` 900（实测 844）：章下字被抬亮＝§9.3 认掉的取舍（真实不透明度 0.96 高于
      拟合上限 `NOMINAL_COVERAGE=0.90`，压字处反解奇异 → 选择保字留晕）。这张框内∩笔画
      43,287px 是 stamp 那张 24,670px 的 1.75 倍，所以数也同比更大（844 对 529）。
    - `max_lift_out` 950（实测 885）：默认 0 的「框外不许抬亮正文」在这张上不可能为 0，
      因为**水印本身就在框外**，压着正文的那一段只能靠结论 6 那条「亮度亏损 + 足迹门」在框外补。
      真卷同口径的对照数：结论 6 记的该带 14 页改 8,651px、结论 9 记的同一张试卷页框外
      高覆盖像素 15,825px——885 落在这些数的量级之内，它拦的是「足迹门失控」而不是「线画密了」。
    """
    rng = np.random.default_rng(53)
    img = canvas(53)
    core, edge = draw_text(img, (70, 90, 1030, 1180), rng)
    clean = img.copy()
    lines = np.zeros(img.shape[:2], np.uint8)
    for k in range(-1400, 2600, 60):
        cv2.line(lines, (k, 90), (k + 1090, 1180), 255, 2)
    stroke = lines > 0
    overlay(img, stroke, STAMP_BLUE, STAMP_ALPHA)
    return {
        "id": "tiled",
        "guard": "结论 5/8：开运算核加大（CORE_OPEN=5）会丢掉平铺本体、暖色晕不外扩（WARM_HALO_GROW=1）只剩暖边",
        "clean": clean,
        "marked": img,
        "boxes": [Box(60, 80, 1040, 620)],
        "stroke": stroke,
        "core": core,
        "edge": edge,
        "ink": core,
        "expect_unblend": 1,
        "expect_masked": 0,
        "expect_dirs": 0,
        "expect_fallback": 0,
        "max_residual": 30.0,
        "max_ripple": 1.70,
        "max_missed": 0,
        "max_ink_moved": 0,
        "max_stray": 0,
        "max_lift": 900,
        "max_lift_out": 950,
    }


def fixture_photo_single_blob():
    """⑥ 立绘上一整块蓝色花瓣（只有一个连通域）→ 不许被认成水印（MIN_SPOTS=2）。

    块的大小按 `_chroma_plan` 的 `saturated` 占比门槛挑（真语料带 [0.003, 0.42] 之内），
    否则这条夹具验的是「太大所以被拒」而不是「只有一个连通域所以被拒」——那是两回事。

    **纸面噪声 σ 必须是 1.5**（第一版写的 4.0 让这条夹具量错了对象）：`_neutral_mask` 的
    门槛里没有一个 `MIN_SPOTS`，它靠 `mask.mean<0.002` 与 `deviation>11` 挡误报，而 σ=4 的
    噪声自己就能在 11 级偏差上凑出 >0.002 的占比——量到的 `masked=1` 是「噪声被当水印」，
    跟「只有一个连通域」无关。σ=1.5 之下 `_neutral_mask` 判不成膜、`_chroma_plan` 的
    `MIN_SPOTS=2`（image_ops.py:250）才是拦住这一块的那道闸，`--canaries` 里把它降到 1
    必须让这条夹具变红。
    """
    rng = np.random.default_rng(67)
    img = canvas(67, base=(210, 205, 200), sigma=1.5)
    clean = img.copy()
    blob = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(blob, (310, 400), (60, 45), 20, 0, 360, 255, -1)
    stroke = blob > 0
    f = img.astype(np.float32)
    f[stroke] = np.array((24, 96, 190), np.float32) * 0.9 + f[stroke] * 0.1
    img[...] = np.clip(f, 0, 255).astype(np.uint8)
    return {
        "id": "blob",
        "guard": "MIN_SPOTS=2：单个彩色块不是水印，降到 1 就会把它反演掉",
        "clean": clean,
        "marked": img,
        "boxes": [Box(200, 320, 420, 480)],
        "stroke": stroke,
        "core": np.zeros(img.shape[:2], bool),
        "edge": np.zeros(img.shape[:2], bool),
        "no_realism": "同上：立绘页没有正文笔画",
        "ink": np.zeros(img.shape[:2], bool),
        "skip_quality": "这里不许修，只许走贴片——残差/漏清没有意义",
        "expect_unblend": 0,
        "expect_masked": 0,
        "expect_dirs": 1,
        "expect_fallback": 0,
        "max_residual": 255.0,
        "max_ripple": 9.99,
        "max_missed": 10**9,
        "max_ink_moved": 0,
        "max_stray": 0,
    }


def fixture_wash_seal():
    """⑦ 印泥不匀的章压正文——与 ① 同一条几何、同一张纸，只差章的 α 有梯度（结论 6）。

    **为什么要第七张**：结论 6 那三件套的第一环「亮度亏损」在 ①–⑥ 上**一次都没点着过火**。
    压在字上时 `deficit = paper_lum − cov·(paper_lum − colour_lum) − lum ≈ 219.3 − 210.7α`
    （`cov ≈ 0.9375α`、`INK_LUM=22.7`、paper≈242），要越过 `TEXT_DEFICIT_LOW=30` 就得 α<0.898。
    ① 的章是均匀 `STAMP_ALPHA=0.96`，deficit 恒为 17 → `band` 里只剩暖色晕那一支漏进来的 406px。
    所以 `--canaries` 第一版把 `TEXT_DEFICIT_LOW` 调到 1e9 时 ①/⑤ 都报「与正常跑一字不差」
    （`work/qa/_r23_mutants.txt`）——**不是判据钝，是这条代码在语料里没有正例**。
    真页上是有的：§9.2 结论 6 记的「该带在 14 页只改 8,651px」量的正是这一段。
    于是把章画淡——但只画**空间梯度**、不整体平移 α：浓端仍是 0.96，`_chroma_plan` 的
    `anchor`（核心云团 raw 投影前 10% 的均值）取在浓段，`cast_split` 那条真页带照旧成立。

    **淡端为什么必须到 0.62**（`work/qa/_r23_calib_wash3.txt`、`_r23_calib_wash4.txt`）：
    lo=0.78/0.84 时把 `TEXT_DEFICIT_LOW` 关掉**指标一字不差**，0.70 只动 0.09，
    0.62 才量到 resid_ink 23.34→25.14。也就是说这张夹具的可用性与其说来自「淡」，
    不如说来自「淡到恰好让第一环有活干」。

    **换来的一条实测边界（`max_stray=3` 的原因，不是放宽判据）**：章整体偏淡会把
    `_chroma_plan` 的 `tint` 拖向纸色、`denom=direction@direction` 变小，于是
    `raw=chroma·direction/denom` 对**全页**正文彩边的 coverage 一起抬高——实测有 3 个像素
    因此越过 `CORE_SWEEP=0.30`（3×3 开运算之后仍在），被框外补扫改掉最多 72 级。
    位置 (386–388, 1085–1086)，**离章 315px**，真值 B−R=36/53/56、成品被中和到 9/−9/−9，
    即字的蓝镶边。这与 `NOMINAL_COVERAGE=1.0` 那个突变体**同形**（同样 stray 3/72 级）——
    同一个「coverage 整体抬高」机制的两条来路，所以这是结论 4/5 那条护栏在淡章上的真实边界。
    三条把它圈住的下界保持 0，不许跟着松：`stray_ink=0`（那 3 px 是镶边，不是正文本体）、
    `lift_out=0`（框外没有一个字被抬亮）、`blank_blue=0`。lo 每往 0.78 收一格 stray 就归 0，
    但那样第一环又没有正例了——**这个取舍就写在这里，别用放宽 `CORE_SWEEP` 去两头讨好。**

    实测（`work/qa/_r23_probe_wash.txt` + `_r23_probe_stray.txt`）：unblend=1（走反演，与 ① 同一条路线）、
    swept=3、ink_moved=246/37 级、lift_in=517、residual=10.24、ripple=0.964、
    resid_paper=(5.82, 9.0)、resid_ink=(23.34, 220.0)、paper_dark=0、missed=0、blank_blue=0、
    warm_in=219 / warm_truth=850（结论 8 的暖边中和 74%）。三个真页保真闸门全过：
    形状（ink 占比 0.0119、ring 色度 26.67、暗字暖/蓝 1.0119）、偏色不对称（蓝 p50 **0.217**
    在 0.16–0.225 带内、暖 0.000）、彩边振幅 28/38/47（跨度 19、蓝占比 0.505）。
    上限的取法与其它夹具一致：**取实测值与「应达到的样子」之间最紧的那一档**，
    留出的余量都写在下面的注释里，任何一个上限被突变体越过才算哨兵成立。
    """
    rng = np.random.default_rng(11)
    img = canvas(11)
    core, edge = draw_text(img, (70, 90, 1030, 1180), rng)
    stroke = stamp_ring(img.shape, (560, 640), 150, 22)
    clean = overlay_faded(img, stroke, STAMP_BLUE, WASH_ALPHA_LO, WASH_ALPHA_HI, 12)
    return {
        "id": "wash",
        "guard": "结论 6：亮度亏损 + 足迹门（章淡到 α<0.90 那一段才点得着火）",
        "clean": clean,
        "marked": img,
        "boxes": [Box(390, 470, 730, 810)],
        "stroke": stroke,
        "core": core,
        "edge": edge,
        "ink": core,
        "expect_unblend": 1,
        "expect_masked": 0,
        "expect_dirs": 0,
        "expect_fallback": 0,
        # 与 ① 的 11.22 同量级；残差由「压字那一半」主导，所以真正的判据在 max_resid_ink
        "max_residual": 12.0,
        # 实测 0.964：>1 就是引擎自己造高频（§8 那条振铃判据的图片版），余量 0.036
        "max_ripple": 1.0,
        "max_missed": 0,
        # 实测 246；`TEXT_FOOT_MED=3`（足迹门放宽）量到 278 → 上限 260 正是把这两档分开
        "max_ink_moved": 260,
        # 实测 37 级；彩边振幅 p90=47，正文改动幅度必须明显小于一整条彩边
        "max_ink_moved_max": 40,
        # 实测 517（① 是 529）：§9.3 认掉的取舍，章下字允许被抬亮，但不许多出 80px
        "max_lift": 600,
        "max_resid_ink": 24.0,
        # 见上面那段：淡章让 coverage 整体抬高，3 个字边像素进补扫；与突变体同形
        "max_stray": 3,
    }


FIXTURES = (
    fixture_blue_stamp,
    fixture_margin_dots,
    fixture_gray_mark,
    fixture_corner_white,
    fixture_tiled_thin,
    fixture_photo_single_blob,
    fixture_wash_seal,
)


def run_fixture(fx):
    """跑一个夹具，返回按真值算出来的指标。"""
    before = fx["marked"].copy()
    after = fx["marked"].copy()
    detail = remove_boxes_inplace(after, fx["boxes"], "smart")
    clean, stroke, boxes = fx["clean"], fx["stroke"], fx["boxes"]
    ink, bg = ink_of(clean)
    boxm = boxes_mask(boxes, after.shape)
    out_box = ~boxm
    # 两圈影响半径各有各的判据，不能合成一圈：
    # halo  = 正文允许带一丝色晕的范围（§9.3 量到 23px），超出就必须逐像素相同；
    # tight = 框外允许被补扫/羽化碰到的范围（紧贴笔画 ±6px），超出就是越界改动。
    halo = _dilate(stroke, INK_GUARD)
    tight = _dilate(stroke, STRAY_GUARD)
    moved = _delta(clean, after).max(axis=2) > 2
    dlum = dlum_delta(clean, after)
    erased, darkened = ink_damage(before, after)
    res_mean, res_max = residual(clean, after, stroke)
    # 水印笔画分两半，残差分开封顶：压在空白纸上的那一半理论上能完全归位（结论 3/9），
    # 压在字形上的那一半是 §9.3 认掉的取舍——混在一个数里，前者会把后者平均掉。
    over_ink = stroke & _dilate(ink, 9)
    on_paper = stroke & ~over_ink
    iw, ib = warm_blue(after, boxm, bg)
    ow, ob = warm_blue(after, out_box, bg)
    tw, tb = warm_blue(clean, boxm, bg)
    still_blue = _cast(after) - _cast(clean)
    in_paper = on_paper & boxm
    # 「该清清不掉」必须同时要求成品在绝对意义上偏蓝。只按「比真值蓝」判会把相反的一种情况算进来：
    # 字形自带的 JPEG 暖边被结论 8 的暖色晕中性化洗掉——tiled 现场实测 cast 从 −43 走到 −8，
    # 差分 +35 却一克蓝水印都不剩。第一版就在这儿把 768px 报成「水印没清掉」，
    # 逐像素分类后拆成两条：那 768px 全部是 cast(clean)<−25 且 cast(after)>−25，记进 warm_wiped。
    missed_px = stroke & out_box & (still_blue > 25) & (_cast(after) > 25)
    # 改动幅度（对「原图」，不是对真值）只服务两条「按级数封顶」的判据
    delta_max = _delta(before, after).max(axis=2)
    metrics = {
        "unblend": detail["unblend_regions"],
        "masked": detail["masked_regions"],
        "dirs": len(detail["directions"]),
        "fallback": detail["fallback_regions"],
        "swept": detail["swept_pixels"],
        "erased": erased,
        "darkened": darkened,
        # 真值正文像素（离笔画 halo 圈以外）被改动 —— 比 erased/darkened 严得多。
        # 计数与幅度成对封顶：`max_ink_moved` 容忍的是「平滑后的 weight 场在正文浅色洗」，
        # 而 `max_ink_moved_max` 把这种洗的**幅度**钉在彩边振幅的一小半以内
        # （真页镶边色度 p50 30.0 / max 48.3，见 `_r23_calib.txt`）——
        # 只有计数没有幅度，等于允许「整页字被抬亮一片」；只有幅度没有计数，等于允许「洗的字越来越多」。
        "ink_moved": int(np.count_nonzero(moved & ink & ~halo)),
        "ink_moved_max": int(delta_max[moved & ink & ~halo].max(initial=0)),
        # 框外、且不在笔画紧影响圈里的改动 —— §11.2 那条「框外差异」的自动版
        "stray": int(np.count_nonzero(moved & out_box & ~tight)),
        # 框外正文被改动：结论 4/5 的承诺是「标题等正文区域 0 改动」，这一项必须是 0
        "stray_ink": int(np.count_nonzero(moved & ink & out_box & ~halo)),
        # 框外笔画像素修完仍然**绝对偏蓝** = 该清清不掉（补扫失效就在这）
        "missed": int(np.count_nonzero(missed_px)),
        # 结论 8 的另一侧：框外字形自带的暖边被一并中性化的像素数。刻意**不封顶**——
        # 真语料没有同口径的对照数（§9.2 结论 6 只记了框内「暖 11,569→864」，框外没有），
        # 所以这一轮只让它进基线被「与基线逐字段相同」那条盯着，跳变一样会红。
        "warm_wiped": int(
            np.count_nonzero(stroke & out_box & (_cast(clean) < -25) & (_cast(after) > -25))
        ),
        "blank_blue": int(
            np.count_nonzero(stroke & out_box & (_cast(after) > 40))
        ),
        "residual": round(res_mean, 2),
        "residual_max": round(res_max, 1),
        "ripple": round(ripple(clean, after, boxm), 3),
        "warm_in": iw,
        "blue_in": ib,
        "warm_out": ow,
        "blue_out": ob,
        "warm_truth": tw,
        "blue_truth": tb,
        "stray_max": int(delta_max[out_box & ~tight].max(initial=0)),
        "residual_p99": round(float(np.percentile(np.abs(dlum[stroke]), 99)), 1) if stroke.any() else 0.0,
        "resid_paper": _stats(dlum, in_paper),
        "resid_ink": _stats(dlum, over_ink & boxm),
        # 压在空白纸上的笔画修完仍比真值暗 25 级以上 = 没归位到纸色（结论 3 的 snap 失效）
        "paper_dark": int(np.count_nonzero(in_paper & (dlum < -25))),
        # 章下字被抬亮的量：§9.3 的已知取舍，按实测封顶，任何上涨都会被基线抓住
        "lift_in": int(np.count_nonzero(ink & boxm & (dlum > 26))),
        "lift_out": int(np.count_nonzero(ink & out_box & (dlum > 26))),
    }
    fx["_after"] = after
    fx["_missed"] = missed_px
    fx["_moved"] = moved
    fx["_halo"] = halo
    fx["_tight"] = tight
    fx["_out_box"] = out_box
    return metrics


def contract(fx, m):
    bad = []
    if m["unblend"] != fx["expect_unblend"]:
        bad.append(f"反演路线 {m['unblend']}≠{fx['expect_unblend']}")
    if m["masked"] != fx["expect_masked"]:
        bad.append(f"掩膜路线 {m['masked']}≠{fx['expect_masked']}")
    if m["dirs"] != fx["expect_dirs"]:
        bad.append(f"贴片路线 {m['dirs']}≠{fx['expect_dirs']}")
    if m["fallback"] != fx["expect_fallback"]:
        bad.append(f"内容填充兜底 {m['fallback']}≠{fx['expect_fallback']}")
    if m["erased"] > 0:
        bad.append(f"正文被抹 {m['erased']}")
    if m["darkened"] > 0:
        bad.append(f"正文被压深 {m['darkened']}")
    if m["ink_moved"] > fx["max_ink_moved"]:
        bad.append(
            f"真值正文被改动 {m['ink_moved']}px>{fx['max_ink_moved']}"
            f"(最大 {m['ink_moved_max']} 级)"
        )
    elif m["ink_moved_max"] > fx.get("max_ink_moved_max", 12):
        bad.append(
            f"真值正文被改动幅度 {m['ink_moved_max']}级>"
            f"{fx.get('max_ink_moved_max', 12)}({m['ink_moved']}px)"
        )
    if m["stray"] > fx["max_stray"]:
        bad.append(f"框外越界改动 {m['stray']}px>{fx['max_stray']}(最大 {m['stray_max']} 级)")
    # 结论 4/5 那句「标题等正文区域必须 0 改动」——这一项不许按实测放宽
    if m["stray_ink"] > fx.get("max_stray_ink", 0):
        bad.append(f"框外正文被改动 {m['stray_ink']}px>{fx.get('max_stray_ink', 0)}")
    if not fx.get("skip_quality"):
        if m["missed"] > fx["max_missed"]:
            bad.append(f"框外水印没清掉 {m['missed']}px>{fx['max_missed']}")
        if m["residual"] > fx["max_residual"]:
            bad.append(f"笔画残差 {m['residual']}>{fx['max_residual']}")
        if m["ripple"] > fx["max_ripple"]:
            bad.append(f"振铃比值 {m['ripple']}>{fx['max_ripple']}")
        # 压在空白纸上的那一半笔画理论上能完全归位（结论 3 的 snap / 结论 9），
        # 所以这里按真值封顶；压字那一半走 max_lift，两条合在一起才既防「留蓝」又防「洗白」。
        # 三个上限的默认值就是「应该达到的样子」；只有实测确实达不到的才在夹具里写松，
        # 松了多少都要在夹具的 docstring 里说明原因，否则等于把判据悄悄关掉。
        if m["resid_paper"][1] > fx.get("max_resid_paper", 12.0):
            bad.append(f"空白纸面笔画残差 p99 {m['resid_paper'][1]}>{fx.get('max_resid_paper', 12.0)}")
        # 「压在字上的那一半笔画」相对真值平均偏多远——§9.3 那条取舍的尺寸。
        # 默认不设上限（各夹具的章浓淡不同，写一个公共上限等于替别人定标准）；
        # 只有 ⑦ 声明它，因为只有那张夹具让结论 6 的「亮度亏损」分支真有活干：
        # 关掉 `TEXT_DEFICIT_LOW` 时 ①–⑥ 一字不变，⑦ 的这一栏从 23.34 涨到 25.14。
        if "max_resid_ink" in fx and m["resid_ink"][0] > fx["max_resid_ink"]:
            bad.append(
                f"字上笔画残差均值 {m['resid_ink'][0]}>上限 {fx['max_resid_ink']}"
                f"（p99 {m['resid_ink'][1]}）"
            )
        if m["paper_dark"] > fx.get("max_paper_dark", 0):
            bad.append(f"空白纸面笔画没归位 {m['paper_dark']}px>{fx.get('max_paper_dark', 0)}")
        if m["lift_in"] > fx.get("max_lift", 0):
            bad.append(f"章下字被抬亮 {m['lift_in']}px>{fx.get('max_lift', 0)}")
        # 框外正文被抬亮：结论 4 那句「整页扫会让字变细发灰」的自动版。默认必须是 0——
        # 这是「框外不动字」最硬的一条判据（§11.5 真卷口径：框外『中性正文抬亮>8』合计 13px），
        # 放宽的夹具要在 docstring 里写清为什么这一张的框外本来就该动。
        if m["lift_out"] > fx.get("max_lift_out", 0):
            bad.append(f"框外正文被抬亮 {m['lift_out']}px>{fx.get('max_lift_out', 0)}")
    # 结论 8 的正题：只要修补动了色度，就必须两边一起动。判据是「框内暗字的**暖边**计数
    # 相对真值降下来」，不是旧稿写的暖/蓝比值——比值在这套夹具里分母恒 0：章是合成纯色，
    # 蓝边全被反演中和（实测 stamp blue_in=0 而真值 828），任何比值都是几十到上百，
    # 只有真语料「蓝 945 不变」那条基准上才成立；而那句话恰恰说明夹具的蓝侧与真页不同源
    # （已作为保真残留写进 §9.2 那条，不在这里假装成立）。
    # 旧注释担心的假通过——「把框内暗字整体抬出暗字集合，两个计数一起掉」——由上面三条各管一段：
    # lift_in（抬亮>26 级）、erased（对原图抬亮>26）、ink_moved_max（改动幅度），不需要比值下界再兜一遍。
    if m["warm_in"] > 0.5 * m["warm_truth"]:
        bad.append(
            f"框内暗字暖边只中和掉 {(1 - m['warm_in'] / max(m['warm_truth'], 1)) * 100:.0f}%："
            f"还剩 {m['warm_in']}px > 真值 {m['warm_truth']}px 的一半"
            f"（蓝 {m['blue_in']}/{m['blue_truth']}）——结论 8：动了色度就得两边一起动"
        )
    return bad


def probe(fx, m):
    """改动落点分解 + 未设上限前的真实量：调阈值时用，判据不看这里。"""
    moved, halo, tight, out_box = fx["_moved"], fx["_halo"], fx["_tight"], fx["_out_box"]
    stroke, ink = fx["stroke"], fx["ink"]
    d = _delta(fx["clean"], fx["_after"]).mean(axis=2)
    inside = ~out_box
    cats = {
        "框内∩笔画": moved & inside & stroke,
        "框内∩笔画紧圈": moved & inside & tight & ~stroke,
        "框内∩正文(离紧圈)": moved & inside & ink & ~tight,
        "框内∩其它": moved & inside & ~tight & ~ink,
        "框外∩笔画": moved & out_box & stroke,
        "框外∩笔画紧圈": moved & out_box & tight & ~stroke,
        "框外∩正文": moved & out_box & ink & ~halo,
        "框外∩其它": moved & out_box & ~tight & ~ink & ~stroke,
    }
    parts = []
    for label, sel in cats.items():
        n = int(np.count_nonzero(sel))
        parts.append(f"{label}={n}" + (f"/{d[sel].max():.0f}" if n else ""))
    print(f"  probe {fx['id']}: " + " ".join(parts))
    print(
        f"    量: residual={m['residual']} p99={m['residual_p99']} max={m['residual_max']} "
        f"ripple={m['ripple']} missed={m['missed']} blank_blue={m['blank_blue']} "
        f"ink_moved={m['ink_moved']}/{m['ink_moved_max']}级 stray={m['stray']}/{m['stray_max']} stray_ink={m['stray_ink']} "
        f"swept={m['swept']} 纸面残差={m['resid_paper']} 字上残差={m['resid_ink']} "
        f"paper_dark={m['paper_dark']} lift_in={m['lift_in']} lift_out={m['lift_out']} "
        f"暖/蓝真值={m['warm_truth']}/{m['blue_truth']}"
    )
    if m["missed"] and not fx.get("skip_quality"):
        ys, xs = np.where(fx["_missed"])
        print(f"    没清掉 {len(ys)}px 第一段 y {ys.min()}-{ys.max()} x {xs.min()}-{xs.max()}")
    if m["warm_wiped"]:
        # 结论 8 那一侧的落点：暖边被中性化洗掉的框外笔画像素（真值偏暖、成品不再偏暖）。
        # 按「压在字上 / 落在空白纸」分开报：前者是设计允许的副作用（§9.2 结论 6/8 那条带），
        # 后者不该出现——空白纸上的笔画没有自己的暖边可洗。
        wiped = stroke & out_box & (_cast(fx["clean"]) < -25) & (_cast(fx["_after"]) > -25)
        on_ink = int(np.count_nonzero(wiped & _dilate(ink, 9)))
        print(
            f"    暖边被洗 warm_wiped={m['warm_wiped']}px"
            f"（压在字上 {on_ink} / 空白纸上 {m['warm_wiped'] - on_ink}）"
        )


# 结论 1 禁的是一种**写法**（「离纸面多远的绝对投影」），不是某个数——没有常量可改坏。
# 这两处正则就是那个写法：image_ops.py:217 把 tint 与 background 各减掉自己的均值，
# direction 才是零均值向量（色度与它的点积因此与亮度无关）；224-226 再把纸面中位数钉在 0。
# 两处一起换成锚在纸色上的绝对向量，才等价于 §9.2 第一条描述的那种投影。
_ABSOLUTE_PROJECTION = (
    (
        re.compile(r"direction = \(tint - tint\.mean\(\)\) - \(background - background\.mean\(\)\)"),
        "direction = tint - background",
    ),
    (
        re.compile(
            r"raw = chroma\.dot\(direction\) / denom - float\(\s*"
            r"np\.median\(chroma\[paper\]\.dot\(direction\) / denom\)\s*\)"
        ),
        "raw = (f - background).dot(direction) / denom",
    ),
)


def patch_absolute_projection():
    """装结论 1 的哨兵：改 `_chroma_plan` 的函数体，回一个复原动作。

    改的是**当前源码**而不是抄一份旧实现，所以引擎日后重排这两行时这里会直接报
    「写法变了」——宁可哨兵当场炸给人看，也不要悄悄退化成一个什么都不改坏的补丁。
    """
    src = inspect.getsource(image_ops._chroma_plan)
    for pattern, repl in _ABSOLUTE_PROJECTION:
        if not pattern.search(src):
            raise AssertionError(
                "_chroma_plan 里那两处零均值锚定的写法变了，结论 1 的哨兵要跟着改："
                f"{repl!r} 这条正则已经匹配不上了"
            )
        src = pattern.sub(repl, src)
    original = image_ops._chroma_plan
    namespace = dict(image_ops.__dict__)
    exec(compile(src, "<canary-结论1>", "exec"), namespace)
    image_ops._chroma_plan = namespace["_chroma_plan"]
    return lambda: setattr(image_ops, "_chroma_plan", original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="把本次指标写成基线文件")
    ap.add_argument("--canaries", action="store_true", help="逐条把常量改坏，断言对应夹具必须变红")
    ap.add_argument("--probe", action="store_true", help="打印改动落点分解与未封顶的真实量")
    ap.add_argument("--keep", action="store_true", help="落盘 _ie_*.png 对照图")
    args = ap.parse_args()

    print(f"语料份数：{len(FIXTURES)}（全部脚本合成，不读用户素材）")
    measured = {}
    builders = {}
    for build in FIXTURES:
        fx = build()
        builders[fx["id"]] = build
        if fx.get("no_realism"):
            print(f"  真值形状 {fx['id']}: 跳过（{fx['no_realism']}）")
        else:
            realism(fx["id"], fx["clean"], fx["core"], fx["edge"], fx.get("chromatic", True))
        # 只有真会走 α 反演的那几张，才需要「偏色不对称」这条保真闸门：
        # 走掩膜/贴片的夹具根本不读覆盖度，它画成双边对称也不影响判据。
        if fx["expect_unblend"] == 1:
            cast_split(fx["id"], fx["marked"])
        # 彩边振幅判的是「正文边画得像不像真页」，所以对象是画了正文的文本页夹具：
        # corner/blob 没有正文笔画，gray 按自己的前提把彩边压到 sat<45 以下，三张都不进这条。
        if not fx.get("no_realism") and fx.get("chromatic", True):
            cast_amplitude(fx["id"], fx["clean"])
        m = run_fixture(fx)
        measured[fx["id"]] = m
        bad = contract(fx, m)
        print(
            f"{fx['id']} boxes={len(fx['boxes'])} unblend={m['unblend']} masked={m['masked']} "
            f"dirs={m['dirs']} fallback={m['fallback']} swept={m['swept']} erased={m['erased']} "
            f"darkened={m['darkened']} ink_moved={m['ink_moved']} stray={m['stray']} "
            f"stray_ink={m['stray_ink']} missed={m['missed']} blank_blue={m['blank_blue']} "
            f"warm_wiped={m['warm_wiped']} "
            f"residual={m['residual']} 纸面={m['resid_paper']} 字上={m['resid_ink']} "
            f"lift_in={m['lift_in']} paper_dark={m['paper_dark']} ripple={m['ripple']} "
            f"暖/蓝 框内 {m['warm_in']}/{m['blue_in']} 真值 {m['warm_truth']}/{m['blue_truth']} "
            f"框外 {m['warm_out']}/{m['blue_out']} 守[{fx['guard']}]"
        )
        if args.probe:
            probe(fx, m)
        check(f"{fx['id']} 契约（路线/正文/越界/漏清/残差/色偏）", not bad, "；".join(bad))
        if args.keep:
            pair = np.concatenate(
                [fx["clean"], np.full((fx["clean"].shape[0], 6, 3), 255, np.uint8), fx["marked"]], axis=1
            )
            cv2.imwrite(os.path.join(QA, f"_ie_{fx['id']}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

    if args.canaries:
        # 每一项都是先拿一次性扫描逐条试过才留下的（留档 `work/qa/_r23_mutants.txt` /
        # `_r23_mutants2.txt`）：第二、三元素是「改坏哪个常量到哪个值」与「用哪份夹具收」。
        # 第一项写成函数时，它是函数体型哨兵（结论 1 那种没有常量的），第二元素当标签用。
        # 不红的组合一律不进这里——留一条永远通过的哨兵比留空更糟，它会让下一轮以为那条结论有人守着。
        canaries = (
            ("CORE_OPEN", 5, "tiled", "结论 5：开运算核加大 → 细平铺水印本体进不了补扫种子"),
            ("INK_CHROMA", 1e9, "dots", "结论 2：白纸判据去掉色度条件 → 水印本体被当正文，空白纸面点不再归位"),
            (patch_absolute_projection, "绝对投影", "stamp", "结论 1：投影改成锚在纸色上的绝对向量 → 正文被算成高覆盖水印"),
            ("CORE_SWEEP", 0.05, "dots", "结论 4：框外补扫门槛放到 0.05 → 正文的 JPEG 彩边被卷进补扫"),
            ("TEXT_FOOT_MED", 3, "wash", "结论 6 足迹门：中值降到 3px → 水印本体笔画也进 band，动到真值正文"),
            ("TEXT_DEFICIT_LOW", 1e9, "wash", "结论 6 亮度亏损：判据关掉 → 淡章段下的字被当白纸提亮，字上残差涨"),
            ("BRIGHT_DEV_HIGH", 60.0, "corner", "结论 7：亮色渐变带上界抬到 60 → alpha 压成骨架，白字辉光整圈留下"),
            ("WARM_HALO_GROW", 1, "tiled", "结论 8：暖色晕不外扩 → 框内暗字只剩暖边，只中和三成"),
            ("WARM_HALO_GROW", 1, "stamp", "结论 8：同上，压在字上的暖边"),
            ("NOMINAL_COVERAGE", 1.0, "stamp", "结论 3：覆盖顶到 1.0 → 反解奇异，白纸笔画被归位过头"),
            ("PAPER_DOT_LUM", 0.999, "dots", "结论 9：邻域白像素占比顶到 1.0 → 空白纸面归位那条整个失效"),
            ("MIN_SPOTS", 1, "blob", "门槛：单个彩色块被认成水印（立绘上的花与眼睛）"),
        )
        print("\n哨兵相位（每条都必须让对应夹具变红；不红说明那条结论没有测试守着）")
        for name, value, fid, why in canaries:
            target = builders[fid]()
            if isinstance(name, str):
                label = f"{name}={value}"
                original = getattr(image_ops, name)
                setattr(image_ops, name, value)
                restore = lambda key=name, old=original: setattr(image_ops, key, old)
            else:
                label = f"{name.__name__}({value})"
                restore = name()
            try:
                m = run_fixture(target)
                bad = contract(target, m)
            finally:
                restore()
            check(
                f"改坏 {label} 必须被 {fid} 抓到",
                bool(bad),
                f"没抓到 → {why}" if not bad else f"抓到：{bad[0]}",
            )

    if args.baseline:
        with open(BASELINE, "w", encoding="utf-8") as handle:
            json.dump(measured, handle, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"基线已写入 {os.path.basename(BASELINE)}（{len(measured)} 份）")
    elif os.path.isfile(BASELINE):
        with open(BASELINE, encoding="utf-8") as handle:
            base = json.load(handle)
        # 两侧必须走同一个口径再比：resid_paper/resid_ink 在内存里是元组，
        # 落进 JSON 再读回来就是数组，直接 != 会让七份夹具天天报「不同」，
        # 而这条判据一旦长期是红的，就真的没人看它了。
        now = json.loads(json.dumps(measured))
        diff = [k for k in now if base.get(k) != now[k]]
        check(f"与基线逐字段相同（{len(measured)} 份夹具）", not diff, f"不同 {len(diff)} 份：{','.join(diff)}")
        for key in diff[:3]:
            print("  基线:", base.get(key))
            print("  本次:", now[key])
    else:
        print("（还没有基线文件；先跑 --baseline）")

    print(f"\n共 {len(results)} 项，失败 {len(fails)} 项")
    for line in fails:
        print("  FAIL", line)
    print("结论=" + ("全部通过" if not fails else f"失败 {len(fails)} 项"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
