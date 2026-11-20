"""对比不同去水印路线在「被水印盖住的字」上的表现。

只输出像素计数与掩膜统计，不读取、不放大页面内容。
路线：
  A 本项目引擎     remover.image_ops.remove_boxes_inplace(smart)
  B 全局反演       固定 alpha 拆解，全页应用（开源 reverse-alpha 类脚本的典型做法）
  C 颜色替换       选中蓝色像素直接换成纸色（简易工具/手机 App 常见做法）
  D 内容填充       OpenCV inpaint（Photoshop 内容识别填充 / IOPaint 一类的重绘思路）
"""

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from remover.detection import detect_watermark  # noqa: E402
from remover.image_ops import (  # noqa: E402
    NOMINAL_COVERAGE,
    _chroma_plan,
    _smoothstep,
    remove_boxes_inplace,
    regions_to_boxes,
)

QA = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.abspath(os.path.join(QA, "..", "uploads"))


def corpus():
    """语料在运行时发现，不写死任何会话号或素材号。

    这份表（§10.1）唯一能观测的对象是「真实页面上被水印压住的字」，所以它读的是
    work/uploads 里用户上传的素材；而 `cleanup_old_files()` 按 72 小时整批删会话目录，
    语料会自己蒸发。这里因此不写死任何 id——没了就如实报「未验」，不拿合成图冒充。
    """
    return sorted(glob.glob(os.path.join(UPLOAD_ROOT, "*", "*", "source.png")))


def load(path):
    with Image.open(path) as opened:
        return np.array(opened.convert("RGB"))


def route_smart(rgb, boxes):
    out = rgb.copy()
    remove_boxes_inplace(out, boxes, "smart")
    return out


def route_unmix(rgb, plan):
    """全局反演：alpha 直接取拟合等级（可到 1.0），全页应用，不分框、不做足迹门。"""
    f = rgb.astype(np.float32)
    alpha = np.clip(plan.coverage / NOMINAL_COVERAGE, 0.0, 1.0)[..., None]
    colour = plan.colour[None, None, :]
    base = (f - alpha * colour) / np.maximum(1.0 - alpha, 0.05)
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def route_replace(rgb, paper):
    """颜色替换：把偏蓝像素直接换成纸色。"""
    f = rgb.astype(np.int16)
    mask = ((f[..., 2] - f[..., 0]) > 40) & (f[..., 2] > 100)
    out = rgb.copy()
    out[mask] = paper
    return out


def route_inpaint(rgb, plan):
    """内容填充：把覆盖度高的像素整块交给 inpaint 重画。"""
    mask = (plan.coverage > 0.25).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(rgb, mask, 4, cv2.INPAINT_TELEA)


def main():
    rows = []
    pages = corpus()
    if not pages:
        print(
            "语料为空：work/uploads 里已没有真实页面"
            "（cleanup_old_files 按 72 小时整批删会话目录）。\n"
            "这张表只能量真实页面上「被水印压住的字」，合成夹具给不出同一判据，"
            "所以本轮**未验**，不算通过也不算失败。"
        )
        return 0
    for index, path in enumerate(pages, start=1):
        rgb = load(path)
        height, width = rgb.shape[:2]
        report = detect_watermark(rgb)
        boxes = regions_to_boxes(report.get("regions", []), width, height, 0)
        plan = _chroma_plan(rgb)
        if plan is None or not boxes:
            print(f"p{index:02d} 跳过（无彩色水印计划或无框）")
            continue

        paper = plan.paper
        paper_lum = float(paper.mean())
        colour_lum = float(plan.colour.mean())
        lum = rgb.astype(np.float32).mean(axis=2)

        # 被水印压住的字：按引擎同一判据（亮度亏损）取，并限制在水印足迹内
        deficit = paper_lum - plan.coverage * (paper_lum - colour_lum) - lum
        text = _smoothstep(deficit, 30.0, 60.0)
        cov8 = np.clip(plan.coverage / NOMINAL_COVERAGE, 0.0, 1.0) * 255.0
        med = cv2.medianBlur(cov8.astype(np.uint8), 7)
        foot = cv2.dilate((med > 60).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        inside = np.zeros((height, width), dtype=bool)
        for box in boxes:
            inside[box.y0:box.y1, box.x0:box.x1] = True
        glyph = (text > 0.5) & foot & inside & (lum < paper_lum * 0.88)
        if int(glyph.sum()) < 40:
            print(f"p{index:02d} 跳过（框内被压住的字只有 {int(glyph.sum())} px）")
            continue

        results = {
            "0 原图": rgb.copy(),
            "A 引擎": route_smart(rgb, boxes),
            "B 全局反演": route_unmix(rgb, plan),
            "C 颜色替换": route_replace(rgb, paper),
            "D 内容填充": route_inpaint(rgb, plan),
        }
        base = rgb.astype(np.float32)
        base_lum = base.mean(axis=2)
        chroma = np.abs(base - base_lum[..., None]).max(axis=2)
        real_ink = (base_lum < paper_lum * 0.6) & (chroma < 34)
        for name, out in results.items():
            o = out.astype(np.float32)
            olum = o.mean(axis=2)
            ink_thr = paper_lum * 0.62
            survive = float(np.mean(olum[glyph] < ink_thr))
            rb = (o[..., 0] - o[..., 2])[glyph]
            red = int(np.count_nonzero(rb >= 20))
            blue = int(
                np.count_nonzero(
                    ((o[..., 2] - o[..., 0])[glyph] > 100) & (o[..., 2][glyph] > 150)
                )
            )
            diff = int(np.count_nonzero(np.abs(o - base).max(axis=2) > 6))
            erased = int(np.count_nonzero(real_ink & (olum - base_lum > 26)))
            darkened = int(np.count_nonzero(real_ink & (base_lum - olum > 26)))
            page_blue = int(
                np.count_nonzero((o[..., 2] - o[..., 0] > 24) & (o[..., 2] > 90))
            )
            rows.append(
                {
                    "page": index,
                    "route": name,
                    "glyph_px": int(glyph.sum()),
                    "survive": survive,
                    "lum_p50": float(np.median(olum[glyph])),
                    "rb_p50": float(np.median(rb)),
                    "red": red,
                    "blue": blue,
                    "diff": diff,
                    "erased": erased,
                    "darkened": darkened,
                    "page_blue": page_blue,
                }
            )
        print(f"p{index:02d} 框={len(boxes)} 被压住的字={int(glyph.sum())}px 完成")

    print()
    header = (
        f"{'路线':<12}{'字存活%':>9}{'亮度p50':>9}{'R-B p50':>9}"
        f"{'发红px':>9}{'残留蓝px':>10}{'改动px':>9}{'抹正文px':>10}{'压深px':>9}"
    )
    print(header)
    print("-" * 74)
    for route in ["0 原图", "A 引擎", "B 全局反演", "C 颜色替换", "D 内容填充"]:
        sel = [r for r in rows if r["route"] == route]
        if not sel:
            continue
        print(
            f"{route:<12}"
            f"{np.mean([r['survive'] for r in sel]) * 100:>8.1f}%"
            f"{np.median([r['lum_p50'] for r in sel]):>9.1f}"
            f"{np.median([r['rb_p50'] for r in sel]):>9.1f}"
            f"{int(np.median([r['red'] for r in sel])):>9}"
            f"{int(np.median([r['blue'] for r in sel])):>10}"
            f"{int(np.median([r['diff'] for r in sel])):>9}"
            f"{int(np.median([r['erased'] for r in sel])):>10}"
            f"{int(np.median([r['darkened'] for r in sel])):>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
