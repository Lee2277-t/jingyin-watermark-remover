import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from remover.detection import detect_watermark  # noqa: E402
from remover.image_ops import remove_boxes_inplace, regions_to_boxes  # noqa: E402
from PIL import Image  # noqa: E402

QA = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.abspath(os.path.join(QA, "..", "uploads"))


def pages():
    out = []
    if not os.path.isdir(UPLOADS):
        return out
    for batch in sorted(os.listdir(UPLOADS)):
        bdir = os.path.join(UPLOADS, batch)
        if not os.path.isdir(bdir):
            continue
        for item in sorted(os.listdir(bdir)):
            src = os.path.join(bdir, item, "source.png")
            if os.path.isfile(src):
                out.append((item, src))
    return out


def blue_count(rgb):
    f = rgb.astype(np.int16)
    b, r = f[..., 2], f[..., 0]
    return int(np.count_nonzero((b - r > 24) & (b > 90)))


def ink_count(rgb):
    lum = rgb.astype(np.float32).mean(axis=2)
    bg = float(np.percentile(lum, 80))
    return int(np.count_nonzero(lum < bg * 0.55)), bg


found = pages()
if not found:
    print(
        "语料为空：work/uploads 里已没有真实页面（cleanup_old_files 按 72 小时整批删会话目录，"
        "源码分享包里也不带素材）。\n"
        "这条核对需要用户上传过的真页，合成夹具替代不了，所以本轮**未验**，不算通过也不算失败；"
        "图片引擎的主判据是 work/qa/_test_image_engine.py。"
    )
    raise SystemExit(0)

for name, src in found:
    with Image.open(src) as opened:
        rgba = np.array(opened)
    rgb = rgba[..., :3].copy()
    t0 = time.time()
    report = detect_watermark(rgb)
    boxes = regions_to_boxes(report.get("regions", []), rgb.shape[1], rgb.shape[0], 0)
    before = rgb.copy()
    detail = remove_boxes_inplace(rgb, boxes, "smart")
    t1 = time.time()
    inks_before, bg = ink_count(before)
    inks_after, _ = ink_count(rgb)
    # 正文笔画只允许变深或持平：任何明显变亮的「中性暗像素」就是被抹掉的正文
    lb = before.astype(np.float32).mean(axis=2)
    la = rgb.astype(np.float32).mean(axis=2)
    chroma = np.abs(before.astype(np.float32) - lb[..., None]).max(axis=2)
    real_ink = (lb < bg * 0.6) & (chroma < 34)
    erased = int(np.count_nonzero(real_ink & (la - lb > 26)))
    darkened = int(np.count_nonzero(real_ink & (lb - la > 26)))
    # 被水印压住的正文：原图偏蓝所以逃过上面那条判据，改用「字形内部的邻域」来认
    glyph = cv2.erode((lb < bg * 0.62).astype(np.uint8), np.ones((3, 3), np.uint8))
    glyph = cv2.dilate(glyph, np.ones((9, 9), np.uint8)).astype(bool)
    washed = int(np.count_nonzero(glyph & (la - lb > 26)))
    part = None
    for i, box in enumerate(boxes):
        sub_b = before[box.y0:box.y1, box.x0:box.x1, :]
        sub_a = rgb[box.y0:box.y1, box.x0:box.x1, :]
        print(
            f"  box{i} ({box.x0},{box.y0})-({box.x1},{box.y1}) "
            f"blue {blue_count(sub_b)}->{blue_count(sub_a)} ink {ink_count(sub_b)[0]}->{ink_count(sub_a)[0]}"
        )
        if box.height > 200 and (part is None or box.height > part[0].height):
            part = (box, sub_b, sub_a)
    print(
        f"{name[:12]} boxes={len(boxes)} unblend={detail['unblend_regions']} "
        f"masked={detail['masked_regions']} fallback={detail['fallback_regions']} "
        f"{t1 - t0:.2f}s page_blue {blue_count(before)}->{blue_count(rgb)} "
        f"page_ink {inks_before}->{inks_after} erased={erased} darkened={darkened} washed={washed}"
    )
    if part is not None:
        box, sub_b, sub_a = part
        pair = np.concatenate([sub_b, np.full((sub_b.shape[0], 6, 3), 255, np.uint8), sub_a], axis=1)
        scale = max(1, min(3, 900 // max(1, pair.shape[1])))
        Image.fromarray(cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)).save(
            os.path.join(QA, f"_{name[:8]}_cmp.png")
        )
