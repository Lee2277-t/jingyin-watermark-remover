"""交付成品核对：成品与离线引擎逐像素一致，另量 erased / 框外改动 / 空白点仍蓝。

用法：python work/qa/_verify_delivered.py <成品批次名> [<拿来对照的旧批次名>]

语料与批次都从磁盘发现，脚本里不写死会话号、素材号，也不留用户的批次名。
它比对的是「用户上传过的真页 vs 已交付的成品」，合成夹具给不出同一判据；
`cleanup_old_files()` 按 72 小时整批删会话目录，语料没了就如实报「未验」。
"""

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from remover.detection import detect_watermark  # noqa: E402
from remover import image_ops as IO  # noqa: E402
from remover.image_ops import _chroma_plan, regions_to_boxes, remove_boxes_inplace  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOADS = os.path.join(BASE, "work", "uploads")
OUTPUT_ROOT = os.path.join(BASE, "成品")
SUFFIX = "_已去水印"


def pages():
    """磁盘上还在的真页：[(素材号, 页面路径, 它在成品目录里该叫什么)]。

    成品名由素材原文件名派生（server 的 _free_output_name 沿用这个名字），
    所以按 meta.json 的 name 配对，而不是按「第几页」——扫盘顺序不等于上传顺序。
    """
    found = []
    for path in sorted(glob.glob(os.path.join(UPLOADS, "*", "*", "source.png"))):
        item_dir = os.path.dirname(path)
        try:
            meta = json.loads(
                open(os.path.join(item_dir, "meta.json"), encoding="utf-8").read()
            )
        except (OSError, ValueError):
            meta = {}
        name = meta.get("name") or os.path.basename(item_dir)
        stem = os.path.splitext(name)[0]
        found.append((os.path.basename(item_dir), path, f"{stem}{SUFFIX}.png"))
    return found


def batch_dir(name):
    path = os.path.join(OUTPUT_ROOT, name)
    if not os.path.isdir(path):
        have = sorted(
            d for d in os.listdir(OUTPUT_ROOT)
            if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
        ) if os.path.isdir(OUTPUT_ROOT) else []
        print(f"成品批次不存在：{name}\n成品/ 现在有的批次：{have or '（空）'}")
        return None
    return path


def read(path):
    with Image.open(path) as opened:
        return np.array(opened)[..., :3]


def main():
    if len(sys.argv) < 2:
        print("用法：python work/qa/_verify_delivered.py <成品批次名> [<拿来对照的旧批次名>]")
        return 0
    new_dir = batch_dir(sys.argv[1])
    if new_dir is None:
        return 1
    old_dir = None
    if len(sys.argv) > 2:
        old_dir = batch_dir(sys.argv[2])
        if old_dir is None:
            return 1
    items = pages()
    if not items:
        print(
            "语料为空：work/uploads 里已没有真实页面（cleanup_old_files 按 72 小时整批删会话目录）。\n"
            "这条核对需要用户上传过的真页，合成夹具替代不了，所以本轮**未验**，"
            "不算通过也不算失败。"
        )
        return 0

    tot = [0, 0, 0, 0]
    paired = skipped = 0
    for item_id, source_path, product_name in items:
        new_path = os.path.join(new_dir, product_name)
        if not os.path.isfile(new_path):
            skipped += 1
            continue
        paired += 1
        src = read(source_path)
        height, width = src.shape[:2]
        padding = max(4, round(min(width, height) * 0.006))
        boxes = regions_to_boxes(
            detect_watermark(src).get("regions", []), width, height, padding
        )
        inside = np.zeros((height, width), bool)
        for box in boxes:
            inside[box.y0:box.y1, box.x0:box.x1] = True
        blank = IO._blank_paper(src, _chroma_plan(src))
        offline = src.copy()
        remove_boxes_inplace(offline, boxes, "smart")
        new = read(new_path)
        same = int(np.array_equal(new, offline))
        lb = src.astype(np.float32).mean(axis=2)
        la = new.astype(np.float32).mean(axis=2)
        bg = float(np.percentile(lb, 80))
        chroma = np.abs(src.astype(np.float32) - lb[..., None]).max(axis=2)
        real_ink = (lb < bg * 0.55) & (chroma < 34)
        erased = int(np.count_nonzero(real_ink & (la - lb > 26)))
        dark = int(np.count_nonzero(real_ink & (lb - la > 26)))
        diff = np.abs(new.astype(np.int32) - src.astype(np.int32)).max(axis=2)
        out_moved = int(np.count_nonzero((diff > 1) & ~inside))
        nb = new.astype(np.int16)[..., 2] - new.astype(np.int16)[..., 0]
        resid_new = int(np.count_nonzero((nb > 40) & blank))
        resid_old = 0
        if old_dir:
            old_path = os.path.join(old_dir, product_name)
            if os.path.isfile(old_path):
                ob = read(old_path).astype(np.int16)
                resid_old = int(np.count_nonzero((ob[..., 2] - ob[..., 0] > 40) & blank))
        tot[0] += out_moved
        tot[1] += resid_new
        tot[2] += resid_old
        tot[3] += erased
        print(
            f"{item_id[:8]} 与离线引擎逐像素一致={same} erased={erased} dark={dark} "
            f"框外改动={out_moved:4d} 空白蓝点 旧={resid_old:4d} 新={resid_new:3d}"
        )
    print(
        f"\n配对 {paired} 份（这批里找不到对应成品的 {skipped} 份已跳过）"
        f" 合计 框外改动={tot[0]} 空白蓝点 旧={tot[2]} -> 新={tot[1]} erased={tot[3]}"
    )
    if paired == 0:
        print("一页都没对上，等于未验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
