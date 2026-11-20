# -*- coding: utf-8 -*-
"""低对比候选检测（remover/detection_lowcontrast.py）的夹具回归。

六类合成夹具断言命中/零误报，再做冒烟：不崩溃、确定性、耗时、逐页候选计数与
detect_watermark 交叉对照。冒烟以六份自绘多尺寸页面为恒在对象，work/uploads 的真语料
只在场上时补跑（它按 72 小时回收会被整批删掉，不能当判据的来源）。
合成图全部自绘，不读取用户素材内容。
"""

from __future__ import annotations

import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from remover.detection import detect_watermark  # noqa: E402
from remover.detection_lowcontrast import low_contrast_regions  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"{tag}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def text_page(h: int = 1600, w: int = 1200, noise: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    page = np.full((h, w, 3), 245, np.uint8)
    for y in range(90, h - 40, 56):
        x = rng.integers(20, 80)
        while x < w - 30:
            seg = int(rng.integers(40, 240))
            cv2.line(page, (x, y), (min(w - 25, x + seg), y), (60, 60, 60), 3)
            x += seg + int(rng.integers(6, 26))
    page = cv2.GaussianBlur(page, (3, 3), 0)
    if noise:
        page = np.clip(
            page.astype(np.float32) + rng.normal(0, noise, page.shape), 0, 255
        ).astype(np.uint8)
    return page


def draw_mark(page: np.ndarray, x: int, y: int) -> None:
    # 低对比灰字块：纸上 245，块 224（偏差 21，落在 [3,26] 带内）
    cv2.rectangle(page, (x, y), (x + 46, y + 30), (224, 228, 234), -1)
    cv2.line(page, (x + 6, y + 7), (x + 40, y + 7), (212, 214, 220), 3)
    cv2.line(page, (x + 6, y + 15), (x + 40, y + 15), (212, 214, 220), 3)
    cv2.line(page, (x + 6, y + 23), (x + 30, y + 23), (212, 214, 220), 3)


def tiled_page() -> np.ndarray:
    page = text_page()
    v = np.array([240, 150])
    u = np.array([-120, 190])
    for i in range(-2, 12):
        for j in range(-2, 8):
            pos = np.array([120, 90]) + i * v + j * u
            x, y = int(pos[1]), int(pos[0])
            if 30 <= x <= 1200 - 90 and 30 <= y <= 1600 - 70:
                draw_mark(page, x, y)
    return page


def stamp_page() -> np.ndarray:
    page = text_page()
    mask = np.zeros((1600, 1200), np.uint8)
    cv2.circle(mask, (600, 800), 130, 1, -1)
    page[mask > 0] = np.clip(
        page[mask > 0].astype(np.int16) - np.array([22, 20, 16]), 0, 255
    ).astype(np.uint8)
    return page


def logo_page() -> np.ndarray:
    page = text_page()
    h, w = 1600, 1200
    cv2.rectangle(page, (w - 150, h - 48), (w - 96, h - 16), (70, 70, 85), -1)
    cv2.rectangle(page, (w - 88, h - 48), (w - 20, h - 16), (70, 70, 85), -1)
    return page


def gradient_page() -> np.ndarray:
    h, w = 1600, 1200
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 180 + 60 * (x / w)
    img[..., 1] = 150 + 40 * np.sin(x / 900) * np.sin(y / 700)
    img[..., 2] = 210 - 40 * (y / h)
    page = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(page, (300, 400), 160, (90, 120, 200), -1)
    cv2.circle(page, (900, 1100), 220, (230, 180, 120), -1)
    return cv2.GaussianBlur(page, (5, 5), 0)


def noise_page() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (1600, 1200, 3), np.uint8)


def bright_mark_page() -> np.ndarray:
    # 既有路线（四角亮白字）的夹具：白字比纸面亮 10 级，必须仍然出正式修复框
    page = text_page()
    h, w = page.shape[:2]
    cv2.putText(
        page, "AI AI", (w - 200, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4, cv2.LINE_AA
    )
    return page


def overlap_share(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return ix * iy / max(1e-9, smaller)


def box_of(region: dict) -> tuple[float, float, float, float]:
    return (
        region["x"],
        region["y"],
        region["x"] + region["w"],
        region["y"] + region["h"],
    )


def detect_contract_checks() -> None:
    print("== 检测契约（detect_watermark）==")
    tiled = detect_watermark(tiled_page())
    k = kinds(tiled["suggestions"])
    check("平铺页 suggestions 命中 tiled", k.get("tiled", 0) >= 2, str(k))
    check("平铺页 suggestions 全带 low_confidence", all(s["low_confidence"] for s in tiled["suggestions"]))
    check("平铺页 suggestions 带中文说明", all(s["label"] == "整页平铺" for s in tiled["suggestions"]))
    check("仅疑似时 found=False 且 regions 为空", tiled["found"] is False and tiled["regions"] == [])
    check("仅疑似时消息提到疑似", "疑似" in tiled["message"], tiled["message"])
    check(
        "平铺 suggestions 归一化在 0..1",
        all(0 <= s["x"] <= 1 and 0 <= s["y"] <= 1 and 0 < s["w"] <= 1 and 0 < s["h"] <= 1 for s in tiled["suggestions"]),
    )

    stamp = detect_watermark(stamp_page())
    check("灰章页 suggestions 命中 stamp", kinds(stamp["suggestions"]).get("stamp", 0) == 1)

    text = detect_watermark(text_page())
    check("纯文字页无 suggestions", text["suggestions"] == [])
    check("纯文字页仍报未找到", text["found"] is False and "未自动找到" in text["message"])

    bright = detect_watermark(bright_mark_page())
    check("白字角标仍走原路线出正式框", bright["found"] is True and len(bright["regions"]) >= 1, bright["message"])
    check(
        "白字角标页疑似框不与修复框重叠",
        not any(
            overlap_share(box_of(s), box_of(r)) > 0.35
            for s in bright["suggestions"]
            for r in bright["regions"]
        ),
        f"regions={len(bright['regions'])} suggestions={len(bright['suggestions'])}",
    )

    logo = detect_watermark(logo_page())
    check(
        "深角标页疑似框不与修复框重叠",
        not any(
            overlap_share(box_of(s), box_of(r)) > 0.35
            for s in logo["suggestions"]
            for r in logo["regions"]
        ),
        f"regions={len(logo['regions'])} suggestions={len(logo['suggestions'])}",
    )
    check("深角标页最多 6 条提示", len(logo["suggestions"]) <= 6)


def kinds(candidates: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cand in candidates:
        out[cand["kind"]] = out.get(cand["kind"], 0) + 1
    return out


def fixture_checks() -> None:
    print("== 合成夹具 ==")
    tiled = low_contrast_regions(tiled_page())
    k = kinds(tiled)
    check("平铺页命中 tiled 候选", k.get("tiled", 0) >= 2, str(k))
    check("平铺页候选都带 low_confidence", all(c["low_confidence"] for c in tiled))

    stamp = low_contrast_regions(stamp_page())
    k = kinds(stamp)
    check("灰章页命中 stamp 候选", k.get("stamp", 0) >= 1, str(k))
    check("灰章候选归一化在 0..1", all(0 <= c["w"] <= 1 and 0 <= c["h"] <= 1 for c in stamp))

    logo = low_contrast_regions(logo_page())
    k = kinds(logo)
    check("深色角标页命中 corner-dark 候选", k.get("corner-dark", 0) >= 1, str(k))
    corner = [c for c in logo if c["kind"] == "corner-dark"]
    check("角标候选落在右下", corner and corner[0]["x"] > 0.6 and corner[0]["y"] > 0.75, str(corner[:1]))

    check("纯文字页零候选", low_contrast_regions(text_page()) == [])
    check("渐变图零候选", low_contrast_regions(gradient_page()) == [])
    check("噪声图零候选", low_contrast_regions(noise_page()) == [])


def determinism_checks() -> None:
    print("== 确定性 ==")
    pages = [tiled_page(), stamp_page(), logo_page(), gradient_page()]
    for i, page in enumerate(pages):
        first = low_contrast_regions(page)
        second = low_contrast_regions(page)
        check(f"夹具 {i} 两次结果一致", first == second, str(len(first)))


def synthetic_sweep() -> list[tuple[str, np.ndarray]]:
    """多尺寸合成页面，专供冒烟阶段。

    旧版只扫 work/uploads/*/*/source.png（用户上传的真页），而 cleanup_old_files()
    按 72 小时整批删会话目录——2026-10-27 起本机实测 0 份，冒烟阶段变成「一份都没跑」，
    「每份耗时 < 4s」也随之 vacuous 通过。「不崩溃 + 耗时上限」这两条判据不该跟着语料一起蒸发，
    所以给它们一份恒在的合成扫描；真页仍在时照样跑，只降为机会性补充。
    """
    return [
        ("合成 平铺 1200x1600", tiled_page()),
        ("合成 灰章 1200x1600", stamp_page()),
        ("合成 角标 1200x1600", logo_page()),
        ("合成 渐变 1200x1600", gradient_page()),
        ("合成 纯文 700x900", text_page(900, 700)),
        ("合成 纯文 1500x2000", text_page(2000, 1500)),
    ]


def run_one(rgb: np.ndarray) -> tuple[int, dict, float]:
    t0 = time.time()
    report = detect_watermark(rgb)
    cands = low_contrast_regions(rgb)
    elapsed = time.time() - t0
    return len(report.get("regions", [])), kinds(cands), elapsed


def tally(cross: dict[str, int], found: int, k: dict[str, int]) -> None:
    key = (
        f"detect={found} cand="
        f"{'|'.join(f'{name}:{n}' for name, n in sorted(k.items())) or '-'}"
    )
    cross[key] = cross.get(key, 0) + 1


def corpus_smoke(limit: int = 0) -> None:
    print("== 冒烟：合成扫描恒在 + 真语料机会性补充 ==")
    cross: dict[str, int] = {}
    total_time = 0.0
    slowest = 0.0
    ran = 0
    for label, rgb in synthetic_sweep():
        found, k, elapsed = run_one(rgb)
        ran += 1
        total_time += elapsed
        slowest = max(slowest, elapsed)
        tally(cross, found, k)
        print(f"  {label} detect={found} {k or {}} t={elapsed:.2f}s")

    sources = sorted(glob.glob(os.path.join(ROOT, "work", "uploads", "*", "*", "source.png")))
    if limit:
        sources = sources[:limit]
    if sources:
        print(f"真语料 {len(sources)} 份，逐份跑 low_contrast_regions + detect_watermark")
        for src in sources:
            with Image.open(src) as opened:
                rgba = np.array(opened)
            found, k, elapsed = run_one(rgba[..., :3].copy())
            ran += 1
            total_time += elapsed
            slowest = max(slowest, elapsed)
            tally(cross, found, k)
            item = os.path.basename(os.path.dirname(src))
            if k:
                print(f"  {item[:12]} detect={found} {k} t={elapsed:.2f}s")
    else:
        print(
            "真语料 0 份（work/uploads 已被 72 小时回收清空）：「真实页面不崩溃 / 真实页耗时」"
            "这一条本轮**未验**，不算通过也不算失败；语料回来后再跑本脚本才成立。"
        )

    print(f"共跑 {ran} 份，总耗时 {total_time:.1f}s，单份最慢 {slowest:.2f}s")
    print("（检测框数 × 候选数）分布：")
    for key, n in sorted(cross.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4} 份  {key}")
    check("每份耗时 < 4s", slowest < 4.0, f"{slowest:.2f}s")
    check("合成扫描跑满 6 份", ran >= 6, f"{ran} 份")
    if sources:
        check("真语料全部跑完无异常", ran == len(sources) + 6, f"{ran} 份")


def main() -> None:
    fixture_checks()
    determinism_checks()
    detect_contract_checks()
    corpus_smoke()
    print()
    if FAILED:
        print(f"FAILED {len(FAILED)} 项: {FAILED}")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
