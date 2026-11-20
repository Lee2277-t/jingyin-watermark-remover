# -*- coding: utf-8 -*-
"""照片重绘（LaMa ONNX）回归：合成照片 + 白 logo，验「框内补干净、框外逐像素不动」。

素材完全自合成（渐变天空 + 地面纹理），不读用户作品。断言分四层：
① 可用性：模型在时 available()；模型缺失时 available() 为假、且错误能落到用户可读的文案；
② 像素：框内残差相对带 logo 原图明显下降，框外改动（>2 级）为 0；
③ 大框：1600×1000 上 740×240 的大 logo 必须走分块推理，四象限都补干净，且优于整块缩到 512；
④ 契约：服务端 /api/photo/warmup、/api/health 的 photo_ready、以及 method=photo 走完整 HTTP 流程。

用法（服务需是第十三轮代码，脚本会先探活）：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_photo_mode.py
"""

from __future__ import annotations

import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from remover import image_ops, photo_inpaint  # noqa: E402
from remover.image_ops import ImageProcessError, regions_to_boxes  # noqa: E402

QA = BASE / "work" / "qa"
CLEAN = QA / "_photo_clean.png"
MARKED = QA / "_photo_marked.png"
OUTPUT = QA / "_photo_out.png"
SMALL = QA / "_photo_small.png"
W, H = 900, 600
LOGO = (600, 490, 230, 80)  # x, y, w, h；离右边 70px、离下边 30px
LW, LH = 1600, 1000
LARGE_LOGO = (440, 500, 740, 240)  # 大 logo：ROI 长边 1024 > TILE_TRIGGER，必须走分块
LARGE_CLEAN = QA / "_photo_large_clean.png"
LARGE_MARKED = QA / "_photo_large_marked.png"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def _canvas(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        ratio = y / height
        canvas[y, :] = (
            int(120 + 80 * ratio),
            int(150 + 60 * ratio),
            int(190 - 60 * ratio),
        )
    noise = rng.integers(-6, 7, (height, width, 1), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ground = round(height * 0.633)
    cv2.rectangle(canvas, (0, ground), (width, height), (110, 128, 106), -1)
    cv2.line(canvas, (0, ground), (width, ground), (150, 165, 148), 2)
    for x in range(0, width, 60):  # 地面纹理：让重绘有结构可借，避免纯色插值
        cv2.line(canvas, (x, ground), (x + 40, height), (96, 112, 96), 3)
    return canvas


def build_images() -> None:
    canvas = _canvas(W, H, 11)
    Image.fromarray(canvas).save(CLEAN)

    marked = canvas.copy()
    x, y, w, h = LOGO
    overlay = marked[y:y + h, x:x + w].copy()
    cv2.putText(overlay, "SAMPLE", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    marked[y:y + h, x:x + w] = cv2.addWeighted(overlay, 0.8, marked[y:y + h, x:x + w], 0.2, 0)
    Image.fromarray(marked).save(MARKED)
    Image.fromarray(canvas[:100, :100]).save(SMALL)


def build_large_images() -> None:
    canvas = _canvas(LW, LH, 23)
    Image.fromarray(canvas).save(LARGE_CLEAN)
    marked = canvas.copy()
    x, y, w, h = LARGE_LOGO
    overlay = marked[y:y + h, x:x + w].copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 255, 255), 6)
    cv2.putText(overlay, "WATERMARK", (26, 104), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(overlay, "PHOTO LOGO", (26, 196), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)
    marked[y:y + h, x:x + w] = cv2.addWeighted(overlay, 0.72, marked[y:y + h, x:x + w], 0.28, 0)
    Image.fromarray(marked).save(LARGE_MARKED)


def region_of_logo() -> dict:
    x, y, w, h = LOGO
    return {"x": x / W, "y": y / H, "w": w / W, "h": h / H}


def region_of_large_logo() -> dict:
    x, y, w, h = LARGE_LOGO
    return {"x": x / LW, "y": y / LH, "w": w / LW, "h": h / LH}


def unavailable_contract() -> None:
    original = photo_inpaint.MODEL_PATH
    try:
        photo_inpaint.MODEL_PATH = QA / "_photo_missing.onnx"
        photo_inpaint._session = None
        check("模型缺失时 available() 为假", not photo_inpaint.available())
        check("模型缺失时有可读原因", "lama_fp32.onnx" in photo_inpaint.unavailable_reason(),
              photo_inpaint.unavailable_reason())
        try:
            photo_inpaint.inpaint_boxes(np.zeros((64, 64, 3), np.uint8), [photo_inpaint.Box(8, 8, 40, 40)])
            raised = False
        except photo_inpaint.PhotoUnavailable:
            raised = True
        check("模型缺失时 inpaint_boxes 抛 PhotoUnavailable", raised)
        try:
            image_ops.process_image_file(SMALL, QA / "_photo_small_out.png", [region_of_logo()], "photo")
            converted = False
        except ImageProcessError as exc:
            converted = "onnxruntime" in str(exc) or "lama_fp32.onnx" in str(exc)
        check("模型缺失时引擎转成可读的 ImageProcessError", converted)
    finally:
        photo_inpaint.MODEL_PATH = original
        photo_inpaint._session = None
    for leftover in (QA / "_photo_missing.onnx", QA / "_photo_small_out.png"):
        if leftover.exists():
            leftover.unlink()
    check("恢复后 available() 为真", photo_inpaint.available(), photo_inpaint.unavailable_reason())


def offline_engine() -> float:
    started = time.perf_counter()
    detail = image_ops.process_image_file(MARKED, OUTPUT, [region_of_logo()], "photo")
    seconds = time.perf_counter() - started
    check("photo 模式返回重绘说明", "照片重绘" in detail["note"], detail["note"])
    check("photo 模式没有退回纹理修补", detail["fallback_regions"] == 0, str(detail["fallback_regions"]))
    check("成品已落盘", OUTPUT.exists() and OUTPUT.stat().st_size > 10_000,
          f"{OUTPUT.stat().st_size:,} B")

    clean = np.asarray(Image.open(CLEAN).convert("RGB")).astype(np.int16)
    marked = np.asarray(Image.open(MARKED).convert("RGB")).astype(np.int16)
    produced = np.asarray(Image.open(OUTPUT).convert("RGB")).astype(np.int16)

    padding = max(4, round(min(W, H) * 0.006))
    box = regions_to_boxes([region_of_logo()], W, H, padding)[0]
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    inside_before = float(np.abs(marked[y0:y1, x0:x1] - clean[y0:y1, x0:x1]).mean())
    inside_after = float(np.abs(produced[y0:y1, x0:x1] - clean[y0:y1, x0:x1]).mean())
    check("框内残差明显下降", inside_after < inside_before * 0.5,
          f"{inside_before:.1f} -> {inside_after:.1f}")

    diff = np.abs(produced - marked)
    outside = diff.copy()
    outside[y0:y1, x0:x1] = 0
    changed = int((outside > 2).sum())
    check("框外逐像素不动（>2 级 = 0）", changed == 0,
          f"改动 {changed} px，最大 {int(outside.max())}")
    check("框外最大偏差 ≤ 3", int(outside.max()) <= 3, str(int(outside.max())))
    check("框内确实被重绘", int((diff[y0:y1, x0:x1] > 2).sum()) > 1000,
          f"{int((diff[y0:y1, x0:x1] > 2).sum()):,} px")
    return seconds, inside_before, inside_after


def tiled_engine() -> tuple[float, float, float, float, float]:
    """大 logo：必须走分块推理；白色笔画清干净、细节不被糊掉、框外逐像素不动。

    分块的意义是「不把大 ROI 缩到 512」（那会把整块补丁放大成一片糊）。所以这里同时
    量化清晰度：框内高频能量要明显高于整块缩放、且不低于原图的 50%。
    """
    padding = max(4, round(min(LW, LH) * 0.006))
    box = regions_to_boxes([region_of_large_logo()], LW, LH, padding)[0]
    bx0, by0, bx1, by1 = box.x0, box.y0, box.x1, box.y1
    original = np.asarray(Image.open(LARGE_MARKED).convert("RGB"))
    marked_rgb = original.copy()

    started = time.perf_counter()
    detail = photo_inpaint.inpaint_boxes(marked_rgb, [box])
    seconds = time.perf_counter() - started
    check("大 logo 走分块推理", detail.get("tiled_regions") == 1, str(detail.get("tiled_regions")))
    check("大 logo 重绘覆盖选区", detail.get("photo_regions") == 1, str(detail.get("photo_regions")))

    # 旧行为对照：ROI 整块缩到 512 再放大
    x0, y0, x1, y1 = photo_inpaint._roi_for(box, LW, LH)
    roi = original[y0:y1, x0:x1].copy()
    mask = np.zeros(roi.shape[:2], np.uint8)
    mask[max(0, by0 - y0):by1 - y0, max(0, bx0 - x0):bx1 - x0] = 255
    single = photo_inpaint._run_model(roi, mask)
    alpha = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0,
        (photo_inpaint.FEATHER * 2 + 1, photo_inpaint.FEATHER * 2 + 1),
        0,
    )[..., None]
    single_rgb = original.copy()
    single_rgb[y0:y1, x0:x1] = np.clip(
        single.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha), 0, 255
    ).astype(np.uint8)

    clean = np.asarray(Image.open(LARGE_CLEAN).convert("RGB")).astype(np.int16)
    marked = original.astype(np.int16)
    tiled = marked_rgb.astype(np.int16)
    single_out = single_rgb.astype(np.int16)
    region = np.s_[by0:by1, bx0:bx1]

    before = float(np.abs(marked[region] - clean[region]).mean())
    after = float(np.abs(tiled[region] - clean[region]).mean())
    check("大框框内残差明显下降", after < before * 0.65, f"{before:.1f} -> {after:.1f}")

    def bright(image: np.ndarray) -> int:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)[region]
        return int((gray > 200).sum())

    bright_before = bright(marked)
    bright_after = bright(tiled)
    check("大 logo 白色笔画已清除", bright_before > 5_000 and bright_after * 50 < bright_before,
          f"亮像素 {bright_before:,} -> {bright_after:,}")

    def detail_energy(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)[region]
        return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())

    hp_clean = detail_energy(clean)
    hp_tiled = detail_energy(tiled)
    hp_single = detail_energy(single_out)
    check("分块没把大框糊成一片", hp_tiled >= hp_single * 1.8,
          f"高频 分块 {hp_tiled:.2f} vs 整块缩放 {hp_single:.2f}")
    check("框内细节接近原图量级", hp_tiled >= hp_clean * 0.5, f"{hp_tiled:.2f} vs 原图 {hp_clean:.2f}")

    gray = cv2.cvtColor(marked_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)[region]
    column_jump = float(np.abs(np.diff(gray, axis=1)).mean(axis=0).max())
    check("分块没有竖向拼接缝", column_jump <= 12, f"最大列间跳变 {column_jump:.1f}")

    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    quadrants = {
        "左上": (bx0, by0, cx, cy),
        "右上": (cx, by0, bx1, cy),
        "左下": (bx0, cy, cx, by1),
        "右下": (cx, cy, bx1, by1),
    }
    for label, (qx0, qy0, qx1, qy1) in quadrants.items():
        q_before = float(np.abs(marked[qy0:qy1, qx0:qx1] - clean[qy0:qy1, qx0:qx1]).mean())
        q_after = float(np.abs(tiled[qy0:qy1, qx0:qx1] - clean[qy0:qy1, qx0:qx1]).mean())
        check(f"大框{label}没有残留", q_after <= 10.0, f"{q_before:.1f} -> {q_after:.1f}")

    diff = np.abs(tiled - marked)
    outside = diff.copy()
    outside[region] = 0
    changed = int((outside > 2).sum())
    check("大框重绘框外逐像素不动", changed == 0, f"改动 {changed} px，最大 {int(outside.max())}")
    check("大框框内确实被重绘", int((diff[region] > 2).sum()) > 20_000,
          f"{int((diff[region] > 2).sum()):,} px")
    return seconds, before, after, hp_single, hp_tiled


def roi_limits() -> None:
    big = photo_inpaint.Box(0, 0, 4000, 3000)
    x0, y0, x1, y1 = photo_inpaint._roi_for(big, 4000, 3000)
    check("超大选框的 ROI 长边被限制在 1024 内", max(x1 - x0, y1 - y0) <= 1024,
          f"{x1 - x0}x{y1 - y0}")
    corner = photo_inpaint.Box(0, 0, 60, 30)
    x0, y0, x1, y1 = photo_inpaint._roi_for(corner, 900, 600)
    check("贴角选框的 ROI 不越出画面", (x0, y0) == (0, 0) and x1 > 60 and y1 > 30,
          f"({x0},{y0})-({x1},{y1})")


def http_contract() -> None:
    runtime = BASE / "work" / "runtime.json"
    if not runtime.exists():
        check("服务在跑（runtime.json 存在）", False)
        return
    port = int(json.loads(runtime.read_text(encoding="utf-8"))["port"])
    root = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        with opener.open(f"{root}/", timeout=5) as resp:
            page = resp.read().decode("utf-8", "replace")
        with opener.open(f"{root}/api/health", timeout=5) as resp:
            health = json.loads(resp.read())
    except Exception as exc:
        check(f"服务可达（{root}）", False, str(exc))
        return
    check("首页带 photo-ready 标记", 'name="photo-ready" content="1"' in page)
    check("首页有照片重绘卡片", 'data-method="photo"' in page)
    check("health 报告 photo_ready", health.get("photo_ready") is True, str(health.get("photo_ready")))

    def token() -> str:
        import re

        return re.search(r'name="app-token" content="([^"]+)"', page).group(1)

    headers = {"Content-Type": "application/json", "X-App-Token": token()}

    def post(path: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{root}{path}", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with opener.open(request, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    status, payload = post("/api/photo/warmup", {})
    check("POST /api/photo/warmup 返回 200", status == 200 and payload.get("ok") is True, str(payload))

    boundary = "----qaPhotoBoundary"
    data = MARKED.read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{MARKED.name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{root}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": token()},
        method="POST",
    )
    with opener.open(request, timeout=120) as resp:
        uploaded = json.loads(resp.read())
    items = uploaded.get("items") or []
    check("上传合成照片成功", len(items) == 1, str(len(items)))
    if not items:
        return
    item_id = items[0]["id"]
    status, job = post(
        "/api/process",
        {"method": "photo", "items": [{"id": item_id, "regions": [region_of_logo()]}]},
    )
    check("method=photo 被接受（202/200）", status == 200 and job.get("id"), f"{status} {job}")
    job_id = job.get("id")
    deadline = time.time() + 180
    final = job
    while job_id and time.time() < deadline:
        with opener.open(f"{root}/api/jobs/{job_id}", timeout=10) as resp:
            final = json.loads(resp.read())
        if final.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)
    check("照片任务跑完", final.get("status") == "done",
          f"{final.get('status')} {final.get('failed_count')}")
    check("任务回显 motion=false", final.get("motion") is False, str(final.get("motion")))
    results = final.get("results") or []
    check("成品说明来自照片重绘", bool(results) and "照片重绘" in results[0].get("note", ""),
          results[0].get("note", "") if results else "无成品")

    request = urllib.request.Request(
        f"{root}/api/items/{item_id}",
        headers={"X-App-Token": token()},
        method="DELETE",
    )
    try:
        with opener.open(request, timeout=30) as resp:
            resp.read()
    except Exception:
        pass


def main() -> int:
    build_images()
    build_large_images()
    unavailable_contract()
    roi_limits()
    seconds, before, after = offline_engine()
    print(f"\n实测：照片重绘 {W}x{H} 单框耗时 {seconds:.2f}s（含模型加载）| 框内残差 {before:.1f} -> {after:.1f}")
    tiled_seconds, tiled_before, tiled_after, hp_single, hp_tiled = tiled_engine()
    print(
        f"实测：大 logo {LARGE_LOGO[2]}x{LARGE_LOGO[3]}（{LW}x{LH}）分块耗时 {tiled_seconds:.2f}s | "
        f"框内残差 {tiled_before:.1f} -> {tiled_after:.1f} | "
        f"高频 分块 {hp_tiled:.2f} vs 整块缩放 {hp_single:.2f}"
    )
    http_contract()
    for path in (CLEAN, MARKED, OUTPUT, SMALL, LARGE_CLEAN, LARGE_MARKED):
        if path.exists():
            path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
