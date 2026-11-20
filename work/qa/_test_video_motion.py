# -*- coding: utf-8 -*-
"""视频动态水印回归（合成素材，不读用户作品）：台标逐帧移动时，跟踪开关有没有真的跟住。

夹具：mandelbrot 背景 + 左上角贴边起点的半透明白台标，每帧右移 3.2px、上下轻微摆动（100 帧）。
同源无台标片段当原画基准，残差 = 台标**当前所在位置**内 |成品 − 原画| 的均值。

断言：
① `_track_box` 能找回已知位移，且对无关画面返回跟丢（保持原位）；
② 固定框（motion=False）只在第一帧位置修复，台标移走后残差依旧很高；
③ 跟踪（motion=True）在整段路径上残差明显更低；
④ 两条路都不动台标路径以外的画面、不丢帧、保留音轨；
⑤ motion 与 fast 同开时仍走逐帧跟踪（delogo 没有移动矩形可用）；
⑥ 视频选「照片重绘」不会启用 LaMa（视频永远走内容填充）。
"""

import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from remover import video_ops  # noqa: E402
from remover.media import probe_video  # noqa: E402

QA = Path(__file__).resolve().parent
BASE = QA.parents[1]
CLEAN = QA / "_mclean.mp4"
MARKED = QA / "_mmarked.mp4"
OUT_FIXED = QA / "_mfixed.mp4"
OUT_TRACK = QA / "_mtrack.mp4"
OUT_FAST = QA / "_mtrack_fast.mp4"
OUT_PHOTO = QA / "_mphoto.mp4"
W, H, FPS, N = 640, 360, 24, 100
LOGO_W, LOGO_H = 120, 48
SAMPLE = list(range(0, N, 10))
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def logo_xy(index: int) -> tuple[int, int]:
    return int(round(3.2 * index)), 30 + int(round(14 * math.sin(index / 9.0)))


def rect_of(index: int) -> tuple[int, int, int, int]:
    x, y = logo_xy(index)
    return x, y, LOGO_W, LOGO_H


def build_clips() -> None:
    if CLEAN.exists() and MARKED.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"mandelbrot=size={W}x{H}:rate={FPS}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={N / FPS + 1}",
         "-t", f"{N / FPS}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(CLEAN)],
        check=True,
    )
    encoder = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{FPS}", "-i", "pipe:0",
         "-i", str(CLEAN), "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(MARKED)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    capture = cv2.VideoCapture(str(CLEAN))
    index = 0
    try:
        while index < N:
            ok, frame = capture.read()
            if not ok:
                break
            x, y = logo_xy(index)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + LOGO_W - 1, y + LOGO_H - 1), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            encoder.stdin.write(frame.tobytes())
            index += 1
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", "replace")
        encoder.wait()
    finally:
        capture.release()
        if encoder.poll() is None:
            encoder.kill()
    if encoder.returncode != 0:
        raise RuntimeError(f"生成移动台标夹具失败：{error[-800:]}")


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    return "Audio:" in (result.stderr or "")


def read_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = frame
                wanted.discard(index)
            index += 1
    finally:
        capture.release()
    return frames


def path_deviation(output: dict[int, np.ndarray], reference: dict[int, np.ndarray]) -> float:
    """台标每一帧实际所在位置里，成品离无台标原画有多远（越小=清得越干净）。"""
    values = []
    for index in sorted(set(output) & set(reference)):
        x, y, w, h = rect_of(index)
        a = output[index][y:y + h, x:x + w].astype(np.int16)
        b = reference[index][y:y + h, x:x + w].astype(np.int16)
        values.append(float(np.abs(a - b).mean()))
    return sum(values) / len(values) if values else 0.0


def unit_tracking() -> None:
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 255, (240, 240), dtype=np.uint8)
    shifted = np.roll(previous, 7, axis=1)
    hit = video_ops._track_box(previous, shifted, video_ops.Box(40, 40, 100, 100))
    ok = hit is not None and abs(hit[0].x0 - 47) <= 1 and abs(hit[0].y0 - 40) <= 1
    check("_track_box 找回已知位移 7px", bool(ok),
          f"{None if hit is None else (hit[0].x0, hit[0].y0, round(hit[1], 3))}")

    other = rng.integers(0, 255, (240, 240), dtype=np.uint8)
    lost = video_ops._track_box(previous, other, video_ops.Box(40, 40, 100, 100))
    check("_track_box 对无关画面判跟丢", lost is None,
          "None" if lost is None else str(round(lost[1], 3)))


def region_of(rect: tuple[int, int, int, int]) -> dict:
    x, y, w, h = rect
    return {"x": x / W, "y": y / H, "w": w / W, "h": h / H}


def run(out: Path, regions: list[dict], method: str, video_mode: str, motion: bool) -> tuple[dict, float]:
    if out.exists():
        out.unlink()
    info = probe_video(MARKED)
    started = time.perf_counter()
    result = video_ops.process_video_file(
        MARKED, out, regions, int(info["width"]), int(info["height"]),
        method=method, on_progress=lambda _v: None,
        video_mode=video_mode, motion=motion,
    )
    return result, time.perf_counter() - started


def http_contract() -> None:
    """服务端把 motion 一路传到引擎：走真实 HTTP，成品说明必须写「动态水印」。"""
    import http.cookiejar
    import json
    import re
    import urllib.error
    import urllib.request

    runtime = QA.parent / "runtime.json"
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
    check("服务是新代码（health 带 photo_ready）", "photo_ready" in health, str(sorted(health)))
    token = re.search(r'name="app-token" content="([^"]+)"', page).group(1)
    headers = {"Content-Type": "application/json", "X-App-Token": token}

    boundary = "----qaMotionBoundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{MARKED.name}"\r\n'.encode(),
        b"Content-Type: video/mp4\r\n\r\n",
        MARKED.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{root}/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": token},
        method="POST",
    )
    with opener.open(request, timeout=180) as resp:
        items = json.loads(resp.read()).get("items") or []
    check("上传移动台标视频成功", len(items) == 1, str(len(items)))
    if not items:
        return
    item_id = items[0]["id"]
    payload = {"method": "inpaint", "motion": True, "items": [{"id": item_id, "regions": [region_of(rect_of(0))]}]}
    request = urllib.request.Request(
        f"{root}/api/process", data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    with opener.open(request, timeout=120) as resp:
        job = json.loads(resp.read())
    check("motion=true 被服务接受", bool(job.get("id")) and job.get("motion") is True, str(job.get("motion")))
    job_id = job.get("id")
    deadline = time.time() + 240
    final = job
    while job_id and time.time() < deadline:
        with opener.open(f"{root}/api/jobs/{job_id}", timeout=10) as resp:
            final = json.loads(resp.read())
        if final.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)
    check("动态水印任务跑完", final.get("status") == "done", str(final.get("status")))
    results = final.get("results") or []
    note = results[0].get("note", "") if results else ""
    check("成品说明来自动态跟踪", "动态水印" in note, note)

    request = urllib.request.Request(
        f"{root}/api/items/{item_id}", headers={"X-App-Token": token}, method="DELETE"
    )
    try:
        with opener.open(request, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError:
        pass

    # 这一趟真写了一个 成品/ 批次；只认「父目录是本项目 成品/」，别的批次（用户自己的）一律不碰
    output_dir = final.get("output_dir")
    if output_dir:
        path = Path(output_dir)
        if path.parent == BASE / "成品":
            shutil.rmtree(path, ignore_errors=True)
            (BASE / "work" / "zips" / f"{job_id}.zip").unlink(missing_ok=True)
            print(f"已移除本轮动态水印回归的测试成品 {path.name}")


def main() -> int:
    build_clips()
    info = probe_video(MARKED)
    frames = int(info.get("frame_count") or 0)
    print(f"夹具 {info.get('width')}x{info.get('height')} {info.get('duration')}s {frames} 帧，台标每帧右移 3.2px")

    unit_tracking()
    first = region_of(rect_of(0))
    fixed, fixed_seconds = run(OUT_FIXED, [first], "inpaint", "quality", False)
    track, track_seconds = run(OUT_TRACK, [first], "inpaint", "quality", True)
    fast, _ = run(OUT_FAST, [first], "inpaint", "fast", True)
    photo = run(OUT_PHOTO, [region_of((200, 150, LOGO_W, LOGO_H))], "photo", "quality", False)[0]

    check("固定框走逐帧修复", fixed["method"] == "frame-inpaint", str(fixed["method"]))
    check("固定框回显 motion=False", fixed["motion"] is False, str(fixed["motion"]))
    check("跟踪走逐帧跟踪", track["method"] == "frame-track", str(track["method"]))
    check("跟踪回显 motion=True", track["motion"] is True, str(track["motion"]))
    check("跟踪 note 说明是动态水印", "动态水印" in track["note"], track["note"])
    check("跟踪 note 不再谎称音频保持不变", "音频保持不变" not in track["note"], track["note"])
    check("跟踪没跟丢（整段都有命中）", "没跟住" not in track["note"], track["note"])

    check("motion + fast 仍走逐帧跟踪", fast["method"] == "frame-track", str(fast["method"]))
    check("motion + fast 仍说明是动态水印", "动态水印" in fast["note"], fast["note"])
    check("视频选照片重绘不会启用 LaMa", photo["method"] == "delogo" and "重绘" not in photo["note"],
          f"{photo['method']} | {photo['note']}")

    wanted = set(SAMPLE)
    clean = read_frames(CLEAN, set(wanted))
    src = read_frames(MARKED, set(wanted))
    out_fixed = read_frames(OUT_FIXED, set(wanted))
    out_track = read_frames(OUT_TRACK, set(wanted))

    before = path_deviation(src, clean)
    after_fixed = path_deviation(out_fixed, clean)
    after_track = path_deviation(out_track, clean)
    check("固定框在整段路径上仍留大量台标", after_fixed > before * 0.5, f"{before:.1f} -> {after_fixed:.1f}")
    check("跟踪在整段路径上明显更干净", after_track < after_fixed * 0.6, f"{after_fixed:.1f} -> {after_track:.1f}")
    check("跟踪清掉了大部分台标", after_track < before * 0.5, f"{before:.1f} -> {after_track:.1f}")

    mask = np.ones((H, W), bool)
    for index in range(N):
        x, y, w, h = rect_of(index)
        mask[max(0, y - 8):min(H, y + h + 8), max(0, x - 8):min(W, x + w + 8)] = False
    sample = SAMPLE[len(SAMPLE) // 2]
    for label, produced in (("固定框", out_fixed), ("跟踪", out_track)):
        diff = np.abs(produced[sample].astype(np.int16) - src[sample].astype(np.int16))
        outside = float(diff[mask].mean())
        check(f"{label}：台标路径以外画面基本没动", outside < 3.0, f"平均差={outside:.2f}")

    for label, path in (("固定框", OUT_FIXED), ("跟踪", OUT_TRACK)):
        out_frames = int(probe_video(path).get("frame_count") or 0)
        check(f"{label}：成品帧数与素材一致（允许差 1）", abs(out_frames - frames) <= 1, f"{frames} -> {out_frames}")
        check(f"{label}：成品保留音轨", has_audio(path))
        check(f"{label}：成品体积正常", path.stat().st_size > 10_000, f"{path.stat().st_size:,} B")

    print(
        f"\n实测取舍：固定框 {fixed_seconds:.2f}s 残差 {after_fixed:.1f} | "
        f"跟踪 {track_seconds:.2f}s 残差 {after_track:.1f} | 带台标原图基准 {before:.1f}"
    )
    print("跟踪每帧多一次模板匹配；跟丢时保持上一帧位置，不猜方向。")
    http_contract()
    for path in (CLEAN, MARKED, OUT_FIXED, OUT_TRACK, OUT_FAST, OUT_PHOTO):
        if path.exists():
            path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
