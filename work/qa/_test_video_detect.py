# -*- coding: utf-8 -*-
"""视频自动识别回归（合成素材，不读用户作品）：单帧看不到的水印，多帧采样能不能找回来。

夹具（640×360 / 4 秒 / 灰底 + 右下角白色半透明方块）：
① 全片都有台标；
② 台标只在 2.4 秒之后出现——这正是只看第 0 秒会漏掉的场景；
③ 无台标（不该凭空出框）。

断言：
① 采样时间覆盖片头/中段/片尾，且用户当前播放位置被优先采样；
② 跨帧聚类：轻微抖动的同一处水印合成一个框，分开的两处仍是两个；
③ 全片台标能识别、位置对；中途才出现的台标也能识别，而直接看第 0 秒识别不到；
④ 无台标片段不出框；
⑤ HTTP `/api/detect` 返回采样帧数，且坐标仍在 0–1。
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import server  # noqa: E402

QA = Path(__file__).resolve().parent
STATIC = QA / "_vd_static.mp4"
LATE = QA / "_vd_late.mp4"
CLEAN = QA / "_vd_clean.mp4"
LONGAUDIO = QA / "_vd_longaudio.mkv"
W, H, FPS, SECONDS = 640, 360, 24, 4
LOGO = {"x": W - 96, "y": H - 40, "w": 96, "h": 40}
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def build_clips() -> None:
    if STATIC.exists() and LATE.exists() and CLEAN.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames = SECONDS * FPS
    late_start = int(2.4 * FPS)
    for path, appear_from in ((STATIC, 0), (LATE, late_start), (CLEAN, frames)):
        encoder = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{FPS}", "-i", "pipe:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            for index in range(frames):
                frame = np.full((H, W, 3), 122, dtype=np.uint8)
                if index >= appear_from:
                    layer = frame.copy()
                    cv2.putText(layer, "AI", (LOGO["x"] + 6, LOGO["y"] + LOGO["h"] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                    frame = cv2.addWeighted(layer, 0.8, frame, 0.2, 0)
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            error = encoder.stderr.read().decode("utf-8", "replace")
            encoder.wait()
        finally:
            if encoder.poll() is None:
                encoder.kill()
        if encoder.returncode != 0:
            raise RuntimeError(f"生成视频夹具失败：{error[-800:]}")


def build_longaudio() -> None:
    """4 秒画面 + 12 秒音轨的 mkv：容器时长比画面长，采样的后半段会落到片尾之外（真实素材踩过）。"""
    if LONGAUDIO.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=12",
         "-i", str(STATIC), "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac",
         str(LONGAUDIO)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"生成长音轨夹具失败：{result.stderr[-500:]}")


def frame_at(path: Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"读不到 {path.name} @ {seconds}s")
    return frame


def near_logo(region: dict) -> bool:
    cx = region["x"] + region["w"] / 2
    cy = region["y"] + region["h"] / 2
    return (
        abs(cx - (LOGO["x"] + LOGO["w"] / 2) / W) < 0.08
        and abs(cy - (LOGO["y"] + LOGO["h"] / 2) / H) < 0.08
    )


def unit_sampling() -> None:
    times = server._video_sample_times(10.0, 0.0)
    check("10 秒片采样 5 帧", len(times) == 5, str(times))
    check("采样覆盖片头与片尾", min(times) < 0.5 and max(times) > 9.0, str(times))
    with_current = server._video_sample_times(10.0, 3.3)
    check("当前播放位置被优先采样", len(with_current) == 6 and abs(with_current[0] - 3.3) < 1e-6,
          str(with_current))
    check("极短片只采一帧", len(server._video_sample_times(0.2, 0.0)) == 1,
          str(server._video_sample_times(0.2, 0.0)))


def unit_clustering() -> None:
    base = {"x": 0.7, "y": 0.8, "w": 0.15, "h": 0.12}
    jitter = [
        [base],
        [{"x": 0.702, "y": 0.798, "w": 0.151, "h": 0.121}],
        [{"x": 0.698, "y": 0.802, "w": 0.149, "h": 0.119}],
    ]
    clusters = server._cluster_frames(jitter)
    check("同一位置跨帧抖动聚成一个框", len(clusters) == 1 and len(clusters[0]["frames"]) == 3,
          str([len(cluster["frames"]) for cluster in clusters]))
    union = server._union_rect(clusters[0]["rects"])
    check("并集框覆盖抖动范围", 0.69 <= union["x"] <= 0.70 and union["w"] >= 0.151, str(union))

    apart = [[{"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}], [{"x": 0.6, "y": 0.6, "w": 0.1, "h": 0.1}]]
    check("分开的两处仍是两个位置", len(server._cluster_frames(apart)) == 2)

    one_off = [[{"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}], [], [], [], []]
    check("只出现一帧的位置不会被采纳",
          not any(len(cluster["frames"]) >= 2 for cluster in server._cluster_frames(one_off)))


def run_detect(path: Path, current: float = 0.0) -> tuple[dict, float]:
    started = time.perf_counter()
    result = server._detect_video(path, current)
    return result, time.perf_counter() - started


def http_contract() -> None:
    import http.cookiejar
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
    except Exception as exc:
        check(f"服务可达（{root}）", False, str(exc))
        return
    token = re.search(r'name="app-token" content="([^"]+)"', page).group(1)
    headers = {"Content-Type": "application/json", "X-App-Token": token}

    boundary = "----qaVideoDetect"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{LATE.name}"\r\n'.encode(),
        b"Content-Type: video/mp4\r\n\r\n",
        LATE.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{root}/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": token},
        method="POST",
    )
    try:
        with opener.open(request, timeout=180) as resp:
            items = json.loads(resp.read()).get("items") or []
    except urllib.error.HTTPError as exc:
        check("上传中途出现台标的视频", False, exc.read().decode("utf-8", "replace")[:200])
        return
    check("上传中途出现台标的视频", len(items) == 1, str(len(items)))
    if not items:
        return
    item_id = items[0]["id"]
    try:
        payload = {"item_id": item_id, "time": 0}
        request = urllib.request.Request(
            f"{root}/api/detect", data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        with opener.open(request, timeout=120) as resp:
            data = json.loads(resp.read())
        check("HTTP 识别：中途出现的台标能找回", bool(data.get("found")), data.get("message"))
        check("HTTP 识别：返回采样帧数", int(data.get("sampled_frames") or 0) >= 5,
              str(data.get("sampled_frames")))
        regions = data.get("regions") or []
        check("HTTP 识别：坐标仍在 0–1",
              all(0 <= r["x"] <= 1 and 0 <= r["y"] <= 1 and r["w"] > 0 and r["h"] > 0 for r in regions),
              str(regions[:2]))
        check("HTTP 识别：位置落在右下角台标处", any(near_logo(r) for r in regions), str(regions[:2]))
    finally:
        request = urllib.request.Request(
            f"{root}/api/items/{item_id}", headers={"X-App-Token": token}, method="DELETE"
        )
        try:
            with opener.open(request, timeout=30) as resp:
                resp.read()
        except urllib.error.HTTPError:
            pass


def synthetic_frame(width: int, height: int, kind: str | None = None, mark_px: int = 32) -> np.ndarray:
    """合成一张「深色流动画面」：坐标确定，不用字体文件，重复跑逐像素一致。"""
    yy, xx = np.mgrid[0:height, 0:width]
    gray = 70.0 + (yy / height) * 90.0 + np.sin(xx / 37.0) * 12.0 + np.sin((xx + yy) / 53.0) * 9.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    frame = np.dstack([gray, (gray * 0.98).astype(np.uint8), (gray * 1.06).clip(0, 255).astype(np.uint8)])
    if kind in ("corner", "center"):
        size = mark_px
        if kind == "corner":
            x0, y0 = width - 4 * size - 40, height - size - 16
        else:
            x0, y0 = (width - 4 * size) // 2, (height - size) // 2
        stroke = max(2, round(size * 0.14))
        step_x = size + max(2, round(size * 0.2))
        for index in range(4):
            gx = x0 + index * step_x
            frame[y0 : y0 + size, gx : gx + stroke] = 255
            frame[y0 : y0 + size, gx + size - stroke : gx + size] = 255
            mid = y0 + size // 2 - stroke // 2
            frame[mid : mid + stroke + 1, gx : gx + size] = 255
    elif kind == "blob":
        # 虚焦高光：峰值只比背景亮 10 级左右，形状与真实光斑一致
        cy, cx = height - 40, width - 60
        distance = ((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 26.0**2)
        lift = (np.exp(-distance) * 10.0)[..., None]
        frame = np.clip(frame.astype(np.float32) + lift, 0, 255).astype(np.uint8)
    return frame


def unit_native_corner() -> None:
    """高分辨率小台标：1280 降采样口径会丢，原生角裁剪补漏能找回；干净画面与光斑不误框。"""
    width, height = 3840, 1600
    for size in (48, 32, 24):
        frame = synthetic_frame(width, height, "corner", size)
        small = np.array(
            Image.fromarray(frame).resize((1280, 533), Image.Resampling.BILINEAR)
        )
        shrunk = server.detection.detect_watermark(small)
        merged = server._detect_frame(frame)["regions"]
        check(f"4K {size}px 台标：合并后正好 1 框（不与降采样那一路重复）",
              len(merged) == 1, str(merged[:2]))
        if size == 24:
            # 24px 落在 1280 口径的门槛之下，这一档只有原生补漏能找回（实测降采样 0 框）
            check("4K 24px 台标：降采样口径找不到，靠原生补漏", not shrunk["regions"] and len(merged) == 1,
                  f"降采样 {len(shrunk['regions'])} 框 / 合并后 {len(merged)} 框")
    clean = server._detect_frame(synthetic_frame(width, height))
    check("4K 干净画面不出框", not clean["regions"], str(clean["regions"][:1]))
    centered = server.detection.corner_regions_native(synthetic_frame(width, height, "center", 48))
    check("画面中央的亮标记不会被当成角标", not centered, str(centered[:1]))
    blob = server.detection.corner_regions_native(synthetic_frame(width, height, "blob"))
    check("角落里的低对比光斑不出框", not blob, str(blob[:1]))


def main() -> int:
    build_clips()
    build_longaudio()
    unit_sampling()
    unit_clustering()
    unit_native_corner()

    static, static_seconds = run_detect(STATIC)
    check("全片台标：能找到", static["found"], static["message"])
    check("全片台标：位置在右下角", any(near_logo(r) for r in static["regions"]), str(static["regions"][:2]))
    check("全片台标：说明写的是采样帧数", "采样" in static["message"], static["message"])

    first_frame = server.detection.detect_watermark(frame_at(LATE, 0.0))
    check("中途才出现的台标：第 0 秒确实识别不到", not first_frame["found"], first_frame["message"])

    late, late_seconds = run_detect(LATE)
    check("中途才出现的台标：多帧采样能找回", late["found"], late["message"])
    check("中途台标：位置在右下角", any(near_logo(r) for r in late["regions"]), str(late["regions"][:2]))

    clean, clean_seconds = run_detect(CLEAN)
    check("无台标片段不出框", not clean["found"], clean["message"])

    # 长音轨：容器 12s、画面只有 4s。采样点必须按画面长度排，否则后半段全部落在片尾之后，
    # 会被 extract_video_frame 抛错静默跳过（实测只剩 2 帧有效采样）。
    container = server.media.probe_video(LONGAUDIO)
    picture = server.media.probe_video_length(LONGAUDIO)
    check("长音轨素材：容器时长确实比画面长",
          container["duration"] > 10 and 3.0 <= picture["duration"] <= 4.2,
          f"容器 {container['duration']:.2f}s/{container['frame_count']}帧 → 画面 {picture['duration']:.2f}s/{picture['frames']}帧")
    long_audio, long_seconds = run_detect(LONGAUDIO)
    expected = len(server._video_sample_times(picture["duration"], 0.0))
    check("长音轨素材：采样点全部落在画面内", long_audio["sampled_frames"] == expected,
          f"应采 {expected} 帧，实际有效 {long_audio['sampled_frames']} 帧")
    check("长音轨素材：台标仍能识别", long_audio["found"], long_audio["message"])

    print(
        f"\n实测：全片台标 {static_seconds:.2f}s | 中途台标 {late_seconds:.2f}s | "
        f"干净片段 {clean_seconds:.2f}s（每段 5 帧采样，检测帧先缩到长边 ≤{server.VIDEO_DETECT_MAX_SIDE}）"
    )
    http_contract()
    for path in (STATIC, LATE, CLEAN, LONGAUDIO):
        if path.exists():
            path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
