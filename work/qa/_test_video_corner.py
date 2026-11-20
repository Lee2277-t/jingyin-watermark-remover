# -*- coding: utf-8 -*-
"""视频链路回归（合成素材，不读用户作品）：角落台标走哪条路、两种模式各差多少、画面/音轨有没有被伤到。

夹具是 mandelbrot 片段 + 右下角 drawbox 半透明白台标，另存一份同源无台标片段当"原画"基准，
残差 = 台标区域内 |成品 − 无台标原画| 的平均值，越小越好。

断言：
① 默认（画质优先）：贴角选框走逐帧修复，残差明显下降；
② 速度优先：同一框内缩后走 delogo，速度确实更快、残差确实更差（这条开关的意义就是让用户自己取舍）；
③ 两种模式都不动台标以外的画面、不丢帧、保留音轨；
④ 不触边的框在两种模式下都走 delogo。
"""

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
CLIP = QA / "_vsrc.mp4"
CLEAN = QA / "_vclean.mp4"
OUT_A = QA / "_vcorner.mp4"
OUT_B = QA / "_vcorner_fast.mp4"
W, H, N = 640, 480, 120
LOGO = {"x": W - 96, "y": H - 40, "w": 96, "h": 40}      # 贴右下角，x1/y1 触边
INNER = {"x": W - 130, "y": H - 76, "w": 96, "h": 40}    # 离边 12px，可交给 delogo
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def build_clips() -> None:
    if CLIP.exists() and CLEAN.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # 用 mandelbrot 当素材：右下角是深色复杂纹理，半透明白台标才有明显偏差（testsrc 该区本来就发白，量不出来）
    inputs = [
        "-f", "lavfi", "-i", f"mandelbrot=size={W}x{H}:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=5",
    ]
    tail = ["-t", f"{N / 24}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    logo = (f"drawbox=x={LOGO['x']}:y={LOGO['y']}:w={LOGO['w']}:h={LOGO['h']}"
            f":color=white@0.75:t=fill")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-vf", logo, *tail, str(CLIP)], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    *tail, str(CLEAN)], check=True)


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    return "Audio:" in (result.stderr or "")


def first_frames(path: Path, count: int = 3) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    while len(out) < count:
        ok, frame = capture.read()
        if not ok:
            break
        out.append(frame)
    capture.release()
    return out


def deviation(frames: list[np.ndarray], reference: list[np.ndarray], logo: dict) -> float:
    """台标区域内离无台标原画平均差多少（越小=水印清得越干净）。"""
    x, y, w, h = logo["x"], logo["y"], logo["w"], logo["h"]
    values = [
        float(np.abs(f[y:y + h, x:x + w].astype(np.int16) - r[y:y + h, x:x + w].astype(np.int16)).mean())
        for f, r in zip(frames, reference)
    ]
    return sum(values) / len(values)


def region_of(logo: dict) -> dict:
    return {"x": logo["x"] / W, "y": logo["y"] / H, "w": logo["w"] / W, "h": logo["h"] / H}


def run(logo: dict, out: Path, mode: str) -> tuple[dict, float]:
    if out.exists():
        out.unlink()
    info = probe_video(CLIP)
    started = time.perf_counter()
    result = video_ops.process_video_file(
        CLIP,
        out,
        [region_of(logo)],
        int(info["width"]),
        int(info["height"]),
        method="smart",
        on_progress=lambda _v: None,
        video_mode=mode,
    )
    return result, time.perf_counter() - started


def main() -> int:
    build_clips()
    info = probe_video(CLIP)
    width, height = int(info["width"]), int(info["height"])
    print(f"夹具 {width}x{height} {info.get('duration')}s {info.get('frame_count')} 帧")

    corner, corner_seconds = run(LOGO, OUT_A, "quality")
    fast, fast_seconds = run(LOGO, OUT_B, "fast")

    check("默认画质优先：贴角台标走逐帧修复", corner["method"].startswith("frame"), str(corner["method"]))
    check("画质优先 note 说明触边原因", "触到画面边缘" in corner["note"], corner["note"])
    check("画质优先回显 mode", corner["video_mode"] == "quality", str(corner["video_mode"]))
    check("不再谎称音频保持不变", "音频保持不变" not in corner["note"], corner["note"])

    check("速度优先：同一贴边框走 delogo", fast["method"] == "delogo", str(fast["method"]))
    check("速度优先 note 说明取舍", "速度优先" in fast["note"] and "可能更少也可能更明显" in fast["note"], fast["note"])
    check("速度优先回显 mode", fast["video_mode"] == "fast", str(fast["video_mode"]))
    check("速度优先确实更快", fast_seconds < corner_seconds, f"{corner_seconds:.2f}s -> {fast_seconds:.2f}s")

    padding = max(1, round(min(width, height) * 0.003))
    inner_rects = video_ops._delogo_rects(
        video_ops.regions_to_boxes([region_of(INNER)], width, height, padding), width, height
    )
    corner_rects = video_ops._delogo_rects(
        video_ops.regions_to_boxes([region_of(LOGO)], width, height, padding), width, height
    )
    corner_inset = video_ops._delogo_rects(
        video_ops.regions_to_boxes([region_of(LOGO)], width, height, padding),
        width, height, allow_inset=True,
    )
    check("不触边的框判定为可 delogo", inner_rects is not None, str(inner_rects))
    check("触边的框拒绝 delogo（画质优先）", corner_rects is None, str(corner_rects))
    check("触边的框在速度优先下内缩放行", corner_inset is not None, str(corner_inset))

    clean, src = first_frames(CLEAN), first_frames(CLIP)
    before = deviation(src, clean, LOGO)
    out_corner = first_frames(OUT_A)
    out_fast = first_frames(OUT_B)
    after = deviation(out_corner, clean, LOGO)
    after_fast = deviation(out_fast, clean, LOGO)
    check("逐帧路残差明显下降", after < before * 0.5, f"{before:.1f} -> {after:.1f}")
    check("速度优先残差确实更差（诚实的代价）", after_fast > after, f"{after:.1f} -> {after_fast:.1f}")
    check("速度优先仍清掉了大部分台标", after_fast < before, f"{before:.1f} -> {after_fast:.1f}")

    x, y, w, h = LOGO["x"], LOGO["y"], LOGO["w"], LOGO["h"]
    mask = np.ones((H, W), bool)
    mask[y - 8:min(y + h + 8, H), x - 8:min(x + w + 8, W)] = False
    for label, frames in (("画质优先", out_corner), ("速度优先", out_fast)):
        diff = np.abs(frames[2].astype(np.int16) - src[2].astype(np.int16))
        outside = float(diff[mask].mean())
        check(f"{label}：台标以外画面基本没动", outside < 3.0, f"平均差={outside:.2f}")
        path = OUT_A if label == "画质优先" else OUT_B
        src_frames = int(info.get("frame_count") or 0)
        out_frames = int(probe_video(path).get("frame_count") or 0)
        check(f"{label}：成品帧数与素材一致（允许差 1）", abs(out_frames - src_frames) <= 1, f"{src_frames} -> {out_frames}")
        check(f"{label}：成品保留音轨", has_audio(path))
        check(f"{label}：成品体积正常", path.stat().st_size > 10_000, f"{path.stat().st_size:,} B")

    speedup = corner_seconds / fast_seconds if fast_seconds else 0
    print(
        f"\n实测取舍：画质优先 {corner_seconds:.2f}s 残差 {after:.1f} | "
        f"速度优先 {fast_seconds:.2f}s 残差 {after_fast:.1f}（快 {speedup:.1f}x，残差多 {after_fast - after:.1f}） | "
        f"带台标原图基准 {before:.1f}"
    )
    print("默认走画质优先；只有用户在界面上打开「视频速度优先」才内缩走 delogo。")
    for path in (CLIP, CLEAN, OUT_A, OUT_B):
        if path.exists():
            path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
