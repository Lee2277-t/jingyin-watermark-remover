"""视频链路端到端回归（第八/九轮）：贴边台标 + 左边缘字标 + 底部滚动条幅，两种模式各跑一遍。

素材全合成，不读用户作品。夹具与「无水印基准」同源同参数，因此残差可直接逐像素量。

跑法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_video_regression.py
保留夹具便于肉眼复看：加 --keep
"""

from __future__ import annotations

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
W, H, FPS, SECONDS = 960, 540, 25, 4
SRC = QA / "_vr_marked.mp4"
CLEAN = QA / "_vr_clean.mp4"
OUT_QUALITY = QA / "_vr_quality.mp4"
OUT_FAST = QA / "_vr_fast.mp4"
FONT = "C\\\\:/Windows/Fonts/arialbd.ttf"

# 三处水印：右下角贴边台标、左边缘贴边字标、底部横贯左右的滚动条幅（三条都触边）
MARKS = [
    {
        "name": "右下台标",
        "filter": (
            f"drawtext=fontfile={FONT}:text='LIVE TV':fontcolor=white@0.82:"
            f"fontsize=34:borderw=2:bordercolor=black@0.55:x=w-tw-2:y=h-th-2"
        ),
        # 归一化选框（用户框的样子），故意贴到画面边
        "region": {"x": (W - 176) / W, "y": (H - 52) / H, "w": 176 / W, "h": 52 / H},
    },
    {
        "name": "左缘字标",
        "filter": (
            f"drawtext=fontfile={FONT}:text='SAMPLE':fontcolor=0x66aaff@0.75:"
            f"fontsize=30:shadowx=1:shadowy=1:shadowcolor=black@0.4:x=0:y=64"
        ),
        "region": {"x": 0.0, "y": 58 / H, "w": 132 / W, "h": 42 / H},
    },
    {
        "name": "底部滚动条幅",
        "filter": (
            f"drawtext=fontfile={FONT}:text='Breaking News ---- 2026 Preview ---- ':fontcolor=white@0.7:"
            f"fontsize=26:box=1:boxcolor=black@0.35:boxborderw=6:"
            f"y=h-th-10:x=mod(t*160\\,w+tw)*-1+tw"
        ),
        "region": {"x": 0.0, "y": (H - 46) / H, "w": 1.0, "h": 46 / H},
    },
]
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def build_clips() -> None:
    if SRC.exists() and CLEAN.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # 背景用两个叠加源：分形纹理（深色细节多，半透明白字才有明显偏差）+ 缓慢平移，避免纯静止画面
    # 两次 ffmpeg 生成共用同一个渐变种子，避免背景差异被计入水印残差。
    base = f"gradients=size={W}x{H}:rate={FPS}:c0=0x203040:c1=0x4a3a55:c2=0x2e4a3b:c3=0x55402a:x0=0:y0=0:x1={W}:y1={H}:seed=0:duration={SECONDS}"
    overlay = f"mandelbrot=size={W // 2}x{H // 2}:rate={FPS}"
    inputs = [
        "-f", "lavfi", "-i", base,
        "-f", "lavfi", "-i", overlay,
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SECONDS}",
    ]
    vf_clean = (
        f"[0:v][1:v]overlay=x='mod(t*90\\,{W})-{W // 2}':y='mod(t*60\\,{H})-{H // 2}':eof_action=repeat[v]"
    )
    marks = [m["filter"] for m in MARKS]
    vf_marked = f"[0:v][1:v]overlay=x='mod(t*90\\,{W})-{W // 2}':y='mod(t*60\\,{H})-{H // 2}':eof_action=repeat," + ",".join(marks) + "[v]"
    tail = [
        "-t", str(SECONDS), "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
    ]
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", vf_clean, *tail, str(CLEAN)], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", vf_marked, *tail, str(SRC)], check=True)


def read_frames(path: Path, count: int = 8) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def pixel_diff(path_a: Path, path_b: Path, count: int = 8) -> tuple[float, float]:
    """返回 (全部帧平均绝对差, 最大单帧平均绝对差)。"""
    a, b = read_frames(path_a, count), read_frames(path_b, count)
    values = [
        float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()) for x, y in zip(a, b)
    ]
    return sum(values) / len(values), max(values)


def residual_in(path: Path, box: dict, clean: Path, marked: Path) -> tuple[float, float]:
    """选框区域内：成品离无水印基准的平均差，以及带标原片离基准的平均差（前者/后者越小越干净）。"""
    frames = read_frames(path, 6)
    base_clean = read_frames(clean, 6)
    base_marked = read_frames(marked, 6)
    x0, y0 = int(box["x"] * W), int(box["y"] * H)
    x1, y1 = min(W, int((box["x"] + box["w"]) * W)), min(H, int((box["y"] + box["h"]) * H))
    here = [
        float(np.abs(f[y0:y1, x0:x1].astype(np.int16) - c[y0:y1, x0:x1].astype(np.int16)).mean())
        for f, c in zip(frames, base_clean)
    ]
    whole = [
        float(np.abs(m[y0:y1, x0:x1].astype(np.int16) - c[y0:y1, x0:x1].astype(np.int16)).mean())
        for m, c in zip(base_marked, base_clean)
    ]
    return sum(here) / len(here), sum(whole) / len(whole)


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    return "Audio:" in (result.stderr or "")


def run(mode: str, out: Path) -> tuple[dict, float]:
    info = probe_video(SRC)
    regions = [m["region"] for m in MARKS]
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    result = video_ops.process_video_file(
        SRC,
        out,
        regions,
        int(info["width"]),
        int(info["height"]),
        method="smart",
        on_progress=lambda _v: None,
        video_mode=mode,
    )
    return result, time.perf_counter() - started


def main() -> int:
    keep = "--keep" in sys.argv
    build_clips()
    info = probe_video(SRC)
    print(
        f"夹具 {info['width']}x{info['height']} {info.get('duration')}s "
        f"{info.get('frame_count')} 帧，{len(MARKS)} 处触边水印"
    )

    quality, quality_seconds = run("quality", OUT_QUALITY)
    fast, fast_seconds = run("fast", OUT_FAST)

    check("画质优先：触边框走逐帧修复", quality["method"].startswith("frame"), str(quality["method"]))
    check("速度优先：触边框内缩走 delogo", fast["method"] == "delogo", str(fast["method"]))
    check("两种模式各自回显", (quality["video_mode"], fast["video_mode"]) == ("quality", "fast"),
          f"{quality['video_mode']} / {fast['video_mode']}")
    check("速度优先确实更快", fast_seconds < quality_seconds,
          f"{quality_seconds:.1f}s -> {fast_seconds:.1f}s（{quality_seconds / fast_seconds:.1f}x）")

    # 选框之外的画面不该被动
    outside_frames_q, outside_frames_f = read_frames(OUT_QUALITY, 6), read_frames(OUT_FAST, 6)
    src_frames = read_frames(SRC, 6)
    mask = np.ones((H, W), bool)
    for mark in MARKS:
        box = mark["region"]
        y0, x0 = int(box["y"] * H), int(box["x"] * W)
        y1, x1 = min(H, int((box["y"] + box["h"]) * H)), min(W, int((box["x"] + box["w"]) * W))
        mask[y0:y1, x0:x1] = False
    for label, frames in (("画质优先", outside_frames_q), ("速度优先", outside_frames_f)):
        diffs = [
            float(np.abs(f.astype(np.int16) - s.astype(np.int16))[mask].mean())
            for f, s in zip(frames, src_frames)
        ]
        out_path = OUT_QUALITY if label == "画质优先" else OUT_FAST
        out_frames = int(probe_video(out_path).get("frame_count") or 0)
        check(f"{label}：选框以外画面基本没动", max(diffs) < 4.0, f"最大单帧平均差={max(diffs):.2f}")
        check(f"{label}：帧数与素材一致（允许差 2）", abs(out_frames - int(info["frame_count"])) <= 2,
              f"{info['frame_count']} -> {out_frames}")

    print("\n各水印区域残差（越小越干净）：成品差值 / 带标原片差值")
    for mark in MARKS:
        quality_value, _ = residual_in(OUT_QUALITY, mark["region"], CLEAN, SRC)
        fast_value, marked_value = residual_in(OUT_FAST, mark["region"], CLEAN, SRC)
        ratio_q = quality_value / marked_value if marked_value else float("inf")
        ratio_f = fast_value / marked_value if marked_value else float("inf")
        print(
            f"  {mark['name']:<10} 带标原片 {marked_value:6.2f} | 画质优先 {quality_value:6.2f}（{ratio_q:.2f}）"
            f" | 速度优先 {fast_value:6.2f}（{ratio_f:.2f}）"
        )
        check(f"{mark['name']}：画质优先清掉大半", ratio_q < 0.45, f"{ratio_q:.2f}")
        check(f"{mark['name']}：速度优先也确有清理", ratio_f < 0.8, f"{ratio_f:.2f}")

    changed_q = pixel_diff(OUT_QUALITY, SRC)
    changed_f = pixel_diff(OUT_FAST, SRC)
    ceiling = pixel_diff(CLEAN, SRC)
    check("画质优先：音轨保留", has_audio(OUT_QUALITY))
    check("速度优先：音轨保留", has_audio(OUT_FAST))
    print(
        f"\n整帧改动幅度（相对带标原片）：画质优先 {changed_q[0]:.2f} / 速度优先 {changed_f[0]:.2f}；"
        f"若三处水印被逐像素完美还原，这个数应当等于 {ceiling[0]:.2f}"
    )
    print("两档都低于完美还原的理论值，说明框内没有全部恢复到无标状态；干净度只看上面的逐区域残差，不看这个整帧数。")
    print(f"耗时：画质优先 {quality_seconds:.1f}s / 速度优先 {fast_seconds:.1f}s")

    if keep:
        print(f"--keep：夹具与成品留在 {QA}")
    else:
        for path in (SRC, CLEAN, OUT_QUALITY, OUT_FAST):
            if path.exists():
                path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
