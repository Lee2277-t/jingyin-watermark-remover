from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import options
from .cancel import CancelRequested
from .image_ops import Box, regions_to_boxes, remove_boxes_inplace
from .media import ffmpeg_path, probe_streams, probe_video, probe_video_length


class VideoProcessError(Exception):
    pass


ProgressCallback = Callable[[float], None]
StopCheck = Callable[[], bool]

# 逐帧跟踪：搜索窗 = 选框长边 × TRACK_WINDOW，命中阈值用 TM_CCOEFF_NORMED。
# 台标每帧位移通常只有几个像素，窗口给到 1.2 倍长边足够；跟丢时保持上一帧位置。
TRACK_WINDOW = 1.2
TRACK_MIN_SCORE = 0.35

# 静态水印掩膜（第十七轮）：水印不动、背景在动，所以「选框内每个像素的时间波动」明显低于框外一圈。
# 只修补掩膜覆盖的笔画像素，框内其余像素逐像素原样保留——这正是逐帧修复此前输给 delogo 的地方：
# 那时水印落在文档页的色度/亮色路线上全被驳回，整框退化成「搬一块邻居贴片」，既搬来异物又留接缝。
MASK_SAMPLES = 6          # 全片等分抽几帧量时间波动（每帧一次 -ss 抽帧，实测约 0.19s/帧）
MASK_PAD = 28             # 选框外扩一圈作为「背景波动」参考环
MASK_CTX = 24             # 修补时四周留给 inpaint 的上下文（避免边缘外推）
MASK_STD_LO = 0.55        # 波动比 ≤0.55 算水印本体（实测时间反推的覆盖度 0.53–0.72）
MASK_STD_HI = 0.90        # ≥0.90 算纯背景；中间线性过渡
MASK_RING_MIN_STD = 6.0   # 框外一圈中位波动低于此值时判「背景本身几乎不动」，分不开水印与背景
MASK_MIN_SHARE = 0.05     # 掩膜至少要覆盖选框的 5%（动态水印会把它冲掉，低于此值交回原路线）
MASK_MAX_SHARE = 0.70     # 覆盖超过 70% 说明框里是整块实心台标，实测这种块状水印整框贴片更贴合
MASK_HEAVY_SHARE = 0.50   # 覆盖过半的掩膜往往连「不动的背景」一起盖住，说明里不能再说「只补笔画」
MASK_FILL_BLUR = 1.2      # 填补结果的高斯半径：TELEA 的等照度延伸会留下条纹，抹平一档
MASK_FEATHER = 1.0        # 掩膜羽化半径，只影响过渡像素

# 成品固定是 .mp4：这些音频编码可以直接装进 MP4（无损搬运，不重编码）。
AUDIO_COPY_CODECS = {"aac", "mp3", "alac", "ac3", "eac3"}
# 文本字幕能转成 MP4 的 mov_text；图形字幕（pgs/dvd_subtitle/dvb）无法转成文本，只能略过。
TEXT_SUBTITLE_CODECS = {"mov_text", "subrip", "ass", "ssa", "webvtt", "text"}

# ffmpeg 管道卡死总闸。视频永远只有一条通道，一条管道卡住就是整批假死，而且用户点「停止」
# 也停不下来——取消只在检查点读闸门，管道不动了检查点也就永远不会再来第二次。
# 判的是「多久没有任何动静」而不是「总共跑了多久」：长片子逐帧本来就是一帧一帧往前走，
# 只要还在出帧/出进度行就不算卡住（4K 掩膜单帧实测 ≤0.5s，delogo 的 -progress 约每秒一行）。
PIPE_IDLE_TIMEOUT = 120.0
PIPE_IDLE_POLL = 1.0
# 收尾（flush + faststart 搬 moov）没有进度信号可看，单独给一个更宽的等待上限。
PIPE_FLUSH_TIMEOUT = 300.0


class _PipeGuard:
    """盯住一路 ffmpeg 子进程：连续 PIPE_IDLE_TIMEOUT 没有任何动静就把它们全部杀掉。

    调用方在每次读到进度行或解出一帧时打一下 `beat()`；`beat` 就是「还在往前走」的唯一证据。
    """

    def __init__(self, processes: list, label: str) -> None:
        self.processes = processes
        self.label = label
        self.reason = ""
        self._beat_at = time.monotonic()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def beat(self) -> None:
        self._beat_at = time.monotonic()

    def _loop(self) -> None:
        while not self._done.wait(PIPE_IDLE_POLL):
            idle = time.monotonic() - self._beat_at
            if idle < PIPE_IDLE_TIMEOUT:
                continue
            self.reason = f"连续 {int(idle)} 秒没有任何进展"
            for process in self.processes:
                if process.poll() is None:
                    process.kill()
            return

    def stop(self) -> None:
        self._done.set()

    def raise_if_stuck(self) -> None:
        if self.reason:
            raise VideoProcessError(
                f"{self.label}卡住了（{self.reason}），已自动停止这条管道；"
                "请重试这一份，反复卡住可换一种修复档位再试"
            )


def _plan_streams(streams: dict) -> dict:
    audio = streams.get("audio") or []
    subtitles = streams.get("subtitles") or []
    video = streams.get("video") or {}
    copy_audio = [item for item in audio if item["codec"] in AUDIO_COPY_CODECS]
    text_subtitles = [item for item in subtitles if item["codec"] in TEXT_SUBTITLE_CODECS]
    copy_all = len(audio) > 0 and len(copy_audio) == len(audio)
    return {
        "audio_count": len(audio),
        "copy_audio": copy_all,
        "audio_indices": [item["index"] for item in copy_audio],
        "subtitle_indices": [item["index"] for item in text_subtitles],
        "skipped_subtitles": len(subtitles) - len(text_subtitles),
        # 画面流是不是 HDR：由传输特性判定，认不出就当 SDR（不在说明里乱写）。
        "hdr": bool(video.get("hdr")),
        "hdr_transfer": video.get("transfer") or "",
        # `-shortest` 会让最后那一帧视频被音轨/字幕的边界截掉（实测 100 帧只剩 97 帧）：
        # 只有重编码音频、且没搬字幕时才用它对齐长度；搬运原音轨时以画面帧数为准。
        "trim": bool(audio) and not copy_all and not text_subtitles,
    }


def _stream_args(plan: dict, input_index: int) -> list[str]:
    """把原片的音轨/字幕轨搬进成品：能直接装的 copy，装不进的音轨重编码，图形字幕略过。"""
    args: list[str] = []
    if plan["copy_audio"]:
        for index in plan["audio_indices"]:
            args += ["-map", f"{input_index}:{index}"]
        args += ["-c:a", "copy"]
    elif plan["audio_count"]:
        args += ["-map", f"{input_index}:a?", "-c:a", "aac", "-b:a", "192k"]
    for index in plan["subtitle_indices"]:
        args += ["-map", f"{input_index}:{index}"]
    if plan["subtitle_indices"]:
        args += ["-c:s", "mov_text"]
    return args


def _streams_note(plan: dict) -> str:
    parts: list[str] = []
    if plan.get("hdr"):
        transfer = plan.get("hdr_transfer") or "HDR"
        parts.append(
            f"素材是 HDR（{transfer}），本工具不做色调映射：成品为 8bit，"
            "亮度与颜色可能异常、色彩标记可能丢失"
        )
    if plan["copy_audio"]:
        parts.append("原音轨已保留")
    elif plan["audio_count"]:
        parts.append("原音轨格式放不进 MP4，已重编码为 AAC")
    if plan["subtitle_indices"]:
        parts.append("字幕已保留")
    if plan["skipped_subtitles"]:
        parts.append(f"{plan['skipped_subtitles']} 条图形字幕无法转成文本，已略过")
    return "；".join(parts)


def _with_streams(note: str, plan: dict) -> str:
    suffix = _streams_note(plan)
    return f"{note}；{suffix}" if suffix else note


def _read_stream(stream, target: list[str], limit: int = 20000) -> None:
    try:
        for line in iter(stream.readline, b""):
            target.append(line.decode("utf-8", errors="replace"))
            while sum(len(item) for item in target) > limit:
                target.pop(0)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _stderr_text(chunks: list[str]) -> str:
    text = "".join(chunks).strip()
    if not text:
        return "未知错误"
    return text[-1800:]


def _delogo_rects(
    boxes: list[Box], width: int, height: int, allow_inset: bool = False
) -> list[tuple[int, int, int, int]] | None:
    """能整框交给 delogo 就返回矩形，否则返回 None（交回逐帧修复）。

    实测 ffmpeg 的 delogo 不允许矩形贴到画面边（贴边直接报 `Logo area is outside of the frame`），
    合法范围是 `1 <= x`、`x + w <= 宽-1`；而外扩过 padding 的角落台标必然 `x0=0`，所以贴角框进不了 delogo。
    同一合成台标夹具量出来的残差：逐帧 **13.0**、内缩 1px 硬走 delogo **29.3**（无台标基准 49.6），
    把贴边那 1px 行列单独剔掉 delogo 反而更差（31.6）——插值本身就输，不是边框条的问题。
    所以内缩只在用户显式选了「速度优先」时做（快约 3.3–4.4 倍；贴边残留两档各有胜负，
    第十五轮 960×400 实拍上速度档反而更干净，所以这条取舍交给用户，不替他要「更干净」的承诺）。
    """
    rects: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
        if x0 < 1 or y0 < 1 or x1 > width - 1 or y1 > height - 1:
            if not allow_inset:
                return None
            x0, y0 = max(x0, 1), max(y0, 1)
            x1, y1 = min(x1, width - 1), min(y1, height - 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return None
        rects.append((x0, y0, x1 - x0, y1 - y0))
    return rects or None


def _touches_edge(boxes: list[Box], width: int, height: int) -> bool:
    return any(
        box.x0 < 1 or box.y0 < 1 or box.x1 > width - 1 or box.y1 > height - 1
        for box in boxes
    )


def _process_with_delogo(
    input_path: Path,
    output_path: Path,
    rects: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    duration: float,
    on_progress: ProgressCallback,
    stream_args: list[str],
    should_stop: StopCheck | None = None,
) -> None:
    filters = [
        f"scale={width}:{height}:flags=lanczos",
        "setsar=1",
    ]
    for x, y, w, h in rects:
        filters.append(f"delogo=x={x}:y={y}:w={w}:h={h}:show=0")
    filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        *stream_args,
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(
        target=_read_stream, args=(process.stderr, stderr_chunks), daemon=True
    )
    stderr_thread.start()
    guard = _PipeGuard([process], "画面内插值修复")
    return_code: int | None = None
    try:
        if process.stdout is None:
            raise VideoProcessError("无法读取视频处理进度")
        for raw_line in process.stdout:
            guard.beat()
            line = raw_line.decode("utf-8", errors="replace").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in ("out_time_us", "out_time_ms") and duration > 0:
                try:
                    seconds = int(value) / 1_000_000
                    on_progress(min(0.99, max(0.01, seconds / duration)))
                except ValueError:
                    pass
            if key == "out_time_ms" and should_stop is not None and should_stop():
                # finally 会 kill 掉 ffmpeg，半成品由 process_video_file 的 .part 分支删掉
                raise CancelRequested("已停止：视频插值修复已中断")
        guard.stop()
        try:
            return_code = process.wait(timeout=PIPE_FLUSH_TIMEOUT)
        except subprocess.TimeoutExpired:
            guard.reason = f"收尾（写出文件头）超过 {int(PIPE_FLUSH_TIMEOUT)} 秒"
    finally:
        guard.stop()
        if process.poll() is None:
            process.kill()
        stderr_thread.join(timeout=2)
    guard.raise_if_stuck()
    if return_code is None or return_code != 0:
        raise VideoProcessError(f"视频编码失败：{_stderr_text(stderr_chunks)}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoProcessError("视频处理没有生成有效文件，请尝试重新框选水印区域")
    on_progress(1.0)


def _read_frame(stream, frame: np.ndarray) -> bool:
    """把**一整帧**读进 `frame` 自己的内存；不足一帧（含 EOF）返回 False。

    逐帧路线每帧要搬 1.5–25 MB，先前「`bytearray.extend` 拼 + `bytes()` 转不可变」
    的写法会让同一帧在管道和我们之间多拷两遍（实测 4K 每帧因此占掉整趟的三成），
    所以直接对预分配缓冲做 `readinto`。管道会给半截帧，这里按字节数续读，
    只有真的读到 0 字节才判结束——**半截帧不能当成一帧发出去**。
    """
    view = memoryview(frame).cast("B")
    filled = 0
    total = view.nbytes
    while filled < total:
        chunk = stream.readinto(view[filled:])
        if not chunk:
            return False
        filled += chunk
    return True


def _track_box(
    previous_gray: np.ndarray, current_gray: np.ndarray, box: Box
) -> tuple[Box, float] | None:
    """用上一帧（修复前）的选框内容做模板，在当前帧的搜索窗里重新定位。

    `TM_CCOEFF_NORMED` 对亮度整体漂移不敏感；命中分数低于 `TRACK_MIN_SCORE` 视为跟丢，
    调用方保持上一帧位置（比乱跳更安全）。
    """
    height, width = current_gray.shape[:2]
    x0, y0 = max(0, box.x0), max(0, box.y0)
    x1, y1 = min(width, box.x1), min(height, box.y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    template = previous_gray[y0:y1, x0:x1]
    box_w, box_h = x1 - x0, y1 - y0
    margin_x = max(10, int(box_w * TRACK_WINDOW))
    margin_y = max(10, int(box_h * TRACK_WINDOW))
    sx0, sy0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
    sx1, sy1 = min(width, x1 + margin_x), min(height, y1 + margin_y)
    search = current_gray[sy0:sy1, sx0:sx1]
    if search.shape[0] < box_h or search.shape[1] < box_w:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if not np.isfinite(score) or score < TRACK_MIN_SCORE:
        return None
    new_x0 = sx0 + int(location[0])
    new_y0 = sy0 + int(location[1])
    return Box(new_x0, new_y0, new_x0 + box_w, new_y0 + box_h), float(score)


def _sample_gray_frames(
    input_path: Path,
    width: int,
    height: int,
    duration: float,
    should_stop: StopCheck | None = None,
) -> list[np.ndarray]:
    """全片等分抽几帧灰度图，只用来量每个像素的时间波动。"""
    if duration <= 0:
        return []
    frames: list[np.ndarray] = []
    size = width * height
    for index in range(MASK_SAMPLES):
        if should_stop is not None and should_stop():
            raise CancelRequested("已停止：逐帧修复已中断")
        moment = duration * (index + 0.5) / MASK_SAMPLES
        command = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{moment:.3f}",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:{height}:flags=bilinear,setsar=1,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if len(result.stdout) >= size:
            frames.append(
                np.frombuffer(result.stdout[:size], dtype=np.uint8)
                .reshape(height, width)
                .astype(np.float32)
            )
    return frames if len(frames) >= 3 else []


def _static_mask_plan(
    frames: list[np.ndarray], box: Box, width: int, height: int
) -> dict | None:
    """在选框内找出「一直没怎么变」的像素——静止水印的笔画。

    框内像素的时间标准差 ÷ 框外一圈的中位标准差 = 该像素被水印盖住的程度：
    完全不透明 ≈ 0，半透明白字 ≈ 1−α，纯背景 ≈ 1。低于 `MASK_STD_LO` 判成水印本体。
    返回 None 表示这条路在这处选框上不成立，交回原来的整框修复。
    """
    cx0, cy0 = max(0, box.x0 - MASK_PAD), max(0, box.y0 - MASK_PAD)
    cx1, cy1 = min(width, box.x1 + MASK_PAD), min(height, box.y1 + MASK_PAD)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    spread = np.stack([frame[cy0:cy1, cx0:cx1] for frame in frames]).std(axis=0)
    inner = np.zeros(spread.shape, np.uint8)
    ix0, iy0 = box.x0 - cx0, box.y0 - cy0
    ix1, iy1 = ix0 + (box.x1 - box.x0), iy0 + (box.y1 - box.y0)
    inner[iy0:iy1, ix0:ix1] = 1
    ring = spread[inner == 0]
    if ring.size < 64:
        return None
    reference = float(np.median(ring))
    if reference < MASK_RING_MIN_STD:
        # 背景本身几乎不动（固定机位空镜、纯色底）：「没怎么变」分不开水印与背景
        return None
    weight = 1.0 - np.clip(
        (spread / reference - MASK_STD_LO) / (MASK_STD_HI - MASK_STD_LO), 0.0, 1.0
    )
    mask = (weight * inner * 255.0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    # 膨胀后仍只落在用户框内：框外像素必须逐像素不变
    mask = (mask.astype(np.float32) / 255.0) * inner
    share = float(mask[inner == 1].mean())
    if share < MASK_MIN_SHARE or share > MASK_MAX_SHARE:
        return None
    return {
        "crop": (cx0, cy0, cx1, cy1),
        "mask": mask,
        "feather": cv2.GaussianBlur(mask, (0, 0), MASK_FEATHER)[..., None],
        "share": share,
    }


def _repair_with_mask(frame: np.ndarray, plan: dict) -> None:
    """只补掩膜内的笔画像素：先用四周像素内插一块底，再羽化混回去。"""
    cx0, cy0, cx1, cy1 = plan["crop"]
    crop = frame[cy0:cy1, cx0:cx1]
    padded = np.pad(crop, ((MASK_CTX, MASK_CTX), (MASK_CTX, MASK_CTX), (0, 0)), mode="edge")
    hole = np.pad((plan["mask"] * 255.0).astype(np.uint8), MASK_CTX)
    filled = cv2.inpaint(padded, hole, 4, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), MASK_FILL_BLUR)
    filled = filled[MASK_CTX:-MASK_CTX, MASK_CTX:-MASK_CTX]
    feather = plan["feather"]
    blended = crop.astype(np.float32) * (1.0 - feather) + filled * feather
    crop[...] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _process_frame_by_frame(
    input_path: Path,
    output_path: Path,
    boxes: list[Box],
    width: int,
    height: int,
    duration: float,
    fps: float,
    method: str,
    on_progress: ProgressCallback,
    motion: bool = False,
    stream_args: list[str] | None = None,
    trim_to_video: bool = True,
    mask_plans: list[dict | None] | None = None,
    should_stop: StopCheck | None = None,
) -> dict:
    usable_fps = fps if fps and fps > 0.1 else 25.0
    estimated_frames = max(1, round(duration * usable_fps)) if duration > 0 else 0
    decode_command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={usable_fps:.6f},scale={width}:{height}:flags=lanczos,setsar=1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    encode_command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{usable_fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        *(stream_args or []),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        *(["-shortest"] if trim_to_video else []),
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    decoder = subprocess.Popen(
        decode_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        encode_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    decode_errors: list[str] = []
    encode_errors: list[str] = []
    threads = [
        threading.Thread(target=_read_stream, args=(decoder.stderr, decode_errors), daemon=True),
        threading.Thread(target=_read_stream, args=(encoder.stderr, encode_errors), daemon=True),
    ]
    for thread in threads:
        thread.start()

    fallback_count = 0
    directions: list[str] = []
    frame_index = 0
    last_report = 0.0
    tracked_boxes = [Box(box.x0, box.y0, box.x1, box.y1) for box in boxes] if motion else boxes
    previous_gray: np.ndarray | None = None
    tracked_frames = 0
    lost_moves = 0
    plans = list(mask_plans or [])
    if motion or len(plans) != len(boxes):
        plans = [None] * len(boxes)
    masked = [(box, plan) for box, plan in zip(boxes, plans) if plan is not None]
    plain = [box for box, plan in zip(boxes, plans) if plan is None]
    guard = _PipeGuard([decoder, encoder], "逐帧修复")
    encode_code: int | None = None
    decode_code: int | None = None
    # 一整批帧复用同一块缓冲：每帧省掉一次 1.5–25 MB 的分配与拷贝。
    # 前提是修复与写管道都在本轮内完成——`stdin.write` 是阻塞的，返回时字节已交给管道，
    # 所以下一帧覆盖它不会改写已经发出去的内容。
    frame = np.empty((height, width, 3), dtype=np.uint8)
    try:
        if decoder.stdout is None or encoder.stdin is None:
            raise VideoProcessError("无法建立视频处理通道")
        while True:
            if not _read_frame(decoder.stdout, frame):
                break
            guard.beat()
            current_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if motion else None
            if motion and previous_gray is not None:
                updated: list[Box] = []
                moved = 0
                for box in tracked_boxes:
                    hit = _track_box(previous_gray, current_gray, box)
                    if hit is None:
                        lost_moves += 1
                        updated.append(box)
                    else:
                        updated.append(hit[0])
                        moved += 1
                tracked_boxes = updated
                if moved:
                    tracked_frames += 1
            for _box, plan in masked:
                _repair_with_mask(frame, plan)
            # 跟踪模式下 plans 一律为 None，修复目标必须是这一帧定位到的 tracked_boxes；
            # 写成固定的 boxes 会让「动态水印」悄悄退化成固定框（残差与固定框一模一样）。
            targets = tracked_boxes if motion else plain
            if targets:
                detail = remove_boxes_inplace(frame, targets, method)
                fallback_count += detail["fallback_regions"]
                directions.extend(detail["directions"])
            previous_gray = current_gray
            encoder.stdin.write(memoryview(frame).cast("B"))
            frame_index += 1
            now = time.monotonic()
            if now - last_report >= 0.25:
                if should_stop is not None and should_stop():
                    raise CancelRequested("已停止：逐帧修复已中断")
                progress = frame_index / estimated_frames if estimated_frames else 0.5
                on_progress(min(0.98, max(0.01, progress)))
                last_report = now
        encoder.stdin.close()
        guard.stop()
        try:
            encode_code = encoder.wait(timeout=PIPE_FLUSH_TIMEOUT)
        except subprocess.TimeoutExpired:
            guard.reason = f"编码器收尾（写出文件头）超过 {int(PIPE_FLUSH_TIMEOUT)} 秒"
        try:
            decode_code = decoder.wait(timeout=PIPE_IDLE_TIMEOUT)
        except subprocess.TimeoutExpired:
            guard.reason = guard.reason or f"解码进程收尾超过 {int(PIPE_IDLE_TIMEOUT)} 秒"
    except (BrokenPipeError, OSError) as exc:
        guard.raise_if_stuck()
        raise VideoProcessError(
            "逐帧视频处理中断，文件可能过大或编码失败。请查看日志或改试更短的视频。"
        ) from exc
    finally:
        guard.stop()
        for process in (decoder, encoder):
            if process.poll() is None:
                process.kill()
        for thread in threads:
            thread.join(timeout=2)

    guard.raise_if_stuck()
    if decode_code is None or decode_code != 0 or frame_index == 0:
        raise VideoProcessError(f"视频逐帧读取失败：{_stderr_text(decode_errors)}")
    if encode_code is None or encode_code != 0:
        raise VideoProcessError(f"视频重新编码失败：{_stderr_text(encode_errors)}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoProcessError("逐帧处理没有生成有效视频")
    on_progress(1.0)
    return {
        "frames": frame_index,
        "fallback_regions": fallback_count,
        "directions": directions,
        "tracked_frames": tracked_frames,
        "lost_moves": lost_moves,
        "mask_boxes": len(masked),
        "plain_boxes": len(plain),
    }


def process_video_file(
    input_path: Path,
    output_path: Path,
    regions: list[dict],
    width: int,
    height: int,
    method: str = options.DEFAULT_METHOD,
    on_progress: ProgressCallback | None = None,
    video_mode: str = options.DEFAULT_VIDEO_MODE,
    motion: bool = False,
    should_stop: StopCheck | None = None,
) -> dict:
    progress = on_progress or (lambda _value: None)
    if should_stop is not None and should_stop():
        raise CancelRequested("已停止：视频处理尚未开始")
    if not regions:
        raise VideoProcessError("没有设置水印区域")
    probe = probe_video(input_path)
    duration = float(probe.get("duration") or 0)
    # 进度按画面轨自己的长度算：容器时长会被长音轨带偏（实测 8.01s 画面 + 30s 音轨只走到 26%）
    length = probe_video_length(input_path)
    if length.get("duration") and 0 < length["duration"] < duration:
        duration = length["duration"]
    fps = float(probe.get("fps") or 0)
    padding = max(1, round(min(width, height) * 0.003))
    boxes = regions_to_boxes(regions, width, height, padding)
    if not boxes:
        raise VideoProcessError("水印区域太小或无效")
    if any(box.width < 3 or box.height < 3 for box in boxes):
        raise VideoProcessError("水印选框太小，请重新框选")

    speed_first = video_mode == options.FAST
    method = options.method_for_video(method)
    edge_box = _touches_edge(boxes, width, height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_name(output_path.stem + ".part.mp4")
    delogo = _delogo_rects(boxes, width, height, allow_inset=speed_first)
    stream_plan = _plan_streams(probe_streams(input_path))
    stream_args = _stream_args(stream_plan, 1)
    # 画质优先 + 智能修复：多花一两秒采样，能按「静止水印」定位笔画就逐帧只补这些像素；
    # 一处理解不出掩膜（背景本身不动、水印在移动、整块实心台标）就整批按原路线走，行为不变。
    mask_plans: list[dict | None] = [None] * len(boxes)
    if not motion and not speed_first and method == options.SMART:
        samples = _sample_gray_frames(input_path, width, height, duration, should_stop)
        if samples:
            mask_plans = [_static_mask_plan(samples, box, width, height) for box in boxes]
    mask_count = sum(1 for plan in mask_plans if plan is not None)
    try:
        if motion:
            # 会移动的水印没有固定矩形可交给 delogo；逐帧的 smart 路线要按整帧拟合色度，太慢，
            # 所以跟踪模式固定用内容填充，并在说明里如实写清楚。
            detail = _process_frame_by_frame(
                input_path,
                part_path,
                boxes,
                width,
                height,
                duration,
                fps,
                options.INPAINT,
                progress,
                motion=True,
                stream_args=stream_args,
                trim_to_video=stream_plan["trim"],
                should_stop=should_stop,
            )
            note = _with_streams("动态水印：已逐帧重新定位选框后修复（耗时较长）", stream_plan)
            if detail.get("lost_moves"):
                note += f"；有 {detail['lost_moves']} 帧没跟住，按上一帧位置修复"
            used = "frame-track"
        elif delogo is not None and not mask_count:
            _process_with_delogo(
                input_path,
                part_path,
                delogo,
                width,
                height,
                duration,
                progress,
                _stream_args(stream_plan, 0),
                should_stop=should_stop,
            )
            detail = {"fallback_regions": 0, "directions": []}
            note = _with_streams(
                "速度优先：贴边选框内缩后走画面内插值，快约 3–4 倍，贴边处残留可能更少也可能更明显"
                if edge_box
                else "视频已用画面内插值修复",
                stream_plan,
            )
            used = "delogo"
        else:
            detail = _process_frame_by_frame(
                input_path,
                part_path,
                boxes,
                width,
                height,
                duration,
                fps,
                method,
                progress,
                stream_args=stream_args,
                trim_to_video=stream_plan["trim"],
                mask_plans=mask_plans,
                should_stop=should_stop,
            )
            base_note = (
                "贴边选框内缩后放不下插值区域，已改用逐帧修复（耗时较长）"
                if speed_first
                else "水印选框触到画面边缘，插值修复放不下，已改用逐帧修复（耗时较长）"
                if edge_box
                else "已按画质优先逐帧修复（耗时较长）"
            )
            masked_boxes = detail.get("mask_boxes") or 0
            heavy_boxes = sum(
                1 for plan in mask_plans
                if plan is not None and plan["share"] > MASK_HEAVY_SHARE
            )
            if masked_boxes == len(boxes) and not heavy_boxes:
                base_note += "；只按水印笔画补像素，框内其余画面原样保留"
            elif masked_boxes:
                parts = []
                if masked_boxes - heavy_boxes:
                    parts.append(f"{masked_boxes - heavy_boxes} 处按水印笔画逐像素修补")
                if heavy_boxes:
                    parts.append(f"{heavy_boxes} 处选框大部分被水印盖住，已整片补画")
                if len(boxes) - masked_boxes:
                    parts.append(f"{len(boxes) - masked_boxes} 处是整块台标，仍整框修复")
                base_note += "；其中 " + "、".join(parts)
            note = _with_streams(base_note, stream_plan)
            used = "frame-mask" if masked_boxes == len(boxes) and not heavy_boxes else f"frame-{method}"
        part_path.replace(output_path)
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return {
        "method": used,
        "regions": len(boxes),
        "frames": detail.get("frames"),
        "note": note,
        "width": width,
        "height": height,
        "video_mode": options.FAST if speed_first else options.QUALITY,
        "motion": bool(motion),
    }
