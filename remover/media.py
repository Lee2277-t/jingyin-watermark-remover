from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"}
BROWSER_VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm"}
MAX_IMAGE_PIXELS = 80_000_000


class MediaError(Exception):
    pass


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def media_kind(suffix: str) -> str | None:
    suffix = suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def clean_name(name: str, fallback: str = "素材") -> str:
    name = Path(name or fallback).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:160] or fallback


def probe_image(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise MediaError("图片像素过大，暂不支持处理")
            return {"width": width, "height": height}
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"无法读取图片：{exc}") from exc


def make_image_thumb(
    source: Path,
    target: Path,
    max_side: int = 560,
    quality: int = 88,
    subsampling: int = 2,
) -> None:
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (246, 244, 239))
                background.paste(image, mask=image.getchannel("A"))
                image = background
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            part_path = target.with_name(target.name + ".part")
            try:
                image.save(part_path, "JPEG", quality=quality, optimize=True, subsampling=subsampling)
                os.replace(part_path, target)
            finally:
                if part_path.exists():
                    os.remove(part_path)
    except Exception as exc:
        raise MediaError(f"无法生成图片预览：{exc}") from exc


def make_preview(source: Path, target: Path) -> None:
    """大图预览统一口径：2400px 长边、JPEG q95、4:4:4（subsampling=0 才不糊水印色边）。"""
    make_image_thumb(source, target, max_side=2400, quality=95, subsampling=0)


def _parse_ffmpeg_duration(stderr: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = 0.0
    frame_count = 0
    width = 0
    height = 0
    duration = 0.0
    if capture.isOpened():
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
        capture.release()

    if duration <= 0 or width <= 0 or height <= 0:
        try:
            result = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-i", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MediaError("读取视频信息超时或失败，请确认视频文件完整") from exc
        parsed_duration = _parse_ffmpeg_duration(result.stderr)
        if parsed_duration:
            duration = parsed_duration
        if width <= 0 or height <= 0:
            match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", result.stderr)
            if match:
                width, height = int(match.group(1)), int(match.group(2))

    if duration <= 0 or width <= 0 or height <= 0:
        raise MediaError("无法读取视频信息，请确认视频文件完整")

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
    }


def probe_video_length(path: Path) -> dict:
    """画面轨自己的长度（秒 + 帧数）；读不到时回空字典，由调用方退回容器口径。

    容器时长会被更长的音轨带偏：实测 8.01s 画面配 30s 音轨的 mkv，cv2 报成 30.03s / 720 帧，
    于是全片采样的后半段时间点全落在片尾之后（`extract_video_frame` 抛错、被静默跳过，
    「全片采样 2 帧」），逐帧进度也只走到 26%。这一趟只解封装不解码，实测 52s 片子 0.09s。
    """
    try:
        result = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-loglevel", "info", "-i", str(path),
             "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
        )
    except Exception:
        return {}
    stats = next((line for line in reversed(result.stderr.splitlines())
                  if "time=" in line and "frame=" in line), "")
    time_match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", stats)
    if not time_match:
        return {}
    hours, minutes, seconds = time_match.groups()
    frames_match = re.search(r"frame=\s*(\d+)", stats)
    return {
        "duration": int(hours) * 3600 + int(minutes) * 60 + float(seconds),
        "frames": int(frames_match.group(1)) if frames_match else 0,
    }


STREAM_LINE = re.compile(
    r"Stream #\d+:(\d+)(?:\[[^\]]*\])?(?:\([^)]*\))?: (Video|Audio|Subtitle): ([A-Za-z0-9_]+)"
)

# ffmpeg 把色彩标记写在画面流那一行的像素格式后面，形如
#   yuv420p10le(tv, bt2020nc/bt2020/smpte2084)
#   yuv420p(tv, bt709, progressive)
# 括号里第一段是 range，第二段才是色彩标记；三段齐全时依次是
# matrix / primaries / transfer（ffmpeg 的 avcodec_string 就是这个顺序），
# 只有一段时是 primaries。
COLOR_TAG = re.compile(r"([a-z0-9_]+)\(([a-z0-9]+),\s*([^)]*)\)", re.IGNORECASE)

# 只有这两个传输特性是 HDR；bt2020 只是色域，不能单凭它判断 HDR。
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}

# 括号里除了色彩标记还会混进扫描方式（progressive / interlaced），
# 只认这几个 ffmpeg 会打印的色彩名，别把 progressive 当成 matrix。
COLOR_NAMES = {
    "bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film",
    "bt2020nc", "bt2020c", "bt2020", "smpte428", "smpte431", "smpte432",
    "smpte2084", "arib-std-b67", "iec61966-2-1", "iec61966-2-4",
    "linear", "log", "log_sqrt", "iec61966-2-2", "bt2020-10", "bt2020-12",
    "unknown",
}


def _video_color(text: str) -> dict:
    """从 `ffmpeg -i` 的 stderr 里认出画面流的像素格式与色彩标记。

    认不出就返回空字典：主流程照跑，只是说明里不写 HDR 那句话。
    """
    video_at = None
    for match in STREAM_LINE.finditer(text):
        if match.group(2) == "Video":
            video_at = match.end()
            break
    if video_at is None:
        return {}
    line_end = text.find("\n", video_at)
    rest = text[video_at : line_end if line_end >= 0 else len(text)]
    tag = COLOR_TAG.search(rest)
    if not tag:
        return {}
    inner = tag.group(3).split(",")[0].strip()
    parts = [item.strip().lower() for item in inner.split("/")]
    parts = [item for item in parts if item in COLOR_NAMES]
    if not parts:
        return {}
    if len(parts) == 1:
        matrix, primaries, transfer = "", parts[0], ""
    else:
        matrix = parts[0] if len(parts) > 0 else ""
        primaries = parts[1] if len(parts) > 1 else ""
        transfer = parts[2] if len(parts) > 2 else ""
    return {
        "pix_fmt": tag.group(1).lower(),
        "range": tag.group(2).lower(),
        "matrix": matrix,
        "primaries": primaries,
        "transfer": transfer,
        "hdr": transfer in HDR_TRANSFERS,
    }


def probe_streams(path: Path) -> dict:
    """列出声轨与字幕轨（索引 + 编码），用于决定成品里哪些能原样带过去。

    ffmpeg 只给 `-i` 时一定会以非零退出（没有输出文件），流信息在 stderr 里，正常。
    读不到就返回空清单：主流程照跑，只是成品不带音轨/字幕。
    """
    try:
        result = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30,
        )
        text = result.stderr or ""
    except (OSError, subprocess.SubprocessError):
        text = ""
    audio: list[dict] = []
    subtitles: list[dict] = []
    for match in STREAM_LINE.finditer(text):
        index, kind, codec = int(match.group(1)), match.group(2), match.group(3).lower()
        if kind == "Audio":
            audio.append({"index": index, "codec": codec})
        elif kind == "Subtitle":
            subtitles.append({"index": index, "codec": codec})
    return {"audio": audio, "subtitles": subtitles, "video": _video_color(text)}


def make_video_poster(source: Path, target: Path, duration: float, max_side: int = 720) -> dict:
    moment = min(1.0, max(0.0, duration * 0.12))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        command = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{moment:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(tmp_path),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        with Image.open(tmp_path) as frame:
            width, height = frame.size
            frame = frame.convert("RGB")
            frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.save(target, "JPEG", quality=88, optimize=True)
        return {"width": width, "height": height}
    except Exception as exc:
        raise MediaError(f"无法提取视频画面：{exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def extract_video_frame(source: Path, time_seconds: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        command = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, time_seconds):.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(tmp_path),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        with Image.open(tmp_path) as frame:
            return np.array(frame.convert("RGB"))
    except Exception as exc:
        raise MediaError(f"无法读取视频当前画面：{exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
