# -*- coding: utf-8 -*-
"""真实影像复核（第十五轮起）：把此前只用合成夹具验过的视频四项，拿到真人实拍/动画正片上重量一遍。

素材（公开授权，留在 work/qa 当长期真实语料）：
- `_real_oceans.mp4` 真人实拍 960×400 / 23.98fps / 46.5s / 一条真实 AAC
- `_real_sintel.mp4` 854×480 / 24fps / 52.2s（够长，用来量 5 个采样点到底覆盖得住多短的台标）
- `_real_bbb.mp4`   320×176 / **两条真实 AAC 音轨**（多音轨搬运的真素材）
- `_real_jelly1080.mp4` 1920×1080 / 29.97fps / 10s 水下实拍（第十七轮的画质优先掩膜与 4K 夹具底片）
- `_bbb4k_raw.mp4` **原生 3840×2160 / 60fps / 180 帧 / 3.0s / H.264 SDR**（第二十四轮）：从真 HEVC 4K
  正片里 `-ss 120` 抠 3 秒重编码；第十七轮的 4K 夹具是 1080p 上采样，这份的细节是原生 4K。
  下载：curl -L -o work/qa/_bbb4k_src.mkv http://distribution.bbb3d.renderfarming.net/video/mp4/bbb-3840x2160-cfg02.mkv
        （很大，取完 3 秒就删；或用 samples.mplayerhq.hu 的 4K 样片）
- `_real_leopard4k.mkv` 真 4K HEVC **Main10 HDR**（bt2020/smpte2084）+ DTS-HD + PGS 字幕的 mkv，
  下载被截断（画面轨只有 27 帧），但**两条视频路线都能在它上面完整跑通**——`hdr` 相就是这么跑的，
  它同时是「HDR 源成品会不会褪色/偏灰」唯一的一份真语料。第二十八轮实测：产品**不做任何色调映射**
  （PQ 值直接当 sRGB 用），于是成品比正确的 HDR→SDR **更亮**——中位亮度 79.2/79.3/79.4 对
  60.9/61.1/61.3（差 25.89–25.98 级），黑位被抬起来、观感偏灰；`quality` 逐帧路径还会丢掉 HDR 色彩
  元数据（成品容器是 `yuv420p` 无标记），`fast`/delogo 单趟 ffmpeg 则保留 bt2020nc/smpte2084。这三条
  都是如实的能力边界，不是待修的 bug（判据是 `hdr` 相那 18 项断言）。
  来源：https://samples.mplayerhq.hu/ （4K 目录里的 HEVC 样片）
- `_gs_pgs_small.mkv` 720×480 / 240 帧 / 无音轨 / **hdmv_pgs_subtitle**
- `_gs_pgs_dragon.mkv` 1920×1080 / 47 帧 / 无音轨 / **hdmv_pgs_subtitle**
- `_gs_dvd.mkv` 352×288 / 830 帧 / **ac3 + dvd_subtitle**
- `_gs_dvb.mkv` 1920×1080 / 42 帧 / **ac3 + dvb_subtitle**（从三节目 .ts 里 remux 出可用节目）
- `_gs_dvb.ts` 同上的原始三节目 TS（`.ts` 不在上传白名单，只记录不修）
  四个图形字幕样片来自 https://samples.mplayerhq.hu/sub/ ；remux 一律带 `-fflags +genpts`，
  否则 dvdsub 包没有时间戳、`-c copy` 直接失败（"Can't write packet with unknown timestamp"）。

水印不是 lavfi 渐变底上的色块：本地 PIL 画出半透明白字、斜向平铺彩章与台标，overlay 到真实画面后
整帧重新过一遍 H.264 —— 字体渲染、4:2:0 色度抽样、压缩噪声都是真的。几何由本脚本决定，所以
ground truth 是精确像素框。

不读用户作品；成品只落 work/qa，只有 `http` 这一相会写一次 成品/ 并在跑完后清掉。

跑法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_real_video.py all
    ... 子命令 build | detect | mode | mask | motion | streams | uhd | subpic | hdr | http | all
    （加 --keep 保留 _rv_* 夹具；_real_* 与 _gs_* 是本脚本的长期语料，cleanup 不碰）
"""

from __future__ import annotations

import http.cookiejar
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import server  # noqa: E402
from remover import media, options, video_ops  # noqa: E402
from remover.detection import detect_watermark  # noqa: E402

QA = Path(__file__).resolve().parent
OCEANS = QA / "_real_oceans.mp4"
SINTEL = QA / "_real_sintel.mp4"
BBB = QA / "_real_bbb.mp4"
# 第十七轮新增：真 1920×1080 水下实拍（test-videos.co.uk 的 Jellyfish 样本，10s / 29.97fps / 无音轨），
# 用来验「静态水印掩膜」在全新画面上成立不成立，以及真分辨率 4K 上的小台标检测。
# 下载：curl -L -o work/qa/_real_jelly1080.mp4 \
#   https://test-videos.co.uk/vids/jellyfish/mp4/h264/1080/Jellyfish_1080_10s_5MB.mp4
JELLY = QA / "_real_jelly1080.mp4"
GT_FILE = QA / "_rv_gt.json"
MASK_GT_FILE = QA / "_rv_mask_gt.json"

CJK_FONT = Path("C:/Windows/Fonts/msyhbd.ttc")
LATIN_FONT = Path("C:/Windows/Fonts/arialbd.ttf")
SECONDS = 8.0
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{result.stderr[-500:]}")


def overlay_clip(base: Path, out: Path, layers: list[str], specs: list[str],
                 duration: float, with_audio: bool = True) -> None:
    """把若干 PNG 叠到真实底片上重编码；layers/specs 一一对应，specs 是 overlay 的参数串。"""
    graph = ""
    previous = "[0:v]"
    for index, spec in enumerate(specs, start=1):
        label = "[v]" if index == len(specs) else f"[m{index}]"
        graph += f"{previous}[{index}:v]overlay={spec}{label};"
        previous = label
    graph = graph.rstrip(";") or "[0:v]null[v]"
    tail = ["-t", str(duration), "-map", "[v]"]
    tail += (["-map", "0:a", "-c:a", "copy"] if with_audio else ["-an"])
    tail += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    inputs = ["-ss", "0", "-i", str(base)]
    for layer in layers:
        inputs += ["-i", layer]
    run_ffmpeg([*inputs, "-filter_complex", graph, *tail, str(out)])


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def region_from_px(box: dict, width: int, height: int, grow: int = 0) -> dict:
    return {
        "x": (box["x"] - grow) / width,
        "y": (box["y"] - grow) / height,
        "w": (box["w"] + grow * 2) / width,
        "h": (box["h"] + grow * 2) / height,
    }


def px_box(region: dict, width: int, height: int) -> dict:
    return {"x": region["x"] * width, "y": region["y"] * height,
            "w": region["w"] * width, "h": region["h"] * height}


# --------------------------------------------------------------------------
# build：造真实带标素材（PIL 画水印 → overlay 到实拍画面 → 重新 H.264）
# --------------------------------------------------------------------------

CLEAN = QA / "_rv_clean.mp4"
MARKED = QA / "_rv_marked.mp4"
MOVE = QA / "_rv_move.mp4"

CORNER = QA / "_rv_corner.png"
STAMP = QA / "_rv_stamp.png"
CENTER_LOGO = QA / "_rv_logo_big.png"
BUG = QA / "_rv_bug.png"
BUG_WHITE = QA / "_rv_bug_white.png"
TRACK_LOGO = QA / "_rv_logo.png"

MOVE_X0, MOVE_DX, MOVE_Y_OFF, MOVE_AMP, MOVE_PERIOD = 80.0, 70.0, 40.0, 16.0, 2.5


def render_corner(width: int, height: int) -> dict:
    img = Image.new("RGBA", (190, 46), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    text = "AI 生成"
    f = font(CJK_FONT, 26)
    draw.text((95, 24), text, font=f, fill=(0, 0, 0, 150), anchor="mm", stroke_width=1)
    draw.text((94, 23), text, font=f, fill=(255, 255, 255, 205), anchor="mm")
    img.save(CORNER)
    return {"x": width - 190, "y": height - 46, "w": 190, "h": 46}


def render_stamp(width: int, height: int) -> tuple[dict, int]:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    tile = Image.new("RGBA", (230, 44), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text(
        (115, 22), "净印 · 样片", font=font(CJK_FONT, 24), fill=(42, 106, 216, 130),
        anchor="mm", stroke_width=1, stroke_fill=(192, 57, 43, 95))
    tile = tile.rotate(-28, expand=True, resample=Image.BICUBIC)
    tiles: list[dict] = []
    pitch_x, pitch_y = 300, 150
    for row in range(-1, height // pitch_y + 2):
        for col in range(-1, width // pitch_x + 2):
            ox = col * pitch_x + (abs(row) % 2) * pitch_x // 2
            oy = row * pitch_y
            img.alpha_composite(tile, (ox, oy))
            if 0.34 < oy / height < 0.66 and 0.34 < ox / width < 0.66:
                tiles.append({"x": ox, "y": oy, "w": tile.size[0], "h": tile.size[1]})
    img.save(STAMP)
    center = tiles[len(tiles) // 2] if tiles else {
        "x": width // 3, "y": height // 3, "w": tile.size[0], "h": tile.size[1]}
    return center, len(tiles)


def render_logo(path: Path, size: tuple[int, int], text: str, latin: bool = True) -> None:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=6, fill=(18, 26, 44, 130))
    draw.text((size[0] / 2, size[1] / 2), text, font=font(LATIN_FONT if latin else CJK_FONT, 26),
              fill=(255, 255, 255, 220), anchor="mm")
    img.save(path)


def build() -> None:
    info = media.probe_video(OCEANS)
    width, height = int(info["width"]), int(info["height"])
    corner = render_corner(width, height)
    stamp_tile, tile_count = render_stamp(width, height)
    center = {"x": width // 2 - 120, "y": height // 2 - 40, "w": 240, "h": 80}
    render_logo(CENTER_LOGO, (240, 80), "SAMPLE HD")
    render_logo(TRACK_LOGO, (128, 44), "LIVE TV")
    render_logo(BUG, (150, 46), "净印频道", latin=False)
    white = Image.new("RGBA", (150, 46), (0, 0, 0, 0))
    ImageDraw.Draw(white).text((75, 23), "AI 生成", font=font(CJK_FONT, 26),
                               fill=(255, 255, 255, 215), anchor="mm",
                               stroke_width=1, stroke_fill=(0, 0, 0, 120))
    white.save(BUG_WHITE)

    overlay_clip(OCEANS, CLEAN, [], [], SECONDS)
    overlay_clip(OCEANS, MARKED,
                 [str(CORNER), str(STAMP), str(CENTER_LOGO)],
                 [f"x={corner['x']}:y={corner['y']}", "x=0:y=0", f"x={center['x']}:y={center['y']}"],
                 SECONDS)
    base_y = height - 44 - MOVE_Y_OFF
    overlay_clip(OCEANS, MOVE, [str(TRACK_LOGO)],
                 [f"x='{MOVE_X0:.0f}+{MOVE_DX:.0f}*t':y='H-h-{MOVE_Y_OFF:.0f}+{MOVE_AMP:.0f}*sin(2*PI*t/{MOVE_PERIOD})'"],
                 6.0)

    gt = {
        "width": width, "height": height, "fps": info.get("fps"),
        "corner": corner, "corner_region": region_from_px(corner, width, height, grow=3),
        "stamp": stamp_tile, "stamp_region": region_from_px(stamp_tile, width, height, grow=3),
        "center": center, "center_region": region_from_px(center, width, height, grow=3),
        "tiles": tile_count,
        "move": {"x0": MOVE_X0, "dx": MOVE_DX, "y0": base_y, "amp": MOVE_AMP,
                 "period": MOVE_PERIOD, "w": 128, "h": 44},
        "bug": {"x": int(media.probe_video(SINTEL)["width"]) - 154, "y": 12, "w": 150, "h": 46},
    }
    GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"真实底片 {width}x{height} {info.get('fps')}fps；平铺章共 {tile_count} 块，"
          f"中央块 {stamp_tile['w']}x{stamp_tile['h']} @({stamp_tile['x']},{stamp_tile['y']})")
    build_detect_clips()
    build_bug_sequences(6.0)


def build_detect_clips() -> None:
    """sintel 全片 52s：台标只出现在开头/中段的各种时长与各种位置，看 5–9 个采样点抓不抓得到。
    其中 mid12 与 mid12b **一样长、只差 5s 位置**——一个跨两个采样点、一个只盖一个，用来把
    「成不成框取决于落在网格哪儿，不是取决于多长」这条边界钉住。"""
    info = media.probe_video(SINTEL)
    duration = float(info["duration"])
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    x, y = gt["bug"]["x"], gt["bug"]["y"]
    windows = [
        ("_rv_detect_intro2.mp4", f"enable='lt(t,2)'"),
        ("_rv_detect_intro6.mp4", f"enable='lt(t,6)'"),
        ("_rv_detect_intro12.mp4", f"enable='lt(t,12)'"),
        ("_rv_detect_mid4.mp4", f"enable='between(t,{duration / 2 - 2},{duration / 2 + 2})'"),
        ("_rv_detect_mid12.mp4", f"enable='between(t,{duration / 2 - 1.1},{duration / 2 + 10.9})'"),
        # 同样 12s 长、整段往前挪 5s：只盖住 1 个采样点 → 过不了 2 帧门槛。长度不是决定因素，落在网格哪儿才是。
        ("_rv_detect_mid12b.mp4", f"enable='between(t,{duration / 2 - 6.1},{duration / 2 + 5.9})'"),
        ("_rv_detect_white6.mp4", f"enable='lt(t,6)'"),
        ("_rv_detect_clean.mp4", None),
    ]
    for name, enable in windows:
        out = QA / name
        if enable is None:
            overlay_clip(SINTEL, out, [], [], duration, with_audio=False)
            continue
        layer = str(BUG_WHITE) if "white" in name else str(BUG)
        overlay_clip(SINTEL, out, [layer], [f"x={x}:y={y}:{enable}"], duration, with_audio=False)


def build_bug_sequences(seconds: float) -> None:
    """旋转 / 缩放台标：逐帧由 PIL 画出，可见范围是解析真值，ffmpeg 只负责合成。

    画布固定在画面里某处（`origin`），台标画在画布正中；所以这一帧台标在画面上的
    真实像素框 = origin + 画布中心 − 本帧图样的一半。
    """
    fps = 24
    frames = int(seconds * fps)
    base = Image.open(TRACK_LOGO).convert("RGBA")
    sequences: dict[str, tuple[Path, dict, tuple[int, int]]] = {}
    for kind in ("rot", "scale"):
        dir_path = QA / f"_rv_seq_{kind}"
        dir_path.mkdir(exist_ok=True)
        for name in dir_path.glob("*.png"):
            name.unlink()
        canvas_size = (200, 200) if kind == "rot" else (300, 170)
        origin = (600, 100)
        boxes: list[dict] = []
        for index in range(frames):
            t = index / fps
            if kind == "rot":
                art = base.rotate(30 * math.sin(t * 1.1), expand=True, resample=Image.BICUBIC)
            else:
                factor = 1.0 + 0.006 * index
                art = base.resize((max(4, int(base.size[0] * factor)), max(4, int(base.size[1] * factor))),
                                  Image.BICUBIC)
            paste = (canvas_size[0] // 2 - art.size[0] // 2, canvas_size[1] // 2 - art.size[1] // 2)
            canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            canvas.paste(art, paste, art)
            canvas.save(dir_path / f"{index:03d}.png")
            boxes.append({"x": origin[0] + paste[0], "y": origin[1] + paste[1],
                          "w": art.size[0], "h": art.size[1]})
        span = {
            "x": min(b["x"] for b in boxes), "y": min(b["y"] for b in boxes),
            "w": max(b["x"] + b["w"] for b in boxes) - min(b["x"] for b in boxes),
            "h": max(b["y"] + b["h"] for b in boxes) - min(b["y"] for b in boxes),
        }
        sequences[kind] = (dir_path, {"span": span, "boxes": boxes}, origin)
        (QA / f"_rv_{kind}_gt.json").write_text(json.dumps(sequences[kind][1]), encoding="utf-8")

    for kind, (dir_path, _data, origin) in sequences.items():
        out = QA / f"_rv_{kind}.mp4"
        run_ffmpeg(["-ss", "0", "-t", str(seconds), "-i", str(OCEANS),
                    "-framerate", str(fps), "-i", str(dir_path / "%03d.png"),
                    "-filter_complex", f"[0:v][1:v]overlay=x={origin[0]}:y={origin[1]}:shortest=1[v]",
                    "-map", "[v]", "-map", "0:a", "-c:a", "copy",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)])


# --------------------------------------------------------------------------
# 第十七轮夹具：1080p 新素材 / 静止背景 / 真 4K 分辨率
# --------------------------------------------------------------------------

JELLY_CLEAN = QA / "_rv_jelly_clean.mp4"
JELLY_MARKED = QA / "_rv_jelly_marked.mp4"
STATIC_CLEAN = QA / "_rv_static_clean.mp4"
STATIC_MARKED = QA / "_rv_static_marked.mp4"
UHD_CLEAN = QA / "_rv_uhd_clean.mp4"
UHD_MARKED = QA / "_rv_uhd_marked.mp4"
JELLY_SECONDS = 6.0
UHD_SECONDS = 4.0


def render_badge(path: Path, text: str, size: int, alpha: int = 205) -> tuple[int, int]:
    """半透明白字 + 黑描边，字号随素材分辨率走（1080p 用 52、4K 小台标用 24）。"""
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    face = font(CJK_FONT, size)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face, stroke_width=1)
    width, height = right - left + 8, bottom - top + 8
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    center = (width / 2, height / 2)
    draw.text(center, text, font=face, fill=(0, 0, 0, 150), anchor="mm", stroke_width=1)
    draw.text((center[0] - 1, center[1] - 1), text, font=face, fill=(255, 255, 255, alpha),
              anchor="mm")
    img.save(path)
    return width, height


def render_plate(path: Path, size: tuple[int, int], text: str, size_pt: int) -> None:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=10, fill=(16, 26, 44, 130))
    draw.text((size[0] / 2, size[1] / 2), text, font=font(LATIN_FONT, size_pt),
              fill=(255, 255, 255, 220), anchor="mm")
    img.save(path)


def build_mask_clips() -> None:
    """① 1080p 真实拍 + 贴边白字（稀疏笔画，掩膜路线该赢）+ 中央整块台标（该退回整框修复）；
    ② 同一帧定格 6s + 贴边白字（背景不动，掩膜原理上不成立，必须退回原路线且不能变差）；
    ③ 把 1080p 上采样到 3840×2160，两个角各放 24px / 48px 小台标，验真分辨率下的原生角标补漏。"""
    if not JELLY.exists():
        raise RuntimeError(f"缺少 {JELLY.name}：见脚本头注释里的下载命令")
    info = media.probe_video(JELLY)
    width, height = int(info["width"]), int(info["height"])
    badge_w, badge_h = render_badge(QA / "_rv_jelly_text.png", "AI 生成", 52)
    corner = {"x": width - badge_w - 8, "y": height - badge_h - 8, "w": badge_w, "h": badge_h}
    plate = {"x": width // 2 - 240, "y": height // 2 - 80, "w": 480, "h": 160}
    render_plate(QA / "_rv_jelly_plate.png", (plate["w"], plate["h"]), "SAMPLE HD", 54)
    overlay_clip(JELLY, JELLY_CLEAN, [], [], JELLY_SECONDS, with_audio=False)
    overlay_clip(JELLY, JELLY_MARKED,
                 [str(QA / "_rv_jelly_text.png"), str(QA / "_rv_jelly_plate.png")],
                 [f"x={corner['x']}:y={corner['y']}", f"x={plate['x']}:y={plate['y']}"],
                 JELLY_SECONDS, with_audio=False)

    still = Image.fromarray(media.extract_video_frame(JELLY, 3.0)).convert("RGBA")
    marked = still.copy()
    marked.alpha_composite(Image.open(QA / "_rv_jelly_text.png").convert("RGBA"),
                           (corner["x"], corner["y"]))
    still.convert("RGB").save(QA / "_rv_static_frame.png")
    marked.convert("RGB").save(QA / "_rv_static_marked.png")
    for source, out in ((QA / "_rv_static_frame.png", STATIC_CLEAN),
                        (QA / "_rv_static_marked.png", STATIC_MARKED)):
        run_ffmpeg(["-loop", "1", "-i", str(source), "-t", str(JELLY_SECONDS), "-r", "24",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(out)])

    uhd_w, uhd_h = 3840, 2160
    small_w, small_h = render_badge(QA / "_rv_uhd_small.png", "AI生成", 24, alpha=215)
    big_w, big_h = render_badge(QA / "_rv_uhd_big.png", "AI生成", 48, alpha=215)
    small = {"x": 40, "y": 32, "w": small_w, "h": small_h}
    big = {"x": uhd_w - big_w - 40, "y": uhd_h - big_h - 40, "w": big_w, "h": big_h}
    run_ffmpeg(["-ss", "0", "-t", str(UHD_SECONDS), "-i", str(JELLY),
                "-vf", f"scale={uhd_w}:{uhd_h}:flags=lanczos,setsar=1",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(UHD_CLEAN)])
    run_ffmpeg(["-ss", "0", "-t", str(UHD_SECONDS), "-i", str(JELLY),
                "-i", str(QA / "_rv_uhd_small.png"), "-i", str(QA / "_rv_uhd_big.png"),
                "-filter_complex",
                f"[0:v]scale={uhd_w}:{uhd_h}:flags=lanczos,setsar=1[base];"
                f"[base][1:v]overlay=x={small['x']}:y={small['y']}[m1];"
                f"[m1][2:v]overlay=x={big['x']}:y={big['y']}[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(UHD_MARKED)])

    MASK_GT_FILE.write_text(json.dumps({
        "width": width, "height": height,
        "corner": corner, "corner_region": region_from_px(corner, width, height, grow=3),
        "plate": plate, "plate_region": region_from_px(plate, width, height, grow=3),
        "uhd_width": uhd_w, "uhd_height": uhd_h,
        "uhd_small": small, "uhd_big": big,
        "uhd_small_region": region_from_px(small, uhd_w, uhd_h, grow=3),
        "uhd_big_region": region_from_px(big, uhd_w, uhd_h, grow=3),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"1080p 夹具 {width}x{height}：贴边白字 {badge_w}x{badge_h}、中央台标 {plate['w']}x{plate['h']}；"
          f"4K 夹具 {uhd_w}x{uhd_h}：小台标 {small_w}x{small_h}、大台标 {big_w}x{big_h}")


# --------------------------------------------------------------------------
# 测量
# --------------------------------------------------------------------------

def frames_at(path: Path, indices: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            out.append(frame)
    capture.release()
    return out


def sample_indices(count: int, how_many: int = 8) -> list[int]:
    if count <= 1:
        return [0]
    step = max(1, count // how_many)
    return [min(count - 1, index * step) for index in range(how_many)]


def slice_box(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    return (max(0, int(box["x"])), max(0, int(box["y"])),
            min(width, int(box["x"] + box["w"])), min(height, int(box["y"] + box["h"])))


def residual_in(path: Path, clean: Path, box: dict, width: int, height: int, frames: int = 8) -> float:
    """选框内 |成品 − 无标基准| 的均值（跨多帧取平均）。"""
    total = int(media.probe_video(path).get("frame_count") or frames)
    indices = sample_indices(total, frames)
    outs, bases = frames_at(path, indices), frames_at(clean, indices)
    x0, y0, x1, y1 = slice_box(box, width, height)
    values = [float(np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16)).mean())
              for a, b in zip(outs, bases)]
    return sum(values) / len(values) if values else 0.0


def outside_diff(out: Path, marked: Path, boxes: list[dict], width: int, height: int) -> float:
    total = int(media.probe_video(out).get("frame_count") or 8)
    indices = sample_indices(total, 6)
    outs, srcs = frames_at(out, indices), frames_at(marked, indices)
    mask = np.ones((height, width), bool)
    for box in boxes:
        x0, y0, x1, y1 = slice_box(px_box(box, width, height), width, height)
        mask[y0:y1, x0:x1] = False
    return max(float(np.abs(a.astype(np.int16) - b.astype(np.int16))[mask].mean())
               for a, b in zip(outs, srcs))


def mark_metrics(out: Path, marked: Path, clean: Path, box: dict, width: int, height: int) -> dict:
    """两个贴近观感的量，而不只看 |成品−基准| 一个数：

    `ghost` = 水印自己那部分像素（|带标−无标| > 25）上还剩多少对比，0 = 擦干净；
    `sharp` = 选框内高频能量相对无标基准的比例，1 = 细节没被抹平（插值修复会明显低于 1）。
    """
    total = int(media.probe_video(out).get("frame_count") or 8)
    indices = sample_indices(total, 8)
    outs, marks, cleans = frames_at(out, indices), frames_at(marked, indices), frames_at(clean, indices)
    x0, y0, x1, y1 = slice_box(box, width, height)
    ghosts, sharp_out, sharp_clean = [], [], []
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    for here, mark, base in zip(outs, marks, cleans):
        window_out = here[y0:y1, x0:x1].astype(np.int16)
        window_mark = mark[y0:y1, x0:x1].astype(np.int16)
        window_base = base[y0:y1, x0:x1].astype(np.int16)
        footprint = np.abs(window_mark - window_base).max(axis=2) > 25
        if footprint.any():
            ghosts.append(float(np.abs(window_out - window_base)[footprint].mean())
                           / float(np.abs(window_mark - window_base)[footprint].mean()))
        lap_out = np.abs(cv2.filter2D(base[y0:y1, x0:x1].astype(np.float32), -1, kernel)).mean()
        lap_here = np.abs(cv2.filter2D(here[y0:y1, x0:x1].astype(np.float32), -1, kernel)).mean()
        if lap_out > 1e-6:
            sharp_out.append(lap_here / lap_out)
        sharp_clean.append(1.0)
    return {
        "ghost": sum(ghosts) / len(ghosts) if ghosts else float("nan"),
        "sharp": sum(sharp_out) / len(sharp_out) if sharp_out else float("nan"),
    }


def temporal_boil(paths: list[tuple[str, Path]], box: dict, width: int, height: int,
                  start: int, count: int) -> dict[str, float]:
    """框内**相邻帧**平均差：逐帧独立修复会不会「沸腾」。素材本身的画面运动是共同基准，
    成品高于素材的那部分才是修复引入的时序抖动；比值 ≤1 说明没引入可感知的闪烁。"""
    x0, y0, x1, y1 = slice_box(box, width, height)
    frames = {label: frames_at(path, list(range(start, start + count))) for label, path in paths}
    result = {}
    for label, seq in frames.items():
        diffs = [float(np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16)).mean())
                 for a, b in zip(seq, seq[1:])]
        result[label] = sum(diffs) / len(diffs) if diffs else float("nan")
    return result


def overlap(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
    return ix * iy / smaller if smaller else 0.0


def run_engine(source: Path, out: Path, regions: list[dict], width: int, height: int,
               mode: str = "quality", motion: bool = False, method: str = "smart") -> tuple[dict, float]:
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    result = video_ops.process_video_file(source, out, regions, width, height, method=method,
                                          on_progress=lambda _v: None, video_mode=mode, motion=motion)
    return result, time.perf_counter() - started


def move_union(move: dict) -> dict:
    xs, ys = [], []
    for index in range(0, 145):
        t = index / 24
        xs.append(move["x0"] + move["dx"] * t)
        ys.append(move["y0"] + move["amp"] * math.sin(2 * math.pi * t / move["period"]))
    return {"x": min(xs) - 4, "y": min(ys) - 4,
            "w": max(xs) + move["w"] - min(xs) + 8, "h": max(ys) + move["h"] - min(ys) + 8}


# --------------------------------------------------------------------------
# 阶段 1：真实素材上的自动识别
# --------------------------------------------------------------------------

def frame0_regions(path: Path) -> int:
    """第十四轮之前的口径：只看第 0 秒那一帧能得几框。"""
    return len(detect_watermark(media.extract_video_frame(path, 0.0))["regions"])


def phase_detect() -> None:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    width, height = gt["width"], gt["height"]
    print("=== 真实素材上的自动识别（第十四轮的多帧采样口径）===")
    cases = [
        ("实拍：角标+平铺章+中央台标", MARKED,
         [gt["corner_region"], gt["stamp_region"], gt["center_region"]], 1),
        ("动画：全片干净（误报检查）", QA / "_rv_detect_clean.mp4", [], 0),
        ("实拍 8s 干净底片（误报检查）", CLEAN, [], 0),
        ("动画：台标只在片头 2s", QA / "_rv_detect_intro2.mp4", None, None),
        ("动画：台标只在片头 6s", QA / "_rv_detect_intro6.mp4", None, None),
        ("动画：台标只在片头 12s", QA / "_rv_detect_intro12.mp4", None, None),
        ("动画：台标只在中段 4s（第 0 秒没有）", QA / "_rv_detect_mid4.mp4", None, None),
        ("动画：台标中段 12s（跨两个采样点）", QA / "_rv_detect_mid12.mp4", None, None),
        ("动画：台标中段 12s（挪前 5s，只盖一个点）", QA / "_rv_detect_mid12b.mp4", None, None),
        ("动画：片头 6s 白字台标", QA / "_rv_detect_white6.mp4", None, None),
    ]
    for name, path, truth, must in cases:
        started = time.perf_counter()
        result = server._detect_video(path, 0.0)
        seconds = time.perf_counter() - started
        regions = result["regions"]
        head = (f"{name:<34} 采样 {result['sampled_frames']} 帧 → {len(regions)} 框"
                f"，疑似 {len(result['suggestions'])} 条，{seconds:.2f}s")
        if truth:
            hits = sum(1 for want in truth if any(overlap(want, got) >= 0.5 for got in regions))
            extras = sum(1 for got in regions if not any(overlap(want, got) >= 0.5 for want in truth))
            print(f"{head}；命中 {hits}/{len(truth)}，多余 {extras}")
            if must is not None and must > 0:
                check(f"识别：{name}（至少 {must} 处）", hits >= must and extras == 0,
                      f"{hits}/{len(truth)} 命中，多余 {extras}")
        elif truth == []:
            print(f"{head}；{result['message'][:36]}")
            check(f"识别：{name} 不乱出框", len(regions) == 0, f"{len(regions)} 框")
        else:
            print(f"{head}\n     ↳ {result['message']}")
    print(f"对照：同一支「第 0 秒没有台标」的片子，只看第 0 秒会得 "
          f"{frame0_regions(QA / '_rv_detect_mid4.mp4')} 框（第十四轮之前的口径）")


# --------------------------------------------------------------------------
# 阶段 2：画质优先 vs 速度优先
# --------------------------------------------------------------------------

def phase_mode() -> None:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    width, height = gt["width"], gt["height"]
    marks = [("右下贴边白字", gt["corner_region"]), ("平铺彩章中央一块", gt["stamp_region"]),
             ("画面中央不贴边台标", gt["center_region"])]
    print("\n=== 真实实拍：画质优先 vs 速度优先 ===")
    quality_out, fast_out = QA / "_rv_quality.mp4", QA / "_rv_fast.mp4"
    boil_start = int(media.probe_video(MARKED).get("frame_count") or 100) // 2
    regions = [m[1] for m in marks]
    quality, quality_seconds = run_engine(MARKED, quality_out, regions, width, height, "quality")
    fast, fast_seconds = run_engine(MARKED, fast_out, regions, width, height, "fast")
    check("画质优先：贴边框走逐帧", quality["method"].startswith("frame"), str(quality["method"]))
    check("速度优先：走 delogo", fast["method"] == "delogo", str(fast["method"]))
    check("速度优先确实更快", fast_seconds < quality_seconds,
          f"{quality_seconds:.2f}s -> {fast_seconds:.2f}s（{quality_seconds / max(fast_seconds, 0.01):.1f}x）")

    print(f"{'区域':<20}{'残差 画质/速度':>18}{'水印剩影 ghost':>18}{'细节 sharp 带标→画质/速度':>26}   谁更干净")
    for name, region in marks:
        box = px_box(region, width, height)
        base = residual_in(MARKED, CLEAN, box, width, height)
        q = residual_in(quality_out, CLEAN, box, width, height)
        f = residual_in(fast_out, CLEAN, box, width, height)
        # 带标原片自己当 sharp 基准：它 >1 的倍数就是「水印本身算多少高频」，成品只有接近 1 才算既擦干净又没抹平
        mb = mark_metrics(MARKED, MARKED, CLEAN, box, width, height)
        mq = mark_metrics(quality_out, MARKED, CLEAN, box, width, height)
        mf = mark_metrics(fast_out, MARKED, CLEAN, box, width, height)
        winner = "画质" if mq["ghost"] < mf["ghost"] else "速度"
        print(f"{name:<20}{base:8.2f}→{q:5.2f}/{f:5.2f}{mq['ghost']:9.2f}/{mf['ghost']:6.2f}"
              f"{mb['sharp']:9.2f}→{mq['sharp']:5.2f}/{mf['sharp']:5.2f}   {winner}")
        boil = temporal_boil([("带标", MARKED), ("画质", quality_out), ("速度", fast_out)],
                             box, width, height, boil_start, 12)
        print(f"{'':<20}相邻帧差 带标 {boil['带标']:.2f} → 画质 {boil['画质']:.2f}"
              f"（{boil['画质'] / max(boil['带标'], 0.01):.2f}x）/ 速度 {boil['速度']:.2f}"
              f"（{boil['速度'] / max(boil['带标'], 0.01):.2f}x）")
        check(f"{name}：修复没引入可感知的时序沸腾",
              boil["画质"] <= boil["带标"] * 1.25 and boil["速度"] <= boil["带标"] * 1.25,
              f"最高 {max(boil['画质'], boil['速度']) / max(boil['带标'], 0.01):.2f}x")
        check(f"{name}：至少一档把水印对比去掉一半", min(mq["ghost"], mf["ghost"]) < 0.5,
              f"ghost 最好 {min(mq['ghost'], mf['ghost']):.2f}")

    for label, out in (("画质优先", quality_out), ("速度优先", fast_out)):
        d = outside_diff(out, MARKED, regions, width, height)
        frames_out = int(media.probe_video(out).get("frame_count") or 0)
        frames_in = int(media.probe_video(MARKED).get("frame_count") or 0)
        check(f"{label}：选框以外几乎没动", d < 4.0, f"最大单帧平均差 {d:.2f}")
        check(f"{label}：帧数与素材一致", abs(frames_out - frames_in) <= 2, f"{frames_in} -> {frames_out}")
    print(f"耗时：画质优先 {quality_seconds:.2f}s / 速度优先 {fast_seconds:.2f}s")


# --------------------------------------------------------------------------
# 阶段 3：静止水印掩膜（第十七轮画质优先新路线）+ 4K 原生角标补漏
# --------------------------------------------------------------------------

def phase_mask() -> None:
    build_mask_clips()
    gt = json.loads(MASK_GT_FILE.read_text(encoding="utf-8"))
    width, height = gt["width"], gt["height"]
    corner, plate = gt["corner"], gt["plate"]
    marks = [("右下贴边白字（笔画）", gt["corner_region"], corner),
             ("中央整块台标（实心）", gt["plate_region"], plate)]
    print("\n=== 1080p 真实拍：静止水印掩膜（画质优先新路线）===")
    quality_out, fast_out = QA / "_rv_jelly_quality.mp4", QA / "_rv_jelly_fast.mp4"
    regions = [gt["corner_region"], gt["plate_region"]]
    quality, quality_seconds = run_engine(JELLY_MARKED, quality_out, regions, width, height, "quality")
    fast, fast_seconds = run_engine(JELLY_MARKED, fast_out, regions, width, height, "fast")
    # 两个框在 1080p 上走两条不同的路：贴边白字能出掩膜（但覆盖过半，说明里如实写成「整片补画」），
    # 中央半透明台标整块都被盖住、出不了掩膜 —— 正是 §12 记的边界，折回整框修复。
    check("1080p：画质优先走逐帧路线（贴边白字出了掩膜）", quality["method"].startswith("frame"),
          str(quality["method"]))
    check("1080p：说明不谎称「只补笔画」（这处覆盖过半）",
          "整片补画" in quality["note"] and "只按水印笔画补像素" not in quality["note"], quality["note"])
    check("1080p：说明如实写出没出掩膜的那一处",
          "1 处是整块台标，仍整框修复" in quality["note"], quality["note"])
    check("1080p：速度优先仍是 delogo", fast["method"] == "delogo", str(fast["method"]))
    print(f"  画质说明：{quality['note']}")

    print(f"{'区域':<22}{'残差 带标→画质/速度':>24}{'ghost 画质/速度':>18}{'sharp 带标→画质/速度':>24}")
    for name, region, box in marks:
        base = residual_in(JELLY_MARKED, JELLY_CLEAN, box, width, height)
        q = residual_in(quality_out, JELLY_CLEAN, box, width, height)
        f = residual_in(fast_out, JELLY_CLEAN, box, width, height)
        mb = mark_metrics(JELLY_MARKED, JELLY_MARKED, JELLY_CLEAN, box, width, height)
        mq = mark_metrics(quality_out, JELLY_MARKED, JELLY_CLEAN, box, width, height)
        mf = mark_metrics(fast_out, JELLY_MARKED, JELLY_CLEAN, box, width, height)
        print(f"{name:<22}{base:8.2f}→{q:5.2f}/{f:5.2f}{mq['ghost']:9.2f}/{mf['ghost']:6.2f}"
              f"{mb['sharp']:9.2f}→{mq['sharp']:5.2f}/{mf['sharp']:5.2f}")
        if box is corner:
            # 画质优先曾经在这里输给 delogo（整框贴片 6.24 残差 + sharp 2.44 振铃）；
            # 掩膜只补笔画像素后应该反超，且不像贴片那样在笔画边上造高频。
            check("1080p 贴边白字：画质优先比速度优先更干净", q < f, f"画质 {q:.2f} / 速度 {f:.2f}")
            check("1080p 贴边白字：笔画修补没有造出振铃（sharp 应接近 1）",
                  mq["sharp"] < mb["sharp"] * 0.5 and mq["sharp"] < 1.5,
                  f"{mb['sharp']:.2f} -> {mq['sharp']:.2f}")
        else:
            check("1080p 中央整块台标：如实退回整框修复（这档仍略输 delogo）",
                  q < base and f < q, f"带标 {base:.2f} / 画质 {q:.2f} / 速度 {f:.2f}")
        boil = temporal_boil([("带标", JELLY_MARKED), ("画质", quality_out), ("速度", fast_out)],
                             box, width, height, 24, 12)
        print(f"{'':<22}相邻帧差 带标 {boil['带标']:.2f} → 画质 {boil['画质']:.2f}"
              f"（{boil['画质'] / max(boil['带标'], 0.01):.2f}x）/ 速度 {boil['速度']:.2f}"
              f"（{boil['速度'] / max(boil['带标'], 0.01):.2f}x）")
        # 角上那块水面本身几乎不动（带标 0.00），比值会除以 0；绝对量 <0.3 级肉眼不可见，按绝对量兜底。
        check(f"{name}：修复没引入可感知的时序沸腾",
              boil["画质"] <= max(boil["带标"] * 1.25, 0.30),
              f"画质 {boil['画质']:.2f}（带标 {boil['带标']:.2f}）")

    for label, out in (("画质优先", quality_out), ("速度优先", fast_out)):
        d = outside_diff(out, JELLY_MARKED, regions, width, height)
        frames_out = int(media.probe_video(out).get("frame_count") or 0)
        frames_in = int(media.probe_video(JELLY_MARKED).get("frame_count") or 0)
        check(f"1080p {label}：选框以外几乎没动", d < 4.0, f"最大单帧平均差 {d:.2f}")
        check(f"1080p {label}：帧数与素材一致", abs(frames_out - frames_in) <= 2, f"{frames_in} -> {frames_out}")
    print(f"耗时：画质优先 {quality_seconds:.2f}s / 速度优先 {fast_seconds:.2f}s"
          f"（{quality_seconds / max(fast_seconds, 0.01):.1f}x）")

    print("\n--- 静止背景（同一帧定格 6s）：掩膜在原理上不成立，必须退回原路线且不能变差 ---")
    static_out = QA / "_rv_static_out.mp4"
    static, static_seconds = run_engine(STATIC_MARKED, static_out, [gt["corner_region"]],
                                        width, height, "quality")
    sbase = residual_in(STATIC_MARKED, STATIC_CLEAN, corner, width, height)
    sout = residual_in(static_out, STATIC_CLEAN, corner, width, height)
    # 背景与台标一起静止时，框内各点的时序波动和框外一样（实测环带中位 0.0），掩膜分不出笔画，
    # 必须整体退回原路线（这处选框在画面内，画质优先的原有行为就是走 delogo）。
    check("静止背景：没有硬套掩膜，退回原有内插值路线", static["method"] == "delogo",
          str(static["method"]))
    check("静止背景：说明只说内插值修复，不提笔画修补",
          "按水印笔画" not in static["note"] and "整片补画" not in static["note"]
          and "内插值" in static["note"], static["note"])
    check("静止背景：退回原路线照样清得掉", sout < sbase * 0.5, f"{sbase:.2f} -> {sout:.2f}")
    print(f"  残差 带标 {sbase:.2f} → 成品 {sout:.2f}；耗时 {static_seconds:.2f}s")

    print("\n--- 4K（3840×2160）原生角标补漏 + 逐帧分流（48px 那处覆盖过半，如实写成整片补画）---")
    detect = server._detect_video(UHD_MARKED, 0.0)
    found = detect["regions"]
    print(f"  识别：{len(found)} 框 / 采样 {detect.get('sampled_frames')} 帧；{detect['message']}")
    for key, label in (("uhd_small", "24px 小台标"), ("uhd_big", "48px 大台标")):
        target = gt[f"{key}_region"]
        hit = max((overlap(target, item) for item in found), default=0.0)
        check(f"4K：{label}被找回且位置对得上", hit > 0.5, f"最大重叠 {hit:.2f}")
    check("4K：没有多出来的误框", len(found) == 2, f"{len(found)} 框")
    uhd_regions = [gt["uhd_small_region"], gt["uhd_big_region"]]
    uhd_out = QA / "_rv_uhd_out.mp4"
    uhd_result, uhd_seconds = run_engine(UHD_MARKED, uhd_out, uhd_regions,
                                         gt["uhd_width"], gt["uhd_height"], "quality")
    # 24px 那处框外环带中位波动只有 3.98（4K 上框很小、邻域本身就几乎不动）→ 分不出笔画，整框修；
    # 48px 那处掩膜占 62.7%（>50% 的「重覆盖」档）→ 走掩膜但说明不许写成「只补笔画」。
    check("4K：逐帧路线如实按「掩膜 / 整框」分流，不谎称全是掩膜",
          uhd_result["method"] == "frame-smart", str(uhd_result["method"]))
    check("4K：说明里既有「整片补画」也有「整块台标，仍整框修复」",
          "整片补画" in uhd_result["note"] and "1 处是整块台标，仍整框修复" in uhd_result["note"],
          uhd_result["note"])
    print(f"  说明：{uhd_result['note']}")
    for key, box in (("uhd_small", gt["uhd_small"]), ("uhd_big", gt["uhd_big"])):
        base = residual_in(UHD_MARKED, UHD_CLEAN, box, gt["uhd_width"], gt["uhd_height"], frames=4)
        out = residual_in(uhd_out, UHD_CLEAN, box, gt["uhd_width"], gt["uhd_height"], frames=4)
        check(f"4K {key}：修复后残差下降", out < base * 0.5, f"{base:.2f} -> {out:.2f}")
    print(f"  逐帧耗时 {uhd_seconds:.2f}s（{UHD_SECONDS:.0f}s 素材；只报告，不拿它当规格）")


# --------------------------------------------------------------------------
# 阶段 4：动态水印跟踪
# --------------------------------------------------------------------------

def phase_motion() -> None:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    width, height = gt["width"], gt["height"]
    move = gt["move"]
    first_box = {"x": move["x0"], "y": move["y0"], "w": move["w"], "h": move["h"]}
    first = region_from_px(first_box, width, height, grow=3)
    union = move_union(move)
    print("\n=== 真实实拍：移动 / 旋转 / 缩放台标 ===")
    fixed_out, track_out = QA / "_rv_fixed.mp4", QA / "_rv_track.mp4"
    fixed, fixed_seconds = run_engine(MOVE, fixed_out, [first], width, height, "quality")
    track, track_seconds = run_engine(MOVE, track_out, [first], width, height, "quality", motion=True)
    base = residual_in(MOVE, CLEAN, union, width, height)
    f = residual_in(fixed_out, CLEAN, union, width, height)
    t = residual_in(track_out, CLEAN, union, width, height)
    print(f"平移台标（6s 走 {move['dx'] * 6:.0f}px）整段残差：带标原片 {base:.2f} / 固定框 {f:.2f} / 跟踪 {t:.2f}"
          f"；耗时 {fixed_seconds:.2f}s -> {track_seconds:.2f}s（{track_seconds / max(fixed_seconds, 0.01):.2f}x）")
    check("平移：跟踪明显优于固定框", t < f * 0.75, f"{f:.2f} -> {t:.2f}")
    # 耗时只报告不断言：合成夹具当年量到 1.25x，实拍这条稳定在 5.8–6.6x（三轮实测），
    # 差别在画面尺寸与逐帧 inpaint 的量——写进 §8.1，不拿它当阈值卡自己。
    check("平移：跟踪后帧数与素材一致",
          int(media.probe_video(track_out).get("frame_count") or 0)
          == int(media.probe_video(MOVE).get("frame_count") or -1),
          str(media.probe_video(track_out).get("frame_count")))
    print(f"  note：{track['note']}")

    for kind, label in (("rot", "旋转"), ("scale", "缩放")):
        source = QA / f"_rv_{kind}.mp4"
        data = json.loads((QA / f"_rv_{kind}_gt.json").read_text(encoding="utf-8"))
        span, boxes = data["span"], data["boxes"]
        first_region = region_from_px(boxes[0], width, height, grow=3)
        out_fixed, out_track = QA / f"_rv_{kind}_fixed.mp4", QA / f"_rv_{kind}_track.mp4"
        fixed_result, _ = run_engine(source, out_fixed, [first_region], width, height, "quality")
        result, _ = run_engine(source, out_track, [first_region], width, height, "quality", motion=True)
        fixed_method = fixed_result["method"]
        union_region = region_from_px(span, width, height)
        box_px = px_box(union_region, width, height)
        base_value = residual_in(source, CLEAN, box_px, width, height)
        fixed_value = residual_in(out_fixed, CLEAN, box_px, width, height)
        track_value = residual_in(out_track, CLEAN, box_px, width, height)
        print(f"{label}台标（占 {span['w']:.0f}x{span['h']:.0f}px）整段残差：带标原片 {base_value:.2f} / "
              f"固定框 {fixed_value:.2f}（{fixed_method}）/ 跟踪 {track_value:.2f}")
        print(f"  note：{result['note']}")
        if kind == "rot":
            # 第十七轮起画质优先会先试「静止水印掩膜」：转动台标里没怎么变的那部分被补掉，
            # 一直在转的笔画留在框内，固定框残差从第十五轮整框贴片的 4.88 变成 11.53。
            # 时序统计分不开这两种情况（掩膜内波动：静止角标核心 11.9 / 旋转核心 11.2），
            # 所以不加会误伤常见角标的门槛，只把边界记下来：旋转/翻转台标开动态水印。
            check("旋转：开动态水印比固定框干净（边界已变，§12 第 11 条）", track_value < fixed_value,
                  f"固定框 {fixed_value:.2f} / 跟踪 {track_value:.2f}")
            check("旋转：跟丢帧数如实写进说明", "没跟住" in result["note"], result["note"])
            check("旋转：两种框都比不修干净", min(fixed_value, track_value) < base_value * 0.75,
                  f"带标 {base_value:.2f} → 最好 {min(fixed_value, track_value):.2f}")
        else:
            check(f"{label}：跟踪不比固定框差", track_value <= fixed_value * 1.15,
                  f"{fixed_value:.2f} -> {track_value:.2f}")


# --------------------------------------------------------------------------
# 阶段 4：真容器 / 真编解码的音轨字幕
# --------------------------------------------------------------------------

VARIANTS = [
    ("webm + opus", "_rv_webm_opus.webm", "reencode", ["-c:v", "libvpx", "-b:v", "900k", "-c:a", "libopus"]),
    ("mov + pcm_s16le", "_rv_mov_pcm.mov", "reencode", ["-c:v", "copy", "-c:a", "pcm_s16le"]),
    ("mp4 + alac", "_rv_mp4_alac.mp4", "copy", ["-c:v", "copy", "-c:a", "alac"]),
    ("mp4 + ac3", "_rv_mp4_ac3.mp4", "copy", ["-c:v", "copy", "-c:a", "ac3", "-b:a", "192k"]),
    ("mp4 + eac3", "_rv_mp4_eac3.mp4", "copy", ["-c:v", "copy", "-c:a", "eac3", "-b:a", "256k"]),
    ("mp4 + mp3", "_rv_mp4_mp3.mp4", "copy", ["-c:v", "copy", "-c:a", "libmp3lame"]),
    ("mkv + ass 文本字幕", "_rv_mkv_ass.mkv", "text", ["-c:s", "ass"]),
    ("可变帧率（mpdecimate）", "_rv_vfr.mp4", "copy",
     ["-vf", "mpdecimate", "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy"]),
]


def pcm_digest(path: Path, index: int = 0) -> tuple[str, int]:
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-map", f"0:a:{index}", "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
        capture_output=True)
    return hashlib.md5(result.stdout).hexdigest(), len(result.stdout)


def build_streams() -> None:
    (QA / "_rv_subs.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 960\nPlayResY: 400\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour\nStyle: Default,Arial,28,&H00FFFFFF\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,first line\n"
        "Dialogue: 0,0:00:04.00,0:00:05.50,Default,second line\n", encoding="utf-8")
    (QA / "_rv_subs.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nfirst line\n\n"
        "2\n00:00:04,000 --> 00:00:05,500\nsecond line\n", encoding="utf-8")
    for label, name, _kind, args in VARIANTS:
        out = QA / name
        try:
            if "ass" in args:
                run_ffmpeg(["-i", str(CLEAN), "-i", str(QA / "_rv_subs.ass"),
                            "-map", "0", "-map", "1", "-c:v", "copy", "-c:a", "copy", "-c:s", "ass", str(out)])
            else:
                run_ffmpeg(["-i", str(CLEAN), *args, str(out)])
        except RuntimeError as error:
            check(f"造夹具：{label}", False, str(error)[:160])
    # `-c copy` 不带 `-map 0` 时 ffmpeg 每种类型只挑一条，bbb 的第二条 AAC 会被静默丢掉（实测踩过）
    run_ffmpeg(["-i", str(BBB), "-map", "0", "-c", "copy", str(QA / "_rv_multi.mp4")])


def phase_streams() -> None:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    width, height = gt["width"], gt["height"]
    center = [gt["center_region"]]
    print("\n=== 真容器 / 真编解码：音轨与字幕 ===")
    build_streams()
    cases = [(label, QA / name, kind, width, height, center) for label, name, kind, _ in VARIANTS]
    cases.append(("bbb 两条真实 AAC 音轨", QA / "_rv_multi.mp4", "copy", 320, 176,
                  [{"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.3}]))
    cases.append(("mkv + 长音轨（30s 音配 8s 画面）", QA / "_rv_longaudio.mkv", "long", width, height, center))
    if not (QA / "_rv_longaudio.mkv").exists():
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=30",
                    "-i", str(CLEAN), "-map", "1:v", "-map", "0:a", "-c:v", "copy",
                    "-c:a", "aac", str(QA / "_rv_longaudio.mkv")])
    for label, source, kind, w, h, regions in cases:
        if not source.exists():
            continue
        before = media.probe_streams(source)
        out = QA / f"_rv_out_{source.stem}.mp4"
        try:
            result, seconds = run_engine(source, out, regions, w, h, "fast")
        except Exception as error:  # noqa: BLE001
            check(f"处理：{label}", False, str(error)[:160])
            continue
        after = media.probe_streams(out)
        in_audio = [item["codec"] for item in before.get("audio") or []]
        out_audio = [item["codec"] for item in after.get("audio") or []]
        in_subs = [item["codec"] for item in before.get("subtitles") or []]
        out_subs = [item["codec"] for item in after.get("subtitles") or []]
        print(f"\n{label}\n  音轨 {in_audio} / 字幕 {in_subs} → 音轨 {out_audio} / 字幕 {out_subs}（{seconds:.2f}s）")
        print(f"  note：{result['note']}")
        check(f"{label}：音轨一条不丢", len(out_audio) == len(in_audio) and in_audio,
              f"{len(in_audio)} → {len(out_audio)}")
        if kind in ("copy", "multi"):
            same = all(pcm_digest(source, i)[0] == pcm_digest(out, i)[0] for i in range(len(in_audio)))
            check(f"{label}：原音轨逐样本一致", same,
                  " ".join(f"轨{i}:{pcm_digest(source, i)[1]}→{pcm_digest(out, i)[1]}B" for i in range(len(in_audio))))
        if kind == "reencode":
            check(f"{label}：如实写明重编码为 AAC", "重编码为 AAC" in result["note"]
                  and out_audio == ["aac"] * len(in_audio), str(out_audio))
        if kind == "text":
            check(f"{label}：文本字幕转成 mov_text", out_subs == ["mov_text"], str(out_subs))
        if kind == "graphic":
            check(f"{label}：图形字幕略过并写明条数",
                  all(codec == "mov_text" for codec in out_subs) and "图形字幕" in result["note"],
                  f"输出字幕 {out_subs}")
        info_in, info_out = media.probe_video(source), media.probe_video(out)
        # 长音轨素材的容器估算帧数不可信（30s 音轨把 8s 画面估成 720 帧），按画面轨自己的帧数核
        frames_in = media.probe_video_length(source).get("frames") or int(info_in.get("frame_count") or 0)
        frames_out = int(info_out.get("frame_count") or 0)
        dur_in, dur_out = float(info_in.get("duration") or 0), float(info_out.get("duration") or 0)
        check(f"{label}：画面没被截尾", frames_out >= frames_in - 3,
              f"{frames_in}→{frames_out} 帧，{dur_in:.2f}→{dur_out:.2f}s")
        if kind == "long":
            check(f"{label}：长音轨被对齐到画面长度", dur_out <= dur_in + 0.5, f"{dur_out:.2f}s")

    # 图形字幕：本机造不出真位图字幕轨（ffmpeg 7.1 限制「文本→文本、位图→位图」，
    # `-c:s dvdsub/dvbsub` 从 srt/ass 直接失败；把视频过滤流映射成字幕轨会静默不写轨道），
    # FATE 样本库也不给目录列表、下不到现成文件。所以这条只能按 ffmpeg **真实回报的 codec 名**
    # 做规划层断言，不能宣称跑过真片子——§12 第 15 条据此保留。
    for codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
        plan = video_ops._plan_streams({
            "audio": [{"index": 1, "codec": "aac"}],
            "subtitles": [{"index": 2, "codec": codec}, {"index": 3, "codec": "subrip"}]})
        note = video_ops._streams_note(plan)
        check(f"图形字幕 {codec}：不进搬运清单且如实写明条数",
              plan["subtitle_indices"] == [3] and plan["skipped_subtitles"] == 1
              and "1 条图形字幕无法转成文本，已略过" in note, note)


# --------------------------------------------------------------------------
# 阶段 5：真 4K 素材（第二十四轮）—— 原生 3840×2160，不是 1080p 上采样
# --------------------------------------------------------------------------

BBB4K = QA / "_bbb4k_raw.mp4"
UHD2_CLEAN = QA / "_rv_bbb4k_clean.mp4"
UHD2_MARKED = QA / "_rv_bbb4k_marked.mp4"
UHD2_GT_FILE = QA / "_rv_bbb4k_gt.json"
UHD2_SECONDS = 3.0


def build_uhd_clips() -> dict:
    """真 4K 底片上画两个角标：左上 24px、右下 48px。

    第十七轮的 4K 夹具是 1080p 用 lanczos 上采样到 3840×2160（分辨率真、细节不真），
    §12 第 1 条据此一直记着「真 4K 带标素材没有正例」。这一份是原生 4K 帧。
    """
    if not BBB4K.exists():
        raise RuntimeError(f"缺少 {BBB4K.name}：见脚本头注释里的下载命令")
    info = media.probe_video(BBB4K)
    width, height = int(info["width"]), int(info["height"])
    small_w, small_h = render_badge(QA / "_rv_bbb4k_small.png", "AI生成", 24, alpha=215)
    big_w, big_h = render_badge(QA / "_rv_bbb4k_big.png", "AI生成", 48, alpha=215)
    small = {"x": 40, "y": 32, "w": small_w, "h": small_h}
    big = {"x": width - big_w - 40, "y": height - big_h - 40, "w": big_w, "h": big_h}
    overlay_clip(BBB4K, UHD2_CLEAN, [], [], UHD2_SECONDS, with_audio=False)
    overlay_clip(BBB4K, UHD2_MARKED,
                 [str(QA / "_rv_bbb4k_small.png"), str(QA / "_rv_bbb4k_big.png")],
                 [f"x={small['x']}:y={small['y']}", f"x={big['x']}:y={big['y']}"],
                 UHD2_SECONDS, with_audio=False)
    gt = {
        "width": width, "height": height, "fps": info.get("fps"),
        "frames": int(info.get("frame_count") or 0),
        "small": small, "big": big,
        "small_region": region_from_px(small, width, height, grow=3),
        "big_region": region_from_px(big, width, height, grow=3),
    }
    UHD2_GT_FILE.write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"原生 4K 底片 {width}x{height} {info.get('fps')}fps {gt['frames']} 帧；"
          f"左上角标 {small_w}x{small_h}、右下角标 {big_w}x{big_h}")
    return gt


def phase_uhd() -> None:
    print("\n=== 原生 4K：自动识别 + 两档修复（不是 1080p 上采样）===")
    gt = build_uhd_clips()
    width, height = gt["width"], gt["height"]
    truth = [gt["small_region"], gt["big_region"]]

    started = time.perf_counter()
    detect = server._detect_video(UHD2_MARKED, 0.0)
    detect_seconds = time.perf_counter() - started
    found = detect["regions"]
    hits = [max((overlap(want, got) for got in found), default=0.0) for want in truth]
    extras = sum(1 for got in found if not any(overlap(want, got) >= 0.5 for want in truth))
    print(f"  识别：采样 {detect.get('sampled_frames')} 帧 → {len(found)} 框"
          f"（命中重叠 {hits[0]:.2f} / {hits[1]:.2f}，多余 {extras}），{detect_seconds:.2f}s")
    print(f"       {detect['message']}")
    check("4K：24px 与 48px 两个角标都被找回", min(hits) > 0.5, f"{hits[0]:.2f} / {hits[1]:.2f}")
    check("4K：没有多出来的误框", extras == 0, f"{len(found)} 框，多余 {extras}")

    # 对照：只走「降采样到 1280 再检测」这一路（第十六轮之前的口径）会漏掉什么。
    frame = media.extract_video_frame(UHD2_MARKED, 0.0)
    scale = server.VIDEO_DETECT_MAX_SIDE / max(frame.shape[1], frame.shape[0])
    small_frame = np.array(Image.fromarray(frame).resize(
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))), Image.LANCZOS))
    only_1280 = detect_watermark(small_frame)["regions"]
    print(f"  对照（只走 1280 口径）：{len(only_1280)} 框 → "
          + "；".join(f"{r['x']:.3f},{r['y']:.3f} {r['w']:.3f}x{r['h']:.3f}" for r in only_1280))
    missed = [i for i, want in enumerate(truth)
              if not any(overlap(want, got) >= 0.5 for got in only_1280)]
    check("4K：只走 1280 口径会漏掉右下那块 48px 角标", 1 in missed, f"漏掉 {missed}")

    regions = truth
    quality_out, fast_out = QA / "_rv_bbb4k_quality.mp4", QA / "_rv_bbb4k_fast.mp4"
    quality, quality_seconds = run_engine(UHD2_MARKED, quality_out, regions, width, height, "quality")
    fast, fast_seconds = run_engine(UHD2_MARKED, fast_out, regions, width, height, "fast")
    check("4K：画质优先走逐帧路线", quality["method"].startswith("frame"), str(quality["method"]))
    check("4K：速度优先走 delogo", fast["method"] == "delogo", str(fast["method"]))
    print(f"  画质说明：{quality['note']}")
    print(f"{'区域':<16}{'残差 带标→画质/速度':>24}{'ghost 画质/速度':>18}{'sharp 带标→画质/速度':>24}")
    for label, key in (("左上 24px", "small"), ("右下 48px", "big")):
        box = gt[key]
        base = residual_in(UHD2_MARKED, UHD2_CLEAN, box, width, height, frames=4)
        q = residual_in(quality_out, UHD2_CLEAN, box, width, height, frames=4)
        f = residual_in(fast_out, UHD2_CLEAN, box, width, height, frames=4)
        mb = mark_metrics(UHD2_MARKED, UHD2_MARKED, UHD2_CLEAN, box, width, height)
        mq = mark_metrics(quality_out, UHD2_MARKED, UHD2_CLEAN, box, width, height)
        mf = mark_metrics(fast_out, UHD2_MARKED, UHD2_CLEAN, box, width, height)
        print(f"{label:<16}{base:8.2f}→{q:5.2f}/{f:5.2f}{mq['ghost']:9.2f}/{mf['ghost']:6.2f}"
              f"{mb['sharp']:9.2f}→{mq['sharp']:5.2f}/{mf['sharp']:5.2f}")
        check(f"4K {label}：画质优先把残差砍到一半以下", q < base * 0.5, f"{base:.2f} -> {q:.2f}")
        check(f"4K {label}：速度优先把残差砍到一半以下", f < base * 0.5, f"{base:.2f} -> {f:.2f}")

    for label, out in (("画质优先", quality_out), ("速度优先", fast_out)):
        d = outside_diff(out, UHD2_MARKED, regions, width, height)
        frames_out = int(media.probe_video(out).get("frame_count") or 0)
        check(f"4K {label}：选框以外几乎没动", d < 4.0, f"最大单帧平均差 {d:.2f}")
        check(f"4K {label}：帧数与素材一致", abs(frames_out - gt["frames"]) <= 2,
              f"{gt['frames']} -> {frames_out}")
    print(f"  耗时：画质优先 {quality_seconds:.2f}s / 速度优先 {fast_seconds:.2f}s"
          f"（{UHD2_SECONDS:.0f}s 素材；负载主导，只报告不拿它当规格）")


# --------------------------------------------------------------------------
# 阶段 6：图形字幕真容器（第二十四轮）—— PGS / DVD 子图 / DVB 子图
# --------------------------------------------------------------------------

SUB_CASES = [
    # (标签, 文件名, 期望的图形字幕 codec, 音轨处理, 是否另有文本字幕)
    ("PGS 720×480（无音轨）", "_gs_pgs_small.mkv", "hdmv_pgs_subtitle", "none", False),
    ("PGS 1920×1080（无音轨）", "_gs_pgs_dragon.mkv", "hdmv_pgs_subtitle", "none", False),
    ("DVD 子图 + ac3", "_gs_dvd.mkv", "dvd_subtitle", "copy", False),
    ("DVB 子图 + ac3", "_gs_dvb.mkv", "dvb_subtitle", "copy", False),
    ("同一容器里图形 + 文本字幕", "_rv_mixed_subs.mkv", "dvd_subtitle", "copy", True),
]
SUB_REGION = [{"x": 0.40, "y": 0.42, "w": 0.22, "h": 0.16}]


def build_mixed_subs() -> None:
    """把 srt 挂到 DVD 子图那个 mkv 上：同一容器里既有图形字幕又有文本字幕。"""
    srt = QA / "_rv_pic_subs.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n第一行测试字幕\n\n"
                   "2\n00:00:04,000 --> 00:00:07,000\n第二行测试字幕\n", encoding="utf-8")
    run_ffmpeg(["-fflags", "+genpts", "-i", str(QA / "_gs_dvd.mkv"), "-i", str(srt),
                "-map", "0", "-map", "1", "-c", "copy", str(QA / "_rv_mixed_subs.mkv")])


def phase_subpic() -> None:
    print("\n=== 图形字幕真容器：PGS / DVD 子图 / DVB 子图 ===")
    if not (QA / "_gs_dvd.mkv").exists():
        raise RuntimeError("缺少 _gs_* 图形字幕样片：见脚本头注释里的来源")
    build_mixed_subs()
    for label, name, codec, audio, has_text in SUB_CASES:
        source = QA / name
        if not source.exists():
            continue
        info = media.probe_video(source)
        width, height = int(info["width"]), int(info["height"])
        before = media.probe_streams(source)
        in_subs = [item["codec"] for item in before.get("subtitles") or []]
        in_audio = [item["codec"] for item in before.get("audio") or []]
        plan = video_ops._plan_streams(before)
        print(f"\n{label}：{width}x{height}，音轨 {in_audio} / 字幕 {in_subs}")
        print(f"  画面轨 {media.probe_video_length(source)}；规划 {plan}")
        check(f"{label}：ffmpeg 真的报出 {codec}", codec in in_subs, str(in_subs))
        check(f"{label}：图形字幕不进搬运清单", plan["skipped_subtitles"] == 1
              and len(plan["subtitle_indices"]) == (1 if has_text else 0), str(plan))

        out = QA / f"_rv_pic_{source.stem}.mp4"
        try:
            result, seconds = run_engine(source, out, SUB_REGION, width, height, "fast")
        except Exception as error:  # noqa: BLE001
            check(f"{label}：能出成品", False, str(error)[:160])
            continue
        after = media.probe_streams(out)
        out_subs = [item["codec"] for item in after.get("subtitles") or []]
        out_audio = [item["codec"] for item in after.get("audio") or []]
        print(f"  输出：音轨 {out_audio} / 字幕 {out_subs}（{seconds:.2f}s）\n  note：{result['note']}")
        # 成品封装固定 MP4：图形字幕无法转文本 → 一条都不该写进成品；文本字幕转 mov_text 保留。
        check(f"{label}：成品里没有图形字幕轨道",
              all(item != codec for item in out_subs) and len(out_subs) == (1 if has_text else 0),
              str(out_subs))
        check(f"{label}：说明如实写「1 条图形字幕无法转成文本，已略过」",
              "1 条图形字幕无法转成文本，已略过" in result["note"], result["note"])
        check(f"{label}：说明{'保留' if has_text else '不提'}文本字幕",
              ("字幕已保留" in result["note"]) == has_text, result["note"])
        if audio == "none":
            check(f"{label}：无音轨素材不多出音轨", out_audio == [] and "音轨" not in result["note"],
                  f"{out_audio} / {result['note']}")
        else:
            same = all(pcm_digest(source, i)[0] == pcm_digest(out, i)[0]
                       for i in range(len(in_audio)))
            check(f"{label}：原音轨逐样本一致且写明保留",
                  same and len(out_audio) == len(in_audio) and "原音轨已保留" in result["note"],
                  " ".join(f"轨{i}:{pcm_digest(source, i)[1]}→{pcm_digest(out, i)[1]}B"
                           for i in range(len(in_audio))))
        frames_in = media.probe_video_length(source).get("frames") or int(info.get("frame_count") or 0)
        frames_out = int(media.probe_video(out).get("frame_count") or 0)
        check(f"{label}：画面没被截尾", frames_out >= frames_in - 3, f"{frames_in}→{frames_out} 帧")

    # `.ts` 不在上传白名单（media.VIDEO_EXTENSIONS），所以三节目 TS 的容器口径差异属于产品范围外，
    # 只把实测数字留档：容器报的比画面轨长（多节目 + 各节目时间基不同），别拿它当回归判据。
    ts = QA / "_gs_dvb.ts"
    if ts.exists():
        print(f"\n边界（产品不接 .ts）：_gs_dvb.ts 容器口径 {media.probe_video(ts)}，"
              f"画面轨 {media.probe_video_length(ts)}")
        check("边界：.ts / .vob 不在上传白名单",
              media.media_kind(".ts") is None and media.media_kind(".vob") is None,
              f"media_kind('.ts')={media.media_kind('.ts')}")


# --------------------------------------------------------------------------
# 阶段 7：端到端 HTTP（真视频走一遍用户流程）
# --------------------------------------------------------------------------

def phase_http() -> None:
    print("\n=== 端到端 HTTP：真视频走一遍用户流程 ===")
    # 端口是扫描出来的（8765–8785 取第一个可用的），写死 8765 换台机器/换端口就必然连不上，
    # 所以从 work/runtime.json 读。
    base = f"http://127.0.0.1:{json.loads((QA.parent / 'runtime.json').read_text(encoding='utf-8'))['port']}"
    page = urllib.request.urlopen(base + "/", timeout=30).read()
    token = page.decode("utf-8").split('name="app-token" content="')[1].split('"')[0]
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def get_json(url: str) -> dict:
        return json.loads(opener.open(url, timeout=60).read())

    def post_json(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-App-Token": token})
        return json.loads(opener.open(request, timeout=300).read())

    boundary = "----rvHttp"
    payload = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
               f"filename=\"{MARKED.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode(), MARKED.read_bytes(), b"\r\n"]
    payload.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        base + "/api/upload", data=b"".join(payload),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": token})
    uploaded = json.loads(opener.open(request, timeout=600).read())
    items = uploaded.get("items") or []
    check("端到端：真视频能上传", bool(items), str(uploaded.get("errors") or ""))
    if not items:
        return
    item = items[0]
    detect = post_json("/api/detect", {"item_id": item["id"], "time": 0})
    print(f"  识别：{len(detect['regions'])} 框 / 疑似 {len(detect['suggestions'])} 条 / 采样 {detect.get('sampled_frames')} 帧")
    print(f"       {detect['message']}")
    job = post_json("/api/process", {"items": [{"id": item["id"], "regions": detect["regions"]}],
                                     "video_mode": "quality", "method": "smart"})
    for _ in range(300):
        state = get_json(f"{base}/api/jobs/{job['id']}")
        if state["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    print(f"  任务：{state['status']} 成功 {state['success_count']} 失败 {state['failed_count']}")
    for result in state["results"]:
        print(f"       成品 {result['name']}（{result['size']}B）说明：{result['note']}")
    check("端到端：识别出框", len(detect["regions"]) > 0, str(len(detect["regions"])))
    check("端到端：出成品", state["success_count"] == 1, state["status"])
    output_dir = Path(state["output_dir"])
    print(f"  清理本次回归批次：{output_dir}")
    shutil.rmtree(output_dir, ignore_errors=True)
    post_json("/api/clear", {})


# --------------------------------------------------------------------------
# 阶段 8：HDR 源（真 4K HEVC Main10 / bt2020 / smpte2084）—— 会不会褪色/偏灰
# --------------------------------------------------------------------------

# 本机 ffmpeg 7.1 没有 zscale / tonemap / hdrmix（第二十八轮实测），所以「正确的 HDR→SDR」参照系
# 在 numpy 里自己算：PQ EOTF → 线性 nit → BT.2020 转 BT.709 → 压到 SDR 白 → sRGB OETF。
HDR_PQ_M1 = 2610.0 / 16384.0
HDR_PQ_M2 = 2523.0 / 4096.0 * 128.0
HDR_PQ_C1 = 3424.0 / 4096.0
HDR_PQ_C2 = 2413.0 / 4096.0 * 32.0
HDR_PQ_C3 = 2392.0 / 4096.0 * 32.0
HDR_SDR_WHITE = 203.0
HDR_BT2020_TO_709 = np.array([
    [1.660491, -0.587641, -0.072850],
    [-0.124363, 1.132915, -0.008352],
    [-0.018154, -0.100597, 1.118751],
], dtype=np.float64)


def _hdr_stream_lines(path: Path) -> list[str]:
    """流清单只在 stderr 里，且 `-i` 不带输出文件必然非零退出——所以不能走 run_ffmpeg。"""
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-v", "info", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    return [line.strip() for line in result.stderr.splitlines() if "Stream #" in line]


def _hdr_decode_10(path: Path, at: float, pix_fmt: str, scratch: Path,
                   width: int, height: int) -> np.ndarray:
    run_ffmpeg(["-ss", f"{at:.3f}", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                "-pix_fmt", pix_fmt, "-f", "rawvideo", str(scratch)])
    data = np.fromfile(str(scratch), dtype="<u2")
    scratch.unlink(missing_ok=True)
    if data.size != 3 * width * height:
        raise RuntimeError(f"解码 {pix_fmt} 得到 {data.size} 个样本，应为 {3 * width * height}")
    return data.reshape(3, height, width)


def _hdr_decode_rgb24(path: Path, at: float, scratch: Path,
                      width: int, height: int) -> np.ndarray:
    run_ffmpeg(["-ss", f"{at:.3f}", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                "-pix_fmt", "rgb24", "-f", "rawvideo", str(scratch)])
    data = np.fromfile(str(scratch), dtype=np.uint8)
    scratch.unlink(missing_ok=True)
    if data.size != 3 * width * height:
        raise RuntimeError(f"解码 rgb24 得到 {data.size} 字节，应为 {3 * width * height}")
    return data.reshape(height, width, 3)


def _hdr_yuv_to_pq_rgb(planes: np.ndarray) -> np.ndarray:
    """10bit limited range 的 YUV 4:4:4 转成 0..1 的 PQ 值，**不做任何色调映射**。

    色度除数取 896 而不是 448：第二十八轮实测 876/448 与 ffmpeg 差 48.94 个码值，
    876/896 + bt2020nc 只差 0.87（最大 3）。
    """
    luma = (planes[0].astype(np.float32) - 64.0) / 876.0
    cb = (planes[1].astype(np.float32) - 512.0) / 896.0
    cr = (planes[2].astype(np.float32) - 512.0) / 896.0
    rgb = np.empty((planes.shape[1], planes.shape[2], 3), dtype=np.float32)
    rgb[..., 0] = luma + 1.47460 * cr
    rgb[..., 1] = luma - 0.164553 * cb - 0.571353 * cr
    rgb[..., 2] = luma + 1.88140 * cb
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def _hdr_gbrp_to_rgb(planes: np.ndarray) -> np.ndarray:
    """`gbrp10le` 的平面顺序是 G、B、R——读成 R、G、B 会差 113/575 个码值。"""
    rgb = np.stack([planes[2], planes[0], planes[1]], axis=-1).astype(np.float32)
    rgb /= 1023.0
    return rgb


def _hdr_pq_eotf(value):
    """PQ 码值 → **相对**亮度（1.0 等于 10000 nit，调用方按需 ×10000）。"""
    powered = np.power(np.clip(value, 0.0, 1.0), 1.0 / HDR_PQ_M2)
    return np.power(np.maximum(powered - HDR_PQ_C1, 0.0) / (HDR_PQ_C2 - HDR_PQ_C3 * powered),
                    1.0 / HDR_PQ_M1)


def _hdr_pq_oetf(nits):
    """**绝对**亮度（nit）→ PQ 码值。"""
    value = np.clip(np.asarray(nits) / 10000.0, 0.0, 1.0)
    powered = np.power(value, HDR_PQ_M1)
    return np.power((HDR_PQ_C1 + HDR_PQ_C2 * powered) / (1.0 + HDR_PQ_C3 * powered), HDR_PQ_M2)


def _hdr_srgb_oetf(linear):
    value = np.clip(linear, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92,
                    1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def _hdr_tone_map(pq_rgb: np.ndarray, white_nits: float = HDR_SDR_WHITE,
                  mode: str = "reinhard") -> np.ndarray:
    """PQ 值当 sRGB 用（产品口径）→ 正确的 HDR→SDR，返回 0..1 的 sRGB。"""
    linear = _hdr_pq_eotf(pq_rgb)
    linear *= 10000.0
    rgb709 = linear @ HDR_BT2020_TO_709.astype(np.float32).T
    del linear
    np.maximum(rgb709, 0.0, out=rgb709)
    rgb709 /= white_nits
    if mode == "clip":
        mapped = np.clip(rgb709, 0.0, 1.0)
    else:
        peak = max(float(np.percentile(rgb709, 99.99)), 1.0)
        mapped = rgb709 * (1.0 + rgb709 / peak) / (1.0 + rgb709)
    del rgb709
    return _hdr_srgb_oetf(mapped)


def _hdr_stats(rgb_u8: np.ndarray) -> dict:
    rgb = rgb_u8.astype(np.float32)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "luma_mean": float(luma.mean()),
        "luma_p50": float(np.percentile(luma, 50)),
        "luma_p90": float(np.percentile(luma, 90)),
        "luma_max": float(luma.max()),
        "chroma_mean": float(chroma.mean()),
        "clip_high": float((rgb_u8 == 255).any(axis=2).mean()),
        "clip_low": float((rgb_u8 == 0).all(axis=2).mean()),
    }


def _hdr_compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = actual.astype(np.int16) - expected.astype(np.int16)
    signed = [float(diff[..., channel].mean()) for channel in range(3)]
    absolute = np.abs(diff)
    p95 = float(np.percentile(absolute, 95))
    mad = float(absolute.mean())
    return {"mad": mad, "signed": signed, "p95_abs": p95}


def phase_hdr() -> None:
    """真 4K HEVC Main10 HDR 源：产品口径（PQ 当 sRGB）离正确映射多远，两条路线的色彩标记还在不在。"""
    print("\n=== HDR 源：产品口径 vs 正确的 HDR→SDR（真 4K HEVC Main10）===")
    source = QA / "_real_leopard4k.mkv"
    if not source.exists():
        print("未验：缺少语料 _real_leopard4k.mkv（真 4K HEVC Main10 HDR 样片），本相没有跑")
        return

    info = media.probe_video(source)
    width, height = int(info["width"]), int(info["height"])
    length = media.probe_video_length(source)
    duration = float(length.get("duration") or 1.0)
    frames_in = int(length.get("frames") or 0)
    times = [round(duration * share, 3) for share in (0.1, 0.5, 0.9)]
    middle = times[1]
    streams = media.probe_streams(source)
    regions = [{"x": 0.42, "y": 0.40, "w": 0.12, "h": 0.06}]
    print(f"  语料 {source.name}：{width}×{height}，画面轨 {frames_in} 帧 / {duration:.3f}s，"
          f"采样 {times}")

    nit_low = float(_hdr_pq_eotf(0.0)) * 10000.0
    nit_high = float(_hdr_pq_eotf(1.0)) * 10000.0
    check("HDR A1：PQ 端点 0→0nit、1→10000nit",
          abs(nit_low) <= 0.15 and abs(nit_high - 10000.0) <= 200.0,
          f"{nit_low:.2f} / {nit_high:.2f} nit")

    worst = 0.0
    for nit in (0.0001, 0.01, 1.0, 12.46, 100.0, 400.0, 1000.0, 4000.0):
        back = float(_hdr_pq_eotf(float(_hdr_pq_oetf(nit)))) * 10000.0
        worst = max(worst, abs(back - nit) / nit)
    check("HDR A2：OETF↔EOTF 互逆（8 个 nit 值）", worst < 1e-6, f"最大相对误差 {worst:.2e}")

    row_gaps = [float(abs(float(row.sum()) - 1.0)) for row in HDR_BT2020_TO_709]
    check("HDR A3：BT.2020→BT.709 矩阵行和为 1（中性灰不变色）",
          max(row_gaps) < 2e-3, f"行和偏差 {[round(gap, 6) for gap in row_gaps]}")

    raw = QA / "_rv_hdr_raw.yuv"
    gbr = QA / "_rv_hdr_gbr.yuv"
    planes = _hdr_decode_10(source, middle, "yuv444p10le", raw, width, height)
    mine = _hdr_yuv_to_pq_rgb(planes)
    del planes
    planes = _hdr_decode_10(source, middle, "gbrp10le", gbr, width, height)
    theirs = _hdr_gbrp_to_rgb(planes)
    del planes
    diff = mine - theirs
    del mine, theirs
    np.abs(diff, out=diff)
    diff *= 1023.0
    code_gap = float(diff.mean())
    del diff
    check("HDR A4：自算 YUV→RGB 与 ffmpeg gbrp10le 解码一致（色度重采样不可能逐比特相同）",
          code_gap <= 6.0, f"平均差 {code_gap:.2f} 个 10bit 码值")

    audio_codecs = [item["codec"] for item in streams["audio"]]
    sub_codecs = [item["codec"] for item in streams["subtitles"]]
    check("HDR B1：认出 DTS 音轨与 PGS 图形字幕",
          any(codec.startswith("dts") for codec in audio_codecs)
          and "hdmv_pgs_subtitle" in sub_codecs,
          f"音轨 {audio_codecs} 字幕 {sub_codecs}")
    plan = video_ops._plan_streams(streams)
    plan_note = video_ops._streams_note(plan)
    check("HDR B2：流规划如实（DTS 重编码为 AAC / 图形字幕略过 / 按画面长度截断）",
          not plan["copy_audio"] and plan["skipped_subtitles"] == 1 and bool(plan["trim"])
          and "重编码为 AAC" in plan_note and "图形字幕无法转成文本" in plan_note,
          plan_note or "（无追加说明）")

    outputs: dict[str, Path] = {}
    for mode in (options.QUALITY, options.FAST):
        out = QA / f"_rv_hdr_{mode}.mp4"
        try:
            result, elapsed = run_engine(source, out, regions, width, height,
                                         mode=mode, method=options.SMART)
        except Exception as error:  # noqa: BLE001 - 这一相要记失败，不能整脚本崩
            check(f"HDR C：{mode} 档能在真 HDR 源上出成品",
                  False, f"{type(error).__name__}: {error}")
            continue
        out_frames = int(media.probe_video_length(out).get("frames") or 0)
        print(f"  {mode}：method={result.get('method')} 耗时 {elapsed:.2f}s")
        print(f"       说明：{result.get('note')}")
        check(f"HDR C：{mode} 档能在真 HDR 源上出成品（帧数 {frames_in}→{out_frames}）",
              out.exists() and out_frames >= frames_in - 3, f"{out_frames} 帧")
        outputs[mode] = out

    fast_tags = " | ".join(_hdr_stream_lines(outputs[options.FAST])) if options.FAST in outputs else ""
    quality_tags = (" | ".join(_hdr_stream_lines(outputs[options.QUALITY]))
                    if options.QUALITY in outputs else "")
    check("HDR C3：速度优先单趟 ffmpeg 保留 HDR 色彩标记（bt2020nc / smpte2084）",
          options.FAST in outputs and "bt2020nc" in fast_tags and "smpte2084" in fast_tags,
          fast_tags or "（无流信息）")
    check("HDR C4：画质优先逐帧重编码丢了 HDR 色彩标记（如实的能力边界，不是待修的 bug）",
          options.QUALITY in outputs and "smpte2084" not in quality_tags,
          quality_tags or "（无流信息）")

    naive_mads: list[float] = []
    mapped_mads: list[float] = []
    dimmer: list[bool] = []
    darker: list[bool] = []
    luma_pairs: list[str] = []
    kept_product = None
    kept_mapped = None
    kept_signed: list[float] = []
    for at in times:
        planes = _hdr_decode_10(source, at, "yuv444p10le", raw, width, height)
        pq_rgb = _hdr_yuv_to_pq_rgb(planes)
        del planes
        naive_u8 = np.round(pq_rgb * 255.0).astype(np.uint8)
        mapped_u8 = np.round(_hdr_tone_map(pq_rgb) * 255.0).astype(np.uint8)
        del pq_rgb
        product = media.extract_video_frame(source, at)
        stats_product = _hdr_stats(product)
        stats_mapped = _hdr_stats(mapped_u8)
        gap_naive = _hdr_compare(naive_u8, product)
        gap_mapped = _hdr_compare(mapped_u8, product)
        naive_mads.append(gap_naive["mad"])
        mapped_mads.append(gap_mapped["mad"])
        dimmer.append(stats_product["luma_p50"] - stats_mapped["luma_p50"] >= 10.0)
        darker.append(sum(gap_mapped["signed"]) < 0.0)
        luma_pairs.append(f"{stats_product['luma_p50']:.0f}→{stats_mapped['luma_p50']:.0f}")
        print(f"  采样 {at:.3f}s：产品口径中位亮度 {stats_product['luma_p50']:.1f}，"
              f"正确映射 {stats_mapped['luma_p50']:.1f}；"
              f"与产品差 {gap_naive['mad']:.2f} / {gap_mapped['mad']:.2f} 级")
        if at == middle:
            kept_product = product.copy()
            kept_mapped = mapped_u8.copy()
            kept_signed = gap_mapped["signed"]
        del naive_u8, mapped_u8, product

    check("HDR D1：产品口径＝「PQ 值当 sRGB」（≤4 级；产品抽帧自带一次 JPEG 往返）",
          max(naive_mads) <= 4.0, f"三时刻 {[round(value, 2) for value in naive_mads]} 级")
    check("HDR D2：正确的 HDR→SDR 与产品成品差 >12 级（三时刻）",
          min(mapped_mads) > 12.0, f"三时刻 {[round(value, 2) for value in mapped_mads]} 级")
    check("HDR D3：正确映射后画面比产品成品更暗（中位亮度三时刻都低 ≥10 级——产品是偏灰那一侧）",
          all(dimmer), "产品→正确 " + "、".join(luma_pairs))
    check("HDR D4：三通道有符号差之和为负（整体偏暗）",
          all(darker), f"中间一帧 RGB 有符号差 {[round(value, 2) for value in kept_signed]}")

    if kept_product is not None:
        for label, frame in (("产品抽帧", kept_product), ("色调映射后", kept_mapped)):
            found = detect_watermark(frame)
            check(f"HDR E：{label}上自动识别仍返回完整契约",
                  all(key in found for key in ("found", "regions", "suggestions", "message")),
                  f"found={found.get('found')} regions={len(found.get('regions') or [])} "
                  f"suggestions={len(found.get('suggestions') or [])}")

    for clip in (QA / "_bbb4k_raw.mp4", OCEANS):
        if not clip.exists():
            print(f"  未验：缺少 SDR 对照素材 {clip.name}，这一条没有跑")
            continue
        clip_info = media.probe_video(clip)
        clip_w, clip_h = int(clip_info["width"]), int(clip_info["height"])
        clip_len = media.probe_video_length(clip)
        at = round(float(clip_len.get("duration") or 1.0) * 0.5, 3)
        stored = _hdr_decode_rgb24(clip, at, QA / "_rv_hdr_sdr.yuv", clip_w, clip_h)
        extracted = media.extract_video_frame(clip, at)
        control = _hdr_compare(stored, extracted)
        check(f"HDR E：SDR 对照 {clip.name} 上「抽出即存」与产品抽帧一致（≤4 级）",
              control["mad"] <= 4.0, f"平均差 {control['mad']:.2f} 级")
        del stored, extracted


def cleanup() -> None:
    for path in list(QA.glob("_rv_*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def main() -> int:
    phases = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or ["all"]
    want_all = "all" in phases
    if want_all or "build" in phases:
        build()
    if want_all or "detect" in phases:
        phase_detect()
    if want_all or "mode" in phases:
        phase_mode()
    if want_all or "mask" in phases:
        phase_mask()
    if want_all or "motion" in phases:
        phase_motion()
    if want_all or "streams" in phases:
        phase_streams()
    if want_all or "uhd" in phases:
        phase_uhd()
    if want_all or "subpic" in phases:
        phase_subpic()
    if want_all or "hdr" in phases:
        phase_hdr()
    if want_all or "http" in phases:
        phase_http()
    print("\n结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    if want_all and "--keep" not in sys.argv:
        cleanup()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
