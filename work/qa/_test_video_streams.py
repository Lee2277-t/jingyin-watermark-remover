# -*- coding: utf-8 -*-
"""视频成品音轨 / 字幕轨回归（合成素材，不读用户作品）。

要证明的事：成品固定是 .mp4，但**没理由把用户的音轨重编码、把字幕丢掉**。
- 能直接装进 MP4 的音轨（aac/mp3/alac/ac3/eac3）必须原样搬过去，解码出来的 PCM 逐样本一致；
- 装不进的音轨（本夹具用 mkv + pcm_s16le）重编码为 AAC，并在说明里如实写；
- 文本字幕转成 mov_text 保留，图形字幕略过；说明文案与实际情况一致；
- 两条编码路线（delogo / 逐帧）都要做到，且成品时长不缩水。

夹具：640×360 / 24fps / 2s 合成画面 + 440Hz 正弦音轨 + 两条英文字幕；另外一份无音轨无字幕。

用法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_video_streams.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from remover import video_ops  # noqa: E402
from remover.media import probe_streams, probe_video  # noqa: E402

QA = BASE / "work" / "qa"
RAW = QA / "_vs_raw.bgr"
SRT = QA / "_vs_subs.srt"
A_MP4 = QA / "_vs_a.mp4"  # aac + mov_text
B_MKV = QA / "_vs_b.mkv"  # pcm_s16le + subrip
C_MP4 = QA / "_vs_c.mp4"  # 无音轨无字幕
D_MKV = QA / "_vs_d.mkv"  # pcm 音轨、无字幕（唯一会启用 -shortest 的组合）
H_MP4 = QA / "_vs_h.mp4"  # 带 HDR 色彩标记（bt2020nc / bt2020 / smpte2084）
OUT_DELOGO = QA / "_vs_out_delogo.mp4"
OUT_FRAME = QA / "_vs_out_frame.mp4"
OUT_B = QA / "_vs_out_b.mp4"
OUT_C = QA / "_vs_out_c.mp4"
OUT_D = QA / "_vs_out_d.mp4"
W, H, FPS, N = 640, 360, 24, 48
INSET_REGION = {"x": 24 / W, "y": 40 / H, "w": 120 / W, "h": 52 / H}  # 不贴边 → delogo
EDGE_REGION = {"x": 0.0, "y": 0.3, "w": 0.16, "h": 0.16}  # 贴左边 → 逐帧
CUE_ONE = "Stream check cue one"
CUE_TWO = "Stream check cue two"
TEMP_FILES = (RAW, SRT, A_MP4, B_MKV, C_MP4, D_MKV, H_MP4, OUT_DELOGO, OUT_FRAME, OUT_B, OUT_C, OUT_D)
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_frames() -> None:
    """一段带角落白字的合成画面，落成裸 rgb24，三份夹具共用。"""
    rng = np.random.default_rng(7)
    with open(RAW, "wb") as handle:
        for index in range(N):
            frame = np.zeros((H, W, 3), np.uint8)
            for y in range(H):
                ratio = y / H
                frame[y, :] = (
                    int(90 + 70 * ratio + 10 * np.sin((y + index * 2) / 14.0)),
                    int(110 + 60 * ratio),
                    int(140 - 40 * ratio),
                )
            frame = np.clip(
                frame.astype(np.int16) + rng.integers(-5, 6, (H, W, 1), dtype=np.int16), 0, 255
            ).astype(np.uint8)
            cv2.putText(frame, "AI", (40, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            handle.write(frame.tobytes())


def write_srt() -> None:
    SRT.write_text(
        "1\n00:00:00,200 --> 00:00:00,900\n" + CUE_ONE + "\n\n"
        "2\n00:00:01,000 --> 00:00:01,700\n" + CUE_TWO + "\n",
        encoding="utf-8",
    )


def build_clips() -> None:
    write_frames()
    write_srt()
    common = [
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", str(RAW),
    ]
    # 用 -t 而不是 -shortest：实测 -shortest 会被最后一条字幕（1.7s）带着走，
    # 把 48 帧画面砍成 40 帧——夹具本身残缺就没法验成品了。
    length = ["-t", f"{N / FPS}"]
    subprocess.run(
        common + [
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={N / FPS + 1}",
            "-i", str(SRT),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-c:s", "mov_text", *length, str(A_MP4),
        ],
        check=True,
    )
    subprocess.run(
        common + [
            "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={N / FPS + 1}",
            "-i", str(SRT),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-c:s", "srt", *length, str(B_MKV),
        ],
        check=True,
    )
    subprocess.run(
        common + [
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(C_MP4),
        ],
        check=True,
    )
    # HDR 标记夹具：画面与 C 同源，只把像素格式与三项色彩标记换成 HDR 的写法。
    # 注意：`-color_trc` 这类输出选项传到 libx264 会落成 unknown，
    # 必须走 `-x264-params` 才写得出 bt2020nc/bt2020/smpte2084 这组标记。
    subprocess.run(
        common + [
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p10le",
            "-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            str(H_MP4),
        ],
        check=True,
    )


def pcm_digest(path: Path, index: int = 0) -> tuple[str, int]:
    result = subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", f"0:a:{index}", "-f", "s16le", "-ac", "2", "-ar", "48000", "-",
        ],
        capture_output=True,
    )
    return hashlib.md5(result.stdout).hexdigest(), len(result.stdout)


def decode_frame_count(path: Path) -> int:
    """真数一遍帧：cv2 的 CAP_PROP_FRAME_COUNT 是「容器时长 × fps」，mkv 里有长音轨时会虚报。"""
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    return count


def subtitle_text(path: Path) -> str:
    result = subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:s:0", "-f", "srt", "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout or ""


def run(input_path: Path, output_path: Path, region: dict, label: str) -> dict:
    info = probe_video(input_path)
    started = time.perf_counter()
    result = video_ops.process_video_file(
        input_path,
        output_path,
        [region],
        int(info["width"]),
        int(info["height"]),
        method="smart",
        on_progress=lambda _value: None,
        video_mode="quality",
    )
    result["elapsed"] = time.perf_counter() - started
    print(f"  {label}: {result['method']} / {result['elapsed']:.2f}s / {result['note']}")
    return result


def main() -> int:
    build_clips()

    probe_a = probe_streams(A_MP4)
    probe_b = probe_streams(B_MKV)
    probe_c = probe_streams(C_MP4)
    check(
        "夹具 A 认出 aac + mov_text",
        [item["codec"] for item in probe_a["audio"]] == ["aac"]
        and [item["codec"] for item in probe_a["subtitles"]] == ["mov_text"],
        str(probe_a),
    )
    check(
        "夹具 B 认出 pcm_s16le + subrip",
        [item["codec"] for item in probe_b["audio"]] == ["pcm_s16le"]
        and [item["codec"] for item in probe_b["subtitles"]] == ["subrip"],
        str(probe_b),
    )
    check("夹具 C 没有音轨与字幕", not probe_c["audio"] and not probe_c["subtitles"], str(probe_c))

    probe_h = probe_streams(H_MP4)
    check(
        "夹具 H 认出 HDR 传输特性（smpte2084）",
        probe_h["video"].get("hdr") is True and probe_h["video"].get("transfer") == "smpte2084",
        str(probe_h.get("video")),
    )
    check(
        "SDR 夹具 C 不被误判成 HDR",
        not probe_c["video"].get("hdr"),
        str(probe_c.get("video")),
    )
    note_h = video_ops._streams_note(video_ops._plan_streams(probe_h))
    check("HDR 素材的成品说明写明不做色调映射", "HDR" in note_h and "色调映射" in note_h, note_h)
    note_c = video_ops._streams_note(video_ops._plan_streams(probe_c))
    check("SDR 素材的说明里没有 HDR 那句", "HDR" not in note_c, note_c or "(空)")

    source_duration = float(probe_video(A_MP4)["duration"])
    source_pcm = pcm_digest(A_MP4)

    delogo = run(A_MP4, OUT_DELOGO, INSET_REGION, "A/内插值")
    check("A 走内插值路线", delogo["method"] == "delogo", str(delogo["method"]))
    check("A 说明写明原音轨已保留", "原音轨已保留" in delogo["note"], delogo["note"])
    check("A 说明写字幕已保留", "字幕已保留" in delogo["note"], delogo["note"])
    check("A 说明不再谎称字幕不保留", "字幕不保留" not in delogo["note"], delogo["note"])
    out_a = probe_streams(OUT_DELOGO)
    check("A 成品音轨仍是 aac", [item["codec"] for item in out_a["audio"]] == ["aac"], str(out_a))
    check("A 成品字幕是 mov_text", [item["codec"] for item in out_a["subtitles"]] == ["mov_text"], str(out_a))
    copied_pcm = pcm_digest(OUT_DELOGO)
    check(
        "A 音轨逐样本与源一致（无损搬运）",
        copied_pcm == source_pcm and copied_pcm[1] > 0,
        f"{copied_pcm[0][:8]} vs {source_pcm[0][:8]} | {copied_pcm[1]:,} B",
    )
    subs = subtitle_text(OUT_DELOGO)
    check("A 两条字幕文本都在", CUE_ONE in subs and CUE_TWO in subs, subs.replace("\n", " | ")[:80])
    duration = float(probe_video(OUT_DELOGO)["duration"])
    check("A 成品时长没缩水", abs(duration - source_duration) <= 0.3, f"{duration:.2f}s / 源 {source_duration:.2f}s")

    frame = run(A_MP4, OUT_FRAME, EDGE_REGION, "A/逐帧")
    check("A 贴边选框走逐帧路线", frame["method"].startswith("frame"), str(frame["method"]))
    check("A 逐帧说明也写原音轨已保留", "原音轨已保留" in frame["note"], frame["note"])
    check("A 逐帧说明也写字幕已保留", "字幕已保留" in frame["note"], frame["note"])
    out_frame = probe_streams(OUT_FRAME)
    check(
        "A 逐帧成品音轨仍是 aac、字幕仍是 mov_text",
        [item["codec"] for item in out_frame["audio"]] == ["aac"]
        and [item["codec"] for item in out_frame["subtitles"]] == ["mov_text"],
        str(out_frame),
    )
    frame_pcm = pcm_digest(OUT_FRAME)
    check(
        "A 逐帧音轨逐样本与源一致",
        frame_pcm == source_pcm and frame_pcm[1] > 0,
        f"{frame_pcm[0][:8]} vs {source_pcm[0][:8]}",
    )
    frame_subs = subtitle_text(OUT_FRAME)
    check("A 逐帧字幕文本都在", CUE_ONE in frame_subs and CUE_TWO in frame_subs, frame_subs.replace("\n", " | ")[:80])
    frame_duration = float(probe_video(OUT_FRAME)["duration"])
    check(
        "A 逐帧成品时长没缩水",
        abs(frame_duration - source_duration) <= 0.3,
        f"{frame_duration:.2f}s / 源 {source_duration:.2f}s",
    )
    copied_frames = decode_frame_count(OUT_FRAME)
    check("A 搬运音轨时帧数不被截掉", abs(copied_frames - N) <= 1, f"{N} -> {copied_frames}")

    converted = run(B_MKV, OUT_B, INSET_REGION, "B/内插值")
    check("B 走内插值路线", converted["method"] == "delogo", str(converted["method"]))
    check("B 说明如实写重编码", "已重编码为 AAC" in converted["note"], converted["note"])
    check("B 说明写字幕已保留", "字幕已保留" in converted["note"], converted["note"])
    out_b = probe_streams(OUT_B)
    check("B 成品音轨变成 aac", [item["codec"] for item in out_b["audio"]] == ["aac"], str(out_b))
    check("B 成品字幕是 mov_text", [item["codec"] for item in out_b["subtitles"]] == ["mov_text"], str(out_b))
    check("B 成品有声音", pcm_digest(OUT_B)[1] > 0, f"{pcm_digest(OUT_B)[1]:,} B")
    b_subs = subtitle_text(OUT_B)
    check("B 字幕文本转换后仍在", CUE_ONE in b_subs and CUE_TWO in b_subs, b_subs.replace("\n", " | ")[:80])

    silent = run(C_MP4, OUT_C, INSET_REGION, "C/内插值")
    check("C 走内插值路线", silent["method"] == "delogo", str(silent["method"]))
    check("C 说明不提音轨字幕", "音轨" not in silent["note"] and "字幕" not in silent["note"], silent["note"])
    out_c = probe_streams(OUT_C)
    check("C 成品没有凭空多出音轨字幕", not out_c["audio"] and not out_c["subtitles"], str(out_c))

    # D：装不进 MP4 的音轨 + 没有字幕 → 只有这一种组合会启用 `-shortest`，要确认它不吞帧
    subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", str(RAW),
            "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={N / FPS + 1}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", str(D_MKV),
        ],
        check=True,
    )
    reencode = run(D_MKV, OUT_D, EDGE_REGION, "D/逐帧")
    check("D 逐帧路线", reencode["method"].startswith("frame"), str(reencode["method"]))
    check("D 说明如实写重编码", "已重编码为 AAC" in reencode["note"], reencode["note"])
    check("D 说明不提字幕", "字幕" not in reencode["note"], reencode["note"])
    check("D 成品保留音轨", probe_streams(OUT_D)["audio"] != [], str(probe_streams(OUT_D)))
    d_output = decode_frame_count(OUT_D)
    check(
        "D 重编码音轨 + 对齐长度时帧数不被截掉",
        abs(d_output - N) <= 1,
        f"{N} -> {d_output}",
    )
    d_duration = float(probe_video(OUT_D)["duration"])
    check(
        "D 长音轨被对齐到画面长度（不多出黑尾）",
        abs(d_duration - N / FPS) <= 0.3,
        f"{d_duration:.2f}s / 画面 {N / FPS:.2f}s",
    )

    for path in TEMP_FILES:
        if path.exists():
            path.unlink()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
