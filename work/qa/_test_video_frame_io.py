# -*- coding: utf-8 -*-
"""逐帧管道读写的字节契约测试（第二十二轮，差距 #5 收口时补的）。

这一轮把 `_read_exact`（`bytearray.extend` + `bytes()`，每帧多拷两整帧）换成
`_read_frame`（`readinto` 直接写进预分配缓冲），并把编码器写入从 `frame.tobytes()`
换成 `memoryview(frame).cast("B")`。**坏法不会报错，只会让每一帧都错位几个字节**，
所以这里量的不是像素、是「搬进来的字节是不是和整块读一模一样」：

① 一帧一次读满；② 管道把一帧切成 1 字节一片仍然读满、不错位；
③ 只剩半帧时判结束并**把半帧丢掉**（不能发出去）；④ 连续两帧复用同一块缓冲，
第二帧里不残留第一帧的字节；⑤ 缓冲必须是连续内存且字节数 == w*h*3（防将来改成
非连续缓冲时静默错位）；⑥ `memoryview(frame).cast("B")` 交给编码器的字节与
`frame.tobytes()` 完全一致；⑦ 真 ffmpeg 端到端：逐帧循环读到的每一帧 sha1，
与「同一支片子一次性整块解码再切帧」的 sha1 **逐帧相同**、帧数相同；
⑧ 端到端交给编码器的**每一次写正好一帧的字节数**（半帧不会被发出去）；
⑨ 截断夹具这一类「原片自己就不完整」的素材，要么明确抛 `VideoProcessError`、
要么帧数不超过原片——**不会凭空造出多出来的一帧**（本机 ffmpeg 对截断 mp4 是直接报错，
所以这条记的是"不静默多帧"，不是"能救回半截"）。
共 12 项。
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from remover import video_ops  # noqa: E402
from remover.image_ops import Box  # noqa: E402
from remover.media import ffmpeg_path  # noqa: E402

QA = ROOT / "work" / "qa"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


class TrickleStream:
    """模拟管道把字节一片一片送过来：每次 readinto 最多吃 `grain` 个字节。"""

    def __init__(self, data: bytes, grain: int) -> None:
        self._data = data
        self._pos = 0
        self._grain = grain

    def readinto(self, view) -> int:
        if self._pos >= len(self._data):
            return 0
        take = min(self._grain, len(view), len(self._data) - self._pos)
        view[:take] = self._data[self._pos : self._pos + take]
        self._pos += take
        return take


class WriteSpy:
    """替编码器的 stdin 记账：逐帧循环每交给它一次，就记下这次写了多少字节。"""

    def __init__(self, stream, sizes: list[int]) -> None:
        self._stream = stream
        self._sizes = sizes

    def write(self, data) -> int:
        self._sizes.append(len(memoryview(data).cast("B")))
        return self._stream.write(data)

    def close(self) -> None:
        self._stream.close()

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def guard_spy(real_guard, processes: list, label: str, sizes: list[int]):
    """卡死闸门的第一个参数就是 [解码器, 编码器]，从这儿换掉编码器的 stdin 最省事——
    不用去动 subprocess（第十九轮踩过：直接给 stdlib 打补丁会泄漏进 imageio_ffmpeg）。"""
    guard = real_guard(processes, label)
    if len(processes) > 1 and processes[1].stdin is not None:
        processes[1].stdin = WriteSpy(processes[1].stdin, sizes)
    return guard


def bulk_decode(path: Path, width: int, height: int) -> list[bytes]:
    """同一支片子一次性整块解码，切成帧——作为逐帧循环的参照。"""
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps=30,scale={width}:{height}:flags=lanczos,setsar=1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    frame_size = width * height * 3
    return [raw[i : i + frame_size] for i in range(0, len(raw) - frame_size + 1, frame_size)]


def build_clip(path: Path, width: int, height: int, frames: int) -> None:
    subprocess.run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate=30",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> int:
    width, height = 64, 48
    frame_size = width * height * 3
    frame_a = np.arange(frame_size, dtype=np.uint8).reshape(height, width, 3)
    frame_b = ((frame_a.astype(np.int32) * 3) % 256).astype(np.uint8)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    ok = video_ops._read_frame(io.BytesIO(frame_a.tobytes()), buffer)
    check("一帧一次读满", ok and (buffer == frame_a).all(), f"bytes={frame_size}")

    ok = video_ops._read_frame(TrickleStream(frame_a.tobytes(), grain=1), buffer)
    check("管道一片一字节仍读满且不错位", ok and (buffer == frame_a).all(), "grain=1")

    ok = video_ops._read_frame(TrickleStream(frame_a.tobytes(), grain=7), buffer)
    check("管道一片七字节仍读满且不错位", ok and (buffer == frame_a).all(), "grain=7")

    half = frame_a.tobytes() + frame_b.tobytes()[: frame_size // 2]
    reads = []
    stream = io.BytesIO(half)
    while video_ops._read_frame(stream, buffer):
        reads.append(memoryview(buffer).cast("B").tobytes())
    check("尾部半帧判结束、不发出去", len(reads) == 1 and reads[0] == frame_a.tobytes(), f"成功读到的帧数={len(reads)}")

    stream = io.BytesIO(frame_a.tobytes() + frame_b.tobytes())
    first = np.zeros((height, width, 3), dtype=np.uint8)
    second = np.zeros((height, width, 3), dtype=np.uint8)
    video_ops._read_frame(stream, first)
    video_ops._read_frame(stream, second)
    check(
        "两帧连续读，第二帧不残留第一帧字节",
        (second == frame_b).all() and (first == frame_a).all(),
        f"第二帧差值={int(np.abs(second.astype(int) - frame_b.astype(int)).max())}",
    )

    reused = np.zeros((height, width, 3), dtype=np.uint8)
    reused[:] = frame_a
    video_ops._read_frame(io.BytesIO(frame_b.tobytes()), reused)
    check("同一块缓冲复用：整帧被覆盖干净", (reused == frame_b).all())

    view = memoryview(buffer).cast("B")
    check(
        "缓冲是连续内存且字节数等于 w*h*3",
        buffer.flags["C_CONTIGUOUS"] and view.nbytes == frame_size,
        f"nbytes={view.nbytes}",
    )
    check(
        "交给编码器的字节与 tobytes() 完全一致",
        memoryview(frame_a).cast("B").tobytes() == frame_a.tobytes(),
        f"sha1={hashlib.sha1(memoryview(frame_a).cast('B')).hexdigest()[:8]}",
    )

    fixture = QA / "_vfi_clip.mp4"
    output = QA / "_vfi_out.mp4"
    truncated = QA / "_vfi_trunc.mp4"
    try:
        build_clip(fixture, 320, 240, 8)
        expected = [hashlib.sha1(f).hexdigest() for f in bulk_decode(fixture, 320, 240)]

        recorded: list[str] = []
        write_sizes: list[int] = []
        real_read = video_ops._read_frame
        real_guard = video_ops._PipeGuard

        def spy(stream, frame):
            ok = real_read(stream, frame)
            if ok:
                recorded.append(hashlib.sha1(memoryview(frame).cast("B")).hexdigest())
            return ok

        video_ops._read_frame = spy
        video_ops._PipeGuard = lambda processes, label: guard_spy(real_guard, processes, label, write_sizes)
        try:
            result = video_ops._process_frame_by_frame(
                fixture,
                output,
                [Box(200, 200, 300, 232)],
                320,
                240,
                8 / 30.0,
                30.0,
                "inpaint",
                lambda _p: None,
            )
        finally:
            video_ops._read_frame = real_read
            video_ops._PipeGuard = real_guard

        check(
            "端到端：逐帧循环读到的帧数与整块解码一致",
            result["frames"] == len(expected) == len(recorded),
            f"循环={result['frames']} 整块={len(expected)} 记录={len(recorded)}",
        )
        check(
            "端到端：每一帧的 sha1 与整块解码逐帧相同",
            recorded == expected,
            f"首个不一致={next((i for i, (a, b) in enumerate(zip(recorded, expected)) if a != b), '无')}",
        )
        check(
            "端到端：交给编码器每次正好一帧（半帧不会被发出去）",
            len(write_sizes) == result["frames"] and all(n == 320 * 240 * 3 for n in write_sizes),
            f"写次数={len(write_sizes)} 每次字节={sorted(set(write_sizes))}",
        )

        raw = fixture.read_bytes()
        truncated.write_bytes(raw[: int(len(raw) * 0.6)])
        recorded.clear()
        video_ops._read_frame = spy
        try:
            tail_frames = 0
            try:
                tail = video_ops._process_frame_by_frame(
                    truncated,
                    output,
                    [Box(200, 200, 300, 232)],
                    320,
                    240,
                    4 / 30.0,
                    30.0,
                    "inpaint",
                    lambda _p: None,
                )
                tail_frames = tail["frames"]
            except video_ops.VideoProcessError:
                tail_frames = -1
        finally:
            video_ops._read_frame = real_read
        check(
            "截断夹具：要么明确报错、要么帧数不超过原片（不会凭空造出半帧）",
            tail_frames == -1 or 0 <= tail_frames <= result["frames"],
            f"成品帧数={tail_frames} 原片帧数={result['frames']}",
        )
    finally:
        for path in (fixture, output, truncated):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    print(f"结论={'全部通过' if not FAILED else '失败 ' + str(len(FAILED)) + ' 项：' + ', '.join(FAILED)}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
