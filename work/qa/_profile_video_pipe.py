"""量 #5：逐帧路线（`_process_frame_by_frame`）的每帧耗时构成。

背景：§10.4 差距 #5 写的是「逐帧路线每帧一次进程间管道往返，对标工具是只修出现水印的那几帧」。
这句话一直没量过——不知道**往返本身**占多少，就不知道该不该为它改帧循环结构。

这台脚本把一趟逐帧处理拆成四项累加：
  读   = `_read_frame`（解码进程 → 我们，真管道；旧写法是 `_read_exact`）
  写   = `encoder.stdin.write`（我们 → 编码进程；它包含编码器自己的回压，不纯是系统调用）
  修   = `remove_boxes_inplace`（逐帧修复，即图片引擎）
  自有 = 总耗时 − 读 − 写 − 修（np.frombuffer().copy()、cvtColor、tobytes()、循环、闸门、进度）
另跑一趟「短路修复」的对照组（修复换成空操作），用总时长差复核「修」这一项。
三项微基准（copy / cvtColor / tobytes）单独在同一尺寸的空帧上量，用来核对「自有」这一栏的量级。
**改前后各跑一遍这台脚本**：微基准里 `copy` 与 `tobytes` 两项正是旧写法每帧多付的整帧拷贝，
新写法（`readinto` 进预分配缓冲 + `memoryview` 直写）不再付；`cvtColor` 只在 `motion=True` 时才付。

夹具全部由 lavfi 现合成，写 `work/qa/_vp_*`，跑完自清（`--keep` 留档）。**不读用户素材。**
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from remover import video_ops  # noqa: E402
from remover.image_ops import Box  # noqa: E402
from remover.media import ffmpeg_path  # noqa: E402

QA = ROOT / "work" / "qa"
KEEP = "--keep" in sys.argv[1:]

SIZES = [
    (960, 540, 90),
    (1920, 1080, 45),
    (3840, 2160, 24),
]


def build_clip(path: Path, width: int, height: int, frames: int) -> None:
    fps = 30.0
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps}",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)


class Accumulator:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0


def instrument(accumulator: Accumulator, original):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            accumulator.seconds += time.perf_counter() - started
            accumulator.calls += 1

    return wrapper


class WriteTimer:
    """包住编码器 stdin：量 write 的墙钟（含编码器回压）。"""

    def __init__(self, stream, accumulator: Accumulator) -> None:
        self._stream = stream
        self._acc = accumulator

    def write(self, data):
        started = time.perf_counter()
        try:
            return self._stream.write(data)
        finally:
            self._acc.seconds += time.perf_counter() - started
            self._acc.calls += 1

    def close(self):
        return self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def run_once(path_in: Path, path_out: Path, width: int, height: int, frames: int,
             repair_enabled: bool) -> dict:
    acc_read = Accumulator()
    acc_write = Accumulator()
    acc_repair = Accumulator()

    real_read = video_ops._read_frame
    real_repair = video_ops.remove_boxes_inplace
    real_guard = video_ops._PipeGuard

    original_repair = real_repair

    def repair_noop(rgb, boxes, method="smart"):
        return {"fallback_regions": 0, "directions": []}

    def guard_spy(processes, label):
        guard = real_guard(processes, label)
        encoder = processes[1] if len(processes) > 1 else None
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin = WriteTimer(encoder.stdin, acc_write)
        return guard

    video_ops._read_frame = instrument(acc_read, real_read)
    video_ops._PipeGuard = guard_spy
    video_ops.remove_boxes_inplace = (
        original_repair if repair_enabled else repair_noop
    )
    if repair_enabled:
        acc_repair_target = original_repair
    else:
        acc_repair_target = repair_noop
    video_ops.remove_boxes_inplace = instrument(acc_repair, acc_repair_target)
    started = time.perf_counter()
    try:
        result = video_ops._process_frame_by_frame(
            path_in,
            path_out,
            [Box(width - 130, height - 46, width - 12, height - 12)],
            width,
            height,
            frames / 30.0,
            30.0,
            "inpaint",
            lambda _p: None,
            motion=False,
            stream_args=None,
            trim_to_video=True,
            mask_plans=None,
            should_stop=None,
        )
    finally:
        video_ops._read_frame = real_read
        video_ops._PipeGuard = real_guard
        video_ops.remove_boxes_inplace = real_repair
    wall = time.perf_counter() - started
    return {
        "wall": wall,
        "frames": result["frames"],
        "read": acc_read.seconds,
        "write": acc_write.seconds,
        "repair": acc_repair.seconds,
        "read_calls": acc_read.calls,
        "write_calls": acc_write.calls,
    }


def micro_benchmark(width: int, height: int, rounds: int = 20) -> dict:
    raw = bytes(width * height * 3)
    copy_s = gray_s = tobytes_s = 0.0
    for _ in range(rounds):
        t0 = time.perf_counter()
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        t1 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        t2 = time.perf_counter()
        payload = frame.tobytes()
        t3 = time.perf_counter()
        copy_s += t1 - t0
        gray_s += t2 - t1
        tobytes_s += t3 - t2
        del gray, payload
    return {
        "copy": copy_s / rounds,
        "cvtColor": gray_s / rounds,
        "tobytes": tobytes_s / rounds,
    }


def report_line(name: str, data: dict) -> None:
    frames = data["frames"] or 1
    own = data["wall"] - data["read"] - data["write"] - data["repair"]
    print(
        f"{name}: 帧数={frames} 总={data['wall']:.2f}s "
        f"读={data['read']:.2f}s({100 * data['read'] / data['wall']:.1f}%) "
        f"写={data['write']:.2f}s({100 * data['write'] / data['wall']:.1f}%) "
        f"修={data['repair']:.2f}s({100 * data['repair'] / data['wall']:.1f}%) "
        f"自有={own:.2f}s({100 * own / data['wall']:.1f}%) "
        f"| 每帧 总={1000 * data['wall'] / frames:.1f}ms "
        f"读={1000 * data['read'] / frames:.1f}ms "
        f"写={1000 * data['write'] / frames:.1f}ms "
        f"修={1000 * data['repair'] / frames:.1f}ms "
        f"自有={1000 * own / frames:.1f}ms"
    )


def main() -> int:
    fixtures = []
    total_read_write = []
    try:
        for width, height, frames in SIZES:
            path_in = QA / f"_vp_{width}x{height}.mp4"
            path_out = QA / f"_vp_{width}x{height}_out.mp4"
            fixtures += [path_in, path_out]
            build_clip(path_in, width, height, frames)
            print(f"== {width}x{height} / {frames} 帧 ==")
            full = run_once(path_in, path_out, width, height, frames, True)
            report_line("逐帧（含修复）", full)
            passthrough = run_once(path_in, path_out, width, height, frames, False)
            report_line("对照（修复短路）", passthrough)
            delta = full["wall"] - passthrough["wall"]
            print(
                f"复核：总时长差={delta:.2f}s vs 累加的修复耗时={full['repair']:.2f}s"
                f"（差 {abs(delta - full['repair']):.2f}s）"
            )
            pipe = full["read"] + full["write"]
            total_read_write.append((f"{width}x{height}", pipe / full["wall"]))
            micro = micro_benchmark(width, height)
            print(
                f"微基准（每帧一次）：copy={1000 * micro['copy']:.1f}ms "
                f"cvtColor={1000 * micro['cvtColor']:.1f}ms "
                f"tobytes={1000 * micro['tobytes']:.1f}ms "
                f"三项合计={1000 * (micro['copy'] + micro['cvtColor'] + micro['tobytes']):.1f}ms"
            )
            print(
                f"往返占整趟的比例：读+写 = {100 * pipe / full['wall']:.1f}%"
            )
        print("== 汇总 ==")
        for name, share in total_read_write:
            print(f"{name}: 读+写 占整趟 {100 * share:.1f}%")
        return 0
    finally:
        if not KEEP:
            for path in fixtures:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
