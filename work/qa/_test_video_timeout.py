# -*- coding: utf-8 -*-
"""ffmpeg 卡死总闸回归：看门狗单元、逐帧路线真卡死、delogo 管道、以及「没触发时一行不变」对照组。

第十九轮新增。判据是「管道没有任何动静 ⇒ 自动杀掉并诚实报错；只要还在出帧/出进度行 ⇒
哪怕整趟跑很久也不该被拦」，外加「装了闸门但没触发时，method、说明、帧数一字不变」。
素材全合成，不读用户作品。

跑法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_video_timeout.py
只跑不需要 ffmpeg 的部分：加 --unit
保留夹具便于肉眼复看：加 --keep

delogo 那一段是**管道级**（喂假 Popen），不是真 ffmpeg 端到端——本机造不出「真会卡住的 ffmpeg」，
日志里按这一点分开命名，不要当成端到端证据读。
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import imageio_ffmpeg

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
QA = BASE / "work" / "qa"
OUT_DIR = QA / "_vt_out"

from remover import media, video_ops  # noqa: E402
from remover.cancel import CancelRequested  # noqa: E402
from remover.media import MediaError  # noqa: E402
from remover.video_ops import VideoProcessError, _PipeGuard  # noqa: E402

W, H, FPS, SECONDS = 480, 270, 30, 2
TOTAL_FRAMES = FPS * SECONDS
CLIP = QA / "_vt_marked.mp4"
# 右下角贴边选框：x+w 顶到画面边 ⇒ 画质优先必然走逐帧（delogo 放不下贴边框）
REGION = {"x": (W - 150) / W, "y": (H - 44) / H, "w": 150 / W, "h": 44 / H}
FAILED: list[str] = []
KEEP = "--keep" in sys.argv
UNIT_ONLY = "--unit" in sys.argv

DEFAULT_IDLE = video_ops.PIPE_IDLE_TIMEOUT
DEFAULT_POLL = video_ops.PIPE_IDLE_POLL
DEFAULT_FLUSH = video_ops.PIPE_FLUSH_TIMEOUT

# 逐帧卡死那一段用的缩短值：真等 120 秒没有意义，但判据必须是「没有动静」而不是「跑得久」
FAST_IDLE = 1.0
FAST_POLL = 0.2
FAST_FLUSH = 5.0
HANG_SECONDS = 6.0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


@contextlib.contextmanager
def patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# video_ops / media 都用 `subprocess.xxx`，只换模块里那个名字，不碰真的 subprocess 模块——
# 全局换掉会连累 imageio_ffmpeg 内部 `platform.machine()` 的 check_output（实测踩过）。
SUBPROCESS_ATTRS = ("DEVNULL", "PIPE", "Popen", "SubprocessError", "TimeoutExpired", "run")


@contextlib.contextmanager
def patched_subprocess(module, **overrides):
    attrs = {name: getattr(subprocess, name) for name in SUBPROCESS_ATTRS}
    attrs.update(overrides)
    with patched(module, "subprocess", types.SimpleNamespace(**attrs)):
        yield


@contextlib.contextmanager
def fast_gate():
    with patched(video_ops, "PIPE_IDLE_TIMEOUT", FAST_IDLE), patched(
        video_ops, "PIPE_IDLE_POLL", FAST_POLL
    ), patched(video_ops, "PIPE_FLUSH_TIMEOUT", FAST_FLUSH):
        yield


def build_clip() -> None:
    """480x270 / 2s 合成片段：移动的分形底，不含任何文字。"""
    if CLIP.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    base = (
        f"gradients=size={W}x{H}:rate={FPS}:c0=0x203040:c1=0x4a3a55:c2=0x2e4a3b:"
        f"c3=0x55402a:x0=0:y0=0:x1={W}:y1={H}:duration={SECONDS}"
    )
    vf = (
        f"[0:v][1:v]overlay=x='mod(t*90\\,{W})-{W // 2}':y='mod(t*60\\,{H})-{H // 2}'"
        f":eof_action=repeat"
    )
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", base,
            "-f", "lavfi", "-i", f"mandelbrot=size={W // 2}x{H // 2}:rate={FPS}",
            "-filter_complex", vf, "-t", str(SECONDS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-an", str(CLIP),
        ],
        check=True,
    )


def run_engine(output_path: Path, **kwargs) -> dict:
    probe = media.probe_video(CLIP)
    return video_ops.process_video_file(
        CLIP,
        output_path,
        [dict(REGION)],
        probe["width"],
        probe["height"],
        # inpaint：跳过静止水印掩膜采样，这一段只测闸门本身，不掺第十七轮那条路线
        method="inpaint",
        video_mode="quality",
        **kwargs,
    )


def frames_of(path: Path) -> int:
    return int(media.probe_video(path)["frame_count"])


def wait_dead(processes: list, limit: float = 3.0) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            return True
        time.sleep(0.1)
    return all(process.poll() is not None for process in processes)


# ---------------------------------------------------------------- 守护线程单元

def unit_guard() -> None:
    check(
        "unit-01 默认口径没被改动",
        (DEFAULT_IDLE, DEFAULT_POLL, DEFAULT_FLUSH) == (120.0, 1.0, 300.0),
        f"idle={DEFAULT_IDLE} poll={DEFAULT_POLL} flush={DEFAULT_FLUSH}",
    )
    with fast_gate():
        hung = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        quick = subprocess.Popen([sys.executable, "-c", "pass"])
        quick.wait(timeout=10)

        guard = _PipeGuard([hung, quick], "测试管道")
        time.sleep(FAST_IDLE + FAST_POLL * 3)
        check("unit-02 没有动静会开火", bool(guard.reason), guard.reason)
        check("unit-03 卡住的进程被杀掉", hung.poll() is not None, f"returncode={hung.poll()}")
        check("unit-04 开火文案带秒数", "秒没有任何进展" in guard.reason, guard.reason)
        check("unit-05 已自己退出的进程不会被误杀", quick.poll() is not None)
        try:
            guard.raise_if_stuck()
            check("unit-06 开火后 raise_if_stuck 抛错", False, "没有抛出")
        except VideoProcessError as exc:
            text = str(exc)
            check(
                "unit-06 开火后 raise_if_stuck 抛错",
                text.startswith("测试管道卡住了") and "已自动停止这条管道" in text,
                text[:60],
            )
        guard.stop()

        calm = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        quiet = _PipeGuard([calm], "安静管道")
        beats = 0
        for _ in range(int(FAST_IDLE / FAST_POLL) + 3):
            time.sleep(FAST_POLL)
            quiet.beat()
            beats += 1
        check("unit-07 持续 beat 不会开火", not quiet.reason, f"beat {beats} 次")
        check("unit-08 beat 续命期间进程仍活着", calm.poll() is None)
        try:
            quiet.raise_if_stuck()
            check("unit-09 没开火时 raise_if_stuck 不抛", True)
        except VideoProcessError as exc:
            check("unit-09 没开火时 raise_if_stuck 不抛", False, str(exc)[:60])
        quiet.stop()
        calm.kill()
        calm.wait(timeout=10)

        late = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        stopped = _PipeGuard([late], "已收尾管道")
        stopped.stop()
        time.sleep(FAST_IDLE + FAST_POLL * 3)
        check("unit-10 stop() 之后不再开火", not stopped.reason and late.poll() is None)
        late.kill()
        late.wait(timeout=10)


# ---------------------------------------------------------------- 逐帧路线（真 ffmpeg）

def frame_route() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = run_engine(OUT_DIR / "_vt_base.mp4")
    base_frames = frames_of(OUT_DIR / "_vt_base.mp4")
    check(
        "frame-01 默认口径能正常出片",
        base_frames >= TOTAL_FRAMES - 2 and baseline["method"].startswith("frame-"),
        f"{base_frames} 帧 method={baseline['method']}",
    )

    # 对照组：闸门在场但从未触发——每帧都慢一点点，整趟远超 idle，却因为一直在出帧而不算卡住
    real_fill = video_ops.remove_boxes_inplace

    def slow_every_frame(frame, boxes, method):
        time.sleep(0.15)
        return real_fill(frame, boxes, method)

    started = time.monotonic()
    with patched(video_ops, "remove_boxes_inplace", slow_every_frame), fast_gate():
        dragged = run_engine(OUT_DIR / "_vt_slow_ok.mp4")
    drag_elapsed = time.monotonic() - started
    check(
        "frame-02 一直有动静就不算卡住（判动静不判总时长）",
        drag_elapsed > FAST_IDLE * 3,
        f"整趟 {drag_elapsed:.1f}s > idle {FAST_IDLE}s 仍跑完",
    )
    check(
        "frame-03 没触发时 method 与说明一字不变",
        dragged["method"] == baseline["method"] and dragged["note"] == baseline["note"],
        f"{dragged['method']} / {dragged['note'][:20]}…",
    )
    check(
        "frame-04 没触发时帧数不变",
        frames_of(OUT_DIR / "_vt_slow_ok.mp4") == base_frames,
        f"{frames_of(OUT_DIR / '_vt_slow_ok.mp4')} vs {base_frames}",
    )

    # 真卡死：第 3 帧的修复步骤睡死过去，看门狗必须杀掉两路 ffmpeg
    hung_state = {"calls": 0, "start": 0.0, "end": 0.0}

    def hang_once(frame, boxes, method):
        hung_state["calls"] += 1
        if hung_state["calls"] == 3:
            hung_state["start"] = time.monotonic()
            time.sleep(HANG_SECONDS)
            hung_state["end"] = time.monotonic()
        return real_fill(frame, boxes, method)

    recorded: list = []
    kill_times: list[float] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            recorded.append(self)

        def kill(self) -> None:
            kill_times.append(time.monotonic())
            super().kill()

    error: VideoProcessError | None = None
    with patched(video_ops, "remove_boxes_inplace", hang_once), patched_subprocess(
        video_ops, Popen=RecordingPopen
    ), fast_gate():
        try:
            run_engine(OUT_DIR / "_vt_hang.mp4")
        except VideoProcessError as exc:
            error = exc
    text = str(error) if error else ""
    check("frame-05 卡住会抛 VideoProcessError", error is not None, text[:48])
    check("frame-06 文案诚实写「卡住了」", "卡住了" in text and "逐帧修复" in text, text[:48])
    check(
        "frame-07 建议可换档位且不再冒「文件可能过大」",
        "换一种修复档位" in text and "文件可能过大" not in text,
        text[-22:],
    )
    check("frame-08 解码与编码两路 ffmpeg 都在场", len(recorded) >= 2, f"{len(recorded)} 路")
    check("frame-09 卡住的管道全被杀掉", wait_dead(recorded), f"{[p.poll() for p in recorded]}")
    check(
        "frame-10 ffmpeg 在代码还卡着时就被杀掉",
        bool(kill_times) and min(kill_times) < hung_state["end"],
        f"杀进程 {min(kill_times) - hung_state['start']:.1f}s vs 卡死 {HANG_SECONDS}s"
        if kill_times and hung_state["end"]
        else f"kill 次数={len(kill_times)}",
    )
    check(
        "frame-11 动手前等满了 idle 窗口（不一有慢帧就误杀）",
        bool(kill_times) and min(kill_times) - hung_state["start"] >= FAST_IDLE,
        f"卡住后 {min(kill_times) - hung_state['start']:.2f}s 开火（idle={FAST_IDLE}）"
        if kill_times
        else "没开火",
    )
    leftovers = [p.name for p in OUT_DIR.glob("_vt_hang*")]
    check(
        "frame-12 半成品与 .part 都没留下",
        not (OUT_DIR / "_vt_hang.mp4").exists() and not leftovers,
        str(leftovers),
    )

    recovered = run_engine(OUT_DIR / "_vt_after_hang.mp4")
    check(
        "frame-13 中止后同一份素材仍可重跑成功",
        frames_of(OUT_DIR / "_vt_after_hang.mp4") == base_frames
        and recovered["note"] == baseline["note"],
        f"{recovered['method']}",
    )


# ---------------------------------------------------------------- delogo 路线（管道级）

class _NullStream:
    def readline(self) -> bytes:
        return b""

    def close(self) -> None:
        pass


class FakeFfmpeg:
    """假 ffmpeg：按脚本吐几行 -progress，之后要么彻底没动静，要么直接 EOF＝进程退出。"""

    def __init__(
        self,
        lines: list[bytes],
        gap: float = 0.0,
        stall: bool = False,
        flush_hangs: bool = False,
    ) -> None:
        self.lines = lines
        self.gap = gap
        self.stall = stall
        self.flush_hangs = flush_hangs
        self.stderr = _NullStream()
        self.stdout = self
        self.gate = threading.Event()
        self.killed = False
        self._alive = True

    def __iter__(self):
        for line in self.lines:
            if self.gap:
                time.sleep(self.gap)
            yield line
        if self.stall:
            self.gate.wait()
        if not self.flush_hangs:
            self._alive = False

    def poll(self):
        return None if self._alive else (-9 if self.killed else 0)

    def kill(self) -> None:
        self.killed = True
        self._alive = False
        self.gate.set()

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0)
        return -9 if self.killed else 0


def delogo_route() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "_vt_delogo.mp4"
    out.write_bytes(b"fake-output")  # 只用来过「成品有效」那道校验，这一段测的是管道
    args = dict(
        rects=[(10, 10, 100, 40)],
        width=W,
        height=H,
        duration=float(SECONDS),
        on_progress=lambda _value: None,
        stream_args=[],
    )
    # ① 进度行彻底断流
    stalled = FakeFfmpeg([b"out_time_ms=100000\n"], stall=True)
    started = time.monotonic()
    with fast_gate(), patched_subprocess(video_ops, Popen=lambda *_a, **_k: stalled):
        try:
            video_ops._process_with_delogo(CLIP, out, **args)
            check("delogo-01 进度行断流会报错", False, "没有抛出")
        except VideoProcessError as exc:
            text = str(exc)
            check(
                "delogo-01 进度行断流会报错",
                "画面内插值修复卡住了" in text and "秒没有任何进展" in text,
                text[:48],
            )
    check("delogo-02 断流时 ffmpeg 被杀", stalled.killed, f"{time.monotonic() - started:.1f}s")

    # ② 一直有进度行 ⇒ 不该开火（哪怕累计时长比 idle 长好几倍）
    busy = FakeFfmpeg(
        [f"out_time_ms={index * 1000}\n".encode() for index in range(1, 9)], gap=FAST_POLL
    )
    with fast_gate(), patched_subprocess(video_ops, Popen=lambda *_a, **_k: busy):
        try:
            video_ops._process_with_delogo(CLIP, out, **args)
            check(
                "delogo-03 持续出进度不算卡住",
                not busy.killed and busy.poll() == 0,
                f"累计 {len(busy.lines) * FAST_POLL:.1f}s > idle {FAST_IDLE}s 仍跑完",
            )
        except VideoProcessError as exc:
            check("delogo-03 持续出进度不算卡住", False, str(exc)[:48])

    # ③ 装了闸门之后第十八轮的取消检查点仍然生效
    canceller = FakeFfmpeg([b"out_time_ms=100000\n", b"out_time_ms=200000\n"], stall=True)
    with fast_gate(), patched_subprocess(video_ops, Popen=lambda *_a, **_k: canceller):
        try:
            video_ops._process_with_delogo(CLIP, out, should_stop=lambda: True, **args)
            check("delogo-04 停止这一批仍优先于卡死报错", False, "没有抛出")
        except CancelRequested as exc:
            check("delogo-04 停止这一批仍优先于卡死报错", "已停止" in str(exc), str(exc)[:36])
        except VideoProcessError as exc:
            check("delogo-04 停止这一批仍优先于卡死报错", False, str(exc)[:48])

    # ④ 收尾（flush + faststart 搬 moov）卡住：看门狗此时已停，靠 PIPE_FLUSH_TIMEOUT 兜底
    flusher = FakeFfmpeg([b"out_time_ms=100000\n"], flush_hangs=True)
    with fast_gate(), patched_subprocess(video_ops, Popen=lambda *_a, **_k: flusher):
        try:
            video_ops._process_with_delogo(CLIP, out, **args)
            check("delogo-05 收尾卡住也会报错", False, "没有抛出")
        except VideoProcessError as exc:
            check(
                "delogo-05 收尾卡住也会报错",
                "卡住了" in str(exc) and "写出文件头" in str(exc),
                str(exc)[:48],
            )
    check("delogo-06 收尾卡住的进程也被收掉", flusher.poll() is not None)
    out.unlink(missing_ok=True)


# ---------------------------------------------------------------- media 兜底

def media_route() -> None:
    bogus = OUT_DIR / "_vt_broken.txt"
    bogus.write_text("not a video", encoding="utf-8")

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

    with patched_subprocess(media, run=raise_timeout):
        try:
            media.probe_video(bogus)
            check("media-01 探测超时也会诚实报错", False, "没有抛出")
        except MediaError as exc:
            check(
                "media-01 探测超时也会诚实报错",
                "读取视频信息超时或失败" in str(exc),
                str(exc)[:40],
            )
    bogus.unlink(missing_ok=True)
    probe = media.probe_video(CLIP)
    check(
        "media-02 正常素材探测不受影响",
        int(probe["frame_count"]) >= TOTAL_FRAMES - 2 and probe["width"] == W,
        f"{probe['frame_count']} 帧 {probe['width']}x{probe['height']}",
    )


def main() -> int:
    print(f"clip={CLIP} idle={DEFAULT_IDLE} poll={DEFAULT_POLL} flush={DEFAULT_FLUSH}")
    unit_guard()
    if not UNIT_ONLY:
        build_clip()
        frame_route()
    delogo_route()
    if not UNIT_ONLY:
        media_route()
    if not KEEP and OUT_DIR.exists():
        for item in sorted(OUT_DIR.glob("*")):
            item.unlink(missing_ok=False)
        OUT_DIR.rmdir()
        if CLIP.exists():
            CLIP.unlink()
    print(f"共 {len(FAILED)} 项失败" + ("" if FAILED else "（全部通过）"))
    if FAILED:
        print("失败项：" + "、".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
