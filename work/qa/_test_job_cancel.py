# -*- coding: utf-8 -*-
"""「停止这一批」回归：引擎检查点、批处理闸门与 HTTP 契约。

第十八轮新增。判据是「停止之后：已完成的成品还在、半成品一个都不留、排队中的不再开工、
停止不算失败」。素材全合成，不读用户作品。

跑法（需要服务在跑，且是含 cancel 路由的新代码）：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_job_cancel.py
只跑引擎部分：加 --unit
保留夹具便于肉眼复看：加 --keep
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
QA = BASE / "work" / "qa"
OUT_DIR = QA / "_jc_out"

from remover import batch_runner, media, video_ops  # noqa: E402
from remover.cancel import CancelRequested  # noqa: E402

W, H, FPS, SECONDS = 480, 270, 30, 10
CLIP = QA / "_jc_marked.mp4"
PAGE = QA / "_jc_page.png"
FONT = "C\\\\:/Windows/Fonts/arialbd.ttf"
# 贴右下角的白字台标：画质优先必须走逐帧，速度优先内缩后走 delogo —— 两条中断点都要覆盖
REGION = {"x": (W - 150) / W, "y": (H - 44) / H, "w": 150 / W, "h": 44 / H}
IMAGE_REGION = {"x": 0.02, "y": 0.02, "w": 0.95, "h": 0.24}
FAILED: list[str] = []
KEEP = "--keep" in sys.argv
UNIT_ONLY = "--unit" in sys.argv

JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JAR))
TOKEN = ""


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_clip() -> None:
    """480x270 / 10s 合成片段：移动的分形底 + 右下贴边白字台标 + 一条真音轨。"""
    if CLIP.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    base = (
        f"gradients=size={W}x{H}:rate={FPS}:c0=0x203040:c1=0x4a3a55:c2=0x2e4a3b:"
        f"c3=0x55402a:x0=0:y0=0:x1={W}:y1={H}:duration={SECONDS}"
    )
    inputs = [
        "-f", "lavfi", "-i", base,
        "-f", "lavfi", "-i", f"mandelbrot=size={W // 2}x{H // 2}:rate={FPS}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SECONDS}",
    ]
    mark = (
        f"drawtext=fontfile={FONT}:text='LIVE TV':fontcolor=white@0.82:fontsize=30:"
        f"borderw=2:bordercolor=black@0.55:x=w-tw-2:y=h-th-2"
    )
    vf = (
        f"[0:v][1:v]overlay=x='mod(t*90\\,{W})-{W // 2}':y='mod(t*60\\,{H})-{H // 2}'"
        f":eof_action=repeat,{mark}[v]"
    )
    tail = [
        "-t", str(SECONDS), "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
    ]
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
         "-filter_complex", vf, *tail, str(CLIP)],
        check=True,
    )


def build_page() -> None:
    """900x600 合成「页面」：斜向排布的半透明白字条，只为走图片路线。"""
    if PAGE.exists():
        return
    ys, xs = np.mgrid[0:600, 0:900]
    canvas = np.stack(
        [
            (200 - 60 * ys / 600).astype(np.uint8),
            (190 - 40 * xs / 900).astype(np.uint8),
            (175 - 20 * (xs + ys) / 1500).astype(np.uint8),
        ],
        axis=-1,
    )
    img = Image.fromarray(canvas, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    for row in range(6):
        for col in range(4):
            draw.text(
                (col * 240 + row * 30 - 40, row * 110 + 20),
                "样张水印",
                fill=(255, 255, 255, 170),
            )
    img.convert("RGB").save(PAGE)


def image_item(item_id: str, target: Path) -> batch_runner.BatchItem:
    return batch_runner.BatchItem(
        item_id=item_id,
        kind="image",
        input_path=PAGE,
        output_path=target,
        regions=[IMAGE_REGION],
        width=900,
        height=600,
    )


def run_engine(target: Path, **kwargs):
    """跑一次视频引擎，返回 (结果或异常, 耗时)。"""
    started = time.perf_counter()
    try:
        detail = video_ops.process_video_file(CLIP, target, [REGION], W, H, **kwargs)
        return detail, time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - 异常本身就是观测对象
        return exc, time.perf_counter() - started


def unit_phase() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # 1. 未开工即停
    result, _ = run_engine(OUT_DIR / "u1.mp4", method="inpaint", should_stop=lambda: True)
    check("引擎：开工前停止 → CancelRequested", isinstance(result, CancelRequested), repr(result)[:70])
    check("引擎：停止原因写明「尚未开始」", isinstance(result, CancelRequested) and "尚未开始" in str(result))
    check(
        "引擎：开工前停止不留成品与残档",
        not (OUT_DIR / "u1.mp4").exists() and not (OUT_DIR / "u1.part.mp4").exists(),
    )

    # 2. 时序掩膜采样阶段的中断点（画质优先开工前多花的那一两秒也得能停）
    try:
        video_ops._sample_gray_frames(CLIP, W, H, float(SECONDS), lambda: True)
        check("引擎：时序采样停止 → CancelRequested", False, "没有抛出")
    except CancelRequested as exc:
        check("引擎：时序采样停止 → CancelRequested", True, str(exc))

    # 3. delogo（速度优先）：进度回调在插值路线的第一个 out_time_ms 就该被看见
    ticks: list[float] = []
    flag = threading.Event()

    def mark_progress(value: float) -> None:
        ticks.append(value)
        flag.set()

    result3, elapsed3 = run_engine(
        OUT_DIR / "u3.mp4",
        method="smart",
        video_mode="fast",
        on_progress=mark_progress,
        should_stop=flag.is_set,
    )
    check(
        "引擎：速度优先插值在进度处停止 → CancelRequested",
        isinstance(result3, CancelRequested),
        f"{type(result3).__name__}: {str(result3)[:50]}｜进度回调 {len(ticks)} 次｜{elapsed3:.2f}s",
    )
    check("引擎：插值停止的说明写明路线", isinstance(result3, CancelRequested) and "插值" in str(result3))
    check(
        "引擎：插值中断后无成品无 .part",
        not (OUT_DIR / "u3.mp4").exists() and not (OUT_DIR / "u3.part.mp4").exists(),
    )

    # 4. 逐帧（画质优先）：报告点每 0.25s 一次，中断必须远早于跑完
    ticks2: list[float] = []
    flag2 = threading.Event()

    def mark_progress2(value: float) -> None:
        ticks2.append(value)
        flag2.set()

    result4, elapsed4 = run_engine(
        OUT_DIR / "u4.mp4",
        method="inpaint",
        video_mode="quality",
        on_progress=mark_progress2,
        should_stop=flag2.is_set,
    )
    check(
        "引擎：逐帧修复在报告点停止 → CancelRequested",
        isinstance(result4, CancelRequested),
        f"{type(result4).__name__}: {str(result4)[:50]}｜进度回调 {len(ticks2)} 次｜{elapsed4:.2f}s",
    )
    check("引擎：逐帧停止的说明写明路线", isinstance(result4, CancelRequested) and "逐帧" in str(result4))
    check(
        "引擎：逐帧中断后无成品无 .part",
        not (OUT_DIR / "u4.mp4").exists() and not (OUT_DIR / "u4.part.mp4").exists(),
    )

    # 5. 不点停止时行为不变：should_stop=None 与永远 False 必须同一条路线、同样的帧数
    full_a, time_a = run_engine(OUT_DIR / "u5a.mp4", method="inpaint", video_mode="quality")
    full_b, time_b = run_engine(
        OUT_DIR / "u5b.mp4", method="inpaint", video_mode="quality", should_stop=lambda: False
    )
    both_ok = isinstance(full_a, dict) and isinstance(full_b, dict)
    check("引擎：闸门存在但从未触发时照样出成品", both_ok, repr(full_b)[:70])
    check(
        "引擎：闸门不影响路线与说明",
        both_ok and full_a["method"] == full_b["method"] and full_a["note"] == full_b["note"],
        f"{full_a.get('method') if both_ok else '?'} / {full_b.get('method') if both_ok else '?'}",
    )
    frames_a = media.probe_video_length(OUT_DIR / "u5a.mp4").get("frames")
    frames_b = media.probe_video_length(OUT_DIR / "u5b.mp4").get("frames")
    check("引擎：两次成品的帧数一致", bool(frames_a) and frames_a == frames_b, f"{frames_a} vs {frames_b}")
    check("引擎：中断比跑完整段更早结束", isinstance(result4, CancelRequested) and elapsed4 < time_a,
          f"中断 {elapsed4:.2f}s vs 完整 {time_a:.2f}s")
    same_bytes = both_ok and md5(OUT_DIR / "u5a.mp4") == md5(OUT_DIR / "u5b.mp4")
    print(f"     （逐比特一致：{same_bytes}——libx264 跨线程不保证，只作参考；两遍完整耗时 {time_a:.1f}s / {time_b:.1f}s）")

    # 6. run_batch：闸门一开始就置位 → 一份都不开工
    missing = QA / "_jc_definitely_missing.mp4"
    pre_items = [
        batch_runner.BatchItem(
            item_id=f"pre{i}",
            kind="video",
            input_path=missing,
            output_path=OUT_DIR / f"pre{i}.mp4",
            regions=[REGION],
            width=W,
            height=H,
        )
        for i in range(3)
    ]
    batch_runner.run_batch(pre_items, workers=2, should_stop=lambda: True)
    check(
        "批处理：停止已置位时全部记 cancelled",
        all(item.cancelled and item.error is None for item in pre_items),
        f"cancelled={sum(1 for i in pre_items if i.cancelled)} error={sum(1 for i in pre_items if i.error)}",
    )
    check(
        "批处理：cancelled 的份没成品、没异常、也没调用引擎",
        all(
            item.result is None
            and item.trace == ""
            and item.error is None
            and not item.output_path.exists()
            for item in pre_items
        ),
        "引擎若被调用，不存在的输入必然留下 error",
    )

    # 7. run_batch：跑完第一份才置位 → 后续不再开工，已完成的照旧保留
    stop_now = threading.Event()
    events: list[tuple[str, str]] = []

    def on_event(item: batch_runner.BatchItem, phase: str, value: float) -> None:
        events.append((item.item_id, phase))
        if phase == "done":
            stop_now.set()

    image_items = [image_item(f"img{i}", OUT_DIR / f"img{i}.png") for i in range(3)]
    workers_used = batch_runner.run_batch(image_items, workers=1, on_event=on_event, should_stop=stop_now.is_set)
    check(
        "批处理：单路时先做完一份再停",
        image_items[0].result is not None and image_items[0].output_path.exists(),
        f"workers={workers_used}",
    )
    check(
        "批处理：后两份被跳过（cancelled，不是 error）",
        all(item.cancelled and item.error is None and item.result is None for item in image_items[1:]),
        f"cancelled={sum(1 for i in image_items[1:] if i.cancelled)}",
    )
    check(
        "批处理：跳过的份不落盘",
        not (OUT_DIR / "img1.png").exists() and not (OUT_DIR / "img2.png").exists(),
    )
    check("批处理：items 顺序仍是提交顺序", [item.item_id for item in image_items] == ["img0", "img1", "img2"])
    check(
        "批处理：每份都有 done 事件（含被跳过的）",
        sum(1 for _item, phase in events if phase == "done") == 3,
        f"{len(events)} 个事件",
    )

    # 8. 停止闸门不得吃掉失败隔离：坏件仍然是 error，好件仍然是 result
    mixed = [image_item("good", OUT_DIR / "good.png")]
    mixed.append(
        batch_runner.BatchItem(
            item_id="broken",
            kind="image",
            input_path=QA / "_jc_broken_input.png",
            output_path=OUT_DIR / "broken.png",
            regions=[IMAGE_REGION],
            width=900,
            height=600,
        )
    )
    batch_runner.run_batch(mixed, workers=1, should_stop=lambda: False)
    check(
        "批处理：坏件仍记 error、不是 cancelled",
        mixed[1].error is not None and not mixed[1].cancelled,
        f"error={type(mixed[1].error).__name__ if mixed[1].error else '无'}",
    )
    check("批处理：同批好件不受影响", mixed[0].result is not None and not mixed[0].cancelled)

    # 9. 完全没有闸门时（老调用方式）三份都出成品
    plain_items = [image_item(f"ok{i}", OUT_DIR / f"ok{i}.png") for i in range(3)]
    batch_runner.run_batch(plain_items, workers=1)
    check(
        "批处理：不传 should_stop 时全部成功",
        all(item.result is not None and not item.cancelled and not item.error for item in plain_items),
    )
    leftovers = sorted(p.name for p in OUT_DIR.glob("*part*"))
    check("引擎：整个过程没有 .part 残档", not leftovers, str(leftovers[:4]))


def request_bytes(url: str, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with OPENER.open(req, timeout=600) as resp:
        return resp.read(), resp.status


def get_json(url: str):
    return json.loads(request_bytes(url)[0])


def post_json(root: str, url: str, payload: dict):
    body, _ = request_bytes(
        root + url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-App-Token": TOKEN},
    )
    return json.loads(body)


def post_expect_error(root: str, url: str, payload: dict | None = None, headers: dict | None = None):
    """POST 期望失败，返回 (状态码, 响应正文片段)。成功则返回 (None, "")。"""
    merged = {"Content-Type": "application/json", **(headers or {})}
    if "X-App-Token" not in merged:
        merged.pop("X-App-Token", None)
    try:
        request_bytes(
            root + url,
            data=json.dumps(payload or {}).encode("utf-8"),
            headers=merged,
            method="POST",
        )
        return None, ""
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")[:120] if exc.fp else ""
        return exc.code, text


def upload_clips(root: str, count: int) -> list[str]:
    boundary = "----qaCancelBoundary"
    payload = CLIP.read_bytes()
    parts: list[bytes] = []
    for index in range(count):
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="files"; filename="台标片段 {index + 1}.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n".encode(),
            payload + b"\r\n",
        ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    result = json.loads(
        request_bytes(
            root + "/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": TOKEN},
        )[0]
    )
    return [item["id"] for item in result["items"]]


def wait_terminal(root: str, job_id: str, seconds: float = 150.0) -> dict:
    job: dict = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = get_json(f"{root}/api/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    return job


def http_phase(root: str) -> None:
    global TOKEN
    page, _ = request_bytes(root + "/")
    match = re.search(rb'name="app-token" content="([^"]+)"', page)
    if not match:
        check("HTTP：取到 app-token", False)
        return
    TOKEN = match.group(1).decode()
    health = get_json(root + "/api/health")
    check("HTTP：服务健康且是新代码（有 cancel 路由）", health.get("app") == "jingyin", str(health.get("instance")))
    code, _ = post_expect_error(root, "/api/jobs/nonexistent/cancel", {}, {"X-App-Token": TOKEN})
    check("HTTP：cancel 路由已注册（未知任务回 404 而非 405）", code == 404, f"code={code}")

    item_ids = upload_clips(root, 6)
    check("HTTP：上传 6 份合成片段", len(item_ids) == 6, str(len(item_ids)))
    created_dirs: list[Path] = []
    try:
        job = post_json(
            root,
            "/api/process",
            {
                "items": [{"id": item_id, "regions": [REGION]} for item_id in item_ids],
                "method": "smart",
                "video_mode": "quality",
            },
        )
        created_dirs.append(Path(job["output_dir"]))
        job_id = job["id"]
        check("HTTP：任务创建", job["status"] in ("queued", "running"), job["status"])

        code, body = post_expect_error(root, f"/api/jobs/{job_id}/cancel")
        check("HTTP：不带 token 停止 → 403", code == 403, f"code={code} {body[:40]}")

        # 等第一份成品出来再按停止：这样「已完成的保留」才是真被验证的
        first_done: dict | None = None
        deadline = time.time() + 120
        while time.time() < deadline:
            snapshot = get_json(f"{root}/api/jobs/{job_id}")
            if snapshot["success_count"] >= 1:
                first_done = snapshot
                break
            time.sleep(0.2)
        check("HTTP：停止前已有成品", first_done is not None, str(first_done and first_done["success_count"]))
        pending_before = {t["id"] for t in (first_done or {}).get("tasks", []) if t["status"] == "pending"}
        done_before = {t["id"] for t in (first_done or {}).get("tasks", []) if t["status"] == "done"}
        running_before = {t["id"] for t in (first_done or {}).get("tasks", []) if t["status"] == "processing"}

        cancelled_at = post_json(root, f"/api/jobs/{job_id}/cancel", {})
        check("HTTP：停止请求被受理", cancelled_at["cancel_requested"] is True, cancelled_at["status"])
        final = wait_terminal(root, job_id)
        check("HTTP：最终状态 cancelled", final["status"] == "cancelled", final["status"])
        check(
            "HTTP：completed 把 cancelled 计入（进度能走到头）",
            final["completed"] == final["total"],
            f"{final['completed']}/{final['total']}",
        )
        check(
            "HTTP：三项计数之和 = 总份数",
            final["success_count"] + final["failed_count"] + final["cancelled_count"] == final["total"],
            f"成功 {final['success_count']} · 停止 {final['cancelled_count']} · 失败 {final['failed_count']} / {final['total']}",
        )
        by_id = {t["id"]: t["status"] for t in final["tasks"]}
        leaked = [tid for tid in pending_before if by_id.get(tid) != "cancelled"]
        check("HTTP：停止时排队的份全部 cancelled", not leaked, str(leaked[:4]))
        moved = [tid for tid in done_before if by_id.get(tid) != "done"]
        check("HTTP：停止前已完成的份仍是 done", not moved, str(moved[:4]))
        check(
            "HTTP：results 份数 = success_count",
            len(final["results"]) == final["success_count"],
            f"{len(final['results'])} vs {final['success_count']}",
        )
        order = [t["id"] for t in final["tasks"] if t["status"] == "done"]
        placed = [r["task_id"] for r in final["results"]]
        check("HTTP：results 仍按任务原序落位", placed == order, f"{placed} vs {order}")
        check("HTTP：说明是「已停止」口径", final["message"].startswith("已停止"), final["message"][:60])
        check("HTTP：cancelled_count 已回显", final["cancelled_count"] >= 1, str(final["cancelled_count"]))
        check(
            "HTTP：被停止的份没有错误文案",
            all(t.get("error") is None for t in final["tasks"] if t["status"] == "cancelled"),
        )

        # 界面上那份「还没做完的文件」清单要能区分「跑一半被停」与「排队时被停」，
        # 而两条停止路径都把 progress 写成 1.0，所以只能靠 started 这个标记。
        started_by_id = {t["id"]: t.get("started") for t in final["tasks"]}
        check(
            "HTTP：tasks 回显 started 且一定是布尔",
            all(isinstance(v, bool) for v in started_by_id.values()),
            str(sorted({repr(v) for v in started_by_id.values()})),
        )
        check(
            "HTTP：停止时排队中的份 started 为假",
            all(started_by_id.get(tid) is False for tid in pending_before),
            str([tid for tid in pending_before if started_by_id.get(tid) is not False][:4]),
        )
        half = [tid for tid in running_before if by_id.get(tid) == "cancelled"]
        check(
            "HTTP：停止时正在处理的那份 started 为真",
            all(started_by_id.get(tid) is True for tid in half),
            f"{len(half)} 份跑一半被停：{[started_by_id.get(t) for t in half]}",
        )
        check(
            "HTTP：视频单路 ⇒ 跑一半被停的不超过 1 份",
            len([v for v in started_by_id.values() if v]) <= 1,
            str(sum(1 for v in started_by_id.values() if v)),
        )
        missing_ids = {t["id"] for t in final["tasks"] if t["status"] in ("cancelled", "error")}
        check(
            "HTTP：清单口径「没有成品的份」= cancelled + error",
            missing_ids == {t["id"] for t in final["tasks"]} - {r["task_id"] for r in final["results"]},
            f"缺 {len(missing_ids)} 份 / 共 {final['total']} 份",
        )

        out_dir = Path(final["output_dir"])
        parts = sorted(p.name for p in out_dir.glob("*part*")) if out_dir.exists() else ["目录不存在"]
        check("HTTP：成品目录里没有半截文件", not parts, str(parts[:4]))
        existing = sorted(p.name for p in out_dir.glob("*.mp4")) if out_dir.exists() else []
        check("HTTP：磁盘上的成品数 = 成功份数", len(existing) == final["success_count"], f"{len(existing)} 个")

        zip_bytes, zip_code = request_bytes(f"{root}/api/jobs/{job_id}/zip")
        names: list[str] = []
        bad = "打包请求失败"
        if zip_code == 200:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                names = archive.namelist()
                bad = archive.testzip()
        check("HTTP：停止后已完成的成品仍可打包下载", zip_code == 200 and bad is None, f"code={zip_code} bad={bad}")
        check(
            "HTTP：ZIP 条目与 results 一致",
            sorted(names) == sorted(r["name"] for r in final["results"]),
            f"{len(names)} 条 vs {len(final['results'])} 条",
        )

        code, body = post_expect_error(root, f"/api/jobs/{job_id}/cancel", {}, {"X-App-Token": TOKEN})
        check("HTTP：结束后再点停止 → 409", code == 409, f"code={code} {body[:40]}")

        # 立刻停（第一份还在跑）：没有成品时打包必须 409
        fresh = post_json(
            root,
            "/api/process",
            {"items": [{"id": item_ids[0], "regions": [REGION]}], "method": "smart", "video_mode": "quality"},
        )
        created_dirs.append(Path(fresh["output_dir"]))
        stopped = post_json(root, f"/api/jobs/{fresh['id']}/cancel", {})
        check("HTTP：新建任务可立即停止", stopped["cancel_requested"] is True, stopped["status"])
        final2 = wait_terminal(root, fresh["id"])
        check("HTTP：立即停止后状态 cancelled", final2["status"] == "cancelled", final2["status"])
        check(
            "HTTP：立即停止的一份记 cancelled 不算失败",
            final2["failed_count"] == 0 and final2["cancelled_count"] + final2["success_count"] == 1,
            f"停止 {final2['cancelled_count']} 成功 {final2['success_count']} 失败 {final2['failed_count']}",
        )
        if final2["results"]:
            check("HTTP：抢跑成功时 zip_ready 为真", final2["zip_ready"] is True, str(final2["zip_ready"]))
        else:
            check("HTTP：没有成品时 zip_ready 为假", final2["zip_ready"] is False, str(final2["zip_ready"]))
            try:
                request_bytes(f"{root}/api/jobs/{fresh['id']}/zip")
                zip2_code = 200
            except urllib.error.HTTPError as exc:
                zip2_code = exc.code
            check("HTTP：没有成品时打包回 409", zip2_code == 409, f"code={zip2_code}")
            out2 = Path(final2["output_dir"])
            check(
                "HTTP：立即停止不留任何成品",
                out2.exists() and not list(out2.glob("*.mp4")),
                str([p.name for p in out2.glob("*")][:3]) if out2.exists() else "目录没建",
            )

        # 不按停止时终态不变
        normal = post_json(
            root,
            "/api/process",
            {"items": [{"id": item_ids[1], "regions": [REGION]}], "method": "smart", "video_mode": "fast"},
        )
        created_dirs.append(Path(normal["output_dir"]))
        final3 = wait_terminal(root, normal["id"])
        check(
            "HTTP：不按停止时仍是 done、cancelled_count 为 0",
            final3["status"] == "done" and final3["cancelled_count"] == 0 and final3["cancel_requested"] is False,
            f"{final3['status']} 停止={final3['cancelled_count']}",
        )
    finally:
        for folder in created_dirs:
            if folder.parent == BASE / "成品":
                shutil.rmtree(folder, ignore_errors=True)
        uploads = BASE / "work" / "uploads"
        for item_id in item_ids:
            for item_dir in uploads.glob(f"*/{item_id}"):
                shutil.rmtree(item_dir, ignore_errors=True)
        for sid_dir in uploads.iterdir():
            if sid_dir.is_dir() and not any(sid_dir.iterdir()):
                sid_dir.rmdir()


def main() -> int:
    build_clip()
    build_page()
    unit_phase()
    if not UNIT_ONLY:
        runtime = BASE / "work" / "runtime.json"
        if not runtime.exists():
            check("HTTP：runtime.json 存在（服务没在跑）", False)
        else:
            port = json.loads(runtime.read_text(encoding="utf-8"))["port"]
            http_phase(f"http://127.0.0.1:{port}")
    if not KEEP:
        CLIP.unlink(missing_ok=True)
        PAGE.unlink(missing_ok=True)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("\n夹具与沙箱已清理。保留夹具请加 --keep；只跑引擎请加 --unit")
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
