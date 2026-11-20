from __future__ import annotations

import atexit
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file, session
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.serving import make_server

from remover import (
    batch_runner,
    detection,
    image_ops,
    media,
    options,
    photo_inpaint,
    state_store,
    video_ops,
)

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
UPLOAD_ROOT = WORK_DIR / "uploads"
ZIP_DIR = WORK_DIR / "zips"
OUTPUT_ROOT = BASE_DIR / "成品"
RUNTIME_FILE = WORK_DIR / "runtime.json"
SESSION_KEY_FILE = WORK_DIR / "secret.key"
LOG_FILE = WORK_DIR / "tool.log"
INDEX_LOCK = threading.RLock()
MAX_BATCH_ITEMS = 120
MAX_REGIONS_PER_ITEM = 16
PORT_RANGE = range(8765, 8786)
JOBS: dict[str, dict] = {}
SESSIONS: dict[str, dict] = {}
SERVER_HOLDER: dict = {}

APP_TOKEN = secrets.token_urlsafe(32)
# 每次进程启动换一个身份；探活时要对得上，否则说明端口上应答的是另一份工具
INSTANCE_ID = uuid.uuid4().hex
# 只放目录指纹不放路径：用来区分「同一台机器上的另一份拷贝」
DIR_TAG = hashlib.sha1(str(BASE_DIR).encode("utf-8")).hexdigest()[:16]
def load_session_secret() -> str:
    """给会话 cookie 签名用的密钥固定落盘：重启服务后旧 cookie 仍然有效，素材列表才回得来。"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        saved = SESSION_KEY_FILE.read_text(encoding="utf-8").strip()
        if len(saved) == 64:
            return saved
    except OSError:
        pass
    secret = secrets.token_hex(32)
    tmp = WORK_DIR / "secret.key.part"
    try:
        tmp.write_text(secret, encoding="utf-8")
        tmp.replace(SESSION_KEY_FILE)
    except OSError:
        tmp.unlink(missing_ok=True)
    return secret


app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)
app.secret_key = load_session_secret()
# 关掉浏览器再打开仍然是同一个工作区；密钥不落盘的话重启服务就会把 cookie 判成无效
app.permanent_session_lifetime = timedelta(days=30)
app.config.update(
    MAX_CONTENT_LENGTH=8 * 1024**3,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    JSON_AS_ASCII=False,
)


def setup_logging() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(formatter)
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    if sys.stdout is not None and os.environ.get("JINGYIN_NO_CONSOLE_LOG") != "1":
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        app.logger.addHandler(stream)


def get_session_store() -> dict:
    sid = session["sid"]
    with INDEX_LOCK:
        store = SESSIONS.get(sid)
        if store is None:
            store = {"items": {}, "created": time.time()}
            SESSIONS[sid] = store
        return store


def public_item(item: dict) -> dict:
    return {
        "id": item["id"],
        "name": item["name"],
        "kind": item["kind"],
        "width": item["width"],
        "height": item["height"],
        "duration": item.get("duration"),
        "suffix": item["suffix"],
        "thumb_url": f"/api/items/{item['id']}/thumb",
        "display_url": f"/api/items/{item['id']}/display",
        "playable": item["kind"] == "image" or item["suffix"] in media.BROWSER_VIDEO_EXTENSIONS,
    }


def find_item(item_id: str) -> dict | None:
    with INDEX_LOCK:
        return get_session_store()["items"].get(item_id)


def public_job(job: dict) -> dict:
    with INDEX_LOCK:
        item_by_task = {task["id"]: task.get("item_id") for task in job.get("tasks", [])}
        results = []
        for result in job.get("results", []):
            results.append(
                {
                    "id": result["id"],
                    "task_id": result["task_id"],
                    # 界面靠它对上「这份素材现在画面上还留着哪些框」，单份重跑要用
                    "item_id": item_by_task.get(result["task_id"]),
                    "name": result["name"],
                    "kind": result["kind"],
                    "size": result["size"],
                    "note": result.get("note", ""),
                    "preview_url": f"/api/jobs/{job['id']}/results/{result['id']}/preview",
                    "download_url": f"/api/jobs/{job['id']}/results/{result['id']}/file?download=1",
                    "before_url": result.get("before_url"),
                }
            )
        tasks = [
            {
                "id": task["id"],
                "name": task["name"],
                "kind": task["kind"],
                "status": task["status"],
                "progress": round(float(task.get("progress", 0)), 4),
                "started": bool(task.get("started", False)),
                "error": task.get("error"),
            }
            for task in job.get("tasks", [])
        ]
        status = job["status"]
        total = len(tasks)
        completed = sum(1 for task in tasks if task["status"] in ("done", "error", "cancelled"))
        remaining = sum(1 for task in tasks if task["status"] != "done")
        return {
            "id": job["id"],
            "status": status,
            "progress": round(float(job.get("progress", 0)), 4),
            "message": job.get("message", ""),
            "tasks": tasks,
            "total": total,
            "completed": completed,
            "success_count": job.get("success_count", 0),
            "failed_count": job.get("failed_count", 0),
            "cancelled_count": job.get("cancelled_count", 0),
            "cancel_requested": bool(job.get("cancel_requested", False)),
            "resumable": status in ("cancelled", "failed") and remaining > 0,
            "remaining_count": remaining,
            "resume_count": int(job.get("resume_count", 0)),
            "results": results,
            "video_mode": options.normalize_video_mode(job.get("video_mode")),
            "motion": options.normalize_motion(job.get("motion")),
            "output_dir": str(job["output_dir"]),
            "zip_ready": bool(results) and status in ("done", "failed", "cancelled"),
        }


def update_job(job_id: str, **changes) -> None:
    with INDEX_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(changes)


def update_task(job_id: str, task_id: str, **changes) -> None:
    with INDEX_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for task in job["tasks"]:
            if task["id"] == task_id:
                task.update(changes)
                break


class RegionLimitError(ValueError):
    """选框数量超上限：必须让调用方报错，不能静默丢掉第 17 个以后的框。"""


def normalize_regions(raw_regions, owner: str = "") -> list[dict]:
    if not isinstance(raw_regions, list):
        return []
    if len(raw_regions) > MAX_REGIONS_PER_ITEM:
        where = f"「{owner}」" if owner else "这份素材"
        raise RegionLimitError(
            f"{where}有 {len(raw_regions)} 个选框，每份素材最多 {MAX_REGIONS_PER_ITEM} 个，"
            "请把相邻的框合并成一个大框后再处理"
        )
    regions: list[dict] = []
    for region in raw_regions:
        if not isinstance(region, dict):
            continue
        try:
            x = float(region.get("x"))
            y = float(region.get("y"))
            width = float(region.get("w"))
            height = float(region.get("h"))
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(value) for value in (x, y, width, height)):
            continue
        x = min(max(x, 0.0), 0.999)
        y = min(max(y, 0.0), 0.999)
        width = min(max(width, 0.0), 1.0 - x)
        height = min(max(height, 0.0), 1.0 - y)
        if width < 0.0015 or height < 0.0015:
            continue
        regions.append(
            {
                "x": round(x, 6),
                "y": round(y, 6),
                "w": round(width, 6),
                "h": round(height, 6),
            }
        )
    return regions


def user_error(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            media.MediaError,
            image_ops.ImageProcessError,
            video_ops.VideoProcessError,
        ),
    ):
        return str(exc)
    return "处理失败，请换一种修复方式或调整选框后重试"


def _make_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = OUTPUT_ROOT / f"去水印_{stamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"去水印_{stamp}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _free_output_name(output_dir: Path, name: str, taken: set[str]) -> str:
    """继续处理时沿用原成品名；只有磁盘上真存在同名文件（比如用户自己放进来的）才让开。"""
    stem, dot, suffix = name.rpartition(".")
    if not dot or not stem:
        stem, suffix = name, ""
    else:
        suffix = f".{suffix}"
    candidate = name
    counter = 2
    while candidate in taken or (output_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _job_active_for_session(sid: str) -> bool:
    with INDEX_LOCK:
        return any(
            job.get("session_id") == sid and job.get("status") in ("queued", "running")
            for job in JOBS.values()
        )


def _run_job(job_id: str) -> None:
    with INDEX_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        tasks = job["tasks"]
        output_dir = job["output_dir"]
        total = len(tasks)
        cancel_event = job.get("cancel_event")
        carried = [task for task in tasks if task.get("status") == "done"]
        carried_ids = {task["id"] for task in carried}
        if carried:
            job["message"] = "正在继续处理剩下的素材"
        else:
            job["message"] = "正在准备处理"
        results_by_task = {result["task_id"]: result for result in job.get("results", [])}

    app.logger.info("Job %s started with %s task(s)", job_id, total)
    # 每份素材的进度分别记账，作业总进度 = 各份完成度之和 / 总数（图片一路 0→1，视频按帧推进）
    # 「继续处理」时已完成的那几份直接从任务状态里还原，不再进批处理队列
    done = len(carried)
    success_count = len(carried)
    failed_count = 0
    cancelled_count = 0
    partial = {task["id"]: (1.0 if task["id"] in carried_ids else 0.0) for task in tasks}
    index_by_id = {task["id"]: index for index, task in enumerate(tasks)}
    slots: list[dict | None] = [None] * total
    for task_id in carried_ids:
        carried_result = results_by_task.get(task_id)
        if carried_result is not None:
            slots[index_by_id[task_id]] = carried_result
    meta: dict[str, tuple[str, Path, Path | None]] = {}
    items: list[batch_runner.BatchItem] = []
    for task in tasks:
        if task["id"] in carried_ids:
            continue
        result_id = uuid.uuid4().hex[:12]
        output_path = output_dir / task["output_name"]
        # 图片成品另存一份 2400px JPEG 预览；视频预览直接用成品文件本身
        preview_path = output_dir / f".preview_{result_id}.jpg" if task["kind"] == "image" else None
        meta[task["id"]] = (result_id, output_path, preview_path)
        items.append(
            batch_runner.BatchItem(
                item_id=task["id"],
                kind=task["kind"],
                input_path=task["source_path"],
                output_path=output_path,
                regions=task["regions"],
                method=task["method"],
                video_mode=options.normalize_video_mode(task.get("video_mode")),
                motion=options.normalize_motion(task.get("motion")),
                width=task["width"],
                height=task["height"],
                preview_path=preview_path,
            )
        )

    def on_event(item: batch_runner.BatchItem, phase: str, value: float) -> None:
        nonlocal done, success_count, failed_count, cancelled_count
        index = index_by_id[item.item_id]
        task = tasks[index]
        with INDEX_LOCK:
            current_job = JOBS.get(job_id)
            if not current_job:
                return
            if phase == "start":
                task["status"] = "processing"
                task["progress"] = 0.01
                partial[item.item_id] = 0.01
                current_job["message"] = f"已完成 {done}/{total} · 正在处理：{task['name']}"
            elif phase == "tick":
                task["progress"] = round(float(value), 4)
                partial[item.item_id] = float(value)
            else:
                done += 1
                partial[item.item_id] = 1.0
                task["progress"] = 1.0
                result_id, output_path, preview_path = meta[item.item_id]
                if item.cancelled:
                    # 停止掉的份数：没有成品，也不算失败——它不进失败计数与错误文案
                    # 两条停止路径都会把 progress 写成 1.0，所以「有没有开过工」只能看是否发过 start
                    task["started"] = task["status"] == "processing"
                    task["status"] = "cancelled"
                    cancelled_count += 1
                    current_job["cancelled_count"] = cancelled_count
                elif item.error is None:
                    slots[index] = {
                        "id": result_id,
                        "task_id": task["id"],
                        "name": task["output_name"],
                        "kind": task["kind"],
                        "file_path": output_path,
                        "preview_path": preview_path or output_path,
                        "size": output_path.stat().st_size,
                        "note": item.result["note"],
                        "before_url": f"/api/items/{task['item_id']}/display",
                    }
                    # 结果按原始顺序落位，前端轮询时列表不会跳
                    current_job["results"] = [filled for filled in slots if filled]
                    task["status"] = "done"
                    success_count += 1
                    current_job["success_count"] = success_count
                    app.logger.info("Processed %s -> %s", task["name"], output_path)
                else:
                    task["status"] = "error"
                    task["error"] = user_error(item.error)
                    failed_count += 1
                    current_job["failed_count"] = failed_count
                    app.logger.error("Task failed for %s: %s", task["name"], item.trace)
            current_job["progress"] = min(0.995, sum(partial.values()) / total)

    # 照片重绘每个并行 worker 都要多做一份模型推理缓冲，按 2 路封顶（再高显存/内存收益很小）
    photo_job = any(task.get("method") == options.PHOTO for task in tasks)
    workers = batch_runner.run_batch(
        items,
        workers=2 if photo_job else None,
        on_event=on_event,
        should_stop=(cancel_event.is_set if cancel_event is not None else None),
    )
    app.logger.info(
        "Job %s finished batch of %s with %s image worker(s)", job_id, total, workers
    )

    with INDEX_LOCK:
        current_job = JOBS.get(job_id)
        if not current_job:
            return
        current_job["progress"] = 1.0
        if current_job.get("cancel_requested"):
            # 用户按了停止：这是这批的最终状态，已完成的结果照旧保留、可下载
            current_job["status"] = "cancelled"
            if success_count:
                parts = [f"已完成 {success_count} 项", f"已停止 {cancelled_count} 项"]
                if failed_count:
                    parts.append(f"失败 {failed_count} 项")
                current_job["message"] = "已停止：" + "、".join(parts) + "；已完成的成品仍可下载"
            else:
                current_job["message"] = "已停止：这一批还没有成品"
        elif success_count:
            current_job["status"] = "done"
            if failed_count:
                current_job["message"] = f"处理完成：成功 {success_count} 项，失败 {failed_count} 项"
            else:
                current_job["message"] = f"处理完成，共 {success_count} 项"
        else:
            current_job["status"] = "failed"
            current_job["message"] = "处理失败，请查看每个文件的错误提示"
        current_job["finished_at"] = time.time()
    app.logger.info(
        "Job %s finished: %s ok, %s failed, %s cancelled",
        job_id,
        success_count,
        failed_count,
        cancelled_count,
    )


@app.before_request
def prepare_request():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.permanent = True
    if request.path.startswith("/api/") and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        token = request.headers.get("X-App-Token", "")
        if not secrets.compare_digest(token, APP_TOKEN):
            return jsonify({"error": "页面校验失败，请刷新后重试"}), 403
    return None


@app.after_request
def set_security_headers(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return jsonify({"error": "上传文件过大，请分批处理"}), 413


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": error.description or error.name}), error.code
    return error


@app.errorhandler(Exception)
def handle_unexpected(error):
    app.logger.error("Unhandled error: %s", traceback.format_exc())
    if request.path.startswith("/api/"):
        return jsonify({"error": "服务出现异常，请重试；如果持续出现请重新启动工具"}), 500
    return "服务出现异常，请重新启动去水印工具。", 500


@app.get("/")
def index():
    return render_template(
        "index.html", app_token=APP_TOKEN, photo_ready=photo_inpaint.available()
    )


@app.post("/api/photo/warmup")
def photo_warmup():
    if not photo_inpaint.available():
        return jsonify({"ok": False, "error": photo_inpaint.unavailable_reason()}), 400
    photo_inpaint.warmup()
    return jsonify({"ok": True, "message": "照片修复模型正在后台加载"})


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "app": "jingyin",
            "version": 1,
            "instance": INSTANCE_ID,
            "dir": DIR_TAG,
            "photo_ready": photo_inpaint.available(),
        }
    )


def _latest_job_for_session(sid: str) -> dict | None:
    with INDEX_LOCK:
        candidates = [job for job in JOBS.values() if job.get("session_id") == sid]
    if not candidates:
        return None
    return max(candidates, key=lambda job: job.get("created_at", 0))


@app.get("/api/state")
def state():
    with INDEX_LOCK:
        items = [public_item(item) for item in list(get_session_store()["items"].values())]
    latest = _latest_job_for_session(session["sid"])
    return jsonify({"items": items, "job": public_job(latest) if latest else None})


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "没有收到文件"}), 400

    sid = session["sid"]
    store = get_session_store()
    uploaded: list[dict] = []
    errors: list[str] = []
    for uploaded_file in files:
        original_name = media.clean_name(uploaded_file.filename or "素材")
        suffix = Path(original_name).suffix.lower()
        kind = media.media_kind(suffix)
        if not kind:
            errors.append(f"{original_name}：不支持这种文件格式")
            continue
        with INDEX_LOCK:
            pending = store.get("pending_uploads", 0)
            if len(store["items"]) + pending >= MAX_BATCH_ITEMS:
                errors.append(f"{original_name}：最多可添加 {MAX_BATCH_ITEMS} 份素材，请先移除一些文件")
                continue
            store["pending_uploads"] = pending + 1
        reserved = True
        item_id = uuid.uuid4().hex[:16]
        item_dir = UPLOAD_ROOT / sid / item_id
        source_path = item_dir / f"source{suffix}"
        try:
            item_dir.mkdir(parents=True, exist_ok=True)
            uploaded_file.save(source_path)
            if source_path.stat().st_size == 0:
                raise media.MediaError("文件是空的")
            thumb_path = item_dir / "thumb.jpg"
            preview_path = item_dir / "preview.jpg"
            if kind == "image":
                info = media.probe_image(source_path)
                media.make_image_thumb(source_path, thumb_path, max_side=520, quality=88)
                media.make_preview(source_path, preview_path)
                duration = None
            else:
                info = media.probe_video(source_path)
                poster_info = media.make_video_poster(
                    source_path, thumb_path, duration=info["duration"], max_side=720
                )
                info["width"] = poster_info["width"]
                info["height"] = poster_info["height"]
                preview_path = thumb_path
                duration = info["duration"]

            item = {
                "id": item_id,
                "name": original_name,
                "kind": kind,
                "suffix": suffix,
                "width": info["width"],
                "height": info["height"],
                "duration": duration,
                "fps": info.get("fps"),
                "frame_count": info.get("frame_count"),
                "source_path": source_path,
                "thumb_path": thumb_path,
                "preview_path": preview_path,
                "created_at": time.time(),
            }
            with INDEX_LOCK:
                store["items"][item_id] = item
                store["pending_uploads"] -= 1
                reserved = False
            try:
                state_store.write_item_meta(item, item_dir)
            except OSError as meta_error:
                # 索引没落盘只影响「重启后还能不能恢复」，不该让这份素材上传失败
                app.logger.warning("Meta write failed for %s: %s", item_id, meta_error)
            uploaded.append(public_item(item))
            app.logger.info("Uploaded %s as %s", original_name, item_id)
        except Exception as exc:
            shutil.rmtree(item_dir, ignore_errors=True)
            app.logger.warning("Upload failed for %s: %s", original_name, exc)
            errors.append(f"{original_name}：{user_error(exc)}")
        finally:
            if reserved:
                with INDEX_LOCK:
                    store["pending_uploads"] -= 1

    status_code = 200 if uploaded else 400
    return jsonify({"items": uploaded, "errors": errors}), status_code


# 视频自动识别：只看一帧会漏掉「片头才有 / 中途才出现」的台标，所以按片长采样多帧再聚合。
# 采样帧先降采样到长边 1280 再检测：归一化坐标不受影响，单帧耗时可控。
# 只有出现在至少 2 个采样帧的位置才自动成框——单帧偶发的白色物体不算水印。
VIDEO_DETECT_FRACTIONS = (0.02, 0.2, 0.5, 0.8, 0.98)
VIDEO_DETECT_MAX_SIDE = 1280
VIDEO_DETECT_MIN_SUPPORT = 2
VIDEO_DETECT_OVERLAP = 0.35
# 真实素材实测：5 个等分点在 52s 片子上缝宽 9–16s，只出现几秒的台标会正好掉在缝里
# （片头 2s / 6s、中段 4s / 12s 四种，5 点全部抓不到；拉到 9 点后片头 6s 能抓回）。
# 所以把超过 7 秒的缝对半补点，总数封顶 9——实拍素材单帧检测 0.28s，9 帧约 2.5s，一次识别只点一次。
VIDEO_DETECT_MAX_SPACING = 7.0
VIDEO_DETECT_MAX_SAMPLES = 9


def _video_sample_times(duration: float, current: float) -> list[float]:
    end = max(0.0, duration - 0.05)
    times: list[float] = []

    def add(value: float) -> None:
        value = min(max(0.0, value), end)
        if all(abs(value - existing) >= 0.2 for existing in times):
            times.append(value)

    if current > 0:
        add(current)
    for fraction in VIDEO_DETECT_FRACTIONS:
        add(duration * fraction)
    if not times:
        return [0.0]
    head = times[:1] if current > 0 else []
    rest = times[len(head):]
    for _ in range(VIDEO_DETECT_MAX_SAMPLES - len(times)):
        gaps = [(rest[index + 1] - rest[index], index) for index in range(len(rest) - 1)]
        widest = max(gaps, default=(0.0, -1))
        if widest[0] <= VIDEO_DETECT_MAX_SPACING:
            break
        middle = (rest[widest[1]] + rest[widest[1] + 1]) / 2
        rest.insert(widest[1] + 1, middle)
    return head + rest


def _detect_frame(rgb: np.ndarray) -> dict:
    height, width = rgb.shape[:2]
    longest = max(height, width)
    small = rgb
    if longest > VIDEO_DETECT_MAX_SIDE:
        scale = VIDEO_DETECT_MAX_SIDE / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    result = detection.detect_watermark(small)
    if small is not rgb:
        # 降采样会把 4K 上的小台标压到角标的尺寸门槛以下（实测 48px 台标 0 框），
        # 四个角再按原生分辨率补一遍；大的角标由降采样那一路负责，两边互不替代。
        extra = [
            item
            for item in detection.corner_regions_native(rgb)
            if not any(
                _overlap_ratio(item, kept) > VIDEO_DETECT_OVERLAP
                for kept in result["regions"]
            )
        ]
        if extra:
            result = dict(result)
            result["regions"] = list(result["regions"]) + extra
            result["found"] = True
    return result


def _overlap_ratio(first: dict, second: dict) -> float:
    width = min(first["x"] + first["w"], second["x"] + second["w"]) - max(first["x"], second["x"])
    height = min(first["y"] + first["h"], second["y"] + second["h"]) - max(first["y"], second["y"])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min(first["w"] * first["h"], second["w"] * second["h"])
    return (width * height) / smaller if smaller > 0 else 0.0


def _cluster_frames(frames: list[list[dict]]) -> list[dict]:
    """把各采样帧检测到的框按重叠度聚类，记下每个位置出现在哪几帧。"""
    clusters: list[dict] = []
    for frame_index, regions in enumerate(frames):
        for region in regions:
            for cluster in clusters:
                if _overlap_ratio(cluster["anchor"], region) > VIDEO_DETECT_OVERLAP:
                    cluster["frames"].add(frame_index)
                    cluster["rects"].append(region)
                    cluster["anchor"] = region
                    break
            else:
                clusters.append(
                    {
                        "frames": {frame_index},
                        "rects": [region],
                        "anchor": region,
                        "sample": region,
                    }
                )
    return clusters


def _union_rect(rects: list[dict]) -> dict:
    x0 = min(rect["x"] for rect in rects)
    y0 = min(rect["y"] for rect in rects)
    x1 = max(rect["x"] + rect["w"] for rect in rects)
    y1 = max(rect["y"] + rect["h"] for rect in rects)
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(x1 - x0, 5),
        "h": round(y1 - y0, 5),
    }


def _detect_video(source: Path, current: float) -> dict:
    duration = float(media.probe_video(source).get("duration") or 0)
    # 容器时长可能比画面长（长音轨 mkv 实测 30.03s 报成画面长度），按画面轨自己的长度排采样点
    length = media.probe_video_length(source)
    if length.get("duration"):
        duration = min(duration, length["duration"]) if duration else length["duration"]
    results: list[dict] = []
    for moment in _video_sample_times(duration, current):
        try:
            results.append(_detect_frame(media.extract_video_frame(source, moment)))
        except media.MediaError:
            continue
    if not results:
        raise media.MediaError("无法读取视频画面，请确认视频文件完整")

    sample_count = len(results)
    threshold = 1 if sample_count == 1 else VIDEO_DETECT_MIN_SUPPORT
    region_clusters = [
        cluster
        for cluster in _cluster_frames([result["regions"] for result in results])
        if len(cluster["frames"]) >= threshold
    ]
    regions = [_union_rect(cluster["rects"]) for cluster in region_clusters]
    suggestion_clusters = [
        cluster
        for cluster in _cluster_frames(
            [result.get("suggestions") or [] for result in results]
        )
        if len(cluster["frames"]) >= threshold
    ][:6]
    suggestions = []
    for cluster in suggestion_clusters:
        suggestion = dict(cluster["sample"])
        suggestion["score"] = max(item.get("score", 0.0) for item in cluster["rects"])
        suggestions.append(suggestion)

    extra = f"；另有 {len(suggestions)} 处低对比疑似位置（虚线框），请核对" if suggestions else ""
    if regions:
        hit_frames = len({index for cluster in region_clusters for index in cluster["frames"]})
        message = (
            f"全片采样 {sample_count} 帧，其中 {hit_frames} 帧找到水印；"
            f"只在极短片段出现的请手动补框" + extra
        )
    elif suggestions:
        message = f"全片采样 {sample_count} 帧未找到明显水印，另有 {len(suggestions)} 处低对比疑似位置（虚线框），请核对"
    else:
        message = f"全片采样 {sample_count} 帧均未找到明显水印，请直接拖动鼠标框选（可框多个）"
    return {
        "found": bool(regions),
        "regions": regions,
        "suggestions": suggestions,
        "message": message,
        "sampled_frames": sample_count,
    }


@app.get("/api/items/<item_id>/thumb")
def item_thumb(item_id: str):
    item = find_item(item_id)
    if not item:
        abort(404)
    return send_file(item["thumb_path"], mimetype="image/jpeg", conditional=True, max_age=600)


@app.get("/api/items/<item_id>/display")
def item_display(item_id: str):
    item = find_item(item_id)
    if not item:
        abort(404)
    if item["kind"] == "image":
        return send_file(item["preview_path"], mimetype="image/jpeg", conditional=True)
    mimetype = mimetypes.guess_type(item["source_path"].name)[0] or "application/octet-stream"
    return send_file(
        item["source_path"],
        mimetype=mimetype,
        conditional=True,
        download_name=item["name"],
    )


@app.post("/api/detect")
def detect_region():
    data = request.get_json(silent=True) or {}
    item = find_item(str(data.get("item_id", "")))
    if not item:
        return jsonify({"error": "找不到这个文件，请重新添加"}), 404
    timestamp = data.get("time", 0)
    try:
        timestamp = max(0.0, float(timestamp))
    except (TypeError, ValueError):
        timestamp = 0.0
    try:
        if item["kind"] == "image":
            with Image.open(item["source_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                rgb = np.array(image)
            return jsonify(detection.detect_watermark(rgb))
        return jsonify(_detect_video(item["source_path"], timestamp))
    except Exception as exc:
        app.logger.warning("Auto detection failed: %s", traceback.format_exc())
        return jsonify({"error": user_error(exc)}), 400


@app.delete("/api/items/<item_id>")
def delete_item(item_id: str):
    sid = session["sid"]
    if _job_active_for_session(sid):
        return jsonify({"error": "正在处理中，暂时不能删除素材"}), 409
    with INDEX_LOCK:
        store = get_session_store()
        item = store["items"].pop(item_id, None)
    if not item:
        return jsonify({"error": "文件不存在"}), 404
    shutil.rmtree(item["source_path"].parent, ignore_errors=True)
    return jsonify({"ok": True})


@app.post("/api/clear")
def clear_items():
    sid = session["sid"]
    if _job_active_for_session(sid):
        return jsonify({"error": "正在处理中，暂时不能清空素材"}), 409
    with INDEX_LOCK:
        store = get_session_store()
        items = list(store["items"].values())
        store["items"].clear()
    for item in items:
        shutil.rmtree(item["source_path"].parent, ignore_errors=True)
    return jsonify({"ok": True})


def parse_job_options(data: dict) -> tuple[str, str, bool]:
    """请求体里三组修复选项的归一化：认不出来的一律回默认档。

    视频贴边选框的取舍交给用户：`quality`＝逐帧（残留少），`fast`＝内缩走 delogo（快 3–4 倍）；
    会移动的水印（台标跟画面一起动）需要逐帧跟踪，不能交给固定矩形。
    """
    return (
        options.normalize_method(data.get("method")),
        options.normalize_video_mode(data.get("video_mode")),
        options.normalize_motion(data.get("motion")),
    )


@app.post("/api/process")
def create_job():
    data = request.get_json(silent=True) or {}
    method, video_mode, motion = parse_job_options(data)
    if method == options.PHOTO and not photo_inpaint.available():
        return jsonify({"error": photo_inpaint.unavailable_reason() or "照片重绘不可用"}), 400
    specs = data.get("items")
    if not isinstance(specs, list) or not specs:
        return jsonify({"error": "请先选择要处理的文件并框出水印"}), 400
    if len(specs) > MAX_BATCH_ITEMS:
        return jsonify({"error": f"一次最多处理 {MAX_BATCH_ITEMS} 个文件，请分批进行"}), 400

    sid = session["sid"]
    store = get_session_store()
    tasks: list[dict] = []
    seen_items: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        item_id = str(spec.get("id", ""))
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        with INDEX_LOCK:
            item = store["items"].get(item_id)
        if not item:
            continue
        try:
            regions = normalize_regions(spec.get("regions"), owner=item["name"])
        except RegionLimitError as exc:
            return jsonify({"error": str(exc)}), 400
        if not regions:
            continue
        tasks.append(
            {
                "id": uuid.uuid4().hex[:10],
                "item_id": item_id,
                "name": item["name"],
                "kind": item["kind"],
                "source_path": item["source_path"],
                "width": item["width"],
                "height": item["height"],
                "suffix": item["suffix"],
                "regions": regions,
                "method": method,
                "video_mode": video_mode,
                "motion": motion,
                "status": "pending",
                "progress": 0,
                "error": None,
            }
        )
    if not tasks:
        return jsonify({"error": "所选文件都还没有有效的水印选框"}), 400

    output_dir = _make_output_dir()
    used_names: set[str] = set()
    for task in tasks:
        suffix = task["suffix"] if task["kind"] == "image" else ".mp4"
        base_name = f"{media.clean_name(Path(task['name']).stem, '素材')[:120]}_已去水印{suffix}"
        candidate = base_name
        counter = 2
        while candidate in used_names:
            candidate = f"{media.clean_name(Path(task['name']).stem, '素材')[:120]}_已去水印_{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        task["output_name"] = candidate

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "session_id": sid,
        "status": "queued",
        "progress": 0,
        "message": "已加入处理队列",
        "tasks": tasks,
        "results": [],
        "success_count": 0,
        "failed_count": 0,
        "cancelled_count": 0,
        "cancel_event": threading.Event(),
        "cancel_requested": False,
        "video_mode": video_mode,
        "motion": motion,
        "output_dir": output_dir,
        "created_at": time.time(),
    }
    with INDEX_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return jsonify(public_job(job))


def get_job(job_id: str) -> dict:
    with INDEX_LOCK:
        job = JOBS.get(job_id)
    if not job or job.get("session_id") != session["sid"]:
        abort(404)
    return job


@app.get("/api/jobs/<job_id>")
def get_job_status(job_id: str):
    job = get_job(job_id)
    with INDEX_LOCK:
        return jsonify(public_job(job))


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    """停止这批任务：已完成的成品全部保留，排队中的不再开工，正在跑的视频停在当前帧。

    逐帧路线在实拍上比插值慢 15–28 倍（一段 6s 片子 16–19s，4K 素材 35–47s），
    没有这道闸就只能等整批跑完；半成品走 `.part` 落盘，停止后不会留下坏文件。
    """
    job = get_job(job_id)
    with INDEX_LOCK:
        if job["status"] not in ("queued", "running"):
            return jsonify({"error": "这一批已经结束了，不需要停止"}), 409
        job["cancel_requested"] = True
        job["message"] = "正在停止：已经开工的那一份停在当前位置，排队中的不再处理"
        event = job.get("cancel_event")
    if event is not None:
        event.set()
    return jsonify(public_job(job))


@app.post("/api/jobs/<job_id>/resume")
def resume_job(job_id: str):
    """继续处理这一批剩下的素材：已完成的不重跑，成品仍落在同一个文件夹里。

    第十八轮的「停止」只解决「停不下来」，停完还得整批重新点一次——已完成的会重复处理，
    成品还会散进第二个目录。这里把没做完的接着跑在同一批、同一个成品目录上。
    """
    job = get_job(job_id)
    with INDEX_LOCK:
        if job["status"] in ("queued", "running"):
            return jsonify({"error": "这一批还在处理，不需要继续处理"}), 409
        if job["status"] not in ("cancelled", "failed"):
            return jsonify({"error": "这一批已经处理完了，没有需要继续的"}), 409
        # 成品被用户从文件夹里移走的那几份算没做完：否则打包会缺文件，界面上还会留一张点不开的卡
        alive: list[dict] = []
        dropped: set[str] = set()
        for result in job.get("results", []):
            if _file_ready(result["file_path"]):
                alive.append(result)
            else:
                dropped.add(result["task_id"])
                preview = Path(result["preview_path"])
                if result["kind"] == "image" and preview != Path(result["file_path"]):
                    try:
                        os.remove(preview)
                    except OSError:
                        pass
        job["results"] = alive
        for task in job["tasks"]:
            if task["id"] in dropped:
                task["status"] = "pending"
                task["progress"] = 0.0
                task["error"] = None
        done_tasks = [task for task in job["tasks"] if task["status"] == "done"]
        pending_tasks = [task for task in job["tasks"] if task["status"] != "done"]
        if not pending_tasks:
            return jsonify({"error": "这一批的素材都已经处理完了，没有需要继续的"}), 409
        output_dir = Path(job["output_dir"])
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return jsonify({"error": f"成品文件夹无法写入：{output_dir}"}), 500
        taken = {task["output_name"] for task in done_tasks}
        for task in pending_tasks:
            task["status"] = "pending"
            task["progress"] = 0.0
            task["error"] = None
            task["started"] = False
            task["output_name"] = _free_output_name(output_dir, task["output_name"], taken)
            taken.add(task["output_name"])
        job["status"] = "queued"
        job["progress"] = round(len(done_tasks) / len(job["tasks"]), 4)
        job["success_count"] = len(done_tasks)
        job["failed_count"] = 0
        job["cancelled_count"] = 0
        job["cancel_requested"] = False
        job["cancel_event"] = threading.Event()
        job["finished_at"] = None
        job["resumed_at"] = time.time()
        job["resume_count"] = int(job.get("resume_count", 0)) + 1
        parts = [f"继续处理剩下的 {len(pending_tasks)} 项"]
        if done_tasks:
            parts.append(f"已完成的 {len(done_tasks)} 项不重复处理")
        if dropped:
            parts.append(f"{len(dropped)} 项成品已被移走，会重新处理")
        job["message"] = "；".join(parts)
    app.logger.info(
        "Job %s resumed: %s remaining, %s already done, %s dropped",
        job_id,
        len(pending_tasks),
        len(done_tasks),
        len(dropped),
    )
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return jsonify(public_job(job))


def find_result(job: dict, result_id: str) -> dict:
    with INDEX_LOCK:
        for result in job.get("results", []):
            if result["id"] == result_id:
                return result
    abort(404)


def _find_result_or_none(job: dict, result_id: str) -> dict | None:
    """同 find_result，但找不到时返回 None——单份重跑要自己决定回什么文案。"""
    with INDEX_LOCK:
        for result in job.get("results", []):
            if result["id"] == result_id:
                return result
    return None


def _rerun_one(job_id: str, result_id: str) -> None:
    """只重跑一条成品：把新结果覆盖到同一个文件名上，同批其他份数一个字节都不动。

    先写 `_rerun` 临时名、成功后才 `os.replace` 覆盖原成品——重跑失败或中途停止时，
    用户手上原来那份还能用，不会留下半截文件。成品变了就把这批的 ZIP 缓存删掉，
    否则下载到的还是旧包（ZIP 复用只比条目名，名字没变就看不出来）。
    """
    with INDEX_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        result = _find_result_or_none(job, result_id)
        if result is None:
            return
        task = next((entry for entry in job["tasks"] if entry["id"] == result["task_id"]), None)
        if task is None:
            return
        output_path = Path(result["file_path"])
        preview_path = Path(result["preview_path"])
        # 视频的预览就是成品本身，两份临时名会是同一个，别重复替换
        same_preview = preview_path == output_path
        temp_output = output_path.with_name(f"{output_path.stem}_rerun{output_path.suffix}")
        temp_preview = (
            temp_output
            if same_preview
            else preview_path.with_name(f"{preview_path.stem}_rerun{preview_path.suffix}")
        )
        cancel_event = job.get("cancel_event")
        item = batch_runner.BatchItem(
            item_id=task["id"],
            kind=task["kind"],
            input_path=Path(task["source_path"]),
            output_path=temp_output,
            regions=task["regions"],
            method=task["method"],
            video_mode=options.normalize_video_mode(task.get("video_mode")),
            motion=options.normalize_motion(task.get("motion")),
            width=task["width"],
            height=task["height"],
            preview_path=temp_preview,
        )

    def on_event(_entry: batch_runner.BatchItem, phase: str, value: float) -> None:
        with INDEX_LOCK:
            current = JOBS.get(job_id)
            if not current:
                return
            if phase == "start":
                current["message"] = f"正在重跑：{task['name']}"
                current["progress"] = 0.01
            elif phase == "tick":
                current["progress"] = min(0.995, float(value))

    app.logger.info("Rerun %s started for %s", job_id, task["name"])
    batch_runner.run_batch(
        [item],
        workers=1,
        on_event=on_event,
        should_stop=(cancel_event.is_set if cancel_event is not None else None),
    )

    with INDEX_LOCK:
        current = JOBS.get(job_id)
        if not current:
            return
        current["progress"] = 1.0
        current["finished_at"] = time.time()
        if item.cancelled:
            task["status"] = "cancelled"
            task["error"] = None
            current["status"] = "cancelled"
            current["message"] = f"已停止重跑：{task['name']}；原来的成品还在"
        elif item.error is not None:
            task["status"] = "error"
            task["error"] = user_error(item.error)
            current["status"] = "failed" if not current.get("results") else "done"
            current["message"] = f"重跑失败：{task['name']}（{task['error']}）；原来的成品还在"
            app.logger.error("Rerun failed for %s: %s", task["name"], item.trace)
        else:
            try:
                os.replace(temp_output, output_path)
                if not same_preview:
                    os.replace(temp_preview, preview_path)
            except OSError as exc:
                task["status"] = "error"
                task["error"] = user_error(exc)
                current["status"] = "failed" if not current.get("results") else "done"
                current["message"] = f"重跑完成但覆盖原文件失败：{user_error(exc)}"
                app.logger.error("Rerun replace failed for %s: %s", task["name"], exc)
            else:
                task["status"] = "done"
                task["error"] = None
                result["size"] = output_path.stat().st_size
                result["note"] = item.result["note"]
                current["status"] = "done"
                current["message"] = f"已重跑：{task['name']}"
                app.logger.info("Rerun %s -> %s", task["name"], output_path)
        # 不论成没成，先把这批的 ZIP 缓存作废：成功时它已经过期，失败时下次下载会重建
        try:
            (ZIP_DIR / f"{job_id}.zip").unlink()
        except OSError:
            pass
        for leftover in (temp_output, temp_preview):
            try:
                leftover.unlink()
            except OSError:
                pass


@app.post("/api/jobs/<job_id>/results/<result_id>/rerun")
def rerun_result(job_id: str, result_id: str):
    """单独重跑这一份：用界面当前的选框与档位，覆盖同一个成品文件。

    和「继续处理剩下的 N 项」分工不同：续跑补的是没做完的，这里重做的是**已经做完、
    但用户想换框或换档位再试一次**的那一份。
    """
    job = get_job(job_id)
    data = request.get_json(silent=True) or {}
    method, video_mode, motion = parse_job_options(data)
    if method == options.PHOTO and not photo_inpaint.available():
        return jsonify({"error": photo_inpaint.unavailable_reason() or "照片重绘不可用"}), 400
    with INDEX_LOCK:
        if job["status"] in ("queued", "running"):
            return jsonify({"error": "这一批还在处理，等它跑完再单独重跑"}), 409
        result = _find_result_or_none(job, result_id)
        if result is None:
            return jsonify({"error": "这条成品已经不在了，请刷新后再试"}), 404
        task = next((entry for entry in job["tasks"] if entry["id"] == result["task_id"]), None)
        if task is None:
            return jsonify({"error": "这条成品对应的素材记录已经不在，请重新上传"}), 404
        source = Path(task["source_path"])
        if not source.is_file():
            return jsonify({"error": "素材已被回收（保留 72 小时），请重新上传这份文件"}), 410
        raw_regions = data.get("regions")
        try:
            regions = normalize_regions(
                raw_regions if raw_regions is not None else task.get("regions"),
                owner=task["name"],
            )
        except RegionLimitError as exc:
            return jsonify({"error": str(exc)}), 400
        if not regions:
            return jsonify({"error": "这份素材还没有选框，请先在画面上框出水印"}), 400
        task["regions"] = regions
        task["method"] = method
        task["video_mode"] = video_mode
        task["motion"] = motion
        task["status"] = "processing"
        task["progress"] = 0.0
        task["error"] = None
        task["started"] = False
        job["status"] = "running"
        job["progress"] = 0.0
        job["cancel_requested"] = False
        job["cancel_event"] = threading.Event()
        job["finished_at"] = None
        job["message"] = f"正在重跑：{task['name']}"
        snapshot = public_job(job)
    app.logger.info("Job %s rerun requested for %s", job_id, task["name"])
    threading.Thread(target=_rerun_one, args=(job_id, result_id), daemon=True).start()
    return jsonify(snapshot)


@app.get("/api/jobs/<job_id>/results/<result_id>/preview")
def result_preview(job_id: str, result_id: str):
    job = get_job(job_id)
    result = find_result(job, result_id)
    preview_path = result["preview_path"]
    if not preview_path.exists():
        abort(404)
    mimetype = "image/jpeg" if result["kind"] == "image" else "video/mp4"
    return send_file(preview_path, mimetype=mimetype, conditional=True)


@app.get("/api/jobs/<job_id>/results/<result_id>/file")
def result_file(job_id: str, result_id: str):
    job = get_job(job_id)
    result = find_result(job, result_id)
    file_path = result["file_path"]
    if not file_path.exists():
        abort(404)
    as_download = request.args.get("download") == "1"
    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=as_download,
        download_name=result["name"],
        conditional=True,
    )


def _file_ready(file_path: Path) -> bool:
    try:
        return file_path.is_file() and file_path.stat().st_size > 0
    except OSError:
        return False


def _zip_is_complete(zip_path: Path, names: list[str]) -> bool:
    """已存在的 ZIP 只有在条目与本次成品完全一致且逐个可读时才复用，半截文件会被重建。"""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return archive.testzip() is None and sorted(archive.namelist()) == sorted(names)
    except Exception:
        return False


def _zip_is_fresh(zip_path: Path, results: list[dict]) -> bool:
    """ZIP 必须比每个成品都新。

    单份重跑会覆盖同名成品：条目名一个没变，`_zip_is_complete` 照样认它完整，
    用户于是下载到旧包。重跑时虽然会尽力删缓存，但 Windows 上刚下载完的文件句柄
    还没释放时删不掉（`unlink` 只能静默失败），所以判新旧这件事必须由下载这一侧兜住。
    """
    try:
        stamp = zip_path.stat().st_mtime
    except OSError:
        return False
    for result in results:
        try:
            if Path(result["file_path"]).stat().st_mtime > stamp:
                return False
        except OSError:
            continue
    return True


def _write_zip(zip_path: Path, results: list[dict], names: list[str]) -> None:
    """先写 .part、关掉后复核可读，再改名替换——避免不可用的 zip 留在 work/zips 里被反复当成品。"""
    part_path = zip_path.with_name(zip_path.name + ".part")
    try:
        with zipfile.ZipFile(part_path, "w", allowZip64=True) as archive:
            for result in results:
                compression = (
                    zipfile.ZIP_STORED if result["kind"] == "video" else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    result["file_path"],
                    arcname=result["name"],
                    compress_type=compression,
                )
        with zipfile.ZipFile(part_path) as check:
            if check.testzip() is not None or sorted(check.namelist()) != sorted(names):
                raise OSError("打包后复核不一致")
        os.replace(part_path, zip_path)
    finally:
        if part_path.exists():
            os.remove(part_path)


@app.get("/api/jobs/<job_id>/zip")
def download_zip(job_id: str):
    job = get_job(job_id)
    with INDEX_LOCK:
        results = list(job.get("results", []))
        status = job["status"]
    if status not in ("done", "failed", "cancelled") or not results:
        return jsonify({"error": "处理还没有完成"}), 409
    names = [result["name"] for result in results]
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{job_id}.zip"
    # 已存在且完整可读的 ZIP 是自成的，成品目录被用户清掉也照样能下载；只有要重打包才需要成品在场
    if not _zip_is_complete(zip_path, names) or not _zip_is_fresh(zip_path, results):
        missing = [name for name, result in zip(names, results) if not _file_ready(result["file_path"])]
        if missing:
            return jsonify({"error": f"有 {len(missing)} 个成品文件已不在磁盘上，请重新处理这一批"}), 409
        try:
            _write_zip(zip_path, results, names)
        except Exception as exc:
            app.logger.error("Job %s zip failed: %s", job_id, exc)
            return jsonify({"error": "打包失败，请查看 work\\tool.log"}), 500
    download_name = f"去水印结果_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(zip_path, as_attachment=True, download_name=download_name)


@app.post("/api/jobs/<job_id>/open-folder")
def open_output_folder(job_id: str):
    job = get_job(job_id)
    output_dir = Path(job["output_dir"])
    if not output_dir.exists():
        return jsonify({"error": "成品文件夹不存在"}), 404
    try:
        if os.name == "nt":
            os.startfile(str(output_dir))
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(output_dir)])
        return jsonify({"ok": True, "path": str(output_dir)})
    except Exception as exc:
        app.logger.warning("Cannot open folder: %s", exc)
        return jsonify({"error": f"无法打开文件夹，请手动前往：{output_dir}"}), 500


@app.post("/api/shutdown")
def shutdown_server():
    sid = session["sid"]
    if _job_active_for_session(sid):
        return jsonify({"error": "还有文件正在处理，请等待完成后再关闭"}), 409
    server = SERVER_HOLDER.get("server")
    if server is None:
        return jsonify({"error": "服务尚未就绪"}), 500

    def stop() -> None:
        time.sleep(0.35)
        server.shutdown()

    threading.Thread(target=stop, daemon=True).start()
    return jsonify({"ok": True})


def cleanup_old_files() -> None:
    """回收中间产物：素材目录 72h、ZIP 7 天，写盘中断留下的 .part 残档 12h。"""
    now = time.time()
    if UPLOAD_ROOT.exists():
        for folder in UPLOAD_ROOT.iterdir():
            try:
                if folder.is_dir() and now - folder.stat().st_mtime > 72 * 3600:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue
    if ZIP_DIR.exists():
        for file in ZIP_DIR.iterdir():
            try:
                if not file.is_file():
                    continue
                age = now - file.stat().st_mtime
                if age > 7 * 24 * 3600 or (file.name.endswith(".part") and age > 12 * 3600):
                    file.unlink(missing_ok=True)
            except OSError:
                continue
    if OUTPUT_ROOT.exists():
        for file in OUTPUT_ROOT.rglob("*"):
            try:
                if not file.is_file() or not file.name.endswith((".part", ".part.mp4")):
                    continue
                if now - file.stat().st_mtime > 12 * 3600:
                    file.unlink(missing_ok=True)
            except OSError:
                continue


def restore_sessions_from_disk() -> None:
    """把 work/uploads 里还活着的素材索引装回内存，让重开浏览器/重启服务后列表不空。

    必须在 cleanup_old_files() 之后调用：先清理再恢复，否则会把正要回收的目录又读回来。
    """
    sessions, items = state_store.rebuild_sessions(UPLOAD_ROOT, SESSIONS, INDEX_LOCK)
    if items:
        app.logger.info("Restored %s item(s) across %s session(s) from disk", items, sessions)


def probe_health(port: int, timeout: float) -> dict | None:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("app") == "jingyin":
        return payload
    return None


def _same_install(payload: dict, expected_instance: str) -> bool:
    """端口上应答的是不是本目录这一份工具。"""
    if expected_instance:
        return payload.get("instance") == expected_instance
    return payload.get("dir") in (None, DIR_TAG)


def find_existing_instance() -> dict | None:
    """找出本目录这份工具当前活着的实例，返回 {"port", "instance"}；没有则 None。

    只按 runtime.json 里的端口探活不够：8 路并行满载时 1.2s 超时会假阴性，
    记录也可能被上一次强杀留下，于是会起第二个服务并把记录覆盖掉，
    前一个变成还在写同一批目录的隐形孤儿进程。所以放宽超时、重试一次，
    再扫一遍端口范围的目录指纹，并在认领前核对实例身份。
    """
    recorded_port: int | None = None
    expected_instance = ""
    if RUNTIME_FILE.exists():
        try:
            data = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
            recorded_port = int(data["port"])
            expected_instance = str(data.get("instance", ""))
            if str(data.get("base_dir", "")) not in ("", str(BASE_DIR)):
                return None
        except (OSError, ValueError, TypeError):
            recorded_port = None

    if recorded_port is not None:
        for attempt in range(2):
            payload = probe_health(recorded_port, 2.5)
            if payload is not None:
                if _same_install(payload, expected_instance):
                    return {"port": recorded_port, "instance": str(payload.get("instance", ""))}
                break
            time.sleep(0.4)

    for port in PORT_RANGE:
        payload = probe_health(port, 0.4)
        if payload is not None and _same_install(payload, ""):
            app.logger.warning("runtime.json 的记录对不上端口 %s 上的应答，按目录指纹复用", port)
            return {"port": port, "instance": str(payload.get("instance", ""))}
    return None


def choose_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sys.platform == "win32":
                # Windows 的 SO_REUSEADDR 允许绑定到另一个服务正在监听的端口。
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("端口 8765-8785 都被占用，请关闭其他程序后重试")


def remove_runtime_file() -> None:
    try:
        if RUNTIME_FILE.exists():
            data = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
            if int(data.get("pid", -1)) == os.getpid():
                RUNTIME_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _recorded_port() -> int | None:
    try:
        return int(json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def write_runtime_file(port: int, pid: int, instance: str) -> None:
    RUNTIME_FILE.write_text(
        json.dumps(
            {
                "port": port,
                "pid": pid,
                "instance": instance,
                "base_dir": str(BASE_DIR),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    setup_logging()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cleanup_old_files()

    existing = find_existing_instance()
    if existing:
        import webbrowser

        if existing["port"] != _recorded_port():
            # 记录指向的端口已经不通、活实例在别的端口上：把记录改对，pid 不属于我们所以写 0
            write_runtime_file(existing["port"], 0, existing.get("instance", ""))
        webbrowser.open(f"http://127.0.0.1:{existing['port']}/")
        return

    # 记录里的实例探不通（上次被强杀，或启动中途被打断）：先删掉残档再抢端口，
    # 否则启动器只看文件是否存在，会对着死端口报「已在浏览器打开」
    if RUNTIME_FILE.exists():
        RUNTIME_FILE.unlink(missing_ok=True)
        app.logger.warning("已清掉探不通的 runtime.json 残档")

    restore_sessions_from_disk()

    port = choose_port()
    server = make_server("127.0.0.1", port, app, threaded=True)
    SERVER_HOLDER["server"] = server
    write_runtime_file(port, os.getpid(), INSTANCE_ID)
    atexit.register(remove_runtime_file)
    app.logger.info("Jingyin server listening on http://127.0.0.1:%s", port)
    if os.environ.get("JINGYIN_NO_BROWSER") != "1":
        import webbrowser

        threading.Timer(0.7, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    try:
        server.serve_forever()
    finally:
        remove_runtime_file()
        app.logger.info("Jingyin server stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write("\n启动失败：\n")
            log.write(traceback.format_exc())
        if sys.stderr is not None:
            traceback.print_exc()
        raise
