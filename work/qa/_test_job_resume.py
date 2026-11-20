# -*- coding: utf-8 -*-
"""「继续处理剩下的素材」回归：停止之后只补没做完的那几份（第二十轮）。

判据是「续跑之后」：已完成的成品字节一个不变、结果 id 不变；剩下的全部做完；
成品仍落在同一个文件夹；打包条目与结果一一对应；被用户移走的成品会被重新处理；
用户自己放进成品夹的同名文件不被覆盖。素材全合成，不读用户作品。

跑法（需要服务在跑，且是含 resume 路由的新代码）：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_job_resume.py
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
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
QA = BASE / "work" / "qa"

W, H, FPS = 480, 270, 30
SHORT_SECONDS = 5
LONG_SECONDS = 12
CLIP = QA / "_jr_marked.mp4"
SLOW = QA / "_jr_slow.mp4"
PAGE = QA / "_jr_page.png"
FONT = "C\\\\:/Windows/Fonts/arialbd.ttf"
# 贴右下角的白字台标：画质优先必须走逐帧，才有足够时间在中间停下来
REGION = {"x": (W - 150) / W, "y": (H - 44) / H, "w": 150 / W, "h": 44 / H}
IMAGE_REGION = {"x": 0.02, "y": 0.02, "w": 0.95, "h": 0.24}
FAILED: list[str] = []
KEEP = "--keep" in sys.argv
CREATED_DIRS: list[Path] = []

JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JAR))
TOKEN = ""


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_file(path: Path) -> str:
    return md5_bytes(path.read_bytes())


def build_clip(target: Path, seconds: int) -> None:
    """480x270 合成片段：移动的分形底 + 右下贴边白字台标 + 一条真音轨。"""
    if target.exists():
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    base = (
        f"gradients=size={W}x{H}:rate={FPS}:c0=0x203040:c1=0x4a3a55:c2=0x2e4a3b:"
        f"c3=0x55402a:x0=0:y0=0:x1={W}:y1={H}:duration={seconds}"
    )
    mark = (
        f"drawtext=fontfile={FONT}:text='LIVE TV':fontcolor=white@0.82:fontsize=30:"
        f"borderw=2:bordercolor=black@0.55:x=w-tw-2:y=h-th-2"
    )
    vf = (
        f"[0:v][1:v]overlay=x='mod(t*90\\,{W})-{W // 2}':y='mod(t*60\\,{H})-{H // 2}'"
        f":eof_action=repeat,{mark}[v]"
    )
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", base,
            "-f", "lavfi", "-i", f"mandelbrot=size={W // 2}x{H // 2}:rate={FPS}",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:sample_rate=48000:duration={seconds}",
            "-filter_complex", vf,
            "-t", str(seconds), "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(target),
        ],
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


def expected_output_name(name: str, kind: str) -> str:
    """与 server.create_job 的命名规则同口径：只用于「用户自己放的同名文件」这一步。"""
    stem = Path(name).stem
    suffix = Path(name).suffix if kind == "image" else ".mp4"
    return f"{stem}_已去水印{suffix}"


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


def _message(text: str) -> str:
    """Flask 的 jsonify 会把中文转成 \\uXXXX，断言文案前先解回来。"""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"]
    return text


def post_status(root: str, url: str, payload: dict | None = None, token: bool = True):
    """POST 并返回 (状态码, 可读正文)，不抛异常。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-App-Token"] = TOKEN
    req = urllib.request.Request(
        root + url,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with OPENER.open(req, timeout=600) as resp:
            return resp.status, _message(resp.read().decode("utf-8", "replace")[:2000])
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:2000] if exc.fp else ""
        return exc.code, _message(text)


def get_status(root: str, url: str):
    """GET 并返回 (状态码, 可读正文)，不抛异常。"""
    try:
        body, code = request_bytes(root + url)
        return code, _message(body.decode("utf-8", "replace")[:2000])
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:2000] if exc.fp else ""
        return exc.code, _message(text)


def other_session_status(root: str, url: str) -> int:
    """换一个没有 cookie 的浏览器会话打同一批：必须看不到。"""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    req = urllib.request.Request(
        root + url,
        data=b"{}",
        headers={"Content-Type": "application/json", "X-App-Token": TOKEN},
        method="POST",
    )
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def upload_files(root: str, entries: list[tuple[Path, str, str]]) -> list[str]:
    boundary = "----qaResumeBoundary"
    parts: list[bytes] = []
    for path, filename, mime in entries:
        payload = path.read_bytes()
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode(),
            payload + b"\r\n",
        ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    result = json.loads(
        request_bytes(
            root + "/api/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-App-Token": TOKEN,
            },
        )[0]
    )
    return [item["id"] for item in result["items"]]


def wait_terminal(root: str, job_id: str, seconds: float = 300.0) -> dict:
    job: dict = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = get_json(f"{root}/api/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    return job


def poll_until(root: str, job_id: str, predicate, seconds: float = 180.0) -> dict:
    job: dict = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = get_json(f"{root}/api/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled") or predicate(job):
            return job
        time.sleep(0.2)
    return job


def zip_entries(root: str, job_id: str):
    body, code = request_bytes(f"{root}/api/jobs/{job_id}/zip")
    if code != 200:
        return None, code
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}, code


def start_job(root: str, specs: list[dict], video_mode: str = "quality") -> dict:
    job = post_json(
        root,
        "/api/process",
        {"items": specs, "method": "smart", "video_mode": video_mode},
    )
    CREATED_DIRS.append(Path(job["output_dir"]))
    return job


def contract_phase(root: str, fast_item: str) -> None:
    code, _ = post_status(root, "/api/jobs/nonexistent/resume")
    check("续跑：resume 路由已注册（未知任务回 404 而非 405）", code == 404, f"code={code}")

    job = start_job(root, [{"id": fast_item, "regions": [REGION]}], video_mode="fast")
    job_id = job["id"]
    check("续跑：对照组任务创建", job["status"] in ("queued", "running"), job["status"])

    code, body = post_status(root, f"/api/jobs/{job_id}/resume", token=False)
    check("续跑：不带 token 继续 → 403", code == 403, f"code={code} {body[:30]}")

    final = wait_terminal(root, job_id)
    check("续跑：对照组跑完是 done、不可续跑", final["status"] == "done" and final["resumable"] is False,
          f"{final['status']} resumable={final['resumable']}")
    code, body = post_status(root, f"/api/jobs/{job_id}/resume")
    check("续跑：已全部做完的批再点继续 → 409", code == 409 and "处理完了" in body,
          f"code={code} {body[:40]}")

    code = other_session_status(root, f"/api/jobs/{job_id}/resume")
    check("续跑：换一个浏览器会话看不到这一批 → 404", code == 404, f"code={code}")


def scenario_main(root: str, specs: list[dict]) -> None:
    """停止 → 删掉一份成品 → 放一个同名文件 → 续跑，全部做完。"""
    job = start_job(root, specs)
    out_dir = Path(job["output_dir"])
    job_id = job["id"]
    total = job["total"]
    check("续跑：混合批（4 视频 + 2 图片）创建成功", total == 6, f"total={total}")
    check(
        "续跑：新批次 resume_count=0、不可续跑",
        job["resume_count"] == 0 and job["resumable"] is False and job["remaining_count"] == total,
        f"{job['resume_count']} / {job['resumable']} / {job['remaining_count']}",
    )

    snap = poll_until(root, job_id, lambda j: j["success_count"] >= 3)
    check(
        "续跑：停止前已有成品（图片已出完 + 一份视频）",
        snap["success_count"] >= 3 and snap["status"] in ("queued", "running"),
        f"成功 {snap['success_count']}/{total} {snap['status']}",
    )
    post_status(root, f"/api/jobs/{job_id}/cancel")
    stopped = wait_terminal(root, job_id)
    check("续跑：第一次停止后状态 cancelled", stopped["status"] == "cancelled", stopped["status"])
    check(
        "续跑：停下后可续跑、剩余份数对得上",
        stopped["resumable"] is True and stopped["remaining_count"] == total - stopped["success_count"],
        f"resumable={stopped['resumable']} 剩余={stopped['remaining_count']} 成功={stopped['success_count']}",
    )
    check(
        "续跑：被停止的份不计失败",
        stopped["failed_count"] == 0 and stopped["cancelled_count"] >= 1,
        f"停止 {stopped['cancelled_count']} 失败 {stopped['failed_count']}",
    )

    carried = {r["id"]: r for r in stopped["results"]}
    carried_md5 = {rid: md5_file(out_dir / r["name"]) for rid, r in carried.items()}
    check(
        "续跑：已完成的成品都在磁盘上",
        bool(carried) and all(
            (out_dir / r["name"]).is_file() and (out_dir / r["name"]).stat().st_size > 0
            for r in carried.values()
        ),
        f"{len(carried)} 份",
    )

    # 用户把一份图片成品连同预览一起从文件夹里移走：续跑应当把它补回来
    image_result = next((r for r in stopped["results"] if r["kind"] == "image"), None)
    old_preview = out_dir / f".preview_{image_result['id']}.jpg" if image_result else Path("")
    check("续跑：图片成品带一份预览缓存", bool(image_result) and old_preview.is_file(), old_preview.name)
    dropped_task_id = image_result["task_id"] if image_result else ""
    if image_result:
        (out_dir / image_result["name"]).unlink()

    # 用户自己往成品夹里放一个同名文件：续跑不能覆盖它
    pending_video = next(
        (t for t in stopped["tasks"] if t["status"] == "cancelled" and t["kind"] == "video"),
        None,
    )
    stray = b"user's own file, must stay untouched"
    stray_path = out_dir / expected_output_name(pending_video["name"], "video") if pending_video else None
    if stray_path is not None:
        stray_path.write_bytes(stray)
    check("续跑：准备好同名占位文件", stray_path is not None and stray_path.is_file(),
          stray_path.name if stray_path else "没有可用的排队视频")

    resumed = post_json(root, f"/api/jobs/{job_id}/resume", {})
    check(
        "续跑：续跑请求被受理（同一批、同一成品目录）",
        resumed["id"] == job_id and resumed["output_dir"] == stopped["output_dir"],
        Path(resumed["output_dir"]).name,
    )
    check(
        "续跑：上一次的「跑一半被停」标记已清零（界面清单不会拿旧状态说话）",
        all(t["started"] is False for t in resumed["tasks"] if t["status"] != "done"),
        str([t["started"] for t in resumed["tasks"] if t["status"] != "done"]),
    )
    check(
        "续跑：resume_count 记为 1、状态回到处理中",
        resumed["resume_count"] == 1 and resumed["status"] in ("queued", "running"),
        f"{resumed['resume_count']} / {resumed['status']}",
    )
    check(
        "续跑：剩余份数把被移走的那份算回来",
        resumed["remaining_count"] == total - stopped["success_count"] + 1,
        str(resumed["remaining_count"]),
    )
    check("续跑：处理中不能打包", resumed["zip_ready"] is False, str(resumed["zip_ready"]))
    code, body = post_status(root, f"/api/jobs/{job_id}/resume")
    check("续跑：处理中再点继续 → 409", code == 409 and "还在处理" in body,
          f"code={code} {body[:40]}")

    done = wait_terminal(root, job_id)
    check("续跑：全部做完后状态 done", done["status"] == "done", done["status"])
    check(
        "续跑：成功份数回到总份数（被移走的那份重新处理了）",
        done["success_count"] == total and done["failed_count"] == 0 and done["cancelled_count"] == 0,
        f"成功 {done['success_count']} 失败 {done['failed_count']} 停止 {done['cancelled_count']} / {total}",
    )
    check("续跑：resume_count 仍是 1", done["resume_count"] == 1, str(done["resume_count"]))
    check(
        "续跑：结果数 = 总份数、每份只出现一次",
        len(done["results"]) == total and len({r["task_id"] for r in done["results"]}) == total,
        f"{len(done['results'])} 条",
    )
    task_order = [t["id"] for t in done["tasks"]]
    check(
        "续跑：结果仍按任务原序落位",
        [r["task_id"] for r in done["results"]] == task_order,
        "顺序一致" if [r["task_id"] for r in done["results"]] == task_order else "顺序乱了",
    )

    by_id = {r["id"]: r for r in done["results"]}
    kept = [rid for rid in carried if rid != (image_result["id"] if image_result else "")]
    kept_ok = all(
        rid in by_id
        and by_id[rid]["name"] == carried[rid]["name"]
        and md5_file(out_dir / by_id[rid]["name"]) == carried_md5[rid]
        for rid in kept
    )
    check("续跑：续跑前就完成的成品没被重跑（结果 id 与字节都不变）", bool(kept) and kept_ok,
          f"{len(kept)} 份对照")

    healed = next((r for r in done["results"] if r["task_id"] == dropped_task_id), None)
    new_preview = out_dir / f".preview_{healed['id']}.jpg" if healed else Path("")
    check(
        "续跑：被移走的那份用新结果补回来",
        bool(healed) and healed["id"] != (image_result["id"] if image_result else ""),
        healed["name"] if healed else "没补回来",
    )
    check(
        "续跑：旧预览清掉、新预览就位",
        bool(healed) and not old_preview.exists() and new_preview.is_file(),
        f"{old_preview.name} → {new_preview.name}",
    )

    if stray_path is not None:
        base_name = stray_path.name
        stem, _, suffix = base_name.rpartition(".")
        renamed = f"{stem}_2.{suffix}"
        check("续跑：用户自己放的同名文件没被覆盖",
              stray_path.is_file() and stray_path.read_bytes() == stray, base_name)
        check("续跑：撞名的成品自动改名（_2）",
              any(r["name"] == renamed for r in done["results"]), renamed)

    files_ok = all(
        (out_dir / r["name"]).is_file() and (out_dir / r["name"]).stat().st_size > 0
        for r in done["results"]
    )
    parts = [p.name for p in out_dir.glob("*part*")]
    check("续跑：每个结果都在磁盘上、没有 .part 残档", files_ok and not parts, str(parts[:3]))

    entries, zip_code = zip_entries(root, job_id)
    want_names = sorted(r["name"] for r in done["results"])
    check(
        "续跑：续跑后打包 200 且条目 = 结果名",
        zip_code == 200 and entries is not None and sorted(entries) == want_names,
        f"code={zip_code} {len(entries) if entries else 0} 条",
    )
    check(
        "续跑：ZIP 里每个条目与磁盘字节一致",
        bool(entries) and all(
            md5_bytes(entries[r["name"]]) == md5_file(out_dir / r["name"]) for r in done["results"]
        ),
        "逐条目 md5 相同" if entries else "没有条目",
    )
    check(
        "续跑：所有任务都是 done、没有错误文案",
        all(t["status"] == "done" and t.get("error") is None for t in done["tasks"]),
        str([t["status"] for t in done["tasks"]]),
    )
    check("续跑：终态说明是「处理完成」口径", done["message"].startswith("处理完成"), done["message"][:50])
    code, body = post_status(root, f"/api/jobs/{job_id}/resume")
    check("续跑：全部做完后再点继续 → 409", code == 409 and "处理完了" in body,
          f"code={code} {body[:40]}")


def scenario_repeat(root: str, specs: list[dict]) -> None:
    """停 → 续 → 再停 → 再续：续跑次数要如实累加，最后一批做完。"""
    job = start_job(root, specs)
    job_id = job["id"]
    total = job["total"]

    poll_until(root, job_id, lambda j: j["success_count"] >= 1)
    post_status(root, f"/api/jobs/{job_id}/cancel")
    first = wait_terminal(root, job_id)
    check(
        "续跑：第一次停止后可续、resume_count 仍为 0",
        first["status"] == "cancelled" and first["resumable"] is True and first["resume_count"] == 0,
        f"{first['status']} resumable={first['resumable']}",
    )

    post_json(root, f"/api/jobs/{job_id}/resume", {})
    second = poll_until(root, job_id, lambda j: j["success_count"] >= 2)
    post_status(root, f"/api/jobs/{job_id}/cancel")
    again = wait_terminal(root, job_id)
    check(
        "续跑：续跑途中再停仍记 resume_count=1",
        again["status"] == "cancelled" and again["resume_count"] == 1,
        f"{again['status']} resume_count={again['resume_count']}",
    )
    check(
        "续跑：第二次停止时已完成数比第一次多",
        again["success_count"] > first["success_count"],
        f"{first['success_count']} → {again['success_count']}",
    )
    check("续跑：第二次停下仍可继续", again["resumable"] is True and again["remaining_count"] >= 1,
          f"剩余 {again['remaining_count']}")

    post_json(root, f"/api/jobs/{job_id}/resume", {})
    done = wait_terminal(root, job_id)
    check(
        "续跑：连着续两次后全部做完",
        done["status"] == "done" and done["success_count"] == total and done["resume_count"] == 2,
        f"{done['status']} 成功 {done['success_count']}/{total} resume_count={done['resume_count']}",
    )
    check(
        "续跑：结果与材料一一对应",
        len(done["results"]) == total and len({r["task_id"] for r in done["results"]}) == total,
        f"{len(done['results'])} 条",
    )
    check(
        "续跑：做完后界面上那份清单不会再列出任何东西",
        not [t for t in done["tasks"] if t["status"] == "cancelled"]
        and not [t for t in done["tasks"] if t["started"]],
        f"started 为真 {sum(1 for t in done['tasks'] if t['started'])} 份",
    )
    entries, zip_code = zip_entries(root, job_id)
    check(
        "续跑：两次续跑后打包仍完整",
        zip_code == 200 and entries is not None and sorted(entries) == sorted(r["name"] for r in done["results"]),
        f"code={zip_code} {len(entries) if entries else 0} 条",
    )
    check("续跑：第二次续跑没有多出停止计数", done["cancelled_count"] == 0 and done["failed_count"] == 0,
          f"停止 {done['cancelled_count']} 失败 {done['failed_count']}")


def scenario_empty(root: str, spec: dict) -> None:
    """一份都没做完就停：仍然可以继续，续跑后真的出成品。"""
    job = start_job(root, [spec])
    job_id = job["id"]
    post_status(root, f"/api/jobs/{job_id}/cancel")
    stopped = wait_terminal(root, job_id)
    check(
        "续跑：一份没做完就停也能续跑",
        stopped["status"] == "cancelled" and not stopped["results"] and stopped["resumable"] is True,
        f"{stopped['status']} 成品 {len(stopped['results'])} 份",
    )
    if not stopped["results"]:
        code, _ = get_status(root, f"/api/jobs/{job_id}/zip")
        check("续跑：没有成品时打包仍回 409", code == 409, f"code={code}")

    post_json(root, f"/api/jobs/{job_id}/resume", {})
    done = wait_terminal(root, job_id)
    check(
        "续跑：没有成品的批次续跑后出了成品",
        done["status"] == "done" and done["success_count"] == 1 and len(done["results"]) == 1,
        f"{done['status']} 成功 {done['success_count']}",
    )
    entries, zip_code = zip_entries(root, job_id)
    check(
        "续跑：续跑后打包可用",
        zip_code == 200 and entries is not None and len(entries) == 1,
        f"code={zip_code} {len(entries) if entries else 0} 条",
    )


def http_phase(root: str) -> None:
    global TOKEN
    page, _ = request_bytes(root + "/")
    match = re.search(rb'name="app-token" content="([^"]+)"', page)
    if not match:
        check("HTTP：取到 app-token", False)
        return
    TOKEN = match.group(1).decode()

    videos = upload_files(
        root,
        [(CLIP, f"台标片段 {index + 1}.mp4", "video/mp4") for index in range(9)]
        + [(SLOW, "长片段 1.mp4", "video/mp4")],
    )
    pages = upload_files(
        root,
        [(PAGE, f"样张 {index + 1}.png", "image/png") for index in range(2)],
    )
    check("HTTP：上传 10 段合成视频 + 2 张合成页", len(videos) == 10 and len(pages) == 2,
          f"{len(videos)} + {len(pages)}")
    if len(videos) != 10 or len(pages) != 2:
        return

    video_specs = [{"id": item_id, "regions": [REGION]} for item_id in videos]
    page_specs = [{"id": item_id, "regions": [IMAGE_REGION]} for item_id in pages]
    contract_phase(root, videos[9])
    # 图片与视频交错，验证续跑后结果仍按任务原序落位
    mixed = [video_specs[0], page_specs[0], video_specs[1], page_specs[1], video_specs[2], video_specs[3]]
    scenario_main(root, mixed)
    scenario_repeat(root, [video_specs[4], video_specs[5], video_specs[6], video_specs[7]])
    scenario_empty(root, video_specs[8])

    uploads = BASE / "work" / "uploads"
    for item_id in videos + pages:
        for item_dir in uploads.glob(f"*/{item_id}"):
            shutil.rmtree(item_dir, ignore_errors=True)
    for sid_dir in uploads.iterdir():
        if sid_dir.is_dir() and not any(sid_dir.iterdir()):
            sid_dir.rmdir()


def main() -> int:
    build_clip(CLIP, SHORT_SECONDS)
    build_clip(SLOW, LONG_SECONDS)
    build_page()
    runtime = BASE / "work" / "runtime.json"
    if not runtime.exists():
        check("HTTP：runtime.json 存在（服务没在跑）", False)
    else:
        port = json.loads(runtime.read_text(encoding="utf-8"))["port"]
        try:
            http_phase(f"http://127.0.0.1:{port}")
        finally:
            for folder in CREATED_DIRS:
                if folder.parent == BASE / "成品":
                    shutil.rmtree(folder, ignore_errors=True)
    if not KEEP:
        CLIP.unlink(missing_ok=True)
        SLOW.unlink(missing_ok=True)
        PAGE.unlink(missing_ok=True)
    print("\n成品批次已清理。保留夹具请加 --keep")
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
