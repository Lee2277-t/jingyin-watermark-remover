# -*- coding: utf-8 -*-
"""端到端验证「视频速度优先 / 画质优先」开关：走本机 HTTP 全流程，不直接调引擎。

夹具用 _video_regression.py 的合成台标片段（贴右边 + 贴下边），先 --keep 生成：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_video_regression.py --keep

断言：video_mode 从请求体贯通到 job/task/成品说明；两种模式各走各的路线；非法取值归回默认。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
import http.cookiejar
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CLIP = BASE / "work" / "qa" / "_vr_marked.mp4"
PORT = json.loads((BASE / "work" / "runtime.json").read_text(encoding="utf-8"))["port"]
ROOT = f"http://127.0.0.1:{PORT}"

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
TOKEN = ""
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def request_bytes(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with opener.open(req, timeout=600) as resp:
        return resp.read(), resp.status


def get_json(url):
    return json.loads(request_bytes(url)[0])


def post_json(url, payload):
    body, _ = request_bytes(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-App-Token": TOKEN},
    )
    return json.loads(body)


def upload(path: Path) -> str:
    boundary = "----qaVideoBoundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
        "Content-Type: video/mp4\r\n\r\n".encode(),
        path.read_bytes() + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    result = json.loads(
        request_bytes(
            ROOT + "/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": TOKEN},
        )[0]
    )
    if not result["items"]:
        raise SystemExit(f"上传失败：{result['errors']}")
    return result["items"][0]["id"]


def run_job(item_id: str, video_mode: str | None) -> dict:
    payload = {
        "items": [
            {
                "id": item_id,
                # 故意贴到画面右下角，逼出两种模式的分支
                "regions": [{"x": 0.78, "y": 0.86, "w": 0.22, "h": 0.14}],
            }
        ],
        "method": "smart",
    }
    if video_mode is not None:
        payload["video_mode"] = video_mode
    started = time.perf_counter()
    job = post_json(ROOT + "/api/process", payload)
    for _ in range(600):
        status = get_json(f"{ROOT}/api/jobs/{job['id']}")
        if status.get("status") in ("done", "failed"):
            break
        time.sleep(0.2)
    status["elapsed"] = time.perf_counter() - started
    return status


def main() -> int:
    global TOKEN
    page, _ = request_bytes(ROOT + "/")
    match = re.search(rb'name="app-token" content="([^"]+)"', page)
    if not match:
        print("未取到 app-token")
        return 1
    TOKEN = match.group(1).decode()
    if not CLIP.exists():
        print(f"缺少夹具 {CLIP}，请先跑 _video_regression.py --keep")
        return 1

    item_id = upload(CLIP)
    print(f"上传完成 item={item_id}")

    outputs: list[Path] = []
    try:
        quality = run_job(item_id, "quality")
        fast = run_job(item_id, "fast")
        default = run_job(item_id, None)
        bogus = run_job(item_id, "whatever")

        for label, status, expect_mode, expect_note in (
            ("画质优先", quality, "quality", "触到画面边缘"),
            ("速度优先", fast, "fast", "速度优先"),
            ("不传参数（应默认画质）", default, "quality", "触到画面边缘"),
            ("非法取值（应归回画质）", bogus, "quality", "触到画面边缘"),
        ):
            outputs.append(Path(status["output_dir"]))
            results = status.get("results") or []
            tasks = status.get("tasks") or []
            check(f"{label}：任务成功", status.get("success_count") == 1 and status.get("failed_count") == 0,
                  f"status={status.get('status')} 成功={status.get('success_count')} 失败={status.get('failed_count')} "
                  f"错误={tasks and tasks[0].get('error')}")
            check(f"{label}：job 回显 video_mode={expect_mode}", status.get("video_mode") == expect_mode,
                  str(status.get("video_mode")))
            if results:
                check(f"{label}：成品说明含「{expect_note}」", expect_note in str(results[0].get("note", "")),
                      str(results[0].get("note", ""))[:100])
                check(f"{label}：成品是 mp4", str(results[0].get("kind")) == "video", str(results[0].get("kind")))
                file_bytes, _ = request_bytes(ROOT + results[0]["download_url"])
                check(f"{label}：成品可下载且非空", len(file_bytes) > 10_000, f"{len(file_bytes):,} B")
            else:
                check(f"{label}：有成品", False, str(status.get("message")))

        print(
            f"\n端到端耗时：画质优先 {quality['elapsed']:.1f}s / 速度优先 {fast['elapsed']:.1f}s "
            f"（含上传解码与 ZIP 前的整段任务时间）"
        )
    finally:
        # 测试成品不留在交付目录，测试素材不留在语料目录
        for folder in outputs:
            if folder.exists() and folder.parent == BASE / "成品":
                shutil.rmtree(folder, ignore_errors=True)
        uploads = BASE / "work" / "uploads"
        for item_dir in uploads.glob(f"*/{item_id}"):
            shutil.rmtree(item_dir, ignore_errors=True)
        for sid_dir in uploads.iterdir():
            if sid_dir.is_dir() and not any(sid_dir.iterdir()):
                sid_dir.rmdir()
    print("已清理测试成品与测试素材。")
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
