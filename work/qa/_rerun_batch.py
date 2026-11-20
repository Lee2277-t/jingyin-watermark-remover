# -*- coding: utf-8 -*-
"""走本机 HTTP 全流程的端到端驱动（上传→自动检测→任务→轮询→ZIP）。

材料是脚本自合成的页面，不读 work/uploads 里的用户素材；端口从 work/runtime.json 读，
ZIP 只落 %TEMP% 做条目校验，不往项目目录写东西；跑完删掉自己上传的素材与成品批次。
"""

import http.cookiejar  # type: ignore
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
PAGE_COUNT = 4
BAND_REGION = {"x": 0.04, "y": 0.04, "w": 0.92, "h": 0.24}
TERMINAL = ("done", "failed", "cancelled")

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
TOKEN = ""
ROOT = ""


def request_bytes(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with opener.open(req, timeout=600) as resp:
        return resp.read(), resp.status


def get_json(url):
    body, _ = request_bytes(url)
    return json.loads(body)


def post_json(url, payload):
    body, _ = request_bytes(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-App-Token": TOKEN},
    )
    return json.loads(body)


def post_files(url, fields):
    boundary = "----qaBoundary4f2a1"
    chunks = []
    for name, filename, content in fields:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n".encode()
        )
        chunks.append(content + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body, _ = request_bytes(
        url,
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-App-Token": TOKEN},
    )
    return json.loads(body)


def make_page(index: int) -> bytes:
    """一页自合成文档：白纸 + 正文横线 + 斜向半透明蓝章 + 右上角亮块。"""
    width, height = 1200, 1600
    page = Image.new("RGB", (width, height), (246, 245, 241))
    draw = ImageDraw.Draw(page)
    y = 140
    while y < height - 60:
        x = 70 + (index * 13) % 40
        while x < width - 80:
            seg = 60 + ((x + y + index * 37) % 150)
            draw.line([x, y, min(width - 70, x + seg), y], fill=(48, 48, 52), width=3)
            x += seg + 14
        y += 46
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    stamp = ImageDraw.Draw(overlay, "RGBA")
    band = int(height * 0.24)
    for offset in range(-band, width + band, 96):
        stamp.line([(offset, 60), (offset + band, 60 + band)], fill=(28, 92, 208, 118), width=16)
    stamp.rectangle([width - 250, 40, width - 70, 104], fill=(255, 255, 255, 210))
    merged = Image.alpha_composite(page.convert("RGBA"), overlay).convert("RGB")
    scratch = Path(tempfile.mkdtemp(prefix="jingyin_e2e_"))
    try:
        out = scratch / f"page{index}.png"
        merged.save(out)
        return out.read_bytes()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def service_root() -> str:
    record = BASE / "work" / "runtime.json"
    if not record.exists():
        raise SystemExit("work/runtime.json 不在：服务没起，先用启动器或 pythonw server.py 起一次")
    return f"http://127.0.0.1:{json.loads(record.read_text(encoding='utf-8'))['port']}"


def main() -> int:
    global TOKEN, ROOT
    ROOT = service_root()
    page, _ = request_bytes(ROOT + "/")
    match = re.search(rb'name="app-token" content="([^"]+)"', page)
    if not match:
        print("未取到 app-token")
        return 1
    TOKEN = match.group(1).decode()
    print(f"服务={ROOT}（端口取自 work/runtime.json）token 已取得")

    output_dir = None
    uploaded: list[str] = []
    try:
        specs = []
        for index in range(PAGE_COUNT):
            name = f"合成页{index + 1}.png"
            result = post_files(ROOT + "/api/upload", [("files", name, make_page(index))])
            item = result["items"][0]
            uploaded.append(item["id"])
            detected = post_json(ROOT + "/api/detect", {"item_id": item["id"]})
            regions = detected.get("regions") or []
            print(f"{name}: 自动选框 {len(regions)} 处 {json.dumps(regions, ensure_ascii=False)}")
            if not regions:
                print("  自动识别没找到（合成页面的水印比真卷淡），改用已知位置的框")
                regions = [BAND_REGION]
            specs.append({"id": item["id"], "regions": regions})

        job = post_json(ROOT + "/api/process", {"items": specs, "method": "smart"})
        job_id = job["id"]
        output_dir = job.get("output_dir")
        print(f"job={job_id} -> {output_dir}")

        started = time.perf_counter()
        status = {}
        for _ in range(3000):
            status = get_json(f"{ROOT}/api/jobs/{job_id}")
            if status.get("status") in TERMINAL:
                break
            time.sleep(0.2)
        elapsed = time.perf_counter() - started
        print(
            f"状态={status.get('status')} 成功={status.get('success_count')} "
            f"失败={status.get('failed_count')} 处理耗时={elapsed:.1f}s（{len(specs)} 份）"
        )
        for task in status.get("tasks", []):
            print(f"  {task.get('name')}: {task.get('status')} {task.get('error') or ''}")
        for result in status.get("results", []):
            print(f"  成品 {result.get('name')} 说明={result.get('note')}")

        zip_bytes, _ = request_bytes(f"{ROOT}/api/jobs/{job_id}/zip")
        zip_path = Path(tempfile.gettempdir()) / f"jingyin_e2e_{job_id}.zip"
        zip_path.write_bytes(zip_bytes)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            broken = archive.testzip()
        print(f"ZIP={zip_path} {len(zip_bytes):,} B 条目={len(names)} 损坏条目={broken}")
        expect = {r["name"] for r in status.get("results", [])}
        if broken or set(names) != expect:
            print(f"FAIL 条目与 results 不一致 缺={expect - set(names)} 多={set(names) - expect}")
            return 1
        print("端到端：上传→检测→任务→轮询→ZIP 全走通，ZIP 条目与 results 一致")
        return 0
    finally:
        for item_id in uploaded:
            try:
                # 非 GET 的 /api/* 一律要 X-App-Token（§5），漏带就是 403，素材会留在盘上
                request_bytes(
                    f"{ROOT}/api/items/{item_id}",
                    method="DELETE",
                    headers={"X-App-Token": TOKEN},
                )
            except Exception as exc:  # noqa: BLE001 收尾失败不该盖掉主结果
                print(f"清理素材 {item_id} 失败：{exc}")
        if output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"已清掉本次成品批次：{output_dir}")


if __name__ == "__main__":
    sys.exit(main())
