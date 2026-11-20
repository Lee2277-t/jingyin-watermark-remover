"""单份重跑回归：只重做一条成品，同批其他份数逐字节不动；失败时原文件还在。

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_rerun.py

要证伪的四件事：
- 重跑**只动那一条**：同批另一份的下载内容逐字节相同（比 md5，不比文件大小）；
- 重跑**用界面传来的新选框与档位**，不是把旧参数再跑一遍；
- 重跑**作废这批的 ZIP 缓存**——ZIP 复用只比条目名，名字不变就看不出来；
- 重跑**不毁掉已有成品**：素材被回收时回 410 且原文件仍可下载，失败路径先写临时名。

临时产物一律落在系统临时目录，跑完自动清理。
"""

from __future__ import annotations

import hashlib
import io
import zipfile
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import server  # noqa: E402

FAILURES: list[str] = []
REGION_A = {"x": 0.06, "y": 0.06, "w": 0.42, "h": 0.10}
REGION_B = {"x": 0.06, "y": 0.30, "w": 0.60, "h": 0.16}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_source(seed: int) -> bytes:
    """带纹理的页面：smart（纹理）与 fill（内容填充）在上面出来的字节不该相同。"""
    image = Image.new("RGB", (360, 480), (246, 244, 239))
    for row in range(0, 456, 18):
        for column in range(0, 336, 18):
            tone = (60 + (row * 7 + seed * 31) % 120, 90 + (column * 5) % 90, 150)
            image.paste(tone, (column + 3, row + 3, column + 15, row + 15))
    image.paste((30, 30, 30), (20 + seed * 40, 20, 150 + seed * 40, 44))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def wait_final(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json() or {}
        if job.get("status") in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.15)
    return job


def digest(client, job_id: str, result_id: str) -> str:
    response = client.get(f"/api/jobs/{job_id}/results/{result_id}/file?download=1")
    return hashlib.md5(response.data).hexdigest() if response.status_code == 200 else ""


def frontend_checks() -> None:
    text = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    source = (ROOT / "server.py").read_text(encoding="utf-8")
    check("结果卡片上有「重跑这份」", "重跑这份" in text)
    check("按钮打的是单份重跑接口", "/results/${encodeURIComponent(result.id)}/rerun" in text)
    check("结果指纹带上尺寸与作业状态", "result.size" in text and "job.status" in text)
    check("服务端注册了单份重跑路由", "/results/<result_id>/rerun" in source)
    check("重跑走临时名、成功才替换", "_rerun{output_path.suffix}" in source and "os.replace(temp_output" in source)
    check("重跑后作废本批 ZIP 缓存", 'ZIP_DIR / f"{job_id}.zip"' in source)


def http_checks() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="jingyin_rerun_"))
    old_output_root = server.OUTPUT_ROOT
    server.OUTPUT_ROOT = temp_dir / "成品"
    client = server.app.test_client()
    headers = {"X-App-Token": server.APP_TOKEN}
    item_ids: list[str] = []
    sid = ""
    job_id = ""
    cache = None
    try:
        client.get("/")
        with client.session_transaction() as sess:
            sid = sess.get("sid", "")

        upload = client.post(
            "/api/upload",
            data={
                "files": [
                    (io.BytesIO(make_source(0)), "甲页.png"),
                    (io.BytesIO(make_source(1)), "乙页.png"),
                ]
            },
            content_type="multipart/form-data",
            headers=headers,
        )
        items = (upload.get_json() or {}).get("items", [])
        check("两份素材上传成功", upload.status_code == 200 and len(items) == 2, str(upload.status_code))
        if len(items) != 2:
            return
        item_ids = [item["id"] for item in items]

        started = client.post(
            "/api/process",
            json={
                "method": "smart",
                "items": [
                    {"id": item_ids[0], "regions": [REGION_A]},
                    {"id": item_ids[1], "regions": [REGION_A]},
                ],
            },
            headers=headers,
        )
        job_id = (started.get_json() or {}).get("id", "")
        check("批次开跑", started.status_code == 200 and bool(job_id), str(started.status_code))
        job = wait_final(client, job_id)
        check("两份都出了成品", job.get("status") == "done" and len(job.get("results", [])) == 2, str(job.get("message")))
        results = job.get("results", [])
        if len(results) != 2:
            return
        first, second = results[0], results[1]
        check(
            "结果里带 item_id，界面能对上这份的选框",
            {first.get("item_id"), second.get("item_id")} == set(item_ids),
            f"{first.get('item_id')} / {second.get('item_id')}",
        )

        before_first = digest(client, job_id, first["id"])
        before_second = digest(client, job_id, second["id"])
        check("两份成品都下载得到", bool(before_first) and bool(before_second))

        # 先取一次 ZIP，把缓存落盘——重跑必须让它作废
        zipped = client.get(f"/api/jobs/{job_id}/zip")
        cache = server.ZIP_DIR / f"{job_id}.zip"
        check("首次打包成功且缓存落盘", zipped.status_code == 200 and cache.exists(), str(zipped.status_code))
        # Windows 上文件句柄没释放就删不掉，这里显式关掉，免得把「删不掉」误判成「没删」
        zipped.close()

        rerun_url = f"/api/jobs/{job_id}/results/{first['id']}/rerun"
        rerun = client.post(rerun_url, json={"method": "fill", "regions": [REGION_B]}, headers=headers)
        check("重跑请求被接受", rerun.status_code == 200, f"{rerun.status_code} {rerun.get_json()}")
        payload = rerun.get_json() or {}
        check("受理时立刻进入 running", payload.get("status") == "running", str(payload.get("status")))
        check("受理文案点名是哪一份", "正在重跑" in (payload.get("message") or ""), payload.get("message"))

        after = wait_final(client, job_id)
        check("重跑后仍然只有两份", len(after.get("results", [])) == 2, str(len(after.get("results", []))))
        check("重跑结束后作业回到 done", after.get("status") == "done", str(after.get("status")))
        check("结束文案写「已重跑」", "已重跑" in (after.get("message") or ""), after.get("message"))

        after_first = digest(client, job_id, first["id"])
        after_second = digest(client, job_id, second["id"])
        check("目标那份内容确实变了（换了档位与选框）", after_first and after_first != before_first)
        check("同批另一份逐字节没动", after_second == before_second, f"{before_second[:8]} -> {after_second[:8]}")
        check("ZIP 缓存被作废", not cache.exists())

        rebuilt = client.get(f"/api/jobs/{job_id}/zip")
        check("重新下载的 ZIP 可读且仍是两份", rebuilt.status_code == 200 and rebuilt.data[:2] == b"PK")
        # 光「缓存被删」还不够：新包里的字节必须真是重跑后的那份
        with zipfile.ZipFile(io.BytesIO(rebuilt.data)) as archive:
            entry = archive.read(first["name"])
        check("新 ZIP 里装的是重跑后的内容", hashlib.md5(entry).hexdigest() == after_first)
        rebuilt.close()

        missing = client.post(
            f"/api/jobs/{job_id}/results/deadbeef0000/rerun", json={"method": "smart"}, headers=headers
        )
        check("找不到的成品回 404", missing.status_code == 404, str(missing.status_code))

        too_many = client.post(
            rerun_url,
            json={"method": "smart", "regions": [REGION_A] * (server.MAX_REGIONS_PER_ITEM + 1)},
            headers=headers,
        )
        check("超框上限仍回 400", too_many.status_code == 400, str(too_many.status_code))

        with server.INDEX_LOCK:
            server.JOBS[job_id]["status"] = "running"
        busy = client.post(rerun_url, json={"method": "smart"}, headers=headers)
        with server.INDEX_LOCK:
            server.JOBS[job_id]["status"] = "done"
        check("这一批还在跑时拒绝重跑", busy.status_code == 409, str(busy.status_code))

        # 素材被回收：回 410，且手上那份成品一个字节都不许动
        task = next(
            (entry for entry in server.JOBS[job_id]["tasks"] if entry["id"] == first["task_id"]), None
        )
        survived = digest(client, job_id, first["id"])
        if task is not None:
            Path(task["source_path"]).unlink(missing_ok=True)
        gone = client.post(rerun_url, json={"method": "smart"}, headers=headers)
        check("素材被回收时回 410", gone.status_code == 410, str(gone.status_code))
        check("素材没了也不动已有成品", digest(client, job_id, first["id"]) == survived)
    finally:
        server.OUTPUT_ROOT = old_output_root
        if cache is not None:
            try:
                cache.unlink()
            except OSError:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)
        for item_id in item_ids:
            for path in server.UPLOAD_ROOT.glob(f"*/{item_id}"):
                shutil.rmtree(path, ignore_errors=True)
        if sid:
            empty = server.UPLOAD_ROOT / sid
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()


def main() -> int:
    frontend_checks()
    http_checks()
    print("\n结果：" + ("全部通过" if not FAILURES else f"{len(FAILURES)} 项不通过：{FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
