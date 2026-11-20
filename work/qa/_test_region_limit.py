"""选框上限回归：一份素材超过 16 个框必须明确报错，不能再静默丢掉第 17 个以后的框。

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_region_limit.py

同时核对前端 static/app.js 里的同一道上限（画不上去 + 提示），以及 regionCount 显示 n/16。
临时产物一律落在系统临时目录，跑完自动清理。
"""

from __future__ import annotations

import io
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


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def region(index: int) -> dict:
    step = 0.004
    return {"x": step, "y": step + index * 0.02, "w": 0.5, "h": 0.012}


def regions(count: int) -> list[dict]:
    return [region(i) for i in range(count)]


def make_source() -> bytes:
    image = Image.new("RGB", (240, 640), (246, 244, 239))
    for row in range(0, 620, 40):
        for column in range(0, 240, 40):
            image.paste((70, 110, 200), (column + 4, row + 4, column + 26, row + 26))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def unit_checks() -> None:
    limit = server.MAX_REGIONS_PER_ITEM
    check("上限常量仍是 16", limit == 16, f"实际 {limit}")

    try:
        server.normalize_regions(regions(limit + 1), owner="测试页.png")
        check("17 个框抛 RegionLimitError", False, "没有抛错")
    except server.RegionLimitError as exc:
        text = str(exc)
        check(
            "17 个框抛 RegionLimitError",
            "测试页.png" in text and "17" in text and "16" in text,
            text,
        )

    check("16 个框正常收下", len(server.normalize_regions(regions(limit))) == limit)
    check("不是列表时按空处理", server.normalize_regions(None) == [])


def http_checks() -> None:
    limit = server.MAX_REGIONS_PER_ITEM
    temp_dir = Path(tempfile.mkdtemp(prefix="jingyin_region_"))
    old_output_root = server.OUTPUT_ROOT
    server.OUTPUT_ROOT = temp_dir / "成品"
    client = server.app.test_client()
    item_id = ""
    sid = ""
    try:
        client.get("/")
        with client.session_transaction() as sess:
            sid = sess.get("sid", "")
        upload = client.post(
            "/api/upload",
            data={"files": (io.BytesIO(make_source()), "测试页.png")},
            content_type="multipart/form-data",
            headers={"X-App-Token": server.APP_TOKEN},
        )
        check("测试素材上传成功", upload.status_code == 200, str(upload.status_code))
        item_id = upload.get_json()["items"][0]["id"]

        over = client.post(
            "/api/process",
            json={"method": "smart", "items": [{"id": item_id, "regions": regions(limit + 3)}]},
            headers={"X-App-Token": server.APP_TOKEN},
        )
        payload = over.get_json() or {}
        text = payload.get("error", "")
        check(
            "17+ 个框被明确拒绝",
            over.status_code == 400 and "19" in text and "测试页.png" in text,
            f"{over.status_code} {text}",
        )

        ok = client.post(
            "/api/process",
            json={"method": "smart", "items": [{"id": item_id, "regions": regions(limit)}]},
            headers={"X-App-Token": server.APP_TOKEN},
        )
        check("16 个框照常开跑", ok.status_code == 200, f"{ok.status_code} {ok.get_json()}")
        job_id = (ok.get_json() or {}).get("id", "")
        deadline = time.time() + 60
        status = ""
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").get_json() or {}
            status = job.get("status", "")
            if status in ("done", "failed"):
                break
            time.sleep(0.2)
        check("16 个框跑完并出成品", status == "done" and job.get("success_count") == 1, str(job))

        bad = client.post(
            "/api/process",
            json={"method": "smart", "items": [{"id": item_id, "regions": regions(1)}]},
            headers={"X-App-Token": "wrong"},
        )
        check("缺令牌仍然先被 403 拦住", bad.status_code == 403, str(bad.status_code))
    finally:
        server.OUTPUT_ROOT = old_output_root
        shutil.rmtree(temp_dir, ignore_errors=True)
        # 上传的测试素材会进 work/uploads，_check.py 的语料因此不能留下残渣
        if item_id:
            for path in server.UPLOAD_ROOT.glob(f"*/{item_id}"):
                shutil.rmtree(path, ignore_errors=True)
        if sid:
            empty = server.UPLOAD_ROOT / sid
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()


def frontend_checks() -> None:
    text = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    check("前端有同一道 16 上限常量", "const MAX_REGIONS = 16;" in text)
    check("画满时拒绝再画新框并提示", "已经画满了" in text)
    check("自动识别加满后不再静默吞掉", "处没能加上" in text)
    check("选框计数显示 n/16", "${regions.length}/${MAX_REGIONS}" in text)
    check("前端不再出现裸写死的 slice(0, 16)", ".slice(0, 16)" not in text)


def main() -> int:
    unit_checks()
    http_checks()
    frontend_checks()
    print("\n结果：" + ("全部通过" if not FAILURES else f"{len(FAILURES)} 项不通过：{FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
