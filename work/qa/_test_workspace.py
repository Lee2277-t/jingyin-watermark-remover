"""工作区持久化回归：关掉浏览器或重启服务后，素材列表与手动画的框必须还在。

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_test_workspace.py

盯的是四件事：① 每条素材的索引落盘成 meta.json；② 启动时扫 work/uploads 能原样装回内存，
**item_id 一字不变**（浏览器的选框就是按它索引的，变了框就成孤儿）；③ cookie 是持久的、
签名密钥落在 work/secret.key，所以换一次进程旧 cookie 仍然有效；④ 删掉的素材不会被复活。
测试素材与临时目录跑完自清，不进 _check.py 语料。
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from inspect import getsource
from pathlib import Path
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import server  # noqa: E402
from remover import state_store  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_source() -> bytes:
    image = Image.new("RGB", (160, 120), (246, 244, 239))
    image.paste((70, 110, 200), (10, 10, 60, 40))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def item_stub(item_id: str, source: Path) -> dict:
    return {
        "id": item_id,
        "name": "测试页.png",
        "kind": "image",
        "suffix": ".png",
        "width": 160,
        "height": 120,
        "duration": None,
        "fps": None,
        "frame_count": None,
        "source_path": source,
        "thumb_path": source.parent / "thumb.jpg",
        "preview_path": source.parent / "preview.jpg",
        "created_at": 1700000000.0,
    }


def unit_checks() -> None:
    temp = Path(tempfile.mkdtemp(prefix="jingyin_ws_"))
    try:
        sid = uuid.uuid4().hex
        item_id = uuid.uuid4().hex[:16]
        item_dir = temp / sid / item_id
        item_dir.mkdir(parents=True)
        source = item_dir / "source.png"
        source.write_bytes(make_source())
        (item_dir / "thumb.jpg").write_bytes(b"jpeg")

        original = item_stub(item_id, source)
        state_store.write_item_meta(original, item_dir)
        meta = item_dir / state_store.META_NAME
        check("meta.json 已落盘", meta.is_file(), str(meta))
        check("落盘不留 .part 残档", not list(item_dir.glob("*.part")))

        loaded = state_store.read_item_meta(meta)
        check(
            "读回的索引与上传时同一条",
            loaded is not None
            and loaded["id"] == item_id
            and loaded["name"] == "测试页.png"
            and isinstance(loaded["source_path"], Path)
            and loaded["source_path"] == source,
            str(loaded)[:120],
        )

        # 源文件被删 → 这条索引必须作废，而不是恢复出一个打不开的空条目
        os.remove(source)
        check("源文件没了就不恢复这条", state_store.read_item_meta(meta) is None)

        meta.write_text("{ 坏掉的 JSON", encoding="utf-8")
        check("meta 损坏时返回 None 不抛", state_store.read_item_meta(meta) is None)

        # 整盘重建
        good = temp / uuid.uuid4().hex
        good_item = good / uuid.uuid4().hex[:16]
        good_item.mkdir(parents=True)
        src = good_item / "source.png"
        src.write_bytes(make_source())
        state_store.write_item_meta(item_stub(good_item.name, src), good_item)
        junk = temp / "not-a-hex-sid"
        (junk / uuid.uuid4().hex[:16]).mkdir(parents=True)

        sessions: dict = {}
        count_sessions, count_items = state_store.rebuild_sessions(good.parent, sessions, server.INDEX_LOCK)
        check("按目录重建出 1 个会话 1 份素材", (count_sessions, count_items) == (1, 1), f"{count_sessions}/{count_items}")
        check("item_id 与目录名一致才收下", good.name in sessions and sessions[good.name]["items"].keys() == {good_item.name})
        check("非十六进制目录被跳过", all(not k.startswith("not-a-hex") for k in sessions))

        # 内存里已有的会话只补缺、不覆盖
        sessions[good.name]["items"][good_item.name] = {"id": good_item.name, "marker": "内存版"}
        state_store.rebuild_sessions(good.parent, sessions, server.INDEX_LOCK)
        check("已有条目不被磁盘版覆盖", sessions[good.name]["items"][good_item.name].get("marker") == "内存版")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def http_checks() -> None:
    client = server.app.test_client()
    first = client.get("/")
    set_cookie = first.headers.get("Set-Cookie", "")
    check(
        "会话 cookie 是持久的（关浏览器不灭）",
        "Max-Age" in set_cookie or "Expires" in set_cookie,
        set_cookie.split(";")[1].strip() if ";" in set_cookie else set_cookie,
    )

    key_file = ROOT / "work" / "secret.key"
    saved = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
    check("签名密钥已落盘且是 64 位十六进制", len(saved) == 64 and all(c in "0123456789abcdef" for c in saved))

    with client.session_transaction() as sess:
        sid = sess.get("sid", "")
    upload = client.post(
        "/api/upload",
        data={"files": (io.BytesIO(make_source()), "持久化测试页.png")},
        content_type="multipart/form-data",
        headers={"X-App-Token": server.APP_TOKEN},
    )
    check("测试素材上传成功", upload.status_code == 200, str(upload.get_json())[:120])
    item_id = (upload.get_json() or {}).get("items", [{}])[0].get("id", "")
    meta = server.UPLOAD_ROOT / sid / item_id / state_store.META_NAME
    check("上传即落盘 meta.json", bool(item_id) and meta.is_file(), str(meta))

    # 假装服务重启：内存清空，走启动时那一步恢复
    server.SESSIONS.clear()
    server.JOBS.clear()
    server.restore_sessions_from_disk()
    state = client.get("/api/state").get_json()
    ids = [entry["id"] for entry in state.get("items", [])]
    check(
        "重启后同一 sid 拿回同一 item_id",
        ids == [item_id],
        f"{ids} vs {item_id}",
    )

    # 真换一个进程：重载模块 → 新 APP_TOKEN、新内存，但读同一把密钥，旧 cookie 仍应有效
    import importlib

    reloaded = importlib.reload(server)
    reloaded.restore_sessions_from_disk()
    reloader_client = reloaded.app.test_client()
    reloader_client.set_cookie("session", client.get_cookie("session").value)
    restored = reloader_client.get("/api/state").get_json()
    re_ids = [entry["id"] for entry in restored.get("items", [])]
    check(
        "换新进程后旧 cookie 仍能认出来",
        re_ids == [item_id] and restored.get("items"),
        f"{re_ids}",
    )
    thumb = reloader_client.get(f"/api/items/{item_id}/thumb")
    check("恢复出来的素材缩略图还能取", thumb.status_code == 200, str(thumb.status_code))

    check("main() 真的挂了启动恢复", "restore_sessions_from_disk()" in getsource(reloaded.main))

    # 用户删掉的素材不能被复活
    re_token = reloaded.APP_TOKEN
    drop = reloader_client.delete(f"/api/items/{item_id}", headers={"X-App-Token": re_token})
    check("删除素材返回成功", drop.status_code == 200, str(drop.get_json()))
    reloaded.SESSIONS.clear()
    reloaded.restore_sessions_from_disk()
    check("删掉的素材不会复活", reloaded.SESSIONS.get(sid, {}).get("items", {}) == {})

    # 72 小时回收照常（整棵根目录指到临时目录再测，绝不动用户真实素材与历史 ZIP）
    roots = {name: getattr(reloaded, name) for name in ("UPLOAD_ROOT", "ZIP_DIR", "OUTPUT_ROOT")}
    sandbox = Path(tempfile.mkdtemp(prefix="jingyin_ws_age_"))
    try:
        reloaded.UPLOAD_ROOT = sandbox / "uploads"
        reloaded.ZIP_DIR = sandbox / "zips"
        reloaded.OUTPUT_ROOT = sandbox / "成品"
        old_sid = reloaded.UPLOAD_ROOT / uuid.uuid4().hex
        old_sid.mkdir(parents=True)
        past = time.time() - 73 * 3600
        os.utime(old_sid, (past, past))
        fresh = reloaded.UPLOAD_ROOT / uuid.uuid4().hex
        fresh.mkdir()
        reloaded.cleanup_old_files()
        check("72 小时以上的素材目录仍被回收", not old_sid.exists())
        check("当天的素材目录不动", fresh.exists())
    finally:
        for name, value in roots.items():
            setattr(reloaded, name, value)
        shutil.rmtree(sandbox, ignore_errors=True)

    # 测试素材留下的空会话目录清干净，不污染 _check.py 语料
    for leftover in (sid,):
        folder = server.UPLOAD_ROOT / leftover
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def upload_capacity_checks() -> None:
    sandbox = Path(tempfile.mkdtemp(prefix="jingyin_upload_"))
    original_root = server.UPLOAD_ROOT
    original_limit = server.MAX_BATCH_ITEMS
    sids = []

    def new_client():
        client = server.app.test_client()
        sid = uuid.uuid4().hex
        with client.session_transaction() as sess:
            sess["sid"] = sid
        sids.append(sid)
        return client, sid

    def send(client, count=1, payload=None, name="容量测试.png"):
        data = make_source() if payload is None else payload
        return client.post(
            "/api/upload",
            data={"files": [(io.BytesIO(data), name) for _ in range(count)]},
            content_type="multipart/form-data",
            headers={"X-App-Token": server.APP_TOKEN},
        )

    def stored(sid):
        with server.INDEX_LOCK:
            store = server.SESSIONS.get(sid, {})
            return len(store.get("items", {})), store.get("pending_uploads", 0)

    def clear(client):
        response = client.post("/api/clear", headers={"X-App-Token": server.APP_TOKEN})
        check("清空素材释放容量", response.status_code == 200)

    def concurrent_upload(client, sid, fail=False):
        entered = threading.Event()
        release = threading.Event()
        outcomes = {}
        exceptions = []
        original_save = FileStorage.save

        def held_save(uploaded_file, destination, *args, **kwargs):
            if uploaded_file.filename == "并发占位.png":
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("测试未释放上传")
                if fail:
                    raise OSError("测试保存失败")
            return original_save(uploaded_file, destination, *args, **kwargs)

        def worker(key, filename):
            try:
                worker_client = server.app.test_client()
                with worker_client.session_transaction() as sess:
                    sess["sid"] = sid
                outcomes[key] = send(worker_client, name=filename)
            except Exception as exc:
                exceptions.append(repr(exc))

        first = threading.Thread(target=worker, args=("first", "并发占位.png"))
        second = threading.Thread(target=worker, args=("second", "并发竞争.png"))
        with patch.object(FileStorage, "save", held_save):
            first.start()
            try:
                ready = entered.wait(timeout=5)
                check("上传处理中能进入受控测试点", ready)
                second.start()
                second.join(timeout=3)
                check("文件写入不占用全局索引锁", not second.is_alive())
                response = outcomes.get("second")
                check("同会话并发上传计入预留容量", response is not None and response.status_code == 400)
            finally:
                release.set()
                first.join()
                if second.ident is not None:
                    second.join()
            check("并发测试线程正常结束", not exceptions, str(exceptions))
        check("并发上传没有突破容量", stored(sid)[0] <= server.MAX_BATCH_ITEMS)
        response = outcomes.get("first")
        expected = 400 if fail else 200
        check("处理中上传的终态正确", response is not None and response.status_code == expected)
        check("并发结束后预留名额归零", stored(sid)[1] == 0, str(stored(sid)))
        if fail:
            check("并发失败释放名额后可以再上传", send(client).status_code == 200)
        check("并发完成后素材数等于上限", stored(sid) == (server.MAX_BATCH_ITEMS, 0), str(stored(sid)))

    try:
        server.UPLOAD_ROOT = sandbox / "uploads"
        client, sid = new_client()
        response = send(client, count=original_limit + 1)
        body = response.get_json() or {}
        check("真实上限的整批上传只接收允许的份数", response.status_code == 200 and len(body.get("items", [])) == original_limit, str(stored(sid)))
        check("超出上限的文件逐份报告且保留部分成功", len(body.get("errors", [])) == 1 and str(original_limit) in body["errors"][0])
        check("整批上传后没有遗留预留名额", stored(sid) == (original_limit, 0), str(stored(sid)))
        folders = set((server.UPLOAD_ROOT / sid).iterdir())
        response = send(client)
        check("累计满额后下一次上传被拒绝", response.status_code == 400 and not response.get_json().get("items"))
        check("满额拒绝不会新建素材目录", set((server.UPLOAD_ROOT / sid).iterdir()) == folders)
        first_id = next(iter(server.SESSIONS[sid]["items"]))
        removed = client.delete(f"/api/items/{first_id}", headers={"X-App-Token": server.APP_TOKEN})
        check("删除一份后能补一份且仍不超限", removed.status_code == 200 and send(client).status_code == 200 and stored(sid) == (original_limit, 0))
        other, other_sid = new_client()
        check("不同会话的上传容量互不占用", send(other).status_code == 200 and stored(other_sid) == (1, 0))
        clear(client)
        clear(other)

        server.MAX_BATCH_ITEMS = 2
        check("第一轮上传成功", send(client).status_code == 200)
        response = send(client, count=2)
        body = response.get_json() or {}
        check("多轮累计容量与单次容量一致", response.status_code == 200 and len(body.get("items", [])) == 1 and len(body.get("errors", [])) == 1 and stored(sid) == (2, 0))
        clear(client)

        for payload, name in ((b"", "空文件.png"), (b"not png", "损坏.png"), (b"text", "不支持.txt")):
            response = send(client, payload=payload, name=name)
            check(f"{name}不占用容量且不留目录", response.status_code == 400 and stored(sid) == (0, 0) and not list((server.UPLOAD_ROOT / sid).glob("*/")))

        failures = (
            (FileStorage, "save", OSError("测试写入失败")),
            (server.media, "probe_image", server.media.MediaError("测试读取失败")),
            (server.media, "make_image_thumb", OSError("测试缩略图失败")),
            (server.media, "make_preview", OSError("测试预览失败")),
        )
        for target, attribute, error in failures:
            with patch.object(target, attribute, side_effect=error):
                response = send(client)
            check(f"{attribute}失败会释放预留容量", response.status_code == 400 and stored(sid) == (0, 0) and not list((server.UPLOAD_ROOT / sid).glob("*/")))

        blocked = sandbox / "blocked"
        blocked.write_bytes(b"not a directory")
        server.UPLOAD_ROOT = blocked
        try:
            response = send(client)
            check("创建目录失败也会释放预留容量", response.status_code == 400 and stored(sid) == (0, 0))
        finally:
            server.UPLOAD_ROOT = sandbox / "uploads"

        with patch.object(server.state_store, "write_item_meta", side_effect=OSError("测试索引写入失败")):
            response = send(client)
        check("索引落盘失败仍保留已上传素材且不泄漏名额", response.status_code == 200 and stored(sid) == (1, 0))
        clear(client)
        response = send(client, payload=b"", count=2)
        check("失败文件不会耗尽后续请求的容量", response.status_code == 400 and send(client, count=2).status_code == 200 and stored(sid) == (2, 0))
        clear(client)

        send(client)
        concurrent_upload(client, sid)
        clear(client)
        send(client)
        concurrent_upload(client, sid, fail=True)
    finally:
        server.UPLOAD_ROOT = original_root
        server.MAX_BATCH_ITEMS = original_limit
        with server.INDEX_LOCK:
            for sid in sids:
                server.SESSIONS.pop(sid, None)
        shutil.rmtree(sandbox, ignore_errors=True)


def frontend_checks() -> None:
    html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    check("界面刷新后会向服务要回状态", "await api('/api/state')" in app_js)
    check("恢复出来的框会照旧画回画布", "loadSavedRegions" in app_js and "已恢复上次框选的" in app_js)
    check("使用说明写明素材保留多久", "72 小时" in html)


def main() -> int:
    unit_checks()
    http_checks()
    upload_capacity_checks()
    frontend_checks()
    print("\n结果：" + ("全部通过" if not FAILURES else f"{len(FAILURES)} 项不通过：{FAILURES}"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
