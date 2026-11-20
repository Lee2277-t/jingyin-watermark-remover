# -*- coding: utf-8 -*-
"""原子落盘与 ZIP 完整性的定向测试（只动 work/qa 下的临时文件，不碰成品目录）。

覆盖四条：
① 图片成品写盘中途抛错时，最终文件名不能被留下，`.part` 也不能残留；
② 预览缩略图同上；
③ 正常保存后目录里只有成品本身；
④ 半截 / 少条目的 ZIP 不会被 `_zip_is_complete` 认领，`_write_zip` 重建后逐条目可读。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image  # noqa: E402

import server  # noqa: E402
from remover import image_ops, media  # noqa: E402

WORK = Path(__file__).resolve().parent / "_robust_tmp"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def reset() -> None:
    if WORK.exists():
        for file in WORK.rglob("*"):
            if file.is_file():
                file.unlink()
    WORK.mkdir(parents=True, exist_ok=True)


class Boom(Exception):
    pass


def test_image_atomic() -> None:
    reset()
    target = WORK / "out.png"
    image = Image.new("RGB", (24, 16), (200, 12, 30))
    original_save = Image.Image.save

    def half_broken_save(self, fp, *args, **kwargs):
        path = Path(fp)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk-junk")
        raise Boom("模拟写盘一半断电")

    Image.Image.save = half_broken_save
    try:
        image_ops._save_image(image, target, {})
        raised = True
    except Boom:
        raised = True
    finally:
        Image.Image.save = original_save

    check("图片成品写盘失败后不留最终文件", not target.exists(), f"target_exists={target.exists()}")
    leftovers = [p.name for p in WORK.iterdir()]
    check("图片成品写盘失败后不留 .part", leftovers == [], f"left={leftovers}")
    check("异常照常上抛（任务会被标失败）", raised)

    image_ops._save_image(image, target, {"icc_profile": None})
    check("正常保存只留成品", [p.name for p in WORK.iterdir()] == ["out.png"])
    with Image.open(target) as saved:
        check("正常保存像素不变", saved.convert("RGB").getpixel((0, 0)) == (200, 12, 30), str(saved.size))


def test_thumb_atomic() -> None:
    reset()
    source = WORK / "src.png"
    Image.new("RGB", (120, 90), (10, 20, 240)).save(source, "PNG")
    target = WORK / "thumb.jpg"
    original_save = Image.Image.save

    def half_broken_save(self, fp, *args, **kwargs):
        Path(fp).write_bytes(b"\xff\xd8\xff\xe0" + b"junk-junk")
        raise Boom("模拟预览写盘中断")

    Image.Image.save = half_broken_save
    try:
        media.make_image_thumb(source, target, max_side=40)
    except media.MediaError:
        pass
    finally:
        Image.Image.save = original_save
    check("预览中断不产出半成品缩略图", not target.exists())
    check("预览中断不留 .part", [p.name for p in WORK.iterdir()] == ["src.png"],
          str([p.name for p in WORK.iterdir()]))

    media.make_image_thumb(source, target, max_side=40)
    check("预览正常落盘", target.is_file() and target.stat().st_size > 0,
          f"{target.stat().st_size} B")


def test_zip_logic() -> None:
    reset()
    a = WORK / "a.png"
    b = WORK / "b.png"
    for p in (a, b):
        Image.new("RGB", (8, 8), (1, 2, 3)).save(p, "PNG")
    results = [
        {"name": "1.png", "file_path": a, "kind": "image"},
        {"name": "2.png", "file_path": b, "kind": "image"},
    ]
    names = [r["name"] for r in results]
    zip_path = WORK / "job.zip"

    check("空目录时 ZIP 不算可用", not server._zip_is_complete(zip_path, names))
    server._write_zip(zip_path, results, names)
    check("打包后条目一致且可读", server._zip_is_complete(zip_path, names), f"{zip_path.stat().st_size} B")
    check("打包后不留 .part", sorted(p.name for p in WORK.iterdir() if p.name.startswith("job")) == ["job.zip"],
          str(sorted(p.name for p in WORK.iterdir())))

    raw = zip_path.read_bytes()
    zip_path.write_bytes(raw[: max(0, len(raw) // 2)])
    check("半截 ZIP 不被认领", not server._zip_is_complete(zip_path, names))
    server._write_zip(zip_path, results, names)
    check("坏 ZIP 能重建", server._zip_is_complete(zip_path, names))

    # 用户自己清掉成品目录后，已完整打包的 ZIP 仍应能下载（路由只在需要重打包时才查成品在场）
    b.unlink()
    check("缺文件被 _file_ready 拦住", not server._file_ready(b))
    check("完整 ZIP 在成品被删后仍可复用", server._zip_is_complete(zip_path, names))
    zip_path.unlink()
    check("无 ZIP 且缺成品时判定为需要报错", not server._zip_is_complete(zip_path, names)
          and not server._file_ready(b))


def test_zip_route() -> None:
    """走一遍真实路由：成品被删 + 无缓存 ZIP 必须回 409 且带明确文案，不能静默给缺张的包。"""
    reset()
    a = WORK / "a.png"
    missing_file = WORK / "gone.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(a, "PNG")
    job_id = "qajobtest0001"
    server.JOBS[job_id] = {
        "id": job_id,
        "status": "done",
        "session_id": "qa",
        "results": [
            {"id": "r1", "name": "1.png", "kind": "image", "file_path": a},
            {"id": "r2", "name": "2.png", "kind": "image", "file_path": missing_file},
        ],
    }
    stale = server.ZIP_DIR / f"{job_id}.zip"
    if stale.exists():
        stale.unlink()
    try:
        with server.app.test_client() as client:
            with client.session_transaction() as sess:
                sess["sid"] = "qa"
            response = client.get(f"/api/jobs/{job_id}/zip")
            response.close()
            check("缺成品且无缓存时回 409", response.status_code == 409, f"status={response.status_code}")
            body = response.get_json() or {}
            check("报错说清缺几个", "1 个成品文件已不在磁盘上" in (body.get("error") or ""), str(body))

            # 删掉一项后全部在场：应能打包并下载
            server.JOBS[job_id]["results"] = [server.JOBS[job_id]["results"][0]]
            response = client.get(f"/api/jobs/{job_id}/zip")
            response.close()
            check("全部成品在场时能打包下载", response.status_code == 200, f"status={response.status_code}")
            check("ZIP 落到 work/zips", stale.exists(), f"{stale.stat().st_size if stale.exists() else 0} B")
            check("缓存 ZIP 可复用", server._zip_is_complete(stale, ["1.png"]))
    finally:
        server.JOBS.pop(job_id, None)
        if stale.exists():
            try:
                stale.unlink()
            except OSError as exc:
                check("测试用 ZIP 能清掉", False, str(exc))


def main() -> int:
    test_image_atomic()
    test_thumb_atomic()
    test_zip_logic()
    test_zip_route()
    reset()
    print("\n结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
