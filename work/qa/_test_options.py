# -*- coding: utf-8 -*-
"""三组修复选项的跨层契约回归（合成素材，不读用户作品）。

`method` / `video_mode` / `motion` 从前在四层里各写各的（server 归一化、BatchItem 默认值、
两个引擎比字面量、前端还有一份默认值 + 三个 localStorage 键），第二十七轮把词汇表收进
`remover/options.py`。**收口本身不改行为**，所以这份回归量的不是像素，而是「收口没收漏」：

① 词汇表与归一化的语义与旧写法逐值相同，**两个默认档冻结**（`smart` / `quality`——后者是
   用户明确要求的「不要改」）；
② 每一层此刻的默认值仍然等于 `options` 里的默认档（用签名/字段元数据问，不靠注释）；
③ 产品代码里再没有第二处把这五个取值写成裸字符串字面量（`tokenize` 只看 STRING token，
   注释、docstring、f-string 都不会误报），并配一条**哨兵**：一个含裸 `"photo"` 的临时模块
   必须被同一个扫描器报出来——不然「0 处」可能只是扫描器坏了；
④ 唯一的真规则「照片重绘在视频上降级成内容填充」由 `method_for_video()` 说一次，
   并拿一段真视频端到端证明成品标记是 `frame-inpaint` 而不是 `frame-photo`；
⑤ HTTP 层：非法取值归到默认档后**回显给界面**，且归一化排在「照片重绘能不能用」判定之前
   （大写 `"PHOTO"` 不该触发 400，因为它已经不是 photo）。

夹具是一段 1 秒 640×360 纯色视频与一张 320×200 的页面图，全部脚本内合成，跑完自清。
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from remover import batch_runner, image_ops, options, photo_inpaint, video_ops  # noqa: E402
import server  # noqa: E402

SCRATCH = BASE / "work" / "qa" / "_op_scratch"
FAILED: list[str] = []

VOCAB = set(options.METHODS) | set(options.VIDEO_MODES)
# 前端与模板**本来就该**写字面量（那是界面自己的文案与卡片标记），扫描范围只到产品 Python 为止。
SCAN_FILES = [BASE / "server.py"] + [
    path
    for path in sorted((BASE / "remover").glob("*.py"))
    if path.name != "options.py"
]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def string_literals(text: str) -> list[tuple[int, object]]:
    """逐行取「常量字符串」的值。f-string 是拼接结果不是常量，跳过；解码失败的 token 同样跳过。

    `generate_tokens` 是生成器，遇到半截文件会在迭代中途抛 `TokenError`——那时候已经取到的
    行仍然是有效证据，所以留住已收集的部分而不是让整份回归崩掉。
    """
    out: list[tuple[int, object]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.STRING:
                continue
            raw = token.string
            quote = raw[0] if raw[0] in "\"'" else raw[-1]
            if "f" in raw[: raw.index(quote)].lower():
                continue
            try:
                out.append((token.start[0], ast.literal_eval(raw)))
            except (ValueError, SyntaxError):
                pass
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def vocabulary_hits(text: str) -> list[tuple[int, str]]:
    return [(line, value) for line, value in string_literals(text) if value in VOCAB]


def options_bindings(text: str) -> list[tuple[int, str]]:
    """这个文件里把名字 `options` 绑定了的每一个位置（行号 + 怎么绑的）。

    `import` 本身也算一次绑定——那正是口径模块该来的地方，所以判据是「恰好一次」而不是「零次」。
    属性访问（`options.SMART`）不绑定名字，不在其列。
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(text)):
        target_names: list[str] = []
        if isinstance(node, ast.Assign):
            target_names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            target_names.append(node.target.id)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            target_names.append(node.target.id)
        elif isinstance(node, ast.Import):
            target_names.extend((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target_names.extend(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                args.append(node.args.vararg)
            if node.args.kwarg:
                args.append(node.args.kwarg)
            target_names.extend(arg.arg for arg in args)
            if node.name == "options":
                hits.append((node.lineno, "def"))
        elif isinstance(node, ast.ExceptHandler) and node.name == "options":
            target_names.append("options")
        hits.extend((getattr(node, "lineno", 0), "bind") for name in target_names if name == "options")
    return hits


def imports_vocabulary(text: str) -> bool:
    """这个文件有没有 `from remover import options` / `from . import options` 这一行。

    只有 import 进来的那个名字才代表词汇表；没 import 的文件里出现局部 `options`
    （`photo_inpaint.py` 的 `ort.SessionOptions()`）是另一回事，不该被这道闸误伤。
    """
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name).split(".")[0] == "options":
                    return True
    return False


# ---------------------------------------------------------------- ① 词汇表与归一化
def vocabulary_checks() -> None:
    check(
        "METHODS 恰好是智能/填充/照片三项且顺序稳定",
        options.METHODS == ("smart", "inpaint", "photo"),
        str(options.METHODS),
    )
    check(
        "VIDEO_MODES 恰好是画质/速度两项",
        options.VIDEO_MODES == ("quality", "fast"),
        str(options.VIDEO_MODES),
    )
    check("修复方式默认档冻结为智能纹理修复", options.DEFAULT_METHOD == "smart", options.DEFAULT_METHOD)
    check(
        "视频档位默认档冻结为画质优先（用户要求「不要改」）",
        options.DEFAULT_VIDEO_MODE == "quality",
        options.DEFAULT_VIDEO_MODE,
    )
    check(
        "normalize_method 认得的三个值原样返回且幂等",
        all(options.normalize_method(options.normalize_method(v)) == v for v in options.METHODS),
    )
    junk_method = [None, "", " ", "Smart", "PHOTO", "photo ", "quality", "fast", 0, 1, 3, [], {}, True]
    check(
        "normalize_method 对非法取值一律归默认档",
        all(options.normalize_method(v) == options.DEFAULT_METHOD for v in junk_method),
        f"非法取值 {len(junk_method)} 个",
    )
    check(
        "normalize_video_mode 认得的两个值原样返回且幂等",
        all(options.normalize_video_mode(options.normalize_video_mode(v)) == v for v in options.VIDEO_MODES),
    )
    junk_mode = [None, "", "Quality", "FAST", "smart", "photo", 0, 1, [], {}, True]
    check(
        "normalize_video_mode 对非法取值一律归画质优先",
        all(options.normalize_video_mode(v) == options.DEFAULT_VIDEO_MODE for v in junk_mode),
        f"非法取值 {len(junk_mode)} 个",
    )
    motion_values = [None, "", "0", "1", 0, 1, [], [0], {}, True, False]
    check(
        "normalize_motion 与旧写法 bool(value) 逐值相同",
        all(options.normalize_motion(v) is bool(v) for v in motion_values),
        f"比了 {len(motion_values)} 个值",
    )
    check(
        "三组词汇互不认领：档位名不是修复方式，修复方式名也不是档位",
        options.normalize_method(options.QUALITY) == options.DEFAULT_METHOD
        and options.normalize_video_mode(options.PHOTO) == options.DEFAULT_VIDEO_MODE,
    )


# ---------------------------------------------------------------- ② 跨层默认值
def layer_checks() -> None:
    inplace_default = inspect.signature(image_ops.remove_boxes_inplace).parameters["method"].default
    image_default = inspect.signature(image_ops.process_image_file).parameters["method"].default
    video_params = inspect.signature(video_ops.process_video_file).parameters
    check(
        "图片两处入口的 method 形参默认值 == 默认档",
        inplace_default == options.DEFAULT_METHOD and image_default == options.DEFAULT_METHOD,
        f"{inplace_default} / {image_default}",
    )
    check(
        "视频入口的三组默认值 == 默认档",
        video_params["method"].default == options.DEFAULT_METHOD
        and video_params["video_mode"].default == options.DEFAULT_VIDEO_MODE
        and video_params["motion"].default is False,
        f"{video_params['method'].default} / {video_params['video_mode'].default} / {video_params['motion'].default}",
    )
    fields = {field.name: field.default for field in dataclasses.fields(batch_runner.BatchItem)}
    check(
        "BatchItem 的三组默认值 == 默认档",
        fields["method"] == options.DEFAULT_METHOD
        and fields["video_mode"] == options.DEFAULT_VIDEO_MODE
        and fields["motion"] is False,
        f"{fields['method']} / {fields['video_mode']} / {fields['motion']}",
    )
    check(
        "空请求体的归一化结果 == 三组默认档",
        server.parse_job_options({}) == (options.DEFAULT_METHOD, options.DEFAULT_VIDEO_MODE, False),
        str(server.parse_job_options({})),
    )

    html = (BASE / "templates" / "index.html").read_text(encoding="utf-8")
    hdr_notes = re.findall(r'<p[^>]*id="videoHdrNote"[^>]*>(.*?)</p>', html, re.S)
    check(
        "视频设置明确提示 HDR 限制，并说明检测到会写进成品说明",
        len(hdr_notes) == 1
        and all(text in hdr_notes[0] for text in (
            "先转成 SDR", "不做色调映射", "8bit", "可能异常", "识别到 HDR",
        ))
        # 不能再写「不代表已检测」：检测已经接上了，那句话现在是不实说明。
        and "不代表已检测到 HDR" not in hdr_notes[0],
    )
    hdr_help = re.findall(r'<strong>HDR 视频</strong><p>(.*?)</p>', html, re.S)
    check(
        "使用说明包含 HDR 处理限制、检测口径与转换建议",
        len(hdr_help) == 1
        and all(text in hdr_help[0] for text in (
            "不做 HDR→SDR 色调映射", "8bit", "可能异常", "转成 SDR",
            "smpte2084", "成品说明",
        ))
        and "不代表已检测到 HDR" not in hdr_help[0],
    )
    cards = re.findall(r'data-method="([^"]+)"', html)
    check(
        "界面每个修复方式各一张卡片，不多不少",
        sorted(cards) == sorted(options.METHODS),
        str(cards),
    )
    selected = [line for line in html.splitlines() if "is-selected" in line and "data-method=" in line]
    check(
        "界面默认选中的卡片 == 默认修复方式",
        len(selected) == 1 and f'data-method="{options.DEFAULT_METHOD}"' in selected[0],
        selected[0].strip()[:80] if selected else "没有 is-selected 卡片",
    )

    js = (BASE / "static" / "app.js").read_text(encoding="utf-8")
    check(
        "前端 state 的三份默认值 == 默认档",
        f"method: '{options.DEFAULT_METHOD}'," in js
        and f"videoMode: '{options.DEFAULT_VIDEO_MODE}'," in js
        and "videoMotion: false," in js,
    )
    check(
        "前端三个 localStorage 键仍在（用户记住的选择不能因为收口而丢）",
        all(key in js for key in ("'jingyin.method'", "'jingyin.videoMode'", "'jingyin.videoMotion'")),
    )


# ---------------------------------------------------------------- ③ 源码扫描
def scan_checks() -> None:
    hits = [(path.relative_to(BASE).as_posix(), line, value) for path in SCAN_FILES for line, value in vocabulary_hits(path.read_text(encoding="utf-8"))]
    check(
        f"产品代码 {len(SCAN_FILES)} 个文件里五组取值再无第二处裸字面量",
        not hits,
        f"命中 {len(hits)} 处：{hits[:4]}" if hits else "0 处",
    )

    SCRATCH.mkdir(parents=True, exist_ok=True)
    canary = SCRATCH / "_canary_bare_literal.py"
    canary.write_text('VALUE = "photo"\nOTHER = "quality"\n', encoding="utf-8")
    found = vocabulary_hits(canary.read_text(encoding="utf-8"))
    check(
        "哨兵：含裸字面量的临时模块必须被同一个扫描器报出来（证明这道闸不是空转）",
        len(found) == 2 and {value for _, value in found} == {"photo", "quality"},
        str(found),
    )
    canary.write_text('VALUE = f"{\'photo\'}"\n', encoding="utf-8")
    check(
        "哨兵反面：f-string 里的同名取值不算命中（扫描器只在常量上判定）",
        not vocabulary_hits(canary.read_text(encoding="utf-8")),
    )

    shadow = []
    scanned = 0
    for path in SCAN_FILES:
        text = path.read_text(encoding="utf-8")
        if not imports_vocabulary(text):
            continue
        scanned += 1
        bindings = options_bindings(text)
        if len(bindings) != 1:
            shadow.append((path.name, bindings))
    check(
        "每个 import 了口径模块的产品文件都只绑定 `options` 那一次（没有被局部变量重新赋值遮蔽）",
        not shadow and scanned >= 4,
        f"查了 {scanned} 个文件" + (f"，遮蔽：{shadow}" if shadow else "，0 处遮蔽"),
    )
    shutil.rmtree(SCRATCH, ignore_errors=True)


# ---------------------------------------------------------------- ④ 视频路线规则
def build_clip() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    clip = SCRATCH / "_op_clip.mp4"
    if clip.exists() and clip.stat().st_size > 0:
        return clip
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x6a7b8c:s=640x360:r=24",
            "-t", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
    )
    return clip


EDGE_REGIONS = [{"x": 0.85, "y": 0.85, "w": 0.15, "h": 0.15}]
INNER_REGIONS = [{"x": 0.30, "y": 0.30, "w": 0.20, "h": 0.20}]


def video_route_checks() -> None:
    check(
        "method_for_video：照片重绘在视频上降级成内容填充",
        options.method_for_video(options.PHOTO) == options.INPAINT,
    )
    check(
        "method_for_video：智能与填充原样返回（它只管一条规则，不是归一化）",
        options.method_for_video(options.SMART) == options.SMART
        and options.method_for_video(options.INPAINT) == options.INPAINT,
    )

    clip = build_clip()
    out_photo = SCRATCH / "_op_photo.mp4"
    result = video_ops.process_video_file(
        clip, out_photo, EDGE_REGIONS, 640, 360, method=options.PHOTO
    )
    check(
        "视频选「照片重绘」：成品标记是 frame-inpaint，不是 frame-photo",
        result["method"] == "frame-inpaint" and "photo" not in result["method"],
        f'{result["method"]} · 说明「{result["note"][:28]}…」',
    )
    check(
        "同一次跑：档位回显是画质优先、说明如实写逐帧修复",
        result["video_mode"] == options.QUALITY and "逐帧修复" in result["note"],
        f'{result["video_mode"]} · {result["frames"]} 帧',
    )
    check("同一次跑：成品帧数与原片一致（24 帧没少）", result["frames"] == 24, str(result["frames"]))

    out_junk = SCRATCH / "_op_junk.mp4"
    junk_result = video_ops.process_video_file(
        clip, out_junk, INNER_REGIONS, 640, 360, method=options.SMART, video_mode="FAST"
    )
    check(
        "直传引擎的垃圾档位落到画质优先（与 normalize_video_mode 同值）",
        junk_result["video_mode"] == options.DEFAULT_VIDEO_MODE,
        f'{junk_result["video_mode"]} · 路线 {junk_result["method"]}',
    )


# ---------------------------------------------------------------- ⑤ HTTP 契约
def make_page() -> bytes:
    image = Image.new("RGB", (320, 200), (247, 247, 243))
    draw = ImageDraw.Draw(image)
    for row in range(6):
        draw.text((12, 12 + row * 24), "正文文字内容示例行", fill=(38, 38, 40))
    stamp = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
    ImageDraw.Draw(stamp).text((4, 6), "水印", fill=(30, 90, 200, 235))
    image.paste(stamp, (188, 150), stamp)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def wait_job(client, job_id: str, token: dict, limit: float = 40.0) -> dict:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}", headers=token).get_json() or {}
        if payload.get("status") in ("done", "failed", "cancelled"):
            return payload
        time.sleep(0.2)
    raise AssertionError("任务没有跑完")


def http_checks() -> None:
    original_output_root = server.OUTPUT_ROOT
    server.OUTPUT_ROOT = Path(tempfile.mkdtemp(prefix="jingyin_op_"))
    client = server.app.test_client()
    headers = {"X-App-Token": server.APP_TOKEN}
    real_available = photo_inpaint.available
    real_reason = photo_inpaint.unavailable_reason
    sid = ""
    try:
        client.get("/")  # 任意一次请求都会由 before_request 落一个 sid，这里只借它建 cookie
        with client.session_transaction() as sess:
            sid = sess.get("sid", "")
        upload = client.post(
            "/api/upload",
            data={"files": (io.BytesIO(make_page()), "契约页.png")},
            content_type="multipart/form-data",
            headers=headers,
        )
        check("测试素材上传成功", upload.status_code == 200, str(upload.status_code))
        if upload.status_code != 200:
            return
        item_id = upload.get_json()["items"][0]["id"]
        body = {"items": [{"id": item_id, "regions": EDGE_REGIONS}]}

        bogus = client.post("/api/process", json={**body, "method": "bogus"}, headers=headers).get_json() or {}
        done = wait_job(client, bogus["id"], headers)
        results = done.get("results") or []
        note = results[0]["note"] if results else ""
        check(
            "非法修复方式跑完全程且说明不是照片重绘（已归到默认档）",
            done.get("success_count") == 1 and "照片重绘" not in note,
            f'{done.get("status")} 成功 {done.get("success_count")} · 说明「{note[:26]}…」',
        )
        check(
            "档位写成大写 FAST 时界面回显是默认档（归一化排在回显之前）",
            client.post("/api/process", json={**body, "video_mode": "FAST"}, headers=headers).get_json()["video_mode"]
            == options.DEFAULT_VIDEO_MODE,
        )
        motion_job = client.post("/api/process", json={**body, "motion": True}, headers=headers).get_json() or {}
        check("motion=true 原样回显", motion_job.get("motion") is True)

        photo_inpaint.available = lambda: False
        photo_inpaint.unavailable_reason = lambda: "模型文件不在 models/ 里"
        try:
            rejected = client.post("/api/process", json={**body, "method": "photo"}, headers=headers)
            upper = client.post("/api/process", json={**body, "method": "PHOTO"}, headers=headers)
        finally:
            photo_inpaint.available = real_available
            photo_inpaint.unavailable_reason = real_reason
        check(
            "模型不可用时 method=photo 回 400 且原因原样给出",
            rejected.status_code == 400 and rejected.get_json()["error"] == "模型文件不在 models/ 里",
            f'{rejected.status_code} {rejected.get_json().get("error", "")}',
        )
        check(
            "归一化排在可用性判定之前：大写 PHOTO 已被归成默认档，不该吃这个 400",
            upper.status_code != 400 or "models" in upper.get_json().get("error", ""),
            f'{upper.status_code} {upper.get_json().get("error", "")}',
        )
    finally:
        photo_inpaint.available = real_available
        photo_inpaint.unavailable_reason = real_reason
        if sid:
            shutil.rmtree(server.UPLOAD_ROOT / sid, ignore_errors=True)
            server.SESSIONS.pop(sid, None)
        shutil.rmtree(server.OUTPUT_ROOT, ignore_errors=True)
        shutil.rmtree(SCRATCH, ignore_errors=True)
        server.OUTPUT_ROOT = original_output_root


def main() -> int:
    vocabulary_checks()
    layer_checks()
    scan_checks()
    video_route_checks()
    http_checks()
    print()
    print("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过：{FAILED}"))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
