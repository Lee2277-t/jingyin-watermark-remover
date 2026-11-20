# -*- coding: utf-8 -*-
"""用本机 Electron 起真窗口跑界面验收（内置浏览器没有可视 surface 时的替代）。

跑法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_electron_qa.py
它会先合成一张无意义测试图（不读用户素材），再交给 _electron_qa.cjs 上传并真拖拽画框。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
# Electron 是本机借来的运行时，不随包分发。换机器时用 JINGYIN_ELECTRON 指过去，
# 指不到就按下面 main() 那条分支跳过界面验收（其余回归不受影响）。
ELECTRON = os.environ.get("JINGYIN_ELECTRON")
QA = BASE / "work" / "qa"
SCRIPT = QA / "_electron_qa.cjs"
FIXTURE = QA / "_ui_fixture.png"
FIXTURE2 = QA / "_ui_lowcontrast.png"
# 「停止这一批」需要一个真的跑得久的任务才抓得住窗口，用公开正片语料（不是用户作品）
VIDEO_FIXTURE = QA / "_real_oceans.mp4"
OUT = QA / "_electron_qa.json"
USER_DATA = QA / "_electron_userdata"


def build_fixture() -> None:
    rng = np.random.default_rng(7)
    height, width = 900, 1200
    yy, xx = np.mgrid[0:height, 0:width]
    red = 120 + 60 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
    green = 100 + 50 * np.sin((xx + yy) / 120.0)
    blue = 140 + 40 * np.cos(xx / 60.0)
    noise = rng.normal(0, 6, size=(height, width, 1))
    rgb = np.dstack([red, green, blue]) + noise
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # 盖一块半透明蓝斜条，让界面里有可框的东西（纯合成，不含任何用户内容）
    image = Image.fromarray(rgb)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pixels = np.array(overlay)
    band = (np.abs((xx - yy) % 260) < 46) & (yy > height * 0.2) & (yy < height * 0.8)
    pixels[band] = (60, 106, 188, 150)
    overlay = Image.fromarray(pixels)
    image = image.convert("RGBA")
    image.alpha_composite(overlay)
    image.convert("RGB").save(FIXTURE)


def build_lowcontrast_fixture() -> None:
    # 低对比平铺水印的合成文档页（与 _test_lowcontrast.py 同一夹具，纯自绘）
    sys.path.insert(0, str(QA))
    import cv2

    from _test_lowcontrast import tiled_page

    page = tiled_page()
    # cv2.imwrite 在中文路径上会静默失败，落盘统一走 PIL
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_RGB2BGR)).save(FIXTURE2)


def main() -> int:
    if not ELECTRON or not Path(ELECTRON).is_file():
        print("请用环境变量 JINGYIN_ELECTRON 指定本机 electron.exe")
        return 1
    # Qoder 的 shell 会注入 ELECTRON_RUN_AS_NODE=1，那样 electron.exe 只当 Node 用，
    # require('electron') 会拿到 npm 桩（一个路径字符串）而不是真 API。必须清掉。
    env = dict(os.environ)
    env.pop("ELECTRON_RUN_AS_NODE", None)
    port = json.loads((BASE / "work" / "runtime.json").read_text(encoding="utf-8"))["port"]
    build_fixture()
    build_lowcontrast_fixture()
    command = [
        ELECTRON,
        str(SCRIPT),
        f"--user-data-dir={USER_DATA}",
        "--no-sandbox",
        f"--out={OUT}",
        f"--upload={FIXTURE}",
        f"--upload2={FIXTURE2}",
        f"--uploadVideo={VIDEO_FIXTURE}" if VIDEO_FIXTURE.exists() else "--uploadVideo=",
        f"--url=http://127.0.0.1:{port}/",
    ]
    if not VIDEO_FIXTURE.exists():
        print(f"提示：没有 {VIDEO_FIXTURE.name}，「停止这一批」那一段会 SKIP")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, env=env)
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", (result.stderr or "")[-2000:])
    for name in ("_ui_switch_on.png", "_ui_region_limit.png", "_ui_suggestion.png", "_ui_mobile.png"):
        path = QA / name
        if path.exists():
            print(f"截图：{path} {path.stat().st_size:,} B")
    if FIXTURE.exists():
        FIXTURE.unlink()
    if FIXTURE2.exists():
        FIXTURE2.unlink()
    # 测试素材不能留在语料目录里，否则 _check.py 会多出一份合成图
    if OUT.exists():
        data = json.loads(OUT.read_text(encoding="utf-8"))
        for key in ("uploadedItemId", "secondItemId", "videoItemId"):
            item_id = data.get(key)
            if not item_id:
                continue
            for item_dir in (BASE / "work" / "uploads").glob(f"*/{item_id}"):
                shutil.rmtree(item_dir, ignore_errors=True)
            print(f"已移除测试素材 item={item_id}")
        for sid_dir in (BASE / "work" / "uploads").iterdir():
            if sid_dir.is_dir() and not any(sid_dir.iterdir()):
                sid_dir.rmdir()
        # 界面自检会真跑三次任务（video_mode 闭环 + motion 闭环 + 停止这一批），三个测试成品都不留在交付目录
        for dir_key, job_key in (
            ("testOutputDir", "testJobId"),
            ("motionOutputDir", "motionJobId"),
            ("cancelOutputDir", "cancelJobId"),
        ):
            output_dir = data.get(dir_key)
            if not output_dir:
                continue
            path = Path(output_dir)
            if path.parent != BASE / "成品":
                continue
            shutil.rmtree(path, ignore_errors=True)
            zip_path = BASE / "work" / "zips" / f"{data.get(job_key)}.zip"
            zip_path.unlink(missing_ok=True)
            print(f"已移除测试成品 {path.name}")
    shutil.rmtree(USER_DATA, ignore_errors=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
