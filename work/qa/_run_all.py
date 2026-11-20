# -*- coding: utf-8 -*-
"""统一回归入口：一条命令跑完 work/qa 的回归脚本，并如实标出哪些根本没跑。

跑法（都用项目内的 venv，裸 python 没有 cv2）：

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_all.py            # 核心档
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_all.py --list      # 只看清单与环境
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_all.py --only engine,corner
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_all.py --slow --ui --real
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe work/qa/_run_all.py --selftest  # 证明这套判据能变红

三条设计前提，都是本项目已经踩过的坑：

1. **跳过不等于验过**。缺服务、缺模型、缺真语料的脚本一律记 SKIP 并写明缺什么；
   一项都没跑出判定时进程退出码是 2，不是 0。这与 _check.py 空语料那句
   「本轮未验，不算通过也不算失败」同一口径。
2. **结论只从脚本自己打印的那一行判定行解析**（``结果：全部通过`` / ``结论=全部通过`` /
   ``共 N 项失败``），项数也从日志数，不按记忆写。脚本退出码非 0 时**绿色判定行不算数**——
   一个只会在末尾崩溃的脚本，如果因为它前面打过「全部通过」就变绿，这套 runner 比没有更糟。
3. **串行跑**。这些脚本共用同一个本机服务实例、视频只有一条通道，成品目录与上传会话
   也都在同一个 work 下；并行会互相抢任务队列，量到的失败不是产品缺陷而是干扰。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
QA = BASE / "work" / "qa"
MODEL = BASE / "models" / "lama_fp32.onnx"
ELECTRON = os.environ.get("JINGYIN_ELECTRON")

GREEN = "全部通过"
# 脚本自己的判定行：三种写法都出现过（结果：/ 结论= / 共 N 项失败），
# 另外 _test_lowcontrast.py 这类只在一行末尾打裸的「全部通过」/「FAILED n 项」。
VERDICT = re.compile(
    r"(结果[:：]|结论\s*[=＝]|项不通过|项失败|失败\s*\d+\s*项|全部通过|全部合格|FAILED\s*\d+\s*项)"
)
# 判定行只可能出现在结尾这几行；逐项的 PASS 行里也可能带「全部通过」这种字样，
# 所以候选行要跳过以 PASS/FAIL 开头的，别把一条通过项当成整份脚本的判定行。
VERDICT_TAIL = 6
# 判定行里的「没跑成」措辞：脚本自己说了未验，就不能算通过
UNVERIFIED = re.compile(r"(本轮未验|等于未验|未验)")

@dataclass(frozen=True)
class Entry:
    file: str
    args: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    cost: str = "core"  # core | slow | ui | real | metric
    scope: str = ""  # 这段到底在验什么（--list 与汇总表都打）
    timeout: float = 600.0
    report_json: str = ""  # 有的脚本另写结构化报告，项数优先按它数

    @property
    def name(self) -> str:
        return Path(self.file).stem.lstrip("_")


# 不在清单里的 work/qa/_*.py：如实说明为什么，免得读清单的人以为「漏了几个回归」。
EXCLUDED = {
    "_image_ops_before_redfix.py": "第三轮修复前的 image_ops 副本，是留档不是回归，跑了也没判据",
    "_cmp_lines.py": "两轮 _check.py 输出的比对工具，需要两个文件名参数，不自己产出结论",
    "_probe_launcher.ps1": "PowerShell 片段，不是 python 脚本",
    "_electron_qa.cjs": "界面验收的实际断言在 Electron 侧，由 _run_electron_qa.py 调它",
    "_hdr_probe.py": "第二十八轮调查阶段的 HDR 探针（PQ 曲线与映射对照），用途已被 _real_video.py 的 hdr 相位吸收，是推导现场记录不是回归",
    "_hdr_data_track.py": "第二十八轮调查阶段的 HDR 色度偏差追踪探针，同样已被 hdr 相位吸收，不是回归",
}

MANIFEST: tuple[Entry, ...] = (
    # ---- 图片引擎 ----
    Entry("_test_image_engine.py", scope="图片修复引擎主判据（自合成夹具 + 冻结基线）"),
    Entry("_test_image_engine.py", ("--canaries",), cost="slow", scope="§9.2 九条结论的哨兵相位"),
    Entry("_check.py", needs=("corpus",), cost="metric", scope="真实语料逐份指标转储（判据是与上一轮逐行比）"),
    Entry(
        "_test_lowcontrast.py",
        scope="低对比候选：合成夹具 + 确定性 + detect_watermark 契约",
    ),
    Entry("_verify_delivered.py", cost="metric", scope="成品与离线引擎逐像素核对（需要 成品/ 里有配对批次）"),
    Entry("_cmp_other_tools.py", needs=("corpus",), cost="metric", scope="与其他去水印路线的对照表测量"),
    # ---- 三组修复选项的跨层口径（server → BatchItem → 两个引擎 → 前端）----
    Entry(
        "_test_options.py",
        scope="method/video_mode/motion 唯一口径：默认档冻结 + 裸字面量扫描（带哨兵）+ 视频降级规则 + HTTP 回显",
    ),
    # ---- 批量与任务循环 ----
    Entry("_parallel_ab.py", scope="同一批 1 路 vs 8 路：逐文件 md5 必须一致"),
    Entry("_test_workspace.py", needs=("http",), scope="素材索引落盘 / 72h 回收 / 会话恢复"),
    Entry("_test_robustness.py", needs=("http",), scope=".part 原子落盘与 ZIP 完整性"),
    Entry("_test_instance.py", needs=("http",), scope="单实例判定与 runtime.json"),
    Entry("_test_region_limit.py", needs=("http",), scope="每份 16 框上限（后端 400 + 前端闸门）"),
    Entry("_test_photo_mode.py", needs=("http", "model"), cost="slow", scope="照片重绘 LaMa（含大框分块）"),
    # ---- 视频 ----
    Entry("_test_video_corner.py", scope="贴边路由与四条说明文案"),
    # --keep 是 §2.3 里写死的顺序契约：_e2e_video_mode.py 用的夹具就是这条留下的。
    # 不加 --keep 的话 e2e 必然「缺少夹具」退出码 1，那是入口自己的顺序错，不是产品回归红。
    Entry("_video_regression.py", ("--keep",), scope="960×540 三处触边夹具，两档逐区域残差"),
    Entry("_test_video_frame_io.py", scope="逐帧管道字节契约（半帧绝不发出去）"),
    Entry("_test_video_streams.py", scope="音轨/字幕搬运四类合成容器"),
    Entry("_test_video_detect.py", needs=("http",), scope="全片采样识别 + 原生角标补漏"),
    Entry("_test_video_motion.py", needs=("http",), scope="动态水印跟踪（含 HTTP 契约）"),
    Entry("_e2e_video_mode.py", needs=("http",), scope="视频档位开关前后端贯通"),
    Entry(
        "_test_job_cancel.py",
        needs=("http",),
        cost="slow",
        scope="停止这一批：三终态计数、清单 started、ZIP 仍可下载",
    ),
    Entry(
        "_test_job_cancel.py",
        ("--unit",),
        cost="core",
        scope="停止这一批：只跑引擎侧四个检查点（不需要服务）",
    ),
    Entry("_test_job_resume.py", needs=("http",), cost="slow", scope="断点续跑：同目录、已完成不重做、自愈"),
    Entry(
        "_test_video_timeout.py",
        ("--unit",),
        cost="core",
        scope="卡死闸门单元（节拍续命 / 无节拍开火）",
    ),
    Entry("_test_video_timeout.py", cost="slow", scope="卡死闸门真 ffmpeg 端到端 + delogo 管道级"),
    # ---- 真实影像与界面（都贵，按需 --real / --ui）----
    Entry("_real_video.py", ("mode",), needs=("real",), cost="real", scope="真片两档残差/ghost/sharp", timeout=1800),
    Entry("_real_video.py", ("mask",), needs=("real",), cost="real", scope="静止水印掩膜（含 1080p 与 4K）", timeout=1800),
    Entry("_real_video.py", ("detect",), needs=("real",), cost="real", scope="真片段采样找回数", timeout=1200),
    Entry("_real_video.py", ("motion",), needs=("real",), cost="real", scope="真片平移/旋转/缩放跟踪", timeout=1800),
    Entry("_real_video.py", ("streams",), needs=("real",), cost="real", scope="真容器音轨字幕逐样本核对", timeout=1800),
    Entry("_real_video.py", ("uhd",), needs=("real",), cost="real", scope="原生 4K 双角标识别与两档", timeout=2400),
    Entry("_real_video.py", ("subpic",), needs=("real",), cost="real", scope="图形字幕真容器端到端", timeout=1800),
    Entry("_real_video.py", ("hdr",), needs=("real",), cost="real", scope="真 4K HDR 源的色调映射与色彩标记", timeout=2400),
    Entry("_real_video.py", ("http",), needs=("real", "http"), cost="real", scope="真片走一遍 HTTP 全流程", timeout=1200),
    Entry("_rerun_batch.py", needs=("http",), cost="real", scope="14 页量级的端到端驱动（写一次成品批次并清掉）"),
    Entry("_profile_video_pipe.py", cost="metric", scope="逐帧路线耗时构成（报告，不是通过/失败）", timeout=1200),
    Entry(
        "_run_electron_qa.py",
        needs=("http", "electron"),
        cost="ui",
        scope="真窗口界面验收（真点击/真拖拽/真停止与续跑）",
        timeout=900,
        report_json="_electron_qa.json",
    ),
)

GROUPS = ("core", "slow", "real", "ui", "metric")


# --------------------------------------------------------------------------- 环境探测


def http_base() -> tuple[str | None, str]:
    """服务是否活着、是不是这个目录自己的实例。返回 (base_url 或 None, 一句话原因)。"""
    runtime = BASE / "work" / "runtime.json"
    if not runtime.is_file():
        return None, "work/runtime.json 不在，本机没有在跑的服务"
    try:
        info = json.loads(runtime.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 记录文件坏了本身就是一种「不能用」
        return None, f"runtime.json 读不出来：{exc}"
    port, pid = info.get("port"), info.get("pid")
    if not port:
        return None, "runtime.json 里没有 port"
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3.0) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 连接失败的原因五花八门，原样带出最有用
        return None, f"端口 {port} 上没有回应（{type(exc).__name__}: {exc}）；runtime.json 记的 pid={pid}"
    if health.get("app") != "jingyin":
        return None, f"端口 {port} 上跑的不是本工具（app={health.get('app')!r}），绝不借用"
    recorded = str(info.get("base_dir", "")).rstrip("\\/")
    if recorded and Path(recorded).resolve() != BASE.resolve():
        return None, f"端口 {port} 上的实例属于 {recorded}，不是 {BASE}；请用本目录的启动器重开服务"
    return f"http://127.0.0.1:{port}", (
        f"端口 {port} 活着（instance={str(health.get('instance',''))[:8]}…，"
        f"photo_ready={health.get('photo_ready')}，started_at={info.get('started_at')}）"
    )


def stale_service(started_at: str | None) -> list[str]:
    """盘上产品代码比在跑的服务新 —— HTTP 段验的仍是旧进程（§2.2 那个坑）。"""
    if not started_at:
        return []
    try:
        started = time.mktime(time.strptime(started_at, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return []
    out = []
    for path in [BASE / "server.py", *sorted((BASE / "remover").glob("*.py"))]:
        if path.is_file() and path.stat().st_mtime > started:
            out.append(path.name)
    for rel in ("templates/index.html", "static/app.js", "static/style.css"):
        path = BASE / rel
        if path.is_file() and path.stat().st_mtime > started:
            out.append(rel)
    return out


def ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 探测失败就让需要 ffmpeg 的脚本自己如实报错
        return None
    return exe if exe and Path(exe).is_file() else None


def corpus_pages() -> int:
    root = BASE / "work" / "uploads"
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.glob("*/*/source.png"))


def real_corpus_present() -> tuple[bool, str]:
    need = ["_real_oceans.mp4", "_real_sintel.mp4", "_real_bbb.mp4", "_real_jelly1080.mp4"]
    have = [n for n in need if (QA / n).is_file()]
    return len(have) >= 1, f"{len(have)}/{len(need)} 份公开正片在位（{'、'.join(have) or '一部都没有'}）"


def probe() -> dict[str, tuple[bool, str]]:
    base, why = http_base()
    pages = corpus_pages()
    ok_real, why_real = real_corpus_present()
    exe = ffmpeg_path()
    model_ok = MODEL.is_file()
    return {
        "http": (base is not None, why),
        "model": (model_ok, f"models/lama_fp32.onnx {'在位 ' + format(MODEL.stat().st_size, ',') + ' B' if model_ok else '缺失：照片重绘整块禁用'}"),
        "corpus": (pages > 0, f"work/uploads 里真实页面 {pages} 份（72h 回收会整批删；0 份时相关脚本只能算未验）"),
        "real": (ok_real, why_real),
        "electron": (
            bool(ELECTRON and Path(ELECTRON).is_file()),
            "需要通过 JINGYIN_ELECTRON 指定本机 electron.exe",
        ),
        "ffmpeg": (exe is not None, exe or "imageio-ffmpeg 没能给出可执行文件"),
    }


# --------------------------------------------------------------------------- 判定解析


def parse_items(*texts: str) -> tuple[int, int]:
    """按脚本自己打的逐项行计数（PASS / [OK] / [PASS]）；判定行里的「共 N 项」优先。"""
    passed = failed = 0
    for text in texts:
        for line in text.splitlines():
            s = line.strip()
            # 「FAILED 3 项: [...]」这种结尾汇总行本身也是 FAIL 开头，
            # 按逐项行数的话会把汇总自己算成一项，项数虚高。
            if re.match(r"^FAILED\s*\d+\s*项", s):
                continue
            if s.startswith(("PASS", "[OK]", "[PASS]")):
                passed += 1
            elif s.startswith(("FAIL", "[FAIL]")):
                failed += 1
    joined = "\n".join(texts)
    m = re.findall(r"共\s*(\d+)\s*项[，,].*?失败\s*(\d+)\s*项", joined)
    if m:
        passed, failed = int(m[-1][0]) - int(m[-1][1]), int(m[-1][1])
    return passed, failed


def verdict_line(*texts: str) -> str:
    """逐个来源找脚本自己打的那一行判定：先看 stdout，再看 stderr。

    不能把两路合并了再看结尾几行——_parallel_ab.py 会把「预期内坏件」的整段回溯写到
    stderr，判定行明明在 stdout 最后，却被几百行回溯挤出窗口，一次真通过会被报成未验。
    """
    for text in texts:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in reversed(lines[-VERDICT_TAIL:]):
            if line.startswith(("PASS", "FAIL", "[OK]", "[FAIL]", "[PASS]")):
                continue
            if VERDICT.search(line):
                return line
    return ""


@dataclass
class Result:
    entry: Entry
    status: str  # OK / FAIL / SKIP / METRIC / UNVERIFIED
    items: tuple[int, int]
    note: str
    seconds: float = 0.0
    log: Path | None = None

    @property
    def mark(self) -> str:
        return {"OK": "通过", "FAIL": "不通过", "SKIP": "跳过", "METRIC": "指标", "UNVERIFIED": "未验"}[self.status]


def classify(entry: Entry, out: str, code: int, err: str = "") -> tuple[str, tuple[int, int], str]:
    """把一份脚本输出收成状态。规则顺序就是重要性顺序。stdout 与 stderr 分开传：
    判定行只可能在脚本自己的结尾，而 stderr 的噪声（预期内坏件的回溯就是这个）不该把它挤走。"""
    merged = out + ("\n" + err if err else "")
    if entry.cost == "metric":
        n = len([l for l in merged.splitlines() if l.strip()])
        return "METRIC", (n, 0), f"指标转储 {n} 行，没有通过/失败判据"
    verdict = verdict_line(out, err)
    items = parse_items(out, err)
    if code != 0:
        tail = (err or merged).strip().splitlines()
        return "FAIL", items, f"退出码 {code}" + (f"：{tail[-1][:160]}" if tail else "") + (
            f"（判定行是 {verdict[:60]}，但进程没干净退出，绿色不算数）" if GREEN in verdict else ""
        )
    if not verdict:
        hint = next((l for l in merged.splitlines() if UNVERIFIED.search(l)), "")
        return "UNVERIFIED", items, hint.strip()[:200] or "脚本没打印判定行，无法据此宣布通过"
    if GREEN in verdict:
        if items[1] > 0:
            return "FAIL", items, f"判定行说「{verdict[:40]}」，但日志里有 {items[1]} 行 FAIL —— 自相矛盾，按不通过算"
        if UNVERIFIED.search(merged):
            hint = next((l for l in merged.splitlines() if UNVERIFIED.search(l)), "")
            return "UNVERIFIED", items, "脚本自己声明有内容未验，不计入通过：" + hint.strip()[:200]
        return "OK", items, verdict[:120]
    return "FAIL", items, verdict[:200]


def cleanup_kept(since: float) -> list[str]:
    """删掉本次为了 §2.3 那条顺序契约而用 --keep 留下的夹具，只认 mtime 晚于本次开跑的。

    用户自己手动跑 `_video_regression.py --keep` 留的那几份不能跟着删——那是他手上的东西，
    而且 next 一轮 e2e 还要靠它。时间戳闸门就是为这一条存在的。
    """
    removed = []
    for path in sorted(QA.glob("_vr_*.mp4")):
        try:
            if path.stat().st_mtime >= since:
                path.unlink()
                removed.append(path.name)
        except OSError:  # noqa: PERF203 删不掉就留在盘上，如实少报，不该把整轮结论带崩
            continue
    return removed


def run_entry(entry: Entry, env: dict[str, tuple[bool, str]], log_dir: Path) -> Result:
    missing = [k for k in entry.needs if not env.get(k, (False, ""))[0]]
    if missing:
        why = "；".join(f"{k}：{env[k][1]}" for k in missing)
        return Result(entry, "SKIP", (0, 0), f"缺前置条件 {('/'.join(missing))} —— {why}")
    path = QA / entry.file
    if not path.is_file():
        return Result(entry, "FAIL", (0, 0), f"{entry.file} 不在 work/qa 里")
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env.pop("ELECTRON_RUN_AS_NODE", None)  # 带着它 electron 只当 Node 用，界面验收会假成功
    start = time.monotonic()
    started_wall = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path), *entry.args],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=entry.timeout,
        )
        out, err, code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        out, err = (exc.stdout or ""), (exc.stderr or "")
        out = out if isinstance(out, str) else out.decode("utf-8", "replace")
        err = err if isinstance(err, str) else err.decode("utf-8", "replace")
        code = 124
        err += f"\n超时 {entry.timeout:.0f}s 被 runner 结束"
    seconds = time.monotonic() - start
    stem = entry.name + ("_" + "_".join(entry.args) if entry.args else "")
    log_name = re.sub(r"[^\w.\-]+", "_", stem.replace(" ", "")) + ".txt"
    log = log_dir / log_name
    log.write_text(f"$ {sys.executable} {entry.file} {' '.join(entry.args)}\n$ exit={code} 用时={seconds:.1f}s\n\n{out}\n--- stderr ---\n{err}\n", encoding="utf-8")
    status, items, note = classify(entry, out, code, err)
    # 结构化报告优先：界面验收的项数以 cjs 写出的 JSON 为准
    if entry.report_json:
        report = QA / entry.report_json
        # 只信本次运行期间写出来的报告。上一轮留下的 _electron_qa.json 会在这次
        # 脚本根本没跑成（比如窗口没起来）时被当成「107 项全过」——那是凭空造出来的绿色。
        fresh = report.is_file() and report.stat().st_mtime >= started_wall
        if fresh:
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                steps = data.get("steps") or []
                failed = data.get("failed") or []
                if steps:
                    items = (sum(1 for s in steps if s.get("ok")), len(failed) or sum(1 for s in steps if not s.get("ok")))
                    note = f"界面验收报告 {len(steps)} 步，失败 {len(failed)} 步" + (f"：{'、'.join(map(str, failed[:5]))}" if failed else "")
                    status = "OK" if not failed and items[1] == 0 else "FAIL"
            except Exception as exc:  # noqa: BLE001 报告读不出来就退回日志判定，别把跳过编成失败
                note += f"；{entry.report_json} 读不出来：{exc}"
        elif report.is_file():
            note += f"；{entry.report_json} 是上一轮留下的（mtime 早于本次启动），不按它计项数"
    return Result(entry, status, items, note, seconds, log)


# --------------------------------------------------------------------------- 自检（能变红的证据）


def selftest(log_dir: Path) -> int:
    """判据自检：该红的必须红、该绿的必须绿、该判未验的必须判未验。
    任何一条不符期望，说明这套 runner 要么会假通过，要么会把一次真通过说成「未验」——后者同样没人当回事。"""
    cases: list[tuple[str, str, int, tuple[str, tuple[int, int], str]]] = []
    # (名字, 喂给 classify 的 stdout, 退出码, 期望 (状态, 期望项数, 说明))
    cases.append(("绿色判定行 + 非零退出码 → 不通过", "PASS a\nPASS b\n结果：全部通过\n", 3, ("FAIL", (2, 0))))
    cases.append(("判定行全部通过但日志里有 FAIL 行 → 不通过", "PASS a\nFAIL b 残差 5 > 1\n结果：全部通过\n", 0, ("FAIL", (1, 1))))
    cases.append(("不通过判定行 → 不通过", "PASS a\nFAIL b\n结果：1 项不通过 -> ['b']\n", 0, ("FAIL", (1, 1))))
    cases.append(("没有判定行 → 未验（不算通过）", "14 份语料跑完，每份一行指标\n", 0, ("UNVERIFIED", (0, 0))))
    cases.append(("脚本自己说「本轮未验」 → 未验（不算通过）", "语料为空：本轮未验\n结论=全部通过\n", 0, ("UNVERIFIED", (0, 0))))
    cases.append(("干净的判定行 → 通过", "PASS a\nPASS b\n共 2 项，失败 0 项\n结论=全部通过\n", 0, ("OK", (2, 0))))
    # 真实存在但写法不同的判定行（_test_lowcontrast.py 就是这种裸结尾），
    # 判据必须认得它们——不然一次真通过会被报成「未验」，比假通过更没人当回事。
    cases.append(("裸「全部通过」结尾 → 通过", "PASS  甲\nPASS  乙\n\n全部通过\n", 0, ("OK", (2, 0))))
    cases.append(("裸「FAILED 2 项」结尾 → 不通过且汇总行不算一项", "PASS  甲\nFAIL  乙\n\nFAILED 2 项: ['乙', '丙']\n", 1, ("FAIL", (1, 1))))
    cases.append(("只有逐项 PASS 行、没有判定行 → 未验", "PASS  低对比检测全部通过\nPASS  甲\n", 0, ("UNVERIFIED", (2, 0))))
    # _parallel_ab.py 这两处的写法（逐项 [OK]、判定行走 stdout、stderr 是预期内坏件的整段回溯）
    # 第一次真跑就是把一次真通过报成了「未验」——判据必须认得它，也必须有哨兵钉住它。
    cases.append(
        (
            "逐项 [OK] + stdout 判定行 + stderr 长回溯 → 通过且项数按 [OK] 数",
            "[OK] 顺序按任务原序 = True\n[OK] 成品逐文件 md5 一致 = True\n结论= 全部通过\n",
            0,
            ("OK", (2, 0)),
        )
    )
    bad = 0
    print("== 判据自检（每条都必须按期望变红/变未验）==")
    for name, text, code, (want_status, want_items) in cases:
        status, items, note = classify(Entry("_synthetic.py"), text, code, "")
        ok = status == want_status and items == want_items
        if not ok:
            bad += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [实得 {status} {items}｜期望 {want_status} {want_items}]")
    # stderr 把判定行挤出窗口这一条要单独验：classify 的两个来源是分开看的，
    # 合成用例里塞不出「几百行 stderr」的真实体量，所以直接量 verdict_line 的取用顺序。
    noise = "\n".join(f"Traceback line {i}" for i in range(400))
    got = verdict_line("结论= 全部通过\n", noise)
    ok = got == "结论= 全部通过"
    bad += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  stderr 几百行回溯不得盖掉 stdout 结尾的判定行  [实得「{got}」]")
    # 真脚本级的端到端证据：把真回归脚本拷到临时目录、在末尾注入一次非零退出，
    # runner 必须仍然报不通过（证明「跑过了但末尾崩了」不会蒙混过关）。
    # 这条自检自己也要有判据：拷件必须真的跑到打印过判定行才算数，
    # 否则它只是因为 import 不上而崩，量到的不是「末尾崩」这件事。
    src = QA / "_test_lowcontrast.py"
    if src.is_file():
        fake = log_dir / "_canary_exit.py"
        fake.write_text(src.read_text(encoding="utf-8") + "\nraise SystemExit(7)\n", encoding="utf-8")
        entry = Entry("_test_lowcontrast.py")
        child_env = dict(os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"
        # 拷件在临时目录，sys.path[0] 就不是项目根；不补这一句它连 remover 都 import 不上，
        # 于是「必须变红」会因为完全错误的原因变红。
        child_env["PYTHONPATH"] = os.pathsep.join([str(BASE), child_env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        proc = subprocess.run(
            [sys.executable, str(fake)],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=600,
        )
        merged = proc.stdout + proc.stderr
        status, items, note = classify(entry, proc.stdout, proc.returncode, proc.stderr)
        ran = bool(verdict_line(proc.stdout, proc.stderr)) and items[0] > 0
        ok = status == "FAIL" and ran
        if not ok:
            bad += 1
        print(
            f"{'PASS' if ok else 'FAIL'}  真脚本拷件跑到判定行后注入 SystemExit(7) → 必须不通过"
            f"  [实得 {status}，退出码 {proc.returncode}，PASS 行 {items[0]} 行，判定行{'在' if ran else '不在'}"
            f"{'' if ran else '：拷件没跑到判定行，这条自检没有说服力'}]"
        )
        fake.unlink(missing_ok=True)
    else:
        bad += 1
        print("FAIL  找不到 _test_lowcontrast.py，端到端那条自检没跑成")
    print(f"\n自检 {len(cases) + 2} 项，不合格 {bad} 项")
    print("结论=" + ("判据可用：该红的都红了" if bad == 0 else f"判据有问题：{bad} 项没按期望变红"))
    return 1 if bad else 0


# --------------------------------------------------------------------------- 主流程


def select(args: argparse.Namespace) -> list[Entry]:
    wanted = set(GROUPS) if args.all else {"core"}
    for g in ("slow", "real", "ui", "metric"):
        if getattr(args, g, False):
            wanted.add(g)
    entries = [e for e in MANIFEST if e.cost in wanted]
    if args.only:
        keys = [k.strip().lstrip("_") for k in args.only.split(",") if k.strip()]
        picked = [
            e
            for e in entries
            if any(e.file == k or e.file == f"_{k}.py" or k in e.name or k in " ".join(e.args) for k in keys)
        ]
        unknown = [k for k in keys if not any(k in e.file or k in " ".join(e.args) for e in MANIFEST)]
        if unknown:
            print("清单里没有这些名字（--list 看全量）：" + "、".join(unknown))
        for k in [k for k in keys if k not in unknown]:
            hit = [e for e in MANIFEST if k in e.file or k in e.name or k in " ".join(e.args)]
            if not any(e in entries for e in hit):
                groups = sorted({e.cost for e in hit})
                flags = " ".join("--" + g for g in groups if g != "core")
                print(
                    f"「{k}」在清单里但属于 {'/'.join(groups)} 档，本次没开这一档"
                    + (f" → 加 {flags} 或 --all" if flags else "")
                    + "；「一个条目都没选中」和「跑了但没有通过」是两回事，别混着读。"
                )
        return picked
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="净印回归统一入口")
    parser.add_argument("--all", action="store_true", help="跑全部档位（含 slow/real/ui/metric）")
    for g in ("slow", "real", "ui", "metric"):
        parser.add_argument(f"--{g}", action="store_true", help=f"额外跑 {g} 档")
    parser.add_argument("--only", help="只跑名字里含这些串的条目，逗号分隔")
    parser.add_argument("--list", action="store_true", dest="list_only", help="只打印清单与环境探测")
    parser.add_argument("--selftest", action="store_true", help="跑判据自检（证明这套判据能变红）")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="保留本次为顺序契约而加的 --keep 夹具（默认只删本次跑出来的那几份 _vr_*.mp4）",
    )
    args = parser.parse_args()

    log_dir = Path(tempfile.gettempdir()) / f"jingyin_qa_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_wall = time.time()

    if args.selftest:
        try:
            return selftest(log_dir)
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    env = probe()
    base, why = http_base()
    started_at = None
    runtime = BASE / "work" / "runtime.json"
    if runtime.is_file():
        try:
            started_at = json.loads(runtime.read_text(encoding="utf-8")).get("started_at")
        except Exception:  # noqa: BLE001 探测已经在 http_base 里报过原因了
            started_at = None
    late = stale_service(started_at)

    print(f"净印回归入口 · 项目目录 {BASE}")
    print("环境探测：")
    for key in ("http", "model", "corpus", "real", "electron", "ffmpeg"):
        ok, why = env[key]
        print(f"  {'在位' if ok else '缺失'}  {key:<9} {why}")
    if late:
        print(f"  ⚠ 盘上这些文件比在跑的服务更新（服务 started_at={started_at}）：{'、'.join(late)}")
        print("    HTTP 与界面那几段验的是**旧进程里的代码**。改过产品代码请先按 §2.2 重启（只改 templates/ 文案也要重启）。")
    entries = select(args)
    print(
        f"\n本次条目 {len(entries)} 个（覆盖 {len({e.file for e in entries})} 个脚本；"
        f"核心 {sum(1 for e in entries if e.cost == 'core')}，其余按档位），串行执行；日志 → {log_dir}"
    )
    if args.list_only:
        for e in entries:
            label = " ".join([e.file, *e.args])
            print(f"  {label:<34} [{e.cost}] 需要 {'/'.join(e.needs) or '—'} —— {e.scope}")
        print("\n不在清单里的 work/qa 文件（有意排除）：")
        for name, why in EXCLUDED.items():
            print(f"  {name} —— {why}")
        shutil.rmtree(log_dir, ignore_errors=True)
        return 0

    results: list[Result] = []
    for i, entry in enumerate(entries, 1):
        label = f"{entry.file}{' ' + ' '.join(entry.args) if entry.args else ''}"
        print(f"[{i}/{len(entries)}] {label} … ", end="", flush=True)
        res = run_entry(entry, env, log_dir)
        results.append(res)
        print(f"{res.mark} 项数 {res.items[0]}+{res.items[1]} 用时 {res.seconds:.1f}s" if res.status != "SKIP" else "跳过")
        if res.status in ("FAIL", "UNVERIFIED") or res.note:
            print(f"      {res.note[:300]}")

    ok = [r for r in results if r.status == "OK"]
    fail = [r for r in results if r.status == "FAIL"]
    skip = [r for r in results if r.status == "SKIP"]
    unver = [r for r in results if r.status == "UNVERIFIED"]
    metric = [r for r in results if r.status == "METRIC"]
    total_items = sum(r.items[0] for r in results)
    print("\n== 汇总 ==")
    print(f"通过 {len(ok)} · 不通过 {len(fail)} · 未验 {len(unver)} · 跳过 {len(skip)} · 指标 {len(metric)}")
    print(f"判定的单项数合计（按各脚本判定行与 PASS 行数，不取交集）：通过项 {total_items}")
    if late:
        print("注意：HTTP / 界面段跑在旧进程上（见上面的 ⚠），这几条「通过」说的是那个进程。")
    if skip:
        print("跳过不等于验过，缺的东西如下：")
        for r in skip:
            print(f"  {r.entry.file} {' '.join(r.entry.args)} —— {r.note[:220]}")
    if unver:
        print("以下条目跑了但没有判定依据：")
        for r in unver:
            print(f"  {r.entry.file} —— {r.note[:220]}")
    if fail:
        print("不通过（日志见上面目录）：")
        for r in fail:
            print(f"  {r.entry.file} {' '.join(r.entry.args)} —— {r.note[:260]}  [{r.log}]")

    kept = [] if args.keep else cleanup_kept(session_wall)
    if kept:
        print("本次 --keep 留下的夹具已清掉：" + "、".join(kept))
    elif args.keep:
        print("--keep：_vr_* 夹具留在 work/qa（下一轮 _e2e_video_mode.py 可直接用）")

    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base": str(BASE),
                "service": base,
                "service_started_at": started_at,
                "stale_files": late,
                "keep": bool(args.keep),
                "cleanup_removed": kept,
                "env": {k: {"ok": v[0], "why": v[1]} for k, v in env.items()},
                "results": [
                    {
                        "file": r.entry.file,
                        "args": list(r.entry.args),
                        "cost": r.entry.cost,
                        "status": r.status,
                        "passed": r.items[0],
                        "failed": r.items[1],
                        "seconds": round(r.seconds, 1),
                        "note": r.note,
                        "log": str(r.log) if r.log else "",
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n机器可读汇总：{manifest_path}")
    if fail:
        return 1
    if not ok:
        print("一项都没跑出「通过」——这一次没有可交付的验证结论（跳过 / 未验 / 指标转储都不算）。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
