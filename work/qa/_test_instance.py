# -*- coding: utf-8 -*-
"""单实例判定的定向测试：只改 work/runtime.json（跑完还原），用假 probe 喂应答。

覆盖这些判定，每条都是「起第二个服务、两个进程写同一批目录」的真实入口：
① 记录里的端口没人应答 → 不认领；② 记录属于另一份拷贝 → 不认领；
③ 端口应答且实例身份对得上 → 认领；④ 身份对不上但目录指纹对 → 仍复用（别起第二个）；
⑤ 应答的是别的目录指纹 → 不认领；⑥ 没有记录但扫描命中活实例 → 复用并准备改写记录；
⑦ 第一次探活超时、重试命中 → 认领（并行满载就是这种）；⑧ 记录字段齐全；
⑨ 已有服务占用端口时，另一份安装不复用该端口。
"""

import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import server  # noqa: E402

FAILED: list[str] = []
BACKUP = server.RUNTIME_FILE.read_text(encoding="utf-8") if server.RUNTIME_FILE.exists() else None
PROBES: dict[int, dict | None] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def fake_probe(table: dict):
    def probe(port: int, timeout: float) -> dict | None:
        return table.get(port)

    return probe


def write_record(payload: dict) -> None:
    server.RUNTIME_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def ours(port: int, instance: str = "inst-live") -> dict:
    return {"ok": True, "app": "jingyin", "version": 1, "instance": instance, "dir": server.DIR_TAG}


def other_copy(port: int) -> dict:
    return {"ok": True, "app": "jingyin", "version": 1, "instance": "inst-other", "dir": "deadbeefdeadbeef"}


def run() -> None:
    original_probe = server.probe_health
    original_range = server.PORT_RANGE
    server.PORT_RANGE = range(8765, 8770)  # 测试用窄范围

    try:
        # ① 记录端口没人应答
        server.probe_health = fake_probe({})
        write_record({"port": 8765, "pid": 1, "instance": "inst-x", "base_dir": str(server.BASE_DIR)})
        check("① 死端口的记录不认领", server.find_existing_instance() is None)

        # ② 记录属于另一份拷贝
        server.probe_health = fake_probe({8766: ours(8766)})
        write_record({"port": 8766, "pid": 1, "instance": "inst-live", "base_dir": r"D:\别的一份\去水印工具"})
        check("② 别的拷贝的记录不认领", server.find_existing_instance() is None)

        # ③ 身份对得上
        write_record({"port": 8766, "pid": 1, "instance": "inst-live", "base_dir": str(server.BASE_DIR)})
        check("③ 身份对得上就认领", server.find_existing_instance() == {"port": 8766, "instance": "inst-live"},
              str(server.find_existing_instance()))

        # ④ 身份对不上、目录指纹对 → 复用（这是"记录被上一次启动写坏"的现实情形）
        write_record({"port": 8766, "pid": 1, "instance": "inst-stale", "base_dir": str(server.BASE_DIR)})
        check("④ 身份不符但同目录仍复用", server.find_existing_instance() == {"port": 8766, "instance": "inst-live"},
              str(server.find_existing_instance()))

        # ⑤ 应答来自别的目录指纹 → 不认领，也不能被扫描命中
        server.probe_health = fake_probe({8766: other_copy(8766)})
        write_record({"port": 8766, "pid": 1, "instance": "inst-live", "base_dir": str(server.BASE_DIR)})
        check("⑤ 别份拷贝在应答时不认领", server.find_existing_instance() is None)

        # ⑥ 没有记录，但扫描命中活实例
        server.probe_health = fake_probe({8768: ours(8768)})
        if server.RUNTIME_FILE.exists():
            server.RUNTIME_FILE.unlink()
        found = server.find_existing_instance()
        check("⑥ 无记录时靠扫描复用", found == {"port": 8768, "instance": "inst-live"}, str(found))

        # ⑦ 第一次探活超时、重试命中（并行满载的假阴性）
        calls = {"n": 0}

        def flaky(port: int, timeout: float) -> dict | None:
            if port != 8769:
                return None
            calls["n"] += 1
            return None if calls["n"] == 1 else ours(port)

        server.probe_health = flaky
        write_record({"port": 8769, "pid": 1, "instance": "inst-live", "base_dir": str(server.BASE_DIR)})
        found = server.find_existing_instance()
        check("⑦ 探活超时后重试能认领", found == {"port": 8769, "instance": "inst-live"},
              f"calls={calls['n']} found={found}")

        # ⑧ 记录字段
        server.write_runtime_file(8765, 4242, "inst-new")
        data = json.loads(server.RUNTIME_FILE.read_text(encoding="utf-8"))
        check("⑧ 记录含端口/身份/目录", data["port"] == 8765 and data["instance"] == "inst-new"
              and data["base_dir"] == str(server.BASE_DIR) and data["pid"] == 4242, str(data))

        # Werkzeug 的监听套接字也允许地址复用，用同样的选项模拟已运行的另一份安装。
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            busy_port = occupied.getsockname()[1]
            server.PORT_RANGE = range(busy_port, busy_port + 1)
            try:
                server.choose_port()
            except RuntimeError:
                check("⑨ 不选择正在监听的端口", True)
            else:
                check("⑨ 不选择正在监听的端口", False)
    finally:
        server.probe_health = original_probe
        server.PORT_RANGE = original_range
        if BACKUP is None:
            if server.RUNTIME_FILE.exists():
                server.RUNTIME_FILE.unlink(missing_ok=True)
        else:
            server.RUNTIME_FILE.write_text(BACKUP, encoding="utf-8")


if __name__ == "__main__":
    run()
    print("\n结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项不通过 -> {FAILED}"))
    sys.exit(1 if FAILED else 0)
