# -*- coding: utf-8 -*-
"""素材索引落盘与服务重启后的恢复（已接入 server.py）。

选框存在浏览器 localStorage 里、按 item_id 索引，所以恢复的关键是**重启后 item_id 不变**：
每个素材目录留一份 meta.json，启动时扫 work/uploads 把索引装回内存即可。
正在跑的任务不恢复——进程没了就是没了，只恢复素材与选框这些人工劳动。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

META_NAME = "meta.json"
ID_RE = re.compile(r"^[0-9a-f]{8,32}$")
_PATH_FIELDS = ("source_path", "thumb_path", "preview_path")
_ITEM_FIELDS = (
    "id",
    "name",
    "kind",
    "suffix",
    "width",
    "height",
    "duration",
    "fps",
    "frame_count",
    "source_path",
    "thumb_path",
    "preview_path",
    "created_at",
)


def write_item_meta(item: dict, item_dir: Path) -> None:
    """把一条素材索引写成 meta.json，口径同成品：先 .part 再 replace。"""
    payload = {}
    for key in _ITEM_FIELDS:
        value = item.get(key)
        payload[key] = str(value) if isinstance(value, Path) else value
    target = item_dir / META_NAME
    tmp = target.with_name(META_NAME + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def read_item_meta(meta_path: Path) -> dict | None:
    """读回一条素材索引；文件坏掉或源文件已不在就返回 None，不抛异常。"""
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    item = {}
    for key in _ITEM_FIELDS:
        if key in payload:
            item[key] = payload[key]
    for key in _PATH_FIELDS:
        value = item.get(key)
        item[key] = Path(value) if value else None
    source = item.get("source_path")
    if not item.get("id") or source is None or not Path(source).exists():
        return None
    if not Path(item["id"]).name.isalnum():
        return None
    return item


def rebuild_sessions(upload_root: Path, sessions: dict, lock) -> tuple[int, int]:
    """扫描 uploads/<sid>/<item_id>/meta.json 重建 SESSIONS，返回 (会话数, 素材数)。"""
    restored_sessions = 0
    restored_items = 0
    if not upload_root.exists():
        return 0, 0
    for sid_dir in sorted(upload_root.iterdir()):
        if not sid_dir.is_dir() or not ID_RE.match(sid_dir.name):
            continue
        items = {}
        for item_dir in sorted(sid_dir.iterdir()):
            if not item_dir.is_dir() or not ID_RE.match(item_dir.name):
                continue
            item = read_item_meta(item_dir / META_NAME)
            if item and item["id"] == item_dir.name:
                items[item["id"]] = item
        if not items:
            continue
        created = min(item.get("created_at") or time.time() for item in items.values())
        with lock:
            store = sessions.get(sid_dir.name)
            if store is None:
                sessions[sid_dir.name] = {"items": items, "created": created}
                restored_sessions += 1
                restored_items += len(items)
            else:
                # 内存里已有的会话只补缺，绝不覆盖正在使用中的条目
                missing = [key for key in items if key not in store["items"]]
                store["items"].update({key: items[key] for key in missing})
                restored_items += len(missing)
    return restored_sessions, restored_items
