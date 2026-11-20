# -*- coding: utf-8 -*-
"""任务取消：一批素材跑起来之后，用户随时可以按「停止」。

视频逐帧路线（画质优先＋静止水印掩膜）比 delogo 慢一到两个数量级（实拍 8s 片段
实测 15.99s vs 0.57s，4K 逐帧 35–47s / 4s 素材），而 ffmpeg 的两个管道没有超时总闸，
所以「开始后只能等它跑完」是会被真实素材触发的死等。

引擎在各自的检查点调用 `should_stop()`，返回真就抛 `CancelRequested`：
- `video_ops.process_video_file` 已有的 `.part` 清理分支会顺手删掉半成品，
  交付目录里不会留下半个能打开的视频；
- `batch_runner` 把它记成 `cancelled` 而不是 `error`，不污染失败计数；
- 排到队但还没开始的那几份直接跳过，不再白跑。
"""

from __future__ import annotations


class CancelRequested(RuntimeError):
    """用户请求停止当前任务。这不是失败，不参与失败计数与错误文案。"""
