# -*- coding: utf-8 -*-
"""三组修复选项的唯一口径：`method` / `video_mode` / `motion`。

从前这三组取值在四层里各写各的：`server.py` 把请求体归一化一遍并写进 task 与 job，
`batch_runner.BatchItem` 自带一份默认值，`video_ops` 再比较一遍字面量（其中
「照片重绘在视频上降级成内容填充」是一条真规则），`image_ops` 逐分支比较 `method`，
前端 `app.js` 还有第三份默认值与三个 `localStorage` 键。新加第四种修复方式时，
漏改哪一层都**不会报错**——只会让那一层把新值当成非法值悄悄归回默认档。

这里只放词汇表与判定，不放状态也不放话术：

- `METHODS` / `VIDEO_MODES` 是全集，`normalize_*` 的默认值与旧写法逐字一致
  （认不出来的一律归 `smart` / `quality`，`motion` 就是 `bool(value)`）；
- `method_for_video()` 只收「视频路线上 `method` 实际会取什么值」这一条规则；
- 400 的文案不归这里（那是 HTTP 层对用户说的话），「照片重绘此刻能不能用」要问
  `photo_inpaint.available()`，那也不是词汇问题。

判定权集中到这里之后，`work/qa/_test_options.py` 才能同时问两件事：各层默认值是不是
还等于这里的默认值，以及产品代码里还有没有第二处把取值写成裸字面量。
"""

from __future__ import annotations

SMART = "smart"
INPAINT = "inpaint"
PHOTO = "photo"
METHODS = (SMART, INPAINT, PHOTO)
DEFAULT_METHOD = SMART

QUALITY = "quality"
FAST = "fast"
VIDEO_MODES = (QUALITY, FAST)
DEFAULT_VIDEO_MODE = QUALITY

# `motion` 只有真假两档，所以它的「全集」不是一个元组而是一句 `bool(value)`。
# 界面把它存进 localStorage 时写的是字符串 "true"/"false" 还是布尔，服务端不关心——
# 与旧写法一致：非空即真。


def normalize_method(value: object) -> str:
    """认不出的修复方式归到智能纹理修复（默认档），与 `server.py` 原来的三元判断同值。"""
    return value if value in METHODS else DEFAULT_METHOD


def normalize_video_mode(value: object) -> str:
    """认不出的档位归到画质优先（默认档）。"""
    return value if value in VIDEO_MODES else DEFAULT_VIDEO_MODE


def normalize_motion(value: object) -> bool:
    return bool(value)


def method_for_video(method: str) -> str:
    """视频路线上 `method` 实际会取的值。

    照片重绘只针对静态图片：LaMa 按整帧推理，视频里每一帧都跑一次的成本没人接受，
    而且逐帧路线本来就在做同样的局部修补，所以选了照片也走内容填充。
    """
    return INPAINT if method == PHOTO else method
