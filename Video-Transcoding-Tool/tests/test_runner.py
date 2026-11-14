"""运行器的纯逻辑测试：进度换算、尾部缓冲、命令参数。"""

from __future__ import annotations

from transcode_tool.runner import (
    _TailKeeper,
    _ratio_from_progress,
    build_progress_args,
)


def test_ratio_basic_and_clamped():
    assert _ratio_from_progress(5_000_000, 10.0, False) == 0.5
    assert _ratio_from_progress(0, 10.0, False) == 0.0
    # 超出总时长的异常时间戳应被夹到 1.0
    assert _ratio_from_progress(99_000_000, 10.0, False) == 1.0


def test_ratio_end_marker_forces_completion():
    assert _ratio_from_progress(None, 10.0, True) == 1.0
    assert _ratio_from_progress(2_000_000, 10.0, True) == 1.0


def test_ratio_unknown_duration():
    # 时长未知：非结束帧返回 None，结束帧返回 1.0
    assert _ratio_from_progress(123, None, False) is None
    assert _ratio_from_progress(None, None, True) == 1.0
    assert _ratio_from_progress(123, 0.0, False) is None


def test_tail_keeper_keeps_only_tail():
    keeper = _TailKeeper([])
    keeper.feed("a" * 100)
    keeper.feed("b" * 100)
    text = keeper.text()
    assert text.endswith("b" * 100)
    assert len(text) <= 4000  # 受内部上限约束，不会无限增长


def test_tail_keeper_handles_chunked_unicode():
    keeper = _TailKeeper([])
    keeper.feed("转码")
    keeper.feed("进度")
    assert keeper.text().endswith("进度")


def test_build_progress_args():
    args = build_progress_args()
    assert args == ["-progress", "pipe:1", "-nostats"]
