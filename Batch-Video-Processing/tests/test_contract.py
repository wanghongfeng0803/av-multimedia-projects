"""稳定错误分类契约的纯逻辑测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import pytest

from bvp.contract import (
    ErrorCategory,
    SKIP_CATEGORIES,
    SkipReason,
    category_is_skip,
    classify_error,
)
from bvp.errors import (
    BinaryNotFoundError,
    FFmpegRunError,
    InputNotFoundError,
    InvalidInputError,
    OutputExistsError,
    TranscodeError,
)


def test_each_core_error_has_stable_category():
    assert BinaryNotFoundError().category is ErrorCategory.BINARY_MISSING
    assert InputNotFoundError("x").category is ErrorCategory.INPUT_NOT_FOUND
    assert InvalidInputError().category is ErrorCategory.NOT_A_VIDEO
    assert OutputExistsError().category is ErrorCategory.OUTPUT_EXISTS
    assert FFmpegRunError("x").category is ErrorCategory.TRANSCODE_FAILED


def test_classify_error_reads_category_attribute_only():
    exc = InvalidInputError("无法识别")
    # 改消息文本不影响分类——批量层不依赖文本
    assert classify_error(exc) is ErrorCategory.NOT_A_VIDEO
    assert classify_error(TranscodeError("随便什么措辞")) is (
        ErrorCategory.TRANSCODE_FAILED
    )


def test_unknown_exception_falls_back_to_internal():
    # 核心契约之外的陌生异常必须有兜底分类，绝不允许逃出批量层
    assert classify_error(OSError("磁盘炸了")) is ErrorCategory.INTERNAL_ERROR
    assert classify_error(RuntimeError("意外")) is ErrorCategory.INTERNAL_ERROR
    assert classify_error(KeyError("k")) is ErrorCategory.INTERNAL_ERROR


def test_skip_boundary_is_closed_set():
    assert category_is_skip(ErrorCategory.NOT_A_VIDEO)
    assert category_is_skip(ErrorCategory.OUTPUT_EXISTS)
    assert not category_is_skip(ErrorCategory.TRANSCODE_FAILED)
    assert not category_is_skip(ErrorCategory.BINARY_MISSING)
    assert not category_is_skip(ErrorCategory.INPUT_NOT_FOUND)
    assert not category_is_skip(ErrorCategory.INTERNAL_ERROR)
    assert SKIP_CATEGORIES == frozenset({
        ErrorCategory.NOT_A_VIDEO, ErrorCategory.OUTPUT_EXISTS
    })


def test_category_values_are_stable_strings():
    # 取值字符串是外部清单的持久化契约，不允许被随手改名
    values = {c.value for c in ErrorCategory}
    assert values == {
        "binary_missing", "input_not_found", "not_a_video",
        "output_exists", "transcode_failed", "encoder_unavailable",
        "internal_error",
    }


def test_skip_reason_enum_is_complete():
    assert {r.value for r in SkipReason} == {
        "non_video", "output_exists", "inaccessible"
    }


def test_category_is_bound_at_class_level():
    # 契约分类是类级绑定：所有实例共享同一分类，且核心侧不接受分类参数
    assert InputNotFoundError.category is ErrorCategory.INPUT_NOT_FOUND
    exc1, exc2 = InputNotFoundError("a"), InputNotFoundError("b")
    assert exc1.category is exc2.category
    with pytest.raises(TypeError):
        # 构造签名不允许调用方传入/篡改分类
        InputNotFoundError("x", category=ErrorCategory.TRANSCODE_FAILED)  # type: ignore[call-arg]
