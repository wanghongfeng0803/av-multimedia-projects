"""命名模板纯逻辑测试：占位符渲染、零填充、安全校验与 ffmpeg 模式翻译。"""

from __future__ import annotations

import os

import pytest

from hls_packager.errors import NamingError
from hls_packager.naming import DEFAULT_NAME_TEMPLATE, NamingConfig


def test_default_template_matches_legacy_filenames():
    # 默认模板在不同序号下都必须与历史 seg_0000.ts 形态逐字节一致
    naming = NamingConfig()
    assert naming.render(0, 0) == "seg_0000.ts"
    assert naming.render(7, 7) == "seg_0007.ts"
    assert naming.render(1234, 1234) == "seg_1234.ts"
    # 超过 4 位宽度自然扩位，不截断
    assert naming.render(10000, 10000) == "seg_10000.ts"


def test_custom_prefix_and_extension():
    naming = NamingConfig(name_prefix="chunk", extension="m4s")
    assert naming.render(0, 0) == "chunk_0000.m4s"
    assert naming.render(2, 5) == "chunk_0005.m4s"


def test_index_uses_run_local_numbering():
    # {index} 与绝对序号解耦：sequence_start=100 时 index 仍从 0 起
    naming = NamingConfig(
        template="{name}-{index:03d}.{ext}",
        name_prefix="v",
        extension="ts",
    )
    names = naming.render_many(3, sequence_start=100)
    assert names == ["v-000.ts", "v-001.ts", "v-002.ts"]


def test_sequence_uses_absolute_numbering():
    naming = NamingConfig(
        template="{name}_{sequence:05d}.{ext}",
        name_prefix="ep",
        extension="mp4",
    )
    names = naming.render_many(2, sequence_start=42)
    assert names == ["ep_00042.mp4", "ep_00043.mp4"]


def test_no_padding_and_custom_literal_layout():
    naming = NamingConfig(
        template="{name}.{index}.{ext}",
        name_prefix="clip",
        extension="mkv",
    )
    assert naming.render(9, 20) == "clip.9.mkv"


def test_render_many_detects_duplicate_names():
    # 漏掉序号占位符本身会被 validate 拦截；这里构造"序号不参与区分"
    # 的退化模板同样在校验阶段失败
    with pytest.raises(NamingError):
        NamingConfig(template="{name}.{ext}", name_prefix="a")


def test_requires_exactly_one_numeric_placeholder():
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_{index:02d}_{sequence:02d}.{ext}")
    with pytest.raises(NamingError):
        NamingConfig(template="{name}.{ext}")


def test_requires_name_placeholder():
    with pytest.raises(NamingError):
        NamingConfig(template="{sequence:04d}.{ext}")


def test_unknown_placeholder_rejected():
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_{timestamp}.{ext}")


def test_unbalanced_braces_rejected_at_construction():
    # 游离花括号与拼写残缺的占位符都应在构造期（而非渲染期）失败
    for bad in (
        "{name}_{sequence:04d.{ext}",
        "{name}_clip}.{ext}",
        "{name}_{seq}.{ext}",
    ):
        with pytest.raises(NamingError):
            NamingConfig(template=bad)


def test_non_zero_pad_spec_rejected():
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_{index:>5}.{ext}")
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_{index:.2f}.{ext}")
    # 非序号字段不接受任何格式说明
    with pytest.raises(NamingError):
        NamingConfig(template="{name:>3}_{sequence:04d}.{ext}")


def test_duplicate_field_rejected():
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_{name}_{sequence:04d}.{ext}")


def test_forbidden_characters_rejected():
    # 模板字面部分含路径分隔符 / Windows 保留字符
    for bad in (
        "{name}_{sequence:04d}/{ext}",
        "dir/{name}_{sequence:04d}.{ext}",
        "{name}_{sequence:04d}.{ext}\x00",
        "{name}_{sequence:04d}:a.{ext}",
    ):
        with pytest.raises(NamingError):
            NamingConfig(template=bad)
    # 前缀与扩展名同样过安全检查
    with pytest.raises(NamingError):
        NamingConfig(name_prefix="a/b")
    with pytest.raises(NamingError):
        NamingConfig(extension="t/s")


def test_literal_percent_rejected():
    with pytest.raises(NamingError):
        NamingConfig(template="{name}_100%_{sequence:04d}.{ext}")


def test_empty_template_rejected():
    with pytest.raises(NamingError):
        NamingConfig(template="   ")


def test_negative_numbers_rejected():
    naming = NamingConfig()
    with pytest.raises(NamingError):
        naming.render(-1, 0)


def test_ffmpeg_pattern_sequence_mode():
    naming = NamingConfig(
        template="{name}_{sequence:05d}.{ext}",
        name_prefix="seg",
        extension="ts",
    )
    pattern, start = naming.ffmpeg_pattern("X:/out", sequence_start=12)
    assert pattern.replace("\\", "/").endswith("X:/out/seg_%05d.ts")
    assert start == 12


def test_ffmpeg_pattern_index_mode_starts_at_zero():
    naming = NamingConfig(
        template="{name}-{index:03d}.{ext}",
        name_prefix="seg",
        extension="m4s",
    )
    pattern, start = naming.ffmpeg_pattern("X:/out", sequence_start=99)
    assert pattern.replace("\\", "/").endswith("X:/out/seg-%03d.m4s")
    # index 模式 ffmpeg 计数器永远从 0 开始，与 MEDIA-SEQUENCE 无关
    assert start == 0


def test_ffmpeg_pattern_no_padding():
    naming = NamingConfig(
        template="{name}.{index}.{ext}",
        name_prefix="c",
        extension="mkv",
    )
    pattern, start = naming.ffmpeg_pattern("out", sequence_start=0)
    assert pattern == os.path.join("out", "c.%d.mkv")
    assert start == 0


def test_numeric_field_property():
    assert NamingConfig(
        template="{name}_{index:02d}.{ext}"
    ).numeric_field == "index"
    assert NamingConfig(
        template="{name}_{sequence:02d}.{ext}"
    ).numeric_field == "sequence"


def test_default_template_constant():
    assert DEFAULT_NAME_TEMPLATE == "{name}_{sequence:04d}.{ext}"
