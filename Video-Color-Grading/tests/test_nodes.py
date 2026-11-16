"""节点抽象、有序节点链与分块引擎串联的纯逻辑测试（不依赖 ffmpeg）。

重点锁定：
- 节点链顺序即处理顺序，调换位置结果确定可预期；
- 3D LUT 节点与亮度/对比度、色相/饱和度节点可任意串联；
- 恒等节点零开销跳过、全恒等链逐字节直通；
- 经进程池（Windows spawn）分发后结果与单进程参考一致、可重复。
"""

from __future__ import annotations

import io
import os

import pytest

from color_grading.chunks import frame_chunk_sizes, map_stream
from color_grading.colorspace import map_chunk as hsv_map_chunk
from color_grading.lut import build_lut
from color_grading.lut3d import parse_cube
from color_grading.nodes import (
    BrightnessContrastNode,
    HueSaturationNode,
    LUT3DNode,
    apply_chain,
    is_identity_chain,
)

W, H = 8, 6


def _frames(count: int) -> bytes:
    return os.urandom(W * H * 3 * count)


def _invert_cube(size: int = 4):
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rr, gg, bb = r / (size - 1), g / (size - 1), b / (size - 1)
                lines.append(f"{1 - rr:.6f} {1 - gg:.6f} {1 - bb:.6f}")
    return parse_cube("\n".join(lines))


def _swap_cube(size: int = 4):
    # R/B 通道交换的 LUT，用于核对网格通道落位。
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rr, gg, bb = r / (size - 1), g / (size - 1), b / (size - 1)
                lines.append(f"{bb:.6f} {gg:.6f} {rr:.6f}")
    return parse_cube("\n".join(lines))


# --------------------------------------------------------------- 节点恒等


def test_identity_chain_detection():
    assert is_identity_chain([])
    assert is_identity_chain([
        BrightnessContrastNode(),
        HueSaturationNode(),
        LUT3DNode(_invert_cube(), 0.0),
    ])
    assert not is_identity_chain([LUT3DNode(_invert_cube(), 1.0)])
    assert not is_identity_chain([BrightnessContrastNode(build_lut(10, 0))])
    assert not is_identity_chain([HueSaturationNode(30, 0)])


def test_lut3d_node_strength_zero_is_identity_and_passthrough():
    data = _frames(1)
    node = LUT3DNode(_invert_cube(), 0.0)
    assert node.is_identity()
    assert node.apply(data) == data


# --------------------------------------------------------------- 顺序语义


def test_chain_order_matches_manual_composition():
    """节点链输出 == 按同顺序手工逐个 apply 的结果。"""
    data = _frames(1)
    cube = _invert_cube()
    table = build_lut(brightness=15, contrast=25)
    chain = [
        BrightnessContrastNode(table),
        LUT3DNode(cube, 1.0),
        HueSaturationNode(40, -20),
    ]
    expected = hsv_map_chunk(
        LUT3DNode(cube, 1.0).apply(data.translate(table)), 40, -20
    )
    assert apply_chain(data, chain) == expected


def test_node_reorder_produces_different_but_deterministic_result():
    """LUT 在前 vs 在后：结果不同（顺序真实生效），但各自可重复。"""
    data = _frames(2)
    cube = _invert_cube()
    table = build_lut(brightness=20, contrast=30)

    first = [LUT3DNode(cube, 1.0), BrightnessContrastNode(table)]
    last = [BrightnessContrastNode(table), LUT3DNode(cube, 1.0)]

    out_first = apply_chain(data, first)
    out_last = apply_chain(data, last)
    assert out_first != out_last  # 非线性变换，顺序不可交换
    # 各自确定：重复计算逐字节一致
    assert apply_chain(data, first) == out_first
    assert apply_chain(data, last) == out_last
    # 手工核对 first 的语义：先 LUT 后 1D 表
    assert out_first == LUT3DNode(cube, 1.0).apply(data).translate(table)
    assert out_last == LUT3DNode(cube, 1.0).apply(data.translate(table))


def test_two_lut3d_nodes_chain_in_order():
    """反色 LUT 串联两次应近似回到原图（顺序确定、节点可重复出现）。"""
    data = _frames(1)
    node = LUT3DNode(_invert_cube(size=32), 1.0)
    out = apply_chain(data, [node, node])
    # 两次反色 + 插值/舍入，允许每通道 ±2 的往返误差。
    for i in range(len(data)):
        assert abs(out[i] - data[i]) <= 2


# --------------------------------------------------------------- 分块引擎集成


def test_map_stream_identity_nodes_passthrough_byte_exact():
    data = _frames(3)
    out = io.BytesIO()
    frames = map_stream(
        io.BytesIO(data), out, W, H,
        nodes=[BrightnessContrastNode(), HueSaturationNode(),
               LUT3DNode(_invert_cube(), 0.0)],
        workers=2, chunk_rows=2,
    )
    assert frames == 3
    assert out.getvalue() == data


def test_map_stream_lut3d_matches_direct_apply():
    data = _frames(3)
    cube = _swap_cube()
    out = io.BytesIO()
    frames = map_stream(
        io.BytesIO(data), out, W, H,
        nodes=[LUT3DNode(cube, 0.8)], workers=2, chunk_rows=3,
    )
    assert frames == 3
    from color_grading.lut3d import apply_chunk
    assert out.getvalue() == apply_chunk(data, cube, 0.8)


def test_map_stream_full_chain_through_process_pool():
    """完整四节点链经 spawn 进程池，结果 == 单进程 apply_chain。"""
    data = _frames(2)
    cube = _invert_cube()
    chain = [
        BrightnessContrastNode(build_lut(-10, 20)),
        HueSaturationNode(35, 25),
        LUT3DNode(cube, 0.6),
    ]
    out = io.BytesIO()
    map_stream(io.BytesIO(data), out, W, H, nodes=chain, workers=2, chunk_rows=2)
    assert out.getvalue() == apply_chain(data, chain)


def test_map_stream_repeated_runs_identical():
    """同一输入两次输出逐字节一致（确定性，跨进程池重复构建）。"""
    data = _frames(2)
    chain = [LUT3DNode(_invert_cube(), 0.43), HueSaturationNode(15, -10)]
    outs = []
    for _ in range(2):
        out = io.BytesIO()
        map_stream(io.BytesIO(data), out, W, H, nodes=chain, workers=2, chunk_rows=3)
        outs.append(out.getvalue())
    assert outs[0] == outs[1]


def test_map_stream_nodes_and_legacy_args_are_mutually_exclusive():
    data = _frames(1)
    with pytest.raises(ValueError):
        map_stream(
            io.BytesIO(data), io.BytesIO(), W, H,
            lut=build_lut(10, 0), nodes=[LUT3DNode(_invert_cube(), 1.0)],
        )


def test_map_stream_legacy_args_keep_working_without_nodes():
    """不传 nodes 时旧 lut/hue/saturation 接口行为不变。"""
    data = _frames(2)
    table = build_lut(brightness=12, contrast=18)
    out_new = io.BytesIO()
    map_stream(io.BytesIO(data), out_new, W, H, table,
               hue=-40, saturation=30, workers=2, chunk_rows=3)
    expected = hsv_map_chunk(data.translate(table), -40, 30)
    assert out_new.getvalue() == expected


def test_map_stream_lut3d_position_first_vs_last_via_chain():
    """通过节点链显式控制 LUT 位置，两种位置都能跑通且结果不同。"""
    data = _frames(1)
    cube = _invert_cube()
    table = build_lut(brightness=25, contrast=10)
    bc = BrightnessContrastNode(table)
    lutn = LUT3DNode(cube, 1.0)

    out_first = io.BytesIO()
    map_stream(io.BytesIO(data), out_first, W, H, nodes=[lutn, bc],
               workers=1, chunk_rows=2)
    out_last = io.BytesIO()
    map_stream(io.BytesIO(data), out_last, W, H, nodes=[bc, lutn],
               workers=1, chunk_rows=2)
    assert out_first.getvalue() != out_last.getvalue()


def test_frame_chunk_sizes_still_valid():
    # 分块几何不受节点化影响。
    assert sum(frame_chunk_sizes(W, H, 2)) == W * H * 3
