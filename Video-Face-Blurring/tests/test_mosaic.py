"""全局块网格与一次性马赛克的纯逻辑测试（不依赖 ffmpeg）。

覆盖需求核心约束：多区域重叠只打一次、共用全局网格相位（重叠处
无拼接缝）、命中块整块均值回填、舍入定点可复现、恒等基线。
"""

import pytest

from vblur.mosaic import BlockTable, apply_mosaic, build_block_table
from vblur.regions import Region


def _frame(width, height, channels=3):
    """生成内容确定的测试帧：像素 (x,y) 通道 c = (x + y*width + c) % 256。"""
    buf = bytearray(width * height * channels)
    for y in range(height):
        for x in range(width):
            base = (y * width + x) * channels
            for c in range(channels):
                buf[base + c] = (x + y * width + c) % 256
    return buf


class TestBuildBlockTable:
    def test_global_grid_phase(self):
        # 区域不按块对齐，命中块原点仍全部对齐 (0,0) 锚定的全局网格。
        table = build_block_table([Region(3, 5, 7, 9)], 64, 64, 8)
        assert table.blocks
        for x, y, w, h in table.blocks:
            assert x % 8 == 0 and y % 8 == 0

    def test_overlap_dedup_single_pass(self):
        # 两个重叠区域：重叠块只出现一次（只打一次的块表级保证）。
        table = build_block_table(
            [Region(0, 0, 12, 8), Region(8, 0, 12, 8)], 64, 64, 8,
        )
        assert table.blocks == ((0, 0, 8, 8), (8, 0, 8, 8), (16, 0, 8, 8))

    def test_edge_blocks_clamped(self):
        # 帧宽 10、块 8：右边缘块被帧边界裁为宽 2；命中块整块回填，
        # 不因区域只覆盖块的一部分而裁小（高度方向同样取整块 8）。
        table = build_block_table([Region(0, 0, 10, 4)], 10, 10, 8)
        assert table.blocks == ((0, 0, 8, 8), (8, 0, 2, 8))

    def test_empty_regions_empty_table(self):
        table = build_block_table([], 64, 64, 8)
        assert table.is_empty

    def test_sorted_by_row_then_column(self):
        table = build_block_table(
            [Region(24, 8, 8, 8), Region(0, 0, 8, 8), Region(8, 0, 8, 8)],
            64, 64, 8,
        )
        keys = [(b[1], b[0]) for b in table.blocks]
        assert keys == sorted(keys)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            build_block_table([], 0, 64, 8)
        with pytest.raises(ValueError):
            build_block_table([], 64, 64, 0)


class TestBlockTableValidation:
    def test_rejects_misaligned_block(self):
        with pytest.raises(ValueError):
            BlockTable(64, 64, 8, ((3, 0, 8, 8),))

    def test_rejects_out_of_bounds_block(self):
        with pytest.raises(ValueError):
            BlockTable(64, 64, 8, ((56, 56, 16, 8),))

    def test_rejects_bad_grid(self):
        with pytest.raises(ValueError):
            BlockTable(64, 64, 0, ())


class TestApplyMosaic:
    def test_block_filled_with_channel_mean(self):
        width, height, block = 8, 8, 4
        frame = _frame(width, height)
        table = build_block_table([Region(0, 0, 4, 4)], width, height, block)
        apply_mosaic(frame, table)
        # 手算块 (0,0)-(3,3) 的逐通道均值。
        src = _frame(width, height)
        for c in range(3):
            vals = [
                src[(y * width + x) * 3 + c]
                for y in range(4) for x in range(4)
            ]
            mean = (sum(vals) + len(vals) // 2) // len(vals)
            for y in range(4):
                for x in range(4):
                    assert frame[(y * width + x) * 3 + c] == mean
        # 块外像素逐字节不动。
        for y in range(height):
            for x in range(width):
                if x >= 4 or y >= 4:
                    o = (y * width + x) * 3
                    assert frame[o:o + 3] == src[o:o + 3]

    def test_rounding_half_up(self):
        # 2 像素块通道值 [0, 1]：sum=1，(1 + 1) // 2 = 1（四舍五入）。
        width, height = 2, 1
        frame = bytearray([0, 0, 0, 1, 1, 1])
        table = BlockTable(width, height, 2, ((0, 0, 2, 1),))
        apply_mosaic(frame, table)
        assert frame == bytearray([1, 1, 1, 1, 1, 1])

    def test_overlap_applied_once(self):
        # 重叠区域与手工单次并集计算逐字节一致（不存在二次均值）。
        width, height, block = 16, 16, 8
        regions = [Region(0, 0, 12, 12), Region(4, 4, 12, 12)]
        frame = _frame(width, height)
        apply_mosaic(frame, build_block_table(regions, width, height, block))
        # 期望：并集块 (0,0),(8,0),(0,8),(8,8) 各自对原图取一次均值。
        src = _frame(width, height)
        expected = bytearray(src)
        for bx, by in ((0, 0), (8, 0), (0, 8), (8, 8)):
            for c in range(3):
                vals = [
                    src[(y * width + x) * 3 + c]
                    for y in range(by, by + 8) for x in range(bx, bx + 8)
                ]
                mean = (sum(vals) + len(vals) // 2) // len(vals)
                for y in range(by, by + 8):
                    for x in range(bx, bx + 8):
                        expected[(y * width + x) * 3 + c] = mean
        assert frame == expected

    def test_empty_table_is_identity(self):
        frame = _frame(8, 8)
        original = bytes(frame)
        apply_mosaic(frame, build_block_table([], 8, 8, 4))
        assert bytes(frame) == original

    def test_block_one_is_identity(self):
        # block=1：每块只有 1 像素，均值即自身 —— 恒等基线。
        frame = _frame(8, 8)
        original = bytes(frame)
        table = build_block_table([Region(0, 0, 8, 8)], 8, 8, 1)
        apply_mosaic(frame, table)
        assert bytes(frame) == original

    def test_deterministic(self):
        a = _frame(16, 16)
        b = bytearray(a)
        table = build_block_table([Region(2, 2, 10, 10)], 16, 16, 8)
        apply_mosaic(a, table)
        apply_mosaic(b, table)
        assert a == b

    def test_grayscale_channels(self):
        # channels=1：灰度帧同样成立。
        width, height = 4, 4
        frame = bytearray(range(16))
        table = build_block_table([Region(0, 0, 2, 2)], width, height, 2)
        apply_mosaic(frame, table, channels=1)
        mean = (0 + 1 + 4 + 5 + 2) // 4  # (sum + n//2)//n = (10+2)//4
        assert frame[0] == frame[1] == frame[4] == frame[5] == mean
        assert frame[2] == 2  # 块外不动
