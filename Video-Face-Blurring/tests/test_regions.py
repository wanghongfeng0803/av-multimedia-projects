"""区域钳制与运行时增删的纯逻辑测试（不依赖 ffmpeg）。

覆盖需求三条硬约束中的两条：

- 钳制：完全越界 / 部分越界 / 负尺寸 / 零面积；
- 运行时增删不影响已稳定帧：增删区域后，未涉及区域的块表逐字节
  不变（无闪烁），移除后回到增前状态。
"""

import pytest

from vblur.regions import Region, RegionManager, clamp_region


class TestClampRegion:
    def test_normal_inside(self):
        assert clamp_region(10, 20, 30, 40, 100, 100) == Region(10, 20, 30, 40)

    def test_negative_size_normalizes(self):
        # 负尺寸 = 终点坐标写法：(10,10) 到 (6,4)。
        assert clamp_region(10, 10, -4, -6, 100, 100) == Region(6, 4, 4, 6)

    def test_negative_size_only_w(self):
        assert clamp_region(10, 5, -10, 8, 100, 100) == Region(0, 5, 10, 8)

    @pytest.mark.parametrize("args,expected", [
        ((-5, 10, 20, 10), Region(0, 10, 15, 10)),    # 左越界
        ((90, 10, 20, 10), Region(90, 10, 10, 10)),   # 右越界
        ((10, -5, 10, 20), Region(10, 0, 10, 15)),    # 上越界
        ((10, 90, 10, 20), Region(10, 90, 10, 10)),   # 下越界
        ((-5, -5, 200, 200), Region(0, 0, 100, 100)),  # 四边同时越界
    ])
    def test_partial_out_of_bounds(self, args, expected):
        assert clamp_region(*args, 100, 100) == expected

    @pytest.mark.parametrize("args", [
        (200, 0, 10, 10),     # 完全在右外
        (-20, 0, 10, 10),     # 完全在左外（右沿 -10 < 0）
        (0, 200, 10, 10),     # 完全在下外
        (0, -20, 10, 10),     # 完全在上外
        (-30, -30, 10, 10),   # 完全在左上外
    ])
    def test_fully_out_of_bounds_returns_none(self, args):
        assert clamp_region(*args, 100, 100) is None

    @pytest.mark.parametrize("args", [
        (10, 10, 0, 10),    # 零宽
        (10, 10, 10, 0),    # 零高
        (10, 10, 0, 0),     # 零面积
    ])
    def test_zero_size_returns_none(self, args):
        assert clamp_region(*args, 100, 100) is None

    def test_exact_frame_coverage(self):
        assert clamp_region(0, 0, 100, 100, 100, 100) == Region(0, 0, 100, 100)

    def test_one_pixel(self):
        r = clamp_region(99, 99, 1, 1, 100, 100)
        assert r == Region(99, 99, 1, 1)
        assert r.area == 1

    def test_invalid_frame_size_raises(self):
        with pytest.raises(ValueError):
            clamp_region(0, 0, 10, 10, 0, 100)
        with pytest.raises(ValueError):
            clamp_region(0, 0, 10, 10, 100, -1)


class TestRegionManager:
    def test_add_returns_monotonic_ids(self):
        m = RegionManager(100, 100)
        assert m.add(0, 0, 10, 10) == 1
        assert m.add(20, 20, 10, 10) == 2

    def test_add_clamps_partially_out_of_bounds(self):
        m = RegionManager(100, 100)
        m.add(-5, 0, 20, 10)
        assert m.regions() == (Region(0, 0, 15, 10),)

    def test_add_fully_out_of_bounds_dropped(self):
        m = RegionManager(100, 100)
        assert m.add(500, 500, 10, 10) is None
        assert m.add(0, 0, 0, 10) is None
        assert m.regions() == ()

    def test_remove(self):
        m = RegionManager(100, 100)
        rid = m.add(0, 0, 10, 10)
        assert m.remove(rid) is True
        assert m.regions() == ()
        assert m.remove(rid) is False  # 重复删除
        assert m.remove(999) is False  # 不存在的 id

    def test_clear(self):
        m = RegionManager(100, 100)
        m.add(0, 0, 10, 10)
        m.add(20, 20, 10, 10)
        m.clear()
        assert m.regions() == ()

    def test_snapshot_isolated_from_later_mutation(self):
        m = RegionManager(100, 100)
        m.add(0, 0, 10, 10)
        snap = m.regions()
        m.add(50, 50, 10, 10)
        m.clear()
        assert snap == (Region(0, 0, 10, 10),)

    def test_invalid_frame_size_raises(self):
        with pytest.raises(ValueError):
            RegionManager(0, 100)


class TestRuntimeStability:
    """运行时增删不影响已稳定帧（无闪烁）的块表级验证。"""

    def test_add_far_region_keeps_existing_blocks(self):
        m = RegionManager(64, 64)
        m.add(0, 0, 16, 16)
        before = m.block_table(16)
        m.add(40, 40, 16, 16)  # 远处新增
        after = m.block_table(16)
        # 既有块逐字节保留，且顺序稳定（新块按 (y,x) 排序追加到末尾）。
        assert after.blocks[:len(before.blocks)] == before.blocks

    def test_remove_restores_exact_previous_table(self):
        m = RegionManager(64, 64)
        m.add(0, 0, 16, 16)
        baseline = m.block_table(16)
        rid = m.add(32, 32, 16, 16)
        assert m.block_table(16) != baseline
        m.remove(rid)
        assert m.block_table(16) == baseline  # 回到增前状态，逐字节相等

    def test_source_caches_until_mutation(self):
        m = RegionManager(64, 64)
        m.add(0, 0, 16, 16)
        src = m.as_source(16)
        t0 = src(0)
        assert src(1) is t0  # 无增删：同一对象，已稳定帧零开销
        m.add(32, 32, 16, 16)
        t1 = src(2)
        assert t1 is not t0
        assert src(3) is t1
        m.clear()
        assert src(4).is_empty

    def test_source_rejects_bad_block(self):
        m = RegionManager(64, 64)
        with pytest.raises(ValueError):
            m.as_source(0)
