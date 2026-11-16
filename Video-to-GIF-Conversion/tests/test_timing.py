"""时序规划测试：帧延迟精确、端点锚定、无累积漂移。"""

import pytest

from vgif.timing import (
    MIN_GIF_DELAY_CS,
    DelayPlan,
    TimingError,
    frame_pick_indices,
    plan_gif_delays,
)


class TestFramePickIndices:
    def test_identity(self):
        assert frame_pick_indices(5, 5) == [0, 1, 2, 3, 4]

    def test_endpoints_anchored_strong_compression(self):
        # 强压缩（仿 60→24）也必须保留首尾帧。
        idx = frame_pick_indices(60, 24)
        assert idx[0] == 0
        assert idx[-1] == 59
        assert len(idx) == 24

    def test_steps_are_q_or_q_plus_one(self):
        idx = frame_pick_indices(30, 12)
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        assert set(gaps) <= {2, 3}
        assert idx[0] == 0 and idx[-1] == 29

    def test_single_output(self):
        assert frame_pick_indices(10, 1) == [0]

    def test_strictly_increasing_and_in_range(self):
        idx = frame_pick_indices(100, 37)
        assert len(idx) == 37
        assert all(a < b for a, b in zip(idx, idx[1:]))
        assert max(idx) == 99

    def test_rejects_more_output_than_source(self):
        with pytest.raises(TimingError):
            frame_pick_indices(3, 5)

    @pytest.mark.parametrize("n_src,n_out", [(0, 1), (1, 0), (0, 0)])
    def test_rejects_nonpositive(self, n_src, n_out):
        with pytest.raises(TimingError):
            frame_pick_indices(n_src, n_out)

    def test_deterministic(self):
        a = frame_pick_indices(123, 47)
        b = frame_pick_indices(123, 47)
        assert a == b


class TestPlanGifDelays:
    def test_total_matches_rounded_duration(self):
        plan = plan_gif_delays(1.0, 12, 12.0)
        # 100cs / 12 帧 → 8 或 9，和恰为 100。
        assert plan.total_delay_cs == 100
        assert set(plan.delays_cs) <= {8, 9}
        assert sum(plan.delays_cs) == 100

    def test_no_accumulating_drift(self):
        # 12fps 直接 round(100/12)=8 会每帧少 0.33cs；均摊后总误差为 0。
        plan = plan_gif_delays(10.0, 120, 12.0)
        assert plan.total_delay_cs == 1000
        assert abs(plan.duration_error) < 0.01
        # 各帧延迟最多相差 1 厘秒。
        assert max(plan.delays_cs) - min(plan.delays_cs) <= 1

    def test_half_up_rounding(self):
        # 总厘秒 = 0.015s*100=1.5 → 半值向上取整为 2cs（非银行家舍入）。
        plan = plan_gif_delays(0.015, 1)
        assert plan.total_delay_cs == 2

    def test_error_bounded_under_one_centisecond(self):
        # 选取均摊延迟 ≥2cs 的时长，验证厘秒化误差始终 <1 厘秒。
        for duration in (0.333, 1.237, 7.777):
            n_out = max(1, int(duration * 10))  # 约 10fps，延迟≈10cs
            plan = plan_gif_delays(duration, n_out, n_out / duration)
            assert abs(plan.duration_error) < 0.01

    def test_rejects_below_safe_delay(self):
        # 100 帧 / 1s = 100fps → 均摊 1cs < 安全下限 2cs。
        with pytest.raises(TimingError, match="50fps"):
            plan_gif_delays(1.0, 100)

    def test_boundary_50fps_allowed(self):
        # 50fps 恰为 2cs，落在安全下限，允许。
        plan = plan_gif_delays(1.0, 50, 50.0)
        assert min(plan.delays_cs) == MIN_GIF_DELAY_CS

    def test_rejects_bad_inputs(self):
        with pytest.raises(TimingError):
            plan_gif_delays(0.0, 10)
        with pytest.raises(TimingError):
            plan_gif_delays(1.0, 0)
        with pytest.raises(TimingError):
            plan_gif_delays(-1.0, 5)

    def test_fps_recorded(self):
        plan = plan_gif_delays(2.0, 24, 12.0)
        assert plan.target_fps == 12.0
        assert isinstance(plan, DelayPlan)

    def test_delay_values_in_gif_field_range(self):
        plan = plan_gif_delays(60.0, 12, 0.2)
        assert all(0 <= d <= 65535 for d in plan.delays_cs)
