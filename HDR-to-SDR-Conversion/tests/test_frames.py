"""FrameProcessor 引擎测试：合成原始帧，验证管线确定性与画质不变量。

不依赖 ffmpeg 的部分直接构造字节缓冲；需要真实解码的端到端测试见
test_pipeline.py。
"""

from __future__ import annotations

import array

import pytest

from hdr_to_sdr.frames import FrameProcessor
from hdr_to_sdr.transfer import TRANSFER_PQ, TRANSFER_HLG
from hdr_to_sdr.colorscience import RANGE_LIMITED, RANGE_FULL


def _make_yuv444p16_frame(width: int, height: int, y: int, u: int, v: int) -> bytes:
    """构造三平面均为常数码值的 yuv444p16le 帧。"""
    plane = array.array("H", [y]) * (width * height)
    u_plane = array.array("H", [u]) * (width * height)
    v_plane = array.array("H", [v]) * (width * height)
    return plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()


def _parse_yuv444p16(frame: bytes, width: int, height: int):
    """解析 yuv444p16le 三平面。注意输出码深由 FrameProcessor.dst_bits 决定，
    默认 8bit 时码值落在 8bit 域（黑 16、中点 128），缓冲宽度仍是 16bit。"""
    samples = array.array("H")
    samples.frombytes(frame)
    plane = width * height
    return samples[0:plane], samples[plane:2 * plane], samples[2 * plane:3 * plane]


class TestProcessorConstruct:
    def test_unknown_transfer_raises(self):
        with pytest.raises(ValueError):
            FrameProcessor("nonsense", 10, RANGE_LIMITED, False)

    def test_pq_peak_from_metadata(self):
        proc = FrameProcessor(TRANSFER_PQ, 10, RANGE_LIMITED, False,
                              peak_nits=4000.0)
        # 4000nit 归一化到 10000nit 域 = 0.4
        assert proc.peak == pytest.approx(0.4)

    def test_pq_peak_default(self):
        proc = FrameProcessor(TRANSFER_PQ, 10, RANGE_LIMITED, False)
        assert proc.peak == pytest.approx(0.1)  # 默认 1000nit

    def test_hlg_peak_unity(self):
        proc = FrameProcessor(TRANSFER_HLG, 10, RANGE_LIMITED, False)
        assert proc.peak == 1.0


class TestYUVPath:
    def _proc(self, transfer=TRANSFER_PQ, **kw):
        return FrameProcessor(transfer, 10, RANGE_LIMITED, False, **kw)

    def test_black_frame(self):
        proc = self._proc()
        # limited 10bit 输入：黑场 Y=64，色差中点 542（注意不是 full range 的 512）
        frame = _make_yuv444p16_frame(8, 4, 64, 542, 542)
        out = proc.process_frame(frame, 8, 4)
        y, u, v = _parse_yuv444p16(out, 8, 4)
        # 输出为 8bit limited 域：黑电平 16，色差中点 128
        assert all(c <= 18 for c in y)
        assert all(c == 128 for c in u)
        assert all(c == 128 for c in v)

    def test_gray_frame_neutral(self):
        """中性灰：输出 U/V 必须等于 8bit 色差中点（不偏色的硬指标）。"""
        proc = self._proc()
        frame = _make_yuv444p16_frame(8, 4, 512, 542, 542)
        out = proc.process_frame(frame, 8, 4)
        _, u, v = _parse_yuv444p16(out, 8, 4)
        assert all(c == 128 for c in u)
        assert all(c == 128 for c in v)

    def test_hlg_gray_neutral(self):
        proc = self._proc(TRANSFER_HLG)
        frame = _make_yuv444p16_frame(8, 4, 512, 542, 542)
        out = proc.process_frame(frame, 8, 4)
        _, u, v = _parse_yuv444p16(out, 8, 4)
        assert all(c == 128 for c in u)
        assert all(c == 128 for c in v)

    def test_output_codes_in_limited_range(self):
        """输出码值不得越过 8bit limited 范围（不过曝/不越界）。"""
        proc = self._proc()
        for y_code in (64, 256, 512, 940, 1023):
            frame = _make_yuv444p16_frame(4, 4, y_code, 542, 542)
            out = proc.process_frame(frame, 4, 4)
            y, u, v = _parse_yuv444p16(out, 4, 4)
            assert all(16 <= c <= 235 for c in y)
            assert all(16 <= c <= 240 for c in u)
            assert all(16 <= c <= 240 for c in v)

    def test_deterministic(self):
        """同一输入两次处理必须逐字节相同。"""
        proc = self._proc()
        frame = _make_yuv444p16_frame(16, 8, 700, 400, 600)
        out1 = proc.process_frame(frame, 16, 8)
        out2 = proc.process_frame(frame, 16, 8)
        assert out1 == out2

    def test_band_parallel_matches_serial(self):
        """分带并行结果必须与串行逐像素一致（并行只切分、不改数学）。"""
        proc = self._proc()
        frame = _make_yuv444p16_frame(32, 16, 700, 400, 600)
        serial = proc.process_frame(frame, 32, 16)
        parallel = proc.process_bands(frame, 32, 16, workers=2, band_height=8)
        assert serial == parallel

    def test_full_range_input(self):
        proc = FrameProcessor(TRANSFER_PQ, 10, RANGE_FULL, False)
        # full range 黑场 Y=0、色差中点 512；输出仍为 8bit limited
        frame = _make_yuv444p16_frame(8, 4, 0, 512, 512)
        out = proc.process_frame(frame, 8, 4)
        y, u, v = _parse_yuv444p16(out, 8, 4)
        assert all(c <= 18 for c in y)
        assert all(c == 128 for c in u)


class TestReinhardClassicEngine:
    def _proc(self, **kw):
        return FrameProcessor(
            TRANSFER_PQ, 10, RANGE_LIMITED, False,
            tone_method="reinhard_classic", **kw
        )

    def test_white_clamped_at_construction(self):
        # 越界白点在构造期即被钳制，处理器上持有确定的合法值
        with pytest.warns(UserWarning):
            proc = self._proc(reinhard_white=-9.0)
        from hdr_to_sdr.tonemap import REINHARD_WHITE_MIN
        assert proc.reinhard_white == REINHARD_WHITE_MIN

    def test_different_white_changes_output(self):
        frame = _make_yuv444p16_frame(8, 4, 900, 542, 542)
        out_w1 = self._proc(reinhard_white=1.0).process_frame(frame, 8, 4)
        out_w4 = self._proc(reinhard_white=4.0).process_frame(frame, 8, 4)
        y1, _, _ = _parse_yuv444p16(out_w1, 8, 4)
        y4, _, _ = _parse_yuv444p16(out_w4, 8, 4)
        # white 大压缩轻 → Y 码值更高（至少有一个像素能区分）
        assert sum(y4) >= sum(y1)
        assert out_w1 != out_w4

    def test_deterministic_two_runs(self):
        # 需求：同输入两次一致（逐字节）
        proc = self._proc(reinhard_white=2.0)
        frame = _make_yuv444p16_frame(16, 8, 700, 400, 600)
        assert proc.process_frame(frame, 16, 8) == proc.process_frame(frame, 16, 8)

    def test_band_parallel_matches_serial_classic(self):
        proc = self._proc(reinhard_white=1.5)
        frame = _make_yuv444p16_frame(32, 16, 700, 400, 600)
        serial = proc.process_frame(frame, 32, 16)
        parallel = proc.process_bands(frame, 32, 16, workers=2, band_height=8)
        assert serial == parallel

    def test_output_in_limited_range(self):
        proc = self._proc()
        for y_code in (64, 512, 940, 1023):
            frame = _make_yuv444p16_frame(4, 4, y_code, 542, 542)
            out = proc.process_frame(frame, 4, 4)
            y, u, v = _parse_yuv444p16(out, 4, 4)
            assert all(16 <= c <= 235 for c in y)
            assert all(16 <= c <= 240 for c in u)
            assert all(16 <= c <= 240 for c in v)


class TestRGBPath:
    def _make_gbrpf32_frame(self, width, height, r, g, b):
        plane = width * height
        gp = array.array("f", [g]) * plane
        bp = array.array("f", [b]) * plane
        rp = array.array("f", [r]) * plane
        return gp.tobytes() + bp.tobytes() + rp.tobytes()

    def _proc(self, transfer=TRANSFER_PQ):
        return FrameProcessor(transfer, 10, RANGE_FULL, True)

    def test_rgb_gray_neutral(self):
        proc = self._proc()
        frame = self._make_gbrpf32_frame(8, 4, 0.5, 0.5, 0.5)
        out = proc.process_frame(frame, 8, 4)
        floats = array.array("f")
        floats.frombytes(out)
        plane = 8 * 4
        g, b, r = floats[0:plane], floats[plane:2 * plane], floats[2 * plane:3 * plane]
        for i in range(plane):
            # 浮点 RGB 路径允许矩阵/OETF 累积误差，但必须远小于一个 8bit 码级
            assert r[i] == pytest.approx(g[i], abs=1e-4)
            assert g[i] == pytest.approx(b[i], abs=1e-4)

    def test_rgb_output_bounded(self):
        proc = self._proc()
        # 输入满幅高亮，输出必须被压到 [0,1]
        frame = self._make_gbrpf32_frame(8, 4, 1.0, 1.0, 1.0)
        out = proc.process_frame(frame, 8, 4)
        floats = array.array("f")
        floats.frombytes(out)
        assert all(0.0 <= x <= 1.0 for x in floats)

    def test_rgb_black(self):
        proc = self._proc()
        frame = self._make_gbrpf32_frame(4, 4, 0.0, 0.0, 0.0)
        out = proc.process_frame(frame, 4, 4)
        floats = array.array("f")
        floats.frombytes(out)
        assert all(x == 0.0 for x in floats)
