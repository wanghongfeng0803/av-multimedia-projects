"""ffmpeg 端到端集成测试：无 ffmpeg 时整体 skip。

验证四件最关键的事：

1. **输出一致性**：帧率 / 尺寸 / 像素格式 / 帧数与输入一致，音轨直通；
2. **打码生效且只糊区域内**：区域内块逐通道动态范围塌缩（细节被
   均值化），区域外像素与输入统计一致；
3. **运行时增删**：逐帧块表源（自动检测契约）中途增删区域，前段帧
   与输入一致（已稳定帧无闪烁），后段帧正确打码；
4. **越界区域**：部分越界被钳制、完全越界被丢弃、负尺寸归一化。

有损编码口径（与 vdenoise 测试同风格）：不做逐字节精确比较，一律
用统计断言——块内逐通道最大差 <= 16（swscale rgb→yuv420p 色度
抽样抖动 + 量化噪声，帧角实测约 14；打码精确性由 test_mosaic.py
的逐字节单测保证），区域外 / 恒等基线用平均绝对差 <= 6（实测
约 3）。testsrc2 在 64x48 下块 (16,16) 的 R 通道动态范围约 65，
打码后塌缩到 <= 16，是「细节被销毁」的稳健证据。
"""

import pytest

from tests.conftest import (  # noqa: F401
    FFMPEG,
    FFPROBE,
    decode_rgb_frame,
    needs_ffmpeg,
    probe_video,
)
from vblur import RegionManager, blur_video, pix_fmt_matches
from vblur.pipeline import _encoder_for

W, H = 64, 48

#: 块内逐通道最大差容差（色度抽样抖动 + 量化噪声，实测 <= 14）
BLOCK_TOL = 16
#: 区域外 / 恒等往返平均绝对差容差（实测约 3）
MEAN_TOL = 6.0


def _block_channel_range(frame, bx, by, bw, bh):
    """rgb24 帧指定矩形内逐通道动态范围 ``[R, G, B]``。

    块不小于 4x4 时内缩 1 像素再测：有损编码的边界振铃集中在块
    边缘一圈，块内核心才是「均值回填」的稳定证据。
    """
    if bw >= 4 and bh >= 4:
        bx, by, bw, bh = bx + 1, by + 1, bw - 2, bh - 2
    out = []
    for c in range(3):
        vals = [
            frame[(y * W + x) * 3 + c]
            for y in range(by, by + bh) for x in range(bx, bx + bw)
        ]
        out.append(max(vals) - min(vals))
    return out


def _assert_block_constant(frame, bx, by, bw, bh, tol=BLOCK_TOL):
    """断言块内逐通道动态范围不超过 ``tol``（均值回填的特征）。"""
    ranges = _block_channel_range(frame, bx, by, bw, bh)
    assert max(ranges) <= tol, f"块 ({bx},{by}) 逐通道范围 {ranges} 超差"


def _mean_abs_diff(a, b):
    assert len(a) == len(b)
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class TestEncoderSelection:
    """编码器/像素格式映射的纯逻辑测试。"""

    def test_yuv_passthrough(self):
        assert _encoder_for("yuv420p") == ("libx264", "yuv420p")
        assert _encoder_for("yuvj420p") == ("libx264", "yuv420p")
        assert _encoder_for("yuv444p") == ("libx264", "yuv444p")

    def test_rgb_switches_encoder(self):
        assert _encoder_for("rgb24") == ("libx264rgb", None)
        assert _encoder_for("gbrp") == ("libx264rgb", None)

    def test_exotic_falls_back(self):
        assert _encoder_for("gray") == ("libx264", "yuv420p")
        assert _encoder_for("nv12") == ("libx264", "yuv420p")

    def test_pix_fmt_matches_rgb_equiv(self):
        assert pix_fmt_matches("gbrp", "rgb24")
        assert pix_fmt_matches("yuv420p", "yuvj420p")
        assert not pix_fmt_matches("yuv420p", "yuv444p")


@needs_ffmpeg
class TestEndToEnd:
    def test_output_consistency(self, video_factory, tmp_path):
        src = video_factory(frames=24, fps=24, with_audio=True)
        dst = tmp_path / "out.mp4"
        stats = blur_video(src, dst, regions=[(8, 8, 24, 16)], block=8)
        assert stats["frames"] == 24

        vin, ain = probe_video(src)
        vout, aout = probe_video(dst)
        assert vout["width"] == vin["width"] == W
        assert vout["height"] == vin["height"] == H
        assert vout["avg_frame_rate"] == vin["avg_frame_rate"]
        assert pix_fmt_matches(vout["pix_fmt"], vin["pix_fmt"])
        assert int(vout["nb_frames"]) == int(vin["nb_frames"]) == 24
        # 时长一致（时间戳口径：帧数 × 帧率）。
        assert abs(float(vout["duration"]) - float(vin["duration"])) < 0.05
        assert ain and aout  # 音轨直通

    def test_mosaic_only_inside_region(self, video_factory, tmp_path):
        src = video_factory(frames=6, fps=6)
        dst = tmp_path / "out.mp4"
        blur_video(src, dst, regions=[(16, 16, 16, 16)], block=8)

        before = decode_rgb_frame(src, 3, W, H)
        after = decode_rgb_frame(dst, 3, W, H)
        # 细节被销毁：源块 (16,16) R 通道动态范围约 65，打码后塌缩。
        assert max(_block_channel_range(before, 16, 16, 8, 8)) >= 32
        # 区域内 2x2 个块各自单色。
        for by in (16, 24):
            for bx in (16, 24):
                _assert_block_constant(after, bx, by, 8, 8)
        # 区域外与输入统计一致（打码不外溢）。
        outside_a = bytearray()
        outside_b = bytearray()
        for y in range(H):
            for x in range(W):
                if 16 <= x < 32 and 16 <= y < 32:
                    continue
                o = (y * W + x) * 3
                outside_a += before[o:o + 3]
                outside_b += after[o:o + 3]
        assert _mean_abs_diff(outside_a, outside_b) <= MEAN_TOL

    def test_runtime_add_via_source(self, video_factory, tmp_path):
        """逐帧块表源契约：中途加入区域，已稳定帧与输入一致。"""
        src = video_factory(frames=12, fps=12)
        dst = tmp_path / "out.mp4"
        manager = RegionManager(W, H)
        source = manager.as_source(8)

        def on_frame(n):
            # 处理完第 5 帧（n 从 1 计）后加入区域，第 6 帧起生效。
            if n == 5:
                manager.add(16, 16, 16, 16)

        blur_video(src, dst, source=source, progress=on_frame)

        # 前段帧：无区域，与输入统计一致（已稳定帧不受后续增删影响）。
        assert _mean_abs_diff(
            decode_rgb_frame(src, 2, W, H), decode_rgb_frame(dst, 2, W, H),
        ) <= MEAN_TOL
        # 后段帧：区域内块单色。
        blurred = decode_rgb_frame(dst, 8, W, H)
        for by in (16, 24):
            for bx in (16, 24):
                _assert_block_constant(blurred, bx, by, 8, 8)

    def test_out_of_bounds_regions(self, video_factory, tmp_path):
        src = video_factory(frames=6, fps=6)
        dst = tmp_path / "out.mp4"
        stats = blur_video(
            src, dst,
            regions=[
                (-10, -10, 20, 20),   # 部分越界 → 钳制为 (0,0,10,10)
                (500, 500, 10, 10),   # 完全越界 → 丢弃
                (30, 30, -8, -8),     # 负尺寸 → (22,22,8,8)
            ],
            block=8,
        )
        assert stats["frames"] == 6
        frame = decode_rgb_frame(dst, 3, W, H)
        _assert_block_constant(frame, 0, 0, 8, 8)    # 钳制后的角落块
        _assert_block_constant(frame, 16, 16, 8, 8)  # 负尺寸归一化命中块

    def test_block_one_identity_baseline(self, video_factory, tmp_path):
        src = video_factory(frames=6, fps=6)
        dst = tmp_path / "out.mp4"
        blur_video(src, dst, regions=[(0, 0, 64, 48)], block=1)
        # block=1 恒等：与输入统计一致（有损往返平均差 <= 6）。
        assert _mean_abs_diff(
            decode_rgb_frame(src, 3, W, H), decode_rgb_frame(dst, 3, W, H),
        ) <= MEAN_TOL
