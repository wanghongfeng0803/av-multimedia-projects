"""管线纯逻辑测试：帧数/尺寸推导与参数校验（不依赖 ffmpeg）。"""

import pytest

from vgif.extract import FrameInfo, Manifest
from vgif import scale as scale_mod
from vgif.pipeline import (
    DEFAULT_COLORS,
    DEFAULT_FPS,
    DEFAULT_MAX_WIDTH,
    PipelineError,
    convert_to_gif,
    derive_output_frame_count,
    derive_target_size,
    resolve_target_size,
)


def make_manifest(n=24, fps=24.0, w=320, h=240):
    frames = tuple(
        FrameInfo(index=i, pts_time=i / fps, duration_time=1.0 / fps)
        for i in range(n)
    )
    return Manifest(width=w, height=h, fps=fps, frames=frames)


class TestDeriveFrameCount:
    def test_integer_fps_ratio(self):
        m = make_manifest(n=24, fps=24)
        # 24fps × 1s → 12fps 目标 = 12 帧。
        assert derive_output_frame_count(m, 12) == 12

    def test_half_up_rounding(self):
        m = make_manifest(n=10, fps=10)  # 时长 1.0s
        # 10 × 0.333 = 3.33 → 3。
        assert derive_output_frame_count(m, 3.33) == 3

    def test_capped_at_source_count(self):
        m = make_manifest(n=5, fps=5)
        # 请求 100fps 也不能凭空造帧。
        assert derive_output_frame_count(m, 100) == 5

    def test_at_least_one(self):
        m = make_manifest(n=24, fps=24)
        assert derive_output_frame_count(m, 0.001) == 1

    def test_rejects_nonpositive_fps(self):
        with pytest.raises(PipelineError):
            derive_output_frame_count(make_manifest(), 0)


class TestDeriveTargetSize:
    def test_no_downscale_when_smaller(self):
        assert derive_target_size(320, 240, 480) == (320, 240)

    def test_keeps_aspect_ratio(self):
        w, h = derive_target_size(1920, 1080, 480)
        assert w == 480
        assert h == 270
        assert abs(h / w - 1080 / 1920) < 0.01

    def test_none_disables_scaling(self):
        assert derive_target_size(1920, 1080, None) == (1920, 1080)

    def test_min_height_one(self):
        w, h = derive_target_size(1000, 1, 480)
        assert (w, h) == (480, 1)


class TestResolveTargetSize:
    def test_legacy_max_width_path_unchanged(self):
        # 旧尺寸参数行为完全保留（默认不做奇偶对齐）。
        assert resolve_target_size(1920, 1080, max_width=480) == (480, 270)
        assert resolve_target_size(320, 240, max_width=480) == (320, 240)
        assert resolve_target_size(1920, 1080, max_width=None) == (1920, 1080)

    def test_align_off_preserves_even_dims(self):
        w, h = resolve_target_size(1920, 1080, max_width=480, align_odd=False)
        assert (w, h) == (480, 270)  # 偶数原样保留

    def test_explicit_width_derives_height(self):
        assert resolve_target_size(320, 240, width=160) == (160, 120)

    def test_explicit_height_derives_width(self):
        assert resolve_target_size(320, 240, height=120) == (160, 120)

    def test_explicit_dim_overrides_max_width(self):
        # 给出显式维度后 max_width 失效。
        w, h = resolve_target_size(
            1920, 1080, max_width=480, width=100
        )
        assert (w, h) == (100, 56)  # round(1080*100/1920)=56

    def test_explicit_both_consistent_accepted(self):
        assert resolve_target_size(320, 240, width=160, height=120) == (
            160, 120
        )
        # ±1px 舍入容差内视为等比。
        assert resolve_target_size(320, 240, width=161, height=121) == (
            161, 121
        )

    def test_explicit_both_inconsistent_raises_no_stretch(self):
        # 比例不符直接报错，绝不静默拉伸画面。
        with pytest.raises(PipelineError, match="宽高比"):
            resolve_target_size(320, 240, width=160, height=200)

    def test_rejects_nonpositive_explicit_dims(self):
        with pytest.raises(PipelineError):
            resolve_target_size(320, 240, width=0)
        with pytest.raises(PipelineError):
            resolve_target_size(320, 240, height=-2)

    def test_upscale_allowed(self):
        assert resolve_target_size(64, 48, width=128) == (128, 96)

    def test_align_odd_rounds_even_dims_down(self):
        # 偶数统一向下取奇：480x270 → 479x269。
        w, h = resolve_target_size(
            1920, 1080, max_width=480, align_odd=True
        )
        assert (w, h) == (479, 269)
        assert w % 2 == 1 and h % 2 == 1

    def test_align_odd_keeps_already_odd_dims(self):
        w, h = resolve_target_size(320, 240, width=161, align_odd=True)
        assert (w, h) == (161, 121)  # 推导结果本身即奇数

    def test_align_odd_clamps_height_to_one(self):
        w, h = resolve_target_size(1000, 1, max_width=480, align_odd=True)
        assert (w, h) == (479, 1)

    def test_aspect_ratio_priority_over_alignment(self):
        # 宽高比优先：先等比取整再 ±1px 对齐，比例偏差有界且不拉伸。
        w, h = resolve_target_size(
            1920, 1080, max_width=480, align_odd=True
        )
        assert w % 2 == 1 and h % 2 == 1
        assert abs(h / w - 1080 / 1920) < 0.01

    def test_resolve_matches_derive_when_align_off(self):
        for args in [
            (1920, 1080, 480), (320, 240, 480), (1000, 1, 480),
            (720, 480, None),
        ]:
            assert resolve_target_size(*args, align_odd=False) == (
                derive_target_size(*args)
            )


class TestStageOrdering:
    """用 monkeypatch 绕开 ffmpeg，验证缩放严格先于量化、调色板基于
    最终尺寸（不依赖 ffmpeg 的纯管线顺序测试）。"""

    def _prepare_workdir(self, tmp_path, src_w=8, src_h=6, n=4, fps=4.0):
        work = tmp_path / "work"
        work.mkdir()
        manifest = make_manifest(n=n, fps=fps, w=src_w, h=src_h)
        for i in range(n):
            (work / f"frame_{i + 1:06d}.rgb").write_bytes(
                bytes((i * 31 + j) % 256 for j in range(src_w * src_h * 3))
            )
        return work, manifest

    def test_scale_before_quantize_and_palette_uses_final_size(
        self, tmp_path, monkeypatch
    ):
        import vgif.pipeline as pl

        work, manifest = self._prepare_workdir(tmp_path)
        monkeypatch.setattr(
            pl, "_load_or_extract", lambda *a, **k: (manifest, False)
        )
        monkeypatch.setattr(
            pl, "write_gif", lambda *a, **k: (tmp_path / "o.gif").write_bytes(b"GIF")
        )

        log = []
        real_resize = scale_mod.resize_rgb
        real_quantize = pl.quantize_frames

        def spy_resize(frame, sw, sh, dw, dh):
            log.append(("scale", dw, dh))
            return real_resize(frame, sw, sh, dw, dh)

        def spy_quantize(frames, width, height, colors):
            log.append(("quantize", width, height))
            # 量化入参帧必须已是最终尺寸。
            assert all(len(f) == width * height * 3 for f in frames)
            return real_quantize(frames, width, height, colors)

        monkeypatch.setattr(pl, "resize_rgb", spy_resize)
        monkeypatch.setattr(pl, "quantize_frames", spy_quantize)

        result = convert_to_gif(
            "in.mp4", tmp_path / "o.gif",
            target_fps=2, max_width=None, width=4,
            work_dir=work,
        )
        stages = [name for name, _, _ in log]
        assert stages.index("scale") < stages.index("quantize")
        # 两个阶段几何一致，均为最终尺寸 4x3。
        assert all(dw == 4 and dh == 3 for _, dw, dh in log)
        assert (result.width, result.height) == (4, 3)

    def test_align_odd_flows_through_pipeline(self, tmp_path, monkeypatch):
        import vgif.pipeline as pl

        work, manifest = self._prepare_workdir(
            tmp_path, src_w=10, src_h=8, n=4, fps=4.0
        )
        monkeypatch.setattr(
            pl, "_load_or_extract", lambda *a, **k: (manifest, False)
        )
        monkeypatch.setattr(
            pl, "write_gif", lambda *a, **k: (tmp_path / "o.gif").write_bytes(b"GIF")
        )
        result = convert_to_gif(
            "in.mp4", tmp_path / "o.gif",
            target_fps=2, max_width=6, align_odd=True, work_dir=work,
        )
        # 等比 6x5（偶数）→ 奇数对齐 5x5。
        assert (result.width, result.height) == (5, 5)


def test_defaults_documented():
    # 锁定默认参数，改动须同步 README 与 docstring 的折中依据。
    assert DEFAULT_FPS == 12
    assert DEFAULT_MAX_WIDTH == 480
    assert DEFAULT_COLORS == 256


def test_manifest_duration_includes_last_frame():
    m = make_manifest(n=10, fps=10)
    # 末帧 PTS 0.9 + 持续 0.1 = 1.0，而非 0.9。
    assert m.duration == pytest.approx(1.0)
