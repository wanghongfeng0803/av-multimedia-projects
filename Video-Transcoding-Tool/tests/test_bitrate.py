"""码率控制模块的单元测试（纯逻辑，不启动 ffmpeg）。

覆盖：码率字符串解析的各种单位与非法输入、硬边界、三种速率模式的
参数推导、五族编码器的参数分派（含 SVT-AV1 禁 maxrate、x264 的
MP4/MKV CBR 差异等实测约束）。
"""

from __future__ import annotations

import pytest

from transcode_tool.bitrate import (
    MAX_BITRATE_BPS,
    MIN_BITRATE_BPS,
    MODE_ABR,
    MODE_CBR,
    MODE_VBR,
    build_bitrate_encoder_args,
    format_bitrate,
    parse_bitrate,
    plan_bitrate,
)
from transcode_tool.codecs import AV1, H264, H265, VP8, VP9


# ---------------------------------------------------------------------------
# parse_bitrate：单位、数值与边界
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("800k", 800_000),
        ("800K", 800_000),
        ("800kbps", 800_000),
        ("800 kbps", 800_000),
        ("2M", 2_000_000),
        ("2.5Mbps", 2_500_000),
        ("2.5 mb/s", 2_500_000),
        ("1G", 1_000_000_000),
        ("1gbps", 1_000_000_000),
        ("1500000", 1_500_000),
        ("1500000bps", 1_500_000),
        ("  300k  ", 300_000),
        ("1.5k", 1_500),
    ],
)
def test_parse_bitrate_units(text, expected):
    assert parse_bitrate(text) == expected


def test_parse_bitrate_numeric_inputs():
    assert parse_bitrate(1500) == 1500
    assert parse_bitrate(1500.0) == 1500
    # 小数结果四舍五入为整数 bps
    assert parse_bitrate("1.234k") == 1234


def test_parse_bitrate_uses_si_decimal_not_1024():
    # 码率单位遵循 ffmpeg 的十进制约定：1k = 1000，而非 1024
    assert parse_bitrate("1k") == 1000
    assert parse_bitrate("1m") == 1_000_000


@pytest.mark.parametrize("bad", [
    "", "   ", "abc", "800x", "100kk", "5t", "1.2.3m", "-800k", "0",
    "0k", "999",          # 低于 1kbps 下限
    "1001m",              # 超过 1Gbps 上限
])
def test_parse_bitrate_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_bitrate(bad)


def test_parse_bitrate_hard_bounds_inclusive():
    # 边界值本身合法
    assert parse_bitrate(MIN_BITRATE_BPS) == MIN_BITRATE_BPS
    assert parse_bitrate(MAX_BITRATE_BPS) == MAX_BITRATE_BPS
    # 紧贴边界外侧非法
    with pytest.raises(ValueError):
        parse_bitrate(MIN_BITRATE_BPS - 1)
    with pytest.raises(ValueError):
        parse_bitrate(MAX_BITRATE_BPS + 1)


def test_parse_bitrate_rejects_bool_and_nan_inf():
    with pytest.raises(ValueError):
        parse_bitrate(True)
    with pytest.raises(ValueError):
        parse_bitrate(float("nan"))
    with pytest.raises(ValueError):
        parse_bitrate(float("inf"))
    with pytest.raises(ValueError):
        parse_bitrate(-1.0)
    with pytest.raises(ValueError):
        parse_bitrate(None)  # type: ignore[arg-type]


def test_format_bitrate():
    assert format_bitrate(800_000) == "800kbps"
    assert format_bitrate(2_000_000) == "2Mbps"
    assert format_bitrate(1234) == "1234bps"
    with pytest.raises(ValueError):
        format_bitrate(True)


# ---------------------------------------------------------------------------
# plan_bitrate：三种模式的派生参数
# ---------------------------------------------------------------------------

def test_plan_vbr_doubles_peak_and_buffer():
    plan = plan_bitrate("1M", MODE_VBR)
    assert plan.target_bps == 1_000_000
    assert plan.max_bps == 2_000_000
    assert plan.buffer_bps == 2_000_000
    assert plan.min_bps == 0


def test_plan_abr_has_no_hard_vbv_constraints():
    plan = plan_bitrate("1M", MODE_ABR)
    assert plan.target_bps == 1_000_000
    assert plan.max_bps == 0
    assert plan.buffer_bps == 0
    assert plan.min_bps == 0


def test_plan_cbr_pins_min_max_buffer_to_target():
    plan = plan_bitrate("800k", MODE_CBR)
    assert plan.min_bps == plan.max_bps == plan.buffer_bps == 800_000


def test_plan_vbr_caps_peak_at_hard_maximum():
    # 目标 600M，2 倍峰值会超过 1G 上限，必须收敛
    plan = plan_bitrate("600M", MODE_VBR)
    assert plan.max_bps == MAX_BITRATE_BPS
    assert plan.buffer_bps == MAX_BITRATE_BPS


def test_plan_default_mode_is_vbr():
    assert plan_bitrate("1M").mode == MODE_VBR


def test_plan_rejects_unknown_mode():
    with pytest.raises(ValueError):
        plan_bitrate("1M", "cqp")


# ---------------------------------------------------------------------------
# build_bitrate_encoder_args：编码器分派
# ---------------------------------------------------------------------------

def _args(codec, encoder_name, plan, preset="medium", container="mp4"):
    enc = codec.resolve_encoder(encoder_name)
    return build_bitrate_encoder_args(
        codec, enc, plan, preset, container_canonical=container
    )


def test_x264_vbr_uses_generic_vbv_and_no_crf():
    plan = plan_bitrate("1M", MODE_VBR)
    args = _args(H264, "libx264", plan, preset="slow")
    assert args[:2] == ["-preset", "slow"]
    assert "-crf" not in args
    assert args[args.index("-b:v") + 1] == "1000000"
    assert args[args.index("-maxrate") + 1] == "2000000"
    assert args[args.index("-bufsize") + 1] == "2000000"
    assert "-minrate" not in args  # VBR 不设下限


def test_x264_abr_only_sets_target():
    plan = plan_bitrate("1M", MODE_ABR)
    args = _args(H264, "libx264", plan)
    assert "-maxrate" not in args and "-bufsize" not in args
    assert "-minrate" not in args
    assert args[args.index("-b:v") + 1] == "1000000"


def test_x264_cbr_in_mp4_omits_nal_hrd():
    # MP4 容器禁止 nal-hrd=cbr，只能给紧凑 VBV 的准 CBR
    plan = plan_bitrate("800k", MODE_CBR)
    args = _args(H264, "libx264", plan, container="mp4")
    assert "-x264-params" not in args
    assert args[args.index("-minrate") + 1] == "800000"
    assert args[args.index("-maxrate") + 1] == "800000"
    assert args[args.index("-bufsize") + 1] == "800000"


def test_x264_cbr_in_mkv_enables_nal_hrd():
    plan = plan_bitrate("800k", MODE_CBR)
    args = _args(H264, "libx264", plan, container="mkv")
    assert "-x264-params" in args
    params = args[args.index("-x264-params") + 1]
    assert "nal-hrd=cbr" in params and "force-cfr=1" in params


def test_x265_cbr_uses_generic_options_and_keeps_hvc1_tag():
    plan = plan_bitrate("800k", MODE_CBR)
    args = _args(H265, "libx265", plan)
    assert args[args.index("-b:v") + 1] == "800000"
    assert args[args.index("-minrate") + 1] == "800000"
    # 注册表上的 hvc1 标签等额外参数必须保留
    assert "-tag:v" in args and "hvc1" in args
    assert "-crf" not in args


def test_vp9_bitrate_mode_drops_constant_quality_marker():
    plan = plan_bitrate("500k", MODE_VBR)
    args = _args(VP9, "libvpx-vp9", plan, preset="fast")
    # 码率模式绝不能再出现常量质量的 -b:v 0
    assert not (args.count("-b:v") and args[args.index("-b:v") + 1] == "0")
    assert args[args.index("-b:v") + 1] == "500000"
    assert args[args.index("-cpu-used") + 1] == "5"  # fast → cpu-used 5
    assert args[args.index("-deadline") + 1] == "good"
    assert "-row-mt" in args
    assert args[args.index("-maxrate") + 1] == "1000000"


def test_vp8_abr_only_target():
    plan = plan_bitrate("400k", MODE_ABR)
    args = _args(VP8, "libvpx", plan)
    assert "-maxrate" not in args and "-bufsize" not in args
    assert args[args.index("-b:v") + 1] == "400000"


def test_svtav1_must_not_emit_maxrate_in_vbr_or_abr():
    # 实测：SVT-AV1 在 ABR 模式收到 -maxrate 会直接报错
    for mode in (MODE_VBR, MODE_ABR):
        plan = plan_bitrate("600k", mode)
        args = _args(AV1, "libsvtav1", plan, preset="8")
        assert "-maxrate" not in args
        assert "-bufsize" not in args
        assert "-minrate" not in args
        assert args[args.index("-preset") + 1] == "8"
        assert args[args.index("-b:v") + 1] == "600000"


def test_svtav1_cbr_uses_private_rc_flag():
    plan = plan_bitrate("600k", MODE_CBR)
    args = _args(AV1, "libsvtav1", plan)
    assert "-svtav1-params" in args
    assert args[args.index("-svtav1-params") + 1] == "rc=1"
    # 即便 CBR 也不向 SVT 传通用 -maxrate
    assert "-maxrate" not in args


def test_rav1e_uses_speed_and_generic_rate():
    plan = plan_bitrate("300k", MODE_CBR)
    args = _args(AV1, "librav1e", plan, preset="fast")
    assert args[args.index("-speed") + 1] == "6"  # fast → speed 6
    assert args[args.index("-b:v") + 1] == "300000"
    assert args[args.index("-maxrate") + 1] == "300000"
    assert "-qp" not in args  # 码率模式不走 QP


def test_libaom_uses_cpu_used_and_generic_rate():
    plan = plan_bitrate("300k", MODE_VBR)
    args = _args(AV1, "libaom-av1", plan, preset="6")
    assert args[args.index("-cpu-used") + 1] == "6"
    assert args[args.index("-b:v") + 1] == "300000"
    assert "-maxrate" in args and "-row-mt" in args


def test_build_rejects_unknown_mode():
    plan = plan_bitrate("1M")
    object.__setattr__(plan, "mode", "bogus")  # frozen dataclass 绕过
    enc = H264.encoder_chain[0]
    with pytest.raises(ValueError):
        build_bitrate_encoder_args(H264, enc, plan, "medium")
