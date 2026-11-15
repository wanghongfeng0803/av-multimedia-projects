"""两路视频左右并排合成的对齐规划（纯逻辑，不依赖 ffmpeg）。

设计目标：对分辨率 / 时长 / 帧率 / 音频不一致的输入给出**确定性**策略——
同样的输入与参数永远得到同样的规划，从而保证输出可复现，而不是直接失败。

确定性对齐策略总览
------------------

分辨率不一致
    画布高度取两路较大者（向上取偶数，可用 ``height`` 覆盖）；每路按显示
    宽高比（含 SAR 修正）等比缩放到该高度，宽度取偶数。``layout=auto``
    （默认）时每路单元格即缩放后的尺寸，输出宽 = 两路宽度之和；
    ``layout=fit`` 时较窄的一路居中加黑边填充到等宽单元格，输出宽 =
    2 × 单元格宽，画面左右严格对称。

时长不一致
    ``duration_mode=pad``（默认）：输出时长取较长者，较短的一路冻结最后
    一帧补齐（tpad clone），其音频（若有）补静音；
    ``duration_mode=trim``：输出时长取较短者，两路均截断。

帧率不一致
    输出帧率取两路较大者（可用 ``fps`` 覆盖），两路经 fps 滤镜统一转为
    该恒定帧率（CFR）。帧率以有理数精确表示（如 30000/1001），避免
    浮点误差积累导致的帧数不确定。

音频缺失 / 格式不一致
    两路都有音频：统一重采样为 48kHz 立体声后等权混音（amix 归一化，
    防止削波）；只有一路有：使用该路（重采样为 48kHz 立体声）；
    都没有：输出无音轨。无论哪种情况，音频都按目标时长补静音 / 截断，
    与视频严格等长。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .errors import InvalidInputError
from .models import MediaInfo

# 时长对齐模式
DURATION_PAD = "pad"
DURATION_TRIM = "trim"
DURATION_MODES = (DURATION_PAD, DURATION_TRIM)

# 布局模式
LAYOUT_AUTO = "auto"
LAYOUT_FIT = "fit"
LAYOUTS = (LAYOUT_AUTO, LAYOUT_FIT)

# 音频模式
AUDIO_AUTO = "auto"
AUDIO_MIX = "mix"
AUDIO_LEFT = "left"
AUDIO_RIGHT = "right"
AUDIO_NONE = "none"
AUDIO_MODES = (AUDIO_AUTO, AUDIO_MIX, AUDIO_LEFT, AUDIO_RIGHT, AUDIO_NONE)

#: 两路帧率都无法探测时的兜底帧率（确定性要求：必须有一个确定值）
DEFAULT_FALLBACK_FPS = Fraction(25, 1)
#: 音频统一重采样目标
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 2
#: 编码器可接受的最小偶数尺寸
MIN_DIMENSION = 2


def parse_fps(text: object) -> Fraction | None:
    """把 ffprobe 的帧率文本解析为精确有理数；无法解析或为 0 时返回 None。

    支持 ``"30000/1001"``、``"25"``、``"29.97"`` 三种形式。
    """
    if text is None:
        return None
    if isinstance(text, Fraction):
        return text if text > 0 else None
    if isinstance(text, (int, float)):
        if text <= 0:
            return None
        return Fraction(text).limit_denominator(1_000_000)
    s = str(text).strip()
    if not s:
        return None
    if "/" in s:
        num, _, den = s.partition("/")
        try:
            n, d = int(num), int(den)
        except ValueError:
            return None
        if n <= 0 or d <= 0:
            return None
        return Fraction(n, d)
    try:
        value = float(s)
    except ValueError:
        return None
    if value <= 0:
        return None
    return Fraction(value).limit_denominator(1_000_000)


def parse_sar(text: object) -> float:
    """解析采样宽高比 ``"4:3"``；缺失或非法时返回 1.0（方形像素）。"""
    if not text:
        return 1.0
    s = str(text).strip()
    if ":" in s:
        num, _, den = s.partition(":")
        try:
            n, d = int(num), int(den)
        except ValueError:
            return 1.0
        if n <= 0 or d <= 0:
            return 1.0
        return n / d
    try:
        value = float(s)
    except ValueError:
        return 1.0
    return value if value > 0 else 1.0


def even_ceil(value: float) -> int:
    """向上取偶数（不小于 MIN_DIMENSION），用于画布高度等「至少容纳」场景。"""
    import math

    v = math.ceil(value)
    if v % 2:
        v += 1
    return max(v, MIN_DIMENSION)


def round_even(value: float) -> int:
    """四舍五入到最近的偶数（不小于 MIN_DIMENSION），用于等比缩放后的宽度。"""
    v = int(round(value))
    if v % 2:
        v = v - 1 if v > MIN_DIMENSION else v + 1
    return max(v, MIN_DIMENSION)


def scaled_width_for_height(display_width: float, height: int, target_height: int) -> int:
    """按显示宽高比把一路视频等比缩放到目标高度后的宽度（偶数）。"""
    return round_even(display_width * target_height / height)


@dataclass
class VideoSidePlan:
    """单路视频的对齐结果。"""

    source_width: int
    source_height: int
    source_duration: float
    source_fps: Fraction | None
    has_audio: bool
    scaled_width: int
    scaled_height: int
    cell_width: int
    cell_height: int
    # 画面在单元格内的偏移（fit 布局下居中填充黑边）
    pad_x: int = 0
    pad_y: int = 0
    # 时长对齐需要冻结最后一帧的秒数（pad 模式下较短的一路 > 0）
    freeze_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_duration": self.source_duration,
            "source_fps": str(self.source_fps) if self.source_fps else None,
            "has_audio": self.has_audio,
            "scaled_width": self.scaled_width,
            "scaled_height": self.scaled_height,
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "freeze_seconds": self.freeze_seconds,
        }


@dataclass
class ComposePlan:
    """一次并排合成的完整确定性规划。"""

    width: int
    height: int
    fps: Fraction
    duration: float
    duration_mode: str
    layout: str
    audio_mode: str  # 解析后的实际模式：mix / left / right / none
    audio_sources: tuple[int, ...]  # 参与输出的输入序号（0=左，1=右）
    sample_rate: int
    channels: int
    left: VideoSidePlan
    right: VideoSidePlan
    warnings: list[str] = field(default_factory=list)

    @property
    def fps_text(self) -> str:
        """帧率的滤镜/显示文本，如 ``30000/1001`` 或 ``25``。"""
        return str(self.fps)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps_text,
            "duration": self.duration,
            "duration_mode": self.duration_mode,
            "layout": self.layout,
            "audio_mode": self.audio_mode,
            "audio_sources": list(self.audio_sources),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "warnings": self.warnings,
        }


def plan_composition(
    left: MediaInfo,
    right: MediaInfo,
    *,
    duration_mode: str = DURATION_PAD,
    layout: str = LAYOUT_AUTO,
    height: int | None = None,
    fps: object = None,
    audio_mode: str = AUDIO_AUTO,
) -> ComposePlan:
    """根据两路探测结果计算确定性的合成规划。

    :param duration_mode: ``pad``（补短到较长者，默认）或 ``trim``（截断到较短者）
    :param layout: ``auto``（单元格即缩放尺寸）或 ``fit``（等宽单元格 + 居中黑边）
    :param height: 画布高度覆盖；省略时取两路较大高度（向上取偶）
    :param fps: 输出帧率覆盖（``"30000/1001"`` / ``30`` / Fraction）；省略时取两路较大帧率
    :param audio_mode: ``auto`` / ``mix`` / ``left`` / ``right`` / ``none``
    :raises InvalidInputError: 缺视频流、尺寸或时长无法确定、参数非法
    """
    if duration_mode not in DURATION_MODES:
        raise InvalidInputError(
            f"未知的时长对齐模式: {duration_mode!r}（可选: {'/'.join(DURATION_MODES)}）"
        )
    if layout not in LAYOUTS:
        raise InvalidInputError(
            f"未知的布局模式: {layout!r}（可选: {'/'.join(LAYOUTS)}）"
        )
    if audio_mode not in AUDIO_MODES:
        raise InvalidInputError(
            f"未知的音频模式: {audio_mode!r}（可选: {'/'.join(AUDIO_MODES)}）"
        )

    warnings: list[str] = []
    warnings.extend(f"左路: {w}" for w in left.warnings)
    warnings.extend(f"右路: {w}" for w in right.warnings)

    left_video = _require_video(left, "左")
    right_video = _require_video(right, "右")

    # ---- 分辨率对齐 ----
    canvas_height = (
        even_ceil(height)
        if height is not None
        else even_ceil(max(left_video.height, right_video.height))  # type: ignore[arg-type]
    )
    left_scaled_w = scaled_width_for_height(
        left_video.display_width or left_video.width,  # type: ignore[arg-type]
        left_video.height,  # type: ignore[arg-type]
        canvas_height,
    )
    right_scaled_w = scaled_width_for_height(
        right_video.display_width or right_video.width,  # type: ignore[arg-type]
        right_video.height,  # type: ignore[arg-type]
        canvas_height,
    )

    if layout == LAYOUT_FIT:
        cell_w = even_ceil(max(left_scaled_w, right_scaled_w))
        left_cell_w = right_cell_w = cell_w
    else:
        left_cell_w, right_cell_w = left_scaled_w, right_scaled_w
    canvas_width = left_cell_w + right_cell_w

    # ---- 帧率对齐 ----
    left_fps = left_video.fps
    right_fps = right_video.fps
    if fps is not None:
        out_fps = parse_fps(fps)
        if out_fps is None:
            raise InvalidInputError(f"无法解析的输出帧率: {fps!r}")
    else:
        known = [f for f in (left_fps, right_fps) if f is not None]
        if known:
            out_fps = max(known)
        else:
            out_fps = DEFAULT_FALLBACK_FPS
            warnings.append(
                f"两路帧率均无法探测，回退到默认帧率 {out_fps}。"
            )

    # ---- 时长对齐 ----
    left_duration = _require_duration(left, "左")
    right_duration = _require_duration(right, "右")
    if duration_mode == DURATION_PAD:
        out_duration = max(left_duration, right_duration)
    else:
        out_duration = min(left_duration, right_duration)
    left_freeze = max(0.0, out_duration - left_duration)
    right_freeze = max(0.0, out_duration - right_duration)

    # ---- 音频对齐 ----
    left_has_audio = bool(left.audio_streams)
    right_has_audio = bool(right.audio_streams)
    resolved_audio_mode, audio_sources = _resolve_audio(
        audio_mode, left_has_audio, right_has_audio, warnings
    )

    left_plan = VideoSidePlan(
        source_width=left_video.width,  # type: ignore[arg-type]
        source_height=left_video.height,  # type: ignore[arg-type]
        source_duration=left_duration,
        source_fps=left_fps,
        has_audio=left_has_audio,
        scaled_width=left_scaled_w,
        scaled_height=canvas_height,
        cell_width=left_cell_w,
        cell_height=canvas_height,
        pad_x=(left_cell_w - left_scaled_w) // 2,
        pad_y=0,
        freeze_seconds=left_freeze,
    )
    right_plan = VideoSidePlan(
        source_width=right_video.width,  # type: ignore[arg-type]
        source_height=right_video.height,  # type: ignore[arg-type]
        source_duration=right_duration,
        source_fps=right_fps,
        has_audio=right_has_audio,
        scaled_width=right_scaled_w,
        scaled_height=canvas_height,
        cell_width=right_cell_w,
        cell_height=canvas_height,
        pad_x=(right_cell_w - right_scaled_w) // 2,
        pad_y=0,
        freeze_seconds=right_freeze,
    )

    return ComposePlan(
        width=canvas_width,
        height=canvas_height,
        fps=out_fps,
        duration=out_duration,
        duration_mode=duration_mode,
        layout=layout,
        audio_mode=resolved_audio_mode,
        audio_sources=audio_sources,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        left=left_plan,
        right=right_plan,
        warnings=warnings,
    )


def _require_video(info: MediaInfo, side: str):
    """取出主视频流并校验尺寸；缺视频流或尺寸未知时抛 InvalidInputError。"""
    video = info.primary_video
    if video is None:
        raise InvalidInputError(f"{side}路输入不含视频流: {info.file_path}")
    if not video.width or not video.height or video.width <= 0 or video.height <= 0:
        raise InvalidInputError(
            f"{side}路视频尺寸无法确定: {info.file_path}，无法规划画布。"
        )
    return video


def _require_duration(info: MediaInfo, side: str) -> float:
    """确定一路的有效时长：优先视频流时长，回退容器时长。

    时长是时长对齐（pad/trim）与帧数确定性的基础，无法确定时只能失败，
    并给出明确原因（此时任何「猜」都会破坏可复现性）。
    """
    video = info.primary_video
    duration = video.duration_seconds if video else None
    if duration is None or duration <= 0:
        duration = info.duration_seconds
    if duration is None or duration <= 0:
        raise InvalidInputError(
            f"{side}路输入时长无法确定: {info.file_path}。"
            "时长对齐需要明确的时长，请先修复容器元数据。"
        )
    return duration


def _resolve_audio(
    audio_mode: str,
    left_has: bool,
    right_has: bool,
    warnings: list[str],
) -> tuple[str, tuple[int, ...]]:
    """把请求的音频模式解析为实际模式与参与混音的输入序号。

    显式指定的模式被严格遵守（指定侧无音频则输出无声并记录警告），
    保证「同参数同结果」；只有 auto/mix 会按实际音轨情况降级。
    """
    if audio_mode == AUDIO_NONE:
        return AUDIO_NONE, ()
    if audio_mode == AUDIO_LEFT:
        if not left_has:
            warnings.append("已指定仅使用左路音频，但左路没有音频流，输出将无音轨。")
            return AUDIO_NONE, ()
        return AUDIO_LEFT, (0,)
    if audio_mode == AUDIO_RIGHT:
        if not right_has:
            warnings.append("已指定仅使用右路音频，但右路没有音频流，输出将无音轨。")
            return AUDIO_NONE, ()
        return AUDIO_RIGHT, (1,)
    if audio_mode == AUDIO_MIX:
        if left_has and right_has:
            return AUDIO_MIX, (0, 1)
        if left_has or right_has:
            present = "左" if left_has else "右"
            warnings.append(f"请求混音但只有{present}路有音频流，输出使用该路音频。")
            return (AUDIO_LEFT, (0,)) if left_has else (AUDIO_RIGHT, (1,))
        warnings.append("请求混音但两路都没有音频流，输出将无音轨。")
        return AUDIO_NONE, ()
    # auto
    if left_has and right_has:
        return AUDIO_MIX, (0, 1)
    if left_has:
        return AUDIO_LEFT, (0,)
    if right_has:
        return AUDIO_RIGHT, (1,)
    return AUDIO_NONE, ()
