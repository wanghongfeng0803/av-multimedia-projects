"""三步管线编排：抽帧 → 采样/缩放/量化 → GIF 编码。

三个阶段通过工作目录（默认系统临时目录下的 ``vgif_*``）解耦，
``manifest.json`` 与 ``frame_*.rgb`` 落盘后，改目标帧率 / 尺寸 / 色深
重跑时可直接复用抽帧产物，不再触碰源视频；量化产物（索引帧）也可在
索引域缩放 / 重采样复用。显式传入 ``work_dir`` 即可在多次调用间持久
复用，这正是「抽帧/量化结果可被后续缩放与采样复用」的落地方式。

缩放阶段的三条硬约束（详见 :mod:`vgif.scale`）：

1. **阶段顺序固定**：缩放严格接在抽帧/采样之后、量化之前，调色板
   永远基于最终尺寸生成；量化后只允许索引域最近邻（``indexed_scale``
   像素风路径），不存在「先量化再 RGB 缩放」。
2. **宽高比与奇数对齐冲突时**：宽高比优先到 1px，奇数对齐其次——
   等比推导后再把宽、高各自规整到最近奇数（偶数向下取奇），接受
   ≤1px 的比例偏差，绝不拉伸画面去凑奇数（``align_odd`` 显式开启
   才生效，默认关闭以保持旧尺寸行为）。
3. **抗锯齿**：缩小走面积平均（box 低通，全源像素加权），放大走
   双线性，由 :func:`vgif.scale.resize_rgb` 逐轴自动选路。

默认参数（体积与画质的折中，依据写在各常量 docstring 处）：

- :data:`DEFAULT_FPS` = 12：100/12 经厘秒均摊为 8/9 厘秒交替，节奏
  均匀且单帧延迟在浏览器安全区；相对常见源帧率（24/25/30）为整数级
  丢帧，取帧节奏规则，体积约为 15fps 方案的八成、25fps 直转的一半。
- :data:`DEFAULT_MAX_WIDTH` = 480：GIF 仅 256 色且 LZW 压缩率有限，
  480px 宽是社媒贴图的常用上限，再宽体积近似按像素平方增长而主观
  画质受 256 色上限制约，收益递减。
- :data:`DEFAULT_COLORS` = 256：用满全局颜色表换取最好量化质量；
  需要压体积时降到 128/64，体积与色阶可显式权衡。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .extract import ExtractError, Manifest, extract_rgb_frames
from .gif_writer import GifEncodeError, write_gif
from .quantize import (
    MAX_GIF_COLORS,
    QuantizedFrame,
    quantize_frames,
)
from .sampling import sample_frames
from .scale import (
    nearest_resize_indexed,
    resize_rgb,
    round_odd,
)
from .timing import TimingError, plan_gif_delays

__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_COLORS",
    "PipelineError",
    "ConversionResult",
    "convert_to_gif",
    "derive_output_frame_count",
    "derive_target_size",
    "resolve_target_size",
]

#: 默认目标帧率（每秒帧数）。详见模块 docstring 的折中依据。
DEFAULT_FPS = 12
#: 默认最大宽度（像素），高度按比例推导。
DEFAULT_MAX_WIDTH = 480
#: 默认调色板色数（用满 GIF 全局颜色表）。
DEFAULT_COLORS = MAX_GIF_COLORS


class PipelineError(RuntimeError):
    """管线阶段失败（包装下层 ExtractError/TimingError 等）。"""


@dataclass(frozen=True)
class ConversionResult:
    """一次转换的可观测结果，供 CLI 与测试断言。

    :ivar output: 输出 GIF 路径。
    :ivar manifest: 抽帧清单（源几何/帧率/时间戳）。
    :ivar frame_count_in: 源帧数。
    :ivar frame_count_out: GIF 帧数。
    :ivar width: GIF 宽。
    :ivar height: GIF 高。
    :ivar colors: 实际调色板条目数（可能少于请求值）。
    :ivar delays_cs: 逐帧厘秒延迟。
    :ivar source_duration: 源时长（秒）。
    :ivar gif_duration: GIF 播放时长（秒）。
    :ivar reused_extract: 是否复用了工作目录里的抽帧产物。
    """

    output: Path
    manifest: Manifest
    frame_count_in: int
    frame_count_out: int
    width: int
    height: int
    colors: int
    delays_cs: tuple
    source_duration: float
    gif_duration: float
    reused_extract: bool

    @property
    def duration_error(self) -> float:
        """GIF 与源的时长偏差（秒），绝对值恒 < 0.01。"""
        return self.gif_duration - self.source_duration


def derive_output_frame_count(manifest: Manifest,
                              target_fps: float) -> int:
    """按目标帧率推导输出帧数（半值向上取整，且不超过源帧数）。

    时长口径与 frconv 一致：``duration = 末帧 PTS + 末帧持续时间``，
    不能只用末帧 PTS，否则平白少算一帧。
    """
    if target_fps <= 0:
        raise PipelineError(f"目标帧率必须为正：{target_fps}")
    duration = manifest.duration
    raw = int(target_fps * duration + 0.5)
    n = max(1, raw)
    return min(n, manifest.frame_count)


def derive_target_size(src_w: int, src_h: int,
                       max_width: int | None) -> tuple:
    """按最大宽度等比推导目标尺寸（**不做奇偶强制**，GIF 无此约束）。

    源宽不超过 ``max_width`` 时保持原尺寸；否则高度按比例四舍五入，
    至少为 1。``max_width=None`` 表示不缩放。

    历史行为保持不变：奇数对齐是可选项，由
    :func:`resolve_target_size` 在本函数结果上再施加。
    """
    if max_width is None or src_w <= max_width:
        return src_w, src_h
    dst_w = max_width
    dst_h = max(1, int(src_h * max_width / src_w + 0.5))
    return dst_w, dst_h


def resolve_target_size(src_w: int, src_h: int,
                        max_width: int | None = DEFAULT_MAX_WIDTH,
                        width: int | None = None,
                        height: int | None = None,
                        align_odd: bool = False) -> tuple:
    """解析全部尺寸参数，返回最终 ``(dst_w, dst_h)``（正整数）。

    参数优先级与取舍规则：

    1. **显式 ``width`` / ``height`` 优先于 ``max_width``**：给出任一
       显式维度即进入显式模式，``max_width`` 被忽略；只给一个维度时，
       另一个按源宽高比推导，不允许把两个维度都给成与源比例不同的
       值——显式模式不做拉伸，同时给出宽高且比例与源不一致时直接
       :class:`PipelineError`，要求调用方在「裁剪」与「变形」之间自行
       抉择，本工具只做等比缩放。
    2. **宽高比与奇数对齐冲突时，宽高比优先到 1px、对齐其次**：先按
       源比例把两个维度都算成整数（比例误差只来自四舍五入），再在
       ``align_odd=True`` 时把宽、高**各自**规整到最近奇数
       （:func:`vgif.scale.round_odd`，偶数向下取奇）。规整只改 ±1px，
       不回头重算另一维，因此比例偏差上界为 1px 量级且画面绝不拉伸；
       ``align_odd=False``（默认）完全保留旧行为，不做任何奇偶改动。
    3. ``max_width`` 模式（无显式维度）维持 :func:`derive_target_size`
       的旧语义；``max_width=None`` 且无显式维度时不缩放。

    :raises PipelineError: 维度非正，或显式宽高与源宽高比不一致。
    """
    if width is None and height is None:
        dst_w, dst_h = derive_target_size(src_w, src_h, max_width)
    else:
        for name, value in (("width", width), ("height", height)):
            if value is not None and (
                not isinstance(value, int) or value <= 0
            ):
                raise PipelineError(f"{name} 必须为正整数：{value!r}")
        if width is not None and height is not None:
            # 显式给出两维：只接受与源比例一致（四舍五入容差 1px），
            # 比例不符即报错，不静默拉伸画面。
            expect_h = max(1, int(src_h * width / src_w + 0.5))
            expect_w = max(1, int(src_w * height / src_h + 0.5))
            if abs(expect_h - height) > 1 or abs(expect_w - width) > 1:
                raise PipelineError(
                    f"显式尺寸 {width}x{height} 与源宽高比 "
                    f"{src_w}:{src_h} 不一致（等比应为约 "
                    f"{width}x{expect_h} 或 {expect_w}x{height}）；"
                    "本工具只做等比缩放，请只给一个维度或改用等比尺寸"
                )
            dst_w, dst_h = width, height
        elif width is not None:
            dst_w = width
            dst_h = max(1, int(src_h * width / src_w + 0.5))
        else:
            dst_h = height  # type: ignore[assignment]
            dst_w = max(1, int(src_w * height / src_h + 0.5))  # type: ignore[operator]

    if align_odd:
        dst_w = round_odd(dst_w)
        dst_h = round_odd(dst_h)
    return dst_w, dst_h


def _load_or_extract(input_path: Path, work_dir: Path,
                     reuse: bool) -> tuple:
    """阶段一：命中工作目录清单则复用，否则重新抽帧。"""
    manifest_path = work_dir / "manifest.json"
    if reuse and manifest_path.is_file():
        import json
        try:
            manifest = Manifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            # 校验帧文件齐全才认定复用有效，否则降级重抽。
            ok = all(
                manifest.frame_path(work_dir, i).is_file()
                for i in range(manifest.frame_count)
            )
            if ok:
                return manifest, True
        except (KeyError, ValueError, OSError):
            pass
    manifest = extract_rgb_frames(input_path, work_dir)
    return manifest, False


def convert_to_gif(input_path: str | Path, output_path: str | Path,
                   target_fps: float = DEFAULT_FPS,
                   max_width: int | None = DEFAULT_MAX_WIDTH,
                   colors: int = DEFAULT_COLORS,
                   work_dir: str | Path | None = None,
                   reuse: bool = True,
                   indexed_scale: bool = False,
                   width: int | None = None,
                   height: int | None = None,
                   align_odd: bool = False) -> ConversionResult:
    """执行完整的视频 → GIF 三步转换。

    阶段顺序固定为**抽帧 → 采样 → 缩放 → 量化 → 编码**，缩放始终在
    量化之前，调色板基于最终尺寸生成。

    :param target_fps: 目标帧率，默认 :data:`DEFAULT_FPS`；均摊厘秒
        延迟低于 2（即 >50fps）时抛 :class:`TimingError`，不静默漂移。
    :param max_width: 缩放后最大宽度，None 表示保持源宽；默认 480。
        被显式 ``width``/``height`` 任一参数覆盖时忽略。旧参数保持
        可用，语义与历史版本完全一致（``align_odd=False`` 时）。
    :param colors: 调色板色数（2..256），实际条目可能更少。
    :param work_dir: 抽帧工作目录；None 时创建临时目录（随用随弃），
        显式给出时可跨调用复用抽帧/量化产物。
    :param reuse: 是否允许复用 ``work_dir`` 中已有的抽帧产物。
    :param indexed_scale: 为 True 时在索引域最近邻缩放（不重新量化，
        适合像素风素材）；默认 False 走 RGB 域缩放（缩小面积平均
        抗锯齿、放大双线性）后重新量化。
    :param width: 显式目标宽（像素，正整数）；给出后 ``max_width``
        失效，高度按源宽高比推导；与 ``height`` 同时给出且比例不符
        时报错，不拉伸画面。
    :param height: 显式目标高（像素，正整数），规则同 ``width``。
    :param align_odd: True 时把最终宽、高各自规整到最近奇数（偶数
        向下取奇）。宽高比与奇数对齐冲突时宽高比优先：先等比取整再
        ±1px 对齐，不拉伸；默认 False，保持旧尺寸行为。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not 1 <= colors <= MAX_GIF_COLORS:
        raise PipelineError(
            f"色数须在 1..{MAX_GIF_COLORS}：{colors}"
        )

    owns_workdir = work_dir is None
    if owns_workdir:
        temp_ctx = tempfile.TemporaryDirectory(prefix="vgif_")
        work_dir = Path(temp_ctx.name)
    else:
        temp_ctx = None
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- 阶段一：抽帧（与帧率/调色板无关，可持久复用）----
        try:
            manifest, reused = _load_or_extract(input_path, work_dir, reuse)
        except ExtractError as exc:
            raise PipelineError(f"抽帧失败：{exc}") from exc

        n_src = manifest.frame_count
        n_out = derive_output_frame_count(manifest, target_fps)
        dst_w, dst_h = resolve_target_size(
            manifest.width, manifest.height,
            max_width=max_width, width=width, height=height,
            align_odd=align_odd,
        )

        # ---- 时序规划：只依赖源时长与输出帧数 ----
        try:
            plan = plan_gif_delays(manifest.duration, n_out, target_fps)
        except TimingError:
            raise

        # ---- 阶段二：采样 → 缩放 → 量化（纯 Python，顺序固定）----
        src_frames = [
            manifest.frame_path(work_dir, i).read_bytes()
            for i in range(n_src)
        ]
        picked, _ = sample_frames(src_frames, n_out)

        if indexed_scale:
            # 像素风复用路径：先在源尺寸量化，再索引域最近邻缩放，
            # 不重新量化、零颜色损失；主路径（下面分支）才是
            # 「缩放严格在量化前」的 RGB 流程。
            palette, qframes = quantize_frames(
                picked, manifest.width, manifest.height, colors
            )
            qframes = [
                QuantizedFrame(
                    indices=nearest_resize_indexed(
                        q.indices, q.width, q.height, dst_w, dst_h
                    ),
                    width=dst_w,
                    height=dst_h,
                )
                for q in qframes
            ]
        else:
            # 缩放必须在量化前：先把 RGB 帧放到最终尺寸，再让中位
            # 切分基于正确尺寸的像素生成调色板，否则调色板失真。
            scaled = [
                resize_rgb(
                    f, manifest.width, manifest.height, dst_w, dst_h
                )
                for f in picked
            ]
            palette, qframes = quantize_frames(scaled, dst_w, dst_h, colors)

        # ---- 阶段三：GIF 编码（延迟原样写入）----
        try:
            write_gif(
                output_path, qframes, palette,
                plan.delays_cs, dst_w, dst_h,
            )
        except GifEncodeError as exc:
            raise PipelineError(f"GIF 编码失败：{exc}") from exc

        return ConversionResult(
            output=output_path,
            manifest=manifest,
            frame_count_in=n_src,
            frame_count_out=n_out,
            width=dst_w,
            height=dst_h,
            colors=len(palette),
            delays_cs=plan.delays_cs,
            source_duration=manifest.duration,
            gif_duration=plan.gif_duration,
            reused_extract=reused,
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
