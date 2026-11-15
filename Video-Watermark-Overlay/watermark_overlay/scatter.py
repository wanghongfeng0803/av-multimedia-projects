"""分散布局的抗裁剪水印：把同一水印按确定性网格 + 抖动散布到整帧。

抗裁剪思路：单点水印（如固定右下角）一旦被裁掉对应区域就彻底丢失；
分散布局把帧划分为 ``rows x cols`` 个单元格，每个单元格放置一份水印
实例——任意局部裁剪都会保留其余区域的实例，四周留白 ``margin``
进一步避免贴边实例被一刀切掉。

布局是**确定性**的：位置抖动与角度抖动都由 ``seed`` 驱动的
``random.Random`` 产生，同一个 seed 得到完全相同的实例位置与角度，
因此批量处理多帧时布局一致、结果可复现（测试可逐字节断言）。

本模块只负责「放在哪」，不碰像素：实例的旋转变换与混合分别复用
:mod:`watermark_overlay.transform` 与 :mod:`watermark_overlay.blend`。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .errors import InvalidParameterError

# scatter_from_dict 允许的字段名（一一对应 ScatterConfig，便于拒绝拼写错误）
_SCATTER_KEYS = {
    "rows",
    "cols",
    "margin",
    "jitter",
    "angle_jitter",
    "seed",
}


@dataclass
class ScatterConfig:
    """分散布局配置。

    - ``rows`` / ``cols``：网格行/列数，实例总数为 rows * cols；
    - ``margin``：四边留白（像素），布局区域向内收缩，防止贴边实例
      被整条裁掉；
    - ``jitter``：单元格内位置抖动幅度，占可动范围的比例 [0, 1]，
      0 表示严格居中，1 表示最大幅度（实例仍不越出本单元格）；
    - ``angle_jitter``：每个实例的随机旋转角幅度（度），[0, 180]，
      实际角度均匀落在 [-angle_jitter, +angle_jitter]；
    - ``seed``：布局随机种子，相同 seed 布局完全一致。
    """

    rows: int = 2
    cols: int = 3
    margin: int = 0
    jitter: float = 0.3
    angle_jitter: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        """逐项校验，失败抛出 :class:`InvalidParameterError`。"""
        for name, value in (("rows", self.rows), ("cols", self.cols)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidParameterError(f"{name} 必须为整数")
            if value < 1:
                raise InvalidParameterError(f"{name} 必须为正整数，收到 {value}")

        if not isinstance(self.margin, int) or isinstance(self.margin, bool):
            raise InvalidParameterError("margin 必须为整数")
        if self.margin < 0:
            raise InvalidParameterError(f"margin 必须非负，收到 {self.margin}")

        if not isinstance(self.jitter, (int, float)) or isinstance(self.jitter, bool):
            raise InvalidParameterError("jitter 必须为数值")
        if not 0.0 <= float(self.jitter) <= 1.0:
            raise InvalidParameterError(
                f"jitter 必须在 [0, 1] 区间内，收到 {self.jitter}"
            )

        if not isinstance(self.angle_jitter, (int, float)) or isinstance(
            self.angle_jitter, bool
        ):
            raise InvalidParameterError("angle_jitter 必须为数值")
        if not math.isfinite(float(self.angle_jitter)):
            raise InvalidParameterError(
                f"angle_jitter 必须为有限数，收到 {self.angle_jitter}"
            )
        if not 0.0 <= float(self.angle_jitter) <= 180.0:
            raise InvalidParameterError(
                f"angle_jitter 必须在 [0, 180] 区间内，收到 {self.angle_jitter}"
            )

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise InvalidParameterError("seed 必须为整数")


@dataclass
class ScatterPlacement:
    """一个水印实例的落点：帧坐标系中的整数原点 + 绕自身中心的旋转角。"""

    x: int
    y: int
    angle: float  # 度，落在 [-angle_jitter, +angle_jitter]


def compute_scatter_layout(
    frame_width: int,
    frame_height: int,
    watermark_width: int,
    watermark_height: int,
    config: ScatterConfig,
) -> list[ScatterPlacement]:
    """计算全部实例的位置与角度，返回长度为 rows * cols 的列表。

    每个实例以其单元格中心为基准，位置抖动不超过单元格内可动范围的
    一半，因此只要水印不大于单元格，实例就始终完整落在本单元格内
    （实例互不重叠）；水印大于单元格时按单元格中心居中，越界部分
    交给混合层逐像素裁剪。

    布局只依赖帧尺寸、水印尺寸与配置，与帧像素内容无关。
    """
    config.validate()

    rng = random.Random(config.seed)
    usable_w = frame_width - 2 * config.margin
    usable_h = frame_height - 2 * config.margin
    cell_w = usable_w / config.cols
    cell_h = usable_h / config.rows

    placements: list[ScatterPlacement] = []
    for row in range(config.rows):
        for col in range(config.cols):
            # 水印在单元格内居中；slack 为单元格内的可动范围（可为负）
            slack_x = cell_w - watermark_width
            slack_y = cell_h - watermark_height
            base_x = config.margin + col * cell_w + slack_x / 2.0
            base_y = config.margin + row * cell_h + slack_y / 2.0

            # 抖动幅度 ≤ 可动范围的一半：实例不会越出本单元格
            range_x = max(0.0, slack_x) / 2.0
            range_y = max(0.0, slack_y) / 2.0
            jx = (rng.random() * 2.0 - 1.0) * config.jitter * range_x
            jy = (rng.random() * 2.0 - 1.0) * config.jitter * range_y
            angle = (rng.random() * 2.0 - 1.0) * float(config.angle_jitter)
            angle = angle + 0.0 if angle else 0.0  # 折叠 -0.0，保持输出整洁

            placements.append(
                ScatterPlacement(
                    x=int(round(base_x + jx)),
                    y=int(round(base_y + jy)),
                    angle=angle,
                )
            )
    return placements


def scatter_from_dict(data: object) -> ScatterConfig:
    """从 JSON 风格字典构建 :class:`ScatterConfig` 并校验。

    期望格式：``{"rows": 2, "cols": 3, "margin": 0, "jitter": 0.3,
    "angle_jitter": 0.0, "seed": 0}``（全部可省略）；未知字段直接拒绝，
    非法值抛出 :class:`InvalidParameterError`。
    """
    if not isinstance(data, dict):
        raise InvalidParameterError("分散布局配置必须是 JSON 对象")
    unknown = sorted(set(data) - _SCATTER_KEYS)
    if unknown:
        raise InvalidParameterError(
            f"分散布局配置含未知字段：{', '.join(unknown)}"
            f"（可选：{', '.join(sorted(_SCATTER_KEYS))}）"
        )
    try:
        config = ScatterConfig(**data)
    except TypeError as exc:
        raise InvalidParameterError(f"分散布局配置字段非法：{exc}") from exc
    config.validate()
    return config
