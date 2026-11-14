"""水印叠加的可配置参数与校验。

所有位置/尺寸/透明度参数在进入核心合成逻辑前统一经过 :meth:`WatermarkConfig.validate`
校验，非法值抛出携带中文说明的 :class:`InvalidParameterError`，
保证核心算法内部无需再防御非法输入。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidParameterError
from .geometry import ANCHOR_CENTER, ANCHORS


@dataclass
class WatermarkConfig:
    """一次水印叠加的完整配置。

    尺寸：``target_width`` / ``target_height`` 至少给一个；只给一个时
    按水印原始宽高比等比推算。均为 ``None`` 时使用原始尺寸。
    """

    # 九宫格锚点，见 geometry.ANCHORS
    anchor: str = ANCHOR_CENTER
    # 锚点定位后的像素级偏移（x 向右、y 向下，允许负值）
    offset_x: int = 0
    offset_y: int = 0
    # 目标尺寸（像素），None 表示不约束该维度
    target_width: int | None = None
    target_height: int | None = None
    # 全局不透明度，0.0 全透明 ~ 1.0 完全不透明
    opacity: float = 1.0

    # 自适应蒙版开关与参数
    adaptive: bool = False
    # 关键区域最大压暗比例：0.0 不处理 ~ 1.0 最强处完全透明
    mask_strength: float = 0.6
    # 细节强度进入/达到满压暗的阈值（0..255）
    detail_low: int = 24
    detail_high: int = 96
    # 局部统计窗口半径（像素）
    mask_radius: int = 2

    def validate(self) -> None:
        """逐项校验配置，失败即抛出 :class:`InvalidParameterError`。"""
        if self.anchor not in ANCHORS:
            raise InvalidParameterError(
                f"未知锚点 {self.anchor!r}，可选值：{', '.join(sorted(ANCHORS))}"
            )

        for name, value in (("offset_x", self.offset_x), ("offset_y", self.offset_y)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidParameterError(f"{name} 必须为整数")

        for name, value in (
            ("target_width", self.target_width),
            ("target_height", self.target_height),
        ):
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise InvalidParameterError(f"{name} 必须为整数或 None")
                if value <= 0:
                    raise InvalidParameterError(f"{name} 必须为正整数，收到 {value}")

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise InvalidParameterError("opacity 必须为数值")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise InvalidParameterError(
                f"opacity 必须在 [0, 1] 区间内，收到 {self.opacity}"
            )

        if not isinstance(self.adaptive, bool):
            raise InvalidParameterError("adaptive 必须为布尔值")

        if not isinstance(self.mask_strength, (int, float)) or isinstance(
            self.mask_strength, bool
        ):
            raise InvalidParameterError("mask_strength 必须为数值")
        if not 0.0 <= float(self.mask_strength) <= 1.0:
            raise InvalidParameterError(
                f"mask_strength 必须在 [0, 1] 区间内，收到 {self.mask_strength}"
            )

        for name, value in (
            ("detail_low", self.detail_low),
            ("detail_high", self.detail_high),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidParameterError(f"{name} 必须为整数")
            if not 0 <= value <= 255:
                raise InvalidParameterError(f"{name} 必须在 [0, 255] 区间内，收到 {value}")
        if not isinstance(self.mask_radius, int) or isinstance(self.mask_radius, bool):
            raise InvalidParameterError("mask_radius 必须为整数")
        if self.mask_radius < 1:
            raise InvalidParameterError(
                f"mask_radius 必须为不小于 1 的正整数，收到 {self.mask_radius}"
            )
        if self.detail_low >= self.detail_high:
            raise InvalidParameterError(
                f"detail_low({self.detail_low}) 必须小于 detail_high({self.detail_high})"
            )
