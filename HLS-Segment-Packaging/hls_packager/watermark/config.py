"""文字水印的不可变配置模型与参数校验。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import WatermarkError

VALID_ANCHORS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)


@dataclass(frozen=True)
class WatermarkConfig:
    """一次文字水印的全部输入参数。"""

    text: str | bytes
    font_files: tuple[str, ...] = field(default_factory=tuple)
    font_size: int = 28
    color: str = "white"
    opacity: float = 1.0
    anchor: str = "bottom-right"
    margin_x: int = 24
    margin_y: int = 24
    padding: int = 6

    def validate(self) -> None:
        if isinstance(self.text, str):
            if not self.text:
                raise WatermarkError("水印文本不能为空")
        elif isinstance(self.text, (bytes, bytearray)):
            if not self.text:
                raise WatermarkError("水印文本不能为空")
        else:
            raise WatermarkError("水印文本必须是 str 或 UTF-8 bytes")

        if not isinstance(self.font_size, int) or self.font_size <= 0:
            raise WatermarkError(f"字号必须为正整数像素，实际为 {self.font_size!r}")
        if self.font_size > 4096:
            raise WatermarkError("字号过大（上限 4096 像素）")

        if not 0.0 <= self.opacity <= 1.0:
            raise WatermarkError(f"不透明度必须在 0~1 之间，实际为 {self.opacity}")

        if self.anchor not in VALID_ANCHORS:
            raise WatermarkError(
                f"锚点 {self.anchor!r} 非法，可选：{', '.join(VALID_ANCHORS)}"
            )

        for name, value in (
            ("margin_x", self.margin_x),
            ("margin_y", self.margin_y),
            ("padding", self.padding),
        ):
            if not isinstance(value, int) or value < 0:
                raise WatermarkError(f"{name} 必须为非负整数，实际为 {value!r}")

        for font_path in self.font_files:
            if not isinstance(font_path, str) or not font_path:
                raise WatermarkError("字体路径必须是非空字符串")
