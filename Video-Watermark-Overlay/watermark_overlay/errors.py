"""统一的异常类型与中文错误提示。

水印叠加过程中所有可预期的错误（非法参数、解码失败、尺寸越界等）
都抛出 :class:`WatermarkError` 或其子类，便于 CLI 统一捕获展示。
"""


class WatermarkError(Exception):
    """水印工具的基础异常。"""


class InvalidParameterError(WatermarkError):
    """配置参数未通过校验时抛出（透明度越界、尺寸为负、锚点未知等）。"""


class ImageDecodeError(WatermarkError):
    """PNG 解码失败：文件损坏、格式不支持或非 PNG 数据。"""
