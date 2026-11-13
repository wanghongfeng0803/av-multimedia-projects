"""统一的异常类型与错误提示。

解析或重封装过程中出现的所有可预期错误都抛出 :class:`RemuxError`，
消息使用中文，便于 CLI 直接展示给用户。
"""


class RemuxError(Exception):
    """输入文件无效或包含基础版本无法处理的结构时抛出。"""
