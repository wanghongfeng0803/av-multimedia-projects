"""EXT-X-KEY 描述模型与属性行渲染（RFC 8216 §4.3.2.4）。

标签形态::

    #EXT-X-KEY:METHOD=AES-128,URI="encryption.key",IV=0x1a2b...

合规要点：

* ``METHOD`` 必须是本工具唯一支持的 ``AES-128``（未加密时根本不输出
  该标签，因此不需要 ``METHOD=NONE``）；
* ``URI`` 是带引号的 quoted-string，属性值中的双引号已在
  :class:`EncryptionConfig` 构造期拒绝，渲染层只负责套引号；
* ``IV`` 为可选 128 位十六进制（``0x`` 前缀）；采用 RFC §5.2 的
  隐式序号 IV 时不输出该属性，播放器按 MEDIA-SEQUENCE 自行派生；
* 属性顺序固定为 METHOD -> URI -> IV，与 RFC 示例及主流播放器一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import EncryptionError
from .aes_cbc import BLOCK_SIZE
from .config import METHOD_AES_128, EncryptionConfig


@dataclass(frozen=True)
class KeyDescriptor:
    """播放列表中一个 EXT-X-KEY 标签的结构化描述。"""

    method: str
    uri: str
    #: 显式 IV；None 表示隐式序号 IV，渲染时省略 IV 属性
    iv: bytes | None = None

    def __post_init__(self) -> None:
        if self.method != METHOD_AES_128:
            raise EncryptionError(
                f"本工具仅支持 {METHOD_AES_128} 方法，实际为 {self.method!r}"
            )
        if not isinstance(self.uri, str) or not self.uri:
            raise EncryptionError("EXT-X-KEY 的 URI 必须是非空字符串")
        if '"' in self.uri or any(ord(ch) < 0x20 for ch in self.uri):
            raise EncryptionError("EXT-X-KEY 的 URI 含引号或控制字符，无法安全渲染")
        if self.iv is not None and len(self.iv) != BLOCK_SIZE:
            raise EncryptionError(
                f"EXT-X-KEY 的 IV 必须为 {BLOCK_SIZE} 字节，实际为 "
                f"{0 if self.iv is None else len(self.iv)} 字节"
            )

    @classmethod
    def from_encryption_config(cls, config: EncryptionConfig) -> "KeyDescriptor":
        """从打包用的加密配置提取播放列表视角的密钥描述。"""
        return cls(method=METHOD_AES_128, uri=config.key_uri, iv=config.iv)


def render_key_tag(descriptor: KeyDescriptor) -> str:
    """渲染完整的 EXT-X-KEY 标签行（不含末尾换行）。"""
    parts = [
        f"#EXT-X-KEY:METHOD={descriptor.method}",
        f'URI="{descriptor.uri}"',
    ]
    if descriptor.iv is not None:
        parts.append(f"IV=0x{descriptor.iv.hex()}")
    return ",".join(parts)
