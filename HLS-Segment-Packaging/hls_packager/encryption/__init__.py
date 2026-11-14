"""HLS AES-128 分片加密能力（RFC 8216 §4.3.2.4 / §5.2，纯标准库实现）。

模块划分（数据单向流动，全部仅依赖 Python 标准库）：

``aes_cbc``    零依赖 AES-128 块密码（FIPS-197）+ CBC（SP 800-38A）
               + PKCS#7 填充，提供流式 update/finish 与一次性函数
``config``     密钥/IV/URI 的解析与严格校验、不可变 EncryptionConfig、
               序号 IV 派生（RFC §5.2）
``segment_io`` 分片流式加密与原子落盘、密钥文件写入
``key_info``   EXT-X-KEY 描述模型与播放列表属性行渲染
"""

from __future__ import annotations

from ..errors import EncryptionError
from .aes_cbc import (
    BLOCK_SIZE,
    KEY_SIZE,
    CBCDecryptor,
    CBCEncryptor,
    aes128_decrypt_block,
    aes128_encrypt_block,
    cbc_decrypt,
    cbc_encrypt,
    pkcs7_pad,
    pkcs7_unpad,
    random_iv,
)
from .config import (
    DEFAULT_KEY_FILENAME,
    METHOD_AES_128,
    EncryptionConfig,
    parse_iv,
    parse_key,
    sequence_iv,
)
from .key_info import KeyDescriptor, render_key_tag
from .segment_io import encrypt_segment_file, write_key_file

__all__ = [
    # 异常
    "EncryptionError",
    # 配置与解析
    "EncryptionConfig",
    "KeyDescriptor",
    "METHOD_AES_128",
    "DEFAULT_KEY_FILENAME",
    "parse_key",
    "parse_iv",
    "sequence_iv",
    "render_key_tag",
    # 分片 IO
    "encrypt_segment_file",
    "write_key_file",
    # 原语（库用户可独立使用/测试）
    "BLOCK_SIZE",
    "KEY_SIZE",
    "aes128_encrypt_block",
    "aes128_decrypt_block",
    "CBCEncryptor",
    "CBCDecryptor",
    "cbc_encrypt",
    "cbc_decrypt",
    "pkcs7_pad",
    "pkcs7_unpad",
    "random_iv",
]
