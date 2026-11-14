"""AES-128 分片加密的不可变配置与密钥/IV 解析。

本模块是加密能力的**输入边界**：所有来自用户的密钥、IV、URI 都在这里
解析并校验，错误一律以带中文说明的 :class:`EncryptionError` 抛出，
下游 :mod:`aes_cbc` 与编排层只接收已规范化的二进制材料。

支持的密钥来源
--------------
* 原始 16 字节 ``bytes``；
* 十六进制字符串（恰好 32 个十六进制字符，如 ``"a1b2..."``）；
* Base64 字符串（解码后恰好 16 字节，如 ``"o7G...="``）；
* 密钥文件：内容为 16 字节原始二进制，或只含 hex/base64 文本；
* :meth:`EncryptionConfig.generate` 调用系统 CSPRNG 现场生成。

IV 策略（RFC 8216 §5.2）
------------------------
* **显式 IV**：构造时传入 16 字节（或 hex/base64 文本），全部分片共用，
  并写入 ``EXT-X-KEY`` 的 ``IV`` 属性；
* **隐式（默认）**：不传 IV 时，每个分片的 IV 等于其 128 位大端
  MEDIA-SEQUENCE 序号，播放列表不写 ``IV`` 属性，由播放器自行派生。
  RFC 允许该做法，但同一密钥下不同打包会话复用序号会导致 IV 重复，
  因此 :meth:`generate` 默认改为随机显式 IV，更稳妥。
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

from ..errors import EncryptionError
from .aes_cbc import BLOCK_SIZE, KEY_SIZE

#: HLS AES-128 方法标识（RFC 8216 §4.3.2.4）
METHOD_AES_128 = "AES-128"
#: 密钥文件在输出目录中的默认文件名，同时是默认的密钥 URI
DEFAULT_KEY_FILENAME = "encryption.key"
#: 十六进制密钥/IV 文本的字符长度
_HEX_KEY_LENGTH = KEY_SIZE * 2
_HEX_IV_LENGTH = BLOCK_SIZE * 2


def _is_hex(text: str) -> bool:
    """判断字符串是否为纯十六进制（长度由调用方另行约束）。"""
    try:
        int(text, 16)
    except ValueError:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in text)


def parse_key(value: bytes | bytearray | str) -> bytes:
    """把多种形态的用户输入规范化为 16 字节 AES-128 密钥。

    :raises EncryptionError: 类型不支持、hex/base64 解码失败或长度不是 16 字节。
    """
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) != KEY_SIZE:
            raise EncryptionError(
                f"原始字节密钥必须恰好为 {KEY_SIZE} 字节，实际为 {len(raw)} 字节"
            )
        return raw

    if not isinstance(value, str):
        raise EncryptionError(
            f"密钥类型必须是 bytes 或 str，实际为 {type(value).__name__}"
        )

    text = value.strip()
    if not text:
        raise EncryptionError("密钥字符串不能为空")
    if len(text) == _HEX_KEY_LENGTH and _is_hex(text):
        try:
            return bytes.fromhex(text)
        except ValueError as exc:  # pragma: no cover - _is_hex 已兜底
            raise EncryptionError(f"十六进制密钥解析失败：{exc}") from exc

    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EncryptionError(
            "密钥字符串既不是 32 位十六进制，也不是合法 Base64；"
            f"请改用 {KEY_SIZE} 字节原始密钥或其 hex/base64 表示"
        ) from exc
    if len(decoded) != KEY_SIZE:
        raise EncryptionError(
            f"Base64 密钥解码后为 {len(decoded)} 字节，"
            f"AES-128 要求恰好 {KEY_SIZE} 字节"
        )
    return decoded


def parse_iv(value: bytes | bytearray | str) -> bytes:
    """把用户输入规范化为 16 字节 IV，规则与 :func:`parse_key` 一致。"""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) != BLOCK_SIZE:
            raise EncryptionError(
                f"原始字节 IV 必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(raw)} 字节"
            )
        return raw

    if not isinstance(value, str):
        raise EncryptionError(
            f"IV 类型必须是 bytes 或 str，实际为 {type(value).__name__}"
        )

    text = value.strip()
    if not text:
        raise EncryptionError("IV 字符串不能为空")
    if text.lower().startswith("0x"):
        # 宽容接受播放列表风格的 0x 前缀，内部统一走 hex 解析
        text = text[2:]
    if len(text) == _HEX_IV_LENGTH and _is_hex(text):
        return bytes.fromhex(text)

    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EncryptionError(
            "IV 字符串既不是 32 位十六进制，也不是合法 Base64；"
            f"IV 必须恰好编码 {BLOCK_SIZE} 字节"
        ) from exc
    if len(decoded) != BLOCK_SIZE:
        raise EncryptionError(
            f"Base64 IV 解码后为 {len(decoded)} 字节，要求恰好 {BLOCK_SIZE} 字节"
        )
    return decoded


def sequence_iv(sequence_number: int) -> bytes:
    """按 RFC 8216 §5.2 把分片序号派生为 128 位大端 IV。"""
    if not isinstance(sequence_number, int) or isinstance(sequence_number, bool):
        raise EncryptionError(
            f"分片序号必须是整数，实际为 {type(sequence_number).__name__}"
        )
    if sequence_number < 0:
        raise EncryptionError(f"分片序号不能为负数，实际为 {sequence_number}")
    try:
        return sequence_number.to_bytes(BLOCK_SIZE, byteorder="big", signed=False)
    except OverflowError as exc:
        raise EncryptionError(
            f"分片序号 {sequence_number} 超出 128 位 IV 的表示范围"
        ) from exc


def _validate_key_uri(uri: str) -> None:
    """校验 EXT-X-KEY 的 URI：非空、无引号/控制字符/非 ASCII。

    RFC 8216 属性值是带引号字符串，双引号与换行必须排除；非 ASCII
    字符应按 RFC 3986 先做百分号编码，这里明确拒绝并给出提示，
    避免产出播放器无法解析的播放列表。
    """
    if not isinstance(uri, str) or not uri:
        raise EncryptionError("密钥 URI 必须是非空字符串")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in uri):
        raise EncryptionError("密钥 URI 不能包含控制字符或换行")
    if '"' in uri:
        raise EncryptionError('密钥 URI 不能包含双引号（"），请改用 %22 编码')
    if any(ord(ch) > 0x7F for ch in uri):
        raise EncryptionError(
            "密钥 URI 不能直接包含非 ASCII 字符，"
            "请按 RFC 3986 做百分号编码后再传入"
        )


@dataclass(frozen=True)
class EncryptionConfig:
    """一次 AES-128 加密打包的完整配置。

    构造时优先用类方法 :meth:`from_key` / :meth:`from_key_file` /
    :meth:`generate`，它们会完成密钥解析与全部前置校验；直接实例化
    只接受已规范化的 16 字节密钥。

    不可变：配置在整条打包流水线中共享，冻结后杜绝中途被篡改。
    """

    key: bytes
    #: 显式 IV；None 表示按分片序号隐式派生（播放列表不写 IV 属性）
    iv: bytes | None = None
    #: 写入 EXT-X-KEY 的 URI，默认相对文件名，与落盘的密钥文件同名
    key_uri: str = DEFAULT_KEY_FILENAME

    def __post_init__(self) -> None:
        # dataclass 冻结后用 object.__setattr__ 规范化，校验仍只做一次
        if not isinstance(self.key, (bytes, bytearray)):
            raise EncryptionError("EncryptionConfig.key 必须是 bytes")
        if len(self.key) != KEY_SIZE:
            raise EncryptionError(
                f"AES-128 密钥必须恰好为 {KEY_SIZE} 字节，实际为 {len(self.key)} 字节"
            )
        if self.iv is not None:
            if not isinstance(self.iv, (bytes, bytearray)):
                raise EncryptionError("EncryptionConfig.iv 必须是 bytes 或 None")
            if len(self.iv) != BLOCK_SIZE:
                raise EncryptionError(
                    f"显式 IV 必须恰好为 {BLOCK_SIZE} 字节，实际为 {len(self.iv)} 字节"
                )
        _validate_key_uri(self.key_uri)

    # -- 构造类方法 -----------------------------------------------------
    @classmethod
    def from_key(
        cls,
        key: bytes | bytearray | str,
        iv: bytes | bytearray | str | None = None,
        key_uri: str = DEFAULT_KEY_FILENAME,
    ) -> "EncryptionConfig":
        """从原始字节 / hex / base64 密钥构造配置，IV 可选且同样支持文本。"""
        parsed_key = parse_key(key)
        parsed_iv = parse_iv(iv) if iv is not None else None
        return cls(key=parsed_key, iv=parsed_iv, key_uri=key_uri)

    @classmethod
    def from_key_file(
        cls,
        path: str,
        iv: bytes | bytearray | str | None = None,
        key_uri: str = DEFAULT_KEY_FILENAME,
    ) -> "EncryptionConfig":
        """从密钥文件构造：16 字节原始二进制，或仅含 hex/base64 的文本文件。"""
        if not isinstance(path, str) or not path:
            raise EncryptionError("密钥文件路径必须是非空字符串")
        try:
            with open(path, "rb") as handle:
                content = handle.read()
        except OSError as exc:
            raise EncryptionError(f"无法读取密钥文件 {path}：{exc}") from exc
        if not content:
            raise EncryptionError(f"密钥文件为空：{path}")
        if len(content) == KEY_SIZE:
            parsed_key = bytes(content)
        else:
            # 文本形态：解码后按 hex/base64 规则解析，报错信息透传
            try:
                text = content.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise EncryptionError(
                    f"密钥文件 {path} 既不是 {KEY_SIZE} 字节原始二进制，"
                    "也不是可读的 hex/base64 文本"
                ) from exc
            parsed_key = parse_key(text)
        parsed_iv = parse_iv(iv) if iv is not None else None
        return cls(key=parsed_key, iv=parsed_iv, key_uri=key_uri)

    @classmethod
    def generate(
        cls,
        key_uri: str = DEFAULT_KEY_FILENAME,
        *,
        random_iv: bool = True,
    ) -> "EncryptionConfig":
        """用系统 CSPRNG 生成全新密钥；默认同时生成随机显式 IV。

        :param random_iv: True（默认）时生成随机 IV 并写入播放列表；
            False 时退化为 RFC 隐式序号 IV，播放列表不写 IV 属性。
        """
        return cls(
            key=os.urandom(KEY_SIZE),
            iv=os.urandom(BLOCK_SIZE) if random_iv else None,
            key_uri=key_uri,
        )

    # -- 运行期行为 -----------------------------------------------------
    def validate(self) -> None:
        """供编排层在流水线入口统一调用；构造期已校验，这里只做断言式复核。"""
        if len(self.key) != KEY_SIZE:  # pragma: no cover - 构造期已拦截
            raise EncryptionError("密钥长度异常，请通过类方法构造 EncryptionConfig")
        if self.iv is not None and len(self.iv) != BLOCK_SIZE:
            raise EncryptionError("IV 长度异常，请通过类方法构造 EncryptionConfig")
        _validate_key_uri(self.key_uri)

    def resolve_iv(self, sequence_number: int) -> bytes:
        """返回指定分片实际使用的 IV：显式 IV 优先，否则序号派生。"""
        return self.iv if self.iv is not None else sequence_iv(sequence_number)

    @property
    def uses_sequence_iv(self) -> bool:
        """是否采用 RFC 隐式序号 IV（播放列表据此决定是否输出 IV 属性）。"""
        return self.iv is None

    @property
    def key_hex(self) -> str:
        """密钥的小写十六进制表示，用于元数据与终端提示。"""
        return self.key.hex()

    @property
    def iv_hex(self) -> str | None:
        """显式 IV 的 0x 前缀十六进制（播放列表属性格式）；隐式 IV 返回 None。"""
        return "0x" + self.iv.hex() if self.iv is not None else None
