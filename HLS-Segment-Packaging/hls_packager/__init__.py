"""HLS MPEG-TS 切片打包工具。

公开 API 一览：

* :func:`package`            一键打包（探测 -> GOP 规划 -> 切片 -> 加密 -> M3U8）；
* :func:`probe_media`        只做 ffprobe 探测；
* :func:`plan_segments`      纯函数的 GOP 对齐分片规划；
* :func:`render_playlist`    渲染 M3U8 文本；
* :class:`EncryptionConfig`  AES-128 分片加密配置（可选能力）；
* :class:`PackagerError`     全部可预期错误的基类。
"""

from __future__ import annotations

from .encryption import (
    BLOCK_SIZE,
    DEFAULT_KEY_FILENAME,
    KEY_SIZE,
    CBCDecryptor,
    CBCEncryptor,
    EncryptionConfig,
    EncryptionError,
    KeyDescriptor,
    aes128_decrypt_block,
    aes128_encrypt_block,
    cbc_decrypt,
    cbc_encrypt,
    encrypt_segment_file,
    parse_iv,
    parse_key,
    pkcs7_pad,
    pkcs7_unpad,
    render_key_tag,
    sequence_iv,
    write_key_file,
)
from .errors import (
    BinaryNotFoundError,
    NamingError,
    PackagerError,
    PlaylistError,
    ProbeError,
    SegmenterError,
    UnsupportedInputError,
    WatermarkError,
)
from .m3u8 import PlaylistConfig, compute_target_duration, render_playlist
from .models import MediaInfo, PackagingResult, SegmentInfo, StreamInfo
from .naming import DEFAULT_NAME_PREFIX, DEFAULT_NAME_TEMPLATE, NamingConfig
from .containers import (
    DEFAULT_CONTAINER,
    SUPPORTED_CONTAINERS,
    CONTAINER_SPECS,
    ContainerSpec,
    get_container,
)
from .packager import package
from .planner import (
    ALIGN_AFTER,
    ALIGN_BEFORE,
    ALIGN_MODES,
    ALIGN_NEAREST,
    plan_segments,
)
from .probe import probe_media
from .watermark import (
    WatermarkConfig,
    build_watermarked_segment_command,
    decode_utf8,
    escape_filter_path,
    render_text_png,
)

__all__ = [
    "package",
    "probe_media",
    "plan_segments",
    "render_playlist",
    "compute_target_duration",
    "PlaylistConfig",
    "MediaInfo",
    "StreamInfo",
    "SegmentInfo",
    "PackagingResult",
    "PackagerError",
    "BinaryNotFoundError",
    "ProbeError",
    "UnsupportedInputError",
    "SegmenterError",
    "PlaylistError",
    "WatermarkError",
    "NamingError",
    "WatermarkConfig",
    "render_text_png",
    "decode_utf8",
    "escape_filter_path",
    "build_watermarked_segment_command",
    # 命名模板与容器规格
    "NamingConfig",
    "DEFAULT_NAME_TEMPLATE",
    "DEFAULT_NAME_PREFIX",
    "ContainerSpec",
    "CONTAINER_SPECS",
    "SUPPORTED_CONTAINERS",
    "DEFAULT_CONTAINER",
    "get_container",
    # GOP 切点对齐模式
    "ALIGN_NEAREST",
    "ALIGN_BEFORE",
    "ALIGN_AFTER",
    "ALIGN_MODES",
    # AES-128 分片加密（RFC 8216）
    "EncryptionConfig",
    "EncryptionError",
    "KeyDescriptor",
    "DEFAULT_KEY_FILENAME",
    "parse_key",
    "parse_iv",
    "sequence_iv",
    "render_key_tag",
    "encrypt_segment_file",
    "write_key_file",
    "KEY_SIZE",
    "BLOCK_SIZE",
    "aes128_encrypt_block",
    "aes128_decrypt_block",
    "CBCEncryptor",
    "CBCDecryptor",
    "cbc_encrypt",
    "cbc_decrypt",
    "pkcs7_pad",
    "pkcs7_unpad",
]

__version__ = "1.2.0"
