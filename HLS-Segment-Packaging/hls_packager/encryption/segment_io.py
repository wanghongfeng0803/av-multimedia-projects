"""分片与密钥文件的加密 IO（流式处理、原子落盘）。

只负责"把已切好的明文 TS 变成密文 TS"这一段物理工作，加密策略与
IV 语义全部来自 :class:`~hls_packager.encryption.config.EncryptionConfig`：

* :func:`write_key_file` 把 16 字节原始密钥写入输出目录（HLS 播放器
  通过 EXT-X-KEY 的 URI 拉取的就是这个裸密钥文件）；
* :func:`encrypt_segment_file` 以 64 KiB 为单位流式加密单个分片，
  先写同目录临时文件再 ``os.replace`` 原子替换，避免半途失败留下
  半截密文被播放列表引用。

为什么不把加密塞进 ffmpeg 切片命令
----------------------------------
本项目坚持自己掌握 M3U8 与密钥材料：密钥不落进命令行（``ps`` 可见）、
不依赖 ffmpeg 对 hls 加密 muxer 的实现细节；切片与加密解耦后，
加密模块可以独立用 NIST 向量与 openssl 做正确性验证。
"""

from __future__ import annotations

import os

from ..errors import EncryptionError
from .aes_cbc import CBCEncryptor
from .config import DEFAULT_KEY_FILENAME, EncryptionConfig

#: 流式加密的读写块大小（64 KiB，兼顾内存占用与系统调用次数）
ENCRYPTION_CHUNK_SIZE = 64 * 1024
#: 临时密文文件的后缀
_TMP_SUFFIX = ".enc.tmp"


def write_key_file(
    config: EncryptionConfig,
    output_dir: str,
    key_filename: str = DEFAULT_KEY_FILENAME,
) -> str:
    """把密钥以 16 字节原始二进制写入输出目录，返回文件绝对路径。

    :raises EncryptionError: 目录不可写或磁盘 IO 失败。
    """
    path = os.path.join(output_dir, key_filename)
    try:
        with open(path, "wb") as handle:
            handle.write(config.key)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EncryptionError(f"密钥文件写入失败 {path}：{exc}") from exc
    return os.path.abspath(path)


def encrypt_segment_file(
    path: str,
    config: EncryptionConfig,
    sequence_number: int,
    *,
    chunk_size: int = ENCRYPTION_CHUNK_SIZE,
) -> int:
    """对单个分片原地执行 AES-128-CBC 加密，返回密文字节数。

    流程：打开明文 -> 按 ``chunk_size`` 喂入 :class:`CBCEncryptor`
    -> 密文写同目录临时文件 -> flush/fsync -> ``os.replace`` 原子覆盖。
    IV 由配置按分片序号解析（显式 IV 全片共用，或序号大端派生）。

    :raises EncryptionError: 文件缺失/为空、序号非法或任意 IO 失败；
        失败时清理临时文件，原明文分片保持不变。
    """
    if not os.path.isfile(path):
        raise EncryptionError(f"待加密分片不存在：{path}")
    if os.path.getsize(path) == 0:
        raise EncryptionError(f"待加密分片为空文件，拒绝加密：{path}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise EncryptionError(
            f"加密块大小必须为正整数，实际为 {chunk_size!r}"
        )

    iv = config.resolve_iv(sequence_number)
    encryptor = CBCEncryptor(config.key, iv)
    tmp_path = path + _TMP_SUFFIX

    try:
        with open(path, "rb") as src, open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(encryptor.update(chunk))
            dst.write(encryptor.finish())
            dst.flush()
            os.fsync(dst.fileno())
        # 全部密文落盘后再原子替换，任何中途失败都不会动到原明文分片
        os.replace(tmp_path, path)
    except EncryptionError:
        _cleanup_tmp(tmp_path)
        raise
    except OSError as exc:
        _cleanup_tmp(tmp_path)
        raise EncryptionError(
            f"分片加密失败 {path}：{exc}（明文分片未被替换）"
        ) from exc

    return os.path.getsize(path)


def _cleanup_tmp(tmp_path: str) -> None:
    """尽力清理临时密文文件，清理失败不掩盖原始异常。"""
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass
