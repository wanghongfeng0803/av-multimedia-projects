"""加密配置、密钥/IV 解析与 EXT-X-KEY 渲染测试（纯逻辑，不依赖 ffmpeg）。"""

from __future__ import annotations

import base64
import os

import pytest

from hls_packager.encryption import (
    DEFAULT_KEY_FILENAME,
    EncryptionConfig,
    EncryptionError,
    KeyDescriptor,
    parse_iv,
    parse_key,
    render_key_tag,
    sequence_iv,
)

KEY = bytes(range(16))
IV = bytes(range(16, 32))


# ---------------------------------------------------------------------------
# parse_key：多来源、容错
# ---------------------------------------------------------------------------
def test_parse_key_accepts_raw_bytes():
    assert parse_key(KEY) == KEY
    assert parse_key(bytearray(KEY)) == KEY


def test_parse_key_accepts_hex():
    assert parse_key(KEY.hex()) == KEY
    assert parse_key("  " + KEY.hex().upper() + "  ") == KEY


def test_parse_key_accepts_base64():
    encoded = base64.b64encode(KEY).decode()
    assert parse_key(encoded) == KEY


def test_parse_key_rejects_wrong_length_bytes():
    with pytest.raises(EncryptionError, match="16 字节"):
        parse_key(b"short")
    with pytest.raises(EncryptionError):
        parse_key(b"x" * 17)


def test_parse_key_rejects_bad_text():
    for bad in ["", "   ", "xyz", "g" * 32, "a" * 31, 12345, None]:
        with pytest.raises(EncryptionError):
            parse_key(bad)


def test_parse_key_base64_wrong_decoded_length():
    # 合法 Base64 但解码后不是 16 字节
    with pytest.raises(EncryptionError, match="16 字节"):
        parse_key(base64.b64encode(b"only-10-bytes").decode())


# ---------------------------------------------------------------------------
# parse_iv
# ---------------------------------------------------------------------------
def test_parse_iv_forms_equivalent():
    assert parse_iv(IV) == IV
    assert parse_iv(IV.hex()) == IV
    assert parse_iv("0x" + IV.hex()) == IV
    assert parse_iv(base64.b64encode(IV).decode()) == IV


def test_parse_iv_rejects_bad_input():
    with pytest.raises(EncryptionError):
        parse_iv(b"short")
    with pytest.raises(EncryptionError):
        parse_iv("zz" * 16)
    with pytest.raises(EncryptionError):
        parse_iv(123)


# ---------------------------------------------------------------------------
# 序号 IV 派生（RFC 8216 §5.2：128 位大端）
# ---------------------------------------------------------------------------
def test_sequence_iv_big_endian():
    assert sequence_iv(0) == b"\x00" * 16
    assert sequence_iv(1) == b"\x00" * 15 + b"\x01"
    assert sequence_iv(255) == b"\x00" * 15 + b"\xff"
    assert sequence_iv(256) == b"\x00" * 14 + b"\x01\x00"
    # 大值：2^127 落在首字节最高位
    huge = 1 << 127
    assert sequence_iv(huge) == b"\x80" + b"\x00" * 15


def test_sequence_iv_rejects_negative_and_bool():
    with pytest.raises(EncryptionError):
        sequence_iv(-1)
    with pytest.raises(EncryptionError):
        sequence_iv(True)


# ---------------------------------------------------------------------------
# EncryptionConfig 构造与校验
# ---------------------------------------------------------------------------
def test_config_from_key_with_and_without_iv():
    cfg = EncryptionConfig.from_key(KEY.hex(), iv=IV.hex())
    assert cfg.key == KEY and cfg.iv == IV
    assert not cfg.uses_sequence_iv
    assert cfg.iv_hex == "0x" + IV.hex()

    cfg_seq = EncryptionConfig.from_key(KEY)
    assert cfg_seq.iv is None and cfg_seq.uses_sequence_iv
    assert cfg_seq.iv_hex is None
    assert cfg_seq.resolve_iv(42) == sequence_iv(42)
    assert cfg.resolve_iv(42) == IV  # 显式 IV 不随序号变化


def test_config_generate_uses_csprng():
    cfg = EncryptionConfig.generate()
    assert len(cfg.key) == 16 and len(cfg.iv) == 16
    # 两次生成的密钥/IV 不应相同（随机失败概率可忽略）
    again = EncryptionConfig.generate()
    assert cfg.key != again.key and cfg.iv != again.iv

    seq_style = EncryptionConfig.generate(random_iv=False)
    assert seq_style.iv is None


def test_config_rejects_bad_uri():
    for bad_uri in ["", 'a"b', "a\nb", "a\x7fb", "中文.key"]:
        with pytest.raises(EncryptionError):
            EncryptionConfig(key=KEY, key_uri=bad_uri)


def test_config_accepts_absolute_https_uri():
    cfg = EncryptionConfig(
        key=KEY, key_uri="https://cdn.example.com/keys/enc.key"
    )
    assert cfg.key_uri.startswith("https://")


def test_config_from_key_file_binary(tmp_path):
    path = tmp_path / "k.bin"
    path.write_bytes(KEY)
    cfg = EncryptionConfig.from_key_file(str(path), iv=IV)
    assert cfg.key == KEY and cfg.iv == IV


def test_config_from_key_file_hex_text(tmp_path):
    path = tmp_path / "k.hex"
    path.write_text(KEY.hex() + "\n", encoding="utf-8")
    assert EncryptionConfig.from_key_file(str(path)).key == KEY


def test_config_from_key_file_errors(tmp_path):
    missing = tmp_path / "nope.key"
    with pytest.raises(EncryptionError):
        EncryptionConfig.from_key_file(str(missing))

    empty = tmp_path / "empty.key"
    empty.write_bytes(b"")
    with pytest.raises(EncryptionError):
        EncryptionConfig.from_key_file(str(empty))

    bad = tmp_path / "bad.key"
    bad.write_bytes(b"not hex nor base64 raw!!")  # 既非 16 字节也非合法文本
    with pytest.raises(EncryptionError):
        EncryptionConfig.from_key_file(str(bad))


def test_validate_is_idempotent():
    cfg = EncryptionConfig.from_key(KEY)
    cfg.validate()  # 不抛异常即可


# ---------------------------------------------------------------------------
# KeyDescriptor 与 EXT-X-KEY 渲染
# ---------------------------------------------------------------------------
def test_render_key_tag_with_explicit_iv():
    tag = render_key_tag(KeyDescriptor.from_encryption_config(
        EncryptionConfig(key=KEY, iv=IV, key_uri="encryption.key")
    ))
    assert tag == (
        '#EXT-X-KEY:METHOD=AES-128,URI="encryption.key",'
        f"IV=0x{IV.hex()}"
    )


def test_render_key_tag_without_iv_omits_attribute():
    tag = render_key_tag(KeyDescriptor.from_encryption_config(
        EncryptionConfig(key=KEY, key_uri="k/enc.key")
    ))
    assert tag == '#EXT-X-KEY:METHOD=AES-128,URI="k/enc.key"'
    assert "IV=" not in tag


def test_key_descriptor_rejects_unsupported_method_and_bad_iv():
    with pytest.raises(EncryptionError):
        KeyDescriptor(method="SAMPLE-AES", uri="x")
    with pytest.raises(EncryptionError):
        KeyDescriptor(method="AES-128", uri="x", iv=b"short")
    with pytest.raises(EncryptionError):
        KeyDescriptor(method="AES-128", uri='bad"uri')


def test_default_key_filename_constant():
    assert DEFAULT_KEY_FILENAME == "encryption.key"
    assert EncryptionConfig(key=KEY).key_uri == DEFAULT_KEY_FILENAME
