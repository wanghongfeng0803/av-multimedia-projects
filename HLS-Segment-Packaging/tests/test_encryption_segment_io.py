"""分片加密 IO 与密钥落盘测试（纯逻辑 + openssl 交叉，不依赖 ffmpeg）。"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from hls_packager.encryption import (
    EncryptionConfig,
    EncryptionError,
    cbc_decrypt,
    encrypt_segment_file,
    sequence_iv,
    write_key_file,
)

HAS_OPENSSL = shutil.which("openssl") is not None
requires_openssl = pytest.mark.skipif(not HAS_OPENSSL, reason="环境中未安装 openssl")


@pytest.fixture()
def segment_file(tmp_path):
    """造一个非块对齐的"分片"，返回 (路径, 原始字节)。"""
    path = tmp_path / "seg_0000.ts"
    data = os.urandom(100_003)  # 100003 % 16 = 3，强制走非对齐填充
    path.write_bytes(data)
    return str(path), data


def test_write_key_file_is_raw_16_bytes(tmp_path):
    cfg = EncryptionConfig.generate()
    path = write_key_file(cfg, str(tmp_path))
    assert os.path.isfile(path)
    assert open(path, "rb").read() == cfg.key
    assert os.path.getsize(path) == 16


def test_write_key_file_failure_is_encryption_error(tmp_path):
    cfg = EncryptionConfig.generate()
    # 以一个普通文件充当输出目录，写入必然失败
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(EncryptionError):
        write_key_file(cfg, str(blocker))


def test_encrypt_segment_roundtrip_our_own_decryptor(segment_file):
    path, original = segment_file
    cfg = EncryptionConfig.generate()
    cipher_size = encrypt_segment_file(path, cfg, sequence_number=0)

    ciphertext = open(path, "rb").read()
    assert cipher_size == len(ciphertext)
    # 密文长度 = 明文向上取整到 16 的倍数（PKCS#7 至少补 1 字节）
    assert len(ciphertext) == ((len(original) // 16) + 1) * 16
    assert ciphertext != original
    assert cbc_decrypt(ciphertext, cfg.key, cfg.iv) == original


def test_encrypt_uses_sequence_iv_when_no_explicit_iv(segment_file):
    path, original = segment_file
    cfg = EncryptionConfig.generate(random_iv=False)
    encrypt_segment_file(path, cfg, sequence_number=9)

    ciphertext = open(path, "rb").read()
    recovered = cbc_decrypt(ciphertext, cfg.key, sequence_iv(9))
    assert recovered == original


def test_encrypt_rejects_missing_empty_and_bad_chunk(tmp_path):
    cfg = EncryptionConfig.generate()
    with pytest.raises(EncryptionError):
        encrypt_segment_file(str(tmp_path / "nope.ts"), cfg, 0)

    empty = tmp_path / "empty.ts"
    empty.write_bytes(b"")
    with pytest.raises(EncryptionError):
        encrypt_segment_file(str(empty), cfg, 0)

    good = tmp_path / "good.ts"
    good.write_bytes(os.urandom(100))
    with pytest.raises(EncryptionError):
        encrypt_segment_file(str(good), cfg, 0, chunk_size=0)


def test_failed_encryption_leaves_plaintext_untouched(tmp_path, monkeypatch):
    cfg = EncryptionConfig.generate()
    path = tmp_path / "seg.ts"
    original = os.urandom(1234)
    path.write_bytes(original)

    # Windows 不按 POSIX 只读位限权，改用 monkeypatch 确定性模拟替换失败
    def boom(src, dst):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(
        "hls_packager.encryption.segment_io.os.replace", boom
    )
    with pytest.raises(EncryptionError):
        encrypt_segment_file(str(path), cfg, 0)

    # 原明文分片必须原样保留，且不残留临时文件
    assert path.read_bytes() == original
    assert not (tmp_path / "seg.ts.enc.tmp").exists()


def test_small_chunk_sizes_still_correct(segment_file):
    path, original = segment_file
    cfg = EncryptionConfig.generate()
    # 用极小的分块强制多次 update，验证流式拼接与大块一致
    encrypt_segment_file(path, cfg, 0, chunk_size=17)
    assert cbc_decrypt(open(path, "rb").read(), cfg.key, cfg.iv) == original


@requires_openssl
def test_openssl_decrypts_segment_with_explicit_iv(segment_file):
    path, original = segment_file
    cfg = EncryptionConfig.generate()
    encrypt_segment_file(path, cfg, 0)

    proc = subprocess.run(
        ["openssl", "enc", "-d", "-aes-128-cbc",
         "-K", cfg.key.hex(), "-iv", cfg.iv.hex()],
        input=open(path, "rb").read(), capture_output=True,
    )
    assert proc.returncode == 0
    assert proc.stdout == original


@requires_openssl
def test_openssl_decrypts_segment_with_sequence_iv(segment_file):
    path, original = segment_file
    cfg = EncryptionConfig.generate(random_iv=False)
    seq = 123
    encrypt_segment_file(path, cfg, seq)

    proc = subprocess.run(
        ["openssl", "enc", "-d", "-aes-128-cbc",
         "-K", cfg.key.hex(), "-iv", sequence_iv(seq).hex()],
        input=open(path, "rb").read(), capture_output=True,
    )
    assert proc.returncode == 0
    assert proc.stdout[:1] == b"\x47" or proc.stdout == original
    assert proc.stdout == original
