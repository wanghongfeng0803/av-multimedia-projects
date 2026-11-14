"""AES-128 / CBC / PKCS#7 原语测试（纯逻辑，不依赖 ffmpeg）。

正确性锚点：
* FIPS-197 附录 B 的 AES-128 单块官方向量；
* NIST SP 800-38A F.2.1/F.2.2 的 CBC 官方向量（无填充原始 CBC，
  本测试只比对数据块，填充块单独验证）；
* 装有 openssl 时做 ``openssl enc -aes-128-cbc`` 字节级交叉验证；
* PKCS#7 去填充的全部畸形分支与流式分块等价性。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from hls_packager.encryption.aes_cbc import (
    BLOCK_SIZE,
    CBCDecryptor,
    CBCEncryptor,
    aes128_decrypt_block,
    aes128_encrypt_block,
    cbc_decrypt,
    cbc_encrypt,
    pkcs7_pad,
    pkcs7_unpad,
)
from hls_packager.errors import EncryptionError

#: SP 800-38A 附录 F 统一使用的 128 位密钥与 IV
FIPS_KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
FIPS_IV = bytes.fromhex("000102030405060708090a0b0c0d0e0f")

#: FIPS-197 附录 B 单块向量
FIPS197_PLAINTEXT = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
FIPS197_CIPHERTEXT = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")

#: SP 800-38A F.2.1（CBC 加密）四组明文与对应密文（openssl -nopad 复核过的真值）
SP800_PLAIN_BLOCKS = [
    "6bc1bee22e409f96e93d7e117393172a",
    "ae2d8a571e03ac9c9eb76fac45af8e51",
    "30c81c46a35ce411e5fbc1191a0a52ef",
    "f69f2445df4f9b17ad2b417be66c3710",
]
SP800_CIPHER_BLOCKS = [
    "7649abac8119b246cee98e9b12e9197d",
    "5086cb9b507219ee95db113a917678b2",
    "73bed6b8e3c1743b7116e69e22229516",
    "3ff1caa1681fac09120eca307586e1a7",
]

HAS_OPENSSL = shutil.which("openssl") is not None
requires_openssl = pytest.mark.skipif(not HAS_OPENSSL, reason="环境中未安装 openssl")


# ---------------------------------------------------------------------------
# FIPS-197 单块
# ---------------------------------------------------------------------------
def test_fips197_appendix_b_encrypt_vector():
    assert aes128_encrypt_block(FIPS197_PLAINTEXT, FIPS_KEY) == FIPS197_CIPHERTEXT


def test_fips197_appendix_b_decrypt_vector():
    assert aes128_decrypt_block(FIPS197_CIPHERTEXT, FIPS_KEY) == FIPS197_PLAINTEXT


def test_block_cipher_is_involution():
    block = os.urandom(BLOCK_SIZE)
    assert aes128_decrypt_block(aes128_encrypt_block(block, FIPS_KEY), FIPS_KEY) == block


def test_rejects_non_16_byte_block_and_key():
    with pytest.raises(EncryptionError):
        aes128_encrypt_block(b"short", FIPS_KEY)
    with pytest.raises(EncryptionError):
        aes128_encrypt_block(FIPS197_PLAINTEXT, b"k" * 15)


# ---------------------------------------------------------------------------
# SP 800-38A CBC 链路（数据块部分逐字节对齐官方向量）
# ---------------------------------------------------------------------------
def test_sp800_38a_cbc_encrypt_data_blocks():
    plaintext = bytes.fromhex("".join(SP800_PLAIN_BLOCKS))
    expected = bytes.fromhex("".join(SP800_CIPHER_BLOCKS))
    # 本实现自带 PKCS#7，密文比无填充官方向量恰好多一个填充块
    ciphertext = cbc_encrypt(plaintext, FIPS_KEY, FIPS_IV)
    assert ciphertext[:len(expected)] == expected
    assert len(ciphertext) == len(expected) + BLOCK_SIZE


def test_sp800_38a_cbc_decrypt_data_blocks():
    ciphertext = bytes.fromhex("".join(SP800_CIPHER_BLOCKS))
    expected = bytes.fromhex("".join(SP800_PLAIN_BLOCKS))
    # 官方向量无填充：流式解密器会缓存最后一块等待去填充，update 只产出
    # 前 3 块；末块用公开的块密码原语按 CBC 链手动解出，拼成完整明文
    decryptor = CBCDecryptor(FIPS_KEY, FIPS_IV)
    recovered = decryptor.update(ciphertext)
    assert recovered == expected[:48]
    last_plain = bytes(
        a ^ b
        for a, b in zip(
            aes128_decrypt_block(ciphertext[-16:], FIPS_KEY),
            ciphertext[-32:-16],
        )
    )
    assert recovered + last_plain == expected


# ---------------------------------------------------------------------------
# PKCS#7 填充
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 32, 1000])
def test_pkcs7_roundtrip(length):
    data = bytes((i % 256 for i in range(length)))
    padded = pkcs7_pad(data)
    assert len(padded) % BLOCK_SIZE == 0
    assert pkcs7_unpad(padded) == data


def test_pkcs7_always_appends_full_block_when_aligned():
    data = b"a" * BLOCK_SIZE
    padded = pkcs7_pad(data)
    assert len(padded) == 2 * BLOCK_SIZE
    assert padded[-BLOCK_SIZE:] == bytes([BLOCK_SIZE]) * BLOCK_SIZE
    assert pkcs7_unpad(padded) == data


def test_unpad_rejects_bad_padding_value():
    # 合法长度但末字节超出 1~16
    with pytest.raises(EncryptionError):
        pkcs7_unpad(b"a" * 15 + b"\x00")
    with pytest.raises(EncryptionError):
        pkcs7_unpad(b"a" * 15 + b"\x11")


def test_unpad_rejects_inconsistent_padding_bytes():
    # 末字节声称填充 4 字节，但前面的填充字节不一致
    malformed = b"a" * 12 + b"\x04\x04\x03\x04"
    with pytest.raises(EncryptionError):
        pkcs7_unpad(malformed)


def test_unpad_rejects_empty_and_misaligned():
    with pytest.raises(EncryptionError):
        pkcs7_unpad(b"")
    with pytest.raises(EncryptionError):
        pkcs7_unpad(b"a" * 17)


# ---------------------------------------------------------------------------
# 流式接口：任意分块喂入与一次性调用等价
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 100, 4096])
def test_streaming_encrypt_matches_one_shot(chunk_size):
    data = os.urandom(5000)
    one_shot = cbc_encrypt(data, FIPS_KEY, FIPS_IV)

    encryptor = CBCEncryptor(FIPS_KEY, FIPS_IV)
    streamed = b"".join(
        encryptor.update(data[i:i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ) + encryptor.finish()
    assert streamed == one_shot


@pytest.mark.parametrize("chunk_size", [1, 16, 17, 4096])
def test_streaming_decrypt_matches_one_shot(chunk_size):
    data = os.urandom(5001)
    ciphertext = cbc_encrypt(data, FIPS_KEY, FIPS_IV)

    decryptor = CBCDecryptor(FIPS_KEY, FIPS_IV)
    streamed = b"".join(
        decryptor.update(ciphertext[i:i + chunk_size])
        for i in range(0, len(ciphertext), chunk_size)
    ) + decryptor.finish()
    assert streamed == data


def test_double_finish_and_update_after_finish_are_rejected():
    encryptor = CBCEncryptor(FIPS_KEY, FIPS_IV)
    encryptor.update(b"abc")
    encryptor.finish()
    with pytest.raises(EncryptionError):
        encryptor.finish()
    with pytest.raises(EncryptionError):
        encryptor.update(b"x")

    decryptor = CBCDecryptor(FIPS_KEY, FIPS_IV)
    decryptor.update(cbc_encrypt(b"abc", FIPS_KEY, FIPS_IV))
    decryptor.finish()
    with pytest.raises(EncryptionError):
        decryptor.finish()


def test_decrypt_rejects_truncated_ciphertext():
    decryptor = CBCDecryptor(FIPS_KEY, FIPS_IV)
    decryptor.update(b"a" * 17)  # 末段只剩 1 字节，无法凑成完整块
    with pytest.raises(EncryptionError):
        decryptor.finish()


def test_wrong_key_is_detected_via_padding():
    ciphertext = cbc_encrypt(b"hello HLS AES-128", FIPS_KEY, FIPS_IV)
    wrong_key = bytes([0x00]) * 16
    # 错误密钥解出的明文填充几乎必然非法（1/256 概率误过，断言失败才需要关注）
    with pytest.raises(EncryptionError):
        cbc_decrypt(ciphertext, wrong_key, FIPS_IV)


def test_bad_iv_length_rejected():
    with pytest.raises(EncryptionError):
        CBCEncryptor(FIPS_KEY, b"short-iv")
    with pytest.raises(EncryptionError):
        CBCDecryptor(FIPS_KEY, b"short-iv")


# ---------------------------------------------------------------------------
# openssl 对等交叉验证（同为 PKCS#7 填充，应逐字节一致）
# ---------------------------------------------------------------------------
@requires_openssl
@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 1000, 4097, 65537])
def test_matches_openssl_encryption(length):
    data = os.urandom(length)
    key, iv = os.urandom(16), os.urandom(16)
    mine = cbc_encrypt(data, key, iv)

    proc = subprocess.run(
        ["openssl", "enc", "-aes-128-cbc", "-K", key.hex(), "-iv", iv.hex()],
        input=data, capture_output=True,
    )
    assert proc.returncode == 0
    assert proc.stdout == mine


@requires_openssl
@pytest.mark.parametrize("length", [0, 1, 16, 33, 12345])
def test_openssl_can_decrypt_our_output(length):
    data = os.urandom(length)
    key, iv = os.urandom(16), os.urandom(16)
    ciphertext = cbc_encrypt(data, key, iv)

    proc = subprocess.run(
        ["openssl", "enc", "-d", "-aes-128-cbc", "-K", key.hex(), "-iv", iv.hex()],
        input=ciphertext, capture_output=True,
    )
    assert proc.returncode == 0
    assert proc.stdout == data


@requires_openssl
def test_tampered_ciphertext_fails_both_us_and_openssl():
    data = os.urandom(1000)
    key, iv = os.urandom(16), os.urandom(16)
    ciphertext = bytearray(cbc_encrypt(data, key, iv))
    ciphertext[-1] ^= 0xFF  # 翻转末块填充区一个比特

    with pytest.raises(EncryptionError):
        cbc_decrypt(bytes(ciphertext), key, iv)
    proc = subprocess.run(
        ["openssl", "enc", "-d", "-aes-128-cbc", "-K", key.hex(), "-iv", iv.hex()],
        input=bytes(ciphertext), capture_output=True,
    )
    assert proc.returncode != 0
