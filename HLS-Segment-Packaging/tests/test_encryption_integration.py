"""AES-128 加密打包端到端集成测试（需要 ffmpeg/ffprobe，openssl 可选）。

验收口径：
* 加密播放列表含合规 EXT-X-KEY（METHOD/URI/IV），位置在首个 EXTINF 之前；
* openssl 解密每个分片后，ffprobe 可正常解析、时长与明文分片一致；
* 显式 IV 与 RFC 序号 IV 两种模式都验证；
* 不加密的旧路径产物逐字节保持旧行为（playlist 无 KEY、分片仍是明文 TS）；
* segments.json 记录加密信息；CLI 加密开关与错误退出码正确。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from hls_packager.cli import main
from hls_packager.encryption import (
    DEFAULT_KEY_FILENAME,
    EncryptionConfig,
    EncryptionError,
    sequence_iv,
)
from hls_packager.errors import PackagerError
from hls_packager.packager import PLAYLIST_FILENAME, package
from tests.conftest import ffprobe_json

pytestmark = pytest.mark.integration

HAS_OPENSSL = shutil.which("openssl") is not None
requires_openssl = pytest.mark.skipif(not HAS_OPENSSL, reason="环境中未安装 openssl")


def _read_playlist(out_dir: str) -> str:
    with open(os.path.join(out_dir, PLAYLIST_FILENAME), encoding="utf-8") as handle:
        return handle.read()


def _openssl_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    proc = subprocess.run(
        ["openssl", "enc", "-d", "-aes-128-cbc", "-K", key.hex(), "-iv", iv.hex()],
        input=ciphertext, capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    return proc.stdout


def test_encrypted_package_playlist_has_key_tag(h264_file, out_dir):
    cfg = EncryptionConfig.generate()
    result = package(
        h264_file, out_dir, target_duration=4.0, encryption=cfg, overwrite=True
    )
    playlist = _read_playlist(out_dir)
    lines = playlist.splitlines()

    key_line = next(line for line in lines if line.startswith("#EXT-X-KEY:"))
    assert (
        f'#EXT-X-KEY:METHOD=AES-128,URI="{DEFAULT_KEY_FILENAME}",' in key_line
    )
    assert f"IV=0x{cfg.iv.hex()}" in key_line
    # KEY 标签必须在第一个 EXTINF 之前
    assert lines.index(key_line) < next(
        i for i, line in enumerate(lines) if line.startswith("#EXTINF:")
    )
    # 密钥文件落盘且为 16 字节裸密钥
    key_path = os.path.join(out_dir, DEFAULT_KEY_FILENAME)
    assert open(key_path, "rb").read() == cfg.key
    assert os.path.getsize(key_path) == 16
    assert result.extra["encryption"]["method"] == "AES-128"


@requires_openssl
def test_openssl_decrypts_every_encrypted_segment(h264_file, out_dir):
    cfg = EncryptionConfig.generate()
    result = package(
        h264_file, out_dir, target_duration=4.0, encryption=cfg, overwrite=True
    )

    for seg in result.segments:
        cipher_path = os.path.join(out_dir, seg.filename)
        ciphertext = open(cipher_path, "rb").read()
        # 密文分片已不是可直接解析的 MPEG-TS
        probe = subprocess.run(
            ["ffprobe", "-v", "error", cipher_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert probe.returncode != 0

        plaintext = _openssl_decrypt(ciphertext, cfg.key, cfg.resolve_iv(seg.sequence_number))
        assert plaintext[:1] == b"\x47"  # MPEG-TS 同步字节

        dec_path = os.path.join(out_dir, f"dec_{seg.filename}")
        with open(dec_path, "wb") as handle:
            handle.write(plaintext)
        data = ffprobe_json(dec_path, "-show_entries", "format=duration")
        assert float(data["format"]["duration"]) == pytest.approx(
            seg.duration, abs=0.05
        )


@requires_openssl
def test_sequence_iv_mode_playlist_and_decrypt(h264_file, out_dir):
    cfg = EncryptionConfig.generate(random_iv=False)
    result = package(
        h264_file, out_dir, target_duration=4.0, encryption=cfg, overwrite=True
    )
    playlist = _read_playlist(out_dir)

    # 序号模式：标签不含 IV 属性
    key_line = next(line for line in playlist.splitlines() if line.startswith("#EXT-X-KEY:"))
    assert "IV=" not in key_line
    assert key_line == f'#EXT-X-KEY:METHOD=AES-128,URI="{DEFAULT_KEY_FILENAME}"'

    for seg in result.segments:
        ciphertext = open(os.path.join(out_dir, seg.filename), "rb").read()
        plaintext = _openssl_decrypt(
            ciphertext, cfg.key, sequence_iv(seg.sequence_number)
        )
        assert plaintext[:1] == b"\x47"


def test_encrypted_segment_size_is_block_aligned(h264_file, out_dir):
    cfg = EncryptionConfig.generate()
    result = package(
        h264_file, out_dir, target_duration=4.0, encryption=cfg, overwrite=True
    )
    for seg in result.segments:
        size = os.path.getsize(os.path.join(out_dir, seg.filename))
        assert size % 16 == 0


def test_metadata_records_encryption_block(h264_file, out_dir):
    cfg = EncryptionConfig.from_key(
        bytes(range(16)), iv=bytes(range(16, 32)), key_uri="https://cdn/k.key"
    )
    result = package(
        h264_file, out_dir, target_duration=4.0, encryption=cfg, overwrite=True
    )
    payload = json.load(open(result.extra["metadata_path"], encoding="utf-8"))
    info = payload["hls"]["encryption"]

    assert info["method"] == "AES-128"
    assert info["key_uri"] == "https://cdn/k.key"
    assert info["iv_mode"] == "explicit"
    assert info["iv_hex"] == "0x" + bytes(range(16, 32)).hex()
    assert info["key_hex"] == bytes(range(16)).hex()
    assert info["padding"] == "PKCS#7"


def test_unencrypted_package_has_no_key_tag_and_plaintext_ts(h264_file, out_dir):
    """旧路径回归：不加密时播放列表不含 KEY，分片首字节就是 TS 同步字节。"""
    result = package(h264_file, out_dir, target_duration=4.0, overwrite=True)
    playlist = _read_playlist(out_dir)

    assert "#EXT-X-KEY" not in playlist
    payload = json.load(open(result.extra["metadata_path"], encoding="utf-8"))
    assert "encryption" not in payload["hls"]
    for seg in result.segments:
        with open(os.path.join(out_dir, seg.filename), "rb") as handle:
            assert handle.read(1) == b"\x47"
    assert "encryption" not in result.extra


def test_encryption_combines_with_media_sequence(h264_file, out_dir):
    # 非零起始序号 + 序号 IV：派生 IV 必须基于绝对 MEDIA-SEQUENCE
    cfg = EncryptionConfig.generate(random_iv=False)
    result = package(
        h264_file, out_dir, target_duration=4.0,
        media_sequence=100, encryption=cfg, overwrite=True,
    )
    first_seq = result.segments[0].sequence_number
    assert first_seq == 100

    if HAS_OPENSSL:
        seg = result.segments[0]
        ciphertext = open(os.path.join(out_dir, seg.filename), "rb").read()
        plaintext = _openssl_decrypt(ciphertext, cfg.key, sequence_iv(100))
        assert plaintext[:1] == b"\x47"


def test_encryption_combines_with_sliding_window(h264_file, out_dir):
    cfg = EncryptionConfig.generate()
    package(
        h264_file, out_dir, target_duration=4.0, window_size=3,
        encryption=cfg, overwrite=True,
    )
    playlist = _read_playlist(out_dir)
    assert "#EXT-X-KEY" in playlist
    assert "#EXT-X-ENDLIST" not in playlist  # 直播窗口语义不被加密破坏


def test_invalid_encryption_config_raises_packager_error(h264_file, out_dir):
    # 加密侧异常是 PackagerError 子类，编排层不吞错、CLI 可据此给退出码 2
    with pytest.raises(EncryptionError):
        bad = EncryptionConfig(key=b"short")  # type: ignore[arg-type]
        package(h264_file, out_dir, encryption=bad, overwrite=True)


# ---------------------------------------------------------------------------
# CLI 端到端
# ---------------------------------------------------------------------------
def test_cli_encrypt_generates_key_and_playlist(h264_file, out_dir, capsys):
    exit_code = main([h264_file, out_dir, "-d", "4", "-y", "--encrypt"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert os.path.isfile(os.path.join(out_dir, DEFAULT_KEY_FILENAME))
    playlist = _read_playlist(out_dir)
    assert "#EXT-X-KEY:METHOD=AES-128" in playlist
    assert "AES-128 加密" in captured.out


def test_cli_encrypt_sequence_iv_mode(h264_file, out_dir):
    exit_code = main([
        h264_file, out_dir, "-d", "4", "-y",
        "--encrypt", "--iv-mode", "sequence",
    ])
    assert exit_code == 0
    key_line = [
        line for line in _read_playlist(out_dir).splitlines()
        if line.startswith("#EXT-X-KEY:")
    ][0]
    assert "IV=" not in key_line


def test_cli_encrypt_with_hex_key_and_iv(h264_file, out_dir):
    key_hex = "aa" * 16
    iv_hex = "bb" * 16
    exit_code = main([
        h264_file, out_dir, "-d", "4", "-y",
        "--key", key_hex, "--iv", iv_hex,
    ])
    assert exit_code == 0
    playlist = _read_playlist(out_dir)
    assert f"IV=0x{iv_hex}" in playlist
    assert open(os.path.join(out_dir, DEFAULT_KEY_FILENAME), "rb").read() == bytes([0xAA]) * 16


def test_cli_bad_key_returns_error_code(h264_file, out_dir):
    exit_code = main([h264_file, out_dir, "-y", "--key", "too-short"])
    assert exit_code == 2


def test_cli_iv_mode_conflict_returns_error_code(h264_file, out_dir):
    exit_code = main([
        h264_file, out_dir, "-y", "--encrypt",
        "--iv", "bb" * 16, "--iv-mode", "sequence",
    ])
    assert exit_code == 2


def test_cli_without_encrypt_keeps_legacy_behavior(h264_file, out_dir):
    exit_code = main([h264_file, out_dir, "-d", "4", "-y"])
    assert exit_code == 0
    assert "#EXT-X-KEY" not in _read_playlist(out_dir)
    assert not os.path.exists(os.path.join(out_dir, DEFAULT_KEY_FILENAME))
