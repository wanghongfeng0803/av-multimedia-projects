"""端到端集成测试：真实 ffmpeg 打包后，独立解析 m3u8 与分片交叉验证。

校验口径全部独立于被测代码自己的输出：

- 直接解析落盘 m3u8 文本的标签与 EXTINF；
- 对每个分片单独跑 ffprobe 实测时长，与 EXTINF 声明值比对；
- 校验分片均可独立解码（片首 IDR）、序号/URI 连续、TARGETDURATION
  覆盖所有分片、末片短尾如实声明。
"""

import json
import re
import subprocess

from tests.conftest import FFMPEG, FFPROBE, needs_ffmpeg

from vhls import pack_hls
from vhls.pipeline import DEFAULT_TARGET_SECONDS


def _ffprobe_duration_ms(path) -> float:
    """独立实测分片视频轨时长（秒，浮点）；缺视频流时回退 format。

    验证方独立解析 ffprobe，但口径与被测代码一致：HLS 按视频轨时间线
    呈现，MPEG-TS format duration 的音频填充不应计入。
    """
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration:stream=duration,codec_type",
        "-of", "json", str(path),
    ]
    data = json.loads(
        subprocess.run(cmd, check=True, capture_output=True).stdout
    )
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and \
                stream.get("duration") not in (None, "N/A"):
            return float(stream["duration"])
    return float(data["format"]["duration"])


def _parse_m3u8(path):
    """极简 m3u8 解析器（测试专用，不依赖任何三方库）。"""
    tags, extinfs, uris = {}, [], []
    media_sequence = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-TARGETDURATION:"):
            tags["target"] = int(line.split(":", 1)[1])
        elif line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            media_sequence = int(line.split(":", 1)[1])
        elif line.startswith("#EXTINF:"):
            extinfs.append(float(line.split(":", 1)[1].rstrip(",")))
        elif line.startswith("#"):
            tags.setdefault("misc", []).append(line)
        else:
            uris.append(line)
    return tags, media_sequence, extinfs, uris


def _assert_independently_decodable(seg_path):
    """分片必须可独立解码：ffmpeg 单独转 null 不报错且能产出帧。"""
    cmd = [
        FFMPEG, "-v", "error", "-i", str(seg_path),
        "-map", "0:v:0", "-f", "null", "-",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@needs_ffmpeg
class TestEndToEnd:
    def test_cfr_pack_full_consistency(self, video_factory, tmp_path):
        # 25fps、150 帧 = 6.000s，默认目标 6s → 1 片完整覆盖。
        src = video_factory(frames=150, fps=25)
        out_dir = tmp_path / "hls"
        result = pack_hls(src, out_dir)

        m3u8 = out_dir / "index.m3u8"
        tags, seq, extinfs, uris = _parse_m3u8(m3u8)
        assert seq == 0
        assert len(extinfs) == result.plan.segment_count == 1
        assert uris == ["seg_00000.ts"]

        # EXTINF 与独立 ffprobe 实测一致（3 位小数容差 1ms）。
        seg = out_dir / uris[0]
        measured = _ffprobe_duration_ms(seg)
        assert abs(measured - extinfs[0]) <= 0.001
        assert extinfs[0] <= tags["target"]
        _assert_independently_decodable(seg)

    def test_multi_segment_sequence_and_target_duration(
        self, video_factory, tmp_path
    ):
        # 25fps、127 帧 = 5.08s，目标 2s → 2 个整片 + 1 个短尾片。
        src = video_factory(frames=127, fps=25)
        out_dir = tmp_path / "hls"
        result = pack_hls(src, out_dir, target_seconds=2.0)
        m3u8 = out_dir / "index.m3u8"
        tags, seq, extinfs, uris = _parse_m3u8(m3u8)

        assert seq == 0
        assert result.plan.segment_count == 3
        assert uris == [f"seg_{i:05d}.ts" for i in range(3)]
        # 所有 EXTINF <= TARGETDURATION，且 TARGETDURATION 是上限 ceil。
        assert all(d <= tags["target"] for d in extinfs)
        assert tags["target"] == max(
            -(-int(round(d * 1000)) // 1000) for d in extinfs
        )

        # 逐片独立实测与声明值误差 <= 1ms（EXTINF 量化分辨率）。
        total = 0.0
        for uri, declared in zip(uris, extinfs):
            seg = out_dir / uri
            assert seg.is_file() and seg.stat().st_size > 0
            measured = _ffprobe_duration_ms(seg)
            assert abs(measured - declared) <= 0.001
            total += measured
            _assert_independently_decodable(seg)

        # 各片实测（视频轨口径）之和与源时长吻合：
        # 切割容差 20ms/片 + 每片 EXTINF 3 位小数量化 1ms。
        source_duration = result.plan.source_duration_ms / 1000.0
        assert abs(total - source_duration) <= 0.02 * len(uris) + 0.001 * len(uris)

        # 末片短尾标记与 EXTINF 准确性。
        last = result.results[-1]
        assert last.spec.is_short
        assert extinfs[-1] == round(last.actual_duration_ms / 1000, 3)
        assert extinfs[-1] < 2.0
        assert result.playlist.last_segment_short

    def test_extinf_sums_match_timeline_no_drift(self, video_factory,
                                                 tmp_path):
        # 长一些的素材（约 20s）验证多片下无累积漂移。
        src = video_factory(frames=500, fps=25)  # 20.0s，目标 3s
        out_dir = tmp_path / "hls"
        result = pack_hls(src, out_dir, target_seconds=3.0)
        _, _, extinfs, _ = _parse_m3u8(out_dir / "index.m3u8")
        assert len(extinfs) >= 6
        declared_total = sum(extinfs)
        source_total = result.plan.source_duration_ms / 1000.0
        # 声明总和与源时长差仅来自各片 3 位小数量化，单切片无漂移。
        assert abs(declared_total - source_total) <= 0.001 * len(extinfs)

    def test_with_audio(self, video_factory, tmp_path):
        src = video_factory(frames=75, fps=25, with_audio=True)  # 3s
        out_dir = tmp_path / "hls_audio"
        pack_hls(src, out_dir, target_seconds=1.0)
        _, _, extinfs, uris = _parse_m3u8(out_dir / "index.m3u8")
        assert len(uris) == 3
        for uri in uris:
            seg = out_dir / uri
            # 音频流确实被打包进去。
            cmd = [
                FFPROBE, "-v", "error", "-show_entries",
                "stream=codec_type", "-of", "json", str(seg),
            ]
            data = json.loads(
                subprocess.run(cmd, check=True, capture_output=True).stdout
            )
            types = {s["codec_type"] for s in data["streams"]}
            assert {"video", "audio"} <= types

    def test_playlist_byte_format(self, video_factory, tmp_path):
        src = video_factory(frames=50, fps=25)
        out_dir = tmp_path / "hls"
        pack_hls(src, out_dir, target_seconds=1.0)
        raw = (out_dir / "index.m3u8").read_bytes()
        # UTF-8、LF 行尾、无 BOM、以 ENDLIST 收尾。
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in raw
        assert raw.endswith(b"#EXT-X-ENDLIST\n")
        # 每个 EXTINF 都是整数秒 + 固定 3 位小数的浮点舍入格式。
        extinf_lines = [
            ln for ln in raw.split(b"\n") if ln.startswith(b"#EXTINF:")
        ]
        assert extinf_lines
        assert all(
            re.fullmatch(rb"#EXTINF:\d+\.\d{3},", ln) for ln in extinf_lines
        )

    def test_default_target_constant(self):
        assert DEFAULT_TARGET_SECONDS == 6.0
