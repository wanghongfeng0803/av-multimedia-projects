"""ABR 端到端集成测试：真实 ffmpeg 打包后，独立核对主列表标注。

校验口径全部独立于被测代码自己的输出：

- 直接解析落盘 master.m3u8 的 EXT-X-STREAM-INF 标签；
- 对每路码率的每个分片单独跑 ffprobe 实测分辨率与时长；
- BANDWIDTH 用「分片文件字节数 × 8 ÷ 实测时长」独立重算比对；
- 各路码率的分片数、EXTINF 序列逐片比对，确认对齐。
"""

import json
import re
import subprocess

from tests.conftest import FFMPEG, FFPROBE, needs_ffmpeg

from vhls import pack_abr, pack_hls
from vhls.abr import Rendition


def _ffprobe_stream_info(path):
    """独立实测分片视频流的 (width, height, duration_s)。"""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "stream=codec_type,width,height,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(
        subprocess.run(cmd, check=True, capture_output=True).stdout
    )
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"]), float(s["duration"])
    raise AssertionError(f"分片无视频流：{path}")


def _parse_master(path):
    """解析主列表，返回 [(bandwidth, width, height, uri), ...]。"""
    entries = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:"):
            attrs = line.split(":", 1)[1]
            bw = int(re.search(r"BANDWIDTH=(\d+)", attrs).group(1))
            res = re.search(r"RESOLUTION=(\d+)x(\d+)", attrs)
            uri = lines[i + 1].strip()
            entries.append((bw, int(res.group(1)), int(res.group(2)), uri))
    return entries


def _parse_variant_playlist(path):
    """解析子列表，返回 (extinfs, uris)。"""
    extinfs, uris = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#EXTINF:"):
            extinfs.append(float(line.split(":", 1)[1].rstrip(",")))
        elif line and not line.startswith("#"):
            uris.append(line)
    return extinfs, uris


@needs_ffmpeg
class TestAbrEndToEnd:
    def test_master_annotations_independently_verifiable(
        self, video_factory, tmp_path
    ):
        # 640x360 源，两路码率：一路缩小、一路源分辨率。
        src = video_factory(
            name="in.mp4", size="640x360", frames=127, fps=25,
        )
        out_dir = tmp_path / "abr"
        result = pack_abr(
            src, out_dir,
            renditions=(
                Rendition(name="180p", height=180, video_bitrate=150_000),
                Rendition(name="360p", height=360, video_bitrate=400_000),
            ),
            target_seconds=2.0,
        )

        entries = _parse_master(out_dir / "master.m3u8")
        assert len(entries) == 2
        # 按 BANDWIDTH 升序。
        assert entries[0][0] <= entries[1][0]

        for bandwidth, width, height, uri in entries:
            sub_dir = out_dir / uri.split("/")[0]
            sub_playlist = out_dir / uri
            assert sub_playlist.is_file(), f"子列表缺失：{uri}"

            extinfs, seg_uris = _parse_variant_playlist(sub_playlist)
            assert len(extinfs) == len(seg_uris) == result.plan.segment_count

            # RESOLUTION 与分片实测分辨率一致（逐片独立 ffprobe）。
            peak_bps = 0
            for seg_uri, extinf in zip(seg_uris, extinfs):
                seg_path = sub_dir / seg_uri
                w, h, dur = _ffprobe_stream_info(seg_path)
                assert (w, h) == (width, height), (
                    f"主列表 RESOLUTION={width}x{height} 与分片实测 "
                    f"{w}x{h} 不一致"
                )
                # EXTINF 与实测时长一致（3 位小数量化容差 1ms）。
                assert abs(dur - extinf) <= 0.001
                # BANDWIDTH 用分片字节数独立重算。
                bps = int(seg_path.stat().st_size * 8 / dur + 0.5)
                peak_bps = max(peak_bps, bps)
            assert bandwidth == peak_bps, (
                f"主列表 BANDWIDTH={bandwidth} 与独立重算峰值 "
                f"{peak_bps} 不一致"
            )

    def test_variants_aligned_same_segments(self, video_factory, tmp_path):
        # 各路码率分片数一致、EXTINF 逐片对齐（不能多片/少片）。
        src = video_factory(
            name="in.mp4", size="640x360", frames=127, fps=25,
        )
        out_dir = tmp_path / "abr"
        result = pack_abr(
            src, out_dir,
            renditions=(
                Rendition(name="180p", height=180, video_bitrate=150_000),
                Rendition(name="360p", height=360, video_bitrate=400_000),
            ),
            target_seconds=2.0,
        )

        all_extinfs = []
        for v in result.variants:
            sub = out_dir / v.rendition.name / "index.m3u8"
            extinfs, uris = _parse_variant_playlist(sub)
            assert len(uris) == result.plan.segment_count
            all_extinfs.append(extinfs)

        # 两路 EXTINF 序列逐片一致（同一组帧切点）。
        assert len(all_extinfs) == 2
        for a, b in zip(*all_extinfs):
            assert abs(a - b) <= 0.001

    def test_no_upscaling_beyond_source(self, video_factory, tmp_path):
        # 请求 720p 但源只有 360p：不放大，RESOLUTION 写实测源分辨率。
        src = video_factory(
            name="in.mp4", size="640x360", frames=50, fps=25,
        )
        out_dir = tmp_path / "abr"
        result = pack_abr(
            src, out_dir,
            renditions=(
                Rendition(name="720p", height=720, video_bitrate=2_500_000),
            ),
            target_seconds=2.0,
        )
        entries = _parse_master(out_dir / "master.m3u8")
        assert len(entries) == 1
        _, width, height, _ = entries[0]
        assert (width, height) == (640, 360), (
            f"源 640x360 不应放大到 720p，实际 RESOLUTION={width}x{height}"
        )

    def test_default_ladder_generates_renditions(
        self, video_factory, tmp_path
    ):
        # 不指定阶梯时按源高度自动生成。
        src = video_factory(
            name="in.mp4", size="640x360", frames=50, fps=25,
        )
        out_dir = tmp_path / "abr"
        result = pack_abr(src, out_dir, target_seconds=2.0)
        assert len(result.variants) >= 1
        entries = _parse_master(out_dir / "master.m3u8")
        assert len(entries) == len(result.variants)

    def test_single_bitrate_still_compatible(self, video_factory, tmp_path):
        # 旧单码率接口行为不变：产物结构与之前一致。
        src = video_factory(frames=127, fps=25)
        out_dir = tmp_path / "hls"
        result = pack_hls(src, out_dir, target_seconds=2.0)
        m3u8 = out_dir / "index.m3u8"
        assert m3u8.is_file()
        text = m3u8.read_text(encoding="utf-8")
        assert "#EXT-X-STREAM-INF" not in text  # 单码率无主列表标签
        assert result.plan.segment_count == 3
        # 分片直接在输出目录（不在子目录）。
        assert (out_dir / "seg_00000.ts").is_file()

    def test_master_byte_format(self, video_factory, tmp_path):
        src = video_factory(
            name="in.mp4", size="640x360", frames=50, fps=25,
        )
        out_dir = tmp_path / "abr"
        pack_abr(
            src, out_dir,
            renditions=(
                Rendition(name="360p", height=360, video_bitrate=400_000),
            ),
            target_seconds=2.0,
        )
        raw = (out_dir / "master.m3u8").read_bytes()
        # UTF-8、LF 行尾、无 BOM。
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in raw
        # EXT-X-STREAM-INF 格式：BANDWIDTH=整数,RESOLUTION=宽x高。
        stream_lines = [
            ln for ln in raw.split(b"\n")
            if ln.startswith(b"#EXT-X-STREAM-INF:")
        ]
        assert stream_lines
        assert all(
            re.fullmatch(
                rb"#EXT-X-STREAM-INF:BANDWIDTH=\d+,RESOLUTION=\d+x\d+",
                ln,
            )
            for ln in stream_lines
        )
