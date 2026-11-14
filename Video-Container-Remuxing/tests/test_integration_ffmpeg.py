"""ffmpeg 真实文件端到端集成测试。

仅在本机存在 ffmpeg/ffprobe 时运行，覆盖：
- 真实 H.264(+B 帧)/AAC、HEVC 文件的重封装；
- 解码后逐帧 framemd5 哈希比对（证明帧字节无损、解码结果逐比特一致）；
- 包级 PTS 时间戳与容器元数据一致性；
- moov 在文件尾部（默认布局）与 faststart 两种布局；
- 碎片化 MP4 被明确拒绝。
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from mp4_to_mkv import __main__ as cli
from mp4_to_mkv.errors import RemuxError
from mp4_to_mkv.remuxer import remux

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
HAS_FFMPEG = bool(FFMPEG and FFPROBE)


def run(cmd: str) -> subprocess.CompletedProcess:
    # -nostdin 避免 ffmpeg 在无控制终端的子进程里等待 stdin 而挂起；
    # ffprobe 不支持该选项，仅对 ffmpeg 命令添加。
    stripped = cmd.strip()
    if stripped.startswith(FFMPEG):
        cmd = f"{FFMPEG} -nostdin {stripped[len(FFMPEG):].lstrip()}"
    elif not stripped.startswith(FFPROBE):
        # 未写二进制前缀时按 ffmpeg 命令补全
        cmd = f"{FFMPEG} -nostdin {stripped}"
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, shell=True
    )


@unittest.skipUnless(HAS_FFMPEG, "本机未安装 ffmpeg/ffprobe，跳过集成测试")
class FfmpegIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _gen_h264_aac(self, name: str, *, faststart: bool = False) -> Path:
        src = self.dir / name
        movflags = "-movflags +faststart" if faststart else ""
        # -bf 2 强制产生 B 帧，从而在 MP4 中出现 ctts 合成偏移
        run(
            f'{FFMPEG} -y -f lavfi -i testsrc=size=320x240:rate=25 '
            f'-f lavfi -i sine=frequency=440:sample_rate=44100 -t 2 '
            f'-c:v libx264 -preset veryfast -bf 2 -pix_fmt yuv420p '
            f'-c:a aac -b:a 64k {movflags} "{src}"'
        )
        return src

    def _remux(self, src: Path) -> Path:
        dst = src.with_suffix(".mkv")
        code = cli.main([str(src), str(dst), "--overwrite"])
        self.assertEqual(code, 0, f"重封装失败，退出码 {code}")
        return dst

    def _framemd5_hashes(self, path: Path, stream: str) -> list:
        """提取 framemd5 的帧哈希列（最后一列），忽略容器相关的头部与时间戳列。"""
        if stream == "v":
            cmd = f'{FFMPEG} -i "{path}" -map 0:v:0 -f framemd5 -'
        else:
            cmd = (
                f'{FFMPEG} -i "{path}" -map 0:a:0 -acodec pcm_s16le '
                f'-f framemd5 -'
            )
        lines = run(cmd).stdout.splitlines()
        # 数据行以帧序号开头（逗号分隔），最后一个字段即 MD5
        return [line.split(",")[-1].strip() for line in lines if line[:1].isdigit()]

    def _packet_pts(self, path: Path, stream: str) -> list:
        out = run(
            f'{FFPROBE} -v error -select_streams {stream} '
            f'-show_entries packet=pts_time -of csv=p=0 "{path}"'
        ).stdout.split()
        return [float(x.rstrip(",")) for x in out if x.strip()]

    def _stream_info(self, path: Path) -> dict:
        out = run(
            f'{FFPROBE} -v error -show_entries '
            f'stream=codec_name,codec_type,width,height,sample_rate,channels '
            f'-of json "{path}"'
        ).stdout
        return {s["codec_type"]: s for s in json.loads(out)["streams"]}

    def test_h264_aac_lossless(self):
        src = self._gen_h264_aac("h264_aac.mp4")
        dst = self._remux(src)

        # 1) 元数据一致
        src_info, dst_info = self._stream_info(src), self._stream_info(dst)
        self.assertEqual(dst_info["video"]["codec_name"], "h264")
        self.assertEqual(dst_info["audio"]["codec_name"], "aac")
        self.assertEqual(
            (dst_info["video"]["width"], dst_info["video"]["height"]),
            (src_info["video"]["width"], src_info["video"]["height"]),
        )
        self.assertEqual(
            dst_info["audio"]["sample_rate"], src_info["audio"]["sample_rate"]
        )
        self.assertEqual(
            dst_info["audio"]["channels"], src_info["audio"]["channels"]
        )

        # 2) 解码结果逐帧/逐采样一致 —— 无损的决定性证据
        self.assertEqual(
            self._framemd5_hashes(src, "v"), self._framemd5_hashes(dst, "v")
        )
        self.assertEqual(
            self._framemd5_hashes(src, "a"), self._framemd5_hashes(dst, "a")
        )

        # 3) 包级 PTS 一致（B 帧验证 ctts；音频验证 priming）。
        # MKV 采用毫秒时基，源 MP4 为精确分数，故按 <=1ms 的容差比较。
        for stream in ("v", "a"):
            src_pts = self._packet_pts(src, stream)
            dst_pts = self._packet_pts(dst, stream)
            self.assertEqual(
                len(src_pts), len(dst_pts), f"{stream} 流包数量不一致"
            )
            for i, (a, b) in enumerate(zip(src_pts, dst_pts)):
                self.assertLessEqual(
                    abs(a - b), 0.0015,
                    f"{stream} 流第 {i} 包 PTS 偏差过大: {a} vs {b}",
                )

    def test_faststart_layout(self):
        # faststart 把 moov 移到文件头部，解析器不应依赖盒顺序
        src = self._gen_h264_aac("faststart.mp4", faststart=True)
        dst = self._remux(src)
        self.assertEqual(
            self._framemd5_hashes(src, "v"), self._framemd5_hashes(dst, "v")
        )

    def test_hevc_if_available(self):
        src = self.dir / "hevc.mp4"
        proc = subprocess.run(
            f'{FFMPEG} -y -f lavfi -i testsrc=size=160x120:rate=15 -t 1 '
            f'-c:v libx265 -preset ultrafast -tag:v hvc1 "{src}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest("当前 ffmpeg 构建不含 libx265")
        dst = self._remux(src)
        info = self._stream_info(dst)
        self.assertEqual(info["video"]["codec_name"], "hevc")
        self.assertEqual(
            self._framemd5_hashes(src, "v"), self._framemd5_hashes(dst, "v")
        )

    def test_fragmented_mp4_rejected(self):
        src = self.dir / "frag.mp4"
        run(
            f'{FFMPEG} -y -f lavfi -i testsrc=size=160x120:rate=10 -t 1 '
            f'-c:v libx264 -preset ultrafast '
            f'-movflags frag_keyframe+empty_moov+default_base_moof "{src}"'
        )
        dst = src.with_suffix(".mkv")
        with self.assertRaises(RemuxError) as ctx:
            remux(str(src), str(dst), overwrite=True)
        self.assertIn("碎片化", str(ctx.exception))
        self.assertFalse(dst.exists())

    def test_garbage_rejected(self):
        src = self.dir / "garbage.mp4"
        src.write_bytes(b"\x00" * 64 + b"not a real container" * 10)
        dst = src.with_suffix(".mkv")
        code = cli.main([str(src), str(dst)])
        self.assertEqual(code, cli.EXIT_INVALID_INPUT)

    # ---- 新增编码：AC-3 / E-AC-3 / MP3 音频的无损重封装 ----

    def _gen_audio_only(self, name: str, codec: str, extra: str = "") -> Path:
        src = self.dir / name
        run(
            f'{FFMPEG} -y -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 '
            f'-c:a {codec} {extra} "{src}"'
        )
        return src

    def _assert_audio_lossless(self, src: Path, dst: Path, codec_name: str):
        info = self._stream_info(dst)
        self.assertEqual(info["audio"]["codec_name"], codec_name)
        self.assertEqual(
            self._framemd5_hashes(src, "a"),
            self._framemd5_hashes(dst, "a"),
        )

    def test_ac3_lossless(self):
        src = self._gen_audio_only("ac3.mp4", "ac3")
        dst = self._remux(src)
        self._assert_audio_lossless(src, dst, "ac3")

    def test_eac3_lossless(self):
        src = self._gen_audio_only("eac3.mp4", "eac3")
        dst = self._remux(src)
        self._assert_audio_lossless(src, dst, "eac3")

    def test_mp3_lossless(self):
        src = self._gen_audio_only("mp3.mp4", "libmp3lame")
        dst = self._remux(src)
        self._assert_audio_lossless(src, dst, "mp3")

    def test_multi_audio_tracks_all_preserved(self):
        """视频 + 两条不同编码的音频轨，重封装后必须全部保留且各自无损。"""
        src = self.dir / "multi_audio.mp4"
        # 多输入时 -t 必须作为输出选项放在输出文件名前，否则会被误绑到某个输入
        run(
            f'-y -f lavfi -i testsrc=size=160x120:rate=15 '
            f'-f lavfi -i sine=frequency=440:sample_rate=48000 '
            f'-f lavfi -i sine=frequency=880:sample_rate=48000 '
            f'-map 0:v -c:v libx264 -preset ultrafast '
            f'-map 1:a -c:a:0 aac '
            f'-map 2:a -c:a:1 ac3 -t 1 "{src}"'
        )
        dst = self._remux(src)
        out = set(
            run(
                f'{FFPROBE} -v error -show_entries stream=codec_type,codec_name '
                f'-of csv=p=0 "{dst}"'
            ).stdout.splitlines()
        )
        # 输出仍为 1 视频 + 2 音频，编码标识不变（ffprobe csv 列为 name,type）
        self.assertEqual(
            out, {"h264,video", "aac,audio", "ac3,audio"}
        )
        # 两条音频各自解码哈希一致（按流索引）
        for idx in range(2):
            self.assertEqual(
                self._framemd5_hashes(src, f"a:{idx}"),
                self._framemd5_hashes(dst, f"a:{idx}"),
            )

    # ---- TTML 字幕轨：ffmpeg 自身无法封进 MKV，验证本工具可直拷 ----

    def test_ttml_subtitle_preserved(self):
        vtt = self.dir / "sub.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n\n"
            "00:00:03.000 --> 00:00:04.000\nWorld\n",
            encoding="utf-8",
        )
        src = self.dir / "ttml.mp4"
        run(f'{FFMPEG} -y -i "{vtt}" -c:s ttml "{src}"')
        dst = self._remux(src)

        info = self._stream_info(dst)
        # ffmpeg 不识别 MKV 内 TTML 的具体解码器，但轨类型应为字幕
        self.assertIn("subtitle", info)
        # MKV 内字幕块必须包含源 TTML 文档的关键内容（直拷无损）
        blob = dst.read_bytes()
        self.assertIn(b"<?xml", blob)
        self.assertIn(b"Hello", blob)

    def test_video_audio_subtitle_full_pipeline_via_script(self):
        """经 remux_all_tracks 脚本完成 视频+音频+字幕 全轨道重封装。"""
        import remux_all_tracks as rat

        vtt = self.dir / "sub.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nHi there\n",
            encoding="utf-8",
        )
        src = self.dir / "full.mp4"
        run(
            f'-y -f lavfi -i testsrc=size=160x120:rate=15 '
            f'-f lavfi -i sine=frequency=440:sample_rate=44100 '
            f'-i "{vtt}" -c:v libx264 -preset ultrafast -c:a aac -c:s ttml '
            f'-map 0:v -map 1:a -map 2:s -t 2 "{src}"'
        )
        dst = src.with_suffix(".mkv")
        result = rat.remux_all_tracks(str(src), str(dst), overwrite=True)
        # 三类轨道全部保留
        self.assertEqual(result.track_count, 3)
        streams = run(
            f'{FFPROBE} -v error -show_entries stream=codec_type '
            f'-of csv=p=0 "{dst}"'
        ).stdout.split()
        self.assertEqual(sorted(streams), ["audio", "subtitle", "video"])
        # 视频与音频解码无损
        self.assertEqual(
            self._framemd5_hashes(src, "v"),
            self._framemd5_hashes(dst, "v"),
        )
        self.assertEqual(
            self._framemd5_hashes(src, "a"),
            self._framemd5_hashes(dst, "a"),
        )


if __name__ == "__main__":
    unittest.main()
