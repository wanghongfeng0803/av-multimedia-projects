"""真实 ffmpeg 端到端集成测试（无 ffmpeg/ffprobe 时整文件自动 skip）。

覆盖从「真实磁盘输入 → 核心转码 → 批量清单 → 对账」的完整链路，与纯逻辑
注入测试互补：这里的假数据全部由 ffmpeg 真实生成，错误分类也由核心真实抛出。
"""

from __future__ import annotations

import os

from bvp.batch import batch_transcode
from bvp.contract import ErrorCategory, SkipReason
from bvp.core import transcode
from bvp.errors import InvalidInputError
from bvp.manifest import STATUS_SKIPPED, STATUS_SUCCESS
from bvp.probe import probe_media
from bvp.verify import reconcile_manifest_files

from conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def test_core_transcodes_real_video_and_verifies(tmp_path, video_factory):
    src = video_factory(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out" / "src_h264.mp4")

    result = transcode(src, out, codec="h264", container="mp4")

    assert result.success
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    info = probe_media(out)
    assert info.primary_video.codec_name == "h264"
    # 成功后不留 ffmpeg 日志
    assert not os.path.exists(out + ".ffmpeg.log")


def test_core_rejects_disguised_non_video(tmp_path):
    fake = str(tmp_path / "pretend.mp4")
    with open(fake, "w", encoding="utf-8") as handle:
        handle.write("this is not a video" * 50)
    try:
        transcode(fake, str(tmp_path / "o.mp4"))
    except InvalidInputError as exc:
        assert exc.category is ErrorCategory.NOT_A_VIDEO
    else:  # pragma: no cover - 真实 ffprobe 必须拒绝
        raise AssertionError("伪装文件应当触发 InvalidInputError")


def test_core_failure_cleans_partial_output(tmp_path, video_factory,
                                            monkeypatch):
    src = video_factory(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "o.mp4")
    # 强制 ffmpeg 失败：把可执行文件指向不存在的路径，进程启动即失败
    monkeypatch.setenv("FFMPEG", str(tmp_path / "no_such_ffmpeg.exe"))
    try:
        transcode(src, out)
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("应当抛出错误")
    # 半成品/输出绝不能残留在磁盘上冒充成功产物
    assert not os.path.exists(out)


def test_batch_end_to_end_mixed_inputs(tmp_path, video_factory):
    in_root = tmp_path / "in"
    out_dir = tmp_path / "out"
    # 两个真实视频（其中一个在嵌套目录）
    video_factory(str(in_root / "a.mp4"))
    video_factory(str(in_root / "sub" / "b.mov"))
    # 伪装的非视频
    disguised = str(in_root / "fake.mp4")
    with open(disguised, "w", encoding="utf-8") as handle:
        handle.write("plain text, not media" * 30)
    # 一个不存在的输入路径
    ghost = str(in_root / "ghost.mp4")

    result = batch_transcode(
        [str(in_root), ghost],
        output_dir=str(out_dir),
        codec="h264",
        container="mp4",
    )

    summary = result.summary
    assert summary["total"] == 4
    assert summary["success"] == 2
    assert summary["skipped"] == 1   # 伪装非视频
    assert summary["failed"] == 1    # 不可访问路径

    # 成功产物真实存在且镜像了嵌套目录
    assert os.path.isfile(str(out_dir / "a_h264.mp4"))
    assert os.path.isfile(str(out_dir / "sub" / "b_h264.mp4"))

    # 跳过条目分类正确
    skip = result.manifest.skipped_entries[0]
    assert skip.status == STATUS_SKIPPED
    assert skip.skip_reason == SkipReason.NON_VIDEO.value
    assert skip.category == ErrorCategory.NOT_A_VIDEO.value

    # 失败条目为不可访问
    failed = result.manifest.failed_entries[0]
    assert failed.category == ErrorCategory.INPUT_NOT_FOUND.value

    # 用独立的文件级复核再对一次账（脱离运行上下文）
    disk_check = reconcile_manifest_files(str(out_dir))
    assert disk_check.ok
    assert disk_check.produced_outputs == 2


def test_batch_second_run_skips_existing_outputs(tmp_path, video_factory):
    src = video_factory(str(tmp_path / "in" / "clip.mp4"))
    out_dir = str(tmp_path / "out")

    first = batch_transcode(
        [str(tmp_path / "in")], output_dir=out_dir,
    )
    assert first.summary["success"] == 1

    second = batch_transcode(
        [str(tmp_path / "in")], output_dir=out_dir,
    )
    # 第二轮默认 skip：输出已存在，记为跳过而非重新转码
    assert second.summary["success"] == 0
    assert second.summary["skipped"] == 1
    entry = second.manifest.skipped_entries[0]
    assert entry.skip_reason == SkipReason.OUTPUT_EXISTS.value
    assert second.reconciliation.ok


def test_batch_output_dir_inside_input_not_rescanned(tmp_path, video_factory):
    root = tmp_path / "media"
    video_factory(str(root / "clip.mp4"))
    out_dir = str(root / "transcoded")

    result = batch_transcode(
        [str(root)], output_dir=out_dir,
    )
    # 输出目录在输入目录内：只能有一个任务，产物不能被当新输入
    assert result.manifest.total == 1
    assert result.summary["success"] == 1
    assert os.path.isfile(os.path.join(out_dir, "clip_h264.mp4"))


def test_batch_h265_mkv_real_transcode(tmp_path, video_factory):
    src = video_factory(str(tmp_path / "src.mp4"))
    result = batch_transcode(
        [src], output_dir=str(tmp_path / "out"),
        codec="h265", container="mkv",
    )
    assert result.summary["success"] == 1
    out = result.manifest.success_entries[0].output_path
    assert out.endswith(".mkv")
    info = probe_media(out)
    assert info.primary_video.codec_name == "hevc"
    assert result.reconciliation.ok


def test_deterministic_manifest_order_with_real_files(
    tmp_path, video_factory
):
    # 输入与输出放在平级目录，避免输出目录位于输入目录内导致产物被重扫
    in_root = tmp_path / "in"
    for name in ("z.mp4", "a.mp4", "m.mp4"):
        video_factory(str(in_root / name))
    r1 = batch_transcode(
        [str(in_root)], output_dir=str(tmp_path / "out1"),
    )
    r2 = batch_transcode(
        [str(in_root)], output_dir=str(tmp_path / "out2"),
    )
    bases1 = [os.path.basename(e.input_path) for e in r1.manifest.entries]
    bases2 = [os.path.basename(e.input_path) for e in r2.manifest.entries]
    assert bases1 == bases2 == ["a.mp4", "m.mp4", "z.mp4"]
    assert all(e.status == STATUS_SUCCESS for e in r1.manifest.entries)
