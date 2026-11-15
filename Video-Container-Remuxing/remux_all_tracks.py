#!/usr/bin/env python3
"""全轨道 MP4 → MKV 重封装（含失败重试）。

在 :mod:`mp4_to_mkv` 包之上做一层薄编排，目标是**完整保留输入中的全部
轨道**——视频轨、所有音频轨、所有字幕轨（TTML / WebVTT）——逐字节无损地
写入同一个 MKV，且不引入任何新的依赖。

失败重试的边界（关键设计）：
- **瞬时性错误才重试**：文件被短暂锁定、杀软扫描占用、网络盘抖动等
  ``OSError``（读取 / 写入阶段）具有临时性，重试可能成功；
- **确定性错误不重试**：:class:`RemuxError` 表示输入本身不被支持或已损坏，
  同样的输入重试多少次结果都一样，立即失败并给出中文原因；
- 默认**流式直写**目标文件（不产生 ``.part`` 临时文件），写失败时由
  单次尝试自行清理半成品目标；``use_temp=True`` 可退回“临时文件 +
  ``os.replace`` 原子替换”，保证覆盖过程中绝不出现半成品。

命令行用法::

    python remux_all_tracks.py input.mp4 output.mkv [--overwrite]
                               [--retries N] [--retry-delay SECONDS]
                               [--temp]

Python 用法::

    from remux_all_tracks import remux_all_tracks, RemuxRetryError
    try:
        result = remux_all_tracks("in.mp4", "out.mkv", retries=3)
    except RemuxRetryError as exc:
        ...  # 重试用尽后仍失败
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 允许“脚本直接运行”与“作为模块导入”两种方式
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp4_to_mkv.errors import RemuxError  # noqa: E402
from mp4_to_mkv.mkv_writer import write_mkv, write_mkv_stream  # noqa: E402
from mp4_to_mkv.mp4_parser import parse_mp4  # noqa: E402

# 与包内 remuxer 保持一致的保护性内存上限
MAX_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # 秒，线性退避的基准间隔

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_RETRY_EXHAUSTED = 3
EXIT_OUT_OF_MEMORY = 4


class RemuxRetryError(Exception):
    """重试用尽后仍因瞬时性错误失败时抛出，``__cause__`` 保留最后一次异常。"""


@dataclass
class RemuxResult:
    """一次成功重封装的汇总信息。"""

    output_bytes: int
    track_count: int
    sample_count: int
    attempts: int  # 实际尝试次数（1 表示首次即成功）


def _remux_attempt(
    input_path: Path, output_path: Path, *, use_temp: bool = False
) -> tuple[int, int, int]:
    """执行单次重封装，返回 (写出字节数, track 数, 样本总数)。

    解析 / 写出过程中的 :class:`RemuxError` 原样上抛（确定性失败，不重试）；
    读写环节的 :class:`OSError` 也原样上抛，交由外层重试逻辑判定。

    两种写出策略：
    - ``use_temp=False``（默认）：流式直写目标路径，不产生临时文件，
      任何异常都在抛出前清理半成品目标文件；
    - ``use_temp=True``：写 ``.part`` 临时文件再 ``os.replace`` 原子替换。
    """
    # 读入前先按文件大小预检，避免超大文件先占满内存才报错。
    # stat 本身的 OSError（如文件瞬态不可访问）交由外层重试。
    size = input_path.stat().st_size
    if size == 0:
        raise RemuxError("输入文件为空（0 字节），不可能包含媒体数据")
    if size > MAX_INPUT_BYTES:
        raise RemuxError(
            f"输入文件 {size / 1024 / 1024:.1f} MiB 超过 "
            f"{MAX_INPUT_BYTES // 1024 // 1024} MiB 的内存处理上限"
        )

    # 读取阶段：网络盘 / 被占用文件可能抛出 OSError
    data = input_path.read_bytes()

    mp4 = parse_mp4(data)
    if not mp4.tracks:
        raise RemuxError("MP4 解析成功但未发现任何 track")
    total_samples = sum(len(t.samples) for t in mp4.tracks)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_temp:
        output_bytes = _write_via_temp(mp4, output_path)
    else:
        output_bytes = _write_streaming(mp4, output_path)

    return output_bytes, len(mp4.tracks), total_samples


def _write_streaming(mp4, output_path: Path) -> int:
    """流式直写：无临时文件，失败时清理半成品目标并原样上抛异常。"""
    try:
        with open(output_path, "wb") as out:
            return write_mkv_stream(mp4, out)
    except BaseException:
        # 保留原异常类型（RemuxError 确定性 / OSError 可重试），仅负责清理
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass
        raise


def _write_via_temp(mp4, output_path: Path) -> int:
    """原子模式：临时文件 + os.replace；失败清理 .part 后原样上抛。"""
    tmp = output_path.with_name(output_path.name + ".part")
    try:
        mkv = write_mkv(mp4)
        tmp.write_bytes(mkv)
        os.replace(tmp, output_path)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    return len(mkv)


def remux_all_tracks(
    input_path: str,
    output_path: str,
    *,
    overwrite: bool = False,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    use_temp: bool = False,
    sleep=time.sleep,
    attempt_fn=None,
) -> RemuxResult:
    """全轨道重封装，带瞬时性错误重试。

    :param retries: 失败后的**额外**重试次数（总尝试数 = retries + 1）。
    :param retry_delay: 线性退避的基准秒数，第 n 次重试前等待 n*retry_delay。
    :param use_temp: ``False``（默认）流式直写、不产生临时文件；
        ``True`` 走临时文件 + 原子替换（覆盖过程不出现半成品）。
    :param sleep: 可注入的等待函数，默认 :func:`time.sleep`（测试时可替换）。
    :param attempt_fn: 可注入的单次尝试函数（测试桩）。默认使用内置实现，
        签名为 ``attempt_fn(src, dst, *, use_temp=...)``；为兼容只接受
        ``(src, dst)`` 两个位置参数的旧式桩，调用时会自动探测其形参。
    :raises RemuxError: 确定性的输入 / 格式错误（不重试）。
    :raises RemuxRetryError: 瞬时性错误在所有尝试后仍失败。
    """
    if attempt_fn is None:
        attempt_fn = _remux_attempt

    src = Path(input_path)
    dst = Path(output_path)

    # 前置校验属于确定性错误，直接抛 RemuxError，不消耗重试次数
    if not src.exists():
        raise RemuxError(f"输入文件不存在: {input_path}")
    if not src.is_file():
        raise RemuxError(f"输入路径不是普通文件: {input_path}")
    if dst.parent.exists() and src.resolve() == dst.resolve():
        raise RemuxError("输出文件不能与输入文件为同一个路径")
    if dst.exists() and not overwrite:
        raise RemuxError(
            f"输出文件已存在: {output_path}（如需覆盖请使用 --overwrite）"
        )
    if retries < 0:
        raise RemuxError("retries 不能为负数")

    # 兼容只声明 (src, dst) 的旧式测试桩；内置实现与新式桩接收 use_temp。
    try:
        accepts_kw = "use_temp" in attempt_fn.__code__.co_varnames
    except AttributeError:
        accepts_kw = False

    def _invoke():
        if accepts_kw:
            return attempt_fn(src, dst, use_temp=use_temp)
        return attempt_fn(src, dst)

    tmp = dst.with_name(dst.name + ".part")
    attempts = 0
    last_os_error: OSError | None = None
    while attempts <= retries:
        attempts += 1
        try:
            output_bytes, track_count, sample_count = _invoke()
            return RemuxResult(
                # 内置实现返回字节数(int)；旧式桩可能返回字节串（len 即字节数），
                # 统一归一化为整数
                output_bytes=(
                    output_bytes if isinstance(output_bytes, int)
                    else len(output_bytes)
                ),
                track_count=track_count,
                sample_count=sample_count,
                attempts=attempts,
            )
        except RemuxError:
            # 确定性失败：重试无意义，直接上抛
            raise
        except OSError as exc:
            # 瞬时性失败。清理可能残留的半成品：
            # - 原子模式（或旧式桩）可能留下 .part；
            # - 内置流式尝试已自行删除半成品目标，这里仅作兜底，
            #   不主动删除自定义桩负责管理的目标文件。
            last_os_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempts <= retries:
                sleep(attempts * retry_delay)

    raise RemuxRetryError(
        f"重封装在 {attempts} 次尝试后仍失败: {last_os_error}"
    ) from last_os_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remux_all_tracks",
        description="把 MP4 全轨道（视频/音频/字幕）无损重封装为 MKV，支持失败重试",
    )
    parser.add_argument("input", help="输入 MP4 文件路径")
    parser.add_argument("output", help="输出 MKV 文件路径")
    parser.add_argument("--overwrite", action="store_true", help="若输出已存在则覆盖")
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"失败后的额外重试次数（默认 {DEFAULT_RETRIES}）",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"线性退避基准秒数（默认 {DEFAULT_RETRY_DELAY}）",
    )
    parser.add_argument(
        "--temp",
        dest="use_temp",
        action="store_true",
        help="改用临时文件 + 原子替换写出（默认流式直写，不产生临时文件）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = remux_all_tracks(
            args.input,
            args.output,
            overwrite=args.overwrite,
            retries=args.retries,
            retry_delay=args.retry_delay,
            use_temp=args.use_temp,
        )
    except RemuxError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except RemuxRetryError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return EXIT_RETRY_EXHAUSTED
    except MemoryError:
        print("错误：内存不足，输入文件超出当前处理能力", file=sys.stderr)
        return EXIT_OUT_OF_MEMORY

    print(
        f"完成：{args.input} -> {args.output}\n"
        f"  轨道数 {result.track_count}（视频/音频/字幕全保留），"
        f"样本数 {result.sample_count}，输出 {result.output_bytes} 字节，"
        f"尝试 {result.attempts} 次（全程未重编码）"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
