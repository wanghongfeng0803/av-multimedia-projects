"""输出路径规划：镜像输入子目录 + 同批冲突裁决，全部是启动前的纯字符串计算。

规划阶段不启动任何子进程，因此可在无 ffmpeg 环境下完整单测。两个必须
处理的覆盖风险：

1. **输出目录位于输入目录内**：递归扫描时必须排除输出目录（由批量层
   传入 ``exclude_dirs``），否则上一轮产物会被当成新一轮输入反复转码；
2. **同批两个输入映射到同一输出**（如 ``a.mov`` 与 ``a.mp4`` 同名异源）：
   不论磁盘上是否已存在，都给后者自动加序号，避免任务间互相覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "EXIST_SKIP",
    "EXIST_OVERWRITE",
    "EXIST_SUFFIX",
    "PlannedJob",
    "plan_outputs",
    "mirror_output_path",
    "default_output_path",
]

#: 输出已存在策略：跳过 / 覆盖 / 自动改名加序号
EXIST_SKIP = "skip"
EXIST_OVERWRITE = "overwrite"
EXIST_SUFFIX = "suffix"

_EXIST_POLICIES = frozenset({EXIST_SKIP, EXIST_OVERWRITE, EXIST_SUFFIX})


@dataclass(frozen=True)
class PlannedJob:
    """单个任务的规划结果（不可变，可在线程间安全共享）。"""

    index: int
    input_path: str           # 绝对输入路径
    output_path: str          # 启动前规划好的绝对输出路径
    source_root: str | None   # 计算镜像相对路径时使用的输入根
    pre_exists: bool          # 规划时输出是否已存在（skip 策略据此跳过）


def plan_outputs(
    input_files: tuple[str, ...],
    input_roots: list[str],
    *,
    output_dir: str | None,
    extension: str,
    codec: str,
    on_exist: str = EXIST_SKIP,
) -> list[PlannedJob]:
    """把扫描出的输入文件规划成带输出路径的有序任务列表。

    :param input_files: :func:`bvp.scanner.scan_inputs` 产出的有序文件
    :param input_roots: 用户给定的原始输入根（文件或目录），用于镜像目录
    :param output_dir: 统一输出目录；None 时产物生成在各自输入旁
    :param extension: 目标容器扩展名（含点，如 ``.mp4``）
    :param codec: 编码规范名，参与默认命名 ``<名>_<编码><扩展名>``
    :param on_exist: 输出已存在策略
    :returns: 与输入同序的 :class:`PlannedJob` 列表
    """
    if on_exist not in _EXIST_POLICIES:
        raise ValueError(
            f"on_exist 必须是 {', '.join(sorted(_EXIST_POLICIES))} 之一，"
            f"收到 {on_exist!r}。"
        )

    roots = [os.path.abspath(r) for r in input_roots if r]
    used_outputs: set[str] = set()
    jobs: list[PlannedJob] = []

    for index, input_path in enumerate(input_files):
        source_root = _find_containing_root(input_path, roots)
        if output_dir is None:
            output_path = default_output_path(input_path, codec, extension)
        else:
            output_path = mirror_output_path(
                input_path, source_root, os.path.abspath(output_dir),
                codec, extension,
            )

        pre_exists = os.path.exists(output_path)

        # 同批输出冲突优先裁决：即使 on_exist=overwrite 也不能让两个任务
        # 写同一个文件，否则后写者覆盖先写者，成功数与产物对不平
        if output_path in used_outputs:
            output_path = _dedupe_path(output_path, used_outputs)
        used_outputs.add(output_path)

        if pre_exists and on_exist == EXIST_SUFFIX:
            output_path = _dedupe_path(output_path, used_outputs)
            used_outputs.add(output_path)
            pre_exists = False

        jobs.append(PlannedJob(
            index=index,
            input_path=input_path,
            output_path=output_path,
            source_root=source_root,
            pre_exists=pre_exists and on_exist == EXIST_SKIP,
        ))

    return jobs


def _find_containing_root(input_path: str, roots: list[str]) -> str | None:
    """找到包含该文件的输入根；文件独立给出时根就是其所在目录。"""
    for root in roots:
        if os.path.isfile(root):
            if os.path.abspath(root) == input_path:
                return os.path.dirname(root)
            continue
        # 路径分隔符边界判断，避免 /a/ab 误匹配 /a/a
        if input_path == root or input_path.startswith(root + os.sep):
            return root
    return os.path.dirname(input_path)


def default_output_path(input_path: str, codec: str,
                        extension: str) -> str:
    """输入旁默认命名 ``<名>_<编码><扩展名>``（含自覆盖保护）。"""
    stem = os.path.splitext(input_path)[0]
    default_path = os.path.abspath(f"{stem}_{codec}{extension}")
    if default_path == os.path.abspath(input_path):
        default_path = os.path.abspath(
            f"{stem}_transcoded{extension}"
        )
    return default_path


def mirror_output_path(
    input_path: str,
    source_root: str | None,
    output_dir: str,
    codec: str,
    extension: str,
) -> str:
    """按输入相对目录在输出目录中镜像同名文件并替换扩展名。

    纯字符串计算；Windows 跨盘符等无法计算相对路径时退化为平铺到
    输出目录，保证不会因为路径规划异常中断整批。
    """
    mirrored_dir = output_dir
    if source_root:
        try:
            relative = os.path.relpath(input_path, source_root)
            mirrored_dir = os.path.join(output_dir, os.path.dirname(relative))
        except ValueError:
            mirrored_dir = output_dir
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.abspath(
        os.path.join(mirrored_dir, f"{stem}_{codec}{extension}")
    )


def _dedupe_path(path: str, used: set[str]) -> str:
    """在 stem 后追加 _2、_3… 直到路径既未被本批占用也不存在于磁盘。"""
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    counter = 2
    while True:
        candidate = os.path.abspath(
            os.path.join(directory, f"{stem}_{counter}{ext}")
        )
        if candidate not in used and not os.path.exists(candidate):
            return candidate
        counter += 1
