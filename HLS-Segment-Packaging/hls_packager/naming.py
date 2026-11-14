"""可配置的分片文件命名模板（纯逻辑、零 IO，便于单测）。

模板语法
--------
模板是一段普通字符串，支持以下占位符：

``{name}``
    文件名主名前缀（旧 ``filename_prefix`` 的泛化），默认 ``seg``；
``{index}``
    **运行内序号**：每个分片在本次打包中的 0 基序号，与 MEDIA-SEQUENCE
    起始值无关，永远从 0 递增；
``{sequence}``
    **绝对序号**：分片的 EXT-X-MEDIA-SEQUENCE 序号，等于
    ``sequence_start + index``；
``{ext}``
    容器扩展名（不含点，由容器规格提供，如 ``ts`` / ``m4s``）。

序号占位符只接受**零填充格式说明符**：``{index:04d}``、``{sequence:05d}``
或不写宽度的 ``{index}``；``{index:>5}``、``{index:.2f}`` 等非零填充
写法一律拒绝。这一限制与 ffmpeg segment muxer 的输出模式串
（``%04d`` 风格）能力对齐——模板最终要翻译成 ffmpeg 的计数器模式，
任意格式说明无法翻译。

一个模板中 ``{index}`` / ``{sequence}`` 有且只能出现一次（ffmpeg 只有
一个输出计数器）；``{name}``、``{ext}`` 各至多一次，且必须出现
``{ext}`` 或在外部显式提供扩展名（由 :class:`~hls_packager.containers.ContainerSpec`
注入），保证产出文件名带正确后缀。

默认模板 ``{name}_{sequence:04d}.{ext}`` 与历史输出 ``seg_0000.ts``
逐字节一致，因此不指定模板时旧调用方行为完全不变。

边界处理
--------
* 模板内禁止路径分隔符（``/``、``\\``）、Windows 保留字符
  （``<>:"|?*``）与控制字符，避免分片逃逸输出目录；
* 字面 ``%`` 被拒绝：它是 ffmpeg 模式串的转义前缀，混入会破坏
  计数器翻译；
* :meth:`NamingConfig.render_many` 会实测渲染全部分片并检测重名，
  冲突（例如把序号占位符写漏导致所有分片同名）直接报错，而非
  静默互相覆盖。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .errors import NamingError

#: 旧版本的文件名形态，作为默认模板保证逐字节兼容
DEFAULT_NAME_TEMPLATE = "{name}_{sequence:04d}.{ext}"
DEFAULT_NAME_PREFIX = "seg"

# 单个花括号占位符：name/index/sequence/ext，可选格式说明
_PLACEHOLDER_RE = re.compile(r"\{(\w+)(?::([^{}]*))?\}")
# 序号占位符允许的格式说明：只允许 0 填充宽度 + 可选 d，如 04 / 06d
_ZERO_PAD_SPEC_RE = re.compile(r"^0(\d+)d?$")
# 文件名禁用字符：Windows 保留集 + 路径分隔符（POSIX 上同样不允许）
_FORBIDDEN_CHARS = set('<>:"/\\|?*')
# 数字占位符（ffmpeg 输出计数器只能有一个）
_NUMERIC_FIELDS = ("index", "sequence")
# 模板中全部合法字段
_ALL_FIELDS = frozenset(("name", "index", "sequence", "ext"))


@dataclass(frozen=True)
class NamingConfig:
    """分片命名模板配置（不可变）。

    :param template: 含占位符的模板字符串；
    :param name_prefix: ``{name}`` 的替换值；
    :param extension: ``{ext}`` 的替换值（不含点），由容器规格注入；
        模板本身带字面后缀、不含 ``{ext}`` 时可为空串。
    """

    template: str = DEFAULT_NAME_TEMPLATE
    name_prefix: str = DEFAULT_NAME_PREFIX
    extension: str = "ts"

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """校验模板与替换值，非法时抛 :class:`NamingError`。"""
        template = self.template
        if not template or not template.strip():
            raise NamingError("命名模板不能为空")
        if "%" in template:
            raise NamingError(
                f"命名模板不能包含字面 %（与 ffmpeg 计数器模式冲突）：{template!r}"
            )

        numeric_count = 0
        seen_fields: set[str] = set()
        cursor = 0
        for match in _PLACEHOLDER_RE.finditer(template):
            # 占位符之间的字面片段做安全检查
            self._check_literal(template[cursor:match.start()])
            cursor = match.end()
            field, spec = match.group(1), match.group(2)
            if field not in _ALL_FIELDS:
                raise NamingError(
                    f"命名模板含未知占位符 {{{field}}}；"
                    f"合法字段：{', '.join(sorted(_ALL_FIELDS))}"
                )
            if field in seen_fields:
                raise NamingError(f"占位符 {{{field}}} 在模板中重复出现")
            seen_fields.add(field)

            if field in _NUMERIC_FIELDS:
                numeric_count += 1
                if spec is not None and not _ZERO_PAD_SPEC_RE.fullmatch(spec):
                    raise NamingError(
                        f"序号占位符 {{{field}}} 只支持零填充格式（如 :04d），"
                        f"实际格式说明为 {spec!r}"
                    )
            elif spec is not None:
                raise NamingError(
                    f"占位符 {{{field}}} 不接受格式说明，实际为 {spec!r}"
                )
        self._check_literal(template[cursor:])  # 末尾字面片段

        if numeric_count != 1:
            raise NamingError(
                "命名模板必须恰好包含一个序号占位符 {index} 或 {sequence}，"
                f"实际包含 {numeric_count} 个"
            )
        if "name" not in seen_fields:
            raise NamingError("命名模板必须包含 {name} 前缀占位符")

        for label, value in (("name_prefix", self.name_prefix),
                             ("extension", self.extension)):
            if not isinstance(value, str) or any(
                ch in _FORBIDDEN_CHARS or ord(ch) < 0x20 for ch in value
            ):
                raise NamingError(
                    f"{label} 含禁用字符或控制字符：{value!r}"
                )

    @staticmethod
    def _check_literal(text: str) -> None:
        """检查模板中的字面文本（非占位符部分）。"""
        # 游离的单花括号不会被占位符正则匹配，留给 str.format 只会在
        # 渲染期抛 ValueError——这里提前到构造期统一报 NamingError
        if "{" in text or "}" in text:
            raise NamingError(
                "命名模板含未配对的花括号（占位符须写为 {name}/"
                "{index}/{sequence}/{ext} 形式）"
            )
        for ch in text:
            if ch in _FORBIDDEN_CHARS:
                raise NamingError(
                    f"命名模板含文件名禁用字符 {ch!r}（不允许路径分隔符与"
                    " <> : \" | ? *）"
                )
            if ord(ch) < 0x20:
                raise NamingError("命名模板含控制字符")

    # ------------------------------------------------------------------ 渲染
    def render(self, index: int, sequence: int) -> str:
        """渲染单个分片的文件名（仅文件名，不含目录）。"""
        if index < 0 or sequence < 0:
            raise NamingError("序号不能为负数")
        values = {
            "name": self.name_prefix,
            "index": index,
            "sequence": sequence,
            "ext": self.extension,
        }
        try:
            rendered = self.template.format(**values)
        except (IndexError, ValueError, KeyError) as exc:
            # validate 已拦截大部分问题，这里兜住格式化层面的残余异常
            raise NamingError(f"命名模板渲染失败：{exc}") from exc
        if not rendered or rendered in (".", ".."):
            raise NamingError("命名模板渲染出空文件名或保留名")
        return rendered

    def render_many(self, count: int, sequence_start: int) -> list[str]:
        """渲染 ``count`` 个分片名并检测重名。

        :param count: 分片数量；
        :param sequence_start: 首片的绝对序号（``{sequence}`` 起点）。
        """
        if count <= 0:
            raise NamingError("分片数量必须为正数")
        names: list[str] = []
        seen: set[str] = set()
        for index in range(count):
            name = self.render(index, sequence_start + index)
            if name in seen:
                raise NamingError(
                    f"命名模板渲染出重复文件名 {name!r}（请确认模板中"
                    "包含且仅包含一个序号占位符）"
                )
            seen.add(name)
            names.append(name)
        return names

    # ------------------------------------------------------ ffmpeg 模式翻译
    @property
    def numeric_field(self) -> str:
        """模板使用的数字占位符字段名（``index`` 或 ``sequence``）。"""
        match = _PLACEHOLDER_RE.search(self.template)
        while match is not None:
            if match.group(1) in _NUMERIC_FIELDS:
                return match.group(1)
            match = _PLACEHOLDER_RE.search(self.template, match.end())
        # validate 已保证必有一个数字占位符
        raise NamingError("模板缺少序号占位符")  # pragma: no cover

    def ffmpeg_pattern(
        self, output_dir: str, sequence_start: int
    ) -> tuple[str, int]:
        """把模板翻译为 ffmpeg segment muxer 的输出模式串与起始序号。

        ffmpeg 只有一个输出计数器，因此：

        * ``{index}`` 模式起始序号恒为 0（运行内序号）；
        * ``{sequence}`` 模式起始序号为 ``sequence_start``（绝对序号）。

        :param sequence_start: 首片的 EXT-X-MEDIA-SEQUENCE 序号；
        :returns: ``(输出模式串, ffmpeg -segment_start_number)``。
        """
        numeric = self.numeric_field
        tokens: list[str] = []
        cursor = 0
        for match in _PLACEHOLDER_RE.finditer(self.template):
            tokens.append(self.template[cursor:match.start()])
            field, spec = match.group(1), match.group(2)
            if field in _NUMERIC_FIELDS:
                width = int(_ZERO_PAD_SPEC_RE.fullmatch(spec).group(1)) \
                    if spec else 0
                # %0Nd：N 为最小位宽；N=0 时用裸 %d（不补零）
                tokens.append(f"%0{width}d" if width else "%d")
            elif field == "name":
                tokens.append(self.name_prefix)
            else:  # ext
                tokens.append(self.extension)
            cursor = match.end()
        tokens.append(self.template[cursor:])
        pattern = "".join(tokens)
        start_number = sequence_start if numeric == "sequence" else 0
        return os.path.join(output_dir, pattern), start_number
