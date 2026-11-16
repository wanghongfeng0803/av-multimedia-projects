"""3D LUT（``.cube``）解析与三线性插值应用。

本模块把一张 3D LUT 建模为调色链路里的一个**独立处理节点**：
它消费 RGB 字节域（ffmpeg ``rgb24``，通道值 [0, 255]）的像素，
输出同一字节域的像素，因此可以和亮度/对比度 1D LUT、色相/饱和度
等其它节点按显式顺序串联（见 :mod:`color_grading.nodes`）。

.cube 格式（Adobe Cube LUT 子集）
--------------------------------

识别并容错处理：

- ``#`` 行注释、空行、CRLF / LF 混用、行首尾及 token 间多余空白；
- ``TITLE "..."``（可出现 0 或 1 次，引号可选，仅保存不参与运算）；
- ``LUT_3D_SIZE N``（必需，2 ≤ N ≤ 256）；
- 可选的 ``DOMAIN_MIN`` / ``DOMAIN_MAX``（见下方「域匹配」）；
- N×N×N 行 RGB 数据，**R 增长最快、G 次之、B 最慢**（Adobe 约定，
  即数据文件中第一行是网格点 (R=0,G=0,B=0)，最后一行是
  (R=N-1,G=N-1,B=N-1)）。

容错的边界：**宽容格式噪声，严格拒绝语义错误**。注释、空白、换行
风格、关键字大小写都容错；而尺寸声明缺失/重复、数据行数与
N³ 不符、token 无法解析为数值、出现 NaN/Inf、1D LUT
（``LUT_1D_SIZE``）等语义问题一律抛 :class:`LUTParseError`，
错误带文件名与行号，可直接定位。

域匹配（不匹配直接报错，不硬套）
----------------------------------

本管道的像素是 [0, 255] 整数、归一化到 [0, 1] 后查网格，因此只接受
**默认域**：未声明 ``DOMAIN_*``，或声明值恰好为单位域
（``DOMAIN_MIN 0 0 0``、``DOMAIN_MAX 1 1 1``）。任何带偏移/缩放/
负数的非单位域无法在字节域上无损套用，抛 :class:`LUTDomainError`；
尺寸（声明 N 与实际数据量）不符抛 :class:`LUTParseError`。

强度混合
--------

``strength`` ∈ [0, 1] 在**输出字节空间**对「原色」与「全量 LUT 结果」
做线性混合，两端边界精确成立：

- ``strength == 0``：节点恒等，输出与输入**逐字节相同**（在节点层
  与 worker 层都走跳过/直通，不产生浮点往返）；
- ``strength == 1``：输出与不带强度的全量 LUT 应用**逐字节相同**。

中间强度：``out = round(orig + (lut_out - orig) * strength)``，再钳到
[0, 255]。采用字节空间整数混合而非归一化浮点混合，正是为了让
``s=0`` 与 ``s=1`` 两个端点在 IEEE 舍入下也严格精确。

确定性
------

整个实现只用 ``math`` 等标准库、固定运算次序，不依赖字典迭代顺序、
环境变量或随机数；同一张 LUT、同一输入、同一强度，任意进程 / 任意
次数调用的输出逐字节一致。
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass

from .errors import LUTDomainError, LUTParseError

#: 允许的最大网格边长。256 已覆盖 8bit 全分辨率，再大没有字节域意义。
MAX_GRID_SIZE = 256

#: 允许的最小网格边长。1 无法构成插值区间。
MIN_GRID_SIZE = 2

#: 强度合法范围。
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.0

#: 单位域常量（逐通道）。
_DOMAIN_MIN_DEFAULT = (0.0, 0.0, 0.0)
_DOMAIN_MAX_DEFAULT = (1.0, 1.0, 1.0)


def clamp_strength(strength: float) -> tuple[float, list[str]]:
    """把强度钳制到 [0, 1]，策略与亮度/对比度钳制一致。

    超范围时发出 ``UserWarning`` 并把中文提示收集进返回列表，
    避免一次手滑中断整段批处理；范围内（含恰好 0 / 1）静默。

    :return: ``(钳制后的强度, 提示消息列表)``
    """
    messages: list[str] = []
    v = float(strength)
    if STRENGTH_MIN <= v <= STRENGTH_MAX:
        return v, messages
    clamped = STRENGTH_MIN if v < STRENGTH_MIN else STRENGTH_MAX
    msg = (
        f"参数 strength={v:g} 超出范围 "
        f"[{STRENGTH_MIN:g}, {STRENGTH_MAX:g}]，已钳制为 {clamped:g}"
    )
    warnings.warn(msg, UserWarning, stacklevel=2)
    messages.append(msg)
    return clamped, messages


@dataclass(frozen=True)
class CubeLUT:
    """一张已解析、已校验的 3D LUT。

    不可变（``frozen=True``），可安全在多进程间通过 initializer 分发、
    被多个节点 / 多次调用复用。

    存储约定：:attr:`data` 是长度 ``N**3 * 3`` 的 ``tuple[float, ...]``，
    按 **R 最快、G 次之、B 最慢** 的展平顺序存放归一化 RGB 值。
    网格点 ``(r, g, b)``（各为 [0, N) 的网格下标）的三元组起始下标为
    ``((b * N + g) * N + r) * 3``。
    """

    size: int
    data: tuple[float, ...]
    title: str = ""
    file_path: str = ""

    def __post_init__(self) -> None:
        if not (MIN_GRID_SIZE <= self.size <= MAX_GRID_SIZE):
            raise LUTParseError(
                f"LUT_3D_SIZE={self.size} 超出允许范围 "
                f"[{MIN_GRID_SIZE}, {MAX_GRID_SIZE}]",
                file_path=self.file_path,
            )
        expected = self.size ** 3 * 3
        if len(self.data) != expected:
            raise LUTParseError(
                f"网格数据长度 {len(self.data)} 与 size={self.size} "
                f"期望的 {expected} 个浮点不符",
                file_path=self.file_path,
            )

    def index_of(self, r: int, g: int, b: int) -> int:
        """网格点 (r, g, b) 的第一个通道（R）在 :attr:`data` 中的下标。"""
        n = self.size
        return ((b * n + g) * n + r) * 3

    def apply_pixel(self, r: int, g: int, b: int) -> tuple[int, int, int]:
        """对单个 8bit RGB 像素做三线性插值，返回 [0, 255] 整数三元组。

        纯函数参考实现（可读性优先）；worker 热路径使用
        :func:`apply_chunk` 内联的同一套数学，两者由测试锁定等价。
        """
        n = self.size
        scale = n - 1
        fr = r / 255.0 * scale
        fg = g / 255.0 * scale
        fb = b / 255.0 * scale
        r0 = min(int(fr), n - 2)
        g0 = min(int(fg), n - 2)
        b0 = min(int(fb), n - 2)
        tr, tg, tb = fr - r0, fg - g0, fb - b0
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

        def at(ri: int, gi: int, bi: int) -> tuple[float, float, float]:
            i = self.index_of(ri, gi, bi)
            d = self.data
            return d[i], d[i + 1], d[i + 2]

        def lerp(a: tuple[float, ...], q: tuple[float, ...], t: float):
            return tuple(a[k] + (q[k] - a[k]) * t for k in range(3))

        # 沿 R 插值得到 4 条棱
        c000, c100 = at(r0, g0, b0), at(r1, g0, b0)
        c010, c110 = at(r0, g1, b0), at(r1, g1, b0)
        c001, c101 = at(r0, g0, b1), at(r1, g0, b1)
        c011, c111 = at(r0, g1, b1), at(r1, g1, b1)
        e00 = lerp(c000, c100, tr)
        e10 = lerp(c010, c110, tr)
        e01 = lerp(c001, c101, tr)
        e11 = lerp(c011, c111, tr)
        # 沿 G
        f0 = lerp(e00, e10, tg)
        f1 = lerp(e01, e11, tg)
        # 沿 B
        out = lerp(f0, f1, tb)
        return (_round_byte(out[0]), _round_byte(out[1]), _round_byte(out[2]))


def _round_byte(value: float) -> int:
    """归一化浮点 [0,1]（允许轻微越界）四舍五入为 [0,255] 字节。

    与 :func:`color_grading.colorspace._round_byte` 同一确定性舍入：
    ``floor(v + 0.5)`` 半值向上，避免银行家舍入的方向歧义。
    """
    q = math.floor(value * 255.0 + 0.5)
    if q < 0:
        return 0
    if q > 255:
        return 255
    return q


def parse_cube(text: str, file_path: str = "") -> CubeLUT:
    """解析 ``.cube`` 文件文本为 :class:`CubeLUT`。

    :param text: 文件全文（调用方负责读取与解码）。
    :param file_path: 仅用于错误定位与结果记录的路径标签。
    :raises LUTParseError: 任何格式 / 尺寸 / 数据语义错误，带行号。
    :raises LUTDomainError: 声明了非单位域。
    """
    size: int | None = None
    title = ""
    domain_min: tuple[float, ...] | None = None
    domain_max: tuple[float, ...] | None = None
    values: list[float] = []
    expected_points: int | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # 容错：去掉行内注释（# 之后的内容）与首尾空白。
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].upper()  # 容错：关键字大小写不敏感

        if keyword == "TITLE":
            # TITLE 可带引号也可不带；剩余 token 重新拼回标题文本。
            title = _parse_title(line, lineno, file_path)
            continue
        if keyword == "LUT_3D_SIZE":
            if size is not None:
                raise LUTParseError("LUT_3D_SIZE 重复声明", file_path, lineno)
            size = _parse_single_int(tokens, "LUT_3D_SIZE", lineno, file_path)
            if not (MIN_GRID_SIZE <= size <= MAX_GRID_SIZE):
                raise LUTParseError(
                    f"LUT_3D_SIZE={size} 超出允许范围 "
                    f"[{MIN_GRID_SIZE}, {MAX_GRID_SIZE}]",
                    file_path, lineno,
                )
            expected_points = size ** 3
            continue
        if keyword == "LUT_1D_SIZE":
            # 明确拒绝 1D LUT：本节点只做 3D，不能拿 1D 硬套成 3D。
            raise LUTParseError(
                "不支持 LUT_1D_SIZE（1D LUT），本节点仅接受 3D LUT",
                file_path, lineno,
            )
        if keyword in ("DOMAIN_MIN", "DOMAIN_MAX"):
            triple = _parse_domain(tokens, keyword, lineno, file_path)
            if keyword == "DOMAIN_MIN":
                if domain_min is not None:
                    raise LUTParseError("DOMAIN_MIN 重复声明", file_path, lineno)
                domain_min = triple
            else:
                if domain_max is not None:
                    raise LUTParseError("DOMAIN_MAX 重复声明", file_path, lineno)
                domain_max = triple
            continue

        # 其余行一律按数据行处理（未知关键字也会在此因无法解析为
        # 数值而报错，不会被静默吞掉）。
        if len(tokens) != 3:
            raise LUTParseError(
                f"无法识别的行或数据列数不为 3（得到 {len(tokens)} 列）: "
                f"{line[:60]!r}",
                file_path, lineno,
            )
        if size is None:
            raise LUTParseError(
                "数据行出现在 LUT_3D_SIZE 声明之前", file_path, lineno
            )
        for tok in tokens:
            values.append(_parse_float(tok, lineno, file_path))

    if size is None:
        raise LUTParseError("缺少必需的 LUT_3D_SIZE 声明", file_path)

    # 域校验：只接受单位域（未声明，或声明确认为 0/1）。
    if domain_min is not None and domain_min != _DOMAIN_MIN_DEFAULT:
        raise LUTDomainError(
            f"不支持非默认 DOMAIN_MIN {domain_min}：本工具仅在归一化 "
            "[0,1] 域上工作，拒绝硬套，请改用单位域的 LUT",
        )
    if domain_max is not None and domain_max != _DOMAIN_MAX_DEFAULT:
        raise LUTDomainError(
            f"不支持非默认 DOMAIN_MAX {domain_max}：本工具仅在归一化 "
            "[0,1] 域上工作，拒绝硬套，请改用单位域的 LUT",
        )

    assert expected_points is not None
    got_points = len(values) // 3
    if len(values) != expected_points * 3:
        raise LUTParseError(
            f"网格数据量不匹配：LUT_3D_SIZE={size} 要求 {expected_points} 个 "
            f"RGB 点（{expected_points * 3} 个值），实际得到 {got_points} 个点 "
            f"（{len(values)} 个值）",
            file_path,
        )

    return CubeLUT(size=size, data=tuple(values), title=title, file_path=file_path)


def _parse_title(line: str, lineno: int, file_path: str) -> str:
    """解析 ``TITLE "..."`` / ``TITLE ...``，容错引号缺失。"""
    rest = line[len("TITLE"):].strip()
    if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
        return rest[1:-1]
    return rest


def _parse_single_int(
    tokens: list[str], name: str, lineno: int, file_path: str
) -> int:
    if len(tokens) != 2:
        raise LUTParseError(
            f"{name} 需要恰好 1 个整数参数（得到 {len(tokens) - 1} 个）",
            file_path, lineno,
        )
    try:
        return int(tokens[1])
    except ValueError:
        raise LUTParseError(
            f"{name} 参数不是整数: {tokens[1]!r}", file_path, lineno
        ) from None


def _parse_domain(
    tokens: list[str], name: str, lineno: int, file_path: str
) -> tuple[float, ...]:
    if len(tokens) != 4:
        raise LUTParseError(
            f"{name} 需要恰好 3 个浮点参数（得到 {len(tokens) - 1} 个）",
            file_path, lineno,
        )
    return tuple(_parse_float(tok, lineno, file_path) for tok in tokens[1:4])


def _parse_float(token: str, lineno: int, file_path: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise LUTParseError(
            f"无法解析为数值: {token!r}", file_path, lineno
        ) from None
    if not math.isfinite(value):
        raise LUTParseError(
            f"数值必须有限，不允许 NaN/Inf: {token!r}", file_path, lineno
        )
    return value


def load_cube(path: str) -> CubeLUT:
    """从磁盘读取并解析 ``.cube`` 文件。

    优先 UTF-8（含 BOM）解码，失败后回退系统默认编码；两种解码都
    失败时抛 :class:`LUTParseError`。读取/解析失败不会产生任何输出
    文件，调用方据此保证「失败保留原图」。

    :raises LUTParseError: 文件无法解码或内容不符合格式。
    :raises LUTDomainError: 域不匹配。
    """
    if not os.path.isfile(path):
        raise LUTParseError(f"LUT 文件不存在或不是普通文件: {path}")
    raw: bytes
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")  # 单字节编码永不抛错，兼容老式 .cube
        except Exception as exc:  # pragma: no cover - latin-1 理论上不会失败
            raise LUTParseError(f"文件无法解码为文本: {path}", path) from exc
    return parse_cube(text, file_path=path)


def apply_chunk(data: bytes, lut: CubeLUT, strength: float = 1.0) -> bytes:
    """对一块 rgb24 裸流应用 3D LUT（worker 热路径，模块级可 pickle）。

    :param data: RGB 字节序、长度为 3 的倍数的裸流。
    :param lut: 已解析的 :class:`CubeLUT`。
    :param strength: [0,1] 混合强度；``0`` 直接返回原字节（精确恒等），
        ``1`` 为全量 LUT（与无强度结果逐字节一致）。
    :raises ValueError: 数据长度不是 3 的倍数。
    """
    n = len(data)
    if n % 3:
        raise ValueError("rgb24 数据长度必须是 3 的倍数")
    if strength == 0.0:
        # 端点精确：强度 0 不做任何浮点往返，原样返回。
        return bytes(data)

    grid = lut.size
    scale = grid - 1
    table = lut.data
    floor = math.floor

    # 预计算每个输入字节的网格锚点与权重，避免逐像素重复乘除。
    # 三方向网格边长相同、输入域相同，故共用一套 lo/frac；
    # 另外预算各方向在展平数组中的字节步长偏移，角点下标只需三次相加。
    lo_tab = [0] * 256
    frac_tab = [0.0] * 256
    r_off = [0] * 256  # 低网格点在 R 方向的通道下标贡献 = lo * 3
    g_off = [0] * 256  # = lo * N * 3
    b_off = [0] * 256  # = lo * N * N * 3
    g_stride = grid * 3
    b_stride = grid * grid * 3
    for v in range(256):
        f = v / 255.0 * scale
        lo = int(f)
        if lo > grid - 2:
            lo = grid - 2  # 末字节 (f=N-1) 锚定到最后区间，权重恰为 1
        lo_tab[v] = lo
        frac_tab[v] = f - lo
        r_off[v] = lo * 3
        g_off[v] = lo * g_stride
        b_off[v] = lo * b_stride

    full = strength == 1.0
    s = strength
    out = bytearray(n)

    for i in range(0, n, 3):
        rv, gv, bv = data[i], data[i + 1], data[i + 2]
        tr, tg, tb = frac_tab[rv], frac_tab[gv], frac_tab[bv]

        # 8 个角点在展平 table 中的通道起始下标（R 最快、G 次之、B 最慢）。
        i000 = b_off[bv] + g_off[gv] + r_off[rv]
        i100 = i000 + 3
        i010 = i000 + g_stride
        i110 = i010 + 3
        i001 = i000 + b_stride
        i101 = i001 + 3
        i011 = i001 + g_stride
        i111 = i011 + 3

        # 三线性插值：先 R（4 棱），再 G（2 面），最后 B，逐通道。
        for channel, off in ((0, 0), (1, 1), (2, 2)):
            a000, a100 = table[i000 + off], table[i100 + off]
            a010, a110 = table[i010 + off], table[i110 + off]
            a001, a101 = table[i001 + off], table[i101 + off]
            a011, a111 = table[i011 + off], table[i111 + off]
            e00 = a000 + (a100 - a000) * tr
            e10 = a010 + (a110 - a010) * tr
            e01 = a001 + (a101 - a001) * tr
            e11 = a011 + (a111 - a011) * tr
            f0 = e00 + (e10 - e00) * tg
            f1 = e01 + (e11 - e01) * tg
            lut_value = f0 + (f1 - f0) * tb
            lut_byte = _round_byte(lut_value)
            if full:
                out[i + off] = lut_byte
            else:
                orig = (rv, gv, bv)[off]
                # 字节空间整数混合：s=0 精确原图、s=1 精确全量 LUT。
                blended = floor(orig + (lut_byte - orig) * s + 0.5)
                out[i + off] = 0 if blended < 0 else 255 if blended > 255 else blended
    return bytes(out)
