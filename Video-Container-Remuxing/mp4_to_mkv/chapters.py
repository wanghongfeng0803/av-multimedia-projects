"""ISO-BMFF 章节（chapter）元数据的字节级解析。

MP4/MOV 容器的章节有两套并存的表达：

1. **QuickTime 章节轨**：一条 ``text``/``sbtl`` handler 的 track，章节标题
   作为文本样本，由 ``trak`` 的章节引用（``chap``）关联到其它轨道——属于
   媒体轨范畴，本项目基础版本不支持文本轨；
2. **``chpl`` 章节清单盒**（Apple/ISO 风格，本模块解析对象）：位于
   ``moov/udta/meta/chpl`` 或直接 ``moov/udta/chpl``，是一份扁平的
   (起始时间, 标题) 清单，结构为::

       chpl (FullBox: version/flags；ffmpeg 实际写 version=1)
         reserved        1 字节（恒 0）
         chapter_count   uint32 BE
         chapter × N:
           start_time    uint64 BE   —— 单位为 mvhd 的 movie timescale
           title_length  uint8 BE    —— 单字节 Pascal 风格长度
           title         UTF-8 字节（长度即 title_length，最长 255）

   注意标题长度字段是 **1 字节 uint8**：这是 ffmpeg / AtomicParsley /
   GPAC 等主流封装器共同遵循的事实布局（早期资料偶有 uint16 误传，
   会把 ``05 49`` 这类“长度+首字符”误读成 1353）。

本模块的设计目标是**在任何输入下都给出健壮、确定、可复现的结果**：

- **多层嵌套**：实际文件中 ``meta`` 可能再裹 ``gnre``/``ilst`` 容器，
  不同封装器放置 ``chpl`` 的层级深浅不一，因此采用有界深度优先搜索在
  moov 子树内定位全部 ``chpl`` 盒，而非硬编码单一路径；
- **损坏与不规则**：盒长度越界时按容器边界 salvage、``chpl`` 负载截断时
  按可用字节恢复条目、声明计数与实际不符时以能解析出的条目为准、
  非法 UTF-8 以替换字符解码、空标题回退为确定性占位名；
- **绝不拖垮主流程**：章节属于附属元数据，任何解析异常都收敛为
  “部分结果 + 诊断信息”，不抛出异常，媒体轨道的重封装不受影响；
- **零拷贝与有界开销**：复用 :mod:`mp4_parser` 已读入内存的盒树与字节
  切片，不复制文件数据；遍历节点数、``chpl`` 负载大小与章节数量均有
  保护性上限，防止恶意/损坏文件引发超线性开销。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from struct import unpack
from typing import TYPE_CHECKING, List, Optional, Tuple

from . import mp4_parser
from .errors import RemuxError

if TYPE_CHECKING:  # 仅类型检查期生效，规避运行期循环导入
    from .mp4_parser import Box

__all__ = ["Chapter", "extract_chapters"]

# ---- 保护性上限（均远高于真实文件的合理规模，仅用于防失控）--------------
_MAX_TRAVERSE_NODES = 200_000   # 盒树 DFS 访问节点上限
_MAX_CHPL_PAYLOAD = 64 * 1024 * 1024  # 单个 chpl 负载 64 MiB 上限
_MAX_CHAPTERS = 100_000         # 单个/聚合后的章节总数上限
_MAX_DFS_DEPTH = 16             # 最大递归深度（真实嵌套不超过个位数）


@dataclass(frozen=True)
class Chapter:
    """一个章节点。时间在解析阶段即换算为 Matroska 单位（纳秒）。"""

    start_ns: int       # 章节起始时间（纳秒，非负）
    title: str          # 章节标题（已做 UTF-8 容错与空值回退）
    warnings: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 容错的盒树遍历
# ---------------------------------------------------------------------------

def _lenient_children(box: Box) -> List[Box]:
    """解析子盒；同级序列损坏时重同步，尽量恢复后续兄弟盒。

    :mod:`mp4_parser` 的 :func:`parse_boxes` 采用严格策略：任一盒长度
    越界即抛 :class:`RemuxError`。章节元数据位于 ``udta`` 这类厂商自由
    扩展区，出现脏字节的概率高，这里在严格解析失败后退化为扫描式
    重同步——从损坏点之后寻找下一个“长度自洽”的盒头继续解析，
    使损坏点之后的 ``chpl`` 仍有机会被找到。
    """
    try:
        return box.children()
    except RemuxError:
        pass
    return _resync_boxes(box.data, box.data_start)


def _resync_boxes(buf: bytes, base_offset: int) -> List[Box]:
    """在损坏的同级盒缓冲区中扫描恢复合法盒。

    盒头布局为 ``size(4) + type(4)``：四字符码位于起始位置 +4。
    判定一个位置是合法盒头的条件：四字符码可打印、声明长度自洽
    （不短于盒头且不越出缓冲区）。扫描步长为 1，仅在严格解析失败的
    错误路径上执行，正常文件不会进入此函数。
    """
    recovered: List[Box] = []
    n = len(buf)
    pos = 0
    while pos + 8 <= n:
        kind = _try_printable_fourcc(buf, pos + 4)
        if kind is None:
            pos += 1
            continue
        size32 = unpack(">I", buf[pos:pos + 4])[0]
        header_size = 8
        size = size32
        if size32 == 1:  # largesize
            if pos + 16 > n:
                pos += 1
                continue
            size = unpack(">Q", buf[pos + 8:pos + 16])[0]
            header_size = 16
        elif size32 == 0:
            size = n - pos
        if size < header_size or pos + size > n:
            pos += 1
            continue
        recovered.append(
            mp4_parser.Box(
                kind, base_offset + pos, header_size,
                buf[pos + header_size:pos + size],
            )
        )
        pos += size
    return recovered


def _try_printable_fourcc(buf: bytes, pos: int) -> Optional[str]:
    """若 buf[pos:pos+4] 是可打印 ASCII（含空格）四字符码则返回，否则 None。"""
    if pos + 4 > len(buf):
        return None
    raw = buf[pos:pos + 4]
    if all(0x20 <= b < 0x7F for b in raw):
        return raw.decode("latin-1")
    return None


def _meta_payload_children(meta: Box) -> List[Box]:
    """解析 ``meta`` 盒的子盒，兼容 FullBox 与普通 Box 两种封装。

    ISO 14496-12 规定 ``meta`` 是 FullBox（负载前 4 字节为 version/flags）；
    但历史上不少封装器（尤其 QuickTime 生态的 ``udta/meta``）把它写成
    无 fullbox 头的普通容器。两种布局解析出的子盒序列可能不同，这里
    分别尝试并按“识别出的有效盒数量 + 是否含 chpl”打分择优。
    """
    candidates: List[List[Box]] = []
    # 布局 A：跳过 4 字节 FullBox 头
    if len(meta.data) >= 4:
        candidates.append(_lenient_parse(meta.data[4:], meta.data_start + 4))
    # 布局 B：整个负载即子盒序列
    candidates.append(_lenient_parse(meta.data, meta.data_start))

    def score(children: List[Box]) -> Tuple[int, int]:
        return (
            1 if any(c.kind == "chpl" for c in children) else 0,
            len(children),
        )

    return max(candidates, key=score)


def _lenient_parse(buf: bytes, base_offset: int) -> List[Box]:
    """带重同步兜底的同级盒解析。"""
    try:
        return mp4_parser.parse_boxes(buf, base_offset=base_offset)
    except RemuxError:
        return _resync_boxes(buf, base_offset)


# ---------------------------------------------------------------------------
# chpl 盒定位（有界 DFS）
# ---------------------------------------------------------------------------

def _find_chpl_boxes(root: Box) -> List[Box]:
    """在盒树中有界深度优先搜索全部 ``chpl`` 盒（文档顺序）。

    开销控制：``udta`` 是厂商自由扩展区，负载小且最易损坏，其整棵
    子树启用“严格解析 + 逐字节重同步”的完整容错；``udta`` 之外（样本
    表等可能达数 MB 的盒）只做**严格探测**，解析失败即视为叶子，
    避免对大负载执行 O(n) 重同步扫描。
    """
    found: List[Box] = []
    nodes = 0

    def walk(box: Box, depth: int, lenient: bool) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_TRAVERSE_NODES or depth > _MAX_DFS_DEPTH:
            return
        if box.kind == "chpl":
            found.append(box)
            # chpl 是叶子数据盒，不再下钻（也防止把其数据误解析为盒）
            return

        if box.kind == "meta":
            children = _meta_payload_children(box)
        elif lenient:
            children = _lenient_children(box)
        else:
            children = _strict_children(box)

        # 进入 udta / meta 后，整棵子树切换到完整容错模式：
        # 二者都是元数据区，负载小且厂商变体多，重同步成本可忽略
        child_lenient = lenient or box.kind in ("udta", "meta")
        for child in children:
            walk(child, depth + 1, child_lenient)

    walk(root, 0, False)
    return found


def _strict_children(box: Box) -> List[Box]:
    """严格子盒解析：任何结构异常都返回空列表（把该盒视为叶子）。"""
    try:
        return box.children()
    except RemuxError:
        return []


# ---------------------------------------------------------------------------
# chpl 负载的字节级解析
# ---------------------------------------------------------------------------

def _parse_chpl_payload(payload: bytes) -> Tuple[List[Tuple[int, str]], List[str]]:
    """解析单个 chpl 盒负载，返回 (原始条目 [(start_time, title)], 诊断)。

    兼容两种字节布局：
    - 标准：FullBox 头(4) + reserved(1) + count(4) + 条目；
    - 无 FullBox 头的变体：reserved(1) + count(4) + 条目。
    两种布局分别试解析，按“干净度优先、条目数次之”的确定性规则择优：
    能恰好消费到负载末尾且条目数与声明计数一致的布局优先，避免错误
    切分偶然产生的“伪条目”被采信。
    """
    standard = _parse_chpl_layout(payload, 4)
    legacy = _parse_chpl_layout(payload, 0)

    if standard is not None and standard.clean:
        return standard.entries, standard.warnings
    if legacy is not None and legacy.clean:
        return legacy.entries, legacy.warnings
    # 两种布局都不干净（输入确有损坏）：取条目较多者，并列时标准布局优先
    candidates = [r for r in (standard, legacy) if r is not None]
    if candidates:
        chosen = max(candidates, key=lambda r: len(r.entries))
        return chosen.entries, chosen.warnings
    if payload:
        return [], ["chpl 盒无法解析出任何章节条目，已忽略"]
    return [], []


@dataclass
class _ChplLayoutResult:
    """单个 chpl 布局的解析结果。"""

    entries: List[Tuple[int, str]]
    warnings: List[str]
    clean: bool          # 条目数匹配声明计数且恰好消费到负载末尾
    declared_count: int


def _parse_chpl_layout(
    payload: bytes, fullbox_header: int
) -> Optional[_ChplLayoutResult]:
    """按指定 FullBox 头长度解析 chpl 负载。

    :return: 解析结果；布局头部都放不下时返回 None，表示该布局不适用。
    """
    warnings: List[str] = []
    body = payload[fullbox_header:]
    # reserved(1) + chapter_count(4) 至少需要 5 字节
    if len(body) < 5:
        return None

    declared_count = unpack(">I", body[1:5])[0]
    if declared_count > _MAX_CHAPTERS:
        warnings.append(
            f"chpl 声明章节数 {declared_count} 超过保护上限 {_MAX_CHAPTERS}，"
            "按上限截断"
        )
        declared_count = _MAX_CHAPTERS

    entries: List[Tuple[int, str]] = []
    pos = 5
    truncated = False
    while len(entries) < declared_count:
        # start_time(8) + title_length(1) 至少 9 字节
        if pos + 9 > len(body):
            truncated = True
            break
        start_time = unpack(">Q", body[pos:pos + 8])[0]
        title_len = body[pos + 8]  # uint8：标题长度为单字节 Pascal 风格
        pos += 9
        title_end = pos + title_len
        if title_end > len(body):
            # 标题被截断：按可用字节恢复，这是损坏文件能保留的最后信息
            title_bytes = body[pos:]
            warnings.append(
                f"第 {len(entries) + 1} 个章节标题声明 {title_len} 字节但负载截断，"
                f"仅恢复 {len(title_bytes)} 字节"
            )
            entries.append((start_time, _decode_title(title_bytes)))
            truncated = True
            break
        entries.append((start_time, _decode_title(body[pos:title_end])))
        pos = title_end

    if truncated:
        warnings.append(
            f"chpl 负载在第 {len(entries) + 1} 个章节处截断，"
            f"已恢复 {len(entries)}/{declared_count} 个条目"
        )
    elif pos < len(body):
        # 计数之外的尾部多余字节：忽略但记录（常见于计数少写或追加私有数据）
        warnings.append(
            f"chpl 解析完 {len(entries)} 个章节后仍有 {len(body) - pos} 字节尾随数据，已忽略"
        )
    elif pos > len(body):  # pragma: no cover - 循环条件已保证不会发生
        warnings.append("chpl 结构异常，解析提前终止")

    # “干净”判定：实际条目数与声明一致、未截断且无尾随字节。
    # 空清单（declared_count=0、body 恰好 5 字节）同样是干净结果。
    clean = (
        not truncated
        and len(entries) == declared_count
        and pos == len(body)
    )
    return _ChplLayoutResult(
        entries=entries,
        warnings=warnings,
        clean=clean,
        declared_count=declared_count,
    )


def _decode_title(raw: bytes) -> str:
    """解码章节标题：非法 UTF-8 以替换字符兜底，并去除结尾 NUL 填充。"""
    text = raw.decode("utf-8", errors="replace")
    # 部分封装器以 NUL 结尾或做定长填充，仅清理结尾，保留标题内部字符
    return text.rstrip("\x00")


# ---------------------------------------------------------------------------
# 顶层提取与归一化
# ---------------------------------------------------------------------------

# ffmpeg / GPAC / AtomicParsley 的 chpl 时间戳事实单位：100 纳秒（10 MHz）。
# 规范文本虽称“movie timescale 单位”，但主流开源工具统一写 100ns 绝对值。
_CHPL_100NS_TICKS_PER_SEC = 10_000_000


def _to_nanoseconds(start_time: int, ticks_per_sec: int) -> int:
    """把给定时间基（每秒 tick 数）的时间戳换算为纳秒（四舍五入，非负钳制）。"""
    if start_time < 0:  # u64 理论上无负数，防御性处理
        return 0
    return (start_time * 1_000_000_000 + ticks_per_sec // 2) // ticks_per_sec


def _choose_chpl_timebase(
    max_start: int, movie_timescale: int, movie_duration_seconds: Optional[float]
) -> Tuple[int, Optional[str]]:
    """在“100ns 事实单位”与“movie timescale 规范单位”间确定性择优。

    背景：chpl 的 start_time 存在两种真实来源——

    - ffmpeg/GPAC/AtomicParsley 写 **100ns 绝对单位**（10 MHz tick，
      每秒 1e7）；
    - 严格遵循早期 Apple 文档的工具写 **movie timescale 单位**
      （每秒 tick 数 = mvhd timescale，常见 24k/44.1k/48k）。

    二者相差约两个数量级。用“末章 tick ÷ 影片时长”估算章节时间戳的
    实际每秒密度：只有当它明显落在 movie timescale 的 2 倍区间内、
    且与 10 MHz 相差 10 倍以上时才切换到 movie timescale（强证据），
    其余情况一律采用主流事实标准 100ns，避免在证据不足时误判。

    :return: (选中的 ticks_per_sec, 诊断消息或 None)。
    """
    if (
        movie_duration_seconds
        and movie_duration_seconds > 0
        and max_start > 0
        and movie_timescale > 0
    ):
        observed_tps = max_start / movie_duration_seconds
        near_movie_ts = (
            movie_timescale / 2 <= observed_tps <= movie_timescale * 2
        )
        far_from_100ns = observed_tps * 10 < _CHPL_100NS_TICKS_PER_SEC
        if near_movie_ts and far_from_100ns:
            return (
                movie_timescale,
                "检测到 chpl 时间戳采用 movie timescale 单位（非 100ns 事实标准）",
            )
    # 缺时长参照或证据不足：采用主流事实标准 100ns
    return _CHPL_100NS_TICKS_PER_SEC, None


def extract_chapters(
    moov: Box,
    movie_timescale: Optional[int],
    movie_duration_seconds: Optional[float] = None,
) -> Tuple[List[Chapter], List[str]]:
    """从 moov 盒树提取并归一化章节清单。

    :param moov: 已解析的 ``moov`` 盒。
    :param movie_timescale: ``mvhd`` 中的 movie timescale；无法取得时
        回退为 100ns 事实单位并记录诊断。
    :param movie_duration_seconds: mvhd 影片时长（秒），用于在 100ns 与
        movie timescale 两种 chpl 时间单位间做有界判定；未知则传 None。
    :return: (章节列表, 诊断信息列表)。章节列表按起始时间稳定排序、
        同时间同标题去重，UID 顺序由列表位置决定，结果完全确定。
        任何输入下都不抛异常。
    """
    warnings: List[str] = []
    if not movie_timescale or movie_timescale <= 0:
        warnings.append(
            "无法取得合法的 mvhd timescale，章节时间戳按 100ns 事实单位解释"
        )
        movie_timescale = _CHPL_100NS_TICKS_PER_SEC

    try:
        chpl_boxes = _find_chpl_boxes(moov)
    except RemuxError as exc:
        return [], [f"章节盒树遍历失败，已放弃章节保留: {exc}"]

    raw_entries: List[Tuple[int, str, Tuple[str, ...]]] = []
    for chpl in chpl_boxes:
        if len(chpl.data) > _MAX_CHPL_PAYLOAD:
            warnings.append(
                f"chpl 负载 {len(chpl.data)} 字节超过上限，该盒仅解析前 "
                f"{_MAX_CHPL_PAYLOAD} 字节"
            )
        entries, box_warnings = _parse_chpl_payload(chpl.data[:_MAX_CHPL_PAYLOAD])
        warnings.extend(box_warnings)
        for start_time, title in entries:
            raw_entries.append((start_time, title, tuple(box_warnings)))

    if not raw_entries:
        return [], warnings

    # 依据最后一个章节落点，在 100ns 事实单位与 movie timescale 间自适应选择
    max_start = max(start_time for start_time, _t, _w in raw_entries)
    ticks_per_sec, timebase_warning = _choose_chpl_timebase(
        max_start, movie_timescale, movie_duration_seconds
    )
    if timebase_warning:
        warnings.append(timebase_warning)

    # 归一化：时间换算 → 稳定排序 → (start_ns, 标题) 去重。
    # 多个 chpl 盒（重复封装/厂商冗余）由此合并为一份确定清单。
    normalized = [
        (
            _to_nanoseconds(start_time, ticks_per_sec),
            title,
            entry_warnings,
        )
        for start_time, title, entry_warnings in raw_entries
    ]
    normalized.sort(key=lambda item: (item[0], item[1]))

    deduped: List[Tuple[int, str, Tuple[str, ...]]] = []
    seen: set = set()
    for start_ns, title, entry_warnings in normalized:
        key = (start_ns, title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start_ns, title, entry_warnings))

    if len(deduped) > _MAX_CHAPTERS:
        warnings.append(
            f"归一化后章节数 {len(deduped)} 超过上限 {_MAX_CHAPTERS}，超出部分丢弃"
        )
        deduped = deduped[:_MAX_CHAPTERS]

    chapters: List[Chapter] = []
    for index, (start_ns, title, entry_warnings) in enumerate(deduped, start=1):
        if not title:
            # 空标题在部分封装器中表示“仅打点”；给出确定性占位名，
            # 编号取归一化后的位置，保证同一输入反复处理结果一致
            title = f"Chapter {index}"
        chapters.append(
            Chapter(start_ns=start_ns, title=title, warnings=entry_warnings)
        )
    return chapters, warnings
