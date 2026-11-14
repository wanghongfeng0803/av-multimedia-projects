"""ISO 14496-30 WebVTT（wvtt）样本 → Matroska S_TEXT/WEBVTT 转换。

MP4 中每个字幕样本是一串小盒：
- ``vttc``：一条 cue，内含 ``payl``（正文，必需）、``iden``（cue id）、
  ``sttg``（cue settings）等可选子盒；
- ``vtte``：空 cue（占位），无负载。

Matroska 的 ``S_TEXT/WEBVTT`` 约定：
- Block 负载只放 cue 正文（UTF-8），起止时间由块时间戳与 BlockDuration 表达；
- cue id 与 settings 放进 BlockGroup 的 BlockAdditions，
  BlockAddID=1 为 cue identifier、=2 为 cue settings（见 Matroska 规范）。

转换只做容器层重组，正文文本逐字节保留、不做任何文本转码。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import RemuxError
from .mp4_parser import parse_boxes

# Matroska WebVTT 的 BlockAddID 取值
BLOCK_ADD_ID_IDENTIFIER = 1   # cue id
BLOCK_ADD_ID_SETTINGS = 2     # cue settings


@dataclass
class WebVttCue:
    """一条转换后的 Matroska WebVTT cue。"""

    payload: bytes                       # Block 负载（cue 正文，可能为空）
    identifier: bytes = b""              # BlockAddID=1 的内容
    settings: bytes = b""                # BlockAddID=2 的内容

    @property
    def has_additions(self) -> bool:
        return bool(self.identifier or self.settings)


def _find_box(buf: bytes, kind: str, base_offset: int = 0):
    for box in parse_boxes(buf, base_offset=base_offset):
        if box.kind == kind:
            return box
    return None


def convert_sample(sample_data: bytes) -> WebVttCue:
    """把一个 wvtt 样本转换为 :class:`WebVttCue`。

    :raises RemuxError: 样本既不含 vttc 也不含 vtte（结构损坏）。
    """
    if not sample_data:
        # 空样本按空 cue 处理，交由上层以普通块写出
        return WebVttCue(b"")

    boxes = parse_boxes(sample_data)
    vttc = next((b for b in boxes if b.kind == "vttc"), None)
    if vttc is None:
        if any(b.kind == "vtte" for b in boxes):
            return WebVttCue(b"")  # 空 cue
        raise RemuxError("wvtt 样本中既无 vttc 也无 vtte，字幕结构损坏")

    # vttc 内子盒顺序不固定，按四字符码分别提取；未知盒忽略
    payl = _find_box(vttc.data, "payl", base_offset=vttc.data_start)
    iden = _find_box(vttc.data, "iden", base_offset=vttc.data_start)
    sttg = _find_box(vttc.data, "sttg", base_offset=vttc.data_start)
    return WebVttCue(
        payload=bytes(payl.data) if payl is not None else b"",
        identifier=bytes(iden.data) if iden is not None else b"",
        settings=bytes(sttg.data) if sttg is not None else b"",
    )
