"""MP4 → MKV 无损重封装工具。

只做容器层重封装（remux）：解析 ISO-BMFF(MP4) 盒结构，提取已编码的
H.264/HEVC/AAC 等样本，原样写入 Matroska(MKV) 容器，全程不做任何
解码 / 重编码，因此是无损且保持原始码流字节不变的。

自 0.3.0 起同时保留 ``moov/udta/meta/chpl`` 封装层章节（chapter）
元数据：字节级解析、容错损坏与不规则结构，并转写为 Matroska
Chapters（EditionEntry/ChapterAtom），时间戳精确换算为纳秒。
"""

__version__ = "0.3.0"
