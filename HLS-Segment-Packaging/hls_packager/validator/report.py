"""校验结果的数据模型（纯数据、零 IO）。

校验器不抛异常表达"校验未通过"——未通过是一种**正常结果**，
用 :class:`ValidationReport` 里的分级 :class:`Issue` 列表表达；
只有"连校验本身都无法进行"（如播放列表文件不可读）才同样
回落为一条 ERROR 级 Issue，保证调用方永远拿到结构化报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---- 问题严重度（字符串常量，便于 JSON 序列化与测试断言） ----
SEVERITY_INFO = "INFO"        # 提示性信息，不影响判定
SEVERITY_WARNING = "WARNING"  # 不规范但通常可播放
SEVERITY_ERROR = "ERROR"      # 明确违规，判定为不通过

#: 严重度排序权重（汇总时取最高级）
_SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_WARNING: 1, SEVERITY_ERROR: 2}


@dataclass(frozen=True)
class Issue:
    """一条校验发现的问题/提示。"""

    severity: str
    #: 机器可读的问题码，如 ``PLAYLIST_NO_EXTM3U`` / ``TS_BAD_SYNC``
    code: str
    #: 面向人的中文描述
    message: str
    #: 关联的分片 URI（播放列表级问题为 None）
    segment: str | None = None
    #: 关联的播放列表行号（1 起；与行无关时为 None）
    line: int | None = None


@dataclass
class SegmentReport:
    """单个分片的校验结果。"""

    uri: str
    #: 解析后的本地路径；远程 URI 或无法解析时为 None
    resolved_path: str | None = None
    #: EXTINF 声明时长（秒）；声明缺失/非法时为 None
    declared_duration: float | None = None
    #: 文件是否存在且可读
    exists: bool = False
    size_bytes: int = 0
    #: 该分片是否带 EXT-X-KEY 加密声明
    encrypted: bool = False
    #: 是否跳过了内容级校验（远程 URI、缺密钥、非 TS 格式等）
    content_skipped: bool = False
    #: TS 探针结果（hls_packager.validator.ts_probe.SegmentProbe）；未探测为 None
    probe: object | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == SEVERITY_ERROR for i in self.issues)


@dataclass
class ValidationReport:
    """一次完整校验的汇总报告。"""

    playlist_path: str
    #: 播放列表级（语法、标签合规）问题
    issues: list[Issue] = field(default_factory=list)
    segments: list[SegmentReport] = field(default_factory=list)
    #: 解析出的分片总数（含校验失败的）
    segment_count: int = 0
    #: EXTINF 声明时长之和（秒）
    declared_total: float = 0.0

    @property
    def all_issues(self) -> list[Issue]:
        """播放列表级 + 各分片级问题的完整列表。"""
        out = list(self.issues)
        for seg in self.segments:
            out.extend(seg.issues)
        return out

    def count(self, severity: str) -> int:
        return sum(1 for i in self.all_issues if i.severity == severity)

    @property
    def error_count(self) -> int:
        return self.count(SEVERITY_ERROR)

    @property
    def warning_count(self) -> int:
        return self.count(SEVERITY_WARNING)

    @property
    def ok(self) -> bool:
        """判定结果：没有任何 ERROR 级问题即视为通过。"""
        return self.error_count == 0

    def summary_text(self) -> str:
        """生成适合终端输出的中文汇总文本。"""
        lines = []
        for issue in self.issues:
            loc = f"第{issue.line}行 " if issue.line else ""
            lines.append(f"  [{issue.severity}] {issue.code}: {loc}{issue.message}")
        for seg in self.segments:
            for issue in seg.issues:
                lines.append(
                    f"  [{issue.severity}] {issue.code}: {seg.uri}: {issue.message}"
                )
        verdict = "通过" if self.ok else "未通过"
        lines.append(
            f"校验{verdict}：{self.segment_count} 个分片，"
            f"声明总时长 {self.declared_total:.3f}s，"
            f"错误 {self.error_count} 个，警告 {self.warning_count} 个"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """机器可读的 JSON 友好结构。"""
        return {
            "playlist": self.playlist_path,
            "ok": self.ok,
            "segment_count": self.segment_count,
            "declared_total": round(self.declared_total, 6),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "segment": i.segment,
                    "line": i.line,
                }
                for i in self.all_issues
            ],
            "segments": [
                {
                    "uri": s.uri,
                    "exists": s.exists,
                    "size_bytes": s.size_bytes,
                    "encrypted": s.encrypted,
                    "declared_duration": s.declared_duration,
                    "ok": s.ok,
                }
                for s in self.segments
            ],
        }
