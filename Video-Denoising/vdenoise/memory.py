"""可用内存探测与分块预算推导（仅标准库）。

长视频靠逐帧流式处理把内存压到 O(单帧)；单帧内部再按这里的预算
分带，把峰值压到 O(预算)。预算默认取可用物理内存的 1/4，并设保底，
避免小内存机器上核心行高被扣成 0。
"""

import os
import sys

__all__ = [
    "available_memory",
    "derive_budget",
    "DEFAULT_FRACTION",
    "MIN_BUDGET",
    "FALLBACK_BYTES",
]

#: 可用内存探测失败时的兜底值（512 MiB）
FALLBACK_BYTES = 512 * 1024 * 1024
#: 预算默认取可用内存的比例
DEFAULT_FRACTION = 0.25
#: 预算保底值，保证最小机器上也能分出合理的带
MIN_BUDGET = 4 * 1024 * 1024


def available_memory():
    """返回当前可用物理内存（字节）；探测失败返回 :data:`FALLBACK_BYTES`。"""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    else:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(size)
        except (ValueError, OSError, AttributeError):
            pass
    return FALLBACK_BYTES


def derive_budget(available=None, fraction=DEFAULT_FRACTION, cap_bytes=None):
    """由可用内存推导单帧分带预算（字节）。

    取 ``available * fraction``，保底 :data:`MIN_BUDGET`；可选封顶
    ``cap_bytes``——显式封顶优先于保底（调用方明确给的预算必须遵守）。
    ``available`` 缺省时现场探测。
    """
    if available is None:
        available = available_memory()
    budget = max(int(available * fraction), MIN_BUDGET)
    if cap_bytes is not None:
        budget = min(budget, cap_bytes)
    return budget
