"""内存探测与预算推导测试（纯逻辑）。"""

from vdenoise.memory import (
    DEFAULT_FRACTION,
    FALLBACK_BYTES,
    MIN_BUDGET,
    available_memory,
    derive_budget,
)


def test_available_memory_positive_int():
    mem = available_memory()
    assert isinstance(mem, int)
    assert mem > 0


def test_fallback_is_512mib():
    assert FALLBACK_BYTES == 512 * 1024 * 1024


def test_derive_budget_fraction():
    avail = 1 << 30  # 1 GiB
    assert derive_budget(available=avail) == int(avail * DEFAULT_FRACTION)


def test_derive_budget_cap():
    assert derive_budget(available=1 << 40, cap_bytes=1 << 20) == (1 << 20)


def test_derive_budget_floor():
    assert derive_budget(available=1) == MIN_BUDGET


def test_derive_budget_custom_fraction():
    avail = 8 * 1024 ** 3
    assert derive_budget(available=avail, fraction=0.5) == 4 * 1024 ** 3
