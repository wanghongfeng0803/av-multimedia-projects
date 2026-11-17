"""原子热加载 :class:`ConfigStore` 测试（不依赖 ffmpeg）。

锁定三条热加载硬约束：

1. 替换是不可变快照的整体切换，读取永远拿到完整一致的配置；
2. 进行中任务持有的旧快照不受后续重载影响；
3. 坏配置不污染现行配置，校验失败时旧快照继续服务。
"""

from __future__ import annotations

import os

import pytest

from bvp.rules.schema import RuleConfigError
from bvp.rules.store import ConfigStore

_CODECS = frozenset({"h264", "h265"})


def _write(path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _config(codec: str = "h264") -> str:
    return """{
  "version": 1,
  "rules": [
    {"name": "r1", "dir": ".", "params": {"codec": "%s"}}
  ]
}""" % codec


def test_initial_bad_config_raises_and_refuses_start(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, '{"version": 1, "rules": []}')  # 空 rules 非法
    with pytest.raises(RuleConfigError):
        ConfigStore(str(path))


def test_snapshot_is_immutable_and_shared(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    first = store.snapshot()
    second = store.snapshot()
    assert first is second  # 未重载时返回同一不可变对象


def test_reload_atomically_swaps_snapshot(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    old = store.snapshot()

    # 强制推进 mtime，避免同一秒内 mtime_ns 分辨率不足导致漏检
    _write(path, _config("h265"))
    os.utime(path, ns=(10**9 * 200, 10**9 * 200))

    result = store.maybe_reload()
    assert result.loaded and result.ok
    new = store.snapshot()
    assert new is not old
    assert new.resolve(str(tmp_path / "a.mp4")).params.codec == "h265"
    # 旧快照仍然自洽可读：进行中任务不受影响
    assert old.resolve(str(tmp_path / "a.mp4")).params.codec == "h264"


def test_unchanged_file_is_not_reloaded(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    result = store.maybe_reload()
    assert result.changed is False and result.loaded is False
    assert store.snapshot() is store.snapshot()


def test_bad_reload_keeps_current_snapshot(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    good = store.snapshot()

    _write(path, '{"version": 1, "rules": []}')  # 非法：空规则集
    os.utime(path, ns=(10**9 * 201, 10**9 * 201))

    result = store.maybe_reload()
    assert not result.ok
    assert isinstance(result.error, RuleConfigError)
    # 现行快照未被污染，继续提供服务
    assert store.snapshot() is good
    assert store.snapshot().resolve(
        str(tmp_path / "a.mp4")
    ).params.codec == "h264"


def test_force_reload_ignores_signature(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    # 文件未变化，force 仍重新解析（内容相同，得到等价新快照）
    result = store.maybe_reload(force=True)
    assert result.loaded


def test_vanished_file_during_reload_keeps_current(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h264"))
    store = ConfigStore(str(path))
    good = store.snapshot()

    # 文件暂时不可见：签名哨兵触发重新加载，加载失败但现行快照不被污染
    os.remove(path)
    result = store.maybe_reload()
    assert not result.ok
    assert store.snapshot() is good

    # 文件恢复后可正常重新加载
    _write(path, _config("h265"))
    os.utime(path, ns=(10**9 * 202, 10**9 * 202))
    result = store.maybe_reload()
    assert result.ok and result.loaded
    assert store.snapshot() is not good


def test_codec_probe_injected_into_validation(tmp_path):
    path = tmp_path / "rules.json"
    _write(path, _config("h265"))
    # 探测声称本机只有 h264：h265 规则必须在构造时被拒绝
    with pytest.raises(RuleConfigError):
        ConfigStore(str(path), codec_probe=lambda: frozenset({"h264"}))
    # 探测返回 h265 时正常
    store = ConfigStore(
        str(path), codec_probe=lambda: frozenset({"h264", "h265"})
    )
    assert store.snapshot() is not None
