"""目录扫描的确定性、嵌套处理与不可访问路径测试（不依赖 ffmpeg）。"""

from __future__ import annotations

import os

from bvp.scanner import (
    DEFAULT_VIDEO_EXTENSIONS,
    scan_inputs,
)


def _touch(path: str, content: str = "x") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def test_flat_scan_collects_video_extensions(tmp_path):
    _touch(str(tmp_path / "a.mp4"))
    _touch(str(tmp_path / "b.txt"))
    _touch(str(tmp_path / "c.MKV"))  # 大写扩展名也应命中
    result = scan_inputs([str(tmp_path)], recursive=False)
    names = sorted(os.path.basename(p) for p in result.files)
    assert names == ["a.mp4", "c.MKV"]
    assert result.problems == ()


def test_recursive_scan_handles_nested_dirs(tmp_path):
    _touch(str(tmp_path / "root.mp4"))
    _touch(str(tmp_path / "sub1" / "deep" / "d.mp4"))
    _touch(str(tmp_path / "sub2" / "e.mov"))
    _touch(str(tmp_path / "sub1" / "note.txt"))
    result = scan_inputs([str(tmp_path)], recursive=True)
    rel = sorted(
        os.path.relpath(p, str(tmp_path)).replace(os.sep, "/")
        for p in result.files
    )
    assert rel == [
        "root.mp4", "sub1/deep/d.mp4", "sub2/e.mov",
    ]


def test_traversal_order_is_deterministic_regardless_of_creation_order(tmp_path):
    # 故意按乱序创建，扫描结果必须确定。注意 Windows/macOS 默认文件系统
    # 大小写不敏感，无法共存仅大小写不同的文件名，因此这里用小写名各不相同
    # 的集合验证排序键的第一键（小写名）；第二键（原名）在大小写敏感系统上
    # 才会体现，排序实现始终同时给出两个键以保证跨平台确定。
    names = ["z.mp4", "a.mp4", "m.mp4", "b.mp4", "c.mp4"]
    for name in names:
        _touch(str(tmp_path / name))
    first = scan_inputs([str(tmp_path)]).files
    second = scan_inputs([str(tmp_path)]).files
    assert first == second
    assert [os.path.basename(p) for p in first] == sorted(
        names, key=lambda n: (n.lower(), n)
    )
    assert [os.path.basename(p) for p in first] == [
        "a.mp4", "b.mp4", "c.mp4", "m.mp4", "z.mp4"
    ]


def test_nested_directory_order_is_depth_first_deterministic(tmp_path):
    _touch(str(tmp_path / "z" / "z.mp4"))
    _touch(str(tmp_path / "a" / "a.mp4"))
    _touch(str(tmp_path / "a" / "sub" / "s.mp4"))
    _touch(str(tmp_path / "top.mp4"))
    result = scan_inputs([str(tmp_path)], recursive=True)
    rel = [
        os.path.relpath(p, str(tmp_path)).replace(os.sep, "/")
        for p in result.files
    ]
    # DFS：a 目录整体先于 z 目录，目录内文件字典序
    assert rel == [
        os.path.join("a", "a.mp4").replace(os.sep, "/"),
        os.path.join("a", "sub", "s.mp4").replace(os.sep, "/"),
        "top.mp4",
        os.path.join("z", "z.mp4").replace(os.sep, "/"),
    ]


def test_explicit_file_bypasses_extension_filter(tmp_path):
    weird = _touch(str(tmp_path / "noext_video"), content="fake")
    result = scan_inputs([weird])
    assert result.files == (os.path.abspath(weird),)


def test_missing_path_is_reported_not_silently_dropped(tmp_path):
    ghost = str(tmp_path / "ghost.mp4")
    result = scan_inputs([ghost])
    assert result.files == ()
    assert len(result.problems) == 1
    problem = result.problems[0]
    assert problem.kind == "missing"
    assert os.path.abspath(ghost) == problem.path


def test_unreadable_directory_is_reported_as_problem(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    _touch(str(blocked / "inside.mp4"))
    try:
        os.chmod(str(blocked), 0o000)
        # Windows 上 chmod 0o000 不一定真正拒绝读取；两种结果都要合法
        result = scan_inputs([str(tmp_path)], recursive=True)
        if result.problems:
            assert result.problems[0].kind == "access_denied"
            assert "blocked" in result.problems[0].path
        else:
            # 权限未生效（常见于 Windows）：文件应被正常收集
            assert any("inside.mp4" in p for p in result.files)
    finally:
        os.chmod(str(blocked), 0o755)


def test_excluded_output_dir_is_not_rescanned(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _touch(str(tmp_path / "src.mp4"))
    _touch(str(out / "previous_h264.mp4"))  # 上一轮产物，同为视频扩展名
    result = scan_inputs(
        [str(tmp_path)],
        recursive=True,
        exclude_dirs=frozenset({os.path.abspath(str(out))}),
    )
    assert [os.path.basename(p) for p in result.files] == ["src.mp4"]


def test_dedup_across_multiple_roots(tmp_path):
    video = _touch(str(tmp_path / "shared.mp4"))
    result = scan_inputs([video, video, str(tmp_path)])
    assert result.files.count(os.path.abspath(video)) == 1


def test_default_extensions_are_lowercase_dotted():
    assert all(e.startswith(".") and e == e.lower()
               for e in DEFAULT_VIDEO_EXTENSIONS)
    assert ".mp4" in DEFAULT_VIDEO_EXTENSIONS
