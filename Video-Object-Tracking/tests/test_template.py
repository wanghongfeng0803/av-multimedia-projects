"""模板匹配纯逻辑测试：三种方法的正确性、边界与粗精搜索一致性。"""

import pytest

from conftest import extract, make_plane, put_pattern, put_rect
from vtrack.template import (
    METHOD_SAD,
    METHOD_SSD,
    METHOD_ZNCC,
    TEMPLATE_METHODS,
    block_downsample,
    extract_patch,
    integral_image,
    match_template,
    rect_sum,
    score_accepted,
    zncc_at,
    best_match,
    best_match_zncc_coarse_fine,
)

W, H = 48, 32
TW, TH = 8, 6


def _frame_with_block(x, y, fill=180, bg=20):
    frame = make_plane(W, H, fill=bg)
    put_rect(frame, W, x, y, TW, TH, fill)
    return frame


def test_extract_patch_roundtrip():
    frame = make_plane(W, H, fill=7)
    put_rect(frame, W, 5, 4, TW, TH, 200)
    patch = extract_patch(frame, W, 5, 4, TW, TH)
    assert len(patch) == TW * TH
    assert set(patch) == {200}


def test_sad_finds_exact_translation_zero_score():
    x, y = 17, 9
    frame = _frame_with_block(x, y)
    tpl = extract(frame, W, x, y, TW, TH)
    bx, by, score = best_match(frame, W, H, tpl, TW, TH, method=METHOD_SAD)
    assert (bx, by) == (x, y)
    assert score == 0.0


def test_ssd_finds_exact_translation_zero_score():
    x, y = 21, 14
    frame = _frame_with_block(x, y)
    tpl = extract(frame, W, x, y, TW, TH)
    bx, by, score = best_match(frame, W, H, tpl, TW, TH, method=METHOD_SSD)
    assert (bx, by) == (x, y)
    assert score == 0.0


def test_sad_score_is_per_pixel_mean():
    frame = make_plane(W, H, fill=100)
    tpl = bytes([110]) * (TW * TH)  # 每像素差 10
    bx, by, score = best_match(frame, W, H, tpl, TW, TH, method=METHOD_SAD)
    assert score == pytest.approx(10.0)


def test_ssd_score_is_per_pixel_mean_square():
    frame = make_plane(W, H, fill=100)
    tpl = bytes([105]) * (TW * TH)  # 每像素差 5
    bx, by, score = best_match(frame, W, H, tpl, TW, TH, method=METHOD_SSD)
    assert score == pytest.approx(25.0)


def test_zncc_perfect_correlation_is_one():
    x, y = 12, 8
    frame = make_plane(W, H, fill=30)
    put_pattern(frame, W, x, y, TW, TH, seed=2)
    tpl = extract(frame, W, x, y, TW, TH)
    bx, by, score = match_template(frame, W, H, tpl, TW, TH,
                                   method=METHOD_ZNCC)
    assert (bx, by) == (x, y)
    assert score == pytest.approx(1.0, abs=1e-12)


def test_zncc_linear_brightness_still_matches():
    """ZNCC 对仿射亮度不变：整体加偏置后仍应高分定位。"""
    x, y = 15, 11
    base = make_plane(W, H, fill=30)
    put_pattern(base, W, x, y, TW, TH, seed=5)
    tpl = extract(base, W, x, y, TW, TH)
    shifted = bytearray(min(255, v + 40) for v in base)
    bx, by, score = match_template(shifted, W, H, tpl, TW, TH,
                                   method=METHOD_ZNCC)
    assert (bx, by) == (x, y)
    assert score > 0.95


def test_zncc_constant_template_returns_minus_one():
    frame = make_plane(W, H, fill=90)
    tpl = bytes([120]) * (TW * TH)  # 零方差
    t_sum, t_sum2 = sum(tpl), sum(v * v for v in tpl)
    assert zncc_at(frame, W, 0, 0, tpl, TW, TH, t_sum, t_sum2) == -1.0


def test_zncc_constant_window_returns_minus_one():
    frame = make_plane(W, H, fill=90)        # 候选窗零方差
    tpl = bytearray(TW * TH)
    put_pattern(tpl, TW, 0, 0, TW, TH, seed=1)
    t_sum, t_sum2 = sum(tpl), sum(v * v for v in tpl)
    assert zncc_at(frame, W, 0, 0, bytes(tpl), TW, TH,
                   t_sum, t_sum2) == -1.0


def test_search_region_is_clipped_and_respected():
    x, y = 30, 20
    frame = _frame_with_block(x, y)
    tpl = extract(frame, W, x, y, TW, TH)
    # 搜索区完全不含真位置时，最佳点只能落在搜索区内
    search = (0, 0, 10, 10)
    bx, by, _ = best_match(frame, W, H, tpl, TW, TH,
                           method=METHOD_SAD, search=search)
    assert 0 <= bx <= 10 and 0 <= by <= 10


def test_clip_search_out_of_bounds():
    frame = _frame_with_block(0, 0)
    tpl = extract(frame, W, 0, 0, TW, TH)
    # 负数与超出范围都应被夹回，不抛异常
    bx, by, _ = best_match(frame, W, H, tpl, TW, TH,
                           method=METHOD_SAD, search=(-9, -9, 999, 999))
    assert 0 <= bx <= W - TW and 0 <= by <= H - TH


def test_integral_image_matches_bruteforce():
    frame = make_plane(W, H)
    put_pattern(frame, W, 3, 2, 20, 15, seed=4)
    ii = integral_image(frame, W, H)
    x, y, rw, rh = 5, 4, 7, 9
    expect = sum(
        frame[(y + r) * W + x + c]
        for r in range(rh) for c in range(rw)
    )
    assert rect_sum(ii, W + 1, x, y, rw, rh) == expect


def test_integral_image_square():
    frame = make_plane(W, H, fill=13)
    put_rect(frame, W, 1, 1, 4, 4, 77)
    ii = integral_image(frame, W, H, square=True)
    total = rect_sum(ii, W + 1, 1, 1, 4, 4)
    assert total == 4 * 4 * 77 * 77


def test_block_downsample_averages_and_dims():
    frame = make_plane(W, H, fill=10)
    put_rect(frame, W, 0, 0, 4, 4, 20)  # 4x4 块均值 20，其余 10
    small, sw, sh = block_downsample(frame, W, H, 4)
    assert (sw, sh) == (W // 4, H // 4)
    assert small[0] == 20
    assert small[sw + 1] == 10


def test_block_downsample_factor_one_no_copy():
    frame = bytes(make_plane(W, H))
    small, sw, sh = block_downsample(frame, W, H, 1)
    assert small is frame and (sw, sh) == (W, H)


def test_coarse_fine_matches_exhaustive_on_textured_target():
    x, y = 26, 17
    frame = make_plane(W, H, fill=25)
    put_pattern(frame, W, x, y, TW, TH, seed=6)
    tpl = extract(frame, W, x, y, TW, TH)
    bx, by, score, used = best_match_zncc_coarse_fine(
        frame, W, H, tpl, TW, TH
    )
    assert (bx, by) == (x, y)
    assert score == pytest.approx(1.0, abs=1e-12)
    # 返回的搜索区是夹过边界的合法闭区间
    x0, y0, x1, y1 = used
    assert x0 == 0 and y0 == 0
    assert x1 == W - TW and y1 == H - TH


def test_match_template_dispatches_all_methods():
    x, y = 9, 13
    frame = make_plane(W, H, fill=40)
    put_pattern(frame, W, x, y, TW, TH, seed=7)
    tpl = extract(frame, W, x, y, TW, TH)
    for method in TEMPLATE_METHODS:
        bx, by, score = match_template(frame, W, H, tpl, TW, TH,
                                       method=method)
        assert (bx, by) == (x, y)
        assert score_accepted(
            method, score,
            {"sad": 1.0, "ssd": 1.0, "zncc": 0.99}[method],
        )


def test_invalid_template_size_rejected():
    frame = make_plane(W, H)
    with pytest.raises(ValueError):
        best_match(frame, W, H, b"", 0, TH, method=METHOD_SAD)
    with pytest.raises(ValueError):
        best_match(frame, W, H, b"x", W + 1, TH, method=METHOD_SAD)


def test_best_match_rejects_zncc_use_coarse_instead():
    frame = make_plane(W, H)
    tpl = bytes(TW * TH)
    with pytest.raises(ValueError):
        best_match(frame, W, H, tpl, TW, TH, method=METHOD_ZNCC)
