"""WatermarkConfig 参数校验测试。"""

import pytest

from watermark_overlay.config import WatermarkConfig
from watermark_overlay.errors import InvalidParameterError


def test_default_config_is_valid():
    WatermarkConfig().validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor": "middle"},
        {"offset_x": 1.5},
        {"offset_y": True},
        {"target_width": 0},
        {"target_height": -3},
        {"target_width": 2.0},
        {"opacity": -0.1},
        {"opacity": 1.1},
        {"opacity": "0.5"},
        {"adaptive": "yes"},
        {"mask_strength": -0.01},
        {"mask_strength": 2},
        {"detail_low": -1},
        {"detail_high": 256},
        {"detail_low": 100, "detail_high": 100},
        {"detail_low": 200, "detail_high": 100},
        {"mask_radius": 0},
        {"mask_radius": 1.5},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(InvalidParameterError):
        WatermarkConfig(**kwargs).validate()


def test_boundary_values_accepted():
    WatermarkConfig(opacity=0.0).validate()
    WatermarkConfig(opacity=1.0).validate()
    WatermarkConfig(mask_strength=0.0).validate()
    WatermarkConfig(mask_strength=1.0).validate()
    WatermarkConfig(detail_low=0, detail_high=255).validate()
    WatermarkConfig(mask_radius=1).validate()


def test_booleans_are_not_accepted_as_ints():
    # bool 是 int 的子类，但尺寸/偏移语义上不接受
    with pytest.raises(InvalidParameterError):
        WatermarkConfig(target_width=True).validate()
