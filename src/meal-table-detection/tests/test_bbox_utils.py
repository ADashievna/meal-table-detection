import pytest

from utils.bbox_utils import calc_ioa_two_boxes, is_inside, IOA_THRESHOLD

def test_calc_ioa_full_overlap():
    inner = outer = (0, 0, 10, 10)
    ioa = calc_ioa_two_boxes(inner, outer)
    assert ioa == pytest.approx(1.0)


def test_calc_ioa_inner_inside_outer():
    inner = (2, 2, 4, 4)
    outer = (0, 0, 10, 10)
    ioa = calc_ioa_two_boxes(inner, outer)
    assert ioa == pytest.approx(1.0)


def test_calc_ioa_partial_overlap():
    inner = (0, 0, 10, 10)
    outer = (2, 2, 8, 8)
    ioa = calc_ioa_two_boxes(inner, outer)
    assert ioa == pytest.approx(0.36, rel=1e-6)


def test_calc_ioa_no_overlap():
    inner = (0, 0, 2, 2)
    outer = (3, 3, 5, 5)
    ioa = calc_ioa_two_boxes(inner, outer)
    assert ioa == 0.0

def test_is_inside_full_overlap_default_threshold():
    inner = outer = (0, 0, 10, 10)
    assert is_inside(inner, outer) is True


def test_is_inside_inner_inside_outer_default_threshold():
    inner = (2, 2, 4, 4)
    outer = (0, 0, 10, 10)
    assert is_inside(inner, outer) is True


def test_is_inside_partial_overlap_below_default_threshold():
    inner = (0, 0, 10, 10)
    outer = (2, 2, 8, 8)
    assert is_inside(inner, outer) is False


def test_is_inside_partial_overlap_custom_threshold():
    inner = (0, 0, 10, 10)
    outer = (2, 2, 8, 8)
    assert is_inside(inner, outer, threshold=0.3) is True