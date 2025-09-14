import logging
import numpy as np
import pytest

import humancompatible.detect.helpers.utils as utils


# =====
# Tests for signed_subgroup_discrepancy
# =====
def test_signed_subgroup_discrepancy_basic_positive():
    subgroup = np.array([1, 1, 0, 0], dtype=int)  # int -> should auto-convert
    y = np.array([1, 0, 1, 0], dtype=int)         # int -> should auto-convert
    val = utils.signed_subgroup_discrepancy(subgroup, y)
    # subgroup among positives: mean([1,0]) = 0.5 ; among negatives: mean([1,0]) = 0.5 => 0.0
    assert val == pytest.approx(0.0)

def test_signed_subgroup_discrepancy_negative_gap():
    subgroup = np.array([True, False, True, False, False, False])
    y = np.array([True, True, False, False, False, True])
    # positives idx: [0,1,5] -> subgroup[1,0,0] mean=1/3; negatives idx [2,3,4] -> [True, False, False] mean=1/3 => 0
    assert utils.signed_subgroup_discrepancy(subgroup, y) == pytest.approx(0.0)

def test_signed_subgroup_discrepancy_shape_mismatch():
    with pytest.raises(AssertionError):
        utils.signed_subgroup_discrepancy(np.array([True, False]), np.array([True]))

def test_signed_subgroup_discrepancy_all_one_raises():
    subgroup = np.array([True, False, True])
    y_all_one = np.array([True, True, True])
    with pytest.raises(ValueError):
        utils.signed_subgroup_discrepancy(subgroup, y_all_one)

def test_signed_subgroup_discrepancy_all_zero_raises():
    subgroup = np.array([True, False, True])
    y_all_zero = np.array([False, False, False])
    with pytest.raises(ValueError):
        utils.signed_subgroup_discrepancy(subgroup, y_all_zero)

def test_signed_subgroup_discrepancy_logs_type_coercion(caplog):
    caplog.set_level(logging.WARNING)
    subgroup = np.array([0, 1, 0, 1], dtype=int)  # not bool
    y = np.array([0, 1, 1, 0], dtype=int)         # not bool
    _ = utils.signed_subgroup_discrepancy(subgroup, y)
    # Two warnings (one for subgroup, one for y)
    messages = [rec.message for rec in caplog.records]
    assert any("instead of bool" in m for m in messages)


# =====
# Tests for evaluate_subgroup_discrepancy
# =====
def test_evaluate_subgroup_discrepancy_is_abs():
    subgroup = np.array([True, False, False, False])
    y = np.array([True, True,  False, False])
    signed = utils.signed_subgroup_discrepancy(subgroup, y)
    absval = utils.evaluate_subgroup_discrepancy(subgroup, y)
    assert absval == pytest.approx(abs(signed))


# =====
# Tests for signed_subgroup_prevalence_diff
# =====
def test_signed_subgroup_prevalence_diff_basic():
    a = np.array([True, False, False, True])   # mean = 0.5
    b = np.array([False, False, True, False])  # mean = 0.25
    assert utils.signed_subgroup_prevalence_diff(a, b) == pytest.approx(0.25 - 0.5)

