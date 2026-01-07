import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import pytest

import humancompatible.detect.helpers.utils as utils


# Helpers for stubbing Bin-like objects
class _Feature:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"_Feature({self.name!r})"
    def __str__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, _Feature) and self.name == other.name

class _Bin:
    def __init__(self, feature, value):
        self.feature = _Feature(feature)
        self.value = value
    def __eq__(self, other):
        return isinstance(other, _Bin) and (self.feature, self.value) == (other.feature, other.value)
    def __repr__(self):
        return f"_Bin({self.feature!r}, {self.value!r})"
    def __str__(self) -> str:
        feat_name = self.feature if isinstance(self.feature, str) else self.feature.name
        return f"{feat_name} = {self.value}"
    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return values == self.value


# =====
# Tests for detect_and_score
# =====
def test_detect_and_score_msd(monkeypatch):
    """
    Ensure detect_and_score:
      - calls most_biased_subgroup and passes the rule into evaluate via method_kwargs['rule']
      - returns (rule, value)
      - does not mutate the caller's method_kwargs
    """
    X = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 1, 0, 0]})
    y = pd.DataFrame({"target": [1, 0, 1, 0]})

    fake_rule = [(0, _Bin("A", 1))]

    captured = {}

    def _fake_most_biased_subgroup(X_, y_, **kwargs):
        assert kwargs["method"] == "MSD"
        return fake_rule

    def _fake_evaluate_biased_subgroup(X_, y_, **kwargs):
        # Verify the rule is forwarded for MSD
        captured["method_kwargs"] = deepcopy(kwargs.get("method_kwargs", {}))
        assert "rule" in captured["method_kwargs"]
        assert captured["method_kwargs"]["rule"] == fake_rule
        return 0.42

    # Monkeypatch into the real modules that utils imports inside the function
    import humancompatible.detect.detect_bias as detect_bias_mod
    import humancompatible.detect.evaluate_bias as evaluate_bias_mod
    monkeypatch.setattr(detect_bias_mod, "most_biased_subgroup", _fake_most_biased_subgroup, raising=True)
    monkeypatch.setattr(evaluate_bias_mod, "evaluate_biased_subgroup", _fake_evaluate_biased_subgroup, raising=True)

    method_kwargs_in = {"time_limit": 1}
    rule, val = utils.detect_and_score(
        X=X, y=y,
        protected_list=["A", "B"],
        continuous_list=None,
        fp_map=None,
        seed=7,
        n_samples=1000,
        method="MSD",
        method_kwargs=method_kwargs_in,
    )

    assert rule == fake_rule
    assert val == pytest.approx(0.42)
    assert "rule" not in method_kwargs_in
    assert method_kwargs_in == {"time_limit": 1}

def test_detect_and_score_linf(monkeypatch):
    """
    Ensure detect_and_score:
      - does not compute a rule for l_inf (rule is None)
      - does NOT inject 'rule' into method_kwargs
    """
    X = pd.DataFrame({"A": [0, 1]})
    y = pd.DataFrame({"target": [1, 0]})

    import humancompatible.detect.detect_bias as detect_bias_mod
    def _boom(*a, **k):  # would indicate wrong branch
        raise AssertionError("MSD path should not be called for method='l_inf'")
    monkeypatch.setattr(detect_bias_mod, "most_biased_subgroup", _boom, raising=True)

    seen_kwargs = {}

    import humancompatible.detect.evaluate_bias as evaluate_bias_mod
    def _fake_eval(X_, y_, **kwargs):
        seen_kwargs.update(kwargs)
        assert "method_kwargs" in seen_kwargs
        assert "rule" not in (seen_kwargs["method_kwargs"] or {})
        return 0.123
    monkeypatch.setattr(evaluate_bias_mod, "evaluate_biased_subgroup", _fake_eval, raising=True)

    method_kwargs = {"feature_involved": "A", "subgroup_to_check": 1, "delta": 0.05}
    rule, val = utils.detect_and_score(
        X=X, y=y,
        protected_list=["A"],
        method="l_inf",
        method_kwargs=method_kwargs,
    )
    assert rule == []
    assert val == pytest.approx(0.123)
    assert method_kwargs == {"feature_involved": "A", "subgroup_to_check": 1, "delta": 0.05}


# =====
# Tests for signed_subgroup_discrepancy
# =====
def test_signed_subgroup_discrepancy_basic_positive():
    subgroup = np.array([1, 1, 0, 0], dtype=int)  # int -> should auto-convert
    y = np.array([1, 0, 1, 0], dtype=int)
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

def test_signed_subgroup_discrepancy_logs(caplog):
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
