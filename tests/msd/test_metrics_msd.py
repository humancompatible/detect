from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

import humancompatible.detect.methods.msd.metrics_msd as metrics


# =====
# Tests for subgroup_gap
# =====
def test_subgroup_gap_signed(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0]})
    y = np.array([[1], [0], [1], [0]], dtype=int)
    rule = [("ignored", "ignored")]

    mask = np.array([True, False, True, False], dtype=bool)

    # Patch mapping and signed helper
    monkeypatch.setattr(
        metrics, "subgroup_map_from_conjuncts_dataframe",
        lambda r, Xdf: mask,
        raising=True,
    )

    seen = {}
    def _fake_signed(subgroup_mask, y_vec):
        # Verify we got the mask and flattened y
        seen["mask"] = deepcopy(subgroup_mask)
        seen["y_ndim"] = y_vec.ndim
        return -0.123

    monkeypatch.setattr(metrics, "signed_subgroup_discrepancy", _fake_signed, raising=True)

    out = metrics.subgroup_gap(rule, X, y, signed=True)
    assert out == pytest.approx(-0.123)
    assert np.array_equal(seen["mask"], mask)
    assert seen["y_ndim"] == 1

def test_subgroup_gap_absolute(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 1, 0]})
    y = pd.Series([1, 0, 1, 0], dtype=int)
    rule = [("ignored", "ignored")]

    mask = np.array([True, True, False, False], dtype=bool)

    monkeypatch.setattr(
        metrics, "subgroup_map_from_conjuncts_dataframe",
        lambda r, Xdf: mask,
        raising=True,
    )

    called = {}
    def _fake_abs(subgroup_mask, y_vec):
        called["ndim"] = y_vec.ndim
        return 0.25

    monkeypatch.setattr(metrics, "evaluate_subgroup_discrepancy", _fake_abs, raising=True)

    out = metrics.subgroup_gap(rule, X, y, signed=False)
    assert out == pytest.approx(0.25)
    assert called["ndim"] == 1

def test_subgroup_gap_propagates_keyerror(monkeypatch):
    # Simulate missing-column error coming from mapping function
    def _boom(rule, Xdf):
        raise KeyError("missing column 'Z'")

    monkeypatch.setattr(metrics, "subgroup_map_from_conjuncts_dataframe", _boom, raising=True)

    with pytest.raises(KeyError, match="missing column 'Z'"):
        metrics.subgroup_gap(
            rule=[("whatever", "bin")],
            X=pd.DataFrame({"A": [0, 1]}),
            y=np.array([0, 1]),
            signed=True,
        )

def test_subgroup_gap_returns_float(monkeypatch):
    monkeypatch.setattr(
        metrics, "subgroup_map_from_conjuncts_dataframe",
        lambda r, Xdf: np.array([True, False], dtype=bool),
        raising=True,
    )
    monkeypatch.setattr(
        metrics, "evaluate_subgroup_discrepancy",
        lambda mask, y_vec: 0.0,
        raising=True,
    )

    out = metrics.subgroup_gap(
        rule=[("r", "u")],
        X=pd.DataFrame({"A": [0, 1]}),
        y=np.array([0, 1]),
        signed=False,
    )
    assert isinstance(out, float)
