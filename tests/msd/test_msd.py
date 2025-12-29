from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import humancompatible.detect.methods.msd.msd as msd


# =====
# Tests for get_conjuncts_MSD
# =====
def test_get_conjuncts_MSD_forwards_args_and_returns_indices(monkeypatch):
    captured = {}

    class _OneRule:
        def find_rule(self, X_bin, y_bin, *, n_min, time_limit, solver_name, verbose):
            captured["n_min"] = n_min
            captured["time_limit"] = time_limit
            captured["solver_name"] = solver_name
            captured["verbose"] = verbose
            # return a deterministic "solution"
            return [2, 5], True

    monkeypatch.setattr(msd, "OneRule", _OneRule, raising=True)

    Xb = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    yb = np.array([1, 0], dtype=bool)

    indices = msd.get_conjuncts_MSD(
        X_bin=Xb, y_bin=yb, time_limit=12, n_min=3, solver="gurobi", verbose=1
    )
    assert indices == [2, 5]
    assert captured == {"n_min": 3, "time_limit": 12, "solver_name": "gurobi", "verbose": 1}

def test_get_conjuncts_MSD_errors(monkeypatch):
    class _OneRule:
        def find_rule(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(msd, "OneRule", _OneRule, raising=True)

    with pytest.raises(ValueError, match="boom"):
        msd.get_conjuncts_MSD(
            X_bin=np.zeros((1, 1), dtype=bool),
            y_bin=np.zeros((1,), dtype=bool),
            time_limit=1,
            n_min=0,
            solver="appsi_highs",
        )


# =====
# Tests for evaluate_MSD
# =====
def test_evaluate_MSD_absolute(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 1, 0]})
    y = np.array([[1], [0], [1], [0]], dtype=int)

    # make subgroup mask deterministic
    mask = np.array([True, False, True, False], dtype=bool)
    called = {}

    def _fake_subgroup_map(rule, X_df):
        assert isinstance(X_df, pd.DataFrame)
        return mask

    def _fake_eval_abs(subgroup_mask, y_vec, verbose):
        assert isinstance(y_vec, np.ndarray)
        assert y_vec.ndim == 1
        called["y"] = deepcopy(y_vec)
        called["mask"] = deepcopy(subgroup_mask)
        return 0.25

    monkeypatch.setattr(msd, "subgroup_map_from_conjuncts_dataframe", _fake_subgroup_map, raising=True)
    monkeypatch.setattr(msd, "evaluate_subgroup_discrepancy", _fake_eval_abs, raising=True)

    val = msd.evaluate_MSD(X=X, y=y, rule=[("anything", "goes")], signed=False, verbose=1)
    assert val == pytest.approx(0.25)
    assert np.array_equal(called["mask"], mask)
    assert np.array_equal(called["y"], np.array([1, 0, 1, 0]))

def test_evaluate_MSD_signed(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 1, 0], dtype=int)

    mask = np.array([False, True, False, True], dtype=bool)
    seen = {}

    def _fake_subgroup_map(rule, X_df):
        return mask

    def _fake_signed(subgroup_mask, y_vec, verbose):
        # record inputs; return a distinct value to ensure correct branch used
        seen["ndim"] = y_vec.ndim
        return -0.123

    monkeypatch.setattr(msd, "subgroup_map_from_conjuncts_dataframe", _fake_subgroup_map, raising=True)
    monkeypatch.setattr(msd, "signed_subgroup_discrepancy", _fake_signed, raising=True)

    val = msd.evaluate_MSD(X=X, y=y, rule=[("r", "u")], signed=True, verbose=1)
    assert val == pytest.approx(-0.123)
    assert seen["ndim"] == 1  # flattened/1D

def test_evaluate_MSD_returns_float(monkeypatch):
    monkeypatch.setattr(
        msd, "subgroup_map_from_conjuncts_dataframe",
        lambda rule, X_df: np.array([True, False], dtype=bool),
        raising=True,
    )
    monkeypatch.setattr(
        msd, "evaluate_subgroup_discrepancy",
        lambda mask, y_vec, verbose: 0.0,
        raising=True,
    )
    out = msd.evaluate_MSD(
        X=pd.DataFrame({"A": [0, 1]}),
        y=np.array([0, 1]),
        rule=[("x", "y")],
        signed=False,
        verbose=1,
    )
    assert isinstance(out, float)
