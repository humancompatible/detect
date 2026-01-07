from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import humancompatible.detect.evaluate_bias as eb


class _Binarizer:
    """Only used to check identity is passed to check_l_inf_gap."""
    pass


# =====
# Tests for evaluate_biased_subgroup (MSD)
# =====
def test_evaluate_biased_subgroup_msd(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 1], "B": [1, 0, 1]})
    y = pd.DataFrame({"target": [1, 0, 1]})

    fake_bin = _Binarizer()
    captured = {}

    def _fake_prepare_dataset(X_, Y_, n_samples, protected_attrs, continuous_feats, feature_processing, verbose):
        X_prot = X_[["A", "B"]]
        y_ser = Y_["target"]
        return fake_bin, X_prot, y_ser

    def _fake_evaluate_MSD(X_prot, y_ser, verbose, **kwargs):
        assert list(X_prot.columns) == ["A", "B"]
        assert isinstance(y_ser, pd.Series)
        captured["method_kwargs"] = deepcopy(kwargs)
        assert "rule" in captured["method_kwargs"]
        return 0.123

    monkeypatch.setattr(eb, "prepare_dataset", _fake_prepare_dataset, raising=True)
    monkeypatch.setattr(eb, "evaluate_MSD", _fake_evaluate_MSD, raising=True)

    rule = [("idx", "dummy-bin")]
    method_kwargs_in = {"rule": rule, "signed": False, "solver": "gurobi"}

    out = eb.evaluate_biased_subgroup(
        X, y,
        protected_list=["A", "B"],
        continuous_list=["A"],
        fp_map={"A": int},
        seed=7,
        n_samples=500,
        method="MSD",
        verbose=1,
        method_kwargs=method_kwargs_in,
    )

    assert out == pytest.approx(0.123)
    assert method_kwargs_in == {"rule": rule, "signed": False, "solver": "gurobi"}

def test_evaluate_biased_subgroup_msd_missing_rule_raises(monkeypatch):
    X = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    y = pd.DataFrame({"target": [1, 0]})

    def _fake_prepare_dataset(X_, Y_, *args, **kwargs):
        return _Binarizer(), X_[["A", "B"]], Y_["target"]

    monkeypatch.setattr(eb, "prepare_dataset", _fake_prepare_dataset, raising=True)

    with pytest.raises(ValueError, match="must include a 'rule'"):
        eb.evaluate_biased_subgroup(
            X, y,
            protected_list=["A", "B"],
            continuous_list=[],
            fp_map={},
            seed=None,
            n_samples=10,
            method="MSD",
            method_kwargs={},   # no 'rule'
        )


# =====
# Tests for evaluate_biased_subgroup (l_inf)
# =====
def test_evaluate_biased_subgroup_linf(monkeypatch):
    X = pd.DataFrame({"A": [0, 1, 1], "B": [1, 0, 1]})
    y = pd.DataFrame({"target": [1, 0, 1]})

    fake_bin = _Binarizer()
    captured = {}

    def _fake_prepare_dataset(X_, Y_, n_samples, protected_attrs, continuous_feats, feature_processing, verbose):
        return fake_bin, X_[["A", "B"]], Y_["target"]

    def _fake_check_l_inf_gap(X_prot, y_ser, *, binarizer, **kwargs):
        assert binarizer is fake_bin
        captured["method_kwargs"] = deepcopy(kwargs)
        # pretend result is "satisfied" (<= delta) -> True
        return True

    monkeypatch.setattr(eb, "prepare_dataset", _fake_prepare_dataset, raising=True)
    monkeypatch.setattr(eb, "check_l_inf_gap", _fake_check_l_inf_gap, raising=True)

    method_kwargs_in = {
        "feature_involved": "A",
        "subgroup_to_check": 1,
        "delta": 0.05,
        "solver": "ignored-here",
    }

    out = eb.evaluate_biased_subgroup(
        X, y,
        protected_list=["A", "B"],
        continuous_list=[],
        fp_map={},
        seed=0,
        n_samples=100,
        method="l_inf",
        verbose=1,
        method_kwargs=method_kwargs_in,
    )

    assert out is True
    assert "solver" in method_kwargs_in

def test_evaluate_biased_subgroup_unknown_method(monkeypatch):
    X = pd.DataFrame({"A": [0, 1]})
    y = pd.DataFrame({"target": [1, 0]})

    def _fake_prepare_dataset(X_, Y_, *args, **kwargs):
        return _Binarizer(), X_, Y_["target"]

    monkeypatch.setattr(eb, "prepare_dataset", _fake_prepare_dataset, raising=True)

    with pytest.raises(ValueError, match="not supported"):
        eb.evaluate_biased_subgroup(
            X, y,
            protected_list=None, continuous_list=None, fp_map=None,
            seed=None, n_samples=5,
            method="NOT_A_METHOD",
            method_kwargs=None,
        )


# =====
# Tests for evaluate_biased_subgroup_csv
# =====
def test_evaluate_biased_subgroup_csv(tmp_path: Path, monkeypatch):
    df = pd.DataFrame({
        "A": [0, 1, 0],
        "B": [1, 1, 0],
        "target": [1, 0, 1],
    })
    csvp = tmp_path / "toy.csv"
    df.to_csv(csvp, index=False)

    captured = {}

    def _fake_evaluate_biased_subgroup(
        X_df, y_df, protected_list, continuous_list, fp_map, seed, n_samples, method, verbose, method_kwargs
    ):
        assert list(X_df.columns) == ["A", "B"]
        assert list(y_df.columns) == ["target"]
        captured["protected_list"] = protected_list
        return 0.5

    monkeypatch.setattr(eb, "evaluate_biased_subgroup", _fake_evaluate_biased_subgroup, raising=True)

    out = eb.evaluate_biased_subgroup_csv(
        csv_path=csvp, target_col="target",
        protected_list=None, continuous_list=None, fp_map=None,
        seed=1, n_samples=10, method="MSD", verbose=1, method_kwargs=None,
    )

    assert out == pytest.approx(0.5)
    # When protected_list=None, defaults to all feature cols
    assert captured["protected_list"] == ["A", "B"]

def test_evaluate_biased_subgroup_csv_missing_target_raises(tmp_path: Path):
    df = pd.DataFrame({"A": [1], "B": [0]})
    csvp = tmp_path / "no_target.csv"
    df.to_csv(csvp, index=False)

    with pytest.raises(ValueError, match="Target column 'target' is missing"):
        eb.evaluate_biased_subgroup_csv(
            csv_path=csvp, target_col="target",
            protected_list=None, continuous_list=None, fp_map=None,
            seed=None, n_samples=10, method="MSD", method_kwargs=None,
        )


# =====
# Tests for evaluate_biased_subgroup_two_samples
# =====
def test_evaluate_biased_subgroup_two_samples(monkeypatch):
    X1 = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    X2 = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 1]})

    captured = {}

    def _fake_evaluate_biased_subgroup(
        X_df, y_df, protected_list, continuous_list, fp_map, seed, n_samples, method, verbose, method_kwargs
    ):
        assert list(X_df.columns) == ["A", "B"]
        # Synthetic target: 0 for X1 rows, 1 for X2 rows
        y = y_df["target"].to_numpy()
        assert (y[: len(X1)] == 0).all()
        assert (y[len(X1):] == 1).all()
        captured["protected_list"] = protected_list
        return 0.7

    monkeypatch.setattr(eb, "evaluate_biased_subgroup", _fake_evaluate_biased_subgroup, raising=True)

    out = eb.evaluate_biased_subgroup_two_samples(
        X1, X2,
        protected_list=None, continuous_list=None, fp_map=None,
        seed=2, n_samples=999, method="MSD", verbose=1, method_kwargs=None,
    )
    assert out == pytest.approx(0.7)
    assert captured["protected_list"] == ["A", "B"]

def test_evaluate_biased_subgroup_two_samples_mismatched_columns_raises():
    X1 = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    X2 = pd.DataFrame({"A": [1, 1], "C": [0, 0]})
    with pytest.raises(ValueError, match="same features"):
        eb.evaluate_biased_subgroup_two_samples(
            X1, X2,
            protected_list=None, continuous_list=None, fp_map=None,
            seed=None, n_samples=10, method="MSD", method_kwargs=None,
        )
