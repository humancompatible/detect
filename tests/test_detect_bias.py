from copy import deepcopy
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

import humancompatible.detect.detect_bias as db


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

class _DataHandler:
    def __init__(self, features: List[Any]):
        self.features = features

class _Binarizer:
    def __init__(self):
        self._encs = [_Bin("A", 0), _Bin("B", 1)]
        self.data_handler = _DataHandler(features=[_Feature("A"), _Feature("B")])

    def encode(self, X, include_binary_negations=True):
        # Shape doesn't matter for our tests; return a plausible bool matrix.
        return np.array([[True, False], [False, True]], dtype=bool)

    def encode_y(self, y):
        return np.array([True, False], dtype=bool)

    def get_bin_encodings(self, include_binary_negations=True):
        return self._encs


# =====
# Tests for most_biased_subgroup
# =====
def test_most_biased_subgroup(monkeypatch):
    """
    - prepare_dataset is called and returns (binarizer, X_prot, y)
    - get_conjuncts_MSD is passed through method_kwargs
    - rule indices are mapped back to (feature_index, encoding_bin)
    """
    X = pd.DataFrame({"A": [0, 1, 1], "B": [1, 0, 1]})
    y = pd.DataFrame({"target": [1, 0, 1]})

    fake_bin = _Binarizer()

    captured = {}

    def _fake_prepare_dataset(X_, y_, n_samples, protected_attrs, continuous_feats, feature_processing, verbose):
        assert list(X_.columns) == ["A", "B"]
        assert y_.shape[0] == X_.shape[0]
        captured["n_samples"] = n_samples
        captured["protected_attrs"] = protected_attrs
        captured["continuous_feats"] = continuous_feats
        captured["feature_processing"] = feature_processing
        captured["verbose"] = verbose
        return fake_bin, X_[["A", "B"]], y_["target"]

    def _fake_get_conjuncts_MSD(X_bin, y_bin, **kwargs):
        captured["method_kwargs"] = deepcopy(kwargs)
        # Pick the second encoding (feature "B")
        return [1]

    monkeypatch.setattr(db, "prepare_dataset", _fake_prepare_dataset, raising=True)
    monkeypatch.setattr(db, "get_conjuncts_MSD", _fake_get_conjuncts_MSD, raising=True)

    rule = db.most_biased_subgroup(
        X, y,
        protected_list=["A", "B"],
        continuous_list=["A"],
        fp_map={"A": int},
        seed=123,
        n_samples=1000,
        method="MSD",
        verbose=1,
        method_kwargs={"time_limit": 7, "n_min": 2},
    )

    # We asked get_conjuncts_MSD to return index 1 => feature "B"
    assert rule == [(1, fake_bin.get_bin_encodings()[1])]

    assert captured["n_samples"] == 1000
    assert captured["protected_attrs"] == ["A", "B"]
    assert captured["continuous_feats"] == ["A"]
    assert captured["feature_processing"] == {"A": int}
    assert captured["verbose"] == 1
    assert captured["method_kwargs"]["time_limit"] == 7
    assert captured["method_kwargs"]["n_min"] == 2

def test_most_biased_subgroup_method_raises(monkeypatch):
    X = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    y = pd.DataFrame({"target": [1, 0]})

    with pytest.raises(ValueError):
        db.most_biased_subgroup(
            X, y, protected_list=["A", "B"], continuous_list=[], fp_map={},
            seed=None, n_samples=10, method="NOT_A_METHOD", method_kwargs={},
        )


# =====
# Tests for most_biased_subgroup_csv
# =====
def test_most_biased_subgroup_csv(tmp_path: Path, monkeypatch):
    df = pd.DataFrame({
        "A": [0, 1, 0],
        "B": [1, 1, 0],
        "target": [1, 0, 1],
    })
    csvp = tmp_path / "toy.csv"
    df.to_csv(csvp, index=False)

    fake_rule = [("idx", _Bin("B", 1))]

    captured = {}

    def _fake_most_biased_subgroup(
        X_df, y_df, protected_list, continuous_list, fp_map, seed, n_samples, method, verbose, method_kwargs
    ):
        assert list(X_df.columns) == ["A", "B"]
        assert list(y_df.columns) == ["target"]
        captured["protected_list"] = protected_list
        return fake_rule

    monkeypatch.setattr(db, "most_biased_subgroup", _fake_most_biased_subgroup, raising=True)

    out = db.most_biased_subgroup_csv(
        csv_path=csvp, target_col="target",
        protected_list=None, continuous_list=None, fp_map=None,
        seed=7, n_samples=100, method="MSD", verbose=1, method_kwargs=None,
    )
    assert out == fake_rule
    assert captured["protected_list"] == ["A", "B"]

def test_most_biased_subgroup_csv_missing_target_raises(tmp_path: Path):
    df = pd.DataFrame({"A": [1], "B": [0]})
    csvp = tmp_path / "no_target.csv"
    df.to_csv(csvp, index=False)

    with pytest.raises(ValueError, match="Target column 'target' is missing"):
        db.most_biased_subgroup_csv(
            csv_path=csvp, target_col="target",
            protected_list=None, continuous_list=None, fp_map=None,
            seed=None, n_samples=10, method="MSD", method_kwargs=None,
        )


# =====
# Tests for most_biased_subgroup_two_samples
# =====
def test_most_biased_subgroup_two_samples(monkeypatch):
    X1 = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    X2 = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 1]})

    fake_rule = [("idx", _Bin("A", 0))]

    captured = {}

    def _fake_most_biased_subgroup(
        X_df, y_df, protected_list, continuous_list, fp_map, seed, n_samples, method, verbose, method_kwargs
    ):
        assert list(X_df.columns) == ["A", "B"]
        # y is 0 for X1 rows, 1 for X2 rows
        y = y_df["target"].to_numpy()
        assert (y[: len(X1)] == 0).all()
        assert (y[len(X1) :] == 1).all()
        captured["protected_list"] = protected_list
        return fake_rule

    monkeypatch.setattr(db, "most_biased_subgroup", _fake_most_biased_subgroup, raising=True)

    out = db.most_biased_subgroup_two_samples(
        X1, X2,
        protected_list=None, continuous_list=None, fp_map=None,
        seed=1, n_samples=999, method="MSD", verbose=1, method_kwargs=None,
    )
    assert out == fake_rule
    assert captured["protected_list"] == ["A", "B"]

def test_most_biased_subgroup_two_samples_mismatched_columns_raises():
    X1 = pd.DataFrame({"A": [0, 1], "B": [1, 0]})
    X2 = pd.DataFrame({"A": [1, 1], "C": [0, 0]})

    with pytest.raises(ValueError, match="same features"):
        db.most_biased_subgroup_two_samples(
            X1, X2,
            protected_list=None, continuous_list=None, fp_map=None,
            seed=None, n_samples=10, method="MSD", method_kwargs=None,
        )
