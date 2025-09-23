import numpy as np
import pandas as pd
import pytest

import humancompatible.detect.methods.l_inf.l_inf as linf_mod


class _Feature:
    def __init__(self, value_mapping):
        self.value_mapping = dict(value_mapping)

class _DataHandler:
    def __init__(self, cols, value_maps):
        self.feature_names = list(cols)                # e.g., ["SEX", "SCHL", "RACE"]
        self.features = [_Feature(value_maps[c]) for c in self.feature_names]

    def encode(self, X: pd.DataFrame, one_hot: bool = False) -> np.ndarray:
        return X[self.feature_names].to_numpy(dtype=int)

class _Binarizer:
    def __init__(self, cols, value_maps):
        self.data_handler = _DataHandler(cols, value_maps)

    def encode_y(self, y: pd.Series | pd.DataFrame) -> np.ndarray:
        arr = y.to_numpy().ravel()
        # Return 0/1 ints; code compares (== 1)
        return arr.astype(int)


# =====
# Tests for check_l_inf_gap
# =====
def test_check_l_inf_gap_calls_linprog_with_expected_shapes_and_returns_1(monkeypatch):
    # Data: three features; subgroup on SEX == 1; ensure subgroup among positives is non-empty
    X = pd.DataFrame({
        "SEX":  [0, 1, 1, 0, 1],
        "SCHL": [2, 2, 1, 0, 1],
        "RACE": [3, 3, 3, 2, 1],
    })
    y = pd.Series([1, 1, 0, 1, 1])  # positives at indices 0,1,3,4; among them SEX==1 at 1 and 4

    cols = ["SEX", "SCHL", "RACE"]
    value_maps = {
        "SEX":  {0: 0, 1: 1},
        "SCHL": {0: 0, 1: 1, 2: 2},
        "RACE": {1: 1, 2: 2, 3: 3},
    }
    binarizer = _Binarizer(cols, value_maps)

    captured = {}

    def _fake_lin_prog_feas(all_rsh, discr_rsh, delta, num_samples=1.0):
        captured["delta"] = delta
        captured["all_shape"] = tuple(np.shape(all_rsh))
        captured["discr_shape"] = tuple(np.shape(discr_rsh))
        return 0

    monkeypatch.setattr(linf_mod, "lin_prog_feas", _fake_lin_prog_feas)

    out = linf_mod.check_l_inf_gap(
        X=X,
        y=y,
        binarizer=binarizer,
        feature_involved="SEX",
        subgroup_to_check=1,
        delta=0.05,
    )

    assert out == 1.0
    assert captured["delta"] == 0.05

    # Expected bins exclude the tested feature (SEX), so use SCHL and RACE
    bins_schl = int(X["SCHL"].max() + 1)  # 3
    bins_race = int(X["RACE"].max() + 1)  # 4
    prod_bins = bins_schl * bins_race     # 12

    assert captured["all_shape"] == (prod_bins, 1)
    assert captured["discr_shape"] == (prod_bins, 1)

def test_check_l_inf_gap_infeasible_returns_0(monkeypatch):
    # we only care about LP status -> return non-zero
    X = pd.DataFrame({"SEX": [1, 1], "SCHL": [0, 1]})
    y = pd.Series([1, 1])
    binarizer = _Binarizer(
        cols=["SEX", "SCHL"],
        value_maps={"SEX": {0: 0, 1: 1}, "SCHL": {0: 0, 1: 1}},
    )

    monkeypatch.setattr(linf_mod, "lin_prog_feas", lambda a, b, delta, num_samples=1.0: 2)
    out = linf_mod.check_l_inf_gap(X, y, binarizer, feature_involved="SEX", subgroup_to_check=1, delta=0.1)
    assert out == 0.0

def test_check_l_inf_gap_requires_positive_delta():
    X = pd.DataFrame({"SEX": [0, 1]})
    y = pd.Series([1, 0])
    binarizer = _Binarizer(cols=["SEX"], value_maps={"SEX": {0: 0, 1: 1}})

    with pytest.raises(ValueError, match="delta must be positive"):
        linf_mod.check_l_inf_gap(X, y, binarizer, "SEX", 1, delta=0.0)

def test_check_l_inf_gap_unknown_feature_raises_keyerror():
    X = pd.DataFrame({"SEX": [0, 1]})
    y = pd.Series([1, 0])
    binarizer = _Binarizer(cols=["SEX"], value_maps={"SEX": {0: 0, 1: 1}})

    with pytest.raises(KeyError, match="Feature 'FOO' not in protected set"):
        linf_mod.check_l_inf_gap(X, y, binarizer, "FOO", 1, delta=0.1)

def test_check_l_inf_gap_invalid_subgroup_value_lists_allowed():
    X = pd.DataFrame({"SEX": [0, 1], "SCHL": [0, 1]})
    y = pd.Series([1, 1])
    binarizer = _Binarizer(
        cols=["SEX", "SCHL"],
        value_maps={"SEX": {0: 0, 1: 1}, "SCHL": {0: 0, 1: 1}},
    )

    with pytest.raises(KeyError) as ei:
        linf_mod.check_l_inf_gap(X, y, binarizer, feature_involved="SEX", subgroup_to_check=999, delta=0.1)

    # Error should include allowed raw values
    msg = str(ei.value)
    assert "not a valid value for 'SEX'" in msg
    assert "Allowed: [0, 1]" in msg
