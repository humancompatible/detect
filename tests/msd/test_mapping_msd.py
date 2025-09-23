import numpy as np
import pandas as pd
import pytest

from humancompatible.detect.methods.msd.mapping_msd import subgroup_map_from_conjuncts_binarized, subgroup_map_from_conjuncts_dataframe


# =====
# Tests for subgroup_map_from_conjuncts_binarized
# =====
def test_binarized_and_logic():
    X = np.array([
        [True,  True,  False, True],
        [True,  False, True,  True],
        [False, True,  True,  False],
        [True,  True,  True,  True],
    ], dtype=bool)

    # Require col 0 AND col 1
    m = subgroup_map_from_conjuncts_binarized([0, 1], X)
    np.testing.assert_array_equal(m, np.array([ True, False, False,  True]))

    # Require only col 2
    m = subgroup_map_from_conjuncts_binarized([2], X)
    np.testing.assert_array_equal(m, np.array([False,  True,  True,  True]))

def test_binarized_empty_returns_all_true():
    X = np.array([[False, False], [True, False], [False, True]], dtype=bool)
    m = subgroup_map_from_conjuncts_binarized([], X)
    np.testing.assert_array_equal(m, np.array([True, True, True]))

def test_binarized_out_of_bounds_raises():
    X = np.zeros((3, 2), dtype=bool)
    with pytest.raises(IndexError):
        subgroup_map_from_conjuncts_binarized([0, 99], X)



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
# Tests for subgroup_map_from_conjuncts_dataframe
# =====
def test_dataframe_single_conjunct():
    X = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 1, 0, 0]})
    rule = [(0, _Bin("A", 1))]
    m = subgroup_map_from_conjuncts_dataframe(rule, X)
    np.testing.assert_array_equal(m, np.array([False, True, True, False]))

def test_dataframe_two_conjuncts_and_logic():
    X = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 1, 0, 0]})
    rule = [(0, _Bin("A", 1)), (1, _Bin("B", 1))]
    m = subgroup_map_from_conjuncts_dataframe(rule, X)
    np.testing.assert_array_equal(m, np.array([False, True, False, False]))

def test_dataframe_missing_column_raises():
    X = pd.DataFrame({"A": [0, 1, 1, 0]})
    rule = [(0, _Bin("B", 1))]  # B not present
    with pytest.raises(KeyError):
        subgroup_map_from_conjuncts_dataframe(rule, X)

def test_dataframe_ignores_rule_index_and_is_stable_to_reordering():
    # The implementation ignores the numeric index in each (idx, Bin) pair and uses Bin.feature.name.
    X = pd.DataFrame({"A": [1, 0, 1], "B": [0, 0, 1]})
    rule = [(999, _Bin("B", 0))]  # bogus positional index on purpose
    m1 = subgroup_map_from_conjuncts_dataframe(rule, X)
    # Reorder columns; result should be identical
    X_reordered = X[["B", "A"]]
    m2 = subgroup_map_from_conjuncts_dataframe(rule, X_reordered)
    np.testing.assert_array_equal(m1, m2)
