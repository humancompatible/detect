import numpy as np
import pandas as pd

from humancompatible.detect.helpers.prepare import prepare_dataset
from humancompatible.detect.binarizer.Binarizer import Binarizer
from humancompatible.detect.data_handler.features import Contiguous, Categorical, Binary


# =====
# Tests for prepare_dataset
# =====
def test_prepare_basic_clean_map_and_drop_constant():
    # Row 3 has None -> should be removed; 'const' is constant -> should be dropped.
    X = pd.DataFrame({
        "cat": ["a", "b", "a", None],
        "num": [1.0, 2.0, 2.0, 3.0],
        "const": [1, 1, 1, 1],
    })
    y = pd.DataFrame({"target": [True, False, True, False]})

    fp = {"cat": {"a": 0, "b": 1}.__getitem__}

    binarizer, Xp, ys = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=10,
        protected_attrs=["cat", "num", "const"],  # include const on purpose
        continuous_feats=["num"],
        feature_processing=fp,
    )

    assert len(Xp) == 3
    assert len(ys) == 3

    assert "const" not in Xp.columns
    assert set(Xp.columns) == {"cat", "num"}

    assert set(Xp["cat"].unique()) <= {0, 1}

    assert isinstance(binarizer, Binarizer)
    ftypes = {f.name: type(f) for f in binarizer.data_handler.features}
    assert issubclass(ftypes["num"], Contiguous)
    assert issubclass(ftypes["cat"], (Binary, Categorical))

def test_prepare_protected_intersection_only():
    X = pd.DataFrame({"A": [0, 1, 2], "B": [1, 1, 0], "C": [5, 6, 7]})
    y = pd.DataFrame({"target": [True, False, True]})

    _, Xp, ys = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=10,
        protected_attrs=["A", "Z"],  # Z doesn’t exist
        continuous_feats=[],
        feature_processing={},
    )

    assert list(Xp.columns) == ["A"]
    assert len(Xp) == 3 and len(ys) == 3

def test_prepare_downsamples_with_choice(monkeypatch):
    # n_max < n -> code path uses np.random.choice
    X = pd.DataFrame({"A": np.arange(10), "B": np.arange(10, 20)})
    y = pd.DataFrame({"target": [True, False] * 5})

    def fake_choice(n, size, replace):
        assert n == 10 and size == 4 and replace is False
        return np.array([9, 1, 5, 0], dtype=int)

    monkeypatch.setattr(np.random, "choice", fake_choice)

    _, Xp, ys = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=4,
        protected_attrs=["A", "B"],
        continuous_feats=[],
        feature_processing={},
    )

    assert list(Xp.index) == [9, 1, 5, 0]
    assert len(ys) == 4

def test_prepare_no_downsample_uses_permutation(monkeypatch):
    # n_max >= n -> code path uses np.random.permutation
    X = pd.DataFrame({"A": [10, 11, 12], "B": [0, 1, 0]})
    y = pd.DataFrame({"target": [True, False, True]})

    def fake_perm(n):
        assert n == 3
        return np.array([2, 0, 1], dtype=int)

    monkeypatch.setattr(np.random, "permutation", fake_perm)

    _, Xp, ys = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=3,
        protected_attrs=["A", "B"],
        continuous_feats=[],
        feature_processing={},
    )

    assert list(Xp.index) == [2, 0, 1]
    assert list(ys.index) == [2, 0, 1]

def test_prepare_binarizer_target_encoding():
    X = pd.DataFrame({"A": [0, 1, 0, 1]})
    y = pd.DataFrame({"target": [True, False, True, False]})

    binarizer, Xp, ys = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=10,
        protected_attrs=["A"],
        continuous_feats=[],
        feature_processing={},
    )

    encoded = binarizer.encode_y(ys)
    np.testing.assert_array_equal(encoded, ys.values.astype(bool).ravel())
