import numpy as np
import pandas as pd

from src.detect.dnf_bias.dnf_bias import prepare_dataset

def _synthetic_dataframe(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "SEX": rng.integers(1, 3, size=n),
            "RAC1P": rng.integers(1, 6, size=n),
            "AGEP": rng.integers(18, 80, size=n),
            "POBP": rng.integers(100, 900, size=n),
            "PINCP": rng.integers(10_000, 80_000, size=n),
        }
    )
    target = pd.DataFrame({"PINCP_gt_40k": (df["PINCP"] > 40_000)})
    return df, target

def test_prepare_dataset_basic():
    X, y = _synthetic_dataframe(50)

    bin_all, bin_prot, X_proc, y_proc, X_prot = prepare_dataset(
        input_data=X,
        target_data=y,
        n_max=30,
        protected_attrs=["SEX", "RAC1P"],
        continuous_feats=["AGEP", "PINCP"],
        feature_processing={"POBP": 100},
        seed=42,
    )

    assert X_proc.shape[0] == 30
    assert y_proc.shape[0] == 30

    assert X_proc["POBP"].max() < 10

    assert set(X_prot.columns) == {"SEX", "RAC1P"}

    enc = bin_all.encode(X_proc.values)
    assert enc.dtype == bool
    
