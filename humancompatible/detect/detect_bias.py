import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from humancompatible.detect.binarizer.Binarizer import Bin
from humancompatible.detect.methods.one_rule import OneRule
from humancompatible.detect.prepare import prepare_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def most_biased_subgroup(
    X: pd.DataFrame,
    y: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> List[Tuple[int, Bin]]:
    
    if seed is not None:
        logger.info(f"Seeding the run with seed={seed}")
        np.random.seed(seed)
    
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    binarizer, X_prot, y = prepare_dataset(
        X,
        y,
        n_samples,
        protected_attrs=protected_list,
        continuous_feats=continuous_list,
        feature_processing=fp_map,
    )

    X_bin = binarizer.encode(X_prot, include_binary_negations=True)
    y_bin = binarizer.encode_y(y)

    if method == "MSD":
        indices = get_conjuncts_MSD(
            X_bin,
            y_bin,
            **method_kwargs
        )


        mio = OneRule()
        indices, _ = mio.find_rule(
            X_bin, y_bin, **method_kwargs
        )
    else:
        raise ValueError(f"Method '{method}' is not supported.")

    encodings = binarizer.get_bin_encodings(include_binary_negations=True)
    feats = binarizer.data_handler.features
    rule = [(feats.index(encodings[i].feature), encodings[i]) for i in indices]

    return rule


def most_biased_subgroup_csv(
    csv_path: Path | str,
    target_col: str,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[float, List[Tuple[int, Bin]]]:

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the CSV file.")
    X_df = df.drop(columns=[target_col])
    y_df = pd.DataFrame(df[target_col])

    if protected_list is None:
        logger.info("Assuming all attributes are protected")
        protected_list = list(X_df.columns)
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    return most_biased_subgroup(
        X_df,
        y_df,
        protected_list,
        continuous_list,
        fp_map,
        seed,
        n_samples,
        method,
        method_kwargs,
    )


def most_biased_subgroup_two_samples(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[float, List[Tuple[int, Bin]]]:
    
    if X1.columns.tolist() != X2.columns.tolist():
        raise ValueError("The samples must have the same features")

    X_df = pd.concat([X1, X2])
    y = np.concatenate([
        np.zeros(X1.shape[0], dtype=int),
        np.ones (X2.shape[0], dtype=int),
    ])
    y_df = pd.DataFrame(y, columns=["target"])

    if protected_list is None:
        logger.info("Assuming all attributes are protected")
        protected_list = X_df.columns.tolist()
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    return most_biased_subgroup(
        X_df,
        y_df,
        protected_list,
        continuous_list,
        fp_map,
        seed,
        n_samples,
        method,
        method_kwargs,
    )
