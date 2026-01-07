import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from humancompatible.detect.binarizer.Binarizer import Bin
from humancompatible.detect.methods.msd import get_conjuncts_MSD
from humancompatible.detect.helpers.prepare import prepare_dataset

logger = logging.getLogger(__name__)


def most_biased_subgroup(
    X: pd.DataFrame,
    y: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    verbose: int = 1,
    method_kwargs: Dict[str, Any] | None = None,
) -> List[Tuple[int, Bin]]:
    """
    Identify the protected subgroup with the largest absolute difference in outcome rates.

    The procedure:
        1. Cleans, encodes, and optionally downsamples the data via `prepare_dataset`.
        2. Runs a subgroup-search routine on the binarised arrays.
           (Currently, only Maximum Subgroup Discrepancy - `method == "MSD"` - is implemented.)

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target column; must have the same number of rows as `X`.
        protected_list (list[str] | None, default None): Names of columns regarded
            as protected attributes. When `None`, every column in `X` is treated
            as protected.
        continuous_list (list[str] | None, default None): Columns that should be
            treated as continuous when building bins.
        fp_map (dict[str, Callable[[Any], int]] | None, default None): Optional per-feature
            recoding map to apply before binarisation.
        seed (int | None, default None): Seed for the random generator controlling
            subsampling and solver randomness.
        n_samples (int, default 1_000_000): Upper bound on the number of rows kept
            after random subsampling.
        method (str, default "MSD"): Subgroup-search routine to invoke. Only
            `"MSD"` is supported at present.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen `method` (for MSD these include
            `time_limit`, `n_min`, `solver`, etc.).

    Returns:
        list[tuple[int, Bin]]: Conjunctive rule describing the most biased subgroup.
        Each element is a pair `(feature_index, Bin)`.

    Raises:
        ValueError: If `method` is not recognised.
    """
    
    if seed is not None:
        if verbose >= 1: logger.info(f"Seeding the run with seed={seed} for searching the `rule`.")
        np.random.seed(seed)
    
    if protected_list is None:
        if verbose >= 1: logger.info("Assuming all attributes are protected")
        protected_list = list(X.columns)
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
        verbose=verbose,
    )

    X_bin = binarizer.encode(X_prot, include_binary_negations=True)
    y_bin = binarizer.encode_y(y)

    if method == "MSD":
        indices = get_conjuncts_MSD(
            X_bin,
            y_bin,
            verbose=verbose,
            **method_kwargs
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
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    verbose: int = 1,
    method_kwargs: Dict[str, Any] | None = None,
) -> List[Tuple[int, Bin]]:
    """
    Load a CSV file, split it into features and target, and return the subgroup
    rule with the largest absolute gap in outcome rates.

    The helper:
        1. Reads the CSV located at `csv_path`.
        2. Separates the target column (`target_col`) from the remaining
           feature columns.
        3. Passes the data to `most_biased_subgroup` along with any optional
           configuration parameters.
        4. Returns the conjunctive rule that characterises the most biased
           protected subgroup.

    Args:
        csv_path (Path | str): Path to the CSV file.
        target_col (str): Name of the target column inside the CSV.
        protected_list (list[str] | None, default None): Columns treated as
            protected attributes. If `None`, every feature column is treated
            as protected.
        continuous_list (list[str] | None, default None): Protected columns
            handled as continuous when creating bins.
        fp_map (dict[str, Callable[[Any], int]] | None, default None): Optional map for
            recoding feature values before binarisation.
        seed (int | None, default None): Seed for random subsampling and any
            solver randomness.
        n_samples (int, default 1_000_000): Maximum number of rows retained
            after random subsampling.
        method (str, default "MSD"): Subgroup-search routine. Only "MSD"
            is currently supported.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen `method`.

    Returns:
        list[tuple[int, Bin]]: Conjunction describing the most biased subgroup,
            where each element is a pair `(feature_index, Bin)`.

    Raises:
        ValueError: If `target_col` is missing from the CSV.
        ValueError: If `method` is unsupported (propagated from
            `most_biased_subgroup`).
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the CSV file.")
    X_df = df.drop(columns=[target_col])
    y_df = pd.DataFrame(df[target_col])

    if protected_list is None:
        if verbose >= 1: logger.info("Assuming all attributes are protected")
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
        protected_list=protected_list,
        continuous_list=continuous_list,
        fp_map=fp_map,
        seed=seed,
        n_samples=n_samples,
        method=method,
        verbose=verbose,
        method_kwargs=method_kwargs,
    )


def most_biased_subgroup_two_samples(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    verbose: int = 1,
    method_kwargs: Dict[str, Any] | None = None,
) -> List[Tuple[int, Bin]]:
    """
    Identify the subgroup whose prevalence differs the most between two datasets.

    The helper:
        1. Verifies that *X1* and *X2* share the same columns.
        2. Concatenates the two frames and builds a synthetic target:
           0 for rows from *X1*, 1 for rows from *X2*.
        3. Forwards the combined data to `most_biased_subgroup` and returns
           the resulting rule.

    Args:
        X1 (pd.DataFrame): First sample.
        X2 (pd.DataFrame): Second sample. Must have identical columns to *X1*.
        protected_list (list[str] | None, default None): Columns to treat as
            protected. If None, every column is treated as protected.
        continuous_list (list[str] | None, default None): Protected columns
            handled as continuous when binning.
        fp_map (dict[str, Callable[[Any], int]] | None, default None): Optional per-feature
            recoding map applied before binarisation.
        seed (int | None, default None): Seed governing subsampling and solver
            randomness.
        n_samples (int, default 1_000_000): Maximum number of rows kept after
            random subsampling.
        method (str, default "MSD"): Subgroup-search routine; only "MSD"
            is currently implemented.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen *method*.

    Returns:
        list[tuple[int, Bin]]: Conjunctive rule describing the subgroup whose
        prevalence gap between *X1* and *X2* is largest.

    Raises:
        ValueError: If *X1* and *X2* do not have identical columns.
        ValueError: If *method* is unsupported (propagated from
            `most_biased_subgroup`).
    """
    
    if X1.columns.tolist() != X2.columns.tolist():
        raise ValueError("The samples must have the same features")

    X_df = pd.concat([X1, X2])
    y = np.concatenate([
        np.zeros(X1.shape[0], dtype=int),
        np.ones(X2.shape[0], dtype=int),
    ])
    y_df = pd.DataFrame(y, columns=["target"])

    if protected_list is None:
        if verbose >= 1: logger.info("Assuming all attributes are protected")
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
        protected_list=protected_list,
        continuous_list=continuous_list,
        fp_map=fp_map,
        seed=seed,
        n_samples=n_samples,
        method=method,
        verbose=verbose,
        method_kwargs=method_kwargs,
    )
