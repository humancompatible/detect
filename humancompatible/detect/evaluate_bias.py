import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List

from humancompatible.detect.methods.l_inf import check_l_inf_gap
from humancompatible.detect.methods.msd import evaluate_MSD
from humancompatible.detect.helpers.prepare import prepare_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def evaluate_biased_subgroup(
    X: pd.DataFrame,
    y: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> float:
    """
    Evaluate how far a *given* subgroup departs from the reference population.

    Workflow  
        1. The data are cleaned, encoded and prepared via
           `prepare_dataset`.  
        2. One of two evaluation routines is run:
           * `method == "MSD"` - compute the Maximum Subgroup Discrepancy for
             the rule supplied in `method_kwargs["rule"]`. 

           * `method == "l_inf"` - call `check_l_inf_gap`, which checks whether
             the subgroup's positive-class histogram deviates from the whole
             sample by more than the supplied `delta`.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target column, same number of rows as `X`.
        protected_list (list[str] | None, default None): Columns regarded as
            protected. If None, every column in `X` is treated as protected.
        continuous_list (list[str] | None, default None): Columns treated as
            continuous when creating bins.
        fp_map (dict[str, Callable] | None, default None): Optional per-feature
            recoding applied before binarisation.
        seed (int | None, default None): Seed for subsampling / solver
            randomness.
        n_samples (int, default 1_000_000): Maximum number of rows kept after
            random subsampling.
        method (str, default "MSD"): Evaluation routine to invoke.
            Supported values: "MSD", "l_inf".
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen `method`.  
            For MSD you must provide `rule`; for l_inf you typically supply
            `feature_involved`, `subgroup_to_check` and `delta`.

    Returns:
        float:  
            * MSD - the subgroup discrepancy (signed or absolute, depending on
              flags inside `method_kwargs`).  
            * l_inf - 1.0 if the subgroup gap is <= `delta`, otherwise 0.0.

    Raises:
        ValueError: If the requested `method` is unknown, or required keys are
            missing from `method_kwargs`.
    """
    
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

    if method == "MSD":
        if "rule" not in method_kwargs:
            raise ValueError("method_kwargs for MSD must include a 'rule'.")
        val = evaluate_MSD(
            X_prot, y, **method_kwargs
        )
    
    elif method == "l_inf":
        val = check_l_inf_gap(
            X_prot, y, binarizer=binarizer, **method_kwargs
        )
        
    else:
        raise ValueError(f"Method '{method}' is not supported.")

    return val


def evaluate_biased_subgroup_csv(
    csv_path: Path | str,
    target_col: str,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> float:
    """
    Load a CSV file, split it into features / target and forward everything to
    `evaluate_biased_subgroup`.

    Workflow  
        1. The file at `csv_path` is being read.
        2. `target_col` is removed from the frame and kept separately
           (`X_df`, `y_df`).  
        3. The helper `evaluate_biased_subgroup` is called with exactly the same
           keyword interface.

    Args:
        csv_path (Path | str): Location of the CSV file.
        target_col (str): Name of the column that holds the target variable.
        protected_list (list[str] | None, default None): Columns regarded as
            protected. If None, every feature column is treated as protected.
        continuous_list (list[str] | None, default None): Columns that should be
            treated as continuous when creating bins.
        fp_map (dict[str, Callable[[Any], int]] | None, default None): Optional
            mapping *column -> recoding-function* applied before binarisation.
        seed (int | None, default None): Seed for any randomness downstream
            (sub-sampling, solver search).
        n_samples (int, default 1_000_000): Maximum number of rows kept after
            random subsampling.
        method (str, default "MSD"): Evaluation routine to invoke.
            Supported values: "MSD", "l_inf".
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen `method`
            (see `evaluate_biased_subgroup` for details).
    
    Returns:
        float:  
            * MSD - subgroup discrepancy value.  
            * l_inf - 1.0 if the gap is <= `delta`, otherwise 0.0.

    Raises:
        ValueError: If `target_col` is absent from the CSV or if the delegated
            call to `evaluate_biased_subgroup` raises a `ValueError`.
    """
    
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

    return evaluate_biased_subgroup(
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


def evaluate_biased_subgroup_two_samples(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> float:
    
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

    return evaluate_biased_subgroup(
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
