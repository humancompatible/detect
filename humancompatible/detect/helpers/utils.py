import logging
from pathlib import Path
from typing import Any, List, Dict, Callable, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from humancompatible.detect.binarizer import Bin

logger = logging.getLogger(__name__)


def detect_and_score(
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.DataFrame] = None,
    *,
    # CSV mode
    csv_path: Optional[Path | str] = None,
    target_col: Optional[str] = None,
    # Two-sample mode
    X1: Optional[pd.DataFrame] = None,
    X2: Optional[pd.DataFrame] = None,
    # Common options
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[[Any], int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    verbose: int = 1,
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[List[Tuple[int, Bin]], float | bool]:
    """
    One-shot helper: find the most biased subgroup and return its score.
    Works with three input modes:

    - DataFrame mode: pass X, y
    - CSV mode: pass csv_path, target_col
    - Two-sample mode: pass X1, X2

    It first calls `most_biased_subgroup()` (or similar, depending on the mode) to obtain the rule, then
    evaluates that rule through `evaluate_biased_subgroup()` (depending on the mode).

    Args:
        X (pd.DataFrame | None): Feature matrix.
        y (pd.DataFrame | None): Single-column target aligned with X.
        csv_path (Path | str | None): Path to a CSV file.
        target_col (str | None): Name of the target column in the CSV.
        X1 (pd.DataFrame | None): First dataset (Two-sample mode).
        X2 (pd.DataFrame | None): Second dataset (Two-sample mode).
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
        method (str, default "MSD"): Subgroup-search routine to invoke.
            `"MSD"` or `"l_inf"` is supported at present.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).
        method_kwargs (dict[str, Any] | None, default None): Extra keyword
            arguments forwarded to the chosen `method` (for MSD these include
            `time_limit`, `n_min`, `solver`, etc.).

    Returns:
        tuple[list[tuple[int, Bin]], float | bool]: A pair containing
        (rule, value):

        * **rule** - list of ``(feature_index, Bin)`` pairs [for method == "l_inf" the rule is an empty list].
        * **value** - MSD [return float] or l_inf [return bool], depending on *method*.

    Raises:
        ValueError: If modes are mixed, required arguments for a mode are missing,
            or `method`/`method_kwargs` are invalid.
    """
    from humancompatible.detect.detect_bias import (
        most_biased_subgroup,
        most_biased_subgroup_csv,
        most_biased_subgroup_two_samples,
    )
    from humancompatible.detect.evaluate_bias import (
        evaluate_biased_subgroup,
        evaluate_biased_subgroup_csv,
        evaluate_biased_subgroup_two_samples,
    )

    mode_csv = (csv_path is not None) or (target_col is not None)
    mode_two = (X1 is not None) or (X2 is not None)
    mode_df = (X is not None) or (y is not None)

    modes_set = sum([bool(mode_csv), bool(mode_two), bool(mode_df)])
    if modes_set != 1:
        raise ValueError(
            "Provide exactly one mode:\n"
            "  - DataFrame mode: X and y\n"
            "  - CSV mode: csv_path and target_col\n"
            "  - Two-sample mode: X1 and X2"
        )


    m_kwargs: Dict[str, Any] = {} if method_kwargs is None else deepcopy(method_kwargs)
    
    if method == "l_inf":
        rule = []
    else:
        if mode_csv:
            if not (csv_path and target_col):
                raise ValueError("CSV mode requires csv_path and target_col.")
            rule = most_biased_subgroup_csv(
                csv_path=csv_path,
                target_col=target_col,
                protected_list=protected_list,
                continuous_list=continuous_list,
                fp_map=fp_map,
                seed=seed,
                n_samples=n_samples,
                method="MSD",
                verbose=verbose,
                method_kwargs=m_kwargs,
            )
        elif mode_two:
            if X1 is None or X2 is None:
                raise ValueError("Two-sample mode requires X1 and X2.")
            rule = most_biased_subgroup_two_samples(
                X1, X2,
                protected_list=protected_list,
                continuous_list=continuous_list,
                fp_map=fp_map,
                seed=seed,
                n_samples=n_samples,
                method="MSD",
                verbose=verbose,
                method_kwargs=m_kwargs,
            )
        else:
            if X is None or y is None:
                raise ValueError("DataFrame mode requires X and y.")
            rule = most_biased_subgroup(
                X, y,
                protected_list=protected_list,
                continuous_list=continuous_list,
                fp_map=fp_map,
                seed=seed,
                n_samples=n_samples,
                method=method,
                verbose=verbose,
                method_kwargs=m_kwargs,
            )
        
        m_kwargs = {**m_kwargs, "rule": rule}

    if mode_csv:
        value = evaluate_biased_subgroup_csv(
            csv_path=csv_path,
            target_col=target_col,
            protected_list=protected_list,
            continuous_list=continuous_list,
            fp_map=fp_map,
            seed=seed,
            n_samples=n_samples,
            method=method,
            verbose=verbose,
            method_kwargs=m_kwargs,
        )
    elif mode_two:
        value = evaluate_biased_subgroup_two_samples(
            X1, X2,
            protected_list=protected_list,
            continuous_list=continuous_list,
            fp_map=fp_map,
            seed=seed,
            n_samples=n_samples,
            method=method,
            verbose=verbose,
            method_kwargs=m_kwargs,
        )
    else:
        value = evaluate_biased_subgroup(
            X, y,
            protected_list=protected_list,
            continuous_list=continuous_list,
            fp_map=fp_map,
            seed=seed,
            n_samples=n_samples,
            method=method,
            verbose=verbose,
            method_kwargs=m_kwargs,
        )

    return rule, value


def signed_subgroup_discrepancy(
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_],
    verbose: int = 1,
) -> float:
    """
    Signed difference in subgroup representation between positive and negative outcomes.

    This metric returns:
        `delta = mean(subgroup | y = 1) - mean(subgroup | y = 0)`

    A positive delta means the subgroup is **over-represented** among positives;  
    a negative delta means it is **under-represented**.

    Args:
        subgroup (np.ndarray[bool]): Boolean mask of subgroup membership;
            shape must match `y`.
        y (np.ndarray[bool]): Boolean outcome labels
            (True = positive, False = negative).
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).
    
    Returns:
        float: Signed difference ``proportion_in_positives - proportion_in_negatives``.
    
    Raises:
        AssertionError: If `subgroup` and `y` have different shapes.
        ValueError: If `y` contains only positives or only negatives.
    
    Examples:
        1. Equal representation => Δ = 0

        >>> subgroup = np.array([True, False, True, False])
        >>> y = np.array([True, False, False, True])
        >>> signed_subgroup_discrepancy(subgroup, y)
        0.0

        2. Over-representation => positive Δ

        >>> subgroup = np.array([True, True, False, False, True])
        >>> y = np.array([True, True,  True,  False, False])
        >>> round(signed_subgroup_discrepancy(subgroup, y), 3)
        0.167  # subgroup is ~16.7 pp more common among positives

        3. Under-representation => negative Δ

        >>> subgroup = np.array([False, False, True, False])
        >>> y = np.array([True, True, False, False])
        >>> round(signed_subgroup_discrepancy(subgroup, y), 2)
        -0.50  # subgroup is 50 pp less common among positives
    """
    assert (
        subgroup.shape == y.shape
    ), f"Vector y and subgroup mapping have different shapes: {y.shape} and {subgroup.shape}, respectively."

    # Convert to boolean arrays if not already
    if subgroup.dtype != bool:
        if verbose >= 1: logger.warning(
            f"Subgroup mapping has dtype {subgroup.dtype} instead of bool. Assuming value for True is 1."
        )
        subgroup = subgroup == 1
    if y.dtype != bool:
        if verbose >= 1: logger.warning(
            f"Vector y has dtype {y.dtype} instead of bool. Assuming value for True is 1."
        )
        y = y == 1

    # Raise ValueError if all outcomes are the same, as proportions cannot be compared
    if np.all(y):
        raise ValueError("All samples are positive. Cannot calculate metric.")
    if np.all(~y):
        raise ValueError("All samples are negative. Cannot calculate metric.")

    # Calculate the mean of `subgroup` values where `y` is True (positive outcomes)
    proportion_in_pos = np.mean(subgroup[y])
    # Calculate the mean of `subgroup` values where `y` is False (negative outcomes)
    proportion_in_neg = np.mean(subgroup[~y])

    return proportion_in_pos - proportion_in_neg


def evaluate_subgroup_discrepancy(
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_],
    verbose: int = 1,
) -> float:
    """
    Absolute subgroup discrepancy abs(delta) between positive and negative outcomes.

    Simply returns the magnitude of `signed_subgroup_discrepancy(subgroup, y)`.
    
    Args:
        subgroup (np.ndarray[bool]): Boolean mask indicating subgroup membership;
            shape must equal that of `y`.
        y (np.ndarray[bool]): Boolean outcome labels (True = positive).
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).

    Returns:
        float: abs(delta) - the absolute difference in subgroup prevalence between
            positives and negatives (fractional units).

    Raises:
        AssertionError: If `subgroup` and `y` have different shapes.
        ValueError: If `y` contains only positives or only negatives.
    """
    return abs(signed_subgroup_discrepancy(subgroup, y, verbose=verbose))


def signed_subgroup_prevalence_diff(
    subgroup_a: np.ndarray[np.bool_],
    subgroup_b: np.ndarray[np.bool_],
) -> float:
    """
    Signed difference in subgroup prevalence between two datasets.

    Computes:
        delta = mean(subgroup_b) - mean(subgroup_a)

    A positive delta means the subgroup is more common in *dataset B* than in
    *dataset A*; a negative delta means the opposite.

    Args:
        subgroup_a (np.ndarray[bool]): Boolean mask for dataset A.
        subgroup_b (np.ndarray[bool]): Boolean mask for dataset B.
            The two arrays not necessarily need to be the same length, 
            but each must be one-dimensional and boolean.

    Returns:
        float: Signed prevalence difference delta.
    """
    return np.mean(subgroup_b) - np.mean(subgroup_a)


def report_subgroup_bias(
    label: str,
    msd: float,
    rule: list[tuple[int, Any]],
    feature_names: dict[str, str],
    value_map: dict[str, dict[Any, str]],
) -> None:
    """
    Print a little report of MSD and its human-readable rule.

    Args:
        label: a name for this sample (e.g. "State FL" or "FL vs NH").
        msd: the numeric MSD value.
        rule: the list of (col_idx, binop) pairs that define the subgroup.
        feature_names: mapping from column-code -> human feature name 
            (eg. from feature_folktables()).
        value_map: mapping from column-code -> {value_code -> human label} 
            (eg. from feature_folktables()).
    """
    print(f"{label}")
    print(f"MSD = {msd:.3f}")
    # raw rule
    raw = " AND ".join(str(r) for _, r in rule)
    print(f"Rule: {raw}")
    # pretty rule
    pretty = []
    for _, binop in rule:
        col = binop.feature.name
        human_feat = feature_names.get(col, col)
        val = binop.value
        human_val = value_map.get(col, {}).get(val, val)
        # TODO this "=" is not robust to other bins - e.g. continuous ones.
        pretty.append(f"{human_feat} = {human_val}")
    print("Explained rule: " + " AND ".join(pretty))
