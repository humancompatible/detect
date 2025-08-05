import logging
from typing import Any, List, Sequence, Tuple
from random import randrange

import numpy as np
import pandas as pd
import scipy.optimize as optimize

logger = logging.getLogger(__name__)


def signed_subgroup_discrepancy(
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_]
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
        logger.warning(
            f"Subgroup mapping has dtype {subgroup.dtype} instead of bool. Assuming value for True is 1."
        )
        subgroup = subgroup == 1
    if y.dtype != bool:
        logger.warning(
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
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_]
) -> float:
    """
    Absolute subgroup discrepancy |delta| between positive and negative outcomes.

    Simply returns the magnitude of `signed_subgroup_discrepancy(subgroup, y)`.
    
    Args:
        subgroup (np.ndarray[bool]): Boolean mask indicating subgroup membership;
            shape must equal that of `y`.
        y (np.ndarray[bool]): Boolean outcome labels (True = positive).

    Returns:
        float: |delta| - the absolute difference in subgroup prevalence between
            positives and negatives (fractional units).

    Raises:
        AssertionError: If `subgroup` and `y` have different shapes.
        ValueError: If `y` contains only positives or only negatives.
    """
    return abs(signed_subgroup_discrepancy(subgroup, y))


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
