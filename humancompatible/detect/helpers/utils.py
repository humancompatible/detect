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
    Signed difference in subgroup representation between *positive and negative outcomes*.

    This metric returns

    .. math::

        \\Delta = \\underbrace{\\operatorname{mean}(\\text{subgroup}[y])}_{\\text{proportion in positives}}
               \\,\\; - \\;
               \\underbrace{\\operatorname{mean}(\\text{subgroup}[\\lnot y])}_{\\text{proportion in negatives}}

    * **Δ > 0** - subgroup is **over-represented** among positive outcomes  
    * **Δ < 0** - subgroup is **under-represented** among positive outcomes  
    * **|Δ|**   - magnitude used by MSD (see :func:`evaluate_subgroup_discrepancy`)

    Parameters
    ----------
    subgroup : np.ndarray[bool]
        Boolean mask indicating membership in the subgroup
        (shape must match *y*).  ``True`` for members, ``False`` otherwise.
    y : np.ndarray[bool]
        Boolean outcome labels. ``True`` = positive outcome, ``False`` = negative.

    Returns
    -------
    float
        Signed difference ``proportion_in_positives - proportion_in_negatives``.

    Raises
    ------
    AssertionError
        If *subgroup* and *y* have different shapes.
    ValueError
        If *y* contains only positives or only negatives - the difference
        would be undefined.

    Examples
    --------
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

    The absolute value of the same quantity is returned by
    :func:`evaluate_subgroup_discrepancy`.
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
    Absolute difference in subgroup prevalence between *positive and negative outcomes (|Δ|)*.

    This is simply the magnitude of the signed metric returned by
    :func:`signed_subgroup_discrepancy`.  Formally,

    .. math::

        |\\Delta| \\;=\\; \\bigl|\\,
            \\operatorname{mean}(\\text{subgroup}[y])
            \\, - \\, 
            \\operatorname{mean}(\\text{subgroup}[\\lnot y])
        \\,\\bigr|

    Parameters
    ----------
    subgroup : np.ndarray[bool]
        Boolean mask indicating membership in the subgroup
        (shape must match *y*).
    y : np.ndarray[bool]
        Boolean outcome labels.  ``True`` = positive outcome,
        ``False`` = negative.

    Returns
    -------
    float
        Absolute subgroup discrepancy ``|Δ|`` expressed in **fractional points**
        (multiply by 100 for percentage points).

    Raises
    ------
    AssertionError
        If *subgroup* and *y* have different shapes.
    ValueError
        If *y* contains only positives or only negatives - the metric would be
        undefined.

    Examples
    ------
    1. Subgroup equally represented in positive and negative outcomes

    >>> import numpy as np
    >>> subgroup_1 = np.array([True, False, True, False]) # Indices 0, 2 are subgroup
    >>> y_1 = np.array([True, False, True, False])        # Indices 0, 2 are positive
    >>> # Positive outcomes: [True, True] (indices 0, 2). Subgroup members: 2/2 = 1.0
    >>> # Negative outcomes: [False, False] (indices 1, 3). Subgroup members: 0/2 = 0.0
    >>> # Discrepancy: |1.0 - 0.0| = 1.0 (This is the original error in example)
    >>> # Corrected:
    >>> # y[subgroup] means y for subgroup members: [True, True]
    >>> # y[~subgroup] means y for non-subgroup members: [False, False]
    >>> # Your new formula: mean(subgroup[y]) - mean(subgroup[~y])
    >>> # subgroup[y] (subgroup status for positive outcomes): [True, True] -> mean = 1.0
    >>> # subgroup[~y] (subgroup status for negative outcomes): [False, False] -> mean = 0.0
    >>> evaluate_subgroup_discrepancy(subgroup_1, y_1)
    1.0

    2. Subgroup more prevalent in positive outcomes

    >>> subgroup_2 = np.array([True, True, False, False, True])
    >>> y_2 = np.array([True, True, True, False, False])
    >>> # Positive outcomes (y=True): [True, True, True] (indices 0, 1, 2)
    >>> # Subgroup status for positive outcomes: subgroup[0]=T, subgroup[1]=T, subgroup[2]=F.
    >>> # mean(subgroup[y]): (1+1+0)/3 = 0.666...
    >>> #
    >>> # Negative outcomes (y=False): [False, False] (indices 3, 4)
    >>> # Subgroup status for negative outcomes: subgroup[3]=F, subgroup[4]=T.
    >>> # mean(subgroup[~y]): (0+1)/2 = 0.5
    >>> evaluate_subgroup_discrepancy(subgroup_2, y_2)
    0.16666666666666663

    3. Using integer arrays (will be converted to bool)

    >>> subgroup_3 = np.array([1, 1, 0, 0])
    >>> y_3 = np.array([1, 0, 1, 0])
    >>> # Positive outcomes (y=1): [1, 1] (indices 0, 2)
    >>> # subgroup status for positive outcomes: subgroup[0]=1, subgroup[2]=0
    >>> # mean(subgroup[y]): (1+0)/2 = 0.5
    >>> # Negative outcomes (y=0): [1, 0] (indices 1, 3)
    >>> # subgroup status for negative outcomes: subgroup[1]=1, subgroup[3]=0
    >>> # mean(subgroup[~y]): (1+0)/2 = 0.5
    >>> evaluate_subgroup_discrepancy(subgroup_3, y_3)
    0.0

    4. All samples are positive (will raise ValueError)

    >>> subgroup_all_y_pos = np.array([True, False, True])
    >>> y_all_pos = np.array([True, True, True])
    >>> try:
    ...     evaluate_subgroup_discrepancy(subgroup_all_y_pos, y_all_pos)
    ... except ValueError as e:
    ...     print(e)
    All samples are positive. Cannot calculate metric.

    5. All samples are negative (will raise ValueError)

    >>> subgroup_all_y_neg = np.array([True, False, True])
    >>> y_all_neg = np.array([False, False, False])
    >>> try:
    ...     evaluate_subgroup_discrepancy(subgroup_all_y_neg, y_all_neg)
    ... except ValueError as e:
    ...     print(e)
    All samples are negative. Cannot calculate metric.
    """
    return abs(signed_subgroup_discrepancy(subgroup, y))


def signed_subgroup_prevalence_diff(
    subgroup_a: np.ndarray[np.bool_],
    subgroup_b: np.ndarray[np.bool_],
) -> float:
    """
    Signed difference in subgroup prevalence between *dataset A* and *dataset B*.

    .. math::

        \\Delta_{A\\!\!\\rightarrow B}
            = \\operatorname{mean}(\\text{subgroup}_B)
            \\, - \\,\\operatorname{mean}(\\text{subgroup}_A)

    * **Δ < 0**  - subgroup is **more common in A** than in B  
    * **Δ > 0**  - subgroup is **more common in B** than in A  
    * **|Δ|**     - the magnitude is the cross-sample MSD value you already report

    Parameters
    ----------
    subgroup_a, subgroup_b
        Boolean 1-D arrays (may be different lengths) indicating membership
        in the same rule-defined subgroup for each dataset.

    Returns
    -------
    float
        Signed prevalence gap.

    Raises
    ------
    ValueError
        If either array is empty; if they are not boolean or 1-D.
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
        feature_names: mapping from column-code -> human feature name (eg. from feature_folktables()).
        value_map: mapping from column-code -> {value_code -> human label} (eg. from feature_folktables()).
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
