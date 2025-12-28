import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from typing import Any, Sequence, Tuple

from .mapping_msd import subgroup_map_from_conjuncts_dataframe
from humancompatible.detect.helpers.utils import signed_subgroup_discrepancy, evaluate_subgroup_discrepancy


def subgroup_gap(
    rule: Sequence[Tuple[int, Any]],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    signed: bool = True,
) -> float:
    """Compute the subgroup discrepancy `delta` or `abs(delta)` for a given rule.

    Args:
        rule: Rule returned by `detect_bias` - list of (col_idx, Bin).
        X: DataFrame containing the protected columns referenced in `rule`.
        y: Binary outcome vector aligned with `X` (1 = positive outcome).
        signed: If True returns signed `delta`, else returns `abs(delta)`.

    Returns:
        float: Subgroup discrepancy (signed or absolute).

    Raises:
        KeyError: If `X` is missing a column required by the rule.
        ValueError: If `y` contains only positives or only negatives.
    """
    mask = subgroup_map_from_conjuncts_dataframe(rule, X)
    fn = signed_subgroup_discrepancy if signed else evaluate_subgroup_discrepancy
    return float(fn(mask, np.asarray(y).ravel()))
