import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from humancompatible.detect.binarizer.Binarizer import Bin
from humancompatible.detect.helpers.utils import evaluate_subgroup_discrepancy, signed_subgroup_discrepancy
from .mapping_msd import subgroup_map_from_conjuncts_dataframe

from .one_rule import OneRule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_conjuncts_MSD(
    X_bin: np.ndarray[np.bool_],
    y_bin: np.ndarray[np.bool_],
    time_limit: int = 600,
    n_min: int = 0,
    solver: str = "appsi_highs",
) -> Tuple[float, List[int]]:
    """
    Run the One-Rule MILP and return the indices of literals that form the
    Maximum-Subgroup-Discrepancy (MSD) rule.

    Args:
        X_bin (np.ndarray[bool]): Binary feature matrix (shape n_samples * n_features).
        y_bin (np.ndarray[bool]): Binary target vector (length n_samples).
        time_limit (int, default 600): Wall-clock limit for the solver, in seconds.
        n_min (int, default 0): Minimum support the subgroup must have.
        solver (str, default "appsi_highs"): Name of the MIP solver recognised by
            Pyomo (e.g. "gurobi", "cplex", "glpk", "xpress", "appsi_highs").

    Returns:
        list[int]: A list of feature-column indices whose conjunction
            defines the subgroup with maximal discrepancy.

    Raises:
        ValueError: Propagated from ``OneRule.find_rule`` when the solver stops
            with an unexpected termination condition.
    """

    mio = OneRule()
    indices, _ = mio.find_rule(
        X_bin, y_bin, n_min=n_min, time_limit=time_limit, solver_name=solver
    )

    return indices


def evaluate_MSD(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    rule: List[Tuple[int, Bin]],
    signed: bool = False,
) -> float:
    """
    Compute the MSD value (delta  or |delta|) for already calculated rules.
    
    Args:
        X (pd.DataFrame): DataFrame with the protected columns referenced by
            `rule`.
        y (pd.Series | np.ndarray): Binary target vector aligned with `X`.
        rule (list[tuple[int, Bin]]): Conjunctive rule describing the subgroup.
            Each element is a pair `(feature_index, Bin)`. Produced by
            `most_biased_subgroup`.
        signed (bool, default False): If True, return the signed subgroup discrepancy;
            otherwise, return the absolute value.
    
    Returns:
        float: The subgroup discrepancy value.
    """
    mask = subgroup_map_from_conjuncts_dataframe(rule, X)
    fn = signed_subgroup_discrepancy if signed else evaluate_subgroup_discrepancy
    return float(fn(mask, np.asarray(y).ravel()))
