import logging
from typing import List, Tuple

import numpy as np

from .one_rule import OneRule
from .utils import evaluate_subgroup_discrepancy, subgroup_map_from_conjuncts_binarized

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_conjuncts_MSD(
    X_bin: np.ndarray[np.bool_],
    y_bin: np.ndarray[np.bool_],
    rule
    
) -> Tuple[float, List[int]]:
    """Computes the Maximum Subgroup Discrepancy (MSD) to detect bias.

    This function utilizes the `OneRule` algorithm to find a rule (a conjunction
    of binary features) that identifies a subgroup with the biggest difference in
    proportions between two samples. The discrepancy is then evaluated.

    Args:
        X_bin (np.ndarray[bool]): A 2D NumPy array of boolean values representing
            the binarized features (input data). Each row is a sample, and each
            column is a binary feature.
        y_bin (np.ndarray[bool]): A 1D NumPy boolean array of the binary target labels
            used to distinguish the two data distributions (True=positive outcome, False=negative outcome).
        time_limit (int, optional): The maximum time in seconds allowed for the
            `OneRule` algorithm to find a rule. Defaults to 600.
        n_min (int, optional): The minimum number of samples a subgroup must have
            to be considered. Defaults to 0.
        solver (str, optional): Which MIP solver to use for the OneRule call. Must be one of
            "appsi_highs", "gurobi", "cplex", "glpk", "xpress", or other Pyomo-compatible solvers.
            Note that solvers other than the 5 mentioned earlier will not follow the `time_limit` parameter.
            Defaults to "appsi_highs".

    Returns:
        Tuple[float, List[int]]: A tuple containing:
            - MSD_val (float): The calculated Maximum Subgroup Discrepancy value for the
              identified subgroup, representing the biggest difference in proportions.
            - rule (List[int]): A list of feature-column indices whose conjunction
              defines the subgroup with maximal discrepancy.
    """

    
    subgroup_map = subgroup_map_from_conjuncts_binarized(rule, X_bin)
    MSD_val = evaluate_subgroup_discrepancy(subgroup_map, y_bin)

    return MSD_val, rule
