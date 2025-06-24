import logging
from typing import List, Tuple

import numpy as np

from .one_rule import OneRule
from .utils import evaluate_subgroup_discrepancy, subgroup_map_from_conjuncts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def compute_MSD(
    X_bin: np.ndarray[np.bool_],
    y_bin: np.ndarray[np.bool_],
    time_limit: int = 600,
    n_min: int = 0,
) -> Tuple[float, List[int]]:
    """Computes the Maximum Subgroup Discrepancy (MSD) to detect bias.

    This function utilizes the `OneRule` algorithm to find a rule (a conjunction
    of binary features) that identifies a subgroup with the biggest difference in
    proportions between two samples. The discrepancy is then evaluated.

    Args:
        X_bin (np.ndarray[bool]): A 2D NumPy array of boolean values representing
            the binarized features (input data). Each row is a sample, and each
            column is a binary feature.
        y_bin (np.ndarray[bool]): A 1D NumPy array of boolean values representing
            the binarized target variable, which in the context of two samples,
            is assumed to distinguish between the two samples.
        time_limit (int, optional): The maximum time in seconds allowed for the
            `OneRule` algorithm to find a rule. Defaults to 600.
        n_min (int, optional): The minimum number of samples a subgroup must have
            to be considered. Defaults to 0.

    Returns:
        Tuple[float, List[int]]: A tuple containing:
            - MSD_val (float): The calculated Maximum Subgroup Discrepancy value for the
              identified subgroup, representing the biggest difference in proportions.
            - rule (List[int]): A list of binary indicies that represent the rule
              (conjunction of binary values) that defines the subgroup with the highest MSD.
    """

    mio = OneRule()
    rule = mio.find_rule(X_bin, y_bin, n_min=n_min, time_limit=time_limit)
    subgroup_map = subgroup_map_from_conjuncts(rule, X_bin)
    MSD_val = evaluate_subgroup_discrepancy(subgroup_map, y_bin)

    return MSD_val, rule
