import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_subgroup_discrepancy(
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_]
) -> float:
    """
    Calculates fairness metric, based on the difference in positive outcomes
    between a specified subgroup and its complement.

    The subgroup discrepancy quantifies the absolute difference in the proportion of positive
    outcomes (where `y` is True) for individuals within the `subgroup` versus
    those outside the `subgroup` (the complement).

    Args:
        subgroup (np.ndarray[bool] or np.ndarray[int]): A boolean (or integer 0/1)
            NumPy array indicating membership in the subgroup. `True` or `1` for
            members of the subgroup, `False` or `0` otherwise. Must have the same
            shape as `y`.
        y (np.ndarray[bool] or np.ndarray[int]): A boolean (or integer 0/1) NumPy
            array representing the true outcomes. `True` or `1` for positive
            outcomes, `False` or `0` for negative outcomes. Must have the same
            shape as `subgroup`.

    Returns:
        float: The absolute difference between the mean of `y` within the `subgroup`
               and the mean of `y` outside the `subgroup`.

    Raises:
        AssertionError: If `subgroup` and `y` have different shapes. (Note: This
                        uses a direct `assert`, which might be disabled in optimized
                        builds. For production, consider `if not ... raise ValueError`.)
        ValueError: If the `subgroup` is empty (contains no `True` values) or if
                    its complement is empty (contains all `True` values).

    Examples:
        >>> import numpy as np
        >>> # Scenario 1: No difference
        >>> subgroup_1 = np.array([True, True, False, False])
        >>> y_1 = np.array([True, False, True, False])
        >>> our_metric(subgroup_1, y_1)
        0.0

        >>> # Scenario 2: Subgroup has higher positive outcome rate
        >>> subgroup_2 = np.array([True, True, True, False, False])
        >>> y_2 = np.array([True, True, False, False, False])
        >>> our_metric(subgroup_2, y_2)
        0.6666666666666666

        >>> # Scenario 3: Using integer arrays (will be converted to bool)
        >>> subgroup_3 = np.array([1, 1, 0, 0])
        >>> y_3 = np.array([1, 0, 1, 0])
        >>> our_metric(subgroup_3, y_3)
        0.0

        >>> # Scenario 4: Subgroup is empty (will raise ValueError)
        >>> subgroup_empty = np.array([False, False, False])
        >>> y_empty = np.array([True, False, True])
        >>> try:
        ...     our_metric(subgroup_empty, y_empty)
        ... except ValueError as e:
        ...     print(e)
        Subgroup is empty. Cannot calculate metric.

        >>> # Scenario 5: Subgroup contains all samples (will raise ValueError)
        >>> subgroup_full = np.array([True, True, True])
        >>> y_full = np.array([True, False, True])
        >>> try:
        ...     our_metric(subgroup_full, y_full)
        ... except ValueError as e:
        ...     print(e)
        Subgroup contains all samples. Cannot calculate metric.
    """
    # trunk-ignore(bandit/B101)
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

    # Check if subgroup or its complement is empty
    if not np.any(subgroup):
        raise ValueError("Subgroup is empty. Cannot calculate metric.")
    if not np.any(~subgroup):
        raise ValueError("Subgroup contains all samples. Cannot calculate metric.")

    mean_subgroup = np.mean(y[subgroup])
    mean_notsubgroup = np.mean(y[~subgroup])

    return np.abs(mean_subgroup - mean_notsubgroup)


def subgroup_map_from_conjuncts(
    conjuncts: List[int], X: np.ndarray[np.bool_]
) -> np.ndarray[np.bool_]:
    """
    Generates a boolean subgroup mapping based on the conjunction (AND) of specified features.

    This function creates a boolean array where each element is `True` only if the
    corresponding row in `X` has `True` values across all columns specified in `conjuncts`.
    Essentially, it identifies individuals who meet all criteria defined by the conjuncts.

    Args:
        conjuncts (List[int]): A list of integer indices (column indices) from the
                               input array `X`. Each index represents a feature
                               that must be `True` for an individual to be included
                               in the subgroup.
        X (np.ndarray[np.bool_]): A 2D NumPy array of boolean values, where rows
                                  represent individuals and columns represent features.

    Returns:
        np.ndarray[np.bool_]: A 1D boolean NumPy array (`mapping`) of the same
                              length as the number of rows in `X`. An element
                              `mapping[i]` is `True` if `X[i, conj]` is `True` for
                              all `conj` in `conjuncts`, and `False` otherwise.

    Raises:
        IndexError: If any index in `conjuncts` is out of bounds for the columns of `X`.

    Examples:
        >>> import numpy as np
        >>> X_data = np.array([
        ...     [True,  True,  False, True],   # Row 0
        ...     [True,  False, True,  True],   # Row 1
        ...     [False, True,  True,  False],  # Row 2
        ...     [True,  True,  True,  True]    # Row 3
        ... ])
        >>>
        >>> # Subgroup where feature at index 0 AND feature at index 1 are True
        >>> conjuncts_1 = [0, 1]
        >>> subgroup_map_from_conjuncts(conjuncts_1, X_data)
        array([ True, False, False,  True])
        >>> # Explanation: Only Row 0 and Row 3 have both X[:,0] and X[:,1] as True.

        >>> # Subgroup where feature at index 2 is True
        >>> conjuncts_2 = [2]
        >>> subgroup_map_from_conjuncts(conjuncts_2, X_data)
        array([False,  True,  True,  True])

        >>> # Subgroup where feature at index 0 AND feature at index 2 are True
        >>> conjuncts_3 = [0, 2]
        >>> subgroup_map_from_conjuncts(conjuncts_3, X_data)
        array([False,  True, False,  True])

        >>> # Test with an empty list of conjuncts (should return all True)
        >>> subgroup_map_from_conjuncts([], X_data)
        array([ True,  True,  True,  True])

        >>> # Test with an invalid conjunct index (will raise IndexError)
        >>> try:
        ...     subgroup_map_from_conjuncts([0, 99], X_data)
        ... except IndexError as e:
        ...     print(e)
        index 99 is out of bounds for axis 1 with size 4
    """
    # Initialize the mapping with all True values. This ensures that if conjuncts
    # is empty, all individuals are included (logical AND of no conditions is True).
    mapping = np.ones((X.shape[0],), dtype=bool)

    # Iterate through each specified conjunct (feature index)
    for conj in conjuncts:
        # Perform a logical AND operation between the current mapping and the
        # specified feature column. This filters down the subgroup.
        mapping &= X[:, conj]  # This will raise IndexError if `conj` is out of bounds
    return mapping
