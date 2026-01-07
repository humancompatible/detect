import numpy as np
import pandas as pd
from typing import List, Tuple, Any


def subgroup_map_from_conjuncts_binarized(
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
        X (np.ndarray[`np.bool_`]): A 2D NumPy array of boolean values, where rows
            represent individuals and columns represent features.

    Returns:
        np.ndarray[`np.bool_`]: A 1D boolean NumPy array (`mapping`) of the same
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
        >>> subgroup_map_from_conjuncts_binarized(conjuncts_1, X_data)
        array([ True, False, False,  True])
        >>> # Explanation: Only Row 0 and Row 3 have both X[:,0] and X[:,1] as True.

        >>> # Subgroup where feature at index 2 is True
        >>> conjuncts_2 = [2]
        >>> subgroup_map_from_conjuncts_binarized(conjuncts_2, X_data)
        array([False,  True,  True,  True])

        >>> # Subgroup where feature at index 0 AND feature at index 2 are True
        >>> conjuncts_3 = [0, 2]
        >>> subgroup_map_from_conjuncts_binarized(conjuncts_3, X_data)
        array([False,  True, False,  True])

        >>> # Test with an empty list of conjuncts (should return all True)
        >>> subgroup_map_from_conjuncts_binarized([], X_data)
        array([ True,  True,  True,  True])

        >>> # Test with an invalid conjunct index (will raise IndexError)
        >>> try:
        ...     subgroup_map_from_conjuncts_binarized([0, 99], X_data)
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


def subgroup_map_from_conjuncts_dataframe(
    rule: List[Tuple[int, Any]], X: pd.DataFrame
) -> np.ndarray[np.bool_]:
    """
    Build a boolean mask for an MSD rule over a pandas DataFrame.

    Each (index, Bin) in *rule* comes from `detect_bias` or
    `detect_bias_two_samples`.  We ignore the positional index and
    use the Bin's `.feature.name`, so this is robust to column re-ordering.

    Args:
        rule (List[Tuple[int, Any]]): The rule identifying the subgroup, 
            as returned by `detect_bias(...)`.
        X (pd.DataFrame): The original (protected-only) DataFrame passed 
            to `detect_bias`. Must contain all columns named 
            in the rule's Bins.

    Returns:
        np.ndarray[`np.bool_`]: A 1-D boolean array where True marks rows 
            belonging to the subgroup.

    Raises:
        KeyError: If `X` is missing a column required by the rule.
    """
    mask = np.ones(len(X), dtype=bool)
    for _idx, binop in rule:
        feat = binop.feature.name
        if feat not in X.columns:
            raise KeyError(f"Column '{feat}' required by rule is missing.")
        col_values = X[feat].to_numpy()
        mask &= binop.evaluate(col_values)
    return mask
