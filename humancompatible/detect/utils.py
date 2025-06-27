import logging
from typing import List, Any

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_subgroup_discrepancy(
    subgroup: np.ndarray[np.bool_], y: np.ndarray[np.bool_]
) -> float:
    """
    Calculates a fairness metric based on the difference in subgroup representation
    between positive and negative outcomes.

    This metric quantifies the absolute difference between:
    1. The proportion of individuals in the `subgroup` among all **positive outcomes**.
    2. The proportion of individuals in the `subgroup` among all **negative outcomes**.

    It effectively measures how differently the subgroup is represented across the
    two outcome classes (positive vs. negative `y`).

    Args:
        subgroup (np.ndarray[bool]): A boolean NumPy array indicating membership
            in the subgroup. `True` or `1` for members of the subgroup,
            `False` or `0` otherwise. Must have the same shape as `y`.
        y (np.ndarray[bool]): A boolean NumPy array representing the true outcomes.
            `True` or `1` for positive outcomes, `False` or `0` for negative outcomes.
            Must have the same shape as `subgroup`.

    Returns:
        float: The absolute difference between the proportion of `subgroup` members
               among positive outcomes and the proportion of `subgroup` members
               among negative outcomes.

    Raises:
        AssertionError: If `subgroup` and `y` have different shapes.
        ValueError: If all `y` values are `True` (all samples are positive) or
                    all `y` values are `False` (all samples are negative),
                    as the metric requires both positive and negative outcomes
                    to form proportions.

    Examples:
        >>> import numpy as np
        >>> # Scenario 1: Subgroup equally represented in positive and negative outcomes
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

        >>> # Scenario 2: Subgroup more prevalent in positive outcomes
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

        >>> # Scenario 3: Using integer arrays (will be converted to bool)
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

        >>> # Scenario 4: All samples are positive (will raise ValueError)
        >>> subgroup_all_y_pos = np.array([True, False, True])
        >>> y_all_pos = np.array([True, True, True])
        >>> try:
        ...     evaluate_subgroup_discrepancy(subgroup_all_y_pos, y_all_pos)
        ... except ValueError as e:
        ...     print(e)
        All samples are positive. Cannot calculate metric.

        >>> # Scenario 5: All samples are negative (will raise ValueError)
        >>> subgroup_all_y_neg = np.array([True, False, True])
        >>> y_all_neg = np.array([False, False, False])
        >>> try:
        ...     evaluate_subgroup_discrepancy(subgroup_all_y_neg, y_all_neg)
        ... except ValueError as e:
        ...     print(e)
        All samples are negative. Cannot calculate metric.
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

    # Raise ValueError if all outcomes are the same, as proportions cannot be compared
    if np.all(y):
        raise ValueError("All samples are positive. Cannot calculate metric.")
    if np.all(~y):
        raise ValueError("All samples are negative. Cannot calculate metric.")

    # Calculate the mean of `subgroup` values where `y` is True (positive outcomes)
    proportion_in_pos = np.mean(subgroup[y])
    # Calculate the mean of `subgroup` values where `y` is False (negative outcomes)
    proportion_in_neg = np.mean(subgroup[~y])

    return np.abs(proportion_in_pos - proportion_in_neg)


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


def _utils_folktables():
    """
    You can find all explanation of resources here:
    https://www.icpsr.umich.edu/web/DSDR/studies/25042/datasets/0002/variables/POBP?archive=dsdr

    In "Search Variables" filed, write the code of uknown abbreviation.
    Or you can use: 
    https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf
    """

    FEATURE_NAMES = {
        "SEX": "Sex",
        "RAC1P": "Race",
        "AGEP": "Age",
        "MAR": "Marital status",
        "POBP": "Place of birth",
        "DIS": "Disability",
        "CIT": "Citizenship",
        "MIL": "Military service",
        "ANC": "Ancestry",
        "NATIVITY": "Foreign or US native",
        "DEAR": "Difficulty hearing",
        "DEYE": "Difficulty seeing",
        "DREM": "Cognitive difficulty",
        "FER": "Gave birth last year",
        "POVPIP": "Income / Poverty threshold",

        "COW": "Class of worker",
        "SCHL": "Educational attainment",
        "OCCP": "Occupation recode",
        "WKHP": "Usual hours worked per week past 12 months",

    }

    PROTECTED_VALUES_MAP = {
        "SEX": {1: "Male", 2: "Female"},
        "RAC1P": {
            1: "White",
            2: "Black",
            3: "American Indian",
            4: "Alaska Native",
            5: "Native tribes specified",
            6: "Asian",
            7: "Pacific Islander",
            8: "Some Other Race",
            9: "Two or More Races",
        },
        # "AGEP": {}, Age is not categorical
        "MAR": {
            1: "Married",
            2: "Widowed",
            3: "Divorced",
            4: "Separated",
            5: "Never married",
        },
        # "POBP": {}, Too many values, one for each country
        "_POBP": {
            0: "US",
            1: "Europe",
            2: "Asia",
            3: "Non-US Americas",
            4: "Africa",
            5: "Oceania",
        },
        "DIS": {1: "With a disability", 2: "Without a disability"},
        "CIT": {
            1: "Born in the US",
            2: "Born in external US teritories",
            3: "Born abroad of US parent(s)",
            4: "Naturalized US citizen",
            5: "Not a US citizen",
        },
        "MIL": {
            None: "N/A (<17 years)",
            1: "On active duty",
            2: "No more active duty",
            3: "Active duty for training",
            4: "Never served",
        },
        "ANC": {
            1: "Single",
            2: "Multiple",
            3: "Unclassified",
            4: "Not reported",
            8: "Hidden",
        },
        "NATIVITY": {1: "Native", 2: "Foreign born"},
        "DEAR": {1: "Yes", 2: "No"},
        "DEYE": {1: "Yes", 2: "No"},
        "DREM": {1: "Yes", 2: "No", None: "N/A (<5 years)"},
        "FER": {1: "Yes", 2: "No", None: "N/A"},

        "SCHL": {
            1:	"No school completed",
            2:	"Nursery school to grade 4",
            3:	"Grade 5 or grade 6",
            4:	"Grade 7 or grade 8",
            5:	"Grade 9",
            6:	"Grade 10",
            7:	"Grade 11",
            8:	"Grade 12 no diploma",
            9:	"High school graduate",
            10:	"Some college, but less than 1 year",
            11:	"One or more years of college, no degree",
            12:	"Associate's degree",
            13:	"Bachelor's degree",
            14:	"Master's degree",
            15:	"Professional school degree",
            16:	"Doctorate degree",
        },
        "COW": {
            # Not implemented yet. Look at docstrings of this function
        },
        "OCCP": {
            # Not implemented yet. Look at docstrings of this function
        },
        "WKHP": {
            # Not implemented yet. Look at docstrings of this function
        },


        # "POVPIP": {}, Poverty ratio has numeric values
    }

    return FEATURE_NAMES, PROTECTED_VALUES_MAP


def report_subgroup_bias(
    label: str,
    msd: float,
    rule: list[tuple[int, Any]],
    selected_columns: list[str],
    feature_names: dict[str, str],
    value_map: dict[str, dict[Any, str]],
) -> None:
    """
    Print a little report of MSD and its human-readable rule.

    Args:
        label: a name for this sample (e.g. "State FL" or "FL vs NH").
        msd: the numeric MSD value.
        rule: the list of (col_idx, binop) pairs that define the subgroup.
        selected_columns: the list of column-codes in the same order you passed to detect_bias.
        feature_names: mapping from column-code -> human feature name (from _utils_folktables()).
        value_map: mapping from column-code -> {value_code -> human label} (from _utils_folktables()).
    """
    print(f"{label}")
    print(f"MSD = {msd:.3f}")
    # raw rule
    raw = " AND ".join(str(r) for _, r in rule)
    print(f"Rule: {raw}")
    # pretty rule
    pretty = []
    for col_idx, binop in rule:
        col = selected_columns[col_idx]
        human_feat = feature_names.get(col, col)
        val = binop.value
        human_val = value_map.get(col, {}).get(val, val)
        pretty.append(f"{human_feat} = {human_val}")
    print("Explained rule: " + " AND ".join(pretty))
