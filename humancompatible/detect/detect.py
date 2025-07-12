import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .binarizer import Bin, Binarizer
from .data_handler import DataHandler
from .MSD import compute_MSD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def prepare_dataset(
    input_data: pd.DataFrame,
    target_data: pd.DataFrame,
    n_max: int,
    protected_attrs: List[str],
    continuous_feats: List[str],
    feature_processing: Dict[str, int],
):
    r"""
    Prepares a dataset by cleaning, preprocessing, sampling, and structuring it for fairness analysis.

    This function performs several steps to get the data ready for further processing,
    especially focusing on handling missing values, applying feature transformations,
    managing feature types (continuous vs. categorical), sampling, and identifying
    protected attributes.

    Args:
        input_data (pd.DataFrame): The input features DataFrame.
        target_data (pd.DataFrame): The target variable DataFrame. It's expected
                                    to have a single column.
        n_max (int): The maximum number of samples to retain. If the dataset size
                     exceeds this, it will be randomly downsampled.
        protected_attrs (List[str]): A list of column names that are considered
                                     protected attributes for fairness analysis.
        continuous_feats (List[str]): A list of column names identified as continuous features.
        feature_processing (Dict[str, int]): A dictionary where keys are column names
                                             and values are mappings (e.g., dictionaries
                                             or functions) to apply for preprocessing
                                             specific features.

    Returns:
        Tuple[Binarizer, pd.DataFrame, pd.Series]: A tuple containing:
            - binarizer_protected (Binarizer): The protected-attributes binarizer.
            - input_data[protected_cols] (pd.DataFrame): The part of the data with protected attributes.
            - target_data (pd.Series): The corresponding target features.

    Raises:
        (No explicit raises from within this function beyond potential pandas/numpy errors
         if data types or operations are mismatched before calling this function.)

    Notes:
        - Rows with any NaN values in `input_data` will be removed.
        - Features with only one unique value after NaN removal will be dropped.
        - The `target_data` is assumed to contain only one column and will be
          converted to a pandas Series for the output.
        - Requires `DataHandler` and `Binarizer` classes to be defined elsewhere
          for `dhandler_protected` and `binarizer_protected` to work correctly.

    Examples:
        >>> # Assuming 'logger', 'DataHandler', and 'Binarizer' are imported/defined
        >>> # For demonstration, let's mock them
        >>> class MockDataHandler:
        ...     def __init__(self, *args, **kwargs): pass
        >>> class MockBinarizer:
        ...     def __init__(self, *args, **kwargs): self.target_positive_vals = kwargs.get('target_positive_vals')
        >>> global DataHandler, Binarizer, logger
        >>> DataHandler = MockDataHandler
        >>> Binarizer = MockBinarizer
        >>> logger = logging.getLogger('test_logger')
        >>> logger.setLevel(logging.DEBUG)
        >>>
        >>> input_df = pd.DataFrame({
        ...     'feat_cat': ['A', 'B', 'A', 'C', 'A', None],
        ...     'feat_cont': [1.0, 2.5, 3.0, 4.5, 5.0, 6.0],
        ...     'protected_sex': ['M', 'F', 'M', 'F', 'M', 'F'],
        ...     'protected_race': ['W', 'B', 'W', 'A', 'B', 'W'],
        ...     'single_val_col': [10, 10, 10, 10, 10, 10]
        ... })
        >>> target_df = pd.DataFrame({'outcome': [0, 1, 0, 1, 0, 1]})
        >>>
        >>> feature_proc = {'feat_cat': {'A': 0, 'B': 1, 'C': 2}}
        >>> protected_attrs_list = ['protected_sex', 'protected_race']
        >>> continuous_features_list = ['feat_cont']
        >>> max_samples = 4
        >>>
        >>> binarizer, protected_df, target_series = prepare_dataset(
        ...     input_df, target_df, max_samples, protected_attrs_list,
        ...     continuous_features_list, feature_proc
        ... )
        >>>
        >>> print("Original input_data shape:", input_df.shape)
        >>> print("Protected features after processing:\\n", protected_df)
        >>> print("\\nTarget series after processing:\\n", target_series)
        >>> print("\\nBinarizer target positive values:", binarizer.target_positive_vals)
    """
    mask = ~input_data.isnull().any(axis=1)
    logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data[mask.values]
    target_data = target_data[mask.values]

    # Preprocess the data
    for col, map_f in feature_processing.items():
        if col in input_data.columns:
            input_data.loc[:, col] = input_data[col].map(map_f)

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        logger.debug(f"Feature {col} has {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            logger.info(
                f"Feature {col} was removed due to having a single unique value"
            )
            continue
        if col not in continuous_feats:
            values[col] = vals
        else:
            bounds[col] = (min(vals), max(vals))

    n = input_data.shape[0]
    if n_max < n:
        samples = np.random.choice(n, size=n_max, replace=False)
    else:
        samples = np.random.permutation(n)

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]

    protected_cols = [col for col in input_data.columns if col in protected_attrs]
    dhandler_protected = DataHandler.from_data(
        input_data[protected_cols],
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )
    binarizer_protected = Binarizer(dhandler_protected, target_positive_vals=[True])

    return binarizer_protected, input_data[protected_cols], target_data


def detect_bias(
    X: pd.DataFrame,
    y: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[float, List[Tuple[int, Bin]]]:
    """Detects bias in a given dataset using specified methods.

    This function prepares the data and then applies a bias detection method,
    such as Maximum Subgroup Discrepancy (MSD), to identify potential biases
    related to protected attributes.

    Args:
        X (pd.DataFrame): The input features DataFrame.
        y (pd.DataFrame): The target variable DataFrame.
        protected_list (List[str] | None, optional): A list of column names that
            are considered protected attributes. If None, all attributes in X
            are assumed to be protected (handled by `detect_bias_csv` or `detect_bias_two_samples`
            if called through them, or should be handled before calling this function directly).
            Defaults to None.
        continuous_list (List[str], optional): A list of column names identified
            as continuous features. Defaults to [].
        fp_map (Dict[str, Callable[[Any], int]], optional): A dictionary for feature
            processing, where keys are column names and values are mapping
            functions to apply for preprocessing specific features. Defaults to {}.
        seed (int | None, optional): A seed for random number generation to ensure
            reproducibility. If None, no specific seed is set. Defaults to None.
        n_samples (int, optional): The maximum number of samples to use from the
            dataset. If the dataset size exceeds this, it will be randomly downsampled.
            Defaults to 1_000_000.
        method (str, optional): The bias detection method to use. Currently, only
            "MSD" (Maximum Subgroup Discrepancy) is implemented. Defaults to "MSD".
        method_kwargs (Dict[str, Any], optional): Additional keyword arguments
            to pass to the chosen bias detection method. Defaults to {}.

    Returns:
        Tuple[float, List[Tuple[int, Bin]]]: A tuple containing:
            - val (float): The calculated bias value. The interpretation depends
              on the `method` used (e.g., MSD value).
            - rule (List[Tuple[int, Bin]]): A list of tuples representing the rule
              or set of conditions that identify the biased subgroup. The exact
              structure depends on the `method`'s output.

    Raises:
        ValueError: If an unsupported `method` is specified.

    Notes:
        - This function internally calls `prepare_dataset` for data preprocessing.
    """

    if seed is not None:
        logger.info(f"Seeding the run with seed={seed}")
        np.random.seed(seed)
    
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    binarizer, X_prot, y = prepare_dataset(
        X,
        y,
        n_samples,
        protected_attrs=protected_list,
        continuous_feats=continuous_list,
        feature_processing=fp_map,
    )

    X_bin = binarizer.encode(X_prot, include_binary_negations=True)
    y_bin = binarizer.encode_y(y)

    if method == "MSD":
        val, indices = compute_MSD(X_bin, y_bin, **method_kwargs)
    else:
        raise ValueError(
            f'Method named "{method}" is not implemented. Try one of [MSD].'
        )

    encodings = binarizer.get_bin_encodings(include_binary_negations=True)
    feats = binarizer.data_handler.features
    rule = [(feats.index(encodings[i].feature), encodings[i]) for i in indices]

    return val, rule


def detect_bias_csv(
    csv_path: Path | str,
    target_col: str,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[float, List[Tuple[int, Bin]]]:
    """Detects bias in a dataset loaded from a CSV file.

    This function reads a dataset from a specified CSV path, separates features
    and target, and then calls the `detect_bias` function to perform bias detection.

    Args:
        csv_path (Path | str): The path to the CSV file containing the dataset.
        target_col (str): The name of the column in the CSV file that represents
            the target variable.
        protected_list (List[str] | None, optional): A list of column names that
            are considered protected attributes. If None, all columns in `X_df`
            (features excluding the target column) will be treated as protected.
            Defaults to None.
        continuous_list (List[str], optional): A list of column names identified
            as continuous features. Defaults to [].
        fp_map (Dict[str, Callable[[Any], int]], optional): A dictionary for feature
            processing, where keys are column names and values are mapping
            functions to apply for preprocessing specific features. Defaults to {}.
        seed (int | None, optional): A seed for random number generation to ensure
            reproducibility. If None, no specific seed is set. Defaults to None.
        n_samples (int, optional): The maximum number of samples to use from the
            dataset. If the dataset size exceeds this, it will be randomly downsampled.
            Defaults to 1_000_000.
        method (str, optional): The bias detection method to use. Currently, only
            "MSD" (Maximum Subgroup Discrepancy) is implemented. Defaults to "MSD".
        method_kwargs (Dict[str, Any], optional): Additional keyword arguments
            to pass to the chosen bias detection method. Defaults to {}.

    Returns:
        Tuple[float, List[Tuple[int, Bin]]]: A tuple containing:
            - val (float): The calculated bias value.
            - rule (List[Tuple[int, Bin]]): The rule or set of conditions
              identifying the biased subgroup.

    Raises:
        ValueError: If the `target_col` is not found in the CSV file.
        ValueError: Propagates `ValueError` from `detect_bias` if an unsupported
                    method is specified.
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the CSV file.")
    X_df = df.drop(columns=[target_col])
    y_df = pd.DataFrame(df[target_col])

    if protected_list is None:
        logger.info("Assuming all attributes are protected")
        protected_list = list(X_df.columns)
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    return detect_bias(
        X_df,
        y_df,
        protected_list,
        continuous_list,
        fp_map,
        seed,
        n_samples,
        method,
        method_kwargs,
    )


def detect_bias_two_samples(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    protected_list: List[str] | None = None,
    continuous_list: List[str] | None = None,
    fp_map: Dict[str, Callable[Any, int]] | None = None,
    seed: int | None = None,
    n_samples: int = 1_000_000,
    method: str = "MSD",
    method_kwargs: Dict[str, Any] | None = None,
) -> Tuple[float, List[Tuple[int, Bin]]]:
    """Detects bias between two distinct samples (datasets).

    This function concatenates two input DataFrames (`X1` and `X2`) and creates
    a synthetic target variable to differentiate between the two samples. It then
    calls `detect_bias` to find any bias based on the combined dataset.

    Args:
        X1 (pd.DataFrame): The first input features DataFrame.
        X2 (pd.DataFrame): The second input features DataFrame.
        protected_list (List[str] | None, optional): A list of column names that
            are considered protected attributes. If None, all columns in the
            concatenated DataFrame will be treated as protected. Defaults to None.
        continuous_list (List[str], optional): A list of column names identified
            as continuous features. Defaults to [].
        fp_map (Dict[str, Callable[[Any], int]], optional): A dictionary for feature
            processing, where keys are column names and values are mapping
            functions to apply for preprocessing specific features. Defaults to {}.
        seed (int | None, optional): A seed for random number generation to ensure
            reproducibility. If None, no specific seed is set. Defaults to None.
        n_samples (int, optional): The maximum number of samples to use from the
            combined dataset. If the dataset size exceeds this, it will be randomly
            downsampled. Defaults to 1_000_000.
        method (str, optional): The bias detection method to use. Currently, only
            "MSD" (Maximum Subgroup Discrepancy) is implemented. Defaults to "MSD".
        method_kwargs (Dict[str, Any], optional): Additional keyword arguments
            to pass to the chosen bias detection method. Defaults to {}.

    Returns:
        Tuple[float, List[Tuple[int, Bin]]]: A tuple containing:
            - val (float): The calculated bias value.
            - rule (List[Tuple[int, Bin]]): The rule or set of conditions
              identifying the biased subgroup based on the two samples.

    Raises:
        ValueError: If `X1` and `X2` do not have the same columns.
        ValueError: Propagates `ValueError` from `detect_bias` if an unsupported
                    method is specified.

    Notes:
        - A synthetic target `y_df` is created where rows from `X1` are labeled
          0 and rows from `X2` are labeled 1. This allows `detect_bias` to
          identify differences between the two samples as "bias".
        - This function relies on `detect_bias` for the core bias detection logic.
    """

    if X1.columns.tolist() != X2.columns.tolist():
        raise ValueError("The samples must have the same features")

    X_df = pd.concat([X1, X2])
    y = np.concatenate([
        np.zeros(X1.shape[0], dtype=int),
        np.ones (X2.shape[0], dtype=int),
    ])
    y_df = pd.DataFrame(y, columns=["target"])

    if protected_list is None:
        logger.info("Assuming all attributes are protected")
        protected_list = X_df.columns.tolist()
    if continuous_list is None:
        continuous_list = []
    if fp_map is None:
        fp_map = {}
    if method_kwargs is None:
        method_kwargs = {}

    return detect_bias(
        X_df,
        y_df,
        protected_list,
        continuous_list,
        fp_map,
        seed,
        n_samples,
        method,
        method_kwargs,
    )
