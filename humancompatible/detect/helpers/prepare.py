import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from humancompatible.detect.binarizer.Binarizer import Binarizer
from humancompatible.detect.data_handler import DataHandler

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
