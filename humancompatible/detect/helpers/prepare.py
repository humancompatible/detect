import logging
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from humancompatible.detect.binarizer.Binarizer import Binarizer
from humancompatible.detect.data_handler import DataHandler

logger = logging.getLogger(__name__)


def prepare_dataset(
    input_data: pd.DataFrame,
    target_data: pd.DataFrame,
    n_max: int,
    protected_attrs: List[str],
    continuous_feats: List[str],
    feature_processing: Dict[str, Callable[[Any], int]],
    verbose: int = 1,
) -> Tuple[Binarizer, pd.DataFrame, pd.Series]:
    """
    Prepares a dataset by cleaning, preprocessing, sampling, and structuring it for fairness analysis.

    This function performs several steps to get the data ready for further processing,
    especially focusing on handling missing values, applying feature transformations,
    managing feature types (continuous vs. categorical), sampling, and identifying
    protected attributes.

    Args:
        input_data (pd.DataFrame): The input features DataFrame.
        target_data (pd.DataFrame): Single-column target vector; 
            same row count as `input_data`.
        n_max (int): The maximum number of samples to retain. If the dataset size
            exceeds this, it will be randomly downsampled.
        protected_attrs (List[str]): A list of column names that are considered
            protected attributes for fairness analysis.
        continuous_feats (List[str]): A list of column names identified as continuous features.
        feature_processing (Dict[str, Callable[[Any], int]]): Mapping from column
            name to a *callable* that converts each raw value to an integer.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).

    Returns:
        Tuple[Binarizer, pd.DataFrame, pd.Series]: A tuple containing:
            - binarizer_protected (Binarizer): The protected-attributes binarizer.
            - input_data[protected_cols] (pd.DataFrame): The part of the data with protected attributes.
            - target_data (pd.Series): The corresponding target features.

    Notes:
        - Rows with any NaN values in `input_data` will be removed.
        - Features with only one unique value after NaN removal will be dropped.
        - The `target_data` is assumed to contain only one column and will be
          converted to a pandas Series for the output.
        - Requires `DataHandler` and `Binarizer` classes to be defined elsewhere
          for `dhandler_protected` and `binarizer_protected` to work correctly.
    """
    mask_x = (~input_data.isnull().any(axis=1)).to_numpy()
    mask_y = (~target_data.isnull().any(axis=1)).to_numpy()
    mask = mask_x & mask_y
    
    if verbose >= 1: 
        logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data.loc[mask].copy()
    target_data = target_data.loc[mask].copy()

    # Preprocess the data
    for col, map_f in feature_processing.items():
        if col in input_data.columns:
            input_data.loc[:, col] = input_data[col].map(map_f)

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        if verbose >= 1: logger.debug(f"Feature {col} has {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            if verbose >= 1: logger.info(
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
