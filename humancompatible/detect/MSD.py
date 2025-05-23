import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .binarizer import Binarizer
from .data_handler import DataHandler
from .one_rule import OneRule
from .utils import evaluate_subgroup_discrepancy, subgroup_map_from_conjuncts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_dataset(csv_path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Loads a dataset from a CSV file and splits it into features (X) and target (y) DataFrames.

    This function reads a CSV file into a pandas DataFrame. It then separates the specified
    target column from the rest of the features, returning them as two separate DataFrames.

    Args:
        csv_path (Path): The path to the CSV file containing the dataset.
        target_col (str): The name of the column in the CSV file that represents
                          the target variable.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
            - X_df (pd.DataFrame): The DataFrame of features (all columns except the target).
            - y_df (pd.DataFrame): The DataFrame containing only the target column.

    Raises:
        ValueError: If the `target_col` is not found in the columns of the loaded CSV file.

    Examples:
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> # Create a dummy CSV file for demonstration
        >>> dummy_data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [7, 8, 9]}
        >>> dummy_df = pd.DataFrame(dummy_data)
        >>> dummy_csv_path = Path("dummy_dataset.csv")
        >>> dummy_df.to_csv(dummy_csv_path, index=False)
        >>>
        >>> X, y = load_dataset(dummy_csv_path, "target")
        >>> print("X DataFrame:\\n", X)
        >>> print("\\ny DataFrame:\\n", y)
        >>>
        >>> # Clean up the dummy file
        >>> dummy_csv_path.unlink()
        X DataFrame:
           feature1  feature2
        0         1         4
        1         2         5
        2         3         6

        y DataFrame:
           target
        0       7
        1       8
        2       9
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the CSV file.")
    X_df = df.drop(columns=[target_col])
    y_df = pd.DataFrame(df[target_col])
    return X_df, y_df


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
        Tuple[object, pd.DataFrame, pd.Series]: A tuple containing:
            - binarizer_protected (object): An instance of `Binarizer` (or similar
                                            data handler) specifically configured
                                            for the protected attributes.
            - input_data[protected_cols] (pd.DataFrame): A DataFrame containing only
                                                         the processed protected attributes.
            - target_data (pd.Series): The processed and sampled target variable.

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
            input_data[col] = input_data[col].map(map_f)

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
        samples = np.random.shuffle(np.arange(n))

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


def compute_MSD(
    csv_path: Path | str,
    target: str,
    protected_list: List[str] | None = None,
    continuous_list: List[str] = [],
    fp_map: Dict[
        str, Callable[[int], int]
    ] = {},  # TODO make this have a more general type (strs are allowed?)
    seed: int | None = None,
    n_samples: int = 1_000_000,
    time_limit: int = 600,
    n_min: int = 0,
) -> Tuple[float, float, List[str]]:
    """Compute MSD on given CSV dataset.

    This function loads a dataset, preprocesses it to identify protected attributes
    and manage feature types, then searches for a subgroup that exhibits the
    Maximum Subgroup Discrepancy (MSD) based on the specified target variable.

    Args:
        csv_path (Path | str): Path to the input CSV file.
        target (str): Name of the column to treat as the binary target variable.
        protected_list (List[str] | None, optional): A list of column names to
            treat as protected attributes. Subgroups are defined over these
            attributes. If `None`, all attributes in the dataset (except the
            target) are assumed to be protected. Defaults to `None`.
        continuous_list (List[str], optional): A list of column names to treat
            as continuous features. Defaults to `[]`.
        fp_map (Dict[str, Callable[[Any], Any] | Dict[Any, Any]], optional):
            A mapping of column names to functions or dictionaries for
            preprocessing (e.g., cardinality reduction or value mapping)
            before binarization. For instance: `{"POBP": (lambda x: x // 100)}`
            or `{"education": {"High School": 0, "College": 1}}`. Defaults to `{}`.
        seed (int | None, optional): Random seed for dataset subsampling. If `None`,
            no seed is set, resulting in non-reproducible sampling. Defaults to `None`.
        n_samples (int, optional): The maximum number of rows to take from the dataset
            after initial cleaning. If the dataset is larger, it will be randomly
            subsampled. Defaults to `1_000_000`.
        time_limit (int, optional): Time budget in seconds for the subgroup discovery
            search algorithm. Defaults to `600`.
        n_min (int, optional): Minimum required support (number of individuals) for
            a subgroup to be considered valid during the search. Defaults to `0`.

    Returns:
        Tuple[float, List[int]]: A tuple containing:
            - **MSD_val** (float): The Maximum Subgroup Discrepancy (MSD) score
              found for the best subgroup. This represents the absolute difference
              in the mean target outcome between the best subgroup and its complement.
            - **rule** (List[int]): A list of integer indices representing the
              conjuncts that define the best subgroup found. These indices correspond
              to the binarized features.

    Raises:
        FileNotFoundError: If the `csv_path` does not exist. (Implicitly raised by `load_dataset`)
        ValueError: If the `target` column is missing from the CSV file, or if a
                    subgroup or its complement becomes empty during `our_metric`
                    calculation due to `n_min` or data characteristics.
        (Other exceptions may be raised by `load_dataset`, `prepare_dataset`,
        `binarizer.encode`, `OneRule.find_rule`, or `evaluate_subgroup_discrepancy`
        depending on their internal implementations.)

    Notes:
        - This function relies on `load_dataset`, `prepare_dataset`, `binarizer`,
          `subgroup_map_from_conjuncts`, `our_metric` (renamed from `evaluate_subgroup_discrepancy`
          in the previous example's internal call), and `OneRule` to be correctly
          defined and imported.
        - The `rule` returned is a list of internal indices from the binarized data,
          not necessarily human-readable column names from the original dataset.
          Further mapping might be needed to interpret them.
    """

    csv_path = Path(csv_path)

    X_df, y_df = load_dataset(csv_path, target)
    if protected_list is None:
        logger.info("Assuming all attributes are protected")
        protected_list = list(X_df.columns)
    if seed is not None:
        logger.info(f"Seeding the run with seed={seed}")
        np.random.seed(seed)

    binarizer, X_prot, y = prepare_dataset(
        X_df,
        y_df,
        n_samples,
        protected_attrs=protected_list,
        continuous_feats=continuous_list,
        feature_processing=fp_map,
    )

    X_bin = binarizer.encode(X_prot, include_binary_negations=True)
    y_bin = binarizer.encode_y(y)

    mio = OneRule()
    rule = mio.find_rule(X_bin, y_bin, n_min=n_min, time_limit=time_limit)
    subgroup_map = subgroup_map_from_conjuncts(rule, X_bin)
    MSD_val = evaluate_subgroup_discrepancy(subgroup_map, y)
    return MSD_val, rule
