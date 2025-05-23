import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .binarizer import Binarizer
from .data_handler import DataHandler
from .helper import subg_generator
from .utils import MMD, TV_binarized, our_metric, wasserstein_distance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

counter = DictConfig({"n_options": 0, "n_checked": 0, "n_skipped": 0})

def load_dataset(csv_path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in CSV.")
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
    seed: int = 0,
):
    mask = ~input_data.isnull().any(axis=1)
    logger.debug(f"Removing {input_data.shape[0] - mask.sum()} rows with nans")
    input_data = input_data[mask.values]
    target_data = target_data[mask.values]

    # Preprocess the data
    for col, div in feature_processing.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(lambda x: int(x) // div)

    values = {}
    bounds = {}
    for col in input_data.columns:
        vals = input_data[col].unique()
        logger.debug(f"{col} has {vals.shape[0]} values")
        if vals.shape[0] <= 1:
            input_data.drop(columns=[col], inplace=True)
            continue
        if col not in continuous_feats:
            values[col] = vals
        else:
            bounds[col] = (min(vals), max(vals))

    # print(values)
    # print(bounds)

    np.random.seed(seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(n_max, n), replace=False)

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )

    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    protected_cols = [col for col in input_data.columns if col in protected_attrs]
    dhandler_protected = DataHandler.from_data(
        input_data[protected_cols],
        target_data,
        categ_map=values,
        bounds_map=bounds,
    )
    binarizer_protected = Binarizer(dhandler_protected, target_positive_vals=[True])

    return (
        binarizer,
        dhandler,
        input_data,
        target_data,
        binarizer_protected,
        input_data[protected_cols],
    )




def prepare_logs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "output.txt"
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
           for h in root.handlers):
        return
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)


def compute_MSD(
    csv: Path | str,
    target: str,
    protected_list: List[str] | None = None, # none assumes all are protected
    continuous_list: List[str] = [],
    fp_map: Dict[str, Callable[]] = {}, # we'd like this to be a mapping to functions {"POBP":lambda x: x//100} for example
    seed: int | None = None,
    n_samples: int = 1_000_000,
    time_limit: int = 600,
    n_min: int = 0,
) -> Tuple[float, float, List[str]]:
    """Running DNF bias detection experiments on given CSV dataset.

    Args:
        csv (Path, str): Path to the input CSV file.
        out (Path, str): Directory where 'output.txt' will be saved.
        target (str): Name of the column to treat as the binary target variable.
        protected_list (List[str]): Comma-separated list of columns to treat as protected attributes; subgroups
            are defined over these attributes.
        continuous_list (List[str], optional): List of columns to treat as continuous features. Defaults to [].
        fp_map (Dict[str, int], optional): Mapping of column names to integer divisors for
            cardinality reduction before binarization (e.g., {"POBP":100}). Defaults to {}.
        model (str, optional): Distance metric to optimize. One of
            - 'MMD' - Maximum Mean Discrepancy
            - 'MSD' - Mean Subgroup Difference
            - 'W1' - 1-Wasserstein (Earth Mover's Distance)
            - 'W2' - 2-Wasserstein
            - 'TV' - Total Variation
            Defaults to 'MMD'.
        seed (int, optional): Random seed for dataset subsampling. Defaults to 0.
        n_samples (int, optional): Maximum number of rows to sample from the dataset. Defaults to 1_000_000.
        train_samples (int, optional): Reserved for future use. Defaults to 100_000.
        time_limit (int, optional): Time budget for the search in seconds. Defaults to 600.
        n_min (int, optional): Minimum subgroup support (number of rows). Defaults to 10.

    Returns:
        Tuple[float, float, List[str]]: A tuple of:
            - max_distance (float): The highest distance value found across all
                valid subgroups.
            - max_msd (float): The MSD score corresponding to the best subgroup.
            - best_literals (List[str]): A list of human-readable strings
                describing the positive/negative bins that define the best subgroup.

    Writes a summary file ('output.txt') into 'out' containing
       detailed information about the run (max distance, subgroup description,
       counters, and timing).
    """

    csv = Path(csv)

    X_df, y_df = load_dataset(csv, target)

    (
        bin_all,
        _handler,
        X_df,
        y_df,
        bin_prot,
        X_prot_df,
    ) = prepare_dataset(
        X_df,
        y_df,
        n_samples,
        protected_attrs=protected_list,
        continuous_feats=continuous_list,
        feature_processing=fp_map,
        seed=seed,
    )

    cfg = DictConfig(
        {
            "model": model,
            "seed": seed,
            "n_samples": n_samples,
            "train_samples": train_samples,
            "time_limit": time_limit,
            "n_min": n_min,
            "result_folder": out,
        }
    )

    prepare_logs(out)
    return run_enumerative(
        binarizer=bin_all,
        X_orig=X_df,
        y_orig=y_df,
        binarizer_protected=bin_prot,
        X_prot_orig=X_prot_df,
        cfg=cfg,
    )


