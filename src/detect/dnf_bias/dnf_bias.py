import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data_handler import DataHandler
from binarizer import Binarizer
from utils import MMD, TV_binarized, our_metric, wasserstein_distance

from helper import subg_generator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

counter = DictConfig({"n_options": 0, "n_checked": 0, "n_skipped": 0})


def _parse_cli() -> Dict[str, str]:
    if len(sys.argv) == 1:
        sys.exit("Usage: python dnf_bias.py dataset_path=<csv> result_folder=<out_dir> [key=value …]")
    args: Dict[str, str] = {}
    for token in sys.argv[1:]:
        if "=" not in token:
            sys.exit(f"Invalid arg '{token}', expected key=value")
        k, v = token.split("=", 1)
        args[k] = v
    if "dataset_path" not in args or "result_folder" not in args:
        sys.exit("Arguments 'dataset_path' and 'result_folder' are required.")
    return args

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


def run_enumerative(
    binarizer,
    X_orig,
    y_orig,
    binarizer_protected,
    X_prot_orig,
    cfg
):

    n_samples = X_orig.shape[0]
    # n_train_samples = cfg.train_samples
    # while n_train_samples > n_samples:
    #     n_train_samples = n_train_samples // 2
    # train_i = np.random.choice(n_samples, size=n_train_samples, replace=False)
    train_mask = np.ones((n_samples,), dtype=bool)
    # train_mask[train_i] = True

    X = binarizer.encode(X_orig[train_mask], include_negations=False)
    X_prot = binarizer_protected.encode(
        X_prot_orig[train_mask], include_negations=False, include_binary_negations=False
    )
    X_categ = np.empty_like(X_prot_orig[train_mask], dtype=int)
    offset = 0
    for i, f in enumerate(binarizer_protected.get_bin_encodings(return_flat=False)):
        j = len(f)
        if j == 1:
            # binary
            X_categ[:, i] = X_prot[:, offset]
        else:
            # categorized
            X_categ[:, i] = np.argmax(X_prot[:, offset : offset + j], axis=1)
        offset += j
    y = binarizer.encode_y(y_orig[train_mask])
    # X_enc = dhandler.encode(X_orig[train_mask])

    n_samples, d = X.shape
    d_prot = X_prot.shape[1]

    d = X_prot.shape[1]
    X0 = X_prot[~y].astype(float)
    X1 = X_prot[y].astype(float)
    if cfg.model == "MMD":
        X0cat = X_categ[~y].astype(float)
        X1cat = X_categ[y].astype(float)

    global counter
    counter.n_options = 0
    counter.n_checked = 0
    counter.n_skipped = 0

    subgroups = subg_generator(X_prot, cfg.n_min, binarizer_protected, counter, logger)

    max_dist = 0
    max_sg = ([], [])
    t_start = time.time()
    for sg_pos, sg_neg in subgroups:
        counter.n_checked += 1
        logger.info(f"{sg_pos}, {sg_neg}")
        if cfg.model == "MSD":
            mask = X_prot[:, sg_pos].all(axis=1) & (~X_prot[:, sg_neg]).all(axis=1)
            dist = our_metric(y, mask)
        else:
            mask0 = (X0[:, sg_pos] == 1).all(axis=1) & (X0[:, sg_neg] == 0).all(axis=1)
            mask1 = (X1[:, sg_pos] == 1).all(axis=1) & (X1[:, sg_neg] == 0).all(axis=1)
            X0_sub = X0[mask0]
            X1_sub = X1[mask1]
            if X0_sub.shape[0] == 0 or X1_sub.shape[0] == 0:
                continue
            # remove the fixed columns
            fixed_idxs = sorted(sg_pos + sg_neg)
            col_mask = []
            cat_col_mask = []
            i = 0  # index of currently sought fixed value
            for f in binarizer_protected.get_bin_encodings(return_flat=False):
                if i >= len(fixed_idxs) or fixed_idxs[i] > offset + len(f):
                    col_mask += [1] * len(f)
                    cat_col_mask.append(1)
                else:
                    col_mask += [0] * len(f)
                    cat_col_mask.append(0)
                    i += 1
            col_mask = np.array(col_mask, dtype=bool)
            if np.all(~col_mask):
                continue
            X0_sub = X0_sub[:, col_mask]
            X1_sub = X1_sub[:, col_mask]

            if cfg.model in ["W1", "W2"]:
                dist = wasserstein_distance(
                    X0_sub, X1_sub, Wtype=cfg.model, true_dimension=X_prot_orig.shape[1]
                )
            elif cfg.model == "TV":
                dist = TV_binarized(X0_sub, X1_sub)
            elif cfg.model == "MMD":
                X0_sub = X0cat[mask0][:, cat_col_mask]  # keeps the shape correctly
                X1_sub = X1cat[mask1][:, cat_col_mask]  # keeps the shape correctly
                dist = MMD(X0_sub, X1_sub)
            else:
                raise ValueError(f"Not implemented for {cfg.model}")

        if dist > max_dist:
            max_dist = dist
            mask = (X_prot[:, sg_pos] == 1).all(axis=1) & (X_prot[:, sg_neg] == 0).all(
                axis=1
            )
            max_MSD = our_metric(y, mask)
            max_sg = (sg_pos, sg_neg)
        if time.time() - t_start >= cfg.time_limit:
            break
    t_tot = time.time() - t_start

    bin_feats = binarizer_protected.get_bin_encodings(include_binary_negations=False)
    term = [bin_feats[r] for r in max_sg[0]]
    term += [bin_feats[r].negate_self() for r in max_sg[1]]

    run_dir = cfg.result_folder

    import os
    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write("RESULT\n")
        out_file.write(f"Max distance: {max_dist} \n")
        out_file.write(f"Max MSD: {max_MSD} \n")
        out_file.write(f"Max subgroup: ({' AND '.join(sorted(map(str, term)))}) \n")
        out_file.write(f"Total options: {counter.n_options} \n")
        out_file.write(f"Checked options: {counter.n_checked} \n")
        out_file.write(f"Of that skipped options: {counter.n_skipped} \n")
        out_file.write(f"Time spent: {t_tot} \n")
        out_file.write(f"True number of training samples: {n_samples} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Full dimension: {d} \n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")




def main():
    cli = _parse_cli()
    csv = Path(cli.pop("dataset_path"))
    out = Path(cli.pop("result_folder"))

    protected_list = cli.pop("protected", "").split(",") if "protected" in cli else []
    protected_list = [c.strip() for c in protected_list if c.strip()]
    continuous_list = cli.pop("continuous", "").split(",") if "continuous" in cli else []
    continuous_list = [c.strip() for c in continuous_list if c.strip()]

    fp_map: Dict[str, int] = {}
    fp_spec = cli.pop("feature_processing", "")
    if fp_spec:
        for item in fp_spec.split(","):
            col, div = item.split(":")
            fp_map[col.strip()] = int(div)

    target = cli.pop("target", csv.stem)

    model = cli.pop("model", "MMD")
    seed = int(cli.pop("seed", 0))
    n_samples = int(cli.pop("n_samples", 1000000))
    train_samples = int(cli.pop("train_samples", 100000))
    time_limit = int(cli.pop("time_limit", 600))
    n_min = int(cli.pop("n_min", 10))

    if cli:
        logger.warning("Ignoring unknown args: %s", list(cli.keys()))

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
        protected_attrs=protected_list, # or [col for col in X_df.columns if col.lower() in {"sex", "race"}],
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

    run_enumerative(
        binarizer=bin_all,
        X_orig=X_df,
        y_orig=y_df,
        binarizer_protected=bin_prot,
        X_prot_orig=X_prot_df,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()


'''
python src/detect/dnf_bias/dnf_bias.py 
        dataset_path=src/detect/dnf_bias/data/ACSIncome_CA.csv 
        result_folder=results_dir 
        target=PINCP
        protected=SEX,RAC1P,AGEP,POBP,_POBP,DIS,CIT,MIL,ANC,NATIVITY,DEAR,DEYE,DREM,FER,POVPIP
        continuous=AGEP,PINCP,WKHP,JWMNP,POVPIP
        feature_processing=POBP:100,OCCP:100,PUMA:100,POWPUMA:1000

        model=MMD
        seed=0
        n_samples=1000000
        train_samples=100000
        time_limit=600
        n_min=10

'''


