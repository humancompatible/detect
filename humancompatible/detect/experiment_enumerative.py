import argparse
import logging
import os
import subprocess
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from humancompatible.detect.scenarios.folktables_scenarios import load_scenario
from humancompatible.detect.utils import MMD, TV_binarized, our_metric, wasserstein_distance
from humancompatible.detect.helper import subg_generator


@hydra.main(version_base="1.3", config_path="conf", config_name="enumerative")
def run_experiment(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    counter = DictConfig({"n_options": 0, "n_checked": 0, "n_skipped": 0})

    try:
        gitcommit = cfg.gitcommit
    except Exception:
        import subprocess
        from hydra.utils import get_original_cwd
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=get_original_cwd(),
        )
        gitcommit = res.stdout.strip()

    binarizer, dhandler, X_orig, y_orig, binarizer_protected, X_prot_orig = (
        load_scenario(cfg.scenario, cfg.seed, cfg.n_samples, state=cfg.state)
    )

    # cfg needs to have
    # state, scenario, seed, n_samples ~ total n of samples, train_samples, model

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

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {gitcommit}\n\n")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        help="Run even if the working tree is dirty "
             "(useful inside notebooks / CI).",
    )

    args, _ = parser.parse_known_args()

    if "--skip-git-check" in sys.argv:
        sys.argv.remove("--skip-git-check")

    skip = args.skip_git_check

    if skip:
        run_experiment()
    else:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        if result.stdout.strip() == "":
            run_experiment()
        else:
            raise RuntimeError(
                "Git status is not clean. Commit or stash changes first, "
                "or rerun with --skip-git-check."
            )

