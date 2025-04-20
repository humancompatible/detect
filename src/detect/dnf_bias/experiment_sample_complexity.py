import logging
import os
import subprocess
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from methods import test_BRCG, test_RIPPER
from one_rule import OneRule
from scenarios.folktables_scenarios import load_scenario
from utils import (
    MMD,
    TV_binarized,
    balance_datasets,
    eval_terms,
    our_metric,
    wasserstein_distance,
)

gitcommit = ""

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="distances")
def run_experiment(cfg: DictConfig):
    binarizer, dhandler, X_orig, y_orig, binarizer_protected, X_prot_orig = (
        load_scenario(cfg.scenario, cfg.seed, cfg.n_samples, state=cfg.state)
    )

    # cfg needs to have
    # state, scenario, seed, n_samples ~ total n of samples, train_samples, model

    np.random.seed(cfg.seed)
    n_samples = X_orig.shape[0]
    sample_sizes = np.ceil(np.geomspace(1000, n_samples, num=5)).astype(int)
    distances = []
    terms = []
    times = []
    opts = []
    true_ns = []
    true_n = None
    for sample_size in sample_sizes:
        train_i = np.random.choice(n_samples, size=sample_size, replace=False)
        train_mask = np.zeros((n_samples,), dtype=bool)
        train_mask[train_i] = True
        # # test_mask = ~train_mask
        # # test_i = np.where(test_mask)[0]

        X = binarizer.encode(X_orig[train_mask], include_negations=False)
        X_prot = binarizer_protected.encode(
            X_prot_orig[train_mask],
            include_negations=False,
            include_binary_negations=False,
        )
        # for the MIO we need a binary feature for each possible value
        X_prot_full = binarizer_protected.encode(
            X_prot_orig[train_mask],
            include_negations=False,
            include_binary_negations=True,
        )
        # for evaluating Ripper, we need all negations
        X_prot_ripper_eval = binarizer_protected.encode(
            X_prot_orig[train_mask], include_negations=True
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

        d = X.shape[1]
        d_prot = X_prot.shape[1]

        t_start = time.time()

        opt = True
        if cfg.model in ["OneRule", "OneRuleBalanceData"]:
            if cfg.model == "OneRuleBalanceData":
                y, X_prot_full = balance_datasets(y, [y, X_prot_full], seed=cfg.seed)
                true_n, d = X_prot_full.shape
            onerule = OneRule()
            conj, opt = onerule.find_rule(
                X_prot_full,
                y,
                verbose=True,
                time_limit=cfg.time_limit,
                n_min=cfg.n_min,
                return_opt_flag=True,
            )
            y_hat = np.ones_like(y, dtype=bool)
            for c in conj:
                y_hat &= X_prot_full[:, c]
            dist = our_metric(y, y_hat)
            d = X_prot_full.shape[1]
            bin_feats = binarizer_protected.get_bin_encodings(
                include_binary_negations=True
            )
            terms.append([bin_feats[r] for r in conj])
        elif cfg.model == "Ripper":
            y, X_prot, X_prot_ripper_eval = balance_datasets(
                y, [y, X_prot, X_prot_ripper_eval], seed=cfg.seed
            )
            true_n, d = X_prot.shape
            y_hat, dnf = test_RIPPER(X_prot, y, X_prot, binarizer_protected)
            y_hat_true = eval_terms(dnf, binarizer_protected, X_prot_ripper_eval)[0]
            if not (np.array(y_hat) == y_hat_true).all():
                logger.warning("There is an issue in the RIPPER changes")
            dist = our_metric(y, y_hat_true)
            _, dnf = test_RIPPER(X_prot, ~y, X_prot, binarizer_protected)
            y_hat_true2 = eval_terms(dnf, binarizer_protected, X_prot_ripper_eval)[0]
            dist = max(dist, our_metric(y, y_hat_true2))
            terms.append(dnf[0])
        elif cfg.model == "BRCG":
            y, X_prot_full = balance_datasets(y, [y, X_prot_full], seed=cfg.seed)
            true_n, d = X_prot_full.shape
            _, dnf = test_BRCG(X_prot_full, y, X_prot_full, binarizer_protected)
            # y_hat is not correct for the returned single conjuntion
            # print((y_hat == eval_terms(dnf, binarizer_protected, X_prot_full)[0]).all())
            y_hat = eval_terms(
                dnf, binarizer_protected, X_prot_full, binary_negs_only=True
            )[0]
            dist = our_metric(y, y_hat)
            _, dnf = test_BRCG(X_prot_full, ~y, X_prot_full, binarizer_protected)
            y_hat2 = eval_terms(
                dnf, binarizer_protected, X_prot_full, binary_negs_only=True
            )[0]
            dist = max(dist, our_metric(y, y_hat2))
            terms.append(dnf[0])
        elif cfg.model in ["W1", "W2"]:
            d = X_prot.shape[1]
            X0 = X_prot[~y].astype(float)
            X1 = X_prot[y].astype(float)
            dist = wasserstein_distance(
                X0, X1, Wtype=cfg.model, true_dimension=X_prot_orig.shape[1]
            )
        elif cfg.model == "TV":
            d = X_prot.shape[1]
            X0 = X_prot[~y].astype(int)
            X1 = X_prot[y].astype(int)
            dist = TV_binarized(X0, X1)
        elif cfg.model == "MMD":
            d = X_categ.shape[1]
            X0 = X_categ[~y].astype(float)
            X1 = X_categ[y].astype(float)
            dist = MMD(X0, X1)
        else:
            raise ValueError(f"Unknown fair classifier {cfg.model}")

        t_tot = time.time() - t_start

        logger.info(
            f"Finished size {sample_size} ({np.where(sample_sizes == sample_size)[0][0]+1}/{len(sample_sizes)})"
        )
        distances.append(dist)
        times.append(t_tot)
        opts.append(opt)
        true_ns.append(true_n if true_n is not None else sample_size)

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {gitcommit}\n\n")
        out_file.write("RESULT\n")
        if cfg.model in ["OneRule", "Ripper", "BRCG", "OneRuleBalanceData"]:
            out_file.write(
                f"Subgroups found: [{' | '.join([' AND '.join(sorted(map(str, term))) for term in terms])}] \n"
            )
        out_file.write(f"Distances reported: {distances} \n")
        out_file.write(f"Times spent: {times} \n")
        out_file.write(f"Optimal/Valid flags: {opts} \n")
        out_file.write(f"True numbers of training samples: {true_ns} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Full dimension: {d} \n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")


if __name__ == "__main__":
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip() == "":
        res = subprocess.run(
            ["git", "rev-list", "--format=%B", "-n", "1", "HEAD"],
            capture_output=True,
            text=True,
        )
        gitcommit = res.stdout.strip()
        run_experiment()
    else:
        raise Exception("Git status is not clean. Commit changes first.")
