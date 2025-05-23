import argparse
import subprocess
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_config():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    return {
        "scenarios": [
            "ACSIncome",
            "ACSPublicCoverage",
            "ACSMobility",
            "ACSEmployment",
            "ACSTravelTime",
            "DifferentStates-HI-ME",
            "DifferentStates-CA-WY",
            "DifferentStates-MS-NH",
            "DifferentStates-MD-MS",
            "DifferentStates-LA-UT",
        ],
        "scenario_titles": {
            "ACSIncome": "ACSIncome (CA)",
            "ACSPublicCoverage": "ACSPublicCoverage (CA)",
            "ACSMobility": "ACSMobility (CA)",
            "ACSEmployment": "ACSEmployment (CA)",
            "ACSTravelTime": "ACSTravelTime (CA)",
            "DifferentStates-HI-ME": "Havaii X Maine",
            "DifferentStates-CA-WY": "California X Wyoming",
            "DifferentStates-MS-NH": "Mississippi X New Hampshire",
            "DifferentStates-MD-MS": "Maryland X Mississippi",
            "DifferentStates-LA-UT": "Louisiana X Utah",
        },
        "methods": [
            "OneRule",
            # "OneRuleBalanceData",
            "BRCG",
            "Ripper",
            "W1",
            "W2",
            "TV",
            "MMD",
        ],
        "base_dir_prefix": os.path.join(
            project_root,
            "humancompatible", "detect", "batch_precomputed", "experiment_sample_complexity"
        ),
        "method_colors": {
            "OneRuleBalanceData": "red",
            "OneRule": "red",
            "BRCG": "orange",
            "Ripper": "magenta",
            "TV": "green",
            "W1": "blue",
            "W2": "cyan",
            "MMD": "brown",
        },
        "method_lines": {
            "OneRuleBalanceData": "--",
            "OneRule": "--",
            "BRCG": "--",
            "Ripper": "--",
            "TV": "-",
            "W1": "-",
            "W2": "-",
            "MMD": "-",
        },
        "method_names": {
            "OneRuleBalanceData": "MSD (ours)",
            "OneRule": "MSD (ours)",
            "BRCG": "MSD (via BRCG)",
            "Ripper": "MSD (via Ripper)",
            "W1": "Wasserstein-1",
            "W2": "Wasserstein-2",
            "TV": "Total Variation",
            "MMD": "MMD",
        },
    }

def extract_data(config):
    """Read all experiment output.txt files into a nested dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: data[method][metric][scenario] = list of parsed values.
    """
    scenarios = config["scenarios"]
    methods = config["methods"]
    base_dir = config["base_dir_prefix"]

    extracted = []
    for method in methods:
        for scenario in scenarios:
            for i in range(5):
                folder = os.path.join(base_dir, f"{method}-{scenario}", str(i))
                out_f = os.path.join(folder, "output.txt")
                if not os.path.isfile(out_f):
                    print(f"passing {i} {out_f}")
                    continue
                with open(out_f, "r", errors="ignore") as f:
                    for line in f:
                        m = re.search(r"Distances reported: \[(.+)\]", line)
                        if m:
                            d = [float(v) for v in m.group(1).split(", ")]
                            extracted.append(("Distance", scenario, method, d))
                        m = re.search(r"Optimal/Valid flags: \[(.+)\]", line)
                        if m:
                            flag = [v.strip()=="True" for v in m.group(1).split(", ")]
                            extracted.append(("Optimal/Valid", scenario, method, flag))
                        m = re.search(r"True numbers of training samples: \[(.+)\]", line)
                        if m:
                            s = [int(v) for v in m.group(1).split(", ")]
                            extracted.append(("# Samples", scenario, method, s))

    data = {}
    for metric, scen, method, val in extracted:
        data.setdefault(method, {})\
            .setdefault(metric, defaultdict(list))[scen].append(val)
    return data

def plot_metrics(data, config, plot_type):
    """Generate and save a 2×5 grid of plots for the requested metric.

    Args:
        data (dict): Extracted data from `extract_data()`.
        config (dict): Configuration dict.
        plot_type (str): One of "RSE", "relative", "base".
    """
    scenarios = config["scenarios"]
    titles = config["scenario_titles"]
    methods = config["methods"]
    colors = config["method_colors"]
    lines = config["method_lines"]
    names = config["method_names"]

    r, c = 2, 5
    fig, axs = plt.subplots(r, c, figsize=(12, 5))

    for method in methods:
        if method not in data:
            continue
        md = data[method]
        for j, scen in enumerate(scenarios):
            if scen not in md.get("Distance", {}):
                continue
            ax = axs[j // c, j % c]

            vals = np.array(md["Distance"][scen])
            validity = np.array(md["Optimal/Valid"][scen], dtype=bool)
            means = [
                np.mean(vals[:, k][validity[:, k]])
                for k in range(vals.shape[1])
                if validity[:, k].any()
            ]
            stds = [
                np.std(vals[:, k][validity[:, k]])
                for k in range(vals.shape[1])
                if validity[:, k].any()
            ]
            x = np.array(md["# Samples"][scen][0])[validity.any(axis=0)]

            if plot_type == "RSE":
                vs = validity.sum(axis=0)
                y = (np.array(stds) / np.sqrt(vs[vs>0])) / np.array(means)
                ax.plot(
                    x, y,
                    marker="x", linestyle=lines[method],
                    color=colors[method], label=names[method]
                )
                ax.set_ylabel("Relative Standard Error" if j % c==0 else "")
            elif plot_type == "relative":
                final = means[-1]
                rs = [
                    np.std(vals[:, k][validity[:, k]]/final)
                    for k in range(vals.shape[1]) if validity[:, k].any()
                ]
                ax.fill_between(
                    x, np.array(means)/final-np.array(rs),
                    np.array(means)/final+np.array(rs),
                    color=colors[method], alpha=0.2
                )
                ax.plot(
                    x, np.array(means)/final,
                    linestyle=lines[method], color=colors[method],
                    label=names[method]
                )
                ax.set_ylabel("Relative distance measure" if j % c==0 else "")
            elif plot_type == "base":
                ax.fill_between(
                    x, np.array(means)-np.array(stds),
                    np.array(means)+np.array(stds),
                    color=colors[method], alpha=0.2
                )
                ax.plot(
                    x, means,
                    linestyle=lines[method], color=colors[method],
                    label=names[method]
                )
                ax.set_ylabel("Distance measures" if j % c==0 else "")
            else:
                raise ValueError("Specify plot type: RSE, relative, or base")

            ax.set_xscale("log")
            ax.set_title(titles[scen])
            ax.grid(True, which="both", ls=":")
            if j // c == 1:
                ax.set_xlabel("Number of samples")

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(methods),
        bbox_to_anchor=(0.5, 0)
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    outname = f"final_complexity_{plot_type}.pdf"
    plt.savefig(outname)
    plt.show()

def main():
    """Parse CLI arg and dispatch to the appropriate plotting function."""
    if len(sys.argv) != 2 or sys.argv[1] not in ("RSE", "relative", "base"):
        print("Usage: python plots_for_paper.py [RSE|relative|base]")
        sys.exit(1)
    plot_type = sys.argv[1]

    cfg = load_config()
    data = extract_data(cfg)
    plot_metrics(data, cfg, plot_type)


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
        main()
    else:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        if result.stdout.strip() == "":
            main()
        else:
            raise RuntimeError(
                "Git status is not clean. Commit or stash changes first, "
                "or rerun with --skip-git-check."
            )

