import argparse
import subprocess
import os
import sys
import re
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
        "methods": [
            "OneRule",
            "MSD",
            # "BRCG",
            # "Ripper",
            "W1",
            "W2",
            "TV",
            "MMD",
        ],
        "base_dir_prefix": os.path.join(
            project_root,
            "humancompatible", "detect", "batch_precomputed", "experiment_enumerative"
        ),
        "base_onerule": os.path.join(
            project_root,
            "humancompatible", "detect", "batch_precomputed", "experiment_sample_complexity"
        ),
        "method_colors": {
            "OneRule": "red",
            "BRCG": "orange",
            "Ripper": "magenta",
            "TV": "green",
            "W1": "blue",
            "W2": "cyan",
            "MMD": "brown",
            "MSD": "black",
        },
        "method_markers": {
            "OneRule": "s",
            # "BRCG": "--",
            # "Ripper": "--",
            "TV": "D",
            "W1": "D",
            "W2": "D",
            "MMD": "D",
            "MSD": "D",
        },
        "method_names": {
            "OneRule": "MSD (ours)",
            # "BRCG": "MSD (via BRCG)",
            # "Ripper": "MSD (via Ripper)",
            "W1": r"MSDD$_{\mathrm{W}_1}$ (naive)",
            "W2": r"MSDD$_{\mathrm{W}_2}$ (naive)",
            "TV": r"MSDD$_\mathrm{TV}$ (naive)",
            "MMD": r"MSDD$_\mathrm{MMD}$ (naive)",
            "MSD": "MSD (naive)",
        },
    }

def extract_data(config):
    """Read all output.txt files and assemble a nested results dict.

    Args:
        config (dict): configuration dict from `load_config`.

    Returns:
        dict: data_dict[method][valname][scenario] = list of values.
    """
    scenarios = config["scenarios"]
    methods = config["methods"]
    base_dir_prefix = config["base_dir_prefix"]
    base_onerule = config["base_onerule"]

    extracted = []
    sub_size = {}

    for method in methods:
        base_dir = base_onerule if method == "OneRule" else base_dir_prefix
        for scenario in scenarios:
            for seed in range(5):
                folder = os.path.join(base_dir, f"{method}-{scenario}", str(seed))
                out_f = os.path.join(folder, "output.txt")
                if not os.path.isfile(out_f):
                    print(f"passing {seed} {out_f}")
                    continue
                with open(out_f, "r", errors="ignore") as f:
                    for line in f:
                        m = re.search(r"Distances reported: \[(.+)\]", line)
                        if m:
                            vals = [float(v) for v in m.group(1).split(", ")]
                            extracted.append(("Distance", scenario, method, vals[-1]))
                        m = re.search(r"Optimal/Valid flags: \[(.+)\]", line)
                        if m:
                            flag = m.group(1).split(", ")[-1].strip() == "True"
                            extracted.append(("Optimal/Valid", scenario, method, flag))
                        m = re.search(r"Total options: (\d+)", line)
                        if m:
                            sz = int(m.group(1))
                            extracted.append(("Number of subgroups", scenario, method, sz))
                            sub_size[scenario] = sz
                        m = re.search(r"Checked options: (\d+)", line)
                        if m:
                            csz = int(m.group(1))
                            extracted.append(
                                ("Number of considered subgroups", scenario, method, csz)
                            )

    for scenario, sz in sub_size.items():
        for _ in range(5):
            extracted.append(("Number of subgroups", scenario, "OneRule", sz))
            extracted.append(
                ("Number of considered subgroups", scenario, "OneRule", sz)
            )

    data = {}
    for valname, scenario, method, val in extracted:
        data.setdefault(method, {}).setdefault(valname, defaultdict(list))[scenario].append(val)

    return data

def plot_enumeration(data, config):
    methods = config["methods"]
    scenarios = config["scenarios"]
    colors = config["method_colors"]
    markers = config["method_markers"]
    names = config["method_names"]

    r, c = 1, 1
    fig, axs = plt.subplots(r, c, figsize=(5, 5))
    axs.grid(True, which="both", ls=":")
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_title("Proportion of subgroups evaluated within 10 minutes")

    for method in methods:
        if method not in data:
            continue
        data_dict = data[method]
        for j, scenario in enumerate(scenarios):
            if scenario not in data_dict["Number of considered subgroups"]:
                continue
            # ax = axs[j // c, j % c]
            ax = axs

            vals = np.array(data_dict["Number of considered subgroups"][scenario])
            if "Optimal/Valid" in data_dict:
                validity = np.array(data_dict["Optimal/Valid"][scenario], dtype=bool)
            else:
                validity = np.ones_like(vals, dtype=bool)
            if not validity.any():
                continue
            y = np.mean(vals[validity])
            x = data_dict["Number of subgroups"][scenario][0]

            valid_samples = validity.sum(axis=0)
            ax.scatter(
                x,
                y,
                s=100 if markers[method] == "D" else 150,
                linewidth=2,
                marker=markers[method],
                facecolor="none",
                # linestyle=method_lines[method],
                color=colors[method],
                label=f"{names[method]}" if j == 0 else None,
                zorder=10,
                alpha=0.5,
            )

            if j % c == 0:
                ax.set_ylabel("Mean number of considered subgroups")
            if j // c == 1:
                ax.set_xlabel("Number of subgroups")
    handles, labels = axs.get_legend_handles_labels()

    # fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=(len(methods) + 1) // 2,
        bbox_to_anchor=(0.5, 0),
    )

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    output_path = "enumeration.pdf"
    plt.savefig(output_path)

    plt.show()

def main():
    cfg = load_config()
    data = extract_data(cfg)
    plot_enumeration(data, cfg)


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

