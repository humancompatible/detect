import os
import re
import sys
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

scenarios = [
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
]
scenario_titles = {
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
}

methods = [
    "OneRule",
    # "OneRuleBalanceData",
    "BRCG",
    "Ripper",
    "W1",
    "W2",
    "TV",
    "MMD",
]

base_dir_prefix = (
    "batch_precomputed/experiment_sample_complexity"  # from cluster, final (hopefully)
)

method_colors = {
    "OneRuleBalanceData": "red",
    "OneRule": "red",
    "BRCG": "orange",
    "Ripper": "magenta",
    "TV": "green",
    "W1": "blue",
    "W2": "cyan",
    "MMD": "brown",
}
method_lines = {
    "OneRuleBalanceData": "--",
    "OneRule": "--",
    "BRCG": "--",
    "Ripper": "--",
    "TV": "-",
    "W1": "-",
    "W2": "-",
    "MMD": "-",
}
method_names = {
    "OneRuleBalanceData": "MSD (ours)",
    "OneRule": "MSD (ours)",
    "BRCG": "MSD (via BRCG)",
    "Ripper": "MSD (via Ripper)",
    "W1": "Wasserstein-1",
    "W2": "Wasserstein-2",
    "TV": "Total Variation",
    "MMD": "MMD",
}


def extract_data():
    extracted_data = []
    base_dir = base_dir_prefix

    for method in methods:
        for scenario in scenarios:
            for i in range(5):
                folder_path = os.path.join(base_dir, f"{method}-{scenario}", str(i))
                output_file = os.path.join(folder_path, "output.txt")
                if not os.path.isfile(output_file):
                    print(f"passing {i} {output_file}")
                    continue

                with open(output_file, "r", errors="ignore") as file:
                    lines = file.readlines()
                    for line in lines:
                        dist = re.search(r"Distances reported: \[(.+)\]", line)
                        if dist:
                            dists = [float(v) for v in dist.group(1).split(", ")]
                            extracted_data.append(("Distance", scenario, method, dists))
                        opt = re.search(r"Optimal/Valid flags: \[(.+)\]", line)
                        if opt:
                            opts = [
                                v.strip() == "True" for v in opt.group(1).split(", ")
                            ]
                            extracted_data.append(
                                ("Optimal/Valid", scenario, method, opts)
                            )
                        nsamples = re.search(
                            r"True numbers of training samples: \[(.+)\]", line
                        )
                        if nsamples:
                            samples = [int(v) for v in nsamples.group(1).split(", ")]
                            extracted_data.append(
                                ("# Samples", scenario, method, samples)
                            )

    data_dict = {}

    for (
        valname,
        scenario,
        method,
        val,
    ) in extracted_data:
        if method not in data_dict:
            data_dict[method] = {}
        if valname not in data_dict[method]:
            data_dict[method][valname] = defaultdict(list)
        data_dict[method][valname][scenario].append(val)

    return data_dict


all_data = extract_data()

r, c = 2, 5
# fig, axs = plt.subplots(r, c, figsize=(15, 6))
fig, axs = plt.subplots(r, c, figsize=(12, 5))

for method in methods:
    if method not in all_data:
        continue
    data_dict = all_data[method]
    for j, scenario in enumerate(scenarios):
        if scenario not in data_dict["Distance"]:
            continue
        ax = axs[j // c, j % c]

        vals = np.array(data_dict["Distance"][scenario])
        validity = np.array(data_dict["Optimal/Valid"][scenario], dtype=bool)
        sorted_mean = [
            np.mean(vals[:, k][validity[:, k]])
            for k in range(vals.shape[1])
            if validity[:, k].any()
        ]
        sorted_std = [
            np.std(vals[:, k][validity[:, k]])
            for k in range(vals.shape[1])
            if validity[:, k].any()
        ]
        x = np.array(data_dict["# Samples"][scenario][0])[validity.any(axis=0)]

        if sys.argv[1] == "RSE":
            valid_samples = validity.sum(axis=0)
            ax.plot(
                x,
                (sorted_std / np.sqrt(valid_samples[valid_samples > 0])) / sorted_mean,
                marker="x",
                linestyle=method_lines[method],
                color=method_colors[method],
                label=f"{method_names[method]}",
            )

            if j % c == 0:
                ax.set_ylabel("Relative Standard Error")
            if j // c == 1:
                ax.set_xlabel("Number of samples")
            ax.set_xscale("log")
            ax.set_title(scenario_titles[scenario])
            ax.grid(True, which="both", ls=":")
            handles, labels = ax.get_legend_handles_labels()

        elif sys.argv[1] == "relative":
            final_mean = sorted_mean[-1]
            sorted_std = [
                np.std(vals[:, k][validity[:, k]] / final_mean)
                for k in range(vals.shape[1])
                if validity[:, k].any()
            ]
            ax.fill_between(
                x,
                (np.array(sorted_mean) / final_mean - np.array(sorted_std)),
                (np.array(sorted_mean) / final_mean + np.array(sorted_std)),
                color=method_colors[method],
                alpha=0.2,
            )
            ax.plot(
                x,
                sorted_mean / final_mean,
                linestyle=method_lines[method],
                color=method_colors[method],
                label=f"{method_names[method]}",
            )

            if j % c == 0:
                ax.set_ylabel("Relative distance measure")
            if j // c == 1:
                ax.set_xlabel("Number of samples")
            ax.set_xscale("log")
            ax.set_title(scenario_titles[scenario])
            ax.grid(True, which="both", ls=":")
            # ax.legend(loc="upper right")
            handles, labels = ax.get_legend_handles_labels()

        elif sys.argv[1] == "base":
            ax.fill_between(
                x,
                (np.array(sorted_mean) - np.array(sorted_std)),
                (np.array(sorted_mean) + np.array(sorted_std)),
                color=method_colors[method],
                alpha=0.2,
            )
            ax.plot(
                x,
                sorted_mean,
                linestyle=method_lines[method],
                color=method_colors[method],
                label=f"{method_names[method]}",
            )

            if j % c == 0:
                ax.set_ylabel("Distance measures")
            if j // c == 1:
                ax.set_xlabel("Number of samples")
            ax.set_xscale("log")
            ax.set_title(scenario_titles[scenario])
            ax.grid(True, which="both", ls=":")
            # ax.legend(loc="upper right")
            handles, labels = ax.get_legend_handles_labels()

        else:
            print("specify what type of a plot you want - base, RSE, relative")
            exit()

# fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1))
fig.legend(
    handles, labels, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, 0)
)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[0, 0.05, 1, 1])
output_path = (
    f"final_complexity_{sys.argv[1]}.pdf"
)
plt.savefig(output_path)

plt.show()
