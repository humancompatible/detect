import os
import re
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
    "MSD",
    # "BRCG",
    # "Ripper",
    "W1",
    "W2",
    "TV",
    "MMD",
]

base_dir_prefix = (
    "batch_precomputed/experiment_enumerative"  # from cluster, final (hopefully)
)

base_onerule = "batch_precomputed/experiment_sample_complexity"

method_colors = {
    "OneRule": "red",
    "BRCG": "orange",
    "Ripper": "magenta",
    "TV": "green",
    "W1": "blue",
    "W2": "cyan",
    "MMD": "brown",
    "MSD": "black",
}
method_lines = {
    "OneRule": "--",
    "BRCG": "--",
    "Ripper": "--",
    "TV": "-",
    "W1": "-",
    "W2": "-",
    "MMD": "-",
    "MSD": "--",
}
method_markers = {
    "OneRule": "s",
    # "BRCG": "--",
    # "Ripper": "--",
    "TV": "D",
    "W1": "D",
    "W2": "D",
    "MMD": "D",
    "MSD": "D",
}
method_names = {
    "OneRule": "MSD (ours)",
    # "BRCG": "MSD (via BRCG)",
    # "Ripper": "MSD (via Ripper)",
    "W1": "MSDD$_{{\\mathrm{{W}}_1}}$ (naive)",
    "W2": "MSDD$_{{\\mathrm{{W}}_2}}$ (naive)",
    "TV": "MSDD$_\\mathrm{{TV}}$ (naive)",
    "MMD": "MSDD$_\\mathrm{{MMD}}$ (naive)",
    "MSD": "MSD (naive)",
}


def extract_data():
    extracted_data = []
    sub_size = {}

    for method in methods:
        base_dir = base_dir_prefix
        if method == "OneRule":
            base_dir = base_onerule
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
                            extracted_data.append(
                                ("Distance", scenario, method, dists[-1])
                            )
                        opt = re.search(r"Optimal/Valid flags: \[(.+)\]", line)
                        if opt:
                            opts = [
                                v.strip() == "True" for v in opt.group(1).split(", ")
                            ]
                            extracted_data.append(
                                ("Optimal/Valid", scenario, method, opts[-1])
                            )
                        nsubgroups = re.search(r"Total options: (\d+)", line)
                        if nsubgroups:
                            subgroups = int(nsubgroups.group(1))
                            extracted_data.append(
                                ("Number of subgroups", scenario, method, subgroups)
                            )
                            sub_size[scenario] = subgroups
                        seen_subgroups = re.search(r"Checked options: (\d+)", line)
                        if seen_subgroups:
                            subgroups = int(seen_subgroups.group(1))
                            extracted_data.append(
                                (
                                    "Number of considered subgroups",
                                    scenario,
                                    method,
                                    subgroups,
                                )
                            )
    for scenario in scenarios:
        extracted_data += [
            ("Number of subgroups", scenario, "OneRule", sub_size[scenario])
        ] * 5
        extracted_data += [
            ("Number of considered subgroups", scenario, "OneRule", sub_size[scenario])
        ] * 5

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

r, c = 1, 1
# fig, axs = plt.subplots(r, c, figsize=(15, 6))
fig, axs = plt.subplots(r, c, figsize=(5, 5))
axs.grid(True, which="both", ls=":")
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_title("Proportion of subgroups evaluated within 10 minutes")

for method in methods:
    if method not in all_data:
        continue
    data_dict = all_data[method]
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
            s=100 if method_markers[method] == "D" else 150,
            linewidth=2,
            marker=method_markers[method],
            facecolor="none",
            # linestyle=method_lines[method],
            color=method_colors[method],
            label=f"{method_names[method]}" if j == 0 else None,
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
