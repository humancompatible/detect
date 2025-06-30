# HumanCompatible · Bias Detection
_A toolbox for spotting subgroup-level bias in data & models_

<div align="center">

**Maximum Subgroup Discrepancy (MSD)** — linear-time bias metric  
*…with an exact MILP solver that also tells you **which** group is affected.*

</div>

---

## Quick install & 60-second demo

```bash
python -m pip install git+https://github.com/humancompatible/detect.git
```

```python
from humancompatible.detect import detect_bias_csv

msd_val, rule = detect_bias_csv(
    csv_path="census.csv",                  # any CSV
    target_col="income_50k",                # binary or numeric target
    protected_list=["race", "age"],         # columns to audit
    method="MSD",                           # one-liner does the rest
)

print(f"MSD = {msd_val:.3f}", "Rule ->", rule)
```

The function returns

- **`msd_val`** – the gap (in percentage‐points) for the worst‐off subgroup  
- **`rule`**     – the subgroup itself, as an interpretable conjunction (e.g.  
  `race = Black AND age ∈ [18,25)`)


## Why another distance?

| Classical metric               | Needs full d‐dim joint? | Sample cost | Drawbacks                                        |
|-------------------------------:|:-----------------------:|:-----------:|-------------------------------------------------|
| Wasserstein, TV, MMD, …        | yes                     | Ω(2^d)      | exponential samples, no subgroup info           |
| **MSD (ours)**              | only protected attrs    | O(d)        | ✓ returns exact subgroup & gap                  |

MSD maximises the absolute mass difference over all protected‐attribute combinations, yet is solvable in practice through an exact Mixed‐Integer optimization that scans the doubly‐exponential space implicitly.

---

## Contents

<!-- Probably for the future -->
<!-- ```{toctree}
:maxdepth: 2
:hidden:

tutorials/quickstart
examples/index
user_guide/index
api/modules
contributing
``` -->

```{toctree}
:maxdepth: 1

api/modules
```


<!-- TODO: I think it would be better to change on the way, as above -->
- [**Tutorial**](https://github.com/humancompatible/detect/blob/main/README.md) -> Your first audit in 5 minutes  
- [**Examples**](https://github.com/humancompatible/detect/blob/main/examples/01_usage.ipynb) -> Step by step example notebook, also on [Folktables](https://github.com/humancompatible/detect/blob/main/examples/02_folktables.ipynb)

---

## Citation

If you use MSD, please cite:

```bibtex
@inproceedings{MSD,
  author = {Jiří Němeček and Mark Kozdoba and Illia Kryvoviaz and Tomáš Pevný and Jakub Mareček},
  title = {Bias Detection via Maximum Subgroup Discrepancy},
  year = {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  series = {KDD '25}
}
```

Looking for the installation matrix, solver details or developer setup?  
Head to the [**README -> Installation**](https://github.com/humancompatible/detect?tab=readme-ov-file#installation-details) section.
