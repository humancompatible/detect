# HumanCompatible · Bias Detection
A toolbox for measuring bias in data & models

<div align="center">

**Maximum Subgroup Discrepancy (MSD)** - bias metric with linear sample complexity  
*...with a MILP formulation that also tells you **which subgroup is most affected**.*

</div>

---

## Quick install & 60-second demo

```bash
python -m pip install git+https://github.com/humancompatible/detect.git
```

```python
from humancompatible.detect import detect_bias_csv

msd_val, rule = detect_bias_csv(
    csv_path="census.csv",                  # any CSV file
    target_col="income_50k",                # binary target
    protected_list=["race", "age"],         # columns to audit
    method="MSD",                           # chosen method
)

print(f"MSD = {msd_val:.3f}", "Rule ->", rule) 
```

The function returns

- **`msd_val`** – the maximum gap (in percentage‐points) between any subgroup and its complement  
- **`rule`** – the raw subgroup encoding as a list of `(feature_index, Bin)` pairs.  
  To get a human‐readable description, do the following:

  ```python
  pretty = " AND ".join(str(cond) for _, cond in rule)
  print("Subgroup:", pretty)
  # -> "Subgroup: Race = Blue AND Age = 0-18"
  ```

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
- [**Examples**](https://github.com/humancompatible/detect/blob/main/examples/) -> Start with a simple [example notebook](https://github.com/humancompatible/detect/blob/main/examples/01_usage.ipynb), or go directly to a [realistic example using Folktables](https://github.com/humancompatible/detect/blob/main/examples/02_folktables.ipynb)

---

## MSD as a distance?
Bias detection can be understood as measuring some distance between two distributions (positive X negative samples, some training dataset X general population data...).

However, most distances have exponential sample complexity, whereas MSD requires a linear number of samples (w.r.t. the dimension) to achieve the same error.

| Classical metric               | Needs full d‐dim joint? | Sample cost | Drawbacks                                        |
|-------------------------------:|:-----------------------:|:-----------:|-------------------------------------------------|
| Wasserstein, TV, MMD, …        | yes                     | Ω(2^d)      | exponential samples, no subgroup info           |
| **MSD (ours)**              | only protected attrs    | O(d)        | ✓ returns exact subgroup & gap                  |

MSD maximises the absolute difference in probability over all protected‐attribute combinations (subgroups), yet is solvable in practice through an exact Mixed‐Integer optimization that scans the doubly‐exponential space effectively.

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
