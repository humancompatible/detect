# HumanCompatible · Bias Detection

A toolbox for measuring bias in data & models

<div align="center">

**Maximum Subgroup Discrepancy (MSD)** - bias metric with linear sample complexity
_...with a MILP formulation that also tells you **which subgroup is most affected**._

</div>

---

## Quick install & 60-second demo

```bash
python -m pip install git+https://github.com/humancompatible/detect.git
```

```python
from humancompatible.detect import detect_and_score

rule, msd_val = detect_and_score(
    csv_path="./data/01_data.csv",
    target_col="Target",
    protected_list=["Race", "Age"],
    method="MSD",
)

print(f"MSD = {msd_val:.3f}\n", f"Rule = {rule}", sep="")
# MSD = 0.111
# Rule = [(0, Bin(<humancompatible.detect.data_handler.features.Categorical.Categorical object at 0x000001C330467A10>, <Operation.EQ: '='>, 'Blue')), (1, Bin(<humancompatible.detect.data_handler.features.Categorical.Categorical object at 0x000001C33051EAD0>, <Operation.EQ: '='>, '0-18'))]
```

The function returns

- **`msd_val`** – the maximum gap (in percentage‐points) between any subgroup and its complement
- **`rule`** – the raw subgroup encoding as a list of `(feature_index, Bin)` pairs.
  To get a human‐readable description, do the following:

  ```python
  pretty = " AND ".join(str(cond) for _, cond in rule)
  print("Subgroup:", pretty)
  # Subgroup: Race = Blue AND Age = 0-18
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

api/detect
```

<!-- TODO: I think it would be better to change on the way, as above -->

- [**Tutorial**](https://github.com/humancompatible/detect/blob/main/README.md) -> Your first audit in 5 minutes
- [**Examples**](https://github.com/humancompatible/detect/blob/main/examples/) -> Start with a simple [example notebook](https://github.com/humancompatible/detect/blob/main/examples/01_usage.ipynb), or go directly to a [realistic example using Folktables](https://github.com/humancompatible/detect/blob/main/examples/02_folktables.ipynb)

---

## MSD as a distance?

Bias detection can be understood as measuring some distance between two distributions (positive X negative samples, some training dataset X general population data...).

However, most distances have exponential sample complexity, whereas MSD requires a linear number of samples (w.r.t. the dimension) to achieve the same error.

|        Classical metric | Needs full d‐dim joint? | Sample cost | Drawbacks                             |
| ----------------------: | :---------------------: | :---------: | ------------------------------------- |
| Wasserstein, TV, MMD, … |           yes           |   Ω(2^d)    | exponential samples, no subgroup info |
|          **MSD (ours)** |  only protected attrs   |    O(d)     | ✓ returns exact subgroup & gap        |

MSD maximises the absolute difference in probability over all protected‐attribute combinations (subgroups), yet is solvable in practice through an exact Mixed‐Integer optimization that scans the doubly‐exponential space effectively.

---

## Citation

If you use MSD, please cite:

```bibtex
@inproceedings{MSD,
  author = {N\v{e}me\v{c}ek, Ji\v{r}\'{\i} and Kozdoba, Mark and Kryvoviaz, Illia and Pevn\'{y}, Tom\'{a}\v{s} and Mare\v{c}ek, Jakub},
  title = {Bias Detection via Maximum Subgroup Discrepancy},
  year = {2025},
  isbn = {9798400714542},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3711896.3736857},
  doi = {10.1145/3711896.3736857},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages = {2174–2185},
  numpages = {12},
  location = {Toronto ON, Canada},
  series = {KDD '25}
}
```

Looking for the installation matrix, solver details or developer setup?
Head to the [**README -> Installation**](https://github.com/humancompatible/detect?tab=readme-ov-file#installation-details) section.
