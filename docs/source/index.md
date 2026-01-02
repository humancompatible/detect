# HumanCompatible · Bias Detection

A toolbox for measuring bias in data & models

<div align="center">

**Maximum Subgroup Discrepancy (MSD)** -- bias metric with linear sample complexity
_...with a MILP formulation that also tells you which subgroup is most affected._
\[[arxiv](https://dl.acm.org/doi/10.1145/3711896.3736857)\]

**ℓ∞** -- fast pass/fail bias test for a chosen subgroup
_...compares subgroup vs overall against a tolerance δ._
\[[arxiv](https://arxiv.org/abs/2502.02623)\]

</div>

---

## Quick install & demo

```bash
python -m pip install humancompatible-detect
```

```python
from humancompatible.detect import detect_and_score

rule, msd_val = detect_and_score(
    csv_path="./data/01_data.csv",
    target_col="Target",
    protected_list=["Race", "Age"],
    method="MSD",
)
```

The function returns

- **`msd_val`** -- the maximum gap (in percentage-points) between any subgroup and its complement
- **`rule`** -- the raw subgroup encoding as a list of `(feature_index, Bin)` pairs.

To get a human-readable description, do the following:

```python
pretty = " AND ".join(str(cond) for _, cond in rule)

print(f"MSD = {msd_val:.3f}")
print("Subgroup:", pretty)
```

## Contents

```{toctree}
:maxdepth: 1

self
api/humancompatible.detect
api/humancompatible.detect.methods.msd
api/humancompatible.detect.methods.l_inf
Tutorial <https://github.com/humancompatible/detect/blob/main/README.md>
Examples <https://github.com/humancompatible/detect/blob/main/examples/>
```

## Featured examples

If you want to jump straight into notebooks:

- **Simple example notebook**: <https://github.com/humancompatible/detect/blob/main/examples/01_basic_usage.ipynb>
- **Folktables example (within-state)**: <https://github.com/humancompatible/detect/blob/main/examples/02_folktables_within-state.ipynb>
- **Folktables example (cross-state)**: <https://github.com/humancompatible/detect/blob/main/examples/03_folktables_cross-state.ipynb>
- and more in the [examples folder](https://github.com/humancompatible/detect/tree/main/examples)

---

## MSD as a distance?

Bias detection can be understood as measuring some distance between two distributions (positive X negative samples, some training dataset X general population data...).

However, most distances have exponential sample complexity, whereas MSD requires a linear number of samples (w.r.t. the dimension) to achieve the same error.

<table class="metrics">
<tr>
  <th>Classical metric</th>
  <th>Needs to look at</th>
  <th>Sample cost</th>
  <th>Drawback</th>
</tr>
<tr>
  <td>Wasserstein, TV, MMD, ...</td>
  <td>full d-dimensional joint</td>
  <td>Ω(2<sup>d</sup>)</td>
  <td>exponential sample cost, no group explanation</td>
</tr>
<tr>
  <td>MSD (ours)</td>
  <td>only protected attrs</td>
  <td>O(d)</td>
  <td>returns exact subgroup & gap</td>
</tr>
</table>

<style>
table.metrics {
  width: 100%;
  border-collapse: collapse;
  margin: 0.75rem 0 1.5rem;
}
table.metrics th, table.metrics td {
  border: 1px solid #ddd;
  padding: 8px 10px;
  text-align: center;
  vertical-align: middle;
}
table.metrics thead th {
  background: #f7f7f7;
  font-weight: 600;
}
</style>


MSD maximises the absolute difference in probability over all protected-attribute combinations (subgroups), yet is solvable in practice through an exact Mixed-Integer optimization that scans the doubly-exponential space effectively.

## Subsampled ℓ∞ norm

This method checks in a very efficient way whether the bias in any subgroup exceeds a given threshold. It is to be selected in the case in which one wants to be sure that a given dataset is compliant with a predefined acceptable bias level for all its subgroups.

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

If you used the ℓ∞ method, please cite:

```bibtex
@misc{matilla2025samplecomplexitybiasdetection,
      title={Sample Complexity of Bias Detection with Subsampled Point-to-Subspace Distances},
      author={M. Matilla, Germán and Mareček, Jakub},
      year={2025},
      eprint={2502.02623},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02623v1},
}
```

Looking for the installation matrix, solver details or developer setup?
Head to the [**README -> Installation**](https://github.com/humancompatible/detect?tab=readme-ov-file#installation-details) section.
