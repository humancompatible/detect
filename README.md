# humancompatible.detect

[![Docs](https://readthedocs.org/projects/humancompatible-detect/badge/?version=latest)](https://humancompatible-detect.readthedocs.io/en/latest)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Pypi](https://img.shields.io/pypi/v/humancompatible-detect)](https://pypi.org/project/humancompatible-detect/)

humancompatible.detect is an open-source toolkit for detecting bias in AI models and their training data.

## AI Fairness

In a fairness auditing, one would generally like to know if two distributions are identical.
These distributions could be a distribution of internal private training data and publicly accessible data from a nationwide census, i.e., a good baseline.
Or one can compare samples classified positively and negatively, to see if groups are represented equally in each class.

In other words, we ask

> Is there _some_ combination of protected attributes (race × age × …) for which people are treated noticeably differently?

Samples belonging to a given combination of protected attributes are called a subgroup.

<!-- Formally, let

* **X** ∈ ℝ<sup>d</sup> be the feature space,
* **P** and **Q** two distributions we want to compare (e.g. training vs census, positives vs negatives),
* **𝒫** ⊂ {1,…,d} the indices of *protected* features (age, sex, race, …).

A **sub-group** *S* is all samples whose protected attributes take one fixed value each.
We must consider every such intersection -- their number is exponential in |𝒫|.
 -->

## Using HumanCompatible.Detect

1. Install the library (in virtual environment if desired):
   ```bash
   pip install humancompatible-detect
   ```
2. Compute the bias ([MSD](#maximum-subgroup-discrepancy-msd) in this case):

   ```python
   from humancompatible.detect import detect_and_score

   # toy example
   # (col 1 = Race, col 2 = Age, col 3 = (binary) target)
   rule_idx, msd = detect_and_score(
       csv_path = "./data/01_data.csv",
       target_col = "Target",
       protected_list = ["Race", "Age"],
       method = "MSD",
   )
   ```

### More to explore

- `examples/01_basic_usage.ipynb` -- a 5-minute notebook reproducing the call above, then translating `rule_idx` back to human-readable conditions.
- `examples/02_folktables_within-state.ipynb` -- a realistic Folktables/ACS Income example that runs MSD within a single state, reports the most affected subgroup, and interprets the signed gap.
- More notebooks live in [`examples/`](examples/), new ones being added over time.

Feel free to start with the light notebook, then dive into the experiments with different datasets.

We also provide [documentation](https://humancompatible-detect.readthedocs.io/en/latest). For more details on installation, see [Installation details](#installation-details).

---

## Methods

### Maximum Subgroup Discrepancy (MSD)

MSD is the subgroup maximal difference in probability mass of a given subgroup, comparing the mass given by each distribution.

<!-- ```math

\text{MSD}(P,Q;\,𝒫)=
\max_{S\;\text{sub-group on }𝒫}\;
\bigl|\;P(S)-Q(S)\;\bigr|.

``` -->

- Naturally, two distributions are _fair_ iff all subgroups have similar mass.
- The **arg max** immediately tells you _which_ group is most disadvantaged as an interpretable attribute-value combination.
- MSD has linear sample complexity, a stark contrast to exponential complexity of other distributional distances (Wasserstein, TV...)

### Subsampled ℓ∞ norm

This method checks in a very efficient way whether the bias in any subgroup exceeds a given threshold. It is to be selected in the case in which one wants to be sure that a given dataset is compliant with a predefined acceptable bias level for all its subgroups.

---

## Installation details

### Requirements

Requirements are included in the `requirements.txt` file. They include:

- **Python ≥ 3.10**

- **A MILP solver** (to solve the mixed-integer program in the case of MSD)
  - The default solver is [HiGHS](https://highs.dev/). This is an open-source solver included in the requirements.
  - A faster, but proprietary solver [Gurobi](https://www.gurobi.com/) can also easily be used. Free academic licences are available. This solver was used in the original paper.
  - We use [Pyomo](https://pyomo.readthedocs.io/) for modelling. This allows for multiple solvers, see the lists of [solver interfaces](https://pyomo.readthedocs.io/en/stable/reference/topical/solvers/index.html) and [persistent solver interfaces](https://pyomo.readthedocs.io/en/stable/reference/topical/appsi/appsi.html). Note that the implementation sets the graceful time limit only for solvers Gurobi, Cplex, HiGHS, Xpress, and GLPK.

### (Optional) create a fresh environment

```bash
python -m venv .venv
# Activate it
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate.bat    # Windows -- cmd.exe
.venv\Scripts\Activate.ps1    # Windows -- PowerShell
```

### Install the package

```bash
python -m pip install humancompatible-detect
```

Developer install (editable):

```bash
git clone https://github.com/humancompatible/detect.git
cd detect
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
```

### Verify it worked

```bash
python -c "from humancompatible.detect import detect_and_score; print('detect imported OK')"
```

If the import fails you'll see:

```bash
ModuleNotFoundError: No module named 'humancompatible'
```

## Why classical distances fail

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

MSD's linear sample complexity is proven in the paper and achieved in practice via an exact Mixed-Integer Optimization that scans the doubly-exponential search space implicitly, returning both the metric value and the rule that realises it.

---

## References

If you use the MSD in your work, please cite the following work:

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

If you liked the ℓ∞ method, please cite:

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
