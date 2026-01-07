# humancompatible.detect

[![Docs](https://readthedocs.org/projects/humancompatible-detect/badge/?version=latest)](https://humancompatible-detect.readthedocs.io/en/latest)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Pypi](https://img.shields.io/pypi/v/humancompatible-detect)](https://pypi.org/project/humancompatible-detect/)

humancompatible.detect is an open-source toolkit for detecting bias in AI models and their training data.

## AI Fairness

In fairness auditing, one would generally like to know if two distributions are identical.
These distributions could be a distribution of internal private training data and publicly accessible data from a nationwide census, i.e., a good baseline.
Or one can compare samples classified positively and negatively, to see if groups are represented equally in each class.

In other words, we ask

> Is there _some_ combination of protected attributes (race × age × …) for which people are treated noticeably differently?

A set of samples belonging to a given combination of protected attributes is called a subgroup.

<!-- Formally, let

* **X** ∈ ℝ<sup>d</sup> be the feature space,
* **P** and **Q** two distributions we want to compare (e.g. training vs census, positives vs negatives),
* **𝒫** ⊂ {1,…,d} the indices of *protected* features (age, sex, race, …).

A **sub-group** *S* is all samples whose protected attributes take one fixed value each.
We must consider every such intersection -- their number is exponential in |𝒫|.
 -->

## Using HumanCompatible.Detect

1. Install the library (in a virtual environment if desired):
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

- [`examples/01_basic_usage.ipynb`](https://github.com/humancompatible/detect/blob/main/examples/01_basic_usage.ipynb) -- a 5-minute notebook reproducing the call above, then translating `rule_idx` back to human-readable conditions.
- [`examples/02_folktables_within-state.ipynb`](https://github.com/humancompatible/detect/blob/main/examples/02_folktables_within-state.ipynb) -- a realistic Folktables/ACS Income example that runs MSD within a single state, reports the most affected subgroup, and interprets the signed gap.
- More notebooks live in [`examples/`](https://github.com/humancompatible/detect/tree/main/examples), new ones being added over time.

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

This method checks in a very efficient way whether the bias in any subgroup exceeds a given threshold. That is, it tells us to which extent, a particular subgroup obtains the positive outcome more or less frequently than the general trend in the dataset. Here, the fact that we can perform a subsampling with guaranties is key. It is the method of choice in cases in which one wants to be sure that a given dataset is compliant with a predefined acceptable bias level for all its subgroups.

---

## Installation details

### Requirements

All Python dependencies are declared in pyproject.toml (core + optional extras).

- **Python ≥ 3.10**

- **A MILP solver** (required for MSD).
  > We use [Pyomo](https://pyomo.readthedocs.io/) for modelling. This allows for multiple solvers, see the lists of [solver interfaces](https://pyomo.readthedocs.io/en/stable/reference/topical/solvers/index.html) and [persistent solver interfaces](https://pyomo.readthedocs.io/en/stable/reference/topical/appsi/appsi.html).

  - Default (recommended): HiGHS -- works out of the box because we install the HiGHS Python bindings (highspy) with the package.

  - Optional commercial solvers (license required): Gurobi / CPLEX / Xpress
  These require a valid installation + license from the vendor. (Some also have free community license, and pip-installable Python APIs.)

  - Optional open-source fallback: GLPK requires the glpsol executable on your system PATH.

- Other dependencies (installed automatically): numpy, pandas, scipy, pyomo, tqdm, etc.

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

### Optional extras

To install with optional commercial solvers:

```bash
python -m pip install "humancompatible-detect[gurobi]"
python -m pip install "humancompatible-detect[cplex]"
python -m pip install "humancompatible-detect[xpress]"
```

Or if you want the notebooks + plotting dependencies:

```bash
python -m pip install "humancompatible-detect[examples]"
```

And if docs/dev dependencies are desired:

```bash
python -m pip install "humancompatible-detect[docs]"
python -m pip install "humancompatible-detect[dev]"
```

### Verify it worked

```bash
python -c "from humancompatible.detect import detect_and_score; print('detect imported OK')"
```

If the import fails you'll see:

```bash
ModuleNotFoundError: No module named 'humancompatible'
```

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
