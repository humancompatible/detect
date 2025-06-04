# detect

## Problem statement

In a fairness (or data-drift) audit we rarely ask  
"*Are the two distributions identical?*".  

The practical question is

> **Is there *any* combination of protected attributes (race × age × …) for which people are treated noticeably differently?**

Formally, let  

* **X** ∈ ℝ<sup>d</sup> be the feature space,  
* **P** and **Q** two distributions we want to compare (e.g. training vs census, positives vs negatives),  
* **𝒫** ⊂ {1,…,d} the indices of *protected* features (age, sex, race, …).

A **sub-group** *S* is all samples whose protected attributes take one fixed value each.  
We must consider every such intersection – their number is exponential in |𝒫|.

---

## Maximum Subgroup Discrepancy (MSD)

We define the distance between **P** and **Q** as the *largest* protected-sub-group gap  

```math

\text{MSD}(P,Q;\,𝒫)=
\max_{S\;\text{sub-group on }𝒫}\;
\bigl|\;P(S)-Q(S)\;\bigr|.

```


* Two distributions are *fair* iff all sub-groups have similar mass.  
* The **arg max** immediately tells you *which* group is most disadvantaged – an interpretable logical rule.

---

## Why classical distances fail

| Distance | Needs to look at | Worst-case samples | Drawback |
|----------|-----------------|--------------------|----------|
| Wasserstein, Total Variation, MMD, … | full *d*-dimensional joint | Ω(2<sup>d</sup>) | exponential sample cost, no group explanation |
| **MSD (ours)** | only the protected marginal | **O(d)** | exact group, human-readable |

MSD’s linear sample complexity is proven in the paper and achieved in practice via an **exact Mixed-Integer Optimisation** that scans the doubly-exponential search space implicitly, returning **both** the metric value and the rule that realises it.

---

## Quick-start

<div align="center">
  <img src="images/motivation_MSD.png" alt="Motivating example" width="550"/>
</div>
<br>


```python
from humancompatible.detect.MSD import compute_MSD

# tiny toy set 
# (col 1 = Race, col 2 = Age-bin, col 3 = binary target) 
msd, rule_idx = compute_MSD(
    csv_path = csv,
    target = "Target",
    protected_list = ["Race", "Age"],
)
```

---

## More to explore
- `examples/01_usage.ipynb` – a 5-minute notebook reproducing the call above,
then translating `rule_idx` back to human-readable conditions.

Feel free to start with the light notebook, then dive into the experiments with different datasets.

---

## Installation

### Requirements
- **Python ≥ 3.10**  

- **A MILP solver** (to solve the mixed-integer program)  
  * Gurobi 10.x is auto-detected if present - free academic licences are available.


### (Optional) create a fresh environment
```bash
python -m venv .venv
# ── Activate it ─────────────────────────────────────────────
# Linux / macOS
source .venv/bin/activate
# Windows – cmd.exe
.venv\Scripts\activate.bat
# Windows – PowerShell
.venv\Scripts\Activate.ps1
```

### Install the package

> **Package not on PyPI yet?**  
> Until we complete the PyPI release you can install the latest snapshot
> straight from GitHub in one line:

```bash
python -m pip install git+https://github.com/humancompatible/detect.git
```

If you prefer an editable (developer) install:

```bash
git clone https://github.com/humancompatible/detect.git
cd detect
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Verify it worked
```bash
python -c "from humancompatible.detect.MSD import compute_MSD; print('MSD imported OK')"
```

If the import fails you’ll see: <br>
`ModuleNotFoundError: No module named 'humancompatible'`.

---

## Documentation

To generate the documentation, install sphinx and run:

```bash
pip install -r docs/requirements.txt
sphinx-apidoc -o docs/source/ humancompatible/detect  -f -e
sphinx-build -M html docs/source/ docs/build/
```

---

## References

A reference implementation of the **Maximum Subgroup Discrepancy (MSD)** metric and the mixed-integer-optimization (MIO) solver that powers it, as introduced in:

> _Bias Detection via Maximum Subgroup Discrepancy_
> Jiří Němeček, Mark Kozdoba, Illia Kryvoviaz, Tomáš Pevný, Jakub Mareček
> ACM KDD 2025
