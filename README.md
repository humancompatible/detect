# detect
## Bias Detection via Maximum Subgroup Discrepancy

A reference implementation of the **Maximum Subgroup Discrepancy (MSD)** metric and the mixed-integer-optimization (MIO) solver that powers it, as introduced in:

> *Bias Detection via Maximum Subgroup Discrepancy*  
> Jiří Němeček, Mark Kozdoba, Illia Kryvoviaz, Tomáš Pevný, Jakub Mareček  
> ACM KDD 2025

---

## Maximum Subgroup Discrepancy (MSD)
Conventional two-sample distances (Wasserstein, Total Variation, MMD, …) have **exponential sample complexity** in high-dimensional or intersectional settings.  
**MSD** side-steps this by maximising the discrepancy **over all protected sub-groups** and enjoys **linear sample complexity** w.r.t. the number of protected attributes.  
It also returns an *interpretable* logical description (a DNF conjunction) of the most-biased subgroup.

---

## Repository Layout

    detect/
    ├── humancompatible/
    │ └── detect/
    │   ├── dnf_bias.py
    │   ├── .py and folders/ binarizer, data_handler, utils, ...
    │   ├── experiment_enumerative.py
    |   ├── experiment_sample_complexity.py
    │   └── plot-maker/
    |     ├── plot_exploration.py
    |     └── plots_for_paper.py
    |
    ├── data/
    |
    ├── examples/
    | ├── 00_experiments/
    |   ├── experiments.ipynb
    |   └── plots.ipynb
    │ └── 01_usage_dnf_bias.ipynb
    │ 
    ├── tests/
    ├── requirements.txt
    └── setup.py

---

## Installation

```bash
# 1. clone the repo
git clone https://github.com/humancompatible/detect.git
cd detect

# 2. create & activate a fresh env
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. editable install
pip install -e .
```

