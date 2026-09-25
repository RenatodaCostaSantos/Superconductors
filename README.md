# Superconductors

Python reproduction of K. Hamidieh, *A data-driven statistical model for predicting the
critical temperature of a superconductor*, Comput. Mater. Sci. 154 (2018)
([arXiv:1803.10260](https://arxiv.org/abs/1803.10260)), as a basis for feature selection
for quantum ML models.

## Setup

```bash
uv sync
# UCI "Superconductivty Data" (train.csv, unique_m.csv)
mkdir -p data/raw && curl -L -o data/raw/sc.zip \
  "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip" && unzip -o data/raw/sc.zip -d data/raw
# Author's R code (optional, for reference)
git clone https://github.com/khamidieh/predict_tc reference/predict_tc
```

## Layout

- `src/supercon/elements.py` – the 86-element property table used by the paper
  (`subset_element_data`, extracted from the author's `tc.RData`).
- `src/supercon/features.py` – port of the R `get_features` / `extract` functions (81 features).
- `scripts/01_build_features.py` – builds `data/processed/features.csv` from `unique_m.csv`.
- `tests/test_features.py` – checks the port reproduces the author's `train.csv` (`uv run pytest`).
- `data/reference/` – element tables exported from `tc.RData`.
- `notebooks/01_feature_walkthrough.ipynb` – step-by-step explanation of the feature code.
