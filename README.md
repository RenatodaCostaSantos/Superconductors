# Superconductors

Python reproduction of K. Hamidieh, *A data-driven statistical model for predicting the
critical temperature of a superconductor*, Computational Materials Science 154, 346–354 (2018)
([doi:10.1016/j.commatsci.2018.07.052](https://doi.org/10.1016/j.commatsci.2018.07.052),
[arXiv:1803.10260](https://arxiv.org/abs/1803.10260)), as a basis for feature selection
for quantum ML models.

> This is an independent reproduction. It is not affiliated with or endorsed by the original author.
> All credit for the method, the data cleaning and the original features goes to Kam Hamidieh
> (see [Credits](#credits-and-data-sources)).

## Setup

```bash
uv sync
# UCI "Superconductivty Data" (train.csv, unique_m.csv)
mkdir -p data/raw && curl -L -o data/raw/sc.zip \
  "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip" && unzip -o data/raw/sc.zip -d data/raw
# Author's R code (optional, for reference)
git clone https://github.com/khamidieh/predict_tc reference/predict_tc
```

## Building the features

One script builds every variant; the person preparing the data chooses the dataset and the element
property table:

```bash
uv run python scripts/build_features.py --dataset {published,cleaned} --elements {pymatgen,mathematica}
```

It writes `data/processed/features_<dataset>_<elements>.csv` (81 features, `critical_temp`,
`material`, `composition`). `--dataset published --elements mathematica` reproduces the paper's
`train.csv`. In code: `featurize(counts, source="pymatgen" | "mathematica")`.

## Layout

- `src/supercon/elements.py` – the 86-element property tables: `"pymatgen"` (open source, default) or
  `"mathematica"` (the author's table, `subset_element_data` extracted from `tc.RData`).
- `src/supercon/features.py` – port of the R `get_features` / `extract` functions (81 features).
- `src/supercon/cleaning.py` – extra cleaning beyond the paper; per-formula decisions in
  `src/supercon/data/cleaning_decisions.csv`.
- `scripts/build_features.py` – builds the feature files (see above); the cleaned dataset also writes
  `cleaning_log.csv`.
- `tests/test_features.py` – checks the port reproduces the author's `train.csv` (`uv run pytest`).
- `tests/test_elements.py` – checks of the two element tables (the pymatgen one against Mathematica).
- `tests/test_cleaning.py` – checks of the extra cleaning.
- `tests/test_data_preparation.py` – checks the published data reflects the preparation steps of
  Section 2 of the paper.
- `data/reference/` – element tables exported from `tc.RData`.
- `notebooks/01_feature_walkthrough.ipynb` – step-by-step explanation of the feature code.
- `notebooks/02_data_preparation_checks.ipynb` – step-by-step check of Section 2 of the paper, and
  data issues the original clean-up did not catch.
- `notebooks/03_extra_cleaning.ipynb` – what the extra cleaning does and how it changes the data.
- `notebooks/04_element_sources.ipynb` – how the pymatgen element table is built and how it compares
  with Mathematica's.

## Two datasets, two element tables

| `--dataset` | Source | Rows |
|---|---|---|
| `published` | `unique_m.csv` as released | 21,263 |
| `cleaned` | `unique_m.csv` after `supercon.cleaning.clean` | 19,264 |

| `--elements` | Source | Notes |
|---|---|---|
| `pymatgen` (default) | pymatgen's open element data | same units and conventions as the paper; **FusionHeat is replaced by MeltingPoint** (no open source has heat of fusion for oxygen) and **Valence** is the largest absolute common oxidation state |
| `mathematica` | the author's table from Mathematica 11.1 `ElementData` | reproduces the paper's `train.csv` exactly |

The extra cleaning fixes 13 formula typos, removes 1,926 records whose oxygen content is unknown
(mostly cuprates written with a bare `O`, e.g. `Y1Ba2Cu3O`) and 73 duplicates that differ only in the
scale of the formula. Both files have a `composition` column (scale-independent key) for grouped
train/test splits. Because the cleaning removes many high-Tc cuprates, the two models should be
compared on the same test compositions.

## Credits and data sources

**Method and original code.** The feature definitions, the data cleaning and the models come from
Kam Hamidieh's paper (above) and the author's R code at
[github.com/khamidieh/predict_tc](https://github.com/khamidieh/predict_tc) (GPL-3.0).
`src/supercon/features.py` is a Python translation of the `get_features` and `extract` functions in
`main_script_production_9.R` from that repository.

**Superconductor data.** `unique_m.csv` and `train.csv` are the author's
[Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) on the UCI
Machine Learning Repository ([doi:10.24432/C53P47](https://doi.org/10.24432/C53P47), CC BY 4.0).
They are not redistributed here; the setup step downloads them. The author derived them from the
[SuperCon database](https://supercon.nims.go.jp/) of the National Institute for Materials Science
(NIMS), Japan, accessed July 24, 2017.

**Element properties.** `src/supercon/data/hamidieh_elements.csv` and `data/reference/*.csv` were
exported from `tc.RData` in the author's repository. According to the paper, the values come from
Wolfram Mathematica 11.1 `ElementData`, with first ionization energies from
[ptable.com](https://ptable.com/) and the La/Ce atomic radii from
[webelements.com](https://www.webelements.com/). The author imputed the La/Ce radii and added 1.5 to
every electron affinity; the values here are unchanged from the author's table.

The default element table is built from [pymatgen](https://pymatgen.org/)'s element data
(`pymatgen.core.Element`); see notebook 04 for the source of each property and the conversions.

**Software.** Formula parsing and the open-source element table use [pymatgen](https://pymatgen.org/)
(the original used the R package [CHNOSZ](https://www.chnosz.net/) and Mathematica).

### Citing

If you use this code, please cite the original paper and dataset:

```bibtex
@article{hamidieh2018datadriven,
  author  = {Hamidieh, Kam},
  title   = {A data-driven statistical model for predicting the critical temperature of a superconductor},
  journal = {Computational Materials Science},
  volume  = {154},
  pages   = {346--354},
  year    = {2018},
  doi     = {10.1016/j.commatsci.2018.07.052}
}

@misc{hamidieh2018data,
  author    = {Hamidieh, Kam},
  title     = {Superconductivty Data},
  year      = {2018},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C53P47}
}
```

## Notes on the reproduction

- The Python features match all 21,263 × 81 values of the author's `train.csv` to within ~1e-10.
- The paper's worked example (Section 2.3, Table 2) is labelled "Re7Zr1", but its numbers
  correspond to Re6Zr1 (Tc = 6.7 K), the material actually in the dataset.
- The author's Mathematica table gives hafnium a first ionization energy of 575 kJ/mol; the accepted
  value (NIST) is 658.5 kJ/mol, as in pymatgen.
- The paper says the La/Ce atomic radii were replaced by covalent radii (207/204 pm); the author's code
  and `train.csv` actually use 195/185 pm (webelements.com).
- A few formulas have misplaced decimal points the original clean-up missed (e.g. `Pr185Ce0.15Cu1O4`),
  and the same composition can appear at different scales (`Si1V3` / `Si0.25V0.75`). The data is used
  as published to reproduce the paper; see notebook 02.

## License

GPL-3.0 (see [LICENSE](LICENSE)), the same license as the original R code this project translates.
The UCI dataset is licensed separately under CC BY 4.0.
