"""Build the 81 composition features from unique_m.csv.

Two choices, both set on the command line:

  --dataset   published  the author's unique_m.csv as released (reproduces the paper)
              cleaned    after the extra cleaning in supercon.cleaning (see notebook 03)
  --elements  pymatgen     open-source element table (default; FusionHeat -> MeltingPoint)
              mathematica  the author's Mathematica table (reproduces the paper's train.csv exactly)

Output: data/processed/features_<dataset>_<elements>.csv with the 81 features, critical_temp,
material and composition (scale-independent key for grouped train/test splits). The cleaned dataset
also writes data/processed/cleaning_log.csv, one row per correction or removal.

Examples:
  uv run python scripts/build_features.py                                   # published, pymatgen
  uv run python scripts/build_features.py --elements mathematica           # the paper's train.csv
  uv run python scripts/build_features.py --dataset cleaned --elements pymatgen
"""

import argparse
from pathlib import Path

import pandas as pd

from supercon import composition_key, featurize
from supercon.cleaning import clean
from supercon.elements import ELEMENTS, SOURCES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def build(dataset: str, elements: str) -> Path:
    unique_m = pd.read_csv(RAW / "unique_m.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    if dataset == "cleaned":
        data, log = clean(unique_m)
        log.to_csv(OUT / "cleaning_log.csv", index=False)
        print(f"cleaning: {len(unique_m)} -> {len(data)} rows ({log['step'].value_counts().to_dict()})")
    else:
        data = unique_m.assign(composition=composition_key(unique_m[ELEMENTS]))

    features = featurize(data[ELEMENTS], source=elements)
    features["critical_temp"] = data["critical_temp"]
    features["material"] = data["material"]
    features["composition"] = data["composition"]

    out = OUT / f"features_{dataset}_{elements}.csv"
    features.to_csv(out, index=False)
    print(f"wrote {features.shape[0]} rows x {features.shape[1]} columns to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["published", "cleaned"], default="published")
    parser.add_argument("--elements", choices=SOURCES, default="pymatgen")
    args = parser.parse_args()
    build(args.dataset, args.elements)
