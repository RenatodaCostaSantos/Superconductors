"""Build the 81 Hamidieh features from the cleaned compositions in unique_m.csv.

Output: data/processed/features.csv with the 81 features, critical_temp and material.
"""

from pathlib import Path

import pandas as pd

from supercon import featurize
from supercon.elements import ELEMENTS

ROOT = Path(__file__).resolve().parents[1]

unique_m = pd.read_csv(ROOT / "data" / "raw" / "unique_m.csv")
features = featurize(unique_m[ELEMENTS])
features["critical_temp"] = unique_m["critical_temp"]
features["material"] = unique_m["material"]

out = ROOT / "data" / "processed" / "features.csv"
out.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(out, index=False)
print(f"Wrote {features.shape[0]} rows x {features.shape[1]} columns to {out}")
