"""Build the 81 features from unique_m.csv after the extra cleaning in supercon.cleaning.

Outputs:
  data/processed/features_cleaned.csv  the 81 features, critical_temp, material and composition
  data/processed/cleaning_log.csv      one row per correction / removal, with the reason
"""

from pathlib import Path

import pandas as pd

from supercon import featurize
from supercon.cleaning import clean
from supercon.elements import ELEMENTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

unique_m = pd.read_csv(ROOT / "data" / "raw" / "unique_m.csv")
cleaned, log = clean(unique_m)

features = featurize(cleaned[ELEMENTS])
features["critical_temp"] = cleaned["critical_temp"]
features["material"] = cleaned["material"]
features["composition"] = cleaned["composition"]

OUT.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT / "features_cleaned.csv", index=False)
log.to_csv(OUT / "cleaning_log.csv", index=False)

print(f"{len(unique_m)} rows -> {len(cleaned)} rows")
print(log["step"].value_counts().to_string())
print(f"Wrote {OUT / 'features_cleaned.csv'} and {OUT / 'cleaning_log.csv'}")
