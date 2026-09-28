"""Python port of Hamidieh's (2018) 81 composition features.

Mirrors ``get_features`` / ``extract`` in ``main_script_production_9.R``,
including its edge-case behaviour:

* an element whose property is missing (NaN) is dropped for that property,
  but the proportions p are *not* renormalized;
* gmean and entropy use |t|; log(0) propagates as in R (gmean -> 0, entropy -> NaN);
* std is the population standard deviation;
* wtd_range = max(p*t) - min(p*t)  (not p1*t1 - p2*t2 for the extreme-t elements).
"""

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from supercon.elements import ELEMENTS, PROPERTIES, load_element_table

STATS = [
    "mean", "wtd_mean", "gmean", "wtd_gmean", "entropy",
    "wtd_entropy", "range", "wtd_range", "std", "wtd_std",
]

FEATURE_NAMES = ["number_of_elements"] + [
    f"{stat}_{suffix}" for suffix in PROPERTIES.values() for stat in STATS
]


def _property_features(t: np.ndarray, p: np.ndarray, present: np.ndarray) -> np.ndarray:
    """The 10 statistics of one property for every material.

    t: (n_elements,) property values; p: (n_materials, n_elements) proportions;
    present: boolean mask of elements in each material.
    """
    mask = present & ~np.isnan(t)
    n = mask.sum(axis=1)
    t = np.broadcast_to(t, p.shape)

    def msum(x):
        return np.where(mask, x, 0.0).sum(axis=1)

    def mmax(x):
        return np.max(x, axis=1, where=mask, initial=-np.inf)

    def mmin(x):
        return np.min(x, axis=1, where=mask, initial=np.inf)

    def entropy(x):
        w = x / msum(x)[:, None]
        return -msum(w * np.log(w))

    abs_t = np.abs(t)
    log_abs_t = np.log(abs_t)
    pt = p * t

    mean = msum(t) / n
    wtd_mean = msum(pt)
    return np.column_stack([
        mean,
        wtd_mean,
        np.exp(msum(log_abs_t) / n),
        np.exp(msum(p * log_abs_t)),
        entropy(abs_t),
        entropy(p * abs_t),
        mmax(t) - mmin(t),
        mmax(pt) - mmin(pt),
        np.sqrt(msum((t - mean[:, None]) ** 2) / n),
        np.sqrt(msum(p * (t - wtd_mean[:, None]) ** 2)),
    ])


def featurize(counts: pd.DataFrame) -> pd.DataFrame:
    """Compute the 81 features from a table of element counts.

    counts: one row per material, columns are element symbols (any subset of
    ELEMENTS; missing columns are treated as zero), values are the formula
    coefficients, e.g. the element columns of ``unique_m.csv``. NaN counts are
    treated as zero (pandas fills absent keys with NaN when built from dicts).
    """
    unknown = set(counts.columns) - set(ELEMENTS)
    if unknown:
        raise ValueError(f"Unsupported elements (only Z <= 86): {sorted(unknown)}")

    c = counts.reindex(columns=ELEMENTS).fillna(0).to_numpy(dtype=float)
    present = c > 0
    p = c / c.sum(axis=1, keepdims=True)
    table = load_element_table()

    blocks = [present.sum(axis=1)[:, None]]
    with np.errstate(divide="ignore", invalid="ignore"):
        for prop in PROPERTIES:
            blocks.append(_property_features(table[prop].to_numpy(dtype=float), p, present))

    return pd.DataFrame(np.hstack(blocks), columns=FEATURE_NAMES, index=counts.index)


def counts_from_formulas(formulas) -> pd.DataFrame:
    """Parse chemical formula strings into an element-count table (via pymatgen)."""
    rows = [Composition(f).get_el_amt_dict() for f in formulas]
    return pd.DataFrame(rows).fillna(0.0)


def composition_key(counts: pd.DataFrame) -> pd.Series:
    """Scale-independent label of each composition, e.g. "Si0.25V0.75" for both Si1V3 and Si0.25V0.75.

    Elements are in atomic-number order with proportions rounded to 10 decimals. Materials
    with the same key have identical features, so the key is the natural group for train/test splits.
    """
    c = counts.reindex(columns=ELEMENTS).fillna(0).to_numpy(dtype=float)
    p = np.round(c / c.sum(axis=1, keepdims=True), 10)
    symbols = np.array(ELEMENTS)
    keys = ["".join(f"{e}{v:.10g}" for e, v in zip(symbols[row > 0], row[row > 0])) for row in p]
    return pd.Series(keys, index=counts.index, name="composition")
