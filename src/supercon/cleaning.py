"""Extra cleaning of unique_m.csv, beyond the steps described in the paper.

Used to build a second, cleaned dataset; the paper is reproduced on the data as published.
Three steps, each recorded in the log returned by ``clean``:

1. **Formula corrections and removals** listed one by one, with a reason, in
   ``data/cleaning_decisions.csv`` (misplaced decimals, capitalization typos, garbled formulas).
2. **Unknown oxygen content**: cuprates whose formula ends in a bare "O" (e.g. "Y1Ba2Cu3O")
   are removed; the bare O is a placeholder, so the parser counts a single oxygen atom.
   Non-cuprates with a bare "O" must each have an explicit decision (some, like SmFeAsO, are correct).
3. **Duplicates across scales**: rows with the same composition key (``Si1V3`` = ``Si0.25V0.75``)
   *and* the same Tc are dropped, keeping the uncorrected record when a corrected one duplicates it.
"""

import re
from importlib.resources import files

import pandas as pd

from supercon.elements import ELEMENTS
from supercon.features import composition_key, counts_from_formulas

BARE_OXYGEN = re.compile(r"O(?![a-z0-9.])")


def load_decisions() -> pd.DataFrame:
    """The per-formula decisions (action: fix / remove / keep), indexed by the original formula."""
    path = files("supercon") / "data" / "cleaning_decisions.csv"
    return pd.read_csv(path, keep_default_na=False).set_index("material")


def has_bare_oxygen(materials: pd.Series) -> pd.Series:
    """True where the formula writes oxygen without a coefficient, e.g. "Y1Ba2Cu3O"."""
    return materials.str.contains(BARE_OXYGEN)


def clean(unique_m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned, log).

    cleaned: the rows of ``unique_m`` that are kept (original index), corrected where needed, plus a
    ``composition`` column. log: one row per action taken (row, material, step, new_material, reason).
    """
    decisions = load_decisions()
    df = unique_m.copy()
    material = unique_m["material"]
    action = material.map(decisions["action"])
    log = []

    # 1. listed corrections: replace the formula and recompute its element counts
    fix = action.eq("fix")
    corrected = material[fix].map(decisions["corrected"])
    new_counts = counts_from_formulas(corrected).reindex(columns=ELEMENTS, fill_value=0.0)
    df.loc[fix, ELEMENTS] = new_counts.to_numpy()
    df.loc[fix, "material"] = corrected
    for row in material.index[fix]:
        log.append((row, material[row], "fix", corrected[row], decisions.at[material[row], "reason"]))

    # 2. removals: listed ones, plus every cuprate with a bare O
    bare_o = has_bare_oxygen(material)
    undecided = bare_o & (unique_m["Cu"] == 0) & action.isna()
    if undecided.any():
        raise ValueError(f"Non-cuprates with a bare O need a decision: {sorted(set(material[undecided]))}")

    listed = action.eq("remove")
    rule = bare_o & (unique_m["Cu"] > 0) & action.isna()
    for row in material.index[listed]:
        log.append((row, material[row], "remove", "", decisions.at[material[row], "reason"]))
    for row in material.index[rule]:
        log.append((row, material[row], "remove", "", "oxygen content unknown (cuprate written with a bare O)"))
    df = df[~(listed | rule)]

    # 3. duplicates of (composition, Tc), preferring records that were not corrected
    df["composition"] = composition_key(df[ELEMENTS])
    order = df.assign(_fixed=fix[df.index]).sort_values("_fixed", kind="stable")
    duplicate = order.duplicated(["composition", "critical_temp"]).reindex(df.index)
    for row in df.index[duplicate]:
        log.append((row, df.at[row, "material"], "duplicate", "",
                    f"same composition ({df.at[row, 'composition']}) and Tc as another row"))
    df = df[~duplicate]

    log = pd.DataFrame(log, columns=["row", "material", "step", "new_material", "reason"])
    return df, log.sort_values(["row", "step"], ignore_index=True)
