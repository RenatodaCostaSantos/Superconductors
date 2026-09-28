"""Checks of the extra cleaning (supercon.cleaning), which goes beyond the paper."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supercon import composition_key, counts_from_formulas, featurize
from supercon.cleaning import clean, has_bare_oxygen, load_decisions
from supercon.elements import ELEMENTS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def unique_m():
    return pd.read_csv(RAW / "unique_m.csv")


@pytest.fixture(scope="module")
def cleaned(unique_m):
    return clean(unique_m)


def test_composition_key_ignores_scale():
    counts = pd.DataFrame([{"Si": 1, "V": 3}, {"Si": 0.25, "V": 0.75}, {"Si": 1, "V": 2}])
    key = composition_key(counts)
    assert key[0] == key[1] == "Si0.25V0.75"
    assert key[2] != key[0]


def test_bare_oxygen_detection():
    formulas = pd.Series(["Y1Ba2Cu3O", "Y1Ba2Cu3O7", "Sm1Fe1As1F0.5O", "Os1B2", "Y1Ba2Cu3HO7", "Bi1Cu1PO"])
    assert has_bare_oxygen(formulas).tolist() == [True, False, True, False, False, True]


def test_decisions_file_is_consistent(unique_m):
    decisions = load_decisions()
    assert decisions.index.is_unique
    assert set(decisions["action"]) <= {"fix", "remove", "keep"}
    assert set(decisions.index) <= set(unique_m["material"])
    assert (decisions["reason"] != "").all()
    fixes = decisions[decisions["action"] == "fix"]
    assert (fixes["corrected"] != "").all()
    assert (decisions.loc[decisions["action"] != "fix", "corrected"] == "").all()
    parsed = counts_from_formulas(fixes["corrected"])
    assert set(parsed.columns) <= set(ELEMENTS)


def test_every_non_cuprate_bare_oxygen_has_a_decision(unique_m):
    m = unique_m["material"]
    needs_decision = set(m[has_bare_oxygen(m) & (unique_m["Cu"] == 0)])
    assert needs_decision <= set(load_decisions().index)


def test_cleaned_has_no_unknown_oxygen(cleaned):
    df, _ = cleaned
    decisions = load_decisions()
    bare = has_bare_oxygen(df["material"])
    assert set(df.loc[bare, "material"]) <= set(decisions.index[decisions["action"] == "keep"])
    assert not df["material"].isin(decisions.index[decisions["action"] != "keep"]).any()


def test_counts_match_formulas(cleaned):
    df, _ = cleaned
    parsed = counts_from_formulas(df["material"]).reindex(columns=ELEMENTS, fill_value=0.0)
    np.testing.assert_allclose(parsed.to_numpy(), df[ELEMENTS].to_numpy(), rtol=1e-12)


def test_no_duplicates_across_scales(cleaned):
    df, _ = cleaned
    assert not df.duplicated(["composition", "critical_temp"]).any()
    np.testing.assert_array_equal(df["composition"], composition_key(df[ELEMENTS]))


def test_decimal_typos_collapse_onto_existing_records(cleaned, unique_m):
    """The corrected formulas duplicate records already in the data; the originals are kept."""
    df, log = cleaned
    for typo, tc in [("Pr185Ce0.15Cu1O4", 20.0), ("La1.85Sr0.15Cu98Ni0.02O4", 21.96), ("Rb3C60C60", 29.0)]:
        row = unique_m.index[unique_m["material"] == typo][0]
        assert set(log.loc[log["row"] == row, "step"]) == {"fix", "duplicate"}
        assert row not in df.index
    assert ((df["material"] == "Pr1.85Ce0.15Cu1O4") & (df["critical_temp"] == 20.0)).sum() == 1


def test_log_accounts_for_every_dropped_row(cleaned, unique_m):
    df, log = cleaned
    dropped = set(unique_m.index) - set(df.index)
    assert dropped == set(log.loc[log["step"].isin(["remove", "duplicate"]), "row"])
    assert df.index.isin(unique_m.index).all()


def test_cleaned_features_are_finite(cleaned):
    df, _ = cleaned
    assert np.isfinite(featurize(df[ELEMENTS]).to_numpy()).all()


def test_only_listed_formulas_change(cleaned, unique_m):
    df, log = cleaned
    changed = df.index[df["material"] != unique_m.loc[df.index, "material"]]
    assert set(changed) <= set(log.loc[log["step"] == "fix", "row"])
    assert all(re.fullmatch(r"[A-Za-z0-9.]+", f) for f in df["material"])
