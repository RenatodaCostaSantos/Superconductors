"""Checks that the published data reflects the preparation steps of Section 2 of the paper.

Section 2.2 (SuperCon clean-up) was done by the author before unique_m.csv was written;
these tests verify its outcome, they do not re-run it. Steps that leave no trace in the
published files (dropping columns, sorting, fixing shifted Tc values, intermediate row
counts) cannot be checked. See notebooks/02_data_preparation_checks.ipynb.

Needs the author's element tables: uv run python scripts/fetch_mathematica_table.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supercon.elements import ELEMENTS, load_element_table, load_mathematica_full_table, properties

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def unique_m():
    return pd.read_csv(RAW / "unique_m.csv")


@pytest.fixture(scope="module")
def raw_elements():
    """Element table before the author's adjustments (element_data in tc.RData)."""
    return load_mathematica_full_table()


# ---- 2.1 Element data -------------------------------------------------------------


def test_86_elements_up_to_radon(raw_elements, unique_m):
    assert raw_elements["AtomicNumber"].tolist() == list(range(1, 87))
    assert [c for c in unique_m.columns if c in ELEMENTS] == ELEMENTS


def test_electron_affinity_shifted_by_1_5(raw_elements):
    shift = load_element_table("mathematica")["ElectronAffinity"] - raw_elements["ElectronAffinity"]
    np.testing.assert_allclose(shift, 1.5)


def test_la_ce_radius_imputation(raw_elements):
    """The paper says covalent radii were used; the code and train.csv use 195/185 pm."""
    assert raw_elements.loc[["La", "Ce"], "AtomicRadius"].isna().all()
    assert raw_elements.loc[["La", "Ce"], "CovalentRadius"].tolist() == [207, 204]
    assert load_element_table("mathematica").loc[["La", "Ce"], "AtomicRadius"].tolist() == [195, 185]


def test_atomic_and_covalent_radius_correlated(raw_elements):
    r = raw_elements[["AtomicRadius", "CovalentRadius"]].dropna()
    assert r.corr().iloc[0, 1] == pytest.approx(0.95, abs=0.005)


def test_other_properties_unchanged(raw_elements):
    used = load_element_table("mathematica")
    for prop in properties("mathematica"):
        if prop == "ElectronAffinity":
            continue
        a, b = used[prop], raw_elements[prop]
        if prop == "AtomicRadius":
            a, b = a.drop(["La", "Ce"]), b.drop(["La", "Ce"])
        np.testing.assert_array_equal(a, b)


# ---- 2.2 SuperCon clean-up ----------------------------------------------------------


def test_step4_implausible_tc_removed(unique_m):
    materials = set(unique_m["material"])
    assert "La0.23Th0.77Pb3" not in materials
    assert "Pb2C1Ag2O6" not in materials
    assert unique_m["critical_temp"].max() <= 203
    er = unique_m[unique_m["material"].str.startswith("Er1Ba2Cu3O7")]
    assert er["critical_temp"].max() < 203


def test_step5_tc_positive_and_present(unique_m):
    assert unique_m["critical_temp"].notna().all()
    assert (unique_m["critical_temp"] > 0).all()


def test_step7_8_formula_strings_clean(unique_m):
    """No O7-X / O5+X notation, no '!', only valid element symbols, no zero coefficients."""
    m = unique_m["material"]
    assert not m.str.contains(r"[^A-Za-z0-9.]").any()
    symbols = set(re.findall(r"[A-Z][a-z]?", " ".join(m)))
    assert symbols <= set(ELEMENTS)  # e.g. no "Yo", "X", "Z"
    counts = unique_m[ELEMENTS]
    assert (counts >= 0).all().all()
    # every element written in the formula has a non-zero count (catches "Sr0", "O0", ...);
    # some formulas repeat an element for different sites (Tl1Sr2Sr0.5Y0.5Cu2O7), so count distinct
    distinct = m.map(lambda f: len(set(re.findall(r"[A-Z][a-z]?", f))))
    assert (distinct == (counts > 0).sum(axis=1)).all()


def test_step8_specific_fixes(unique_m):
    materials = set(unique_m["material"])
    for removed in ["Y1Ba2Cu3O6050", "Hg1234O10", "Nd185Ce0.15Cu1O4",
                    "Bi1.6Pb0.4Sr2Cu3Ca2O1013", "Y1Ba2Cu285Ni0.15O7"]:
        assert removed not in materials
    for kept in ["Nd1.85Ce0.15Cu1O4", "Bi1.6Pb0.4Sr2Cu3Ca2O10.13", "Y1Ba2Cu2.85Ni0.15O7"]:
        assert kept in materials


def test_step9_column_names(unique_m):
    assert unique_m.columns[-2:].tolist() == ["critical_temp", "material"]


def test_duplicates_removed(unique_m):
    assert len(unique_m) == 21263
    assert not unique_m.duplicated(ELEMENTS + ["critical_temp"]).any()


# ---- Cross-check with Section 3.1 (Tables 3 and 4, Figure 3) --------------------------


def test_tc_summary_table4(unique_m):
    tc = unique_m["critical_temp"]
    assert tc.min() == 0.00021
    assert tc.quantile(0.25) == pytest.approx(5.4, abs=0.05)
    assert tc.median() == 20
    assert tc.quantile(0.75) == 63
    assert tc.max() == 185
    assert tc.mean() == pytest.approx(34.4, abs=0.05)
    assert tc.std() == pytest.approx(34.25, abs=0.01)  # paper rounds to 34.2


def test_iron_cuprate_counts_table3(unique_m):
    assert (unique_m["Fe"] > 0).sum() == 2339
    assert ((unique_m["Cu"] > 0) & (unique_m["O"] > 0)).sum() == 10532
    assert (unique_m["O"] > 0).mean() == pytest.approx(0.56, abs=0.005)
