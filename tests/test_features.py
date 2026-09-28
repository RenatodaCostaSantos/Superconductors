"""Checks that the Python port reproduces Hamidieh's (2018) features.

These use the author's element table (source="mathematica"); the pymatgen table is checked in
test_elements.py.

Reference data (UCI "Superconductivty Data", id 464) must be in data/raw/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supercon import counts_from_formulas, feature_names, featurize
from supercon.elements import ELEMENTS, load_element_table

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FEATURE_NAMES = feature_names("mathematica")


@pytest.fixture(scope="module")
def unique_m():
    return pd.read_csv(RAW / "unique_m.csv")


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(RAW / "train.csv")


def test_element_table():
    table = load_element_table("mathematica")
    assert table.shape == (86, 8)
    assert table.loc["La", "AtomicRadius"] == 195
    assert table.loc["Ce", "AtomicRadius"] == 185
    assert table.loc["He", "ElectronAffinity"] == 1.5  # 0 + 1.5 shift
    missing = table.isna().stack()
    assert sorted(missing[missing].index) == [("At", "Density"), ("Po", "ThermalConductivity")]


def test_paper_table2_example():
    """Thermal-conductivity features of Table 2 of the paper (2 decimals).

    The paper labels the example "Re7Zr1", but its numbers (p1 = 6/7, Tc = 6.7 K)
    are those of Re6Zr1, row 21240 of unique_m.csv. There is no Re7Zr1 in the data.
    """
    x = featurize(pd.DataFrame([{"Re": 6, "Zr": 1}]), source="mathematica").iloc[0]
    expected = {
        "mean": 35.5, "wtd_mean": 44.43, "gmean": 33.23, "wtd_gmean": 43.21,
        "entropy": 0.63, "wtd_entropy": 0.26, "range": 25, "wtd_range": 37.86,
        "std": 12.5, "wtd_std": 8.75,
    }
    for stat, value in expected.items():
        assert x[f"{stat}_ThermalConductivity"] == pytest.approx(value, abs=0.005)
    assert x["number_of_elements"] == 2


def test_missing_property_is_dropped_without_renormalizing():
    """R's get_features drops NaN elements but keeps the original proportions."""
    x = featurize(pd.DataFrame([{"Po": 1, "Cu": 3}]), source="mathematica").iloc[0]
    cu = load_element_table("mathematica").loc["Cu", "ThermalConductivity"]
    assert x["mean_ThermalConductivity"] == pytest.approx(cu)
    assert x["wtd_mean_ThermalConductivity"] == pytest.approx(0.75 * cu)
    assert x["range_ThermalConductivity"] == 0
    assert x["number_of_elements"] == 2


def test_rows_with_different_elements():
    """Building counts from dicts leaves NaN for absent elements; they must count as 0."""
    counts = pd.DataFrame([{"Re": 6, "Zr": 1}, {"Mg": 1, "B": 2}])
    x = featurize(counts)
    assert np.isfinite(x.to_numpy()).all()
    single = featurize(pd.DataFrame([{"Mg": 1, "B": 2}]))
    np.testing.assert_allclose(x.iloc[[1]].to_numpy(), single.to_numpy())


def test_rows_aligned(unique_m, train):
    assert len(unique_m) == len(train) == 21263
    np.testing.assert_array_equal(unique_m["critical_temp"], train["critical_temp"])
    assert list(train.columns[:81]) == FEATURE_NAMES


def test_reproduces_train_csv(unique_m, train):
    ours = featurize(unique_m[ELEMENTS], source="mathematica")
    ref = train[FEATURE_NAMES]
    assert np.isfinite(ours.to_numpy()).all()
    np.testing.assert_allclose(ours.to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-12)


def test_formula_parsing_matches_unique_m(unique_m):
    parsed = counts_from_formulas(unique_m["material"]).reindex(columns=ELEMENTS, fill_value=0.0)
    np.testing.assert_allclose(parsed.to_numpy(), unique_m[ELEMENTS].to_numpy(), rtol=1e-12)
