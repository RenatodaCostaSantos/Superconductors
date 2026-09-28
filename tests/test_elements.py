"""Checks of the element property tables (supercon.elements), in particular the pymatgen one."""

import numpy as np
import pandas as pd
import pytest

from supercon import feature_names, featurize
from supercon.elements import ELEMENTS, SOURCES, load_element_table, properties


@pytest.fixture(scope="module")
def pmg():
    return load_element_table("pymatgen")


@pytest.fixture(scope="module")
def mma():
    return load_element_table("mathematica")


def test_default_source_is_pymatgen():
    pd.testing.assert_frame_equal(load_element_table(), load_element_table("pymatgen"))
    assert feature_names() == feature_names("pymatgen")


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        load_element_table("wikipedia")


def test_same_layout(pmg, mma):
    assert pmg.shape == mma.shape == (86, 8)
    assert list(pmg.index) == list(mma.index) == ELEMENTS
    assert list(pmg.columns) == list(properties("pymatgen"))
    # only FusionHeat differs: pymatgen has no heat of fusion
    assert [c for c in pmg.columns if c not in mma.columns] == ["MeltingPoint"]
    assert [c for c in mma.columns if c not in pmg.columns] == ["FusionHeat"]


def test_feature_names_per_source():
    for source in SOURCES:
        assert len(feature_names(source)) == 81
    diff = set(feature_names("pymatgen")) ^ set(feature_names("mathematica"))
    assert {n.rsplit("_", 1)[1] for n in diff} == {"MeltingPoint", "FusionHeat"}


def test_pymatgen_missing_values(pmg):
    missing = pmg.isna().stack()
    # astatine is barely characterised; everything else is filled
    assert {s for s, _ in missing[missing].index} == {"At"}
    assert sorted(p for _, p in missing[missing].index) == ["AtomicRadius", "Density"]


@pytest.mark.parametrize("prop, rtol, min_agree", [
    ("AtomicMass", 0.005, 85),
    ("FirstIonizationEnergy", 0.02, 80),
    ("AtomicRadius", 0.02, 84),
    ("Density", 0.02, 70),
    ("ThermalConductivity", 0.02, 85),
])
def test_close_to_mathematica(pmg, mma, prop, rtol, min_agree):
    """Same units and conventions as the paper: most elements agree within a few percent."""
    agree = np.isclose(pmg[prop], mma[prop], rtol=rtol)
    assert agree.sum() >= min_agree


def test_la_ce_radius_matches_author_imputation(pmg):
    assert pmg.loc[["La", "Ce"], "AtomicRadius"].tolist() == [195, 185]


def test_gas_density_at_stp(pmg, mma):
    """Gases get the ideal-gas density at 0 degC and 1 atm, as Mathematica does."""
    gases = ["H", "He", "N", "O", "F", "Ne", "Cl", "Ar", "Kr", "Xe"]
    np.testing.assert_allclose(pmg.loc[gases, "Density"], mma.loc[gases, "Density"], rtol=0.02)


def test_electron_affinity_convention(pmg, mma):
    """No stable anion -> 0, then +1.5 as in the paper; so the minimum is 1.5 in both tables."""
    assert pmg["ElectronAffinity"].min() == pytest.approx(1.5)
    assert pmg.loc["He", "ElectronAffinity"] == pytest.approx(1.5)
    assert pmg.loc["O", "ElectronAffinity"] == pytest.approx(mma.loc["O", "ElectronAffinity"], rel=0.02)


def test_valence_is_max_common_oxidation_state(pmg, mma):
    assert pmg.loc[["H", "O", "F", "La", "Zr", "Re", "He"], "Valence"].tolist() == [1, 2, 1, 3, 4, 7, 0]
    assert (pmg["Valence"] == mma["Valence"]).sum() >= 70


def test_melting_point_tracks_fusion_heat(pmg, mma):
    """Melting point stands in for heat of fusion; the two are strongly related (Richards' rule)."""
    both = pmg["MeltingPoint"].notna() & mma["FusionHeat"].notna()
    r = np.corrcoef(np.log(pmg.loc[both, "MeltingPoint"]), np.log(mma.loc[both, "FusionHeat"]))[0, 1]
    assert r > 0.85


def test_featurize_with_pymatgen_is_finite():
    x = featurize(pd.DataFrame([{"Y": 1, "Ba": 2, "Cu": 3, "O": 7}, {"Mg": 1, "B": 2}, {"Hg": 1, "Ba": 2, "Ca": 2, "Cu": 3, "O": 8}]))
    assert list(x.columns) == feature_names("pymatgen")
    assert np.isfinite(x.to_numpy()).all()
