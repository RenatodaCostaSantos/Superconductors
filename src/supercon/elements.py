"""Elemental property tables for the 86 elements (Z = 1..86).

Two interchangeable sources, selected with ``source``:

* ``"pymatgen"`` (default) — built from pymatgen's open element data. Same 8-property layout as
  the paper, except that **FusionHeat is replaced by MeltingPoint** (pymatgen has no heat of fusion)
  and **Valence** is the largest absolute common oxidation state (pymatgen has no "valence").
* ``"mathematica"`` — the author's table: ``subset_element_data`` extracted from ``tc.RData`` in
  https://github.com/khamidieh/predict_tc, from Mathematica 11.1 ``ElementData`` (first ionization
  energy from ptable.com). Use it to reproduce the paper's ``train.csv`` exactly.

Both tables use the paper's units and conventions: AtomicMass (amu), FirstIonizationEnergy (kJ/mol),
AtomicRadius (pm), Density (kg/m^3, at standard temperature and pressure), ElectronAffinity
(kJ/mol, shifted by +1.5 to avoid log(0)), FusionHeat (kJ/mol) or MeltingPoint (K),
ThermalConductivity (W/(m K)), Valence (no units). Missing values are NaN.
"""

import warnings
from functools import cache
from importlib.resources import files

import pandas as pd
from pymatgen.core import Element

SOURCES = ("pymatgen", "mathematica")

# Property -> suffix used in the feature names. Order defines the column order of the features.
_MATHEMATICA_PROPERTIES = {
    "AtomicMass": "atomic_mass",
    "FirstIonizationEnergy": "fie",
    "AtomicRadius": "atomic_radius",
    "Density": "Density",
    "ElectronAffinity": "ElectronAffinity",
    "FusionHeat": "FusionHeat",
    "ThermalConductivity": "ThermalConductivity",
    "Valence": "Valence",
}
_PYMATGEN_PROPERTIES = {
    ("MeltingPoint" if k == "FusionHeat" else k): ("MeltingPoint" if k == "FusionHeat" else v)
    for k, v in _MATHEMATICA_PROPERTIES.items()
}

# The 86 elements (Z = 1..86) considered in the paper, in unique_m.csv order.
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

EV_TO_KJ_PER_MOL = 96.48533212
GAS_CONSTANT = 8.314462618  # J/(mol K)
STP_TEMPERATURE, STP_PRESSURE = 273.15, 101325.0  # K, Pa
DIATOMIC_GASES = {"H", "N", "O", "F", "Cl"}
ELECTRON_AFFINITY_SHIFT = 1.5  # kJ/mol, as in the paper


def properties(source: str = "pymatgen") -> dict[str, str]:
    """Property name -> feature-name suffix for the given source, in feature order."""
    _check(source)
    return dict(_PYMATGEN_PROPERTIES if source == "pymatgen" else _MATHEMATICA_PROPERTIES)


def load_element_table(source: str = "pymatgen") -> pd.DataFrame:
    """Return the 86 x 8 property table indexed by element symbol, in ELEMENTS order."""
    _check(source)
    table = _pymatgen_table() if source == "pymatgen" else _mathematica_table()
    return table.copy()


def _check(source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


@cache
def _mathematica_table() -> pd.DataFrame:
    """The author's table, with the La/Ce radius imputation and +1.5 EA shift already applied."""
    path = files("supercon") / "data" / "hamidieh_elements.csv"
    table = pd.read_csv(path).set_index("Element")
    return table.loc[ELEMENTS, list(_MATHEMATICA_PROPERTIES)].astype(float)


def _get(e: Element, attribute: str):
    """pymatgen attribute, or None when pymatgen has no data (the gaps are handled below)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No data available")
        return getattr(e, attribute)


def _value(x):
    return float("nan") if x is None else float(x)


def _density(e: Element) -> float:
    """Density at standard temperature and pressure (0 degC, 1 atm), in kg/m^3.

    Solids: pymatgen's density of solid. Gases at STP: ideal-gas density of the molecule
    (H2, N2, O2, F2, Cl2 are diatomic). Liquids at STP (Hg, Br): atomic mass / molar volume.
    """
    solid, boiling, molar_volume = (_get(e, a) for a in ("density_of_solid", "boiling_point", "molar_volume"))
    if solid is not None:
        return float(solid)
    if boiling is not None and boiling < STP_TEMPERATURE:
        molar_mass = float(e.atomic_mass) * (2 if e.symbol in DIATOMIC_GASES else 1) / 1000  # kg/mol
        return molar_mass * STP_PRESSURE / (GAS_CONSTANT * STP_TEMPERATURE)
    if molar_volume is not None:
        return float(e.atomic_mass) / float(molar_volume) * 1000  # g/cm^3 -> kg/m^3
    return float("nan")


def _radius(e: Element) -> float:
    """Calculated atomic radius (as in Mathematica), falling back to the empirical radius, in pm."""
    r = _get(e, "atomic_radius_calculated")
    return _value(r if r is not None else _get(e, "atomic_radius")) * 100  # angstrom -> pm


def _electron_affinity(e: Element) -> float:
    """In kJ/mol; elements without a stable anion (negative value in pymatgen) get 0, as in Mathematica."""
    return max(_value(_get(e, "electron_affinity")) * EV_TO_KJ_PER_MOL, 0.0) + ELECTRON_AFFINITY_SHIFT


@cache
def _pymatgen_table() -> pd.DataFrame:
    rows = {}
    for symbol in ELEMENTS:
        e = Element(symbol)
        rows[symbol] = {
            "AtomicMass": float(e.atomic_mass),
            "FirstIonizationEnergy": float(e.ionization_energies[0]) * EV_TO_KJ_PER_MOL,
            "AtomicRadius": _radius(e),
            "Density": _density(e),
            "ElectronAffinity": _electron_affinity(e),
            "MeltingPoint": _value(_get(e, "melting_point")),
            "ThermalConductivity": _value(_get(e, "thermal_conductivity")),
            "Valence": float(max((abs(x) for x in _get(e, "common_oxidation_states") or ()), default=0)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")[list(_PYMATGEN_PROPERTIES)]
