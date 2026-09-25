"""Elemental property table used by Hamidieh (2018).

The table is ``subset_element_data`` extracted from ``tc.RData`` in
https://github.com/khamidieh/predict_tc. It originates from Mathematica 11.1
``ElementData`` (first ionization energy from ptable.com) and already includes
the author's adjustments:

* AtomicRadius of La and Ce imputed as 195 and 185 pm (webelements.com);
* ElectronAffinity shifted by +1.5 kJ/mol to avoid log(0).

Po and At have missing Density / ThermalConductivity (NaN).
"""

from importlib.resources import files

import pandas as pd

# Order matters: it defines the column order of the 81 features.
PROPERTIES = {
    "AtomicMass": "atomic_mass",
    "FirstIonizationEnergy": "fie",
    "AtomicRadius": "atomic_radius",
    "Density": "Density",
    "ElectronAffinity": "ElectronAffinity",
    "FusionHeat": "FusionHeat",
    "ThermalConductivity": "ThermalConductivity",
    "Valence": "Valence",
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


def load_element_table() -> pd.DataFrame:
    """Return the 86 x 8 property table indexed by element symbol, in ELEMENTS order."""
    path = files("supercon") / "data" / "hamidieh_elements.csv"
    table = pd.read_csv(path).set_index("Element")
    return table.loc[ELEMENTS, list(PROPERTIES)]
