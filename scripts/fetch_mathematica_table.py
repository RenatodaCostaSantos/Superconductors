"""Extract the author's Mathematica element tables from tc.RData (optional pipeline step).

Only needed for ``--elements mathematica`` (reproducing the paper exactly); the default element table
is built from pymatgen. The tables are not redistributed with this repository: this script takes
``tc.RData`` from the author's repository https://github.com/khamidieh/predict_tc (GPL-3.0), pinned to
a fixed commit and checked against its SHA-256, and writes to data/external/ (ignored by git):

  mathematica_elements.csv          subset_element_data: the 8 properties used by the paper, with the
                                    author's La/Ce radius imputation and +1.5 electron-affinity shift
  mathematica_element_data_full.csv element_data: all 35 ElementData properties, before adjustments

Usage:
  uv run python scripts/fetch_mathematica_table.py            # downloads tc.RData (~20 MB) if needed
  uv run python scripts/fetch_mathematica_table.py --rdata reference/predict_tc/tc.RData
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from supercon.elements import MATHEMATICA_DIR, MATHEMATICA_FULL_FILE, MATHEMATICA_FILE
from supercon.rdata_reader import read_rdata, to_dataframe

COMMIT = "ab093ab19a4f34ec7c7bd0342afe0f7d26f0f8b6"  # khamidieh/predict_tc, 2018-11-11
URL = f"https://raw.githubusercontent.com/khamidieh/predict_tc/{COMMIT}/tc.RData"
SHA256 = "934690cfa39572b350c28a32ac2c8330d3e2c437ca49b5432adb88449005f888"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_rdata(path: Path | None) -> Path:
    if path is None:
        path = MATHEMATICA_DIR / "tc.RData"
        if not path.exists():
            print(f"downloading {URL}")
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, path)
    if sha256(path) != SHA256:
        raise SystemExit(f"{path} does not match the expected tc.RData (SHA-256 {SHA256})")
    return path


def main(rdata: Path | None) -> None:
    objects = read_rdata(get_rdata(rdata))
    subset = to_dataframe(objects["subset_element_data"])
    full = to_dataframe(objects["element_data"])

    MATHEMATICA_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(MATHEMATICA_DIR / MATHEMATICA_FILE, index=False)
    full.to_csv(MATHEMATICA_DIR / MATHEMATICA_FULL_FILE, index_label="name")
    print(f"wrote {MATHEMATICA_DIR / MATHEMATICA_FILE} {subset.shape} and "
          f"{MATHEMATICA_DIR / MATHEMATICA_FULL_FILE} {full.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rdata", type=Path, help="use a local tc.RData instead of downloading it")
    main(parser.parse_args().rdata)
