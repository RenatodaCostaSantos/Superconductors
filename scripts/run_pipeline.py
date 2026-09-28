"""Run the whole data pipeline, from the raw UCI files to every feature file.

Steps (each skipped when its output already exists, unless --force):

  1. download the UCI "Superconductivty Data" (train.csv, unique_m.csv) into data/raw/
  2. extract the author's Mathematica element table into data/external/
     (scripts/fetch_mathematica_table.py; skip with --no-mathematica)
  3. build data/processed/features_<dataset>_<elements>.csv for every combination of
     --dataset {published, cleaned} and --elements {pymatgen, mathematica}
     (scripts/build_features.py); the cleaned dataset also writes cleaning_log.csv

Usage:
  uv run python scripts/run_pipeline.py                   # everything
  uv run python scripts/run_pipeline.py --no-mathematica  # open-source element table only
"""

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from supercon.elements import MATHEMATICA_DIR, MATHEMATICA_FILE

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UCI_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


def download_uci(force: bool) -> None:
    if not force and (RAW / "unique_m.csv").exists() and (RAW / "train.csv").exists():
        print(f"[1/3] UCI data already in {RAW}", flush=True)
        return
    print(f"[1/3] downloading {UCI_URL}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "superconduct.zip"
    urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW, members=["train.csv", "unique_m.csv"])


def main(mathematica: bool, force: bool) -> None:
    download_uci(force)

    if not mathematica:
        print("[2/3] Mathematica element table skipped (--no-mathematica)", flush=True)
    elif not force and (MATHEMATICA_DIR / MATHEMATICA_FILE).exists():
        print(f"[2/3] Mathematica element table already in {MATHEMATICA_DIR}", flush=True)
    else:
        print("[2/3] extracting the Mathematica element table", flush=True)
        run("fetch_mathematica_table.py")

    print("[3/3] building feature files", flush=True)
    for elements in ["pymatgen", "mathematica"] if mathematica else ["pymatgen"]:
        for dataset in ["published", "cleaned"]:
            run("build_features.py", "--dataset", dataset, "--elements", elements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mathematica", action="store_true", help="skip the author's Mathematica table")
    parser.add_argument("--force", action="store_true", help="download and extract again even if files exist")
    args = parser.parse_args()
    main(mathematica=not args.no_mathematica, force=args.force)
