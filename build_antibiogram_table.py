"""
build_antibiogram_table.py
==========================
Combines the per-isolate antibiogram files (data/raw/antibiograms/*.tsv, copied from each
BioSample page) into ONE table with a BioSample column, using data/raw/antibiogram_sources.csv
to say which file belongs to which BioSample.

    python build_antibiogram_table.py

Values are copied as text exactly as supplied; nothing is recalculated. To add an isolate, put
its antibiogram in data/raw/antibiograms/, add a row to antibiogram_sources.csv, and run this script.
(The Data Upload page in the app can also take a combined file directly.)
"""
import csv
from pathlib import Path

import data_loader as dl

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = RAW / "antibiograms_combined.csv"


def main() -> None:
    with open(RAW / "antibiogram_sources.csv", newline="", encoding="utf-8") as fh:
        sources = list(csv.DictReader(fh))
    rows, header = [], None
    for s in sources:
        df, _ = dl.load_ast_table(RAW / "antibiograms" / s["file"], biosample=s["biosample"])
        header = list(df.columns)
        rows += df.fillna("").to_dict("records")
        print(f"{s['isolate']:14s} {s['biosample']}  {len(df)} results  ({s['file']})")
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows for {len(sources)} isolates -> {OUT}")


if __name__ == "__main__":
    main()
