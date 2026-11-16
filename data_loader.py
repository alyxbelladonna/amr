"""
data_loader.py
==============
Reads and VALIDATES the input files. Nothing in this file changes the meaning
of the data: values are read as text exactly as supplied (e.g. the measurement
sign "<=" and the number "18" stay separate strings).

Every loader accepts either a file path or an uploaded file object (so the same
functions work for local files and for Streamlit uploads).

Loaders return (data, warnings) where `warnings` is a list of plain-English
messages. Serious problems raise DataValidationError with a clear message.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import IO, Iterator, Union

import pandas as pd

PathOrBuffer = Union[str, Path, IO]


class DataValidationError(ValueError):
    """Raised when an input file is missing, malformed or lacks required columns."""


# --------------------------------------------------------------------------- #
# Column requirements
# --------------------------------------------------------------------------- #
GENOTYPE_REQUIRED = ["BioSample", "AMR genotypes"]
GENOTYPE_EXPECTED = [
    "Organism group", "Strain", "Isolate identifiers", "Isolate", "Create date",
    "Location", "Isolation source", "Isolation type", "SNP cluster",
    "BioSample", "Assembly", "AMR genotypes",
]
AST_REQUIRED = [
    "Antibiotic", "Resistance phenotype", "Measurement sign", "Measurement",
    "Measurement units", "Laboratory typing method", "Testing standard",
]
KNOWN_PHENOTYPES = {"resistant", "susceptible", "intermediate", "non-susceptible",
                    "susceptible-dose dependent", "not defined"}

# NCBI BioSample accessions look like SAMN12345678, SAMEA1234567, SAMD00012345
BIOSAMPLE_RE = re.compile(r"^SAM(N|EA?|D)\d+$")

# Characters allowed in nucleotide FASTA sequence lines (IUPAC + gap/stop)
_FASTA_SEQ_RE = re.compile(r"^[ACGTNRYSWKMBDHVUacgtnryswkmbdhvu\-\*]+$")


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _read_text(source: PathOrBuffer, label: str = "file") -> str:
    """Return the full text of a path or uploaded file object."""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise DataValidationError(f"{label}: file not found: {path}")
            raw = path.read_bytes()
        elif hasattr(source, "getvalue"):
            raw = source.getvalue()
        else:
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
    except OSError as exc:
        raise DataValidationError(f"{label}: could not be read ({exc}).") from exc
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{label}: not a UTF-8 text file ({exc}).") from exc


def _read_table(source: PathOrBuffer, label: str) -> pd.DataFrame:
    """Read a CSV or TSV (separator auto-detected) with every value kept as text."""
    text = _read_text(source, label)
    if not text.strip():
        raise DataValidationError(f"{label}: the file is empty.")
    first_line = text.splitlines()[0]
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str,
                         na_values=[""], keep_default_na=False)
    except Exception as exc:  # pandas raises several different error types
        raise DataValidationError(f"{label}: could not be parsed as a table ({exc}).") from exc
    df.columns = [str(c).strip().lstrip("#") for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{label}: missing required column(s) {missing}. "
            f"Columns found: {list(df.columns)}"
        )


def _get(d: dict, *keys, default=None):
    """Safely walk nested dictionaries."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# --------------------------------------------------------------------------- #
# 1. Genotype table (NCBI Pathogen Detection "isolates.csv")
# --------------------------------------------------------------------------- #
def load_genotype_csv(source: PathOrBuffer) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    df = _read_table(source, "Genotype CSV")
    _require_columns(df, GENOTYPE_REQUIRED, "Genotype CSV")

    absent = [c for c in GENOTYPE_EXPECTED if c not in df.columns]
    if absent:
        warnings.append(f"Genotype CSV: optional column(s) not present: {absent}")

    bad = df["BioSample"].dropna()
    bad = bad[~bad.str.strip().str.match(BIOSAMPLE_RE)]
    if len(bad):
        warnings.append(f"Genotype CSV: {len(bad)} BioSample value(s) do not look like "
                        f"NCBI accessions (e.g. {bad.iloc[0]!r}).")
    n_missing_bs = int(df["BioSample"].isna().sum())
    if n_missing_bs:
        warnings.append(f"Genotype CSV: {n_missing_bs} row(s) have no BioSample and "
                        "cannot be matched to AST data.")
    if "Organism group" in df.columns:
        others = sorted(set(df["Organism group"].dropna()) - {"Staphylococcus aureus"})
        if others:
            warnings.append(f"Genotype CSV: rows for other organisms found: {others}")
    return df, warnings


# --------------------------------------------------------------------------- #
# 2. AST / antibiogram table (NCBI BioSample antibiogram, TSV or CSV)
# --------------------------------------------------------------------------- #
def load_ast_table(source: PathOrBuffer,
                   biosample: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Read an antibiogram. The BioSample antibiogram table has no BioSample column,
    so if it is absent you must say which BioSample the table belongs to.
    Values (sign, measurement, units...) are kept as the exact text supplied.
    """
    warnings: list[str] = []
    df = _read_table(source, "AST table")
    df = df.dropna(how="all").copy()
    _require_columns(df, AST_REQUIRED, "AST table")

    if "BioSample" not in df.columns:
        if not biosample or not BIOSAMPLE_RE.match(biosample.strip()):
            raise DataValidationError(
                "AST table has no 'BioSample' column. Supply the BioSample "
                "accession it belongs to (e.g. SAMN47742561).")
        df.insert(0, "BioSample", biosample.strip())
        warnings.append(f"AST table: BioSample {biosample.strip()} was supplied by the "
                        "user (the file itself has no BioSample column).")
    else:
        bad = df["BioSample"].dropna()
        bad = bad[~bad.str.strip().str.match(BIOSAMPLE_RE)]
        if len(bad):
            raise DataValidationError(
                f"AST table: invalid BioSample value(s), e.g. {bad.iloc[0]!r}.")

    unknown = sorted(set(df["Resistance phenotype"].dropna().str.strip().str.lower())
                     - KNOWN_PHENOTYPES)
    if unknown:
        warnings.append(f"AST table: unrecognised phenotype label(s) kept as-is: {unknown}")
    if df["Antibiotic"].isna().any():
        raise DataValidationError("AST table: some rows have no antibiotic name.")
    if df["Resistance phenotype"].isna().any():
        warnings.append("AST table: some rows have no resistance phenotype.")

    dup = df.duplicated(subset=["BioSample", "Antibiotic"], keep=False)
    if dup.any():
        warnings.append(f"AST table: {int(dup.sum())} rows repeat the same BioSample + "
                        "antibiotic; check for conflicting results.")
    return df.reset_index(drop=True), warnings


# --------------------------------------------------------------------------- #
# 3. Isolate metadata CSV (optional upload)
# --------------------------------------------------------------------------- #
def load_metadata_csv(source: PathOrBuffer) -> tuple[pd.DataFrame, list[str]]:
    df = _read_table(source, "Metadata CSV")
    _require_columns(df, ["BioSample"], "Metadata CSV")
    warnings: list[str] = []
    dup = int(df["BioSample"].duplicated().sum())
    if dup:
        warnings.append(f"Metadata CSV: {dup} duplicate BioSample row(s); first kept.")
        df = df.drop_duplicates("BioSample", keep="first")
    return df.reset_index(drop=True), warnings


# --------------------------------------------------------------------------- #
# 4. NCBI Datasets genome package
# --------------------------------------------------------------------------- #
def find_genome_package_files(folder: PathOrBuffer) -> dict[str, list[Path]]:
    """Look inside a downloaded NCBI Datasets package and list what is really there."""
    root = Path(folder)
    if not root.is_dir():
        raise DataValidationError(f"Genome folder not found: {root}")
    return {
        "fasta": sorted(list(root.rglob("*.fna")) + list(root.rglob("*.fasta"))
                        + list(root.rglob("*.fa"))),
        "assembly_report": sorted(root.rglob("assembly_data_report.jsonl")),
        "sequence_report": sorted(list(root.rglob("sequence_report.jsonl"))
                                  + list(root.rglob("sequence_report.tsv"))),
        "data_summary": sorted(root.rglob("data_summary.tsv")),
    }


def read_genome_package_zip(source: PathOrBuffer,
                            max_total_bytes: int = 500_000_000) -> dict[str, list[tuple[str, io.BytesIO]]]:
    """
    Open an NCBI Datasets .zip IN MEMORY (nothing is written to disk) and return the
    useful files as {'fasta': [(name, BytesIO)], 'assembly_report': [...], 'sequence_report': [...]}.
    """
    import zipfile

    try:
        zf = zipfile.ZipFile(source if not isinstance(source, Path) else str(source))
    except zipfile.BadZipFile as exc:
        raise DataValidationError("Genome package: not a valid .zip file.") from exc
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if sum(i.file_size for i in infos) > max_total_bytes:
            raise DataValidationError("Genome package: zip is larger than the allowed size.")
        found: dict[str, list[tuple[str, io.BytesIO]]] = {
            "fasta": [], "assembly_report": [], "sequence_report": []}
        for info in infos:
            low = info.filename.lower()
            base = Path(info.filename).name
            if low.endswith((".fna", ".fasta", ".fa")):
                key = "fasta"
            elif base.lower() == "assembly_data_report.jsonl":
                key = "assembly_report"
            elif base.lower() in ("sequence_report.tsv", "sequence_report.jsonl"):
                key = "sequence_report"
            else:
                continue
            found[key].append((base, io.BytesIO(zf.read(info))))
    if not any(found.values()):
        raise DataValidationError(
            "Genome package: no FASTA, assembly_data_report.jsonl or sequence report found inside.")
    return found


def _iter_lines(source: PathOrBuffer) -> Iterator[str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataValidationError(f"FASTA: file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from fh
    else:
        data = source.getvalue() if hasattr(source, "getvalue") else source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        yield from io.StringIO(data)


def genome_stats_from_fasta(source: PathOrBuffer) -> tuple[dict, pd.DataFrame]:
    """
    Stream a genome FASTA and compute assembly statistics ourselves.
    Returns (summary_dict, per_contig_dataframe). Sequences are not kept in memory.
    """
    contigs: list[dict] = []
    name = None
    length = gc = n_count = 0

    def flush():
        if name is not None:
            contigs.append({"contig": name, "length": length, "gc_count": gc,
                            "n_count": n_count,
                            "gc_percent": round(100 * gc / length, 2) if length else 0.0})

    for lineno, line in enumerate(_iter_lines(source), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            name = line[1:].split()[0] if len(line) > 1 else f"unnamed_{len(contigs) + 1}"
            length = gc = n_count = 0
            continue
        if name is None:
            raise DataValidationError("FASTA: file does not start with a '>' header line.")
        if not _FASTA_SEQ_RE.match(line):
            raise DataValidationError(
                f"FASTA: unexpected characters on line {lineno} (is this a nucleotide FASTA?).")
        up = line.upper()
        length += len(up)
        gc += up.count("G") + up.count("C")
        n_count += up.count("N")
    flush()

    if not contigs:
        raise DataValidationError("FASTA: no sequences found.")

    df = pd.DataFrame(contigs)
    lengths = sorted(df["length"], reverse=True)
    total = int(sum(lengths))
    running, n50, l50 = 0, 0, 0
    for i, ln in enumerate(lengths, start=1):
        running += ln
        if running >= total / 2:
            n50, l50 = ln, i
            break
    gc_total = int(df["gc_count"].sum())
    summary = {
        "n_contigs": len(df),
        "total_length": total,
        "gc_count": gc_total,
        "gc_percent": round(100 * gc_total / total, 2) if total else 0.0,
        "n_count": int(df["n_count"].sum()),
        "n50": int(n50),
        "l50": int(l50),
        "largest_contig": int(max(lengths)),
        "smallest_contig": int(min(lengths)),
    }
    return summary, df


def load_assembly_reports(source: PathOrBuffer) -> list[dict]:
    """Parse assembly_data_report.jsonl into one tidy dict per assembly."""
    text = _read_text(source, "Assembly data report")
    reports = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Assembly data report: line {n} is not valid JSON.") from exc
        if "accession" not in d:
            raise DataValidationError(f"Assembly data report: line {n} has no 'accession'.")
        attrs = {a.get("name"): a.get("value")
                 for a in (_get(d, "assemblyInfo", "biosample", "attributes", default=[]) or [])
                 if isinstance(a, dict)}
        reports.append({
            "accession": d["accession"],
            "assembly_name": _get(d, "assemblyInfo", "assemblyName"),
            "database": _get(d, "sourceDatabase") or _get(d, "assemblyInfo", "sourceDatabase"),
            "organism": _get(d, "organism", "organismName"),
            "biosample": _get(d, "assemblyInfo", "biosample", "accession"),
            "bioproject": _get(d, "assemblyInfo", "bioprojectAccession"),
            "assembly_level": _get(d, "assemblyInfo", "assemblyLevel"),
            "submission_date": _get(d, "assemblyInfo", "releaseDate"),
            "stats": _get(d, "assemblyStats", default={}),
            "annotation": _get(d, "annotationInfo", default={}),
            "biosample_attributes": attrs,
        })
    if not reports:
        raise DataValidationError("Assembly data report: no records found.")
    return reports


def metadata_from_reports(reports: list[dict]) -> pd.DataFrame:
    """One metadata row per BioSample, built from the BioSample attributes NCBI supplies."""
    rows, seen = [], set()
    # Prefer GenBank (GCA_) records over RefSeq copies (GCF_)
    for r in sorted(reports, key=lambda x: not str(x["accession"]).startswith("GCA_")):
        bs = r.get("biosample")
        if not bs or bs in seen:
            continue
        seen.add(bs)
        rows.append({"BioSample": bs, "Assembly accession": r["accession"],
                     "Organism": r.get("organism"), **r["biosample_attributes"]})
    return pd.DataFrame(rows)


def load_sequence_report_tsv(source: PathOrBuffer) -> pd.DataFrame:
    df = _read_table(source, "Sequence report")
    _require_columns(df, ["Seq length"], "Sequence report")
    df["Seq length"] = pd.to_numeric(df["Seq length"], errors="coerce")
    return df


def crosscheck_genome(fasta_summary: dict, report: dict) -> list[str]:
    """Compare statistics we computed from the FASTA with those NCBI reported."""
    stats, msgs = report.get("stats") or {}, []
    checks = [
        ("total length", fasta_summary["total_length"], stats.get("totalSequenceLength")),
        ("contig count", fasta_summary["n_contigs"], stats.get("numberOfContigs")),
        ("contig N50", fasta_summary["n50"], stats.get("contigN50")),
        ("GC base count", fasta_summary["gc_count"], stats.get("gcCount")),
    ]
    for label, ours, theirs in checks:
        if theirs is None:
            msgs.append(f"Genome check: NCBI report has no value for {label}.")
        elif int(ours) == int(theirs):
            msgs.append(f"Genome check OK: {label} = {int(ours):,} (FASTA matches NCBI report).")
        else:
            msgs.append(f"Genome check MISMATCH: {label} FASTA={int(ours):,} vs report={int(theirs):,}.")
    return msgs
