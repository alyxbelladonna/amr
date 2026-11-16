"""
preprocessing.py
================
Cleans the loaded tables and integrates genotype + AST + metadata.

Scientific rules implemented here:
  * Gene-call statuses (COMPLETE, PARTIAL, PARTIAL_END_OF_CONTIG, HMM, POINT,
    MISTRANSLATION) are kept SEPARATE. A gene that is not listed is "not reported",
    never "confirmed absent", and partial/HMM calls are never turned into absence.
  * Genotype and AST are matched ONLY by BioSample accession.
  * Isolates without a match stay in their own tables (nothing is dropped or invented).
  * AST measurement signs and values are preserved exactly as supplied.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATUS_ORDER = ["COMPLETE", "PARTIAL", "PARTIAL_END_OF_CONTIG", "HMM", "POINT", "MISTRANSLATION"]

_QUOTED = re.compile(r'"([^"]+)"')


# --------------------------------------------------------------------------- #
# Genotype parsing
# --------------------------------------------------------------------------- #
def normalize_biosample(value) -> str | float:
    """Standardise an accession so matching is not defeated by spaces or case."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    v = str(value).strip().upper()
    return v if v else np.nan


def parse_genotype_string(value) -> list[tuple[str, str]]:
    """
    'mecA=COMPLETE,gyrA_S84L=POINT' -> [('mecA','COMPLETE'), ('gyrA_S84L','POINT')]
    An entry without '=' is kept with status 'UNPARSED' so nothing silently disappears.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return []
    calls = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            gene, status = token.rsplit("=", 1)
            calls.append((gene.strip(), status.strip().upper()))
        else:
            calls.append((token, "UNPARSED"))
    return calls


def _display_name(row: pd.Series) -> str:
    if pd.notna(row.get("Strain")):
        return str(row["Strain"])
    ids = _QUOTED.findall(str(row.get("Isolate identifiers", "")))
    if ids:
        return ids[0]
    return str(row.get("Isolate", "unknown"))


def prepare_genotype_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per isolate record, with tidy helper columns added (originals kept)."""
    out = df.copy()
    out["biosample"] = out["BioSample"].map(normalize_biosample)
    out["record_id"] = out["Isolate"] if "Isolate" in out.columns else out.index.astype(str)
    out["isolate_name"] = out.apply(_display_name, axis=1)
    out["n_calls"] = out["AMR genotypes"].map(lambda s: len(parse_genotype_string(s)))
    # The same BioSample can appear in several records (same sample sequenced/assembled twice)
    counts = out["biosample"].map(out["biosample"].value_counts())
    out["biosample_record_count"] = counts.fillna(1).astype(int)
    return out


def build_genotype_long(geno: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (isolate record, gene call) with its status."""
    rows = []
    for record_id, biosample, name, genotype_str in zip(
            geno["record_id"], geno["biosample"], geno["isolate_name"], geno["AMR genotypes"]):
        for gene, status in parse_genotype_string(genotype_str):
            rows.append((record_id, biosample, name, gene, status,
                         "point_mutation" if status == "POINT" else "gene"))
    return pd.DataFrame(rows, columns=["record_id", "biosample", "isolate_name",
                                       "gene", "status", "call_type"])


def genes_by_status(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: per record, the genes found at each status (kept separate)."""
    if long_df.empty:
        return pd.DataFrame(columns=["record_id"])
    wide = (long_df.groupby(["record_id", "status"])["gene"]
            .apply(lambda s: ", ".join(sorted(s))).unstack("status"))
    wide = wide.reindex(columns=[s for s in STATUS_ORDER if s in wide.columns]
                        + [c for c in wide.columns if c not in STATUS_ORDER])
    wide.columns = [f"genes_{c}" for c in wide.columns]
    return wide.reset_index()


# --------------------------------------------------------------------------- #
# AST cleaning
# --------------------------------------------------------------------------- #
def normalize_ast(ast: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy column names and add matching helpers. The original text of the sign and
    measurement is preserved; `measurement_value` is an extra numeric copy for plotting.
    """
    out = pd.DataFrame({
        "biosample": ast["BioSample"].map(normalize_biosample),
        "antibiotic": ast["Antibiotic"].str.strip().str.lower(),
        "phenotype_reported": ast["Resistance phenotype"],
        "phenotype": ast["Resistance phenotype"].str.strip().str.lower(),
        "measurement_sign": ast["Measurement sign"],
        "measurement": ast["Measurement"],
        "measurement_units": ast["Measurement units"],
        "method": ast["Laboratory typing method"],
        "standard": ast["Testing standard"],
    })
    out["measurement_value"] = pd.to_numeric(out["measurement"], errors="coerce")
    out["measurement_reported"] = (out["measurement_sign"].fillna("") + out["measurement"].fillna("")
                                   + " " + out["measurement_units"].fillna("")).str.strip()
    return out


def summarise_ast(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per BioSample listing antibiotics by phenotype."""
    if ast_clean.empty:
        return pd.DataFrame(columns=["biosample", "n_ast_tests", "ast_resistant",
                                     "ast_intermediate", "ast_susceptible"])

    def join(s):
        return ", ".join(sorted(s))
    g = ast_clean.groupby("biosample")
    summary = pd.DataFrame({
        "n_ast_tests": g.size(),
        "ast_resistant": g.apply(lambda d: join(d.loc[d.phenotype == "resistant", "antibiotic"]),
                                 include_groups=False),
        "ast_intermediate": g.apply(lambda d: join(d.loc[d.phenotype == "intermediate", "antibiotic"]),
                                    include_groups=False),
        "ast_susceptible": g.apply(lambda d: join(d.loc[d.phenotype == "susceptible", "antibiotic"]),
                                   include_groups=False),
    })
    return summary.reset_index()


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
@dataclass
class IntegrationResult:
    integrated_all: pd.DataFrame        # every genotype record, AST/metadata attached where available
    integrated_matched: pd.DataFrame    # only records whose BioSample has AST data
    genotype_only: pd.DataFrame         # genotype records with NO AST data (kept separate)
    ast_only: pd.DataFrame              # AST rows whose BioSample is NOT in the genotype table
    ast_long_matched: pd.DataFrame      # one row per matched record x antibiotic
    genotype_long: pd.DataFrame         # one row per record x gene call
    summary: dict = field(default_factory=dict)


def integrate(genotype: pd.DataFrame, ast_clean: pd.DataFrame,
              metadata: pd.DataFrame | None = None) -> IntegrationResult:
    geno = prepare_genotype_table(genotype)
    long_df = build_genotype_long(geno)

    table = geno.merge(genes_by_status(long_df), on="record_id", how="left")

    if metadata is not None and len(metadata):
        meta = metadata.copy()
        meta["biosample"] = meta["BioSample"].map(normalize_biosample)
        meta = meta.drop(columns=["BioSample"]).drop_duplicates("biosample")
        meta = meta.add_prefix("meta_").rename(columns={"meta_biosample": "biosample"})
        table = table.merge(meta, on="biosample", how="left")

    ast_summary = summarise_ast(ast_clean)
    table = table.merge(ast_summary, on="biosample", how="left")
    table["has_ast"] = table["n_ast_tests"].notna()
    table["n_ast_tests"] = table["n_ast_tests"].fillna(0).astype(int)

    matched = table[table["has_ast"]].copy()
    genotype_only = table[~table["has_ast"]].copy()
    ast_only = ast_clean[~ast_clean["biosample"].isin(set(geno["biosample"].dropna()))].copy()

    ast_long_matched = ast_clean.merge(
        geno[["record_id", "biosample", "isolate_name"]], on="biosample", how="inner")

    summary = {
        "n_genotype_records": len(geno),
        "n_unique_biosamples": int(geno["biosample"].nunique()),
        "n_records_with_duplicate_biosample": int((geno["biosample_record_count"] > 1).sum()),
        "n_ast_rows": len(ast_clean),
        "n_ast_biosamples": int(ast_clean["biosample"].nunique()),
        "n_matched_records": len(matched),
        "n_matched_biosamples": int(matched["biosample"].nunique()),
        "n_genotype_only_records": len(genotype_only),
        "n_ast_only_biosamples": int(ast_only["biosample"].nunique()),
        "n_gene_calls": len(long_df),
        "n_distinct_genes": int(long_df["gene"].nunique()),
    }
    return IntegrationResult(table, matched, genotype_only, ast_only,
                             ast_long_matched, long_df, summary)


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values."""
    n = len(df)
    rep = pd.DataFrame({"column": df.columns,
                        "n_missing": df.isna().sum().values})
    rep["pct_missing"] = (100 * rep["n_missing"] / n).round(1) if n else 0.0
    return rep.sort_values("n_missing", ascending=False).reset_index(drop=True)
