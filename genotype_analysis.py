"""
genotype_analysis.py
====================
Summaries of AMR gene calls. All functions work on the "long" table made by
preprocessing.build_genotype_long(): one row per (isolate record, gene call) with
columns record_id, biosample, isolate_name, gene, status, call_type.

Reminder of the scientific rules:
  * A gene that is not listed for an isolate is "not reported", NOT "confirmed absent".
  * COMPLETE, PARTIAL, PARTIAL_END_OF_CONTIG, HMM, POINT and MISTRANSLATION are always
    counted separately; they are never merged into a single "present" flag.
  * Frequencies are counted per isolate RECORD (a gene counts once per record).
"""
from __future__ import annotations

import pandas as pd

from preprocessing import STATUS_ORDER

# Plain-English meaning of each status, shown next to results in the dashboard.
STATUS_NOTES = {
    "COMPLETE": "Full-length match to the reference gene.",
    "PARTIAL": "Only part of the gene was matched. Evidence is incomplete; "
               "this is NOT a confirmed gene.",
    "PARTIAL_END_OF_CONTIG": "Gene runs off the end of a contig in the assembly, so it is "
                             "truncated by assembly breaks, not necessarily by biology. "
                             "Presence is uncertain.",
    "HMM": "Detected only by a profile (HMM) match, not a full-length reference match. "
           "Treat as lower-confidence evidence.",
    "POINT": "A resistance-associated point mutation was detected in a chromosomal gene.",
    "MISTRANSLATION": "The gene sequence is present but is predicted to be mistranslated "
                      "(e.g. frameshift/premature stop); function is uncertain.",
    "UNPARSED": "Entry could not be split into gene and status.",
}

# Coarser grouping used for the "complete vs partial" chart. Still kept distinct from each other.
STATUS_GROUPS = {
    "COMPLETE": "Complete",
    "PARTIAL": "Partial",
    "PARTIAL_END_OF_CONTIG": "Partial",
    "HMM": "HMM",
    "POINT": "Point mutation",
    "MISTRANSLATION": "Mistranslation",
    "UNPARSED": "Unparsed",
}
GROUP_ORDER = ["Complete", "Partial", "HMM", "Point mutation", "Mistranslation", "Unparsed"]


def genotype_summary(long_df: pd.DataFrame, n_records: int | None = None) -> dict:
    """Headline numbers for the Home dashboard."""
    if long_df.empty:
        return {"n_records": n_records or 0, "n_records_with_calls": 0, "n_gene_calls": 0,
                "n_distinct_genes": 0, "n_distinct_point_mutations": 0, "status_counts": {}}
    genes = long_df[long_df["call_type"] == "gene"]
    points = long_df[long_df["call_type"] == "point_mutation"]
    return {
        "n_records": n_records if n_records is not None else long_df["record_id"].nunique(),
        "n_records_with_calls": int(long_df["record_id"].nunique()),
        "n_gene_calls": int(len(long_df)),
        "n_distinct_genes": int(genes["gene"].nunique()),
        "n_distinct_point_mutations": int(points["gene"].nunique()),
        "status_counts": long_df["status"].value_counts().to_dict(),
    }


def status_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    """Number of gene calls at each status (all statuses listed, zeros included)."""
    counts = long_df["status"].value_counts()
    order = STATUS_ORDER + [s for s in counts.index if s not in STATUS_ORDER]
    out = pd.DataFrame({"status": order, "n_calls": [int(counts.get(s, 0)) for s in order]})
    out["group"] = out["status"].map(STATUS_GROUPS)
    out["meaning"] = out["status"].map(STATUS_NOTES)
    return out


def gene_frequency(long_df: pd.DataFrame, n_records: int | None = None) -> pd.DataFrame:
    """
    One row per gene: in how many isolate records it was reported (at any status) and
    how many at each status. pct_records = share of ALL records, so it is comparable
    between genes. Sorted most frequent first.
    """
    cols = ["gene", "call_type", "n_records_any", "pct_records"] + [f"n_{s}" for s in STATUS_ORDER]
    if long_df.empty:
        return pd.DataFrame(columns=cols)
    total = n_records if n_records else long_df["record_id"].nunique()

    per_status = (long_df.groupby(["gene", "status"])["record_id"].nunique()
                  .unstack(fill_value=0))
    for s in STATUS_ORDER:
        if s not in per_status.columns:
            per_status[s] = 0
    per_status = per_status[STATUS_ORDER].add_prefix("n_")

    out = per_status.copy()
    out["n_records_any"] = long_df.groupby("gene")["record_id"].nunique()
    out["pct_records"] = (100 * out["n_records_any"] / total).round(1)
    out["call_type"] = long_df.groupby("gene")["call_type"].agg(
        lambda s: "point_mutation" if (s == "point_mutation").any() else "gene")
    out = out.reset_index()[cols]
    return out.sort_values(["n_records_any", "gene"], ascending=[False, True]).reset_index(drop=True)


def status_group_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    """Gene calls per evidence group (Complete / Partial / HMM / Point mutation ...)."""
    if long_df.empty:
        return pd.DataFrame(columns=["group", "n_calls"])
    g = long_df["status"].map(STATUS_GROUPS).value_counts()
    return pd.DataFrame({"group": GROUP_ORDER,
                         "n_calls": [int(g.get(x, 0)) for x in GROUP_ORDER]})


def point_mutations(long_df: pd.DataFrame, n_records: int | None = None) -> pd.DataFrame:
    """Frequency table restricted to point mutations (status POINT)."""
    pts = long_df[long_df["status"] == "POINT"]
    if pts.empty:
        return pd.DataFrame(columns=["mutation", "gene_locus", "n_records", "pct_records"])
    total = n_records if n_records else long_df["record_id"].nunique()
    out = pts.groupby("gene")["record_id"].nunique().rename("n_records").reset_index()
    out = out.rename(columns={"gene": "mutation"})
    out.insert(1, "gene_locus", out["mutation"].str.split("_").str[0])
    out["pct_records"] = (100 * out["n_records"] / total).round(1)
    return out.sort_values(["n_records", "mutation"], ascending=[False, True]).reset_index(drop=True)


def isolate_profile(long_df: pd.DataFrame, record_id: str) -> pd.DataFrame:
    """Every gene call for ONE isolate record, with what its status means."""
    sub = long_df[long_df["record_id"] == record_id].copy()
    status_rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    sub["_r"] = sub["status"].map(status_rank).fillna(99)
    sub = sub.sort_values(["_r", "gene"]).drop(columns="_r")
    sub["meaning"] = sub["status"].map(STATUS_NOTES)
    return sub[["gene", "status", "call_type", "meaning"]].reset_index(drop=True)


def isolate_gene_matrix(long_df: pd.DataFrame, record_ids: list[str] | None = None,
                        genes: list[str] | None = None, top_n: int | None = None) -> pd.DataFrame:
    """
    Isolate x gene table whose cells hold the STATUS text (e.g. 'COMPLETE', 'HMM').
    Empty cell = not reported (which is not the same as absent).
    """
    df = long_df
    if record_ids is not None:
        df = df[df["record_id"].isin(record_ids)]
    if genes is not None:
        df = df[df["gene"].isin(genes)]
    elif top_n:
        keep = df.groupby("gene")["record_id"].nunique().nlargest(top_n).index
        df = df[df["gene"].isin(keep)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(label=df["isolate_name"] + " (" + df["record_id"] + ")")
    mat = df.pivot_table(index="label", columns="gene", values="status",
                         aggfunc=lambda s: "/".join(sorted(set(s))), fill_value="")
    return mat


def genes_per_isolate(long_df: pd.DataFrame) -> pd.DataFrame:
    """How many gene calls each isolate record has, split by evidence group."""
    if long_df.empty:
        return pd.DataFrame()
    tmp = long_df.assign(group=long_df["status"].map(STATUS_GROUPS))
    out = (tmp.pivot_table(index=["record_id", "isolate_name"], columns="group",
                           values="gene", aggfunc="count", fill_value=0)
           .reindex(columns=GROUP_ORDER, fill_value=0))
    out["total_calls"] = out.sum(axis=1)
    return out.reset_index().sort_values("total_calls", ascending=False).reset_index(drop=True)
