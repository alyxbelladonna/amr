"""
phenotype_analysis.py
=====================
Summaries of antibiotic susceptibility testing (AST) results. Works on the table made by
preprocessing.normalize_ast().

Rules followed here:
  * Phenotype labels (resistant / susceptible / intermediate) are used EXACTLY as reported
    by the laboratory. They are NOT re-interpreted against breakpoints.
  * Measurement signs are preserved. A value such as ">=24 mm" is a bound, not an exact
    zone diameter, and is labelled that way (never plotted as if it were exact).
"""
from __future__ import annotations

import pandas as pd

PHENOTYPE_ORDER = ["resistant", "intermediate", "susceptible"]

# What a reported sign means about the true measurement
_SIGN_KIND = {
    "==": "exact", "=": "exact",
    ">=": "lower bound (true value >= reported)", ">": "lower bound (true value > reported)",
    "<=": "upper bound (true value <= reported)", "<": "upper bound (true value < reported)",
}


def measurement_kind(sign) -> str:
    if sign is None or (isinstance(sign, float) and pd.isna(sign)) or str(sign).strip() == "":
        return "no sign reported"
    return _SIGN_KIND.get(str(sign).strip(), f"unrecognised sign '{sign}'")


def antibiogram_table(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """The antibiogram as supplied (row order preserved), one row per isolate x antibiotic."""
    out = ast_clean[["biosample", "antibiotic", "phenotype_reported", "measurement_sign",
                     "measurement", "measurement_units", "method", "standard"]].copy()
    out["measurement_reported"] = ast_clean["measurement_reported"]
    out["measurement_meaning"] = ast_clean["measurement_sign"].map(measurement_kind)
    return out.rename(columns={"phenotype_reported": "phenotype"}).reset_index(drop=True)


def phenotype_counts(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """Number of test results per phenotype (resistant/intermediate/susceptible always listed)."""
    counts = ast_clean["phenotype"].value_counts()
    labels = PHENOTYPE_ORDER + [p for p in counts.index if p not in PHENOTYPE_ORDER]
    return pd.DataFrame({"phenotype": labels, "n_results": [int(counts.get(p, 0)) for p in labels]})


def antibiotics_by_phenotype(ast_clean: pd.DataFrame, biosample: str | None = None) -> dict[str, list[str]]:
    """{'resistant': [...], 'intermediate': [...], 'susceptible': [...]} for one or all isolates."""
    df = ast_clean if biosample is None else ast_clean[ast_clean["biosample"] == biosample]
    out = {p: [] for p in PHENOTYPE_ORDER}
    for ab, ph in zip(df["antibiotic"], df["phenotype"]):
        out.setdefault(ph, []).append(ab)
    return {k: sorted(set(v)) for k, v in out.items()}


def zone_diameters(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Zone-diameter / MIC values exactly as reported, plus what each sign means.
    `plot_value` is the numeric value for charting only; check `measurement_meaning`
    before treating it as an exact measurement.
    """
    df = ast_clean[ast_clean["measurement"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["biosample", "antibiotic", "phenotype", "measurement_reported",
                                     "measurement_meaning", "plot_value", "method"])
    out = pd.DataFrame({
        "biosample": df["biosample"], "antibiotic": df["antibiotic"],
        "phenotype": df["phenotype_reported"],
        "measurement_reported": df["measurement_reported"],
        "measurement_meaning": df["measurement_sign"].map(measurement_kind),
        "plot_value": df["measurement_value"],
        "measurement_units": df["measurement_units"],
        "method": df["method"],
    })
    return out.reset_index(drop=True)


def ast_matrix(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """Isolate x antibiotic table of phenotypes (empty cell = not tested / not reported)."""
    return ast_clean.pivot_table(index="biosample", columns="antibiotic", values="phenotype",
                                 aggfunc="first", fill_value="")


def resistance_by_antibiotic(ast_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Per antibiotic: how many DISTINCT BioSamples were tested and the phenotype split.
    With a single isolate these are all 0/1 and must not be read as a resistance RATE.
    """
    g = ast_clean.groupby("antibiotic")
    out = pd.DataFrame({"n_tested": g["biosample"].nunique()})
    for p in PHENOTYPE_ORDER:
        out[f"n_{p}"] = g.apply(lambda d, p=p: d.loc[d.phenotype == p, "biosample"].nunique(),
                                include_groups=False)
    other = out["n_tested"] - out[[f"n_{p}" for p in PHENOTYPE_ORDER]].sum(axis=1)
    out["n_other"] = other
    return out.reset_index().sort_values("antibiotic").reset_index(drop=True)


def label_availability(ast_long_matched: pd.DataFrame) -> pd.DataFrame:
    """
    For each antibiotic: how many matched ISOLATES (distinct BioSamples) carry a
    resistant vs susceptible label. model.py uses this to decide whether training is
    even statistically meaningful.
    """
    df = ast_long_matched[ast_long_matched["phenotype"].isin(["resistant", "susceptible"])]
    if df.empty:
        return pd.DataFrame(columns=["antibiotic", "n_isolates_labelled",
                                     "n_resistant", "n_susceptible"])
    out = df.groupby("antibiotic").apply(
        lambda d: pd.Series({
            "n_isolates_labelled": d["biosample"].nunique(),
            "n_resistant": d.loc[d.phenotype == "resistant", "biosample"].nunique(),
            "n_susceptible": d.loc[d.phenotype == "susceptible", "biosample"].nunique(),
        }), include_groups=False)
    return out.reset_index().sort_values("antibiotic").reset_index(drop=True)
