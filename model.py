"""
model.py
========
Two clearly separated parts. Never mix them up in the user interface:

PART A - RULE-BASED MARKER DEMONSTRATION (always available)
    A small, ILLUSTRATIVE lookup of well-known resistance-associated genes/mutations
    per antibiotic. It reports which markers are present in an isolate's genotype table
    and compares that with the laboratory result. It is NOT a trained model, makes NO
    prediction and has no accuracy. The lookup is textbook-level knowledge written by
    the prototype author and has NOT been validated against CARD, ResFinder or CLSI.

PART B - SUPERVISED MACHINE-LEARNING PIPELINE (guarded)
    Feature extraction -> missing-value handling -> split by isolate -> Random Forest ->
    metrics. A guard (assess_trainability) refuses to train or report any metric unless
    there are enough INDEPENDENT labelled isolates in both classes. With the data in this
    project (one labelled isolate) it correctly refuses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# =============================================================================
# PART A - rule-based marker demonstration
# =============================================================================

# How much weight a gene-call status deserves as evidence that the gene is really there.
EVIDENCE_LEVEL = {
    "COMPLETE": "confirmed",
    "POINT": "confirmed",
    "PARTIAL": "uncertain",
    "PARTIAL_END_OF_CONTIG": "uncertain",
    "HMM": "uncertain",
    "MISTRANSLATION": "uncertain",
    "UNPARSED": "uncertain",
}

RULE_BASED_BANNER = ("RULE-BASED DEMONSTRATION - this is NOT a machine-learning prediction. "
                     "No model was trained or used. The marker table is illustrative and unvalidated.")


@dataclass(frozen=True)
class Marker:
    pattern: str   # regular expression that must match the WHOLE gene / mutation name
    role: str      # 'primary' (counted), 'regulatory' or 'context' (shown, but not counted)
    note: str


# ILLUSTRATIVE knowledge table (see module docstring). Edit freely; verify before relying on it.
RULES: dict[str, dict] = {
    "penicillin": {"class": "beta-lactam", "markers": [
        Marker(r"blaZ", "primary", "Beta-lactamase (penicillinase) that breaks down penicillin."),
        Marker(r"mec[AC]", "primary", "Altered penicillin-binding protein (PBP2a/PBP2c) with low affinity for beta-lactams."),
        Marker(r"blaI|blaR1", "regulatory", "Regulators of blaZ expression; not resistance determinants on their own."),
        Marker(r"mecR1|mecI", "regulatory", "Regulators of mecA expression; not resistance determinants on their own."),
    ]},
    "cefoxitin": {"class": "beta-lactam (methicillin/oxacillin surrogate)", "markers": [
        Marker(r"mec[AC]", "primary", "Altered penicillin-binding protein (PBP2a/PBP2c); the usual basis of methicillin resistance (MRSA). "
                                       "A cefoxitin disk result is used as a surrogate for methicillin/oxacillin resistance."),
        Marker(r"mecR1|mecI", "regulatory", "Regulators of mecA expression; not resistance determinants on their own."),
    ]},
    "gentamicin": {"class": "aminoglycoside", "markers": [
        Marker(r"aac\(6'\)-Ie/aph\(2''\)-Ia", "primary", "Bifunctional aminoglycoside-modifying enzyme acting on gentamicin."),
    ]},
    "kanamycin": {"class": "aminoglycoside", "markers": [
        Marker(r"aph\(3'\)-IIIa|ant\(4'\)-Ia", "primary", "Aminoglycoside-modifying enzyme acting on kanamycin."),
    ]},
    "streptomycin": {"class": "aminoglycoside", "markers": [
        Marker(r"ant\(6\)-Ia", "primary", "Aminoglycoside nucleotidyltransferase acting on streptomycin."),
    ]},
    "tetracycline": {"class": "tetracycline", "markers": [
        Marker(r"tet\((K|L)\)", "primary", "Efflux pump that exports tetracycline (usually plasmid-borne)."),
        Marker(r"tet\((M|O)\)", "primary", "Ribosomal protection protein."),
        Marker(r"tet\(38\)", "context", "Chromosomal efflux gene reported in almost all S. aureus; not counted as a marker here."),
    ]},
    "erythromycin": {"class": "macrolide", "markers": [
        Marker(r"erm\(.+\)", "primary", "rRNA methylase giving macrolide-lincosamide-streptogramin B resistance."),
        Marker(r"msr\(A\)", "primary", "ABC efflux pump for macrolides and streptogramin B."),
        Marker(r"mph\(C\)", "primary", "Macrolide phosphotransferase."),
    ]},
    "clindamycin": {"class": "lincosamide", "markers": [
        Marker(r"erm\(.+\)", "primary", "rRNA methylase; clindamycin resistance may be inducible and need a D-test to detect."),
        Marker(r"lnu\(A\)", "primary", "Lincosamide nucleotidyltransferase."),
        Marker(r"lsa\(E\)", "primary", "ABC-F protein giving lincosamide / pleuromutilin / streptogramin A resistance."),
    ]},
    "enrofloxacin": {"class": "fluoroquinolone", "markers": [
        Marker(r"gyrA_(S84[A-Z]|E88[A-Z])", "primary", "Mutation in the gyrA quinolone-resistance-determining region."),
        Marker(r"parC_(S80[A-Z]|E84[A-Z])", "primary", "Mutation in the parC (grlA) quinolone-resistance-determining region."),
        Marker(r"mepA", "context", "Efflux pump gene reported in nearly all isolates; not counted as a marker here."),
    ]},
    "linezolid": {"class": "oxazolidinone", "markers": [
        Marker(r"cfr|optrA|poxtA", "primary", "Acquired ribosomal protection / methylation gene conferring linezolid resistance."),
    ]},
    "trimethoprim-sulfamethoxazole": {"class": "folate-pathway inhibitor", "markers": [
        Marker(r"dfr[A-Z]\d*", "primary", "Trimethoprim-insensitive dihydrofolate reductase."),
        Marker(r"folP_.+", "context", "folP mutations are linked to sulfonamide resistance in some studies; not counted here."),
    ]},
    "chloramphenicol": {"class": "phenicol", "markers": [
        Marker(r"cat(\(.+\))?|fexA|cfr", "primary", "Chloramphenicol acetyltransferase / phenicol exporter / methylase."),
    ]},
    "fosfomycin": {"class": "fosfomycin", "markers": [
        Marker(r"fosB", "primary", "Fosfomycin-modifying enzyme; its effect on the laboratory phenotype varies."),
    ]},
}

# Other names mapped onto the rule entries above
RULE_ALIASES = {
    "oxacillin": "cefoxitin", "methicillin": "cefoxitin",
    "ciprofloxacin": "enrofloxacin", "levofloxacin": "enrofloxacin", "norfloxacin": "enrofloxacin",
    "moxifloxacin": "enrofloxacin",
    "sulfamethoxazole-trimethoprim": "trimethoprim-sulfamethoxazole",
    "co-trimoxazole": "trimethoprim-sulfamethoxazole", "tmp-smx": "trimethoprim-sulfamethoxazole",
}

OUTCOME_TEXT = {
    "MARKER_REPORTED": "Resistance-associated marker reported (complete / confirmed call)",
    "UNCERTAIN_EVIDENCE": "Only uncertain marker evidence (partial, end-of-contig, HMM or mistranslated call)",
    "NO_MARKER_REPORTED": "No associated marker reported in the genotype table (this is not proof of absence)",
    "NO_RULES": "No marker rules defined for this antibiotic",
}


def resolve_rule_key(antibiotic: str) -> str | None:
    a = str(antibiotic).strip().lower()
    a = RULE_ALIASES.get(a, a)
    return a if a in RULES else None


def rule_antibiotics() -> list[str]:
    return sorted(RULES)


def knowledge_table() -> pd.DataFrame:
    rows = [(ab, spec["class"], m.pattern, m.role, m.note)
            for ab, spec in RULES.items() for m in spec["markers"]]
    return pd.DataFrame(rows, columns=["antibiotic", "class", "gene_pattern", "role", "note"])


def evaluate_markers(calls: pd.DataFrame, antibiotic: str) -> dict:
    """
    Look up the markers for `antibiotic` among ONE isolate's gene calls
    (`calls` needs columns gene and status, e.g. from genotype_analysis.isolate_profile).
    Returns the matched calls and an outcome code. It never says 'resistant' or
    'susceptible' - only which evidence was found.
    """
    key = resolve_rule_key(antibiotic)
    cols = ["gene", "status", "evidence", "role", "note"]
    if key is None:
        return {"antibiotic_class": None, "matches": pd.DataFrame(columns=cols), "outcome": "NO_RULES",
                "outcome_text": OUTCOME_TEXT["NO_RULES"]}
    rows = []
    for m in RULES[key]["markers"]:
        rx = re.compile(m.pattern)
        for gene, status in zip(calls["gene"], calls["status"]):
            if rx.fullmatch(gene):
                rows.append((gene, status, EVIDENCE_LEVEL.get(status, "uncertain"), m.role, m.note))
    matches = pd.DataFrame(rows, columns=cols).drop_duplicates(["gene", "status", "role"]).reset_index(drop=True)
    primary = matches[matches["role"] == "primary"]
    if (primary["evidence"] == "confirmed").any():
        outcome = "MARKER_REPORTED"
    elif len(primary):
        outcome = "UNCERTAIN_EVIDENCE"
    else:
        outcome = "NO_MARKER_REPORTED"
    return {"antibiotic_class": RULES[key]["class"], "matches": matches, "outcome": outcome,
            "outcome_text": OUTCOME_TEXT[outcome]}


def compare_with_lab(outcome: str, phenotype: str | None) -> str:
    """
    A descriptive sentence putting the marker evidence next to the laboratory result.
    This is a reading aid for ONE isolate, not a measure of accuracy.
    """
    if phenotype is None or (isinstance(phenotype, float) and pd.isna(phenotype)):
        return "No laboratory result is available for this isolate and antibiotic (it is NOT assumed susceptible)."
    p = str(phenotype).strip().lower()
    if outcome == "NO_RULES":
        return f"Laboratory result: {p}. No marker rules exist for this antibiotic, so no comparison is made."
    if p == "resistant":
        return {
            "MARKER_REPORTED": "Laboratory result is resistant and a complete marker call is reported: the marker is consistent with the result.",
            "UNCERTAIN_EVIDENCE": "Laboratory result is resistant and the marker evidence is incomplete: compatible with the result, but it cannot confirm it.",
            "NO_MARKER_REPORTED": "Laboratory result is resistant but no associated marker is reported: possibly a mechanism outside this table, an assembly gap, or a missed call.",
        }[outcome]
    if p == "susceptible":
        return {
            "MARKER_REPORTED": "Laboratory result is susceptible although a marker is reported: markers do not always produce resistance (regulation, expression, borderline values).",
            "UNCERTAIN_EVIDENCE": "Laboratory result is susceptible and the marker evidence is only partial: the incomplete call may not encode a working gene. This does not prove it.",
            "NO_MARKER_REPORTED": "Laboratory result is susceptible and no associated marker is reported: nothing in the table points to resistance.",
        }[outcome]
    return f"Laboratory result is {p}: intermediate results are not compared with markers here."


def isolate_marker_summary(calls: pd.DataFrame, ast_for_isolate: pd.DataFrame) -> pd.DataFrame:
    """One row per tested antibiotic: lab result next to marker evidence (rule-based, descriptive)."""
    rows = []
    for _, a in ast_for_isolate.iterrows():
        ev = evaluate_markers(calls, a["antibiotic"])
        prim = ev["matches"][ev["matches"]["role"] == "primary"]
        rows.append({
            "antibiotic": a["antibiotic"],
            "lab_result": a["phenotype_reported"],
            "lab_measurement": a["measurement_reported"],
            "marker_evidence": ev["outcome_text"],
            "primary_markers": ", ".join(f"{g} ({s})" for g, s in zip(prim["gene"], prim["status"])) or "-",
            "reading": compare_with_lab(ev["outcome"], a["phenotype"]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# PART B - supervised machine-learning pipeline (guarded)
# =============================================================================

# Prototype thresholds below which training is refused. They are rules of thumb, not a statistical
# guarantee: even above them a small test set gives imprecise metrics. Adjust deliberately.
MIN_ISOLATES_TOTAL = 50
MIN_ISOLATES_PER_CLASS = 15
POSITIVE_CLASS = "resistant"
NEGATIVE_CLASS = "susceptible"


class ModelNotTrainable(Exception):
    """Raised when the data cannot support a meaningful model."""


def assess_trainability(ast_long_matched: pd.DataFrame, antibiotic: str) -> dict:
    """
    Decide whether a supervised model for `antibiotic` is statistically meaningful.
    Counts INDEPENDENT isolates (distinct BioSamples). Records that share a BioSample are one
    isolate, not several. Intermediate and other results are excluded from binary training.
    """
    df = ast_long_matched[ast_long_matched["antibiotic"] == str(antibiotic).strip().lower()]
    lab = df[df["phenotype"].isin([POSITIVE_CLASS, NEGATIVE_CLASS])]
    n_r = int(lab.loc[lab["phenotype"] == POSITIVE_CLASS, "biosample"].nunique())
    n_s = int(lab.loc[lab["phenotype"] == NEGATIVE_CLASS, "biosample"].nunique())
    n_total = int(lab["biosample"].nunique())
    excluded = int(df.loc[~df["phenotype"].isin([POSITIVE_CLASS, NEGATIVE_CLASS]), "biosample"].nunique())
    reasons = []
    if n_total < MIN_ISOLATES_TOTAL:
        reasons.append(f"Only {n_total} independent labelled isolate(s); at least {MIN_ISOLATES_TOTAL} are required.")
    if n_r < MIN_ISOLATES_PER_CLASS:
        reasons.append(f"Only {n_r} resistant isolate(s); at least {MIN_ISOLATES_PER_CLASS} are required.")
    if n_s < MIN_ISOLATES_PER_CLASS:
        reasons.append(f"Only {n_s} susceptible isolate(s); at least {MIN_ISOLATES_PER_CLASS} are required.")
    return {"antibiotic": str(antibiotic).strip().lower(), "n_independent_isolates": n_total,
            "n_resistant": n_r, "n_susceptible": n_s, "n_excluded_non_binary": excluded,
            "trainable": not reasons, "reasons": reasons}


def trainability_table(ast_long_matched: pd.DataFrame) -> pd.DataFrame:
    """Readiness of every antibiotic that has AST data."""
    if ast_long_matched.empty:
        return pd.DataFrame(columns=["antibiotic", "n_independent_isolates", "n_resistant",
                                     "n_susceptible", "trainable", "why_not"])
    rows = []
    for ab in sorted(ast_long_matched["antibiotic"].unique()):
        a = assess_trainability(ast_long_matched, ab)
        rows.append({"antibiotic": ab, "n_independent_isolates": a["n_independent_isolates"],
                     "n_resistant": a["n_resistant"], "n_susceptible": a["n_susceptible"],
                     "trainable": a["trainable"], "why_not": " ".join(a["reasons"]) or "-"})
    return pd.DataFrame(rows)


def build_feature_matrix(long_df: pd.DataFrame, record_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature extraction. Each gene gives TWO binary features, so statuses stay separate:
        '<gene>|confirmed'  1 if called COMPLETE or POINT
        '<gene>|uncertain'  1 if called PARTIAL, PARTIAL_END_OF_CONTIG, HMM or MISTRANSLATION
    Missing-value handling: a gene that is not reported is coded 0 in BOTH columns, which means
    'no call', NOT 'confirmed absent'. Records with no gene calls at all are returned in
    `no_calls` so the caller can exclude them (no calls could mean no data).
    """
    sub = long_df[long_df["record_id"].isin(record_ids)].copy()
    sub["feature"] = sub["gene"] + "|" + sub["status"].map(EVIDENCE_LEVEL).fillna("uncertain")
    X = pd.crosstab(sub["record_id"], sub["feature"]).clip(upper=1)
    missing = [r for r in record_ids if r not in X.index]
    X = X.reindex(record_ids).fillna(0).astype(int)
    return X, missing


@dataclass
class ModelResult:
    antibiotic: str
    n_train: int
    n_test: int
    train_counts: dict
    test_counts: dict
    confusion: pd.DataFrame            # rows = actual, columns = predicted
    precision: float
    recall: float
    f1: float
    accuracy: float
    feature_importance: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def training_set(long_df: pd.DataFrame, ast_long_matched: pd.DataFrame,
                 antibiotic: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Features X and labels y with ONE record per BioSample (independent isolates only)."""
    ab = str(antibiotic).strip().lower()
    lab = ast_long_matched[(ast_long_matched["antibiotic"] == ab)
                           & ast_long_matched["phenotype"].isin([POSITIVE_CLASS, NEGATIVE_CLASS])]
    lab = lab.sort_values("record_id").drop_duplicates("biosample", keep="first")
    X, no_calls = build_feature_matrix(long_df, lab["record_id"].tolist())
    lab = lab[~lab["record_id"].isin(no_calls)]
    X = X.loc[lab["record_id"]]
    X = X.loc[:, X.sum(axis=0) > 0]
    y = pd.Series(lab["phenotype"].values, index=lab["record_id"].values, name="phenotype")
    return X, y, {"n_excluded_no_calls": len(no_calls)}


def train_random_forest(long_df: pd.DataFrame, ast_long_matched: pd.DataFrame, antibiotic: str,
                        test_size: float = 0.25, n_estimators: int = 300,
                        random_state: int = 42) -> ModelResult:
    """
    Random Forest baseline. Refuses (ModelNotTrainable) if the guard says the data are insufficient.
    Because there is one row per isolate, a stratified train/test split of the rows IS a split by isolate.
    """
    status = assess_trainability(ast_long_matched, antibiotic)
    if not status["trainable"]:
        raise ModelNotTrainable(" ".join(status["reasons"]))
    X, y, info = training_set(long_df, ast_long_matched, antibiotic)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ModelNotTrainable("After removing records without gene calls, a class has fewer than 2 isolates.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                 random_state=random_state, n_jobs=1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    labels = [NEGATIVE_CLASS, POSITIVE_CLASS]
    cm = confusion_matrix(y_te, pred, labels=labels)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=[POSITIVE_CLASS],
                                                 average=None, zero_division=0)
    importance = (pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
                  .sort_values("importance", ascending=False).head(15).reset_index(drop=True))
    warns = [f"The test set has only {len(y_te)} isolates, so these metrics are imprecise.",
             "Metrics are for resistant as the positive class. The model uses gene calls only and was "
             "not externally validated. Isolates from the same farm or clone are not independent, which "
             "can make results look better than they are.",
             "Not a clinically validated diagnostic."]
    if info["n_excluded_no_calls"]:
        warns.append(f"{info['n_excluded_no_calls']} isolate(s) without any gene calls were excluded.")
    return ModelResult(
        antibiotic=str(antibiotic).lower(), n_train=len(y_tr), n_test=len(y_te),
        train_counts=y_tr.value_counts().to_dict(), test_counts=y_te.value_counts().to_dict(),
        confusion=pd.DataFrame(cm, index=[f"actual {c}" for c in labels],
                               columns=[f"predicted {c}" for c in labels]),
        precision=float(p[0]), recall=float(r[0]), f1=float(f[0]),
        accuracy=float(accuracy_score(y_te, pred)), feature_importance=importance, warnings=warns)


PIPELINE_STEPS = [
    ("Feature extraction (gene x status, binary)", "implemented - runs on all genotype records"),
    ("Missing-value handling ('no call' coded 0, never 'absent')", "implemented"),
    ("Independent-isolate labels (one record per BioSample)", "implemented"),
    ("Training guard (minimum isolates per class)", "implemented - decides whether training may run"),
    ("Train/test split by isolate (stratified)", "implemented - runs only if the guard passes"),
    ("Random Forest baseline", "implemented - runs only if the guard passes"),
    ("Confusion matrix, precision, recall, F1", "implemented - shown only after a valid training run"),
]
