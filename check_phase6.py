"""
check_phase6.py - verifies the 4-isolate antibiogram data and everything that depends on it:

    python check_phase6.py
"""
import subprocess
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import data_loader as dl
import genotype_analysis as ga
import model as mdl
import phenotype_analysis as pa
import preprocessing as pp
import report_generator as rg

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)

# --- the combined file must be an exact copy of the four source files -------------------------
src = pd.read_csv(RAW / "antibiogram_sources.csv", dtype=str)
combined, notes = dl.load_ast_table(RAW / "antibiograms_combined.csv")
assert len(combined) == 36 and combined["BioSample"].nunique() == 4
for s in src.itertuples():
    orig, _ = dl.load_ast_table(RAW / "antibiograms" / s.file, biosample=s.biosample)
    part = combined[combined["BioSample"] == s.biosample].reset_index(drop=True)
    pd.testing.assert_frame_equal(orig, part, check_dtype=False)
print("OK  combined table equals the four source files, value for value")

geno, _ = dl.load_genotype_csv(RAW / "isolates.csv")
ast_clean = pp.normalize_ast(combined)
res = pp.integrate(geno, ast_clean, None)
print({k: v for k, v in res.summary.items() if "match" in k or "ast" in k})
assert res.summary["n_matched_biosamples"] == 4 and res.summary["n_ast_only_biosamples"] == 0
assert res.summary["n_genotype_only_records"] == 1525
names = dict(zip(res.integrated_matched.biosample, res.integrated_matched.isolate_name))
assert names == {"SAMN47742561": "Nivedi_IN131", "SAMN47742560": "Nivedi_IN237",
                 "SAMN47742564": "Nivedi_IN183", "SAMN47742563": "Nivedi_IN184"}, names

# --- exact values preserved (signs, intermediate '==') -----------------------------------------
def val(bs, ab):
    r = ast_clean[(ast_clean.biosample == bs) & (ast_clean.antibiotic == ab)].iloc[0]
    return r.phenotype, r.measurement_reported
assert val("SAMN47742560", "enrofloxacin") == ("intermediate", "==16 mm")
assert val("SAMN47742564", "linezolid") == ("resistant", "<=6 mm")
assert val("SAMN47742563", "erythromycin") == ("susceptible", ">=25 mm")
assert val("SAMN47742564", "erythromycin") == ("intermediate", "==22 mm")
counts = pa.phenotype_counts(ast_clean).set_index("phenotype")["n_results"].to_dict()
print("phenotype counts:", counts)
assert counts == {"resistant": 9, "intermediate": 3, "susceptible": 24}, counts
assert pa.measurement_kind("==") == "exact"

# --- label readiness: still far below the training threshold ------------------------------------
tt = mdl.trainability_table(res.ast_long_matched)
print(tt[["antibiotic", "n_independent_isolates", "n_resistant", "n_susceptible", "trainable"]].to_string(index=False))
exp = {"cefoxitin": 4, "chloramphenicol": 4, "enrofloxacin": 3, "erythromycin": 2, "gentamicin": 4,
       "linezolid": 4, "penicillin": 4, "tetracycline": 4, "trimethoprim-sulfamethoxazole": 4}
# enrofloxacin (IN237) and erythromycin (IN237, IN183) have intermediate results, which are
# deliberately excluded from binary resistant/susceptible training labels
assert dict(zip(tt.antibiotic, tt.n_independent_isolates)) == exp
assert not tt["trainable"].any()
try:
    mdl.train_random_forest(res.genotype_long, res.ast_long_matched, "cefoxitin")
    raise SystemExit("training must still be refused with 4 labelled isolates")
except mdl.ModelNotTrainable:
    print("OK  training still refused (4 labelled isolates < 50)")

# --- rule-based readings for all four isolates (descriptive; compare by eye) ----------------------
for bs, nm in names.items():
    rid = res.integrated_matched.set_index("biosample").loc[bs, "record_id"]
    summ = mdl.isolate_marker_summary(ga.isolate_profile(res.genotype_long, rid), ast_clean[ast_clean.biosample == bs])
    print(f"\n{nm}")
    print(summ[["antibiotic", "lab_result", "primary_markers"]].to_string(index=False))
    assert len(summ) == 9

# --- dashboard + PDF ---------------------------------------------------------------------------------
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
at.run()
assert not at.exception, at.exception
m = {x.label: x.value for x in at.metric}
assert m["AST records (results)"] == "36" and m["Isolates with AST"] == "4", m
PAGES = list(at.sidebar.radio[0].options)
for page in PAGES:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
at.sidebar.radio[0].set_value("5. Antibiotic Susceptibility").run()
for i in range(len(at.selectbox[0].options)):      # every isolate's AST page
    at.selectbox[0].select_index(i).run()
    assert not at.exception, i
at.sidebar.radio[0].set_value("6. Prediction Demonstration").run()
n_iso = len(at.selectbox[0].options)
assert n_iso == 4, n_iso
for i in range(n_iso):                             # every isolate on the prediction page
    at.selectbox[0].select_index(i).run()
    tested = [j for j, a in enumerate(at.selectbox[1].options) if "lab result available" in a]
    assert len(tested) == 9, len(tested)
    for j in tested:
        at.selectbox[1].select_index(j).run()
        assert not at.exception, (i, j)
print(f"\nOK  all pages, and prediction page for 4 isolates x 9 antibiotics")
at.sidebar.radio[0].set_value("7. Machine Learning").run()
assert any("Training is disabled for every antibiotic" in e.value for e in at.error)

fasta = next((ROOT / "data" / "genome").rglob("GCA_*.fna"))
gs, contigs = dl.genome_stats_from_fasta(fasta)
reports = dl.load_assembly_reports(ROOT / "data" / "genome" / "assembly_data_report.jsonl")
rep = next(r for r in reports if r["accession"].startswith("GCA_"))
genome = {"summary": gs, "contigs": contigs, "report": rep, "reports": reports,
          "crosscheck": dl.crosscheck_genome(gs, rep), "fasta_name": fasta.name}
meta = dl.metadata_from_reports(reports)
res_m = pp.integrate(geno, ast_clean, meta)
pdf = rg.build_pdf_report(res_m, ast_clean, genome, {"ast": "data/raw/antibiograms_combined.csv"}, tt)
out = ROOT.parent / "amr_check_report_4isolates.pdf"
out.write_bytes(pdf)
txt = subprocess.run(["pdftotext", "-layout", str(out), "-"], capture_output=True, text=True).stdout
for needle in ["Nivedi_IN237", "Nivedi_IN183", "Nivedi_IN184", "==16 mm", "<=6 mm", "NO MODEL WAS TRAINED"]:
    assert needle in txt, needle
print(f"OK  PDF with 4 isolates: {len(pdf) / 1024:.0f} KB -> {out}")
print("\nALL PHASE 6 CHECKS PASSED")
