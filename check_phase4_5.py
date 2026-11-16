"""
check_phase4_5.py - verifies model.py, the PDF report and the new dashboard pages:

    python check_phase4_5.py

IMPORTANT: part 3 builds a SYNTHETIC toy dataset purely to prove that the training code path
works when enough labelled isolates exist. That toy data is created only inside this test
file; it never reaches the app, the report or your project data.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import data_loader as dl
import genotype_analysis as ga
import model as mdl
import phenotype_analysis as pa
import preprocessing as pp
import report_generator as rg

ROOT = Path(__file__).parent
pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 80)

geno, _ = dl.load_genotype_csv(ROOT / "data" / "raw" / "isolates.csv")
ast, _ = dl.load_ast_table(ROOT / "data" / "raw" / "biosample_metadata.txt", biosample="SAMN47742561")
ast_clean = pp.normalize_ast(ast)
genome_dir = ROOT / "data" / "genome"
reports = dl.load_assembly_reports(genome_dir / "assembly_data_report.jsonl")
res = pp.integrate(geno, ast_clean, dl.metadata_from_reports(reports))
long_df = res.genotype_long
RID = "PDT003377204.1"

print("== 1. Rule-based markers for Nivedi_IN131 (REAL data) ==")
calls = ga.isolate_profile(long_df, RID)
summary = mdl.isolate_marker_summary(calls, ast_clean)
print(summary[["antibiotic", "lab_result", "marker_evidence", "primary_markers"]].to_string(index=False))
out = dict(zip(summary.antibiotic, summary.marker_evidence))
prim = dict(zip(summary.antibiotic, summary.primary_markers))
assert out["cefoxitin"] == mdl.OUTCOME_TEXT["MARKER_REPORTED"] and "mecA (COMPLETE)" in prim["cefoxitin"]
assert "blaZ (COMPLETE)" in prim["penicillin"] and "aac(6')-Ie/aph(2'')-Ia (COMPLETE)" in prim["gentamicin"]
assert out["tetracycline"] == mdl.OUTCOME_TEXT["UNCERTAIN_EVIDENCE"], "tet(L) is end-of-contig: uncertain"
assert "tet(L) (PARTIAL_END_OF_CONTIG)" in prim["tetracycline"] and "tet(38)" not in prim["tetracycline"]
for ab in ["erythromycin", "enrofloxacin", "linezolid", "chloramphenicol", "trimethoprim-sulfamethoxazole"]:
    assert out[ab] == mdl.OUTCOME_TEXT["NO_MARKER_REPORTED"], ab
ev = mdl.evaluate_markers(calls, "oxacillin")           # alias
assert ev["outcome"] == "MARKER_REPORTED"
assert mdl.evaluate_markers(calls, "vancomycin")["outcome"] == "NO_RULES"
assert "NOT assumed susceptible" in mdl.compare_with_lab("NO_MARKER_REPORTED", None)
print("  OK")

print("\n== 2. Training guard on REAL data ==")
tt = mdl.trainability_table(res.ast_long_matched)
print(tt.to_string(index=False))
assert not tt["trainable"].any() and (tt["n_independent_isolates"] == 1).all()
try:
    mdl.train_random_forest(long_df, res.ast_long_matched, "cefoxitin")
    raise SystemExit("training must be refused with one labelled isolate")
except mdl.ModelNotTrainable as e:
    print("  OK  training refused:", str(e)[:90], "...")
X, none = mdl.build_feature_matrix(long_df, res.integrated_all.record_id.tolist())
print(f"  feature matrix for ALL records: {X.shape[0]} x {X.shape[1]}; records without calls: {len(none)}")
assert X.shape[0] == 1529 and (X.values <= 1).all()
assert X.loc[RID, "blaR1|uncertain"] == 1 and X.loc[RID, "blaR1|confirmed"] == 0, "HMM stays 'uncertain'"
assert X.loc[RID, "mecA|confirmed"] == 1

print("\n== 3. Training code path on SYNTHETIC TOY DATA (test only, never shown in app) ==")
rng = np.random.default_rng(0)
n = 120
rows, ast_rows = [], []
for i in range(n):
    resistant = i % 2 == 0
    bs, rid = f"SAMN9{i:07d}", f"TOY{i:04d}"
    genes = [("tet(38)", "COMPLETE")]
    if resistant:
        genes.append(("mecA", "COMPLETE" if rng.random() < 0.9 else "PARTIAL"))
    elif rng.random() < 0.1:
        genes.append(("mecA", "PARTIAL"))
    for g in ("blaZ", "mepA", "fosB"):
        if rng.random() < 0.5:
            genes.append((g, "COMPLETE"))
    for g, s in genes:
        rows.append((rid, bs, f"toy{i}", g, s, "gene"))
    ast_rows.append((rid, bs, "cefoxitin", "resistant" if resistant else "susceptible"))
toy_long = pd.DataFrame(rows, columns=["record_id", "biosample", "isolate_name", "gene", "status", "call_type"])
toy_ast = pd.DataFrame(ast_rows, columns=["record_id", "biosample", "antibiotic", "phenotype"])
assert mdl.assess_trainability(toy_ast, "cefoxitin")["trainable"]
mr = mdl.train_random_forest(toy_long, toy_ast, "cefoxitin")
print(f"  train={mr.n_train} test={mr.n_test} P={mr.precision:.2f} R={mr.recall:.2f} F1={mr.f1:.2f}")
print(mr.confusion.to_string())
assert mr.n_train + mr.n_test == n and mr.confusion.values.sum() == mr.n_test
assert set(mr.feature_importance.feature.head(3)) & {"mecA|confirmed", "mecA|uncertain"}
# duplicate BioSample records must count once
dup = pd.concat([toy_ast, toy_ast.assign(record_id=toy_ast.record_id + "_dup")])
assert mdl.assess_trainability(dup, "cefoxitin")["n_independent_isolates"] == n
# too few in one class -> refused
skew = toy_ast[(toy_ast.phenotype == "resistant") | (toy_ast.index % 20 == 1)]
assert not mdl.assess_trainability(skew, "cefoxitin")["trainable"]
print("  OK  trains when enough isolates exist; counts BioSamples once; refuses skewed classes")

print("\n== 4. PDF report (REAL data) ==")
trainability = mdl.trainability_table(res.ast_long_matched)
fasta = next(genome_dir.rglob("GCA_*.fna"))
gsum, contigs = dl.genome_stats_from_fasta(fasta)
rep = next(r for r in reports if r["accession"].startswith("GCA_"))
genome = {"summary": gsum, "contigs": contigs, "report": rep, "reports": reports,
          "crosscheck": dl.crosscheck_genome(gsum, rep), "fasta_name": fasta.name}
sources = {"genotype": "data/raw/isolates.csv", "ast": "data/raw/biosample_metadata.txt",
           "genome": "data/genome/"}
pdf = rg.build_pdf_report(res, ast_clean, genome, sources, trainability)
out = ROOT.parent / "amr_check_report.pdf"
out.write_bytes(pdf)
assert pdf[:4] == b"%PDF" and len(pdf) > 20_000
print(f"  OK  PDF built: {len(pdf) / 1024:.0f} KB -> {out}")
# no fabricated metrics when nothing was trained
import subprocess
txt = subprocess.run(["pdftotext", "-layout", str(out), "-"], capture_output=True, text=True).stdout
assert "NO MODEL WAS TRAINED" in txt and "Random Forest result" not in txt
assert "<=18 mm" in txt and "tet(L)" in txt and "Limitations" in txt
print("  OK  report says no model was trained; shows signs; lists limitations")

print("\n== 5. Dashboard pages 6-8 ==")
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.run()
assert not at.exception, at.exception
PAGES = list(at.sidebar.radio[0].options)
for page in PAGES:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    print("  OK ", page)
at.sidebar.radio[0].set_value("6. Prediction Demonstration").run()
assert any("NOT a machine-learning prediction" in e.value for e in at.error)
antibiotics = at.selectbox[1].options
for ab in [a for a in antibiotics if "lab result available" in a]:
    at.selectbox[1].select(ab).run()
    assert not at.exception, ab
print(f"  OK  prediction page works for all {sum('lab result' in a for a in antibiotics)} tested antibiotics")
at.selectbox[1].select(next(a for a in antibiotics if a.startswith("vancomycin") or a == antibiotics[0])).run()
assert not at.exception
at.radio[0].set_value("All records (search by name)").run()
assert not at.exception
at.text_input[0].set_value("FDA").run()
assert not at.exception and len(at.selectbox[0].options) > 1
print("  OK  prediction page: search over all records")
at.sidebar.radio[0].set_value("7. Machine Learning").run()
assert any("Training is disabled for every antibiotic" in e.value for e in at.error)
assert not at.metric or all("Precision" not in m.label for m in at.metric)
print("  OK  ML page: training disabled, no performance metrics shown")
at.sidebar.radio[0].set_value("8. Report").run()
next(b for b in at.button if b.label == "Generate PDF report").click().run()
assert not at.exception and not at.error, [e.value for e in at.error]
assert any("Report ready" in x.value for x in at.success)
print("  OK  report page generates the PDF")
print("\nALL PHASE 4 + 5 CHECKS PASSED")
