"""
check_phase3.py - tests the dashboard WITHOUT opening a browser:

    python check_phase3.py

1. Loads every page of the Streamlit app with Streamlit's built-in test harness and fails
   if any page raises an exception.
2. Tests the upload code paths (uploaded files are in-memory objects, not paths).
3. Builds every chart once.
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import data_loader as dl
import genotype_analysis as ga
import phenotype_analysis as pa
import preprocessing as pp
import visualization as viz

ROOT = Path(__file__).parent

print("== 1. App pages ==")
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
at.run()
assert not at.exception, at.exception
PAGES = list(at.sidebar.radio[0].options)
assert len(PAGES) == 8

for page in PAGES:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    print(f"  OK  {page}  (metrics: {len(at.metric)}, tables: {len(at.dataframe)})")

# gene page with each scope option
at.sidebar.radio[0].set_value(PAGES[3]).run()
for scope in ["Records with AST data", "Name contains...", "All records"]:
    at.radio[0].set_value(scope).run()
    assert not at.exception, f"scope {scope}: {[e.value for e in at.exception]}"
    print(f"  OK  gene page scope = {scope}")

print("\n== 2. Upload paths (in-memory files) ==")
raw = (ROOT / "data" / "raw")
geno, _ = dl.load_genotype_csv(io.BytesIO((raw / "isolates.csv").read_bytes()))
assert len(geno) == 1529
ast, w = dl.load_ast_table(io.BytesIO((raw / "biosample_metadata.txt").read_bytes()), biosample="SAMN47742561")
assert len(ast) == 9
try:
    dl.load_ast_table(io.BytesIO((raw / "biosample_metadata.txt").read_bytes()))
    raise SystemExit("AST without a BioSample should have been rejected")
except dl.DataValidationError as e:
    print("  OK  AST without BioSample rejected:", str(e)[:60], "...")
for bad, fn, label in [(b"a,b\n1,2\n", dl.load_genotype_csv, "genotype missing columns"),
                       (b"", dl.load_genotype_csv, "empty file"),
                       (b"Antibiotic\tx\nfoo\tbar\n", lambda f: dl.load_ast_table(f, "SAMN1"), "AST missing columns")]:
    try:
        fn(io.BytesIO(bad))
        raise SystemExit(f"{label} should have been rejected")
    except dl.DataValidationError:
        print(f"  OK  rejected: {label}")
try:
    dl.genome_stats_from_fasta(io.BytesIO(b">x\nACGT123\n"))
    raise SystemExit("bad FASTA should have been rejected")
except dl.DataValidationError:
    print("  OK  rejected: FASTA with non-nucleotide characters")

# whole NCBI package as a zip, built in memory from the project's genome folder
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in (ROOT / "data" / "genome").rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(ROOT / "data" / "genome"))
buf.seek(0)
pk = dl.read_genome_package_zip(buf)
summ, contigs = dl.genome_stats_from_fasta(pk["fasta"][0][1])
assert summ["n_contigs"] == 63 and len(pk["assembly_report"]) == 1
print("  OK  genome .zip read in memory:", pk["fasta"][0][0])

# integration with NO AST must not crash
res0 = pp.integrate(geno, pp.normalize_ast(pd.DataFrame(columns=dl.AST_REQUIRED + ["BioSample"])), None)
assert res0.summary["n_matched_records"] == 0 and res0.summary["n_genotype_only_records"] == 1529
print("  OK  integration works with no AST data")

print("\n== 3. Charts ==")
ast_clean = pp.normalize_ast(ast)
res = pp.integrate(geno, ast_clean, dl.metadata_from_reports(dl.load_assembly_reports(
    ROOT / "data" / "genome" / "assembly_data_report.jsonl")))
lg = res.genotype_long
freq = ga.gene_frequency(lg, 1529)
figs = {
    "status_bar": viz.status_bar(ga.status_counts(lg)),
    "gene_frequency_bar": viz.gene_frequency_bar(freq, 15),
    "point_mutation_bar": viz.point_mutation_bar(ga.point_mutations(lg, 1529)),
    "matrix_heatmap": viz.matrix_heatmap(ga.isolate_gene_matrix(lg, ["PDT003377204.1"], top_n=12)),
    "genes_per_isolate_hist": viz.genes_per_isolate_hist(ga.genes_per_isolate(lg)),
    "phenotype_bar": viz.phenotype_bar(pa.phenotype_counts(ast_clean)),
    "zone_diameter_chart": viz.zone_diameter_chart(pa.zone_diameters(ast_clean)),
    "integration_bar": viz.integration_bar(res.summary),
    "missing_bar": viz.missing_bar(pp.missing_data_report(res.integrated_all)),
}
for name, fig in figs.items():
    assert len(fig.data) > 0, name
    print(f"  OK  {name}")
mfig = viz.contig_length_figure(contigs, summ["n50"])
assert mfig is not None
print("  OK  contig_length_figure (Matplotlib)")
print("\nALL PHASE 3 CHECKS PASSED")
