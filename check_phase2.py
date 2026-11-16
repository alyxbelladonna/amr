"""
check_phase2.py - verifies the analysis modules on YOUR real data:

    python check_phase2.py
"""
from pathlib import Path

import pandas as pd

import data_loader as dl
import genotype_analysis as ga
import phenotype_analysis as pa
import preprocessing as pp

ROOT = Path(__file__).parent
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70)

geno, _ = dl.load_genotype_csv(ROOT / "data" / "raw" / "isolates.csv")
ast, _ = dl.load_ast_table(ROOT / "data" / "raw" / "biosample_metadata.txt", biosample="SAMN47742561")
ast_clean = pp.normalize_ast(ast)
res = pp.integrate(geno, ast_clean)
long_df = res.genotype_long
n_rec = res.summary["n_genotype_records"]

print("== Genotype: ALL 1,529 records ==")
print(ga.genotype_summary(long_df, n_rec))
print("\nStatus counts:")
print(ga.status_counts(long_df)[["status", "n_calls", "group"]].to_string(index=False))
print("\nTop 12 genes/mutations:")
print(ga.gene_frequency(long_df, n_rec).head(12).to_string(index=False))
print("\nTop 6 point mutations:")
print(ga.point_mutations(long_df, n_rec).head(6).to_string(index=False))

print("\n== Genotype: Nivedi_IN131 only ==")
rid = "PDT003377204.1"
print(ga.genotype_summary(long_df[long_df.record_id == rid], 1))
print(ga.isolate_profile(long_df, rid)[["gene", "status", "call_type"]].to_string(index=False))

print("\n== Gene matrix: the 6 Nivedi isolates, 8 most common genes ==")
nivedi = res.integrated_all[res.integrated_all.isolate_name.str.startswith("Nivedi")].record_id.tolist()
print(ga.isolate_gene_matrix(long_df, nivedi, top_n=8).to_string())

print("\n== Phenotype ==")
print(pa.antibiogram_table(ast_clean).drop(columns=["biosample", "method", "standard"]).to_string(index=False))
print("\n", pa.phenotype_counts(ast_clean).to_string(index=False))
print("\n", pa.antibiotics_by_phenotype(ast_clean, "SAMN47742561"))
print("\nLabel availability (matched isolates):")
print(pa.label_availability(res.ast_long_matched).to_string(index=False))

# ------------------------------------------------------------------ assertions
sc = dict(zip(ga.status_counts(long_df).status, ga.status_counts(long_df).n_calls))
assert sc == {"COMPLETE": 10386, "PARTIAL": 547, "PARTIAL_END_OF_CONTIG": 301, "HMM": 343,
              "POINT": 3309, "MISTRANSLATION": 259}, sc
assert sum(sc.values()) == res.summary["n_gene_calls"] == 15145
gf = ga.gene_frequency(long_df, n_rec)
assert (gf["n_records_any"] <= n_rec).all() and (gf["pct_records"] <= 100).all()
assert (gf[[f"n_{s}" for s in pp.STATUS_ORDER]].sum(axis=1) >= gf["n_records_any"]).all()

s131 = ga.genotype_summary(long_df[long_df.record_id == rid], 1)
assert (s131["n_gene_calls"], s131["n_distinct_genes"], s131["n_distinct_point_mutations"]) == (9, 9, 0)
mat = ga.isolate_gene_matrix(long_df, [rid])
assert mat.iloc[0]["blaR1"] == "HMM" and mat.iloc[0]["mecA"] == "COMPLETE"

zd = pa.zone_diameters(ast_clean).set_index("antibiotic")
assert zd.loc["cefoxitin", "measurement_reported"] == "<=18 mm"
assert zd.loc["cefoxitin", "measurement_meaning"].startswith("upper bound")
assert zd.loc["tetracycline", "measurement_meaning"].startswith("lower bound")
assert pa.phenotype_counts(ast_clean).set_index("phenotype")["n_results"].to_dict() == {
    "resistant": 3, "intermediate": 0, "susceptible": 6}
la = pa.label_availability(res.ast_long_matched)
assert (la["n_isolates_labelled"] == 1).all(), "only one labelled isolate: ML must stay disabled"
print("\nALL PHASE 2 CHECKS PASSED")
