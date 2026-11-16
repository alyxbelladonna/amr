"""
check_phase1.py - run this to verify Phase 1 against YOUR real data:

    python check_phase1.py

It prints what was loaded and stops with an AssertionError if something that
should be true of your data is not.
"""
from pathlib import Path

import pandas as pd

import data_loader as dl
import preprocessing as pp

ROOT = Path(__file__).parent
GENOTYPE = ROOT / "data" / "raw" / "isolates.csv"
AST = ROOT / "data" / "raw" / "biosample_metadata.txt"
GENOME_DIR = ROOT / "data" / "genome"
TARGET_BIOSAMPLE = "SAMN47742561"   # Nivedi_IN131

pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)

print("== 1. Genotype table ==")
geno, w = dl.load_genotype_csv(GENOTYPE)
print(f"rows={len(geno)}  columns={len(geno.columns)}")
for m in w:
    print("  warning:", m)

print("\n== 2. AST table ==")
ast, w = dl.load_ast_table(AST, biosample=TARGET_BIOSAMPLE)
for m in w:
    print("  note:", m)
ast_clean = pp.normalize_ast(ast)
print(ast_clean[["biosample", "antibiotic", "phenotype", "measurement_sign",
                 "measurement", "measurement_units", "method", "standard"]].to_string(index=False))

print("\n== 3. Genome package ==")
files = dl.find_genome_package_files(GENOME_DIR)
for k, v in files.items():
    print(f"  {k}: {[p.name for p in v]}")
fasta = [p for p in files["fasta"] if p.name.startswith("GCA_")][0]
fasta_summary, contigs = dl.genome_stats_from_fasta(fasta)
print(" ", fasta_summary)
reports = dl.load_assembly_reports(files["assembly_report"][0])
report = next(r for r in reports if r["accession"].startswith("GCA_"))
for m in dl.crosscheck_genome(fasta_summary, report):
    print("  ", m)
metadata = dl.metadata_from_reports(reports)
print(metadata.T.to_string(header=False))

print("\n== 4. Integration ==")
res = pp.integrate(geno, ast_clean, metadata)
for k, v in res.summary.items():
    print(f"  {k}: {v}")

print("\n== 5. Nivedi_IN131 combined record ==")
row = res.integrated_matched.iloc[0]
for col in ["record_id", "biosample", "isolate_name", "Assembly", "genes_COMPLETE",
            "genes_PARTIAL_END_OF_CONTIG", "genes_HMM", "genes_POINT",
            "ast_resistant", "ast_intermediate", "ast_susceptible",
            "meta_collection_date", "meta_host", "meta_host_disease", "meta_geo_loc_name"]:
    print(f"  {col}: {row.get(col)}")

print("\n== 6. Missing data (genotype+metadata table, top 8) ==")
print(pp.missing_data_report(res.integrated_all).head(8).to_string(index=False))

# ------------------------------------------------------------------ assertions
assert len(geno) == 1529
assert res.summary["n_matched_biosamples"] == 1, "exactly one genotype BioSample should match AST"
assert res.summary["n_ast_only_biosamples"] == 0, "AST BioSample must exist in genotype table"
assert res.summary["n_genotype_only_records"] == 1528
assert set(res.integrated_matched["biosample"]) == {TARGET_BIOSAMPLE}
assert row["ast_resistant"] == "cefoxitin, gentamicin, penicillin"
assert row["ast_intermediate"] == ""  # empty: no intermediate results in this antibiogram
assert row["ast_susceptible"] == ("chloramphenicol, enrofloxacin, erythromycin, linezolid, "
                                  "tetracycline, trimethoprim-sulfamethoxazole")
cef = ast_clean[ast_clean.antibiotic == "cefoxitin"].iloc[0]
assert (cef.measurement_sign, cef.measurement) == ("<=", "18"), "sign/value must be preserved"
assert fasta_summary["n_contigs"] == 63 and fasta_summary["n50"] == 225471
assert fasta_summary["total_length"] == 2741350
long_in131 = res.genotype_long[res.genotype_long.biosample == TARGET_BIOSAMPLE]
status = dict(zip(long_in131.gene, long_in131.status))
assert status["mecA"] == "COMPLETE" and status["blaR1"] == "HMM"
assert status["ant(6)-Ia"] == "PARTIAL_END_OF_CONTIG" and status["tet(L)"] == "PARTIAL_END_OF_CONTIG"
print("\nALL PHASE 1 CHECKS PASSED")
