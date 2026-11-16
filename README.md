# AI-Based Antimicrobial Resistance (AMR) Prototype - *Staphylococcus aureus*

A beginner-friendly Streamlit dashboard that places **AMR gene calls** (NCBI Pathogen Detection),
**laboratory antibiotic susceptibility results** (NCBI BioSample antibiogram), **isolate metadata** and
**genome-assembly information** (NCBI Datasets) side by side, and demonstrates a machine-learning workflow.

> **Academic proof-of-concept. NOT a clinically validated diagnostic system.**
> Nothing here should be used to guide treatment.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10 or newer is recommended. The app opens in your browser. Your real data is already in the `data/`
folder, so every page works immediately.

## What the app does

| Page | What you get |
|---|---|
| 1. Home Dashboard | Counts of isolates, genes, mutations, AST records; sample metadata; genome statistics |
| 2. Data Upload | Upload genotype CSV, AST file, metadata CSV, genome FASTA or the whole NCBI `.zip`; every file is validated |
| 3. Data Integration | Matching by BioSample, matched / genotype-only / AST-only tables, missing-data report, CSV downloads |
| 4. AMR Gene Analysis | Complete vs partial calls, gene frequency, point mutations, isolate-by-gene status heatmap |
| 5. Antibiotic Susceptibility | Resistant / intermediate / susceptible lists, antibiogram, zone diameters with signs preserved |
| 6. Prediction Demonstration | Pick an isolate and antibiotic: gene evidence, laboratory result, marker explanation (**rule-based**) |
| 7. Machine Learning | Pipeline, a guard that decides if training is meaningful, Random Forest if (and only if) it is |
| 8. Report | Downloadable PDF report |

## Project files

| File | Role |
|---|---|
| `app.py` | Streamlit interface (sidebar navigation, 8 pages) |
| `data_loader.py` | Reads and validates CSV/TSV/FASTA/JSONL/ZIP inputs; computes genome statistics from the FASTA |
| `preprocessing.py` | Parses gene calls, normalises AST, integrates everything by BioSample |
| `genotype_analysis.py` | Gene frequencies, status counts, point mutations, isolate profiles |
| `phenotype_analysis.py` | Antibiogram tables, phenotype counts, label availability |
| `model.py` | Rule-based marker demonstration **and** the guarded Random Forest pipeline |
| `visualization.py` | Plotly (interactive) and Matplotlib (PDF) charts |
| `report_generator.py` | PDF report (ReportLab) |
| `build_antibiogram_table.py` | Combines the per-isolate antibiograms into one table with a BioSample column |
| `check_phase1.py` ... `check_phase6.py` | Test scripts that verify the code against your real data |
| `.streamlit/config.toml` | White / light-blue theme |

## Data layout

```
data/
  raw/isolates.csv                     NCBI Pathogen Detection isolates table (genotypes)
  raw/antibiograms/*.tsv               One antibiogram per isolate, copied from its BioSample page
  raw/antibiogram_sources.csv          Which file belongs to which BioSample (and how that was identified)
  raw/antibiograms_combined.csv        Built from the two items above - this is what the app loads
  raw/biosample_metadata.txt           Antibiogram of Nivedi_IN131 (kept for the test scripts)
  genome/GCA_061100565.1/*.fna         Genome FASTA of Nivedi_IN131 (63 contigs)
  genome/assembly_data_report.jsonl, sequence_report.tsv, data_summary.tsv
```

Antibiograms currently loaded (4 isolates, 9 antibiotics each, disk diffusion, CLSI):

| Isolate | BioSample | How the file was matched to the BioSample |
|---|---|---|
| Nivedi_IN131 | SAMN47742561 | confirmed by the project owner |
| Nivedi_IN237 | SAMN47742560 | file name `IN237.txt`; confirmed by project owner |
| Nivedi_IN183 | SAMN47742564 | file name `biosample564.txt`; confirmed by project owner |
| Nivedi_IN184 | SAMN47742563 | file name `IN184.txt`; confirmed by project owner |

**To add an isolate:** save its antibiogram table (as copied from BioSample) in `data/raw/antibiograms/`,
add a row to `data/raw/antibiogram_sources.csv`, then run `python build_antibiogram_table.py`.
Alternatively, upload a combined CSV with a `BioSample` column on the **Data Upload** page. Required columns:
`Antibiotic, Resistance phenotype, Measurement sign, Measurement, Measurement units,
Laboratory typing method, Testing standard`. A file without a `BioSample` column needs the accession typed in.

Sample metadata (host, location, dates) currently exists only for Nivedi_IN131, because it comes from the
genome package's BioSample attributes. Other isolates show "no metadata" until you upload a metadata CSV.

## Scientific rules the code follows

1. No data are invented. AST labels are only those you supplied.
2. A gene that is not listed is **"not reported"**, never "confirmed absent".
3. Detection statuses stay separate: `COMPLETE`, `PARTIAL`, `PARTIAL_END_OF_CONTIG`, `HMM`, `POINT`,
   `MISTRANSLATION`. Only COMPLETE and POINT count as confirmed evidence; the rest are "uncertain".
4. Genotype and AST are matched **only by BioSample**. Unmatched isolates stay in separate tables and are
   **not assumed susceptible**.
5. AST phenotypes and measurement signs are kept exactly as supplied. `>=24 mm` is a bound, not an exact diameter.
6. Rule-based output is always labelled as such and is never presented as an ML prediction.
7. No model is trained unless the guard passes (see below). Otherwise no accuracy, precision, recall or F1 is shown.

## When will machine learning switch on?

`model.py` refuses to train unless an antibiotic has at least **50 independent labelled isolates** and at least
**15 in each class** (resistant and susceptible). Isolates are counted by BioSample, so duplicate records count once.
The thresholds are `MIN_ISOLATES_TOTAL` and `MIN_ISOLATES_PER_CLASS` at the top of `model.py`; they are rules of
thumb, not a statistical guarantee. With the current data (at most 4 labelled isolates per antibiotic) training is correctly disabled.
Intermediate results are excluded from the binary resistant/susceptible labels.

## Limitations

- Only four isolates have laboratory AST results (and only 2-4 labelled isolates per antibiotic), far below the
  minimum for training or validating a supervised model.
- The marker table in `model.py` is **illustrative, textbook-level knowledge written by the prototype author**.
  It has **not** been validated against CARD, ResFinder or CLSI. Check it before relying on it.
- Gene and mutation calls come from NCBI Pathogen Detection. The app does **not** re-detect genes from the FASTA;
  it only computes assembly statistics from it.
- The genome is a fragmented draft (63 contigs), so partial calls can be caused by assembly breaks.
- The genotype file is a convenience collection, not a representative sample. Do not generalise its gene
  frequencies to dairy cattle or to India.
- Some BioSamples appear in several records, and isolates from the same herd or clone are not independent.
- Phenotypes are not re-interpreted against CLSI breakpoints.
- Metadata are as submitted to NCBI and were not independently checked.

## Verifying your installation

```bash
python check_phase1.py      # loaders + integration
python check_phase2.py      # gene and phenotype analysis
python check_phase3.py      # dashboard pages, uploads, charts
python check_phase4_5.py    # rule-based markers, training guard, PDF, pages 6-8
python check_phase6.py      # the 4-isolate antibiogram data, end to end
```

Each ends with `ALL ... CHECKS PASSED`. `check_phase4_5.py` uses a synthetic toy dataset **only inside the test**
to prove the training code works when enough isolates exist; that data never reaches the app or the report.
It also needs the `pdftotext` command (part of poppler) to inspect the PDF; if you do not have it, that one
assertion will fail but the app itself is unaffected.

## Data sources

NCBI Pathogen Detection Isolates Browser, NCBI BioSample (antibiogram), NCBI Datasets (genome package for
`GCA_061100565.1`, BioProject PRJNA1245146).
