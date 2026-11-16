"""
report_generator.py
===================
Builds the downloadable PDF report with ReportLab. Everything in the report comes from the
loaded data; the only fixed text is explanatory wording and the list of limitations.

    pdf_bytes = build_pdf_report(res, ast_clean, genome, sources, trainability_df)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import genotype_analysis as ga
import model as mdl
import phenotype_analysis as pa
import preprocessing as pp
import visualization as viz

DISCLAIMER = ("Academic proof-of-concept prototype. NOT a clinically validated diagnostic system. "
              "Nothing in this report should be used to guide treatment.")

LIMITATIONS = [
    "Only a handful of isolates have real laboratory susceptibility results in this project. That is far too "
    "few to train or validate a supervised model, so no machine-learning performance (accuracy, precision, "
    "recall, F1) is reported. The Random Forest pipeline exists but is disabled by a guard until enough "
    "independent labelled isolates are supplied.",
    "The prediction demonstration is RULE-BASED, not machine learning. Its gene-to-antibiotic table is "
    "illustrative textbook-level knowledge and has not been validated against CARD, ResFinder or CLSI. "
    "Comparisons with the laboratory result describe a single isolate and are not a measure of accuracy.",
    "Gene and mutation calls are read from NCBI Pathogen Detection (AMRFinderPlus-based). This tool does not "
    "re-detect resistance genes from the genome FASTA; it only computes assembly statistics from it.",
    "A gene that is not listed is 'not reported', never 'confirmed absent'. Partial, end-of-contig, HMM and "
    "mistranslated calls are kept separate and treated as uncertain evidence. The genome is a fragmented draft "
    "assembly, so partial calls can result from assembly breaks.",
    "AST phenotypes are used exactly as reported (disk diffusion, as stated in the source). They are not "
    "re-interpreted against breakpoints. Values with <= or >= signs are bounds, not exact zone diameters.",
    "The genotype file is a convenience collection from NCBI, not a random or representative sample of any "
    "population. Gene frequencies must not be generalised to dairy cattle or to India.",
    "Some BioSamples appear in more than one genotype record, and isolates from the same herd, village or clone "
    "are not independent. Independence between isolates cannot be assessed from these files.",
    "Sample metadata are as submitted to the NCBI BioSample database and were not independently checked.",
    "This is not a clinical decision-support tool and makes no claim of clinical reliability.",
]

NAVY = colors.HexColor("#0b3c5d")
BLUE = colors.HexColor("#1f6fb2")
LIGHT = colors.HexColor("#eaf4fc")
GRID = colors.HexColor("#cfe3f5")

_ss = getSampleStyleSheet()
STY = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=20,
                            textColor=NAVY, alignment=0, spaceAfter=4),
    "sub": ParagraphStyle("s", parent=_ss["Normal"], fontSize=10, textColor=colors.HexColor("#35526b"),
                          spaceAfter=8),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                         textColor=NAVY, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                         textColor=BLUE, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("b", parent=_ss["Normal"], fontSize=9.5, leading=13, spaceAfter=5),
    "small": ParagraphStyle("sm", parent=_ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#444444")),
    "cell": ParagraphStyle("c", parent=_ss["Normal"], fontSize=8, leading=10),
    "cellh": ParagraphStyle("ch", parent=_ss["Normal"], fontSize=8, leading=10, fontName="Helvetica-Bold",
                            textColor=colors.white),
    "warn": ParagraphStyle("w", parent=_ss["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#7a3b00")),
}


def P(text, style="body") -> Paragraph:
    return Paragraph(escape(str(text)), STY[style])


def _cell(v, header=False) -> Paragraph:
    s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    return Paragraph(escape(s), STY["cellh" if header else "cell"])


def table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    data = [[_cell(c, header and i == 0) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(style))
    return t


def kv_table(pairs: list[tuple], widths=(5.5 * cm, 11.5 * cm)) -> Table:
    t = table([[k, v] for k, v in pairs], list(widths), header=False)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT)]))
    return t


def box(text: str, style="warn", bg="#fff4e0", border="#f2b134") -> Table:
    t = Table([[Paragraph(escape(text), STY[style])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def fig_image(fig, width_cm: float = 16.5) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    w, h = fig.get_size_inches()
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "Academic prototype - not clinically validated. S. aureus AMR prototype.")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(GRID)
    canvas.line(1.8 * cm, 1.5 * cm, A4[0] - 1.8 * cm, 1.5 * cm)
    canvas.restoreState()


def _fmt(n) -> str:
    return f"{int(n):,}"


def build_pdf_report(res, ast_clean: pd.DataFrame, genome: dict | None, sources: dict,
                     trainability: pd.DataFrame, model_results: dict | None = None,
                     generated_at: datetime | None = None) -> bytes:
    """Return the report as PDF bytes."""
    generated_at = generated_at or datetime.now(timezone.utc)
    summ = res.summary
    long_df = res.genotype_long
    n_rec = summ["n_genotype_records"]
    gsum = ga.genotype_summary(long_df, n_rec)
    matched = res.integrated_matched
    story = []

    # ------------------------------------------------------------------ title
    story += [P("AMR Prototype Report: Staphylococcus aureus", "title"),
              P(f"Genotype, phenotype and genome analysis | generated {generated_at:%d %B %Y, %H:%M} UTC", "sub"),
              box(DISCLAIMER), Spacer(1, 6)]

    # ------------------------------------------------------------------ 1 overview
    story += [P("1. Project overview", "h1"),
              P("This prototype places AMR gene and mutation calls from NCBI Pathogen Detection next to laboratory "
                "antibiotic susceptibility results (NCBI BioSample antibiogram) and genome-assembly information for the "
                "same isolates, matched by BioSample accession. It demonstrates a machine-learning workflow, but "
                "trains a model only when enough genuine, independent labelled isolates exist.")]
    src_rows = [["Input", "Source used"]] + [[k, v] for k, v in sources.items()]
    story += [table(src_rows, [4 * cm, 13 * cm]), Spacer(1, 4)]

    # ------------------------------------------------------------------ 2 dataset summary
    story += [P("2. Dataset summary", "h1")]
    story += [kv_table([
        ("Genotype records", _fmt(n_rec)),
        ("Unique BioSamples", _fmt(summ["n_unique_biosamples"])),
        ("Records sharing a BioSample", _fmt(summ["n_records_with_duplicate_biosample"])),
        ("Gene calls (all statuses)", _fmt(gsum["n_gene_calls"])),
        ("Distinct AMR genes / point mutations",
         f"{_fmt(gsum['n_distinct_genes'])} genes / {_fmt(gsum['n_distinct_point_mutations'])} point mutations"),
        ("AST results (rows)", _fmt(summ["n_ast_rows"])),
        ("Isolates with AST data", _fmt(summ["n_ast_biosamples"])),
        ("Matched (genotype + AST)", f"{_fmt(summ['n_matched_records'])} record(s)"),
        ("Genotype only (no AST; not assumed susceptible)", _fmt(summ["n_genotype_only_records"])),
        ("AST only (BioSample not in genotype table)", _fmt(summ["n_ast_only_biosamples"])),
    ])]

    if not matched.empty:
        for n_row, (_, row) in enumerate(matched.head(5).iterrows()):
            meta = [(c[5:].replace("_", " "), row[c]) for c in matched.columns
                    if c.startswith("meta_") and pd.notna(row[c])]
            pairs = [("Isolate", f"{row['isolate_name']} ({row['record_id']})"), ("BioSample", row["biosample"]),
                     ("Genome assembly", row.get("Assembly") if pd.notna(row.get("Assembly")) else "n/a")] + meta
            head = [P("Isolates with laboratory AST data", "h2")] if n_row == 0 else []
            story += [KeepTogether(head + [kv_table(pairs), Spacer(1, 4)])]

    if genome and genome.get("summary"):
        gs, rep = genome["summary"], genome.get("report") or {}
        story += [P("Genome assembly", "h2"), kv_table([
            ("Assembly", f"{rep.get('accession', genome.get('fasta_name'))} ({rep.get('assembly_name', 'n/a')})"),
            ("Genome size", f"{_fmt(gs['total_length'])} bp ({gs['total_length'] / 1e6:.2f} Mb)"),
            ("Contigs / N50 / L50", f"{_fmt(gs['n_contigs'])} / {_fmt(gs['n50'])} bp / {gs['l50']}"),
            ("Largest / smallest contig", f"{_fmt(gs['largest_contig'])} bp / {_fmt(gs['smallest_contig'])} bp"),
            ("GC content", f"{gs['gc_percent']:.2f}% (ambiguous bases N: {_fmt(gs['n_count'])})"),
            ("Statistics computed from FASTA", "Cross-checked against the NCBI assembly report: " +
             ("all values match." if genome.get("crosscheck") and all("OK" in m for m in genome["crosscheck"])
              else "see the dashboard for details.")),
        ])]
        if genome.get("contigs") is not None:
            story += [Spacer(1, 4), fig_image(viz.contig_length_figure(genome["contigs"], gs["n50"]), 15)]
    else:
        story += [P("No genome FASTA was loaded.", "body")]

    # ------------------------------------------------------------------ 3 AMR genes
    story += [PageBreak(), P("3. AMR gene analysis", "h1"),
              P(f"Summary over all {_fmt(n_rec)} genotype records. Detection statuses are kept separate. "
                "A gene that is not listed is 'not reported', not 'confirmed absent'.")]
    sc = ga.status_counts(long_df)
    story += [fig_image(viz.status_counts_figure_mpl(sc), 14)]
    story += [table([["Status", "Gene calls", "Meaning"]] +
                    [[r.status, _fmt(r.n_calls), r.meaning] for r in sc.itertuples()],
                    [4 * cm, 2.3 * cm, 10.7 * cm])]
    freq = ga.gene_frequency(long_df, n_rec)
    story += [Spacer(1, 6), fig_image(viz.gene_frequency_figure_mpl(freq, 12), 14)]
    story += [P("Most frequent genes and mutations", "h2"),
              table([["Gene / mutation", "Type", "Records", "% of records", "Complete", "Partial*", "HMM", "Point"]] +
                    [[r.gene, r.call_type.replace("_", " "), _fmt(r.n_records_any), f"{r.pct_records:.1f}",
                      _fmt(r.n_COMPLETE), _fmt(r.n_PARTIAL + r.n_PARTIAL_END_OF_CONTIG), _fmt(r.n_HMM), _fmt(r.n_POINT)]
                     for r in freq.head(12).itertuples()],
                    [4 * cm, 2.4 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm, 1.6 * cm]),
              P("*Partial = PARTIAL + PARTIAL_END_OF_CONTIG. A record with several statuses for one gene counts in each column.", "small")]
    pm = ga.point_mutations(long_df, n_rec)
    if not pm.empty:
        story += [P("Most frequent point mutations", "h2"),
                  table([["Mutation", "Locus", "Records", "% of records"]] +
                        [[r.mutation, r.gene_locus, _fmt(r.n_records), f"{r.pct_records:.1f}"] for r in pm.head(8).itertuples()],
                        [5 * cm, 3 * cm, 3 * cm, 3 * cm])]
    for _, row in matched.head(5).iterrows():
        prof = ga.isolate_profile(long_df, row["record_id"])
        story += [P(f"Gene calls for {row['isolate_name']}", "h2"),
                  table([["Gene / mutation", "Status", "What the status means"]] +
                        [[r.gene, r.status, r.meaning] for r in prof.itertuples()], [3.8 * cm, 5 * cm, 8.2 * cm])]

    # ------------------------------------------------------------------ 4 antibiogram
    story += [PageBreak(), P("4. Antibiogram findings", "h1")]
    if ast_clean.empty:
        story += [P("No AST data were loaded.")]
    else:
        story += [P("Laboratory phenotypes are shown exactly as reported and are not re-interpreted. "
                    "A value with <= or >= is a bound, not an exact zone diameter.")]
        names = res.integrated_all.groupby("biosample")["isolate_name"].agg(lambda s: ", ".join(sorted(set(s)))).to_dict()
        for bs in sorted(ast_clean["biosample"].unique())[:5]:
            sub = ast_clean[ast_clean["biosample"] == bs]
            g = pa.antibiotics_by_phenotype(ast_clean, bs)
            story += [P(f"{names.get(bs, 'Isolate not in genotype table')} ({bs})", "h2"),
                      kv_table([("Resistant", ", ".join(g["resistant"]) or "none"),
                                ("Intermediate", ", ".join(g["intermediate"]) or "none reported"),
                                ("Susceptible", ", ".join(g["susceptible"]) or "none"),
                                ("Method / standard", f"{', '.join(sorted(sub['method'].dropna().unique()))} / "
                                                      f"{', '.join(sorted(sub['standard'].dropna().unique()))}")]),
                      Spacer(1, 4),
                      table([["Antibiotic", "Phenotype", "Reported measurement", "What the sign means"]] +
                            [[r.antibiotic, r.phenotype, r.measurement_reported, r.measurement_meaning]
                             for r in pa.antibiogram_table(sub).itertuples()],
                            [4.8 * cm, 2.6 * cm, 3.4 * cm, 6.2 * cm])]

        # rule-based comparison
        rows_added = False
        for _, mrow in matched.head(5).iterrows():
            calls = ga.isolate_profile(long_df, mrow["record_id"])
            subast = ast_clean[ast_clean["biosample"] == mrow["biosample"]]
            summary = mdl.isolate_marker_summary(calls, subast)
            if not rows_added:
                story += [Spacer(1, 6), P("Marker evidence next to laboratory results (rule-based)", "h1"),
                          box(mdl.RULE_BASED_BANNER)]
                rows_added = True
            story += [P(mrow["isolate_name"], "h2"),
                      table([["Antibiotic", "Lab result", "Marker evidence", "Primary markers (status)"]] +
                            [[r.antibiotic, r.lab_result, r.marker_evidence, r.primary_markers]
                             for r in summary.itertuples()], [4.4 * cm, 2.1 * cm, 5.8 * cm, 4.7 * cm]),
                      P("Descriptive reading, one isolate: this is not an accuracy measure.", "small")]

    # ------------------------------------------------------------------ 5 model status
    story += [PageBreak(), P("5. Machine-learning model status", "h1")]
    trainable = trainability[trainability["trainable"]] if not trainability.empty else trainability
    if trainability.empty or trainable.empty:
        story += [box("NO MODEL WAS TRAINED. There are too few independent labelled isolates for a statistically "
                      "meaningful supervised model. Accuracy, precision, recall and F1 are deliberately not reported "
                      "because with so few isolates they would be meaningless and misleading.", bg="#fdecea", border="#c0392b")]
    else:
        story += [P("Some antibiotics have enough labelled isolates to train a model. Results appear below only if a "
                    "model was trained in the dashboard during this session.")]
    story += [Spacer(1, 4),
              P(f"Training guard: at least {mdl.MIN_ISOLATES_TOTAL} independent labelled isolates in total and "
                f"{mdl.MIN_ISOLATES_PER_CLASS} in each of the resistant and susceptible classes. Isolates are counted "
                "by BioSample, so records sharing a BioSample count once."),
              table([["Pipeline step", "Status"]] + [[a, b] for a, b in mdl.PIPELINE_STEPS], [9 * cm, 8 * cm])]
    if not trainability.empty:
        story += [P("Label readiness per antibiotic", "h2"),
                  table([["Antibiotic", "Independent isolates", "Resistant", "Susceptible", "Trainable", "Why not"]] +
                        [[r.antibiotic, _fmt(r.n_independent_isolates), _fmt(r.n_resistant), _fmt(r.n_susceptible),
                          "yes" if r.trainable else "no",
                          "-" if r.trainable else "Below the minimum number of isolates"] for r in trainability.itertuples()],
                        [4.4 * cm, 2.4 * cm, 1.9 * cm, 2.3 * cm, 1.7 * cm, 4.3 * cm])]
    for ab, mr in (model_results or {}).items():
        story += [P(f"Random Forest result: {ab}", "h2"),
                  kv_table([("Train / test isolates", f"{mr.n_train} / {mr.n_test}"),
                            ("Precision / recall / F1 (resistant)", f"{mr.precision:.2f} / {mr.recall:.2f} / {mr.f1:.2f}"),
                            ("Accuracy", f"{mr.accuracy:.2f}")]),
                  table([[""] + list(mr.confusion.columns)] +
                        [[idx] + [str(v) for v in r] for idx, r in zip(mr.confusion.index, mr.confusion.values)],
                        [5 * cm, 4.5 * cm, 4.5 * cm])]
        story += [P(w, "small") for w in mr.warnings]

    # ------------------------------------------------------------------ 6 limitations
    story += [P("6. Limitations", "h1")]
    story += [P(f"{i}. {t}") for i, t in enumerate(LIMITATIONS, start=1)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=2 * cm,
                            title="S. aureus AMR prototype report", author="AMR prototype")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
