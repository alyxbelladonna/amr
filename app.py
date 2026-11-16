"""
app.py - Streamlit dashboard

Run with:   streamlit run app.py

ACADEMIC PROTOTYPE. Not a clinically validated diagnostic system.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

import data_loader as dl
import report_generator as rg
import genotype_analysis as ga
import model as mdl
import phenotype_analysis as pa
import preprocessing as pp
import visualization as viz

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_GENOME = ROOT / "data" / "genome"
DEFAULT_GENOTYPE = DATA_RAW / "isolates.csv"
DEFAULT_AST = DATA_RAW / "antibiograms_combined.csv"
# Built by build_antibiogram_table.py from the per-isolate antibiograms in data/raw/antibiograms/.
# data/raw/antibiogram_sources.csv records which BioSample each source file belongs to.

DISCLAIMER = ("**Academic proof-of-concept prototype - not a clinically validated diagnostic system.** "
              "Nothing shown here should be used to guide treatment.")

PAGES = ["1. Home Dashboard", "2. Data Upload", "3. Data Integration", "4. AMR Gene Analysis",
         "5. Antibiotic Susceptibility", "6. Prediction Demonstration", "7. Machine Learning",
         "8. Report"]

st.set_page_config(page_title="S. aureus AMR Prototype", page_icon="🧬", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetric"] {background:#eaf4fc; border:1px solid #cfe3f5; border-radius:10px;
                             padding:12px 16px;}
div[data-testid="stMetricLabel"] p {font-size:0.85rem; color:#35526b;}
h1, h2, h3 {color:#0b3c5d;}
section[data-testid="stSidebar"] {border-right:1px solid #cfe3f5;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- small UI helpers
def show_df(df: pd.DataFrame, index: bool = False, height: int | None = None) -> None:
    kw = {"hide_index": not index}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:                      # older Streamlit versions
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def cols_present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def meta_view(row: pd.Series) -> pd.DataFrame:
    items = [(c[5:].replace("_", " "), row[c]) for c in row.index
             if c.startswith("meta_") and pd.notna(row[c])]
    return pd.DataFrame(items, columns=["Field", "Value"])


# ----------------------------------------------------------------------------- genome block
def process_genome(fasta_sources, report_sources, seq_sources) -> dict:
    """Turn genome files (paths or uploaded buffers) into one dictionary of results."""
    block = {"summary": None, "contigs": None, "reports": [], "report": None, "crosscheck": [],
             "fasta_name": None, "sequence_report": None}
    for _, src in report_sources:
        block["reports"] += dl.load_assembly_reports(src)
    if fasta_sources:
        name, src = next((s for s in fasta_sources if s[0].startswith("GCA_")), fasta_sources[0])
        block["fasta_name"] = name
        block["summary"], block["contigs"] = dl.genome_stats_from_fasta(src)
    if block["reports"]:
        acc = re.match(r"(GC[AF]_\d+\.\d+)", block["fasta_name"] or "")
        block["report"] = next((r for r in block["reports"]
                                if acc and r["accession"] == acc.group(1)), block["reports"][0])
    if block["summary"] and block["report"]:
        block["crosscheck"] = dl.crosscheck_genome(block["summary"], block["report"])
    if seq_sources:
        block["sequence_report"] = dl.load_sequence_report_tsv(seq_sources[0][1])
    return block


@st.cache_data(show_spinner="Loading project data...")
def load_default_data() -> dict:
    """Load whatever exists in the data/ folder. Missing files are reported, never assumed."""
    out = {"geno": None, "ast": None, "genome": None, "notes": [], "sources": {}}
    if DEFAULT_GENOTYPE.is_file():
        try:
            out["geno"], w = dl.load_genotype_csv(DEFAULT_GENOTYPE)
            out["notes"] += w
            out["sources"]["geno"] = f"Project file: data/raw/{DEFAULT_GENOTYPE.name}"
        except dl.DataValidationError as e:
            out["notes"].append(str(e))
    else:
        out["notes"].append(f"Genotype file not found: {DEFAULT_GENOTYPE}")
    if DEFAULT_AST.is_file():
        try:
            out["ast"], w = dl.load_ast_table(DEFAULT_AST)
            out["notes"] += w
            out["sources"]["ast"] = f"Project file: data/raw/{DEFAULT_AST.name}"
        except dl.DataValidationError as e:
            out["notes"].append(str(e))
    else:
        out["notes"].append(f"AST file not found: {DEFAULT_AST}")
    if DATA_GENOME.is_dir():
        try:
            f = dl.find_genome_package_files(DATA_GENOME)
            out["genome"] = process_genome(
                [(p.name, p) for p in f["fasta"]],
                [(p.name, p) for p in f["assembly_report"]],
                [(p.name, p) for p in f["sequence_report"] if p.suffix == ".tsv"])
            out["sources"]["genome"] = "Project folder: data/genome/"
        except dl.DataValidationError as e:
            out["notes"].append(str(e))
    return out


@st.cache_data(show_spinner="Integrating datasets...")
def compute_integration(geno: pd.DataFrame, ast_raw: pd.DataFrame | None, metadata: pd.DataFrame | None):
    if ast_raw is None or ast_raw.empty:
        ast_clean = pp.normalize_ast(pd.DataFrame(columns=dl.AST_REQUIRED + ["BioSample"]))
    else:
        ast_clean = pp.normalize_ast(ast_raw)
    return pp.integrate(geno, ast_clean, metadata), ast_clean


def init_state() -> None:
    st.session_state.setdefault("uploads", {})
    st.session_state.setdefault("upload_sources", {})
    st.session_state.setdefault("upload_sig", {})
    st.session_state.setdefault("upload_msgs", {})
    st.session_state.setdefault("uploader_gen", 0)


def get_state() -> dict:
    """Combine project data with any uploads (an upload replaces its project counterpart)."""
    base = load_default_data()
    up = st.session_state["uploads"]
    geno = up.get("geno", base["geno"])
    ast = up.get("ast", base["ast"])
    genome = up.get("genome", base["genome"])
    sources = {**base["sources"], **st.session_state["upload_sources"]}
    metadata = up.get("metadata")
    if metadata is not None:
        sources["metadata"] = st.session_state["upload_sources"].get("metadata", "Uploaded")
    elif genome and genome["reports"]:
        metadata = dl.metadata_from_reports(genome["reports"])
        sources["metadata"] = "BioSample attributes in NCBI assembly_data_report.jsonl"
    S = {"geno": geno, "ast_raw": ast, "genome": genome, "metadata": metadata,
         "sources": sources, "notes": base["notes"], "res": None, "ast_clean": None}
    if geno is not None:
        S["res"], S["ast_clean"] = compute_integration(geno, ast, metadata)
    return S


def need_genotype(S: dict) -> bool:
    if S["res"] is None:
        st.warning("No genotype data is loaded. Add `isolates.csv` to `data/raw/` or use **2. Data Upload**.")
        return False
    return True


# ----------------------------------------------------------------------------- page 1
def page_home(S: dict) -> None:
    st.title("🧬 AI-Based Antimicrobial Resistance (AMR) Prototype")
    st.markdown("##### *Staphylococcus aureus* - genotype, phenotype and genome analysis")
    st.warning(DISCLAIMER)
    st.write(
        "This prototype shows how bacterial genetic information (AMR gene and mutation calls from NCBI "
        "Pathogen Detection) can be placed next to laboratory antibiotic susceptibility results and the "
        "genome assembly of the same isolate. Because only a very small number of isolates have real "
        "laboratory results, the prediction page uses a clearly labelled **rule-based** demonstration "
        "rather than a trained model.")
    if not need_genotype(S):
        return
    res, ast_clean, summary = S["res"], S["ast_clean"], S["res"].summary
    gsum = ga.genotype_summary(res.genotype_long, summary["n_genotype_records"])

    st.subheader("Dataset overview")
    c = st.columns(4)
    c[0].metric("Isolate records", f"{summary['n_genotype_records']:,}")
    c[1].metric("Unique BioSamples", f"{summary['n_unique_biosamples']:,}")
    c[2].metric("Distinct AMR genes", f"{gsum['n_distinct_genes']:,}")
    c[3].metric("Distinct point mutations", f"{gsum['n_distinct_point_mutations']:,}")
    c = st.columns(4)
    c[0].metric("AST records (results)", f"{summary['n_ast_rows']:,}")
    c[1].metric("Isolates with AST", f"{summary['n_matched_biosamples']:,}")
    c[2].metric("Antibiotics tested", f"{ast_clean['antibiotic'].nunique():,}")
    c[3].metric("Gene calls (all statuses)", f"{gsum['n_gene_calls']:,}")
    st.caption("Counts of AMR genes are over all isolates in the genotype file; only isolates "
               "with a matching BioSample have laboratory AST results.")

    st.subheader("Organism and sample information")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Organism:** *Staphylococcus aureus*  \n"
                    "**Gram stain:** positive coccus  \n"
                    "**Relevance:** common cause of bovine mastitis; methicillin-resistant strains "
                    "(MRSA) carry *mecA*.")
    with right:
        matched = res.integrated_matched
        if matched.empty:
            st.info("No isolate currently has both genotype and AST data.")
        else:
            labels = (matched["isolate_name"] + " (" + matched["biosample"] + ")").tolist()
            pick = st.selectbox("Isolate with AST data", labels) if len(labels) > 1 else labels[0]
            row = matched.iloc[labels.index(pick)]
            if len(labels) == 1:
                st.markdown(f"**Isolate with AST data:** {pick}")
            mv = meta_view(row)
            if mv.empty:
                st.caption("No BioSample metadata available for this isolate.")
            else:
                show_df(mv)

    st.subheader("Genome information")
    g = S["genome"]
    if not g or not g["summary"]:
        st.info("No genome FASTA loaded. Add a genome package under `data/genome/` or use "
                "**2. Data Upload**.")
    else:
        gs, rep = g["summary"], g["report"] or {}
        c = st.columns(5)
        c[0].metric("Assembly", rep.get("accession") or g["fasta_name"] or "n/a")
        c[1].metric("Genome size", f"{gs['total_length'] / 1e6:.2f} Mb")
        c[2].metric("Contigs", f"{gs['n_contigs']:,}")
        c[3].metric("N50", f"{gs['n50']:,} bp")
        c[4].metric("GC content", f"{gs['gc_percent']:.2f}%")
        st.caption(f"File: {g['fasta_name']} | largest contig {gs['largest_contig']:,} bp | smallest "
                   f"{gs['smallest_contig']:,} bp | ambiguous bases (N): {gs['n_count']:,}. "
                   "A fragmented draft assembly can split genes across contigs, which is one reason a "
                   "gene may be reported as partial.")
        with st.expander("Contig lengths and cross-check with the NCBI report"):
            fig = viz.contig_length_figure(g["contigs"], gs["n50"])
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            for m in g["crosscheck"]:
                (st.success if "OK" in m else st.warning)(m)
            if rep:
                st.caption(f"NCBI-reported: {rep.get('assembly_name')} | BioProject {rep.get('bioproject')} | "
                           f"BioSample {rep.get('biosample')}. NCBI rounds its reported GC% to the "
                           f"nearest 0.5, so it can differ slightly from the exact value above.")

    with st.expander("Data sources and loading notes"):
        for k, v in S["sources"].items():
            st.write(f"- **{k}**: {v}")
        for n in S["notes"]:
            st.write(f"- {n}")


# ----------------------------------------------------------------------------- page 2
def _store(key: str, value, source: str, sig, msgs: list[str]) -> None:
    st.session_state["uploads"][key] = value
    st.session_state["upload_sources"][key] = source
    st.session_state["upload_sig"][key] = sig
    st.session_state["upload_msgs"][key] = ("ok", msgs)


def _fail(key: str, sig, message: str) -> None:
    st.session_state["upload_sig"][key] = sig
    st.session_state["upload_msgs"][key] = ("error", [message])


def _show_msgs(key: str) -> None:
    kind, msgs = st.session_state["upload_msgs"].get(key, (None, []))
    for m in msgs:
        (st.error if kind == "error" else st.warning)(m)
    if kind == "ok" and key in st.session_state["upload_sources"]:
        st.success(f"Active: {st.session_state['upload_sources'][key]}")


def page_upload(S: dict) -> None:
    st.title("2. Data Upload")
    st.write("Upload your own files to replace the project data. Each file is checked for the required "
             "columns before it is used. Files stay in your browser session only.")
    st.warning(DISCLAIMER)
    gen = st.session_state["uploader_gen"]
    sig_store = st.session_state["upload_sig"]

    st.subheader("Genotype CSV (NCBI Pathogen Detection isolates table)")
    st.caption(f"Required columns: {', '.join(dl.GENOTYPE_REQUIRED)}.")
    f = st.file_uploader("Genotype CSV", type=["csv"], key=f"up_geno_{gen}", label_visibility="collapsed")
    if f is not None and sig_store.get("geno") != (f.name, f.size):
        try:
            df, w = dl.load_genotype_csv(f)
            _store("geno", df, f"Uploaded: {f.name}", (f.name, f.size), w)
        except dl.DataValidationError as e:
            _fail("geno", (f.name, f.size), str(e))
    _show_msgs("geno")

    st.subheader("AST CSV / TSV (antibiogram)")
    st.caption("Required columns: " + ", ".join(dl.AST_REQUIRED) +
               ". A 'BioSample' column is optional if you enter the accession below.")
    bs = st.text_input("BioSample accession for this antibiogram (only needed if the file has no "
                       "BioSample column)", value="", placeholder="e.g. SAMN47742561", key=f"bs_{gen}")
    f = st.file_uploader("AST file", type=["csv", "tsv", "txt"], key=f"up_ast_{gen}",
                         label_visibility="collapsed")
    if f is not None:
        sig = (f.name, f.size, bs.strip())
        if sig_store.get("ast") != sig:
            try:
                df, w = dl.load_ast_table(f, biosample=bs.strip() or None)
                _store("ast", df, f"Uploaded: {f.name}", sig, w)
            except dl.DataValidationError as e:
                _fail("ast", sig, str(e))
    _show_msgs("ast")

    st.subheader("Metadata CSV (optional)")
    st.caption("Required column: BioSample. Other columns (host, location, dates...) are shown as-is. "
               "If you upload none, metadata comes from the genome package's BioSample attributes.")
    f = st.file_uploader("Metadata CSV", type=["csv", "tsv", "txt"], key=f"up_meta_{gen}",
                         label_visibility="collapsed")
    if f is not None and sig_store.get("metadata") != (f.name, f.size):
        try:
            df, w = dl.load_metadata_csv(f)
            _store("metadata", df, f"Uploaded: {f.name}", (f.name, f.size), w)
        except dl.DataValidationError as e:
            _fail("metadata", (f.name, f.size), str(e))
    _show_msgs("metadata")

    st.subheader("Genome FASTA or NCBI genome package")
    st.caption("Accepts a genome FASTA (.fna/.fa/.fasta), assembly_data_report.jsonl, sequence_report.tsv, "
               "or the whole NCBI Datasets .zip. Uploading genome files replaces the project genome data.")
    files = st.file_uploader("Genome files", type=["fna", "fa", "fasta", "jsonl", "tsv", "zip"],
                             accept_multiple_files=True, key=f"up_genome_{gen}",
                             label_visibility="collapsed")
    if files:
        sig = tuple(sorted((x.name, x.size) for x in files))
        if sig_store.get("genome") != sig:
            try:
                fasta, rep, seq = [], [], []
                for x in files:
                    low = x.name.lower()
                    if low.endswith(".zip"):
                        pk = dl.read_genome_package_zip(x)
                        fasta += pk["fasta"]
                        rep += pk["assembly_report"]
                        seq += [s for s in pk["sequence_report"] if s[0].endswith(".tsv")]
                    elif low.endswith((".fna", ".fa", ".fasta")):
                        fasta.append((x.name, x))
                    elif low.endswith(".jsonl") and "sequence" not in low:
                        rep.append((x.name, x))
                    elif low.endswith(".tsv"):
                        seq.append((x.name, x))
                block = process_genome(fasta, rep, seq)
                _store("genome", block, "Uploaded: " + ", ".join(x.name for x in files), sig, [])
            except dl.DataValidationError as e:
                _fail("genome", sig, str(e))
    _show_msgs("genome")

    st.divider()
    if st.button("Revert to project data (clear all uploads)"):
        for k in ("uploads", "upload_sources", "upload_sig", "upload_msgs"):
            st.session_state[k] = {}
        st.session_state["uploader_gen"] += 1
        st.rerun()
    st.caption("Currently active sources: " + "; ".join(f"{k} = {v}" for k, v in S["sources"].items()))


# ----------------------------------------------------------------------------- page 3
def antibiotic_level_table(res) -> pd.DataFrame:
    gene_cols = cols_present(res.integrated_matched, ["record_id"] + [f"genes_{s}" for s in pp.STATUS_ORDER])
    return res.ast_long_matched.merge(res.integrated_matched[gene_cols], on="record_id", how="left")


def page_integration(S: dict) -> None:
    st.title("3. Data Integration")
    st.write("Genotype and AST data are matched **only by BioSample accession**. Isolates without a match "
             "are kept in separate tables and nothing is filled in or guessed.")
    if not need_genotype(S):
        return
    res, summ = S["res"], S["res"].summary
    c = st.columns(4)
    c[0].metric("Matched isolate records", summ["n_matched_records"])
    c[1].metric("Genotype only", f"{summ['n_genotype_only_records']:,}")
    c[2].metric("AST only (BioSamples)", summ["n_ast_only_biosamples"])
    c[3].metric("Records sharing a BioSample", summ["n_records_with_duplicate_biosample"])
    if summ["n_records_with_duplicate_biosample"]:
        st.caption("Some BioSamples appear in more than one genotype record (the same sample sequenced "
                   "or assembled twice). An antibiogram for such a BioSample attaches to every record "
                   "that shares it; these are not independent isolates.")
    show_fig(viz.integration_bar(summ))

    tabs = st.tabs(["Matched (genotype + AST)", "Antibiotic-level view", "Genotype only", "AST only",
                    "Missing data"])
    with tabs[0]:
        m = res.integrated_matched
        if m.empty:
            st.info("No BioSample is present in both the genotype and AST tables.")
        else:
            show = cols_present(m, ["isolate_name", "biosample", "record_id", "Assembly", "meta_collection_date",
                                    "meta_host", "meta_host_disease", "meta_geo_loc_name"]
                                + [f"genes_{s}" for s in pp.STATUS_ORDER]
                                + ["ast_resistant", "ast_intermediate", "ast_susceptible"])
            show_df(m[show])
        st.download_button("Download matched table (CSV)", csv_bytes(m), "integrated_matched.csv", "text/csv")
    with tabs[1]:
        al = antibiotic_level_table(res)
        if al.empty:
            st.info("No matched isolates.")
        else:
            st.caption("One row per matched isolate and antibiotic: laboratory result next to the gene "
                       "calls, with each detection status kept in its own column.")
            show_df(al.drop(columns=["measurement_value"], errors="ignore"))
        st.download_button("Download antibiotic-level table (CSV)", csv_bytes(al),
                           "integrated_antibiotic_level.csv", "text/csv")
    with tabs[2]:
        go = res.genotype_only
        st.caption(f"{len(go):,} genotype records have no AST data here. They are NOT assumed susceptible.")
        show_df(go[cols_present(go, ["isolate_name", "biosample", "record_id", "Assembly", "Location",
                                     "Isolation source", "AMR genotypes"])], height=420)
    with tabs[3]:
        ao = res.ast_only
        if ao.empty:
            st.success("Every AST BioSample was found in the genotype table.")
        else:
            st.warning("These AST rows have a BioSample that is not in the genotype table.")
            show_df(ao)
    with tabs[4]:
        rep = pp.missing_data_report(res.integrated_all)
        st.caption("Columns that are empty for every record (e.g. Serovar) are listed for transparency. "
                   "Metadata (meta_*) exists only for isolates covered by the genome package.")
        show_fig(viz.missing_bar(rep))
        show_df(rep, height=300)
    st.divider()
    st.download_button("Download full integrated dataset - all genotype records (CSV)",
                       csv_bytes(res.integrated_all), "integrated_all_records.csv", "text/csv")


# ----------------------------------------------------------------------------- page 4
def page_genes(S: dict) -> None:
    st.title("4. AMR Gene Analysis")
    if not need_genotype(S):
        return
    res = S["res"]
    tbl, long_df = res.integrated_all, res.genotype_long
    with st.expander("How to read detection statuses (important)"):
        for s in pp.STATUS_ORDER:
            st.markdown(f"- **{s}** - {ga.STATUS_NOTES[s]}")
        st.markdown("- **Not listed** = *not reported*. This is **not** proof that a gene is absent.")

    scope = st.radio("Isolate records to analyse", ["All records", "Records with AST data", "Name contains..."],
                     horizontal=True)
    if scope == "All records":
        ids = set(tbl["record_id"])
    elif scope == "Records with AST data":
        ids = set(tbl.loc[tbl["has_ast"], "record_id"])
    else:
        text = st.text_input("Isolate name contains", value="Nivedi")
        ids = set(tbl.loc[tbl["isolate_name"].str.contains(text, case=False, na=False, regex=False), "record_id"])
    sub = long_df[long_df["record_id"].isin(ids)]
    n_rec = len(ids)
    if n_rec == 0:
        st.info("No isolate records in this selection.")
        return

    gs = ga.genotype_summary(sub, n_rec)
    c = st.columns(4)
    c[0].metric("Isolate records", f"{n_rec:,}")
    c[1].metric("Gene calls", f"{gs['n_gene_calls']:,}")
    c[2].metric("Distinct genes", f"{gs['n_distinct_genes']:,}")
    c[3].metric("Distinct point mutations", f"{gs['n_distinct_point_mutations']:,}")

    st.subheader("Complete versus partial calls")
    left, right = st.columns([3, 2])
    with left:
        show_fig(viz.status_bar(ga.status_counts(sub)))
    with right:
        show_df(ga.status_group_counts(sub))
        st.caption("Partial = PARTIAL + PARTIAL_END_OF_CONTIG. Statuses are shown separately in the chart.")

    st.subheader("Gene frequency")
    freq = ga.gene_frequency(sub, n_rec)
    top_n = st.slider("Number of genes in the chart", 5, min(40, max(5, len(freq))), min(15, max(5, len(freq))))
    show_fig(viz.gene_frequency_bar(freq, top_n))
    with st.expander("Full frequency table"):
        show_df(freq)
        st.download_button("Download gene frequency (CSV)", csv_bytes(freq), "gene_frequency.csv", "text/csv")

    st.subheader("Point mutations")
    pm = ga.point_mutations(sub, n_rec)
    if pm.empty:
        st.info("No point mutations were reported for this selection.")
    else:
        l2, r2 = st.columns([3, 2])
        with l2:
            show_fig(viz.point_mutation_bar(pm))
        with r2:
            show_df(pm, height=360)

    st.subheader("Isolate-wise gene distribution")
    labels = (tbl.set_index("record_id").loc[sorted(ids), "isolate_name"] + " (" + pd.Index(sorted(ids)) + ")")
    options = labels.tolist()
    if len(options) > 40:
        st.caption(f"{len(options):,} records selected; choose up to 40 to display as a matrix.")
        preferred = [o for o, rid in zip(options, sorted(ids)) if rid in set(tbl.loc[tbl.has_ast, "record_id"])]
        chosen = st.multiselect("Isolates to display", options, default=(preferred or options)[:10],
                                max_selections=40)
    else:
        chosen = options
    chosen_ids = [o.rsplit("(", 1)[1].rstrip(")") for o in chosen]
    top_genes = st.slider("Genes shown (most frequent first)", 5, 40, 12)
    if chosen_ids:
        mat = ga.isolate_gene_matrix(sub, chosen_ids, top_n=top_genes)
        if mat.empty:
            st.info("No gene calls for the chosen isolates.")
        else:
            show_fig(viz.matrix_heatmap(mat))
            st.markdown(viz.status_legend_html(), unsafe_allow_html=True)
    else:
        st.info("Select at least one isolate.")

    st.subheader("Single-isolate gene profile")
    pick = st.selectbox("Isolate record", options, index=0)
    prof = ga.isolate_profile(long_df, pick.rsplit("(", 1)[1].rstrip(")"))
    show_df(prof)


# ----------------------------------------------------------------------------- page 5
def page_ast(S: dict) -> None:
    st.title("5. Antibiotic Susceptibility Analysis")
    if S["ast_clean"] is None or S["ast_clean"].empty:
        st.info("No AST data loaded. Add an antibiogram under **2. Data Upload**.")
        return
    ast = S["ast_clean"]
    names = {}
    if S["res"] is not None:
        t = S["res"].integrated_all
        names = t.groupby("biosample")["isolate_name"].agg(lambda s: ", ".join(sorted(set(s)))).to_dict()
    biosamples = sorted(ast["biosample"].unique())
    label_to_bs = {f"{b} - {names.get(b, 'not in genotype table')}": b for b in biosamples}
    pick = label_to_bs[st.selectbox("Isolate (BioSample)", list(label_to_bs))]
    sub = ast[ast["biosample"] == pick]
    st.caption("Phenotype labels are shown exactly as reported by the laboratory. They are not re-interpreted "
               "against breakpoints. Method: " + ", ".join(sorted(sub["method"].dropna().unique())) +
               " | Standard: " + ", ".join(sorted(sub["standard"].dropna().unique())))

    groups = pa.antibiotics_by_phenotype(ast, pick)
    c = st.columns(3)
    for col, (label, key) in zip(c, [("Resistant", "resistant"), ("Intermediate", "intermediate"),
                                     ("Susceptible", "susceptible")]):
        with col:
            st.metric(label, len(groups.get(key, [])))
            if groups.get(key):
                st.markdown("\n".join(f"- {a}" for a in groups[key]))
            else:
                st.caption("None reported" if key == "intermediate" else "None")

    left, right = st.columns([2, 3])
    with left:
        show_fig(viz.phenotype_bar(pa.phenotype_counts(sub)))
    with right:
        st.subheader("Antibiogram")
        show_df(pa.antibiogram_table(sub).drop(columns=["biosample"]))
    st.subheader("Zone diameters")
    zd = pa.zone_diameters(sub)
    if zd.empty:
        st.info("No zone diameter or MIC values were reported.")
    else:
        show_fig(viz.zone_diameter_chart(zd))
        st.caption("Signs are preserved from the source. `<=` and `>=` values are bounds (the true diameter "
                   "is at most / at least the number), so bar lengths are not exact diameters.")
    if ast["biosample"].nunique() > 1:
        st.subheader("All isolates: results per antibiotic")
        show_df(pa.resistance_by_antibiotic(ast))
        st.caption("Counts of tested isolates; with few isolates these are not resistance rates.")


# ----------------------------------------------------------------------------- page 6
def _record_options(S: dict, key: str) -> tuple[list[str], dict]:
    """Choose which isolate records can be picked; returns (labels, label -> record_id)."""
    tbl = S["res"].integrated_all
    scope = st.radio("Isolates to choose from", ["Isolates with AST data", "All records (search by name)"],
                     horizontal=True, key=f"{key}_scope")
    if scope == "Isolates with AST data":
        sub = tbl[tbl["has_ast"]]
    else:
        text = st.text_input("Isolate name contains", value="Nivedi", key=f"{key}_filter")
        sub = tbl[tbl["isolate_name"].str.contains(text, case=False, na=False, regex=False)]
        if len(sub) > 500:
            st.caption(f"{len(sub):,} matches; showing the first 500. Refine the search.")
            sub = sub.head(500)
    labels = {f"{r.isolate_name} ({r.record_id})": r.record_id for r in sub.itertuples()}
    return list(labels), labels


def page_prediction(S: dict) -> None:
    st.title("6. Prediction Demonstration")
    st.error(mdl.RULE_BASED_BANNER)
    st.write("Pick an isolate and an antibiotic to see the genetic evidence, the laboratory result (if one "
             "exists) and an explanation of the resistance-associated markers. This page **does not predict** "
             "resistance. It only looks up known markers and shows them next to the real result.")
    if not need_genotype(S):
        return
    res, ast = S["res"], S["ast_clean"]
    options, mapping = _record_options(S, "pred")
    if not options:
        st.info("No isolate records match. Try another selection.")
        return
    c1, c2 = st.columns(2)
    with c1:
        iso_label = st.selectbox("Isolate", options)
    rid = mapping[iso_label]
    row = res.integrated_all[res.integrated_all["record_id"] == rid].iloc[0]
    bs = row["biosample"]
    tested = sorted(ast.loc[ast["biosample"] == bs, "antibiotic"].unique()) if not ast.empty else []
    ab_options = sorted(set(mdl.rule_antibiotics()) | set(tested))
    with c2:
        ab_labels = {f"{a}  (lab result available)" if a in tested else a: a for a in ab_options}
        antibiotic = ab_labels[st.selectbox("Antibiotic", list(ab_labels))]

    calls = ga.isolate_profile(res.genotype_long, rid)
    ev = mdl.evaluate_markers(calls, antibiotic)
    lab = ast[(ast["biosample"] == bs) & (ast["antibiotic"] == antibiotic)] if not ast.empty else ast
    pheno = lab["phenotype"].iloc[0] if len(lab) else None

    st.subheader("1. Genetic evidence")
    if ev["outcome"] == "NO_RULES":
        st.info("No marker rules are defined for this antibiotic in the prototype's table.")
    else:
        st.caption(f"Antibiotic class: {ev['antibiotic_class']}")
        if ev["matches"].empty:
            st.warning("No associated gene or mutation is reported for this isolate. "
                       "**This is not proof that the marker is absent** (not reported is not the same as absent).")
        else:
            show_df(ev["matches"][["gene", "status", "evidence", "role"]])
        st.markdown(f"**Marker finding:** {ev['outcome_text']}")

    st.subheader("2. Laboratory AST result")
    if len(lab):
        r = lab.iloc[0]
        c = st.columns(2)
        c[0].metric("Phenotype (as reported)", str(r["phenotype_reported"]).title())
        c[1].metric("Measurement", r["measurement_reported"] or "n/a")
        st.caption(f"Sign meaning: {pa.measurement_kind(r['measurement_sign'])}. "
                   f"Method: {r['method']} | Standard: {r['standard']}")
    else:
        st.info("No laboratory result is available for this isolate and antibiotic. "
                "It is **not** assumed to be susceptible.")

    st.subheader("3. Resistance-associated marker explanation")
    key = mdl.resolve_rule_key(antibiotic)
    if key:
        spec = mdl.RULES[key]
        rows = [(m.pattern, m.role, m.note) for m in spec["markers"]]
        show_df(pd.DataFrame(rows, columns=["Marker (gene / mutation pattern)", "Role", "Why it matters"]))
        st.caption("Roles: **primary** markers are counted as evidence; **regulatory** and **context** entries are "
                   "shown for explanation but are not counted.")
    st.markdown("**Reading for this isolate:** " + mdl.compare_with_lab(ev["outcome"], pheno))
    st.caption("This sentence describes one isolate. It is not an accuracy measurement and not a prediction.")

    if row["biosample_record_count"] > 1:
        st.warning(f"This BioSample appears in {row['biosample_record_count']} genotype records.")
    with st.expander("All gene calls for this isolate"):
        show_df(calls)
    with st.expander("Illustrative marker knowledge table used by this page"):
        st.warning("Written by the prototype author from textbook-level knowledge. It has **not** been validated "
                   "against CARD, ResFinder or CLSI. Check it against those sources before relying on it.")
        show_df(mdl.knowledge_table())
    if not ast.empty:
        with st.expander("Marker evidence vs laboratory result, all tested antibiotics of this isolate"):
            sub = ast[ast["biosample"] == bs]
            if sub.empty:
                st.caption("This isolate has no AST data.")
            else:
                show_df(mdl.isolate_marker_summary(calls, sub))


# ----------------------------------------------------------------------------- page 7
def page_ml(S: dict) -> None:
    st.title("7. Machine Learning")
    st.warning(DISCLAIMER)
    if not need_genotype(S):
        return
    res = S["res"]
    st.write("A Random Forest baseline is part of the pipeline, but a model is only trained when enough genuine, "
             "independent, labelled isolates exist. Otherwise training and evaluation would not be statistically "
             "meaningful, so the app refuses to train and shows **no** performance numbers.")
    st.subheader("Pipeline")
    show_df(pd.DataFrame(mdl.PIPELINE_STEPS, columns=["Step", "Status"]))

    st.subheader("Is training statistically meaningful?")
    st.caption(f"Guard: at least {mdl.MIN_ISOLATES_TOTAL} independent labelled isolates in total and "
               f"{mdl.MIN_ISOLATES_PER_CLASS} in each class (resistant / susceptible) per antibiotic. "
               "Isolates are counted by BioSample, so duplicate records count once. The thresholds are prototype "
               "rules of thumb and are set at the top of model.py.")
    tt = mdl.trainability_table(res.ast_long_matched)
    if tt.empty:
        st.error("**Training is disabled.** There are no labelled isolates in the matched data.")
        return
    show_df(tt)
    ok = tt[tt["trainable"]]["antibiotic"].tolist()
    if not ok:
        n_max = int(tt["n_independent_isolates"].max())
        st.error(f"**Training is disabled for every antibiotic.** The most labelled isolates for any antibiotic "
                 f"is {n_max}. A model trained and tested on so few isolates cannot be validated: a split would "
                 "leave nothing (or one isolate) to test, and any accuracy reported would be meaningless. "
                 "Precision, recall, F1 and the confusion matrix are therefore **not shown**.")
        st.info("To enable training: add real antibiograms (disk diffusion or MIC) for many more isolates on the "
                "**2. Data Upload** page and make sure their BioSample accessions appear in the genotype file.")
    else:
        st.success(f"Enough labelled isolates for: {', '.join(ok)}.")
        ab = st.selectbox("Antibiotic to model", ok)
        if st.button("Train Random Forest"):
            try:
                st.session_state.setdefault("model_results", {})[ab] = mdl.train_random_forest(
                    res.genotype_long, res.ast_long_matched, ab)
            except mdl.ModelNotTrainable as e:
                st.error(f"Training refused: {e}")
        mr = st.session_state.get("model_results", {}).get(ab)
        if mr is not None:
            c = st.columns(4)
            c[0].metric("Precision (resistant)", f"{mr.precision:.2f}")
            c[1].metric("Recall (resistant)", f"{mr.recall:.2f}")
            c[2].metric("F1 (resistant)", f"{mr.f1:.2f}")
            c[3].metric("Test isolates", mr.n_test)
            l, r = st.columns(2)
            with l:
                show_fig(viz.confusion_matrix_heatmap(mr.confusion))
            with r:
                show_fig(viz.feature_importance_bar(mr.feature_importance))
            for w in mr.warnings:
                st.warning(w)

    st.subheader("Features that would be used")
    st.caption("Each gene gives two binary features, 'confirmed' (COMPLETE or POINT) and 'uncertain' (partial, "
               "end-of-contig, HMM or mistranslated). A gene that is not reported is coded 0 in both, meaning "
               "'no call', never 'confirmed absent'.")
    X, no_calls = mdl.build_feature_matrix(res.genotype_long, res.integrated_all["record_id"].tolist())
    c = st.columns(3)
    c[0].metric("Records", f"{X.shape[0]:,}")
    c[1].metric("Features", f"{X.shape[1]:,}")
    c[2].metric("Records without gene calls", len(no_calls))
    with st.expander("Feature matrix preview (first 20 rows, first 20 columns)"):
        show_df(X.iloc[:20, :20], index=True)


# ----------------------------------------------------------------------------- page 8
def page_report(S: dict) -> None:
    st.title("8. Report")
    st.write("Generate a PDF with the project overview, dataset summary, AMR gene analysis, antibiogram findings, "
             "model status and limitations. It uses the data currently loaded in the app.")
    st.warning(DISCLAIMER)
    if not need_genotype(S):
        return
    res = S["res"]
    trained = st.session_state.get("model_results", {})
    if trained:
        st.caption("Models trained this session will be included: " + ", ".join(trained))
    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Building report..."):
            try:
                st.session_state["report_pdf"] = rg.build_pdf_report(
                    res, S["ast_clean"], S["genome"], S["sources"],
                    mdl.trainability_table(res.ast_long_matched), trained)
            except Exception as exc:  # show the cause instead of a stack trace
                st.session_state.pop("report_pdf", None)
                st.error(f"The report could not be built: {exc}")
    pdf = st.session_state.get("report_pdf")
    if pdf:
        st.success(f"Report ready ({len(pdf) / 1024:.0f} KB).")
        st.download_button("Download PDF report", pdf, "amr_prototype_report.pdf", "application/pdf")
        st.caption("The report reflects the data loaded when it was generated. Generate it again after "
                   "uploading new files.")


# ----------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    S = get_state()
    with st.sidebar:
        st.markdown("## 🧬 AMR Prototype")
        st.caption("*Staphylococcus aureus*")
        page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("Academic proof-of-concept. Not a clinically validated diagnostic system.")
        if S["res"] is not None:
            st.caption(f"Isolates with AST data: **{S['res'].summary['n_matched_biosamples']}**")

    if page == PAGES[0]:
        page_home(S)
    elif page == PAGES[1]:
        page_upload(S)
    elif page == PAGES[2]:
        page_integration(S)
    elif page == PAGES[3]:
        page_genes(S)
    elif page == PAGES[4]:
        page_ast(S)
    elif page == PAGES[5]:
        page_prediction(S)
    elif page == PAGES[6]:
        page_ml(S)
    else:
        page_report(S)

main()
