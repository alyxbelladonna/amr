"""
visualization.py
================
All charts for the dashboard. Plotly is used for interactive charts in the app;
Matplotlib is used for the static genome figure (which the PDF report also reuses).

Colour conventions (same everywhere, so a colour always means the same thing):
  gene-call status  -> STATUS_COLORS      phenotype -> PHENO_COLORS
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed; works inside Streamlit and when making PDFs
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import STATUS_ORDER

PALETTE = {"primary": "#1f6fb2", "light": "#eaf4fc", "mid": "#8ec1ea",
           "dark": "#0b3c5d", "grey": "#c9d3dc", "text": "#1b2a38"}

STATUS_COLORS = {
    "COMPLETE": "#1f6fb2", "PARTIAL": "#8ec1ea", "PARTIAL_END_OF_CONTIG": "#f2b134",
    "HMM": "#7a6bc4", "POINT": "#2a9d8f", "MISTRANSLATION": "#b08968", "UNPARSED": "#999999",
}
STATUS_ABBREV = {"COMPLETE": "C", "PARTIAL": "P", "PARTIAL_END_OF_CONTIG": "P-end",
                 "HMM": "HMM", "POINT": "pt", "MISTRANSLATION": "Mis"}
PHENO_COLORS = {"resistant": "#c0392b", "intermediate": "#f2b134", "susceptible": "#2a9d8f"}
NOT_REPORTED_COLOR = "#f1f4f7"


def _style(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", title=title, height=height,
                      font=dict(family="Inter, Segoe UI, Arial", color=PALETTE["text"], size=13),
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(title_text=""))
    return fig


# ----------------------------------------------------------------------------- genes
def status_legend_html() -> str:
    """Small HTML colour key for gene-call statuses (used under the heatmap)."""
    chips = [f'<span style="display:inline-block;margin:0 10px 4px 0;">'
             f'<span style="display:inline-block;width:12px;height:12px;background:{c};'
             f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>{s}</span>'
             for s, c in STATUS_COLORS.items() if s != "UNPARSED"]
    chips.append(f'<span style="display:inline-block;margin:0 10px 4px 0;">'
                 f'<span style="display:inline-block;width:12px;height:12px;background:{NOT_REPORTED_COLOR};'
                 f'border:1px solid #cfd8e0;border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                 f'not reported</span>')
    return "<div style='font-size:0.85rem'>" + "".join(chips) + "</div>"


def status_bar(status_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(status_df, x="status", y="n_calls", color="status", text="n_calls",
                 color_discrete_map=STATUS_COLORS, category_orders={"status": STATUS_ORDER})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Gene calls")
    return _style(fig, "Gene calls by detection status", 360)


def gene_frequency_bar(freq: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Stacked horizontal bars: records per gene, split by detection status."""
    top = freq.head(top_n)
    rows = []
    for _, r in top.iterrows():
        for s in STATUS_ORDER:
            if r[f"n_{s}"] > 0:
                rows.append({"gene": r["gene"], "status": s, "n_records": int(r[f"n_{s}"])})
    long = pd.DataFrame(rows, columns=["gene", "status", "n_records"])
    fig = px.bar(long, x="n_records", y="gene", color="status", orientation="h",
                 color_discrete_map=STATUS_COLORS, category_orders={"status": STATUS_ORDER})
    fig.update_yaxes(categoryorder="array", categoryarray=list(top["gene"])[::-1], title="")
    fig.update_xaxes(title="Isolate records (a record with two statuses counts in each)")
    return _style(fig, f"Top {len(top)} genes / mutations by frequency",
                  max(320, 24 * len(top) + 120))


def point_mutation_bar(pm: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = pm.head(top_n)
    fig = px.bar(top, x="n_records", y="mutation", orientation="h", text="n_records",
                 color_discrete_sequence=[STATUS_COLORS["POINT"]])
    fig.update_yaxes(categoryorder="array", categoryarray=list(top["mutation"])[::-1], title="")
    fig.update_xaxes(title="Isolate records")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, "Point mutations", max(300, 26 * len(top) + 110))


def matrix_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Isolate x gene heatmap coloured by detection STATUS. Blank = not reported (not 'absent')."""
    keys = [""] + STATUS_ORDER
    code = {k: i for i, k in enumerate(keys)}
    mixed_code = len(keys)                       # cells holding more than one status, e.g. COMPLETE/HMM
    colors = [NOT_REPORTED_COLOR] + [STATUS_COLORS[s] for s in STATUS_ORDER] + ["#555555"]
    n = len(colors)
    colorscale = []
    for i, c in enumerate(colors):
        colorscale += [[i / n, c], [(i + 1) / n, c]]

    z = matrix.apply(lambda col: col.map(lambda v: code.get(v, mixed_code)))
    text = matrix.apply(lambda col: col.map(lambda v: STATUS_ABBREV.get(v, "multi" if v else "")))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(matrix.columns), y=list(matrix.index), text=text.values,
        texttemplate="%{text}", textfont=dict(size=10, color="white"),
        customdata=matrix.values, colorscale=colorscale, zmin=0, zmax=n, showscale=False,
        xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}<br>status: %{customdata}<extra></extra>"))
    fig.update_xaxes(tickangle=-45, side="bottom")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Isolate-wise gene distribution (cell = detection status)",
                  max(260, 26 * len(matrix) + 170))


def genes_per_isolate_hist(gpi: pd.DataFrame) -> go.Figure:
    fig = px.histogram(gpi, x="total_calls", nbins=min(30, max(5, int(gpi["total_calls"].max()))),
                       color_discrete_sequence=[PALETTE["primary"]])
    fig.update_layout(xaxis_title="Gene calls per isolate record", yaxis_title="Isolate records",
                      bargap=0.05)
    return _style(fig, "Distribution of gene calls per isolate", 320)


# ----------------------------------------------------------------------------- phenotype
def phenotype_bar(counts: pd.DataFrame) -> go.Figure:
    fig = px.bar(counts, x="phenotype", y="n_results", color="phenotype", text="n_results",
                 color_discrete_map=PHENO_COLORS,
                 category_orders={"phenotype": ["resistant", "intermediate", "susceptible"]})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="AST results")
    return _style(fig, "AST results by phenotype", 340)


def zone_diameter_chart(zd: pd.DataFrame) -> go.Figure:
    """
    Bars show the reported number; the label shows the sign exactly as supplied.
    Many NCBI antibiogram values are bounds ('>=24 mm'), so bars are NOT exact diameters.
    """
    zd = zd.dropna(subset=["plot_value"])
    fig = px.bar(zd, x="plot_value", y="antibiotic", orientation="h", color=zd["phenotype"].str.lower(),
                 color_discrete_map=PHENO_COLORS, text="measurement_reported",
                 hover_data={"measurement_meaning": True, "plot_value": False})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="", categoryorder="array", categoryarray=list(zd["antibiotic"])[::-1])
    fig.update_xaxes(title="Reported zone diameter (mm) - see sign in label")
    return _style(fig, "Reported zone diameters (signs preserved)", max(300, 40 * len(zd) + 120))


# ----------------------------------------------------------------------------- integration
def integration_bar(summary: dict) -> go.Figure:
    df = pd.DataFrame({
        "group": ["Matched (genotype + AST)", "Genotype only", "AST only"],
        "count": [summary["n_matched_records"], summary["n_genotype_only_records"],
                  summary["n_ast_only_biosamples"]]})
    fig = px.bar(df, x="count", y="group", orientation="h", text="count", color="group",
                 color_discrete_sequence=[PALETTE["primary"], PALETTE["mid"], PALETTE["grey"]])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="Records (AST only: BioSamples)", yaxis_title="")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Dataset integration by BioSample", 260)


def missing_bar(report: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = report.head(top_n)
    fig = px.bar(top, x="pct_missing", y="column", orientation="h", text="pct_missing",
                 color_discrete_sequence=[PALETTE["mid"]])
    fig.update_yaxes(categoryorder="array", categoryarray=list(top["column"])[::-1], title="")
    fig.update_xaxes(title="% of records missing", range=[0, 110])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, "Missing data (columns with most gaps)", max(300, 24 * len(top) + 110))


# ----------------------------------------------------------------------------- genome (Matplotlib)
def contig_length_figure(contigs: pd.DataFrame, n50: int | None = None):
    """Static Matplotlib bar chart of contig lengths (largest first), log scale."""
    df = contigs.sort_values("length", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(1, len(df) + 1), df["length"], color=PALETTE["primary"], width=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Contig rank (largest to smallest)")
    ax.set_ylabel("Length (bp, log scale)")
    ax.set_title(f"Contig lengths ({len(df)} contigs)", fontsize=11, color=PALETTE["dark"])
    if n50:
        ax.axhline(n50, color="#c0392b", linestyle="--", linewidth=1)
        ax.text(len(df) * 0.98, n50 * 1.15, f"N50 = {n50:,} bp", ha="right", color="#c0392b", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- model results
def confusion_matrix_heatmap(cm: pd.DataFrame) -> go.Figure:
    """Confusion matrix of a valid training run (rows = actual, columns = predicted)."""
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=list(cm.columns), y=list(cm.index), text=cm.values, texttemplate="%{text}",
        textfont=dict(size=18), colorscale=[[0, "#ffffff"], [1, PALETTE["primary"]]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>isolates: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (test isolates)", 340)


def feature_importance_bar(imp: pd.DataFrame) -> go.Figure:
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 color_discrete_sequence=[PALETTE["primary"]])
    fig.update_yaxes(categoryorder="array", categoryarray=list(imp["feature"])[::-1], title="")
    fig.update_xaxes(title="Random Forest importance")
    return _style(fig, "Most influential features", max(300, 24 * len(imp) + 110))


# ----------------------------------------------------------------------------- static figures for the PDF
def status_counts_figure_mpl(status_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    colors = [STATUS_COLORS.get(s, "#999999") for s in status_df["status"]]
    ax.bar(status_df["status"], status_df["n_calls"], color=colors)
    for i, v in enumerate(status_df["n_calls"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Gene calls")
    ax.set_title("Gene calls by detection status", fontsize=10, color=PALETTE["dark"])
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def gene_frequency_figure_mpl(freq: pd.DataFrame, top_n: int = 12):
    top = freq.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(top) + 1.2))
    left = [0] * len(top)
    for s in STATUS_ORDER:
        vals = top[f"n_{s}"].tolist()
        if sum(vals) == 0:
            continue
        ax.barh(top["gene"], vals, left=left, color=STATUS_COLORS[s], label=s)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("Isolate records")
    ax.set_title(f"Top {len(top)} genes / mutations", fontsize=10, color=PALETTE["dark"])
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
