"""Druglike COV/MAT vs attempted-K charts for the Streamlit dashboard.

Kept next to the Streamlit entrypoint so Community Cloud loads it from the
checkout even when a stale casf-benchmark wheel is installed. No RDKit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

PLOT_LABELS = (
    "qwen_1p7b_fsq_bigdata_step47023",
    "loqi_druglike",
    "nextmol_dmt_l_druglike",
    "torsional_diffusion_druglike",
    "mcf_drugs_l_druglike",
    "flowr_druglike",
)

CHARTS = (
    {
        "title": "Coverage vs sample size (mean)",
        "ylabel": "Coverage (mean over 23 molecules)",
        "recall": "cov_r_mean",
        "precision": "cov_p_mean",
    },
    {
        "title": "Matching vs sample size (mean)",
        "ylabel": "Matching RMSD (Å, mean over 23 molecules)",
        "recall": "mat_r_mean",
        "precision": "mat_p_mean",
    },
    {
        "title": "Coverage vs sample size (median)",
        "ylabel": "Coverage (median over 23 molecules)",
        "recall": "cov_r_median",
        "precision": "cov_p_median",
    },
    {
        "title": "Matching vs sample size (median)",
        "ylabel": "Matching RMSD (Å, median over 23 molecules)",
        "recall": "mat_r_median",
        "precision": "mat_p_median",
    },
)


def plot_display_name(row: pd.Series) -> str:
    generator = str(row.get("generator", "") or "")
    if generator.lower() == "qwen":
        parts = ["Qwen"]
        size = row.get("model_size")
        tokenizer = row.get("tokenizer")
        if pd.notna(size) and str(size) not in {"", "nan"}:
            parts.append(str(size))
        if pd.notna(tokenizer) and str(tokenizer) not in {"", "nan"}:
            parts.append(str(tokenizer))
        return " ".join(parts)
    display = row.get("display_label")
    if pd.notna(display) and str(display) not in {"", "nan"}:
        return str(display)
    return str(row.get("label", ""))


def filter_plot_summary(
    summary: pd.DataFrame, labels: tuple[str, ...] = PLOT_LABELS
) -> pd.DataFrame:
    if summary.empty or "label" not in summary.columns:
        return summary.copy()
    frame = summary[summary["label"].astype(str).isin(labels)].copy()
    if frame.empty:
        return frame
    frame["plot_label"] = frame.apply(plot_display_name, axis=1)
    order = {label: index for index, label in enumerate(labels)}
    frame["_order"] = frame["label"].map(order)
    return frame.sort_values(["_order", "k"]).drop(columns="_order")


def long_metric_pair(frame: pd.DataFrame, recall_col: str, precision_col: str) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "method": row["plot_label"],
                "k": row["k"],
                "series": "Recall (R)",
                "value": row[recall_col],
            }
        )
        rows.append(
            {
                "method": row["plot_label"],
                "k": row["k"],
                "series": "Precision (P)",
                "value": row[precision_col],
            }
        )
    return pd.DataFrame(rows)


def k_chart(frame: pd.DataFrame, *, recall_col: str, precision_col: str, ylabel: str, title: str):
    import altair as alt

    long = long_metric_pair(frame, recall_col, precision_col)
    method_order = list(dict.fromkeys(frame["plot_label"].astype(str)))
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "k:Q",
                scale=alt.Scale(type="log"),
                title="K (generated conformers per molecule)",
            ),
            y=alt.Y("value:Q", title=ylabel),
            color=alt.Color("method:N", sort=method_order, title="Method"),
            strokeDash=alt.StrokeDash("series:N", title=""),
            tooltip=["method:N", "k:Q", "series:N", "value:Q"],
        )
        .properties(title=title, height=320)
    )


def render_druglike_k_charts(summary: pd.DataFrame) -> None:
    frame = filter_plot_summary(summary)
    if frame.empty:
        return
    st.subheader("Coverage and matching vs sample size")
    st.caption(
        "Attempted-K on the raw ~1000-conformer pool (random subsample, 3 seeds, "
        "23 molecules). Solid = recall, dashed = precision. Panel: Qwen 1.7B FSQ, "
        "LOQI, NExT-Mol DMT-L, Torsional Diffusion, MCF drugs-L, FlowR."
    )
    for index in range(0, len(CHARTS), 2):
        columns = st.columns(2)
        for column, spec in zip(columns, CHARTS[index : index + 2]):
            if spec["recall"] not in frame.columns or spec["precision"] not in frame.columns:
                continue
            with column:
                st.altair_chart(
                    k_chart(
                        frame,
                        recall_col=spec["recall"],
                        precision_col=spec["precision"],
                        ylabel=spec["ylabel"],
                        title=spec["title"],
                    ),
                    use_container_width=True,
                )
