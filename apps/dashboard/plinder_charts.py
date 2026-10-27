"""PLINDER recovery and sample-size charts for the Streamlit dashboard.

Kept next to the Streamlit entrypoint so Community Cloud loads it from the
checkout. No RDKit.
"""

from __future__ import annotations

import pandas as pd

MANUSCRIPT_LABELS = (
    "qwen_1p7b_fsq_bigdata_step47023",
    "loqi_druglike",
    "nextmol_dmt_l_druglike",
    "torsional_diffusion_druglike",
    "mcf_drugs_l_druglike",
    "flowr_druglike",
    "rdkit_random_raw_druglike",
    "torsion_raw_druglike",
)

HEADLINE_THRESHOLDS = (0.5, 0.75, 1.0)


def plot_display_name(row: pd.Series) -> str:
    generator = str(row.get("generator", "") or "")
    display = row.get("display_label")
    label = str(row.get("label", "") or "")
    if pd.notna(display) and str(display) not in {"", "nan", label}:
        return str(display)
    if generator.lower() == "qwen":
        parts = ["Qwen"]
        size = row.get("model_size")
        tokenizer = row.get("tokenizer")
        if pd.notna(size) and str(size) not in {"", "nan"}:
            parts.append(str(size))
        if pd.notna(tokenizer) and str(tokenizer) not in {"", "nan"}:
            parts.append(str(tokenizer))
        return " ".join(parts)
    if pd.notna(display) and str(display) not in {"", "nan"}:
        return str(display)
    return label


def method_choices(labels: list[str]) -> list[str]:
    """Manuscript methods first, then any other labels in their existing order."""
    present = list(dict.fromkeys(str(label) for label in labels))
    first = [label for label in MANUSCRIPT_LABELS if label in present]
    rest = [label for label in present if label not in first]
    return first + rest


def with_plot_labels(frame: pd.DataFrame, labels: tuple[str, ...] | list[str]) -> pd.DataFrame:
    if frame.empty or "label" not in frame.columns:
        return frame.copy()
    selected = frame[frame["label"].astype(str).isin([str(label) for label in labels])].copy()
    if selected.empty:
        return selected
    selected["plot_label"] = selected.apply(plot_display_name, axis=1)
    order = {str(label): index for index, label in enumerate(labels)}
    selected["_order"] = selected["label"].astype(str).map(order)
    sort_columns = ["_order"]
    if "threshold" in selected.columns:
        sort_columns.append("threshold")
    if "k" in selected.columns:
        sort_columns.append("k")
    return selected.sort_values(sort_columns).drop(columns="_order")


def coverage_at_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if frame.empty or "threshold" not in frame.columns:
        return frame.copy()
    rounded = pd.to_numeric(frame["threshold"], errors="coerce").round(2)
    return frame.loc[rounded == round(float(threshold), 2)].copy()


def recovery_chart(frame: pd.DataFrame):
    import altair as alt

    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "method": row["plot_label"],
                "threshold": row["threshold"],
                "series": "COV-R",
                "value": row["cov_r_mean"],
            }
        )
        rows.append(
            {
                "method": row["plot_label"],
                "threshold": row["threshold"],
                "series": "COV-P",
                "value": row["cov_p_mean"],
            }
        )
    long = pd.DataFrame(rows)
    method_order = list(dict.fromkeys(frame["plot_label"].astype(str)))
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="RMSD threshold (Å)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", title="Coverage (mean over molecules)"),
            color=alt.Color("method:N", sort=method_order, title="Method"),
            strokeDash=alt.StrokeDash("series:N", title=""),
            tooltip=["method:N", "threshold:Q", "series:N", "value:Q"],
        )
        .properties(title="Coverage vs RMSD threshold", height=360)
    )
