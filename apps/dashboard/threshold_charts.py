"""CASF hit-rate and cluster-count vs RMSD-threshold charts.

Uses values already stored on ``comparison_rows``. Kept next to the Streamlit
entrypoint so Community Cloud loads it from the checkout. No RDKit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

PANEL_FAMILIES = (
    "qwen_1p7b_fsq_bigdata_pretrain",
    "loqi_raw",
    "nextmol_dmt_l_raw",
    "torsional_diffusion_raw",
    "mcf_drugs_l_raw",
    "flowr_raw",
    "rdkit_random_raw",
)

PLOT_LABELS = {
    "qwen_1p7b_fsq_bigdata_pretrain": "Qwen 1.7B FSQ",
    "loqi_raw": "LOQI",
    "nextmol_dmt_l_raw": "NExT-Mol DMT-L",
    "torsional_diffusion_raw": "Torsional Diffusion",
    "mcf_drugs_l_raw": "MCF drugs-L",
    "flowr_raw": "FlowR",
    "rdkit_random_raw": "RDKit random (raw)",
}

TIERS = ("fixed", "dynamic", "chembl_count")

HIT_THRESHOLDS = (
    ("casf_hit_0p25", 0.25),
    ("casf_hit_0p5", 0.5),
    ("casf_hit_0p75", 0.75),
    ("casf_hit_2p0", 2.0),
)
CLUSTER_THRESHOLDS = (
    ("mean_clusters_0p5", 0.5),
    ("mean_clusters_1p0", 1.0),
    ("mean_clusters_2p0", 2.0),
    ("mean_clusters_3p0", 3.0),
)


def plot_label(family: object, display_label: object = "") -> str:
    key = str(family)
    if key in PLOT_LABELS:
        return PLOT_LABELS[key]
    text = str(display_label or "")
    return text if text and text != "nan" else key


def filter_panel(
    frame: pd.DataFrame,
    *,
    ligand_set: str,
    tier: str,
    family: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "row_type" in out.columns:
        out = out[out["row_type"].astype(str) == "generation"]
    if "ligand_set" in out.columns:
        out = out[out["ligand_set"].astype(str) == str(ligand_set)]
    if "family" in out.columns:
        out = out[out["family"].astype(str).isin(PANEL_FAMILIES)]
        if family != "All":
            out = out[out["family"].astype(str) == str(family)]
    if tier != "All" and "tier" in out.columns:
        out = out[out["tier"].astype(str) == str(tier)]
    if out.empty:
        return out
    out = out.copy()
    out["plot_label"] = [
        plot_label(row.get("family"), row.get("display_label"))
        for row in out.to_dict("records")
    ]
    order = {name: index for index, name in enumerate(PANEL_FAMILIES)}
    out["_order"] = out["family"].map(order)
    sort_cols = [col for col in ("_order", "tier") if col in out.columns]
    return out.sort_values(sort_cols).drop(columns="_order")


def available_thresholds(
    frame: pd.DataFrame, spec: tuple[tuple[str, float], ...]
) -> list[tuple[str, float]]:
    found = []
    for column, threshold in spec:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            found.append((column, threshold))
    return found


def melt_thresholds(
    frame: pd.DataFrame, spec: tuple[tuple[str, float], ...] | list[tuple[str, float]]
) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        for column, threshold in spec:
            if column not in row:
                continue
            value = row[column]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            rows.append(
                {
                    "method": row.get("plot_label") or plot_label(row.get("family"), row.get("display_label")),
                    "tier": str(row.get("tier", "")),
                    "threshold": float(threshold),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def threshold_chart(long: pd.DataFrame, *, ylabel: str, title: str, method_order: list[str]):
    import altair as alt

    if long.empty:
        return None
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="RMSD threshold (Å)"),
            y=alt.Y("value:Q", title=ylabel),
            color=alt.Color("method:N", sort=method_order, title="Method"),
            tooltip=["method:N", "tier:N", "threshold:Q", "value:Q"],
        )
        .properties(title=title, height=280)
    )


def _method_order(frame: pd.DataFrame) -> list[str]:
    labels = []
    for family in PANEL_FAMILIES:
        match = frame[frame["family"].astype(str) == family]
        if match.empty:
            continue
        labels.append(str(match["plot_label"].iloc[0]))
    return labels


def _render_metric_row(
    frame: pd.DataFrame,
    *,
    spec: tuple[tuple[str, float], ...],
    ylabel: str,
    title: str,
    selected_tier: str,
) -> None:
    thresholds = available_thresholds(frame, spec)
    if not thresholds:
        return
    method_order = _method_order(frame)
    if selected_tier == "All":
        tiers = [tier for tier in TIERS if (frame["tier"].astype(str) == tier).any()]
        columns = st.columns(max(len(tiers), 1))
        for column, tier in zip(columns, tiers):
            long = melt_thresholds(frame[frame["tier"].astype(str) == tier], thresholds)
            chart = threshold_chart(
                long,
                ylabel=ylabel,
                title=f"{title} ({tier})",
                method_order=method_order,
            )
            if chart is None:
                continue
            with column:
                st.altair_chart(chart, use_container_width=True)
        return
    long = melt_thresholds(frame, thresholds)
    chart = threshold_chart(
        long,
        ylabel=ylabel,
        title=f"{title} ({selected_tier})",
        method_order=method_order,
    )
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)


def render_threshold_charts(
    comparison_rows: pd.DataFrame,
    *,
    ligand_set: str,
    tier: str,
    family: str,
) -> None:
    frame = filter_panel(comparison_rows, ligand_set=ligand_set, tier=tier, family=family)
    if frame.empty:
        return
    st.subheader("Hit rate and cluster count vs RMSD threshold")
    st.caption(
        "Crystal-pose Hit@threshold and mean greedy cluster counts at the thresholds "
        "already stored on the comparison table. Panel: Qwen 1.7B FSQ, LOQI, NExT-Mol "
        "DMT-L, Torsional Diffusion, MCF drugs-L, FlowR, RDKit random (raw). Follows "
        "the ligand-set / tier / family controls in the sidebar."
    )
    if tier != "All":
        hit_col, cluster_col = st.columns(2)
        with hit_col:
            _render_metric_row(
                frame,
                spec=HIT_THRESHOLDS,
                ylabel="Hit rate (fraction of ligands)",
                title="Hit rate",
                selected_tier=tier,
            )
        with cluster_col:
            _render_metric_row(
                frame,
                spec=CLUSTER_THRESHOLDS,
                ylabel="Mean clusters per ligand",
                title="Cluster count",
                selected_tier=tier,
            )
        return
    st.markdown("**Hit rate**")
    _render_metric_row(
        frame,
        spec=HIT_THRESHOLDS,
        ylabel="Hit rate (fraction of ligands)",
        title="Hit rate",
        selected_tier="All",
    )
    st.markdown("**Cluster count**")
    _render_metric_row(
        frame,
        spec=CLUSTER_THRESHOLDS,
        ylabel="Mean clusters per ligand",
        title="Cluster count",
        selected_tier="All",
    )
