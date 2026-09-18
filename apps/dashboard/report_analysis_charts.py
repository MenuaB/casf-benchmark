"""Interactive report analysis charts for the public Streamlit dashboard.

All-method charts are reduced to exact summary statistics before being sent to
the browser. Per-ligand charts show one selectable method at a time. This keeps
the 1,236-ligand reference set responsive without sampling any analysis rows.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


REFERENCE_METHOD = "CASF crystal"
RMSD_ORDER_METHOD = "ChEMBL3D ground truth"
PANEL_COLUMNS = 4
ENERGY_Y_DOMAIN = [-200.0, 420.0]
RMSD_Y_DOMAIN = [0.0, 2.6]
REFERENCE_METHOD_ORDER = (
    "Torsion perturb (raw)",
    "Torsion (raw)",
    "Torsional Diffusion",
    "RDKit random (raw)",
    "RDKit (raw)",
    "MCF drugs-L",
    "NextMol DMT-L",
    "NExT-Mol DMT-L",
    "RDKit random (minimized)",
    "RDKit (minimized)",
    "Torsion perturb (minimized)",
    "Torsion (minimized)",
    "LOQI",
    RMSD_ORDER_METHOD,
    "ChEMBL3D GT",
)
CATEGORY_COLORS = {
    "Qwen": "#4A3AA7",
    "Other generation": "#EB6834",
    "Reference": "#2A78D6",
    "Minimized": "#1BAF7A",
}


def method_category(method: object, row_type: object = "", family: object = "") -> str:
    label = str(method).lower()
    family_label = str(family).lower()
    if str(row_type) == "reference" or any(
        token in label for token in ("chembl", "loqi", "reference", "crystal")
    ):
        return "Reference"
    if "minimized" in label or "minimised" in label or "minimized" in family_label:
        return "Minimized"
    if "qwen" in label or "qwen" in family_label:
        return "Qwen"
    return "Other generation"


def filter_report_rows(
    frame: pd.DataFrame,
    *,
    ligand_set: str,
    tier: str,
    family: str,
) -> pd.DataFrame:
    """Select one ligand set/tier/family while retaining reference rows."""

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "ligand_set" in out.columns:
        out = out[out["ligand_set"].astype(str) == str(ligand_set)]

    is_reference = (
        out["row_type"].astype(str) == "reference"
        if "row_type" in out.columns
        else pd.Series(False, index=out.index)
    )
    if tier != "All" and "tier" in out.columns:
        out = out[is_reference | (out["tier"].astype(str) == str(tier))]
        is_reference = (
            out["row_type"].astype(str) == "reference"
            if "row_type" in out.columns
            else pd.Series(False, index=out.index)
        )
    if family != "All" and "family" in out.columns:
        out = out[is_reference | (out["family"].astype(str) == str(family))]

    label_column = "display_label" if "display_label" in out.columns else "method"
    out = out.copy()
    out["plot_method"] = out[label_column].astype(str)
    out["plot_category"] = [
        method_category(
            row.get("plot_method"),
            row.get("row_type"),
            row.get("family"),
        )
        for row in out.to_dict("records")
    ]
    return out.reset_index(drop=True)


def available_tiers(
    frame: pd.DataFrame,
    *,
    ligand_set: str,
    family: str,
) -> list[str]:
    """Return generation tiers containing rows for the active view."""

    out = frame.copy()
    if "ligand_set" in out.columns:
        out = out[out["ligand_set"].astype(str) == str(ligand_set)]
    if "row_type" in out.columns:
        out = out[out["row_type"].astype(str) == "generation"]
    if family != "All" and "family" in out.columns:
        out = out[out["family"].astype(str) == str(family)]
    if "tier" not in out.columns:
        return []
    preferred = ("fixed", "dynamic", "chembl_count")
    found = set(out["tier"].dropna().astype(str))
    return [tier for tier in preferred if tier in found] + sorted(found - set(preferred))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _method_sort(summary: pd.DataFrame, metric: str, ascending: bool) -> list[str]:
    return (
        summary.dropna(subset=[metric])
        .sort_values([metric, "method"], ascending=[ascending, True])
        ["method"]
        .astype(str)
        .tolist()
    )


def aggregate_method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all available per-ligand values for each method."""

    rows: list[dict[str, object]] = []
    cluster_column = next(
        (
            name
            for name in ("greedy_clusters_0p5", "clusters_0p5")
            if name in frame.columns
        ),
        None,
    )
    for method, group in frame.groupby("plot_method", sort=False):
        best = _numeric(group, "casf_best_rmsd")
        typical = _numeric(group, "casf_median_rmsd")
        energy = _numeric(group, "energy_median")
        energy_std = _numeric(group, "energy_std")
        clusters = (
            _numeric(group, cluster_column)
            if cluster_column is not None
            else pd.Series(np.nan, index=group.index)
        )
        valid_best = best.dropna()
        valid_typical = typical.dropna()
        valid_energy = energy.dropna()
        valid_energy_std = energy_std.dropna()
        valid_clusters = clusters.dropna()
        rows.append(
            {
                "method": str(method),
                "category": str(group["plot_category"].iloc[0]),
                "n_ligands": int(group["mol_id"].nunique())
                if "mol_id" in group.columns
                else int(len(group)),
                "n_best_rmsd": int(valid_best.size),
                "mean_best_rmsd": float(valid_best.mean())
                if not valid_best.empty
                else np.nan,
                "median_best_rmsd": float(valid_best.median())
                if not valid_best.empty
                else np.nan,
                "mean_typical_rmsd": float(valid_typical.mean())
                if not valid_typical.empty
                else np.nan,
                "hit_0p5": float((valid_best <= 0.5).mean())
                if not valid_best.empty
                else np.nan,
                "hit_0p75": float((valid_best <= 0.75).mean())
                if not valid_best.empty
                else np.nan,
                "median_energy": float(valid_energy.median())
                if not valid_energy.empty
                else np.nan,
                "median_energy_std": float(valid_energy_std.median())
                if not valid_energy_std.empty
                else np.nan,
                "mean_clusters_0p5": float(valid_clusters.mean())
                if not valid_clusters.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def tukey_box_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    method_column: str = "plot_method",
    category_column: str = "plot_category",
) -> pd.DataFrame:
    """Compute exact Tukey box statistics from every finite row."""

    rows: list[dict[str, object]] = []
    if metric not in frame.columns:
        return pd.DataFrame()
    for method, group in frame.groupby(method_column, sort=False):
        values = (
            pd.to_numeric(group[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values()
        )
        if values.empty:
            continue
        q1, median, q3 = (float(value) for value in values.quantile([0.25, 0.5, 0.75]))
        iqr = q3 - q1
        inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        rows.append(
            {
                "method": str(method),
                "category": str(group[category_column].iloc[0])
                if category_column in group.columns
                else "Other generation",
                "count": int(values.size),
                "whisker_low": float(inside.min()),
                "q1": q1,
                "median": median,
                "q3": q3,
                "whisker_high": float(inside.max()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def energy_delta_rows(
    frame: pd.DataFrame,
    *,
    reference_method: str = REFERENCE_METHOD,
) -> pd.DataFrame:
    """Return matched method minus reference energy for every ligand."""

    if "energy_median" not in frame.columns or "mol_id" not in frame.columns:
        return pd.DataFrame()
    reference = frame[
        frame["plot_method"].astype(str) == str(reference_method)
    ][["mol_id", "energy_median"]].copy()
    reference["reference_energy"] = pd.to_numeric(
        reference.pop("energy_median"), errors="coerce"
    )
    methods = frame[
        frame["plot_method"].astype(str) != str(reference_method)
    ][["mol_id", "plot_method", "plot_category", "energy_median"]].copy()
    methods["energy_median"] = pd.to_numeric(
        methods["energy_median"], errors="coerce"
    )
    paired = methods.merge(reference, on="mol_id", how="inner")
    paired["energy_delta"] = (
        paired["energy_median"] - paired["reference_energy"]
    )
    return paired.dropna(
        subset=["energy_median", "reference_energy", "energy_delta"]
    ).reset_index(drop=True)


def _filter_methods(frame: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    selected = {str(method) for method in methods}
    return frame[frame["plot_method"].astype(str).isin(selected)].copy()


def report_method_order(
    methods: Iterable[str],
    *,
    reference_last: bool = False,
) -> list[str]:
    """Use the HTML report order, then append new methods deterministically."""

    unique = list(dict.fromkeys(str(method) for method in methods))
    preferred = [method for method in REFERENCE_METHOD_ORDER if method in unique]
    remaining = [method for method in unique if method not in set(preferred)]
    remaining.sort(
        key=lambda method: (
            0 if "qwen" in method.lower() else 1,
            method.casefold(),
        )
    )
    ordered = [*preferred, *remaining]
    if reference_last and REFERENCE_METHOD in ordered:
        ordered.remove(REFERENCE_METHOD)
        ordered.append(REFERENCE_METHOD)
    return ordered


def ranked_bar_chart(
    summary: pd.DataFrame,
    *,
    metric: str,
    title: str,
    axis_title: str,
    ascending: bool,
):
    import altair as alt

    data = summary.dropna(subset=[metric]).copy()
    order = _method_sort(data, metric, ascending)
    height = max(280, 22 * len(data))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{metric}:Q", title=axis_title, scale=alt.Scale(zero=False)),
            y=alt.Y("method:N", sort=order, title=None),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=list(CATEGORY_COLORS),
                    range=list(CATEGORY_COLORS.values()),
                ),
                title="Method type",
            ),
            tooltip=[
                "method:N",
                "category:N",
                "n_ligands:Q",
                alt.Tooltip(f"{metric}:Q", format=".3f"),
            ],
        )
        .properties(title=title, height=height)
    )


def box_summary_chart(
    summary: pd.DataFrame,
    *,
    title: str,
    axis_title: str,
    median_ascending: bool = True,
):
    import altair as alt

    if summary.empty:
        return None
    order = (
        summary.sort_values(
            ["median", "method"], ascending=[median_ascending, True]
        )["method"]
        .astype(str)
        .tolist()
    )
    base = alt.Chart(summary).encode(
        y=alt.Y("method:N", sort=order, title=None),
        tooltip=[
            "method:N",
            "category:N",
            "count:Q",
            alt.Tooltip("minimum:Q", format=".3f"),
            alt.Tooltip("q1:Q", format=".3f"),
            alt.Tooltip("median:Q", format=".3f"),
            alt.Tooltip("q3:Q", format=".3f"),
            alt.Tooltip("maximum:Q", format=".3f"),
        ],
    )
    whisker = base.mark_rule(color="#707782").encode(
        x=alt.X("whisker_low:Q", title=axis_title, scale=alt.Scale(zero=False)),
        x2="whisker_high:Q",
    )
    box = base.mark_bar(size=15, cornerRadius=2).encode(
        x=alt.X("q1:Q", title=axis_title, scale=alt.Scale(zero=False)),
        x2="q3:Q",
        color=alt.Color(
            "category:N",
            scale=alt.Scale(
                domain=list(CATEGORY_COLORS),
                range=list(CATEGORY_COLORS.values()),
            ),
            title="Method type",
        ),
    )
    median = base.mark_tick(color="white", thickness=2, size=15).encode(
        x=alt.X("median:Q", title=axis_title, scale=alt.Scale(zero=False))
    )
    return (whisker + box + median).properties(
        title=title,
        height=max(280, 22 * len(summary)),
    )


def diversity_recovery_chart(summary: pd.DataFrame):
    import altair as alt

    data = summary.dropna(
        subset=["mean_clusters_0p5", "mean_best_rmsd"]
    ).copy()
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_circle(opacity=0.82, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "mean_clusters_0p5:Q",
                title="Mean clusters per ligand at 0.5 Å",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "mean_best_rmsd:Q",
                title="Mean best RMSD (Å; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=list(CATEGORY_COLORS),
                    range=list(CATEGORY_COLORS.values()),
                ),
                title="Method type",
            ),
            size=alt.Size("n_best_rmsd:Q", title="Ligands"),
            tooltip=[
                "method:N",
                "category:N",
                "n_best_rmsd:Q",
                alt.Tooltip("mean_clusters_0p5:Q", format=".2f"),
                alt.Tooltip("mean_best_rmsd:Q", format=".3f"),
                alt.Tooltip("hit_0p75:Q", format=".3f"),
            ],
        )
        .properties(
            title="Geometric diversity versus bound-pose recovery",
            height=420,
        )
    )


def energy_detail_chart(frame: pd.DataFrame, method: str):
    import altair as alt

    data = frame[frame["plot_method"].astype(str) == str(method)].copy()
    data["energy_median"] = _numeric(data, "energy_median")
    data["energy_std"] = _numeric(data, "energy_std").fillna(0).clip(lower=0)
    data = data.dropna(subset=["energy_median"]).sort_values(
        ["energy_median", "mol_id"]
    )
    if data.empty:
        return None
    data["ligand_rank"] = np.arange(1, len(data) + 1)
    data["energy_low"] = data["energy_median"] - data["energy_std"]
    data["energy_high"] = data["energy_median"] + data["energy_std"]
    base = alt.Chart(data).encode(
        x=alt.X("ligand_rank:Q", title="Ligands, sorted by median energy"),
        tooltip=[
            "mol_id:N",
            alt.Tooltip("energy_median:Q", format=".3f"),
            alt.Tooltip("energy_std:Q", format=".3f"),
        ],
    )
    error = base.mark_rule(opacity=0.35, color="#2A78D6").encode(
        y=alt.Y(
            "energy_low:Q",
            title="Median energy ± standard deviation",
            scale=alt.Scale(zero=False),
        ),
        y2="energy_high:Q",
    )
    points = base.mark_circle(size=18, color="#2A78D6", opacity=0.72).encode(
        y=alt.Y(
            "energy_median:Q",
            title="Median energy ± standard deviation",
            scale=alt.Scale(zero=False),
        )
    )
    return (error + points).properties(
        title=f"Per-ligand energy — {method}",
        height=360,
    )


def energy_panel_domain(frame: pd.DataFrame) -> list[float] | None:
    """Return the shared domain used by the source HTML report."""

    values = (
        _numeric(frame, "energy_median")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return None
    return ENERGY_Y_DOMAIN.copy()


def energy_small_multiple_chart(
    frame: pd.DataFrame,
    method: str,
    *,
    y_domain: list[float] | None,
):
    """Build one panel for the report's independently sorted energy grid."""

    import altair as alt

    data = frame[frame["plot_method"].astype(str) == str(method)].copy()
    data["energy_median"] = _numeric(data, "energy_median")
    data["energy_std"] = _numeric(data, "energy_std").clip(lower=0)
    data = data.dropna(subset=["energy_median"]).sort_values(
        ["energy_median", "mol_id"]
    )
    if data.empty:
        return None
    data["ligand_rank"] = np.arange(1, len(data) + 1)
    data["energy_low"] = data["energy_median"] - data["energy_std"]
    data["energy_high"] = data["energy_median"] + data["energy_std"]
    if method == REFERENCE_METHOD:
        color = "#C8552E"
    elif method_category(method) == "Qwen":
        color = "#8A3B1E"
    else:
        color = "#2A6F97"
    y = alt.Y(
        "energy_median:Q",
        title="Median energy (kcal/mol)",
        scale=alt.Scale(domain=y_domain, zero=False),
    )
    base = alt.Chart(data).encode(
        x=alt.X(
            "ligand_rank:Q",
            title="Ligands, sorted within panel",
            scale=alt.Scale(domain=[1, max(1, len(data))]),
        ),
        tooltip=[
            "mol_id:N",
            alt.Tooltip("energy_median:Q", format=".3f", title="Median energy"),
            alt.Tooltip("energy_std:Q", format=".3f", title="Energy std"),
        ],
    )
    points = base.mark_circle(
        size=18, color=color, opacity=0.75, clip=True
    ).encode(y=y)
    chart = points
    if method != REFERENCE_METHOD and data["energy_std"].notna().any():
        error = base.mark_rule(color=color, opacity=0.35, clip=True).encode(
            y=alt.Y(
                "energy_low:Q",
                title="Median energy (kcal/mol)",
                scale=alt.Scale(domain=y_domain, zero=False),
            ),
            y2="energy_high:Q",
        )
        chart = error + points
    return chart.properties(title=method, height=240)


def energy_delta_distribution_chart(frame: pd.DataFrame):
    """Show exact energy deltas as Tukey boxes with every ligand overlaid."""

    import altair as alt

    rows = energy_delta_rows(frame)
    if rows.empty:
        return None
    summary = tukey_box_summary(
        rows,
        metric="energy_delta",
        method_column="plot_method",
        category_column="plot_category",
    )
    summary["absolute_median"] = summary["median"].abs()
    order = (
        summary.sort_values(["absolute_median", "method"])["method"]
        .astype(str)
        .tolist()
    )
    jittered: list[pd.DataFrame] = []
    for _, group in rows.sort_values(["plot_method", "mol_id"]).groupby(
        "plot_method", sort=False
    ):
        group = group.copy()
        group["jitter"] = np.linspace(-0.4, 0.4, len(group))
        jittered.append(group)
    points_data = pd.concat(jittered, ignore_index=True)

    base = alt.Chart(summary).encode(
        y=alt.Y("method:N", sort=order, title=None),
        tooltip=[
            "method:N",
            "count:Q",
            alt.Tooltip("q1:Q", format=".3f"),
            alt.Tooltip("median:Q", format=".3f"),
            alt.Tooltip("q3:Q", format=".3f"),
        ],
    )
    whisker = base.mark_rule(color="#707782").encode(
        x=alt.X(
            "whisker_low:Q",
            title=f"Median energy minus {REFERENCE_METHOD} (kcal/mol)",
            scale=alt.Scale(zero=False),
        ),
        x2="whisker_high:Q",
    )
    box = base.mark_bar(size=15, color="#2A78D6", opacity=0.78).encode(
        x="q1:Q", x2="q3:Q"
    )
    median = base.mark_tick(color="white", thickness=2, size=15).encode(
        x="median:Q"
    )
    points = (
        alt.Chart(points_data)
        .mark_circle(size=10, color="#303640", opacity=0.22)
        .encode(
            x=alt.X(
                "energy_delta:Q",
                title=f"Median energy minus {REFERENCE_METHOD} (kcal/mol)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("plot_method:N", sort=order, title=None),
            yOffset=alt.YOffset(
                "jitter:Q",
                scale=alt.Scale(domain=[-0.5, 0.5], range=[-8, 8]),
            ),
            tooltip=[
                "mol_id:N",
                "plot_method:N",
                alt.Tooltip("energy_delta:Q", format=".3f"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#F28E2B", strokeDash=[5, 4])
        .encode(x="zero:Q")
    )
    return (zero + whisker + box + median + points).properties(
        title=f"Deviation from {REFERENCE_METHOD}",
        height=max(320, 24 * len(summary)),
    )


def energy_reference_chart(
    frame: pd.DataFrame,
    method: str,
    *,
    reference_method: str = REFERENCE_METHOD,
    y_domain: list[float] | None = None,
):
    import altair as alt

    method_columns = ["mol_id", "energy_median"]
    if "energy_std" in frame.columns:
        method_columns.append("energy_std")
    method_rows = frame[
        frame["plot_method"].astype(str) == str(method)
    ][method_columns].copy()
    if "energy_std" not in method_rows.columns:
        method_rows["energy_std"] = np.nan
    reference_rows = frame[
        frame["plot_method"].astype(str) == str(reference_method)
    ][["mol_id", "energy_median"]].copy()
    paired = method_rows.merge(
        reference_rows,
        on="mol_id",
        suffixes=("_method", "_reference"),
    )
    for column in ("energy_median_method", "energy_median_reference", "energy_std"):
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
    paired = paired.dropna(
        subset=["energy_median_method", "energy_median_reference"]
    )
    if paired.empty:
        return None
    paired["energy_std"] = paired["energy_std"].clip(lower=0)
    paired["energy_low"] = paired["energy_median_method"] - paired["energy_std"]
    paired["energy_high"] = paired["energy_median_method"] + paired["energy_std"]

    reference_rows["energy_median"] = pd.to_numeric(
        reference_rows["energy_median"], errors="coerce"
    )
    reference_order = (
        reference_rows.dropna(subset=["energy_median"])
        .sort_values(["energy_median", "mol_id"])["mol_id"]
        .astype(str)
        .tolist()
    )
    order = {ligand: index + 1 for index, ligand in enumerate(reference_order)}
    paired["ligand_rank"] = paired["mol_id"].astype(str).map(order)
    paired = paired.dropna(subset=["ligand_rank"]).sort_values("ligand_rank")
    long = paired.melt(
        id_vars=["mol_id", "ligand_rank"],
        value_vars=["energy_median_method", "energy_median_reference"],
        var_name="series",
        value_name="energy_median",
    )
    long["series"] = long["series"].map(
        {
            "energy_median_method": "Method",
            "energy_median_reference": reference_method,
        }
    )
    error = (
        alt.Chart(paired.dropna(subset=["energy_std"]))
        .mark_rule(color="#2A78D6", opacity=0.35, clip=True)
        .encode(
            x=alt.X(
                "ligand_rank:Q",
                title=f"Ligands, ordered by {reference_method} energy",
                scale=alt.Scale(domain=[1, max(1, len(reference_order))]),
            ),
            y=alt.Y(
                "energy_low:Q",
                title="Median energy (kcal/mol)",
                scale=alt.Scale(domain=y_domain, zero=False),
            ),
            y2="energy_high:Q",
        )
    )
    points = (
        alt.Chart(long)
        .mark_circle(size=18, opacity=0.72, clip=True)
        .encode(
            x=alt.X(
                "ligand_rank:Q",
                title=f"Ligands, ordered by {reference_method} energy",
                scale=alt.Scale(domain=[1, max(1, len(reference_order))]),
            ),
            y=alt.Y(
                "energy_median:Q",
                title="Median energy (kcal/mol)",
                scale=alt.Scale(domain=y_domain, zero=False),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Method", reference_method],
                    range=["#2A78D6", "#F28E2B"],
                ),
                title=None,
            ),
            tooltip=[
                "mol_id:N",
                "series:N",
                alt.Tooltip("energy_median:Q", format=".3f"),
            ],
        )
    )
    return (error + points).properties(title=method, height=240)


def _rmsd_ligand_order(
    frame: pd.DataFrame,
    order_method: str = RMSD_ORDER_METHOD,
) -> list[str]:
    order_rows = frame[frame["plot_method"].astype(str) == str(order_method)][
        ["mol_id", "casf_best_rmsd"]
    ].copy()
    order_rows["casf_best_rmsd"] = _numeric(order_rows, "casf_best_rmsd")
    order_rows = order_rows.dropna(subset=["casf_best_rmsd"])
    if order_rows.empty:
        order_rows = frame[["mol_id", "casf_best_rmsd"]].copy()
        order_rows["casf_best_rmsd"] = _numeric(order_rows, "casf_best_rmsd")
        order_rows = (
            order_rows.dropna()
            .groupby("mol_id", as_index=False)["casf_best_rmsd"]
            .median()
        )
    return (
        order_rows.sort_values(["casf_best_rmsd", "mol_id"])["mol_id"]
        .astype(str)
        .tolist()
    )


def rmsd_all_methods_chart(
    frame: pd.DataFrame,
    *,
    order_frame: pd.DataFrame | None = None,
):
    """Overlay all methods on one shared ligand order without sampling."""

    import altair as alt

    data = frame[["mol_id", "plot_method", "casf_best_rmsd"]].copy()
    data["casf_best_rmsd"] = _numeric(data, "casf_best_rmsd")
    data = data.dropna(subset=["casf_best_rmsd"])
    if data.empty:
        return None
    ligand_order = _rmsd_ligand_order(
        frame if order_frame is None else order_frame
    )
    order = {ligand: index + 1 for index, ligand in enumerate(ligand_order)}
    data["ligand_rank"] = data["mol_id"].astype(str).map(order)
    data = data.dropna(subset=["ligand_rank"])
    points = (
        alt.Chart(data)
        .mark_circle(size=16, opacity=0.6)
        .encode(
            x=alt.X(
                "ligand_rank:Q",
                title=f"Ligands, ordered by best RMSD in {RMSD_ORDER_METHOD}",
                scale=alt.Scale(domain=[1, max(1, len(ligand_order))]),
            ),
            y=alt.Y(
                "casf_best_rmsd:Q",
                title="Best RMSD to bound pose (Å)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color("plot_method:N", title="Method"),
            tooltip=[
                "mol_id:N",
                "plot_method:N",
                alt.Tooltip("casf_best_rmsd:Q", format=".3f"),
            ],
        )
    )
    thresholds = (
        alt.Chart(pd.DataFrame({"threshold": [0.5, 1.0]}))
        .mark_rule(color="#303640", strokeDash=[5, 4], opacity=0.45)
        .encode(y="threshold:Q")
    )
    return (points + thresholds).properties(
        title="Per ligand: best RMSD, all methods together",
        height=440,
    )


def rmsd_panel_domain(frame: pd.DataFrame) -> list[float] | None:
    values = (
        pd.concat(
            [
                _numeric(frame, "casf_best_rmsd"),
                _numeric(frame, "casf_median_rmsd"),
            ],
            ignore_index=True,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return None
    return RMSD_Y_DOMAIN.copy()


def rmsd_distribution_chart(frame: pd.DataFrame):
    """Show exact Tukey boxes, every ligand, and the report hit thresholds."""

    import altair as alt

    data = frame[
        ["mol_id", "plot_method", "plot_category", "casf_best_rmsd"]
    ].copy()
    data["casf_best_rmsd"] = _numeric(data, "casf_best_rmsd")
    data = data.dropna(subset=["casf_best_rmsd"])
    if data.empty:
        return None
    summary = tukey_box_summary(frame, metric="casf_best_rmsd")
    order = (
        summary.sort_values(["median", "method"])["method"].astype(str).tolist()
    )
    jittered: list[pd.DataFrame] = []
    for _, group in data.sort_values(["plot_method", "mol_id"]).groupby(
        "plot_method", sort=False
    ):
        group = group.copy()
        group["jitter"] = np.linspace(-0.4, 0.4, len(group))
        jittered.append(group)
    points_data = pd.concat(jittered, ignore_index=True)

    base = alt.Chart(summary).encode(
        y=alt.Y("method:N", sort=order, title=None),
        tooltip=[
            "method:N",
            "count:Q",
            alt.Tooltip("q1:Q", format=".3f"),
            alt.Tooltip("median:Q", format=".3f"),
            alt.Tooltip("q3:Q", format=".3f"),
        ],
    )
    whisker = base.mark_rule(color="#707782").encode(
        x=alt.X(
            "whisker_low:Q",
            title="Best RMSD to bound pose (Å; lower is better)",
            scale=alt.Scale(zero=True),
        ),
        x2="whisker_high:Q",
    )
    box = base.mark_bar(size=15, color="#2A78D6", opacity=0.72).encode(
        x="q1:Q",
        x2="q3:Q",
    )
    median = base.mark_tick(color="white", thickness=2, size=15).encode(
        x="median:Q"
    )
    points = (
        alt.Chart(points_data)
        .mark_circle(size=10, color="#303640", opacity=0.22)
        .encode(
            x=alt.X(
                "casf_best_rmsd:Q",
                title="Best RMSD to bound pose (Å; lower is better)",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y("plot_method:N", sort=order, title=None),
            yOffset=alt.YOffset(
                "jitter:Q",
                scale=alt.Scale(domain=[-0.5, 0.5], range=[-8, 8]),
            ),
            tooltip=[
                "mol_id:N",
                "plot_method:N",
                alt.Tooltip("casf_best_rmsd:Q", format=".3f"),
            ],
        )
    )
    thresholds = (
        alt.Chart(pd.DataFrame({"threshold": [0.5, 1.0]}))
        .mark_rule(color="#303640", strokeDash=[5, 4], opacity=0.45)
        .encode(x="threshold:Q")
    )
    return (thresholds + whisker + box + median + points).properties(
        title="Distribution of per-ligand best RMSD",
        height=max(320, 24 * len(summary)),
    )


def rmsd_detail_chart(
    frame: pd.DataFrame,
    method: str,
    *,
    y_domain: list[float] | None = None,
):
    import altair as alt

    data = frame[frame["plot_method"].astype(str) == str(method)][
        ["mol_id", "casf_best_rmsd", "casf_median_rmsd"]
    ].copy()
    data["casf_best_rmsd"] = _numeric(data, "casf_best_rmsd")
    data["casf_median_rmsd"] = _numeric(data, "casf_median_rmsd")
    data = data.dropna(subset=["casf_best_rmsd"]).sort_values(
        ["casf_best_rmsd", "mol_id"]
    )
    if data.empty:
        return None
    data["ligand_rank"] = np.arange(1, len(data) + 1)
    paired = data.dropna(subset=["casf_median_rmsd"])
    long = data.melt(
        id_vars=["mol_id", "ligand_rank"],
        value_vars=["casf_best_rmsd", "casf_median_rmsd"],
        var_name="metric",
        value_name="rmsd",
    ).dropna()
    long["metric"] = long["metric"].map(
        {
            "casf_best_rmsd": "Best conformer",
            "casf_median_rmsd": "Typical conformer",
        }
    )
    connector = (
        alt.Chart(paired)
        .mark_rule(color="#A9AFB8", opacity=0.45)
        .encode(
            x=alt.X("ligand_rank:Q", title="Ligands, sorted by best RMSD"),
            y=alt.Y(
                "casf_best_rmsd:Q",
                title="RMSD to bound pose (Å)",
                scale=alt.Scale(domain=y_domain, zero=True),
            ),
            y2="casf_median_rmsd:Q",
        )
    )
    points = (
        alt.Chart(long)
        .mark_circle(size=20, opacity=0.72)
        .encode(
            x=alt.X("ligand_rank:Q", title="Ligands, sorted by best RMSD"),
            y=alt.Y(
                "rmsd:Q",
                title="RMSD to bound pose (Å)",
                scale=alt.Scale(domain=y_domain, zero=True),
            ),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["Best conformer", "Typical conformer"],
                    range=["#2A78D6", "#F28E2B"],
                ),
                title=None,
            ),
            tooltip=[
                "mol_id:N",
                "metric:N",
                alt.Tooltip("rmsd:Q", format=".3f"),
            ],
        )
    )
    thresholds = (
        alt.Chart(pd.DataFrame({"threshold": [0.25, 0.5, 0.75]}))
        .mark_rule(color="#303640", strokeDash=[4, 4], opacity=0.38)
        .encode(y="threshold:Q")
    )
    return (connector + points + thresholds).properties(title=method, height=260)


def render_report_analysis(
    per_ligand: pd.DataFrame,
    *,
    ligand_set: str,
    tier: str,
    family: str,
) -> None:
    """Render the all-data report analysis immediately before Extended Analysis."""

    import altair as alt

    # The reference-set view can exceed Altair's 5,000-row safety limit. These
    # figures intentionally contain every selected ligand and are never sampled.
    alt.data_transformers.disable_max_rows()

    st.divider()
    st.header("Per-ligand conformer report")
    st.caption(
        "Energy, diversity, and bound-pose recovery computed from every available "
        "per-ligand row. All-method summaries use the complete selected tier; "
        "per-ligand figures include every selected method and are never sampled."
    )

    tiers = available_tiers(
        per_ligand,
        ligand_set=ligand_set,
        family=family,
    )
    if not tiers:
        st.info("No generation tiers are available for this selection.")
        return

    if tier == "All":
        default_index = tiers.index("chembl_count") if "chembl_count" in tiers else 0
        plot_tier = st.selectbox(
            "Report tier",
            tiers,
            index=default_index,
            key="report_analysis_tier",
            help="The main sidebar is set to All; choose one tier for paired per-ligand analysis.",
        )
    else:
        plot_tier = tier

    frame = filter_report_rows(
        per_ligand,
        ligand_set=ligand_set,
        tier=plot_tier,
        family=family,
    )
    if frame.empty:
        st.info("No per-ligand rows match the active dashboard filters.")
        return

    all_methods = report_method_order(
        frame["plot_method"].dropna().astype(str).unique()
    )
    generation_methods = report_method_order(
        frame.loc[
            frame["row_type"].astype(str) == "generation",
            "plot_method",
        ].dropna().astype(str).unique()
    )
    reference_methods = [
        method
        for method in all_methods
        if method not in set(generation_methods)
    ]

    with st.expander("Report graph methods", expanded=False):
        selected_generation = st.multiselect(
            "Generation methods",
            generation_methods,
            default=generation_methods,
            key="report_analysis_methods",
            help="All matching generation methods are included by default.",
        )
        include_references = st.checkbox(
            "Include reference methods",
            value=True,
            key="report_analysis_references",
        )

    selected_methods = list(selected_generation)
    if include_references:
        selected_methods.extend(reference_methods)
    selected_methods = list(dict.fromkeys(selected_methods))
    if not selected_methods:
        st.warning("Select at least one method for the report graphs.")
        return
    selected = _filter_methods(frame, selected_methods)
    summary = aggregate_method_summary(selected)

    overview_tab, energy_tab, rmsd_tab = st.tabs(
        ["Overview", "Energy per ligand", "Bound-pose RMSD"]
    )
    with overview_tab:
        left, right = st.columns(2)
        with left:
            best = summary.dropna(subset=["mean_best_rmsd"])
            if not best.empty:
                st.altair_chart(
                    ranked_bar_chart(
                        best,
                        metric="mean_best_rmsd",
                        title="Mean best RMSD by method",
                        axis_title="Mean best RMSD (Å; lower is better)",
                        ascending=True,
                    ),
                    width="stretch",
                )
        with right:
            clusters = summary.dropna(subset=["mean_clusters_0p5"])
            if not clusters.empty:
                st.altair_chart(
                    ranked_bar_chart(
                        clusters,
                        metric="mean_clusters_0p5",
                        title="Mean geometric diversity by method",
                        axis_title="Mean clusters per ligand at 0.5 Å",
                        ascending=False,
                    ),
                    width="stretch",
                )
        scatter = diversity_recovery_chart(summary)
        if scatter is not None:
            st.altair_chart(scatter, width="stretch")

    with energy_tab:
        energy_methods = report_method_order(
            [
                method
                for method in selected_methods
                if _numeric(
                    selected[selected["plot_method"].astype(str) == method],
                    "energy_median",
                ).notna().any()
            ],
            reference_last=True,
        )
        if energy_methods:
            st.subheader("One by one, side by side")
            st.caption(
                "Each panel is sorted by its own median energy and uses the same Y axis. "
                f"The {REFERENCE_METHOD} panel is the real bound pose: one conformer per "
                "ligand, with no error bars."
            )
            energy_domain = energy_panel_domain(selected)
            columns = st.columns(PANEL_COLUMNS)
            for index, method in enumerate(energy_methods):
                with columns[index % PANEL_COLUMNS]:
                    chart = energy_small_multiple_chart(
                        selected,
                        method,
                        y_domain=energy_domain,
                    )
                    if chart is not None:
                        st.altair_chart(chart, width="stretch")

        st.subheader("All methods on one chart: deviation from CASF crystal")
        st.caption(
            "Median-energy difference from the real bound pose. Box = median and IQR; "
            "dots = every matched ligand."
        )
        chart = energy_delta_distribution_chart(selected)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        else:
            st.info(
                f"No matched {REFERENCE_METHOD} energy rows are available for this view."
            )

        comparison_methods = [
            method for method in energy_methods if method != REFERENCE_METHOD
        ]
        if comparison_methods:
            st.subheader("Each method individually against CASF crystal")
            st.caption(
                "Every panel uses the same ligand order, sorted by CASF crystal energy. "
                "Orange = real bound pose; blue = method."
            )
            columns = st.columns(PANEL_COLUMNS)
            for index, method in enumerate(comparison_methods):
                with columns[index % PANEL_COLUMNS]:
                    chart = energy_reference_chart(
                        selected,
                        method,
                        y_domain=energy_domain,
                    )
                    if chart is not None:
                        st.altair_chart(chart, width="stretch")

    with rmsd_tab:
        st.subheader("Per ligand: best RMSD, all methods together")
        st.caption(
            f"One ligand has one X position, ordered by {RMSD_ORDER_METHOD}. "
            "Dashed lines mark 0.5 Å and 1.0 Å."
        )
        chart = rmsd_all_methods_chart(selected, order_frame=frame)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

        rmsd_methods = [
            method
            for method in selected_methods
            if _numeric(
                selected[selected["plot_method"].astype(str) == method],
                "casf_best_rmsd",
            ).notna().any()
        ]
        if rmsd_methods:
            st.subheader("One by one, side by side")
            st.caption(
                "Each panel is sorted by its own best RMSD. Blue = best conformer; "
                "orange = typical conformer; gray = their gap. Dashed lines mark "
                "0.25 Å, 0.5 Å, and 0.75 Å."
            )
            rmsd_domain = rmsd_panel_domain(selected)
            columns = st.columns(PANEL_COLUMNS)
            for index, method in enumerate(rmsd_methods):
                with columns[index % PANEL_COLUMNS]:
                    chart = rmsd_detail_chart(
                        selected,
                        method,
                        y_domain=rmsd_domain,
                    )
                    if chart is not None:
                        st.altair_chart(chart, width="stretch")

        st.subheader("Overall distribution of best RMSD by method")
        st.caption(
            "Boxplot across every available ligand, sorted by median best RMSD."
        )
        chart = rmsd_distribution_chart(selected)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
