"""Generate report-style figures from long-form conformer benchmark data.

The input has one row per ligand and method. Generic column names and the
equivalent dashboard names are accepted:

* ligand_id or mol_id
* method or display_label
* best_rmsd or casf_best_rmsd
* typical_rmsd or casf_median_rmsd
* energy_median and energy_std

tier and category are optional. Missing metric values are allowed; a figure
family is skipped when none of its required values are available.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALIASES: dict[str, tuple[str, ...]] = {
    "ligand_id": ("ligand_id", "mol_id"),
    "method": ("display_label", "method"),
    "best_rmsd": ("best_rmsd", "casf_best_rmsd"),
    "typical_rmsd": ("typical_rmsd", "casf_median_rmsd"),
    "energy_median": ("energy_median",),
    "energy_std": ("energy_std",),
    "tier": ("tier",),
    "category": ("category",),
}

PALETTE = {
    "ours": "#4A3AA7",
    "baseline": "#EB6834",
    "reference": "#2A78D6",
    "minimized": "#1BAF7A",
    "crystal": "#F28E2B",
    "connector": "#A9AFB8",
    "grid": "#DEE2E7",
    "text": "#303640",
}


@dataclass(frozen=True)
class ReportPlotConfig:
    """Selection and presentation options for the report plot suite."""

    reference_method: str = "CASF crystal"
    rmsd_order_method: str = "ChEMBL3D ground truth"
    tier: str | None = None
    methods: tuple[str, ...] | None = None
    category_by_method: Mapping[str, str] | None = None
    energy_ylim: tuple[float, float] | None = None
    rmsd_ylim: tuple[float, float] | None = None
    rmsd_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75)
    seed: int = 0


def load_plot_data(path: str | Path, *, table: str = "per_ligand_long") -> pd.DataFrame:
    """Load long-form plot data from CSV, Parquet, or SQLite."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Invalid SQLite table name: {table!r}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    raise ValueError(f"Unsupported input format {suffix!r}; use CSV, Parquet, or SQLite.")


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical not in out.columns:
            source = next((name for name in aliases if name in out.columns), None)
            if source is not None:
                rename[source] = canonical
    out = out.rename(columns=rename)

    # Dashboard rows carry both an internal method key and a human-readable label.
    if "display_label" in out.columns:
        out["method"] = out["display_label"]

    missing = [name for name in ("ligand_id", "method") if name not in out.columns]
    if missing:
        raise ValueError(
            "Input is missing required column(s): "
            + ", ".join(missing)
            + ". Expected one row per ligand and method."
        )

    out["ligand_id"] = out["ligand_id"].astype(str)
    out["method"] = out["method"].astype(str)
    for name in ("best_rmsd", "typical_rmsd", "energy_median", "energy_std"):
        if name in out.columns:
            out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def _select_rows(frame: pd.DataFrame, config: ReportPlotConfig) -> pd.DataFrame:
    out = _canonicalize(frame)
    always_keep = {config.reference_method, config.rmsd_order_method}

    if config.tier is not None and "tier" in out.columns:
        tier = out["tier"].astype(str)
        out = out[
            (tier == config.tier)
            | (tier == "reference")
            | out["method"].isin(always_keep)
        ]

    if config.methods is not None:
        out = out[out["method"].isin(set(config.methods) | always_keep)]

    duplicates = out.duplicated(["ligand_id", "method"], keep=False)
    if duplicates.any():
        examples = out.loc[duplicates, ["ligand_id", "method"]].drop_duplicates().head(3)
        rendered = ", ".join(
            f"{row.ligand_id}/{row.method}" for row in examples.itertuples()
        )
        raise ValueError(
            "Multiple rows remain for the same ligand and method "
            f"({rendered}). Select a tier or pre-filter the input."
        )
    return out


def _has_metric(frame: pd.DataFrame, name: str) -> bool:
    return name in frame.columns and frame[name].notna().any()


def _category(method: str, frame: pd.DataFrame, config: ReportPlotConfig) -> str:
    if config.category_by_method and method in config.category_by_method:
        return config.category_by_method[method]
    if "category" in frame.columns:
        values = frame.loc[frame["method"] == method, "category"].dropna()
        if not values.empty and str(values.iloc[0]) in PALETTE:
            return str(values.iloc[0])

    label = method.lower()
    if method == config.reference_method:
        return "crystal"
    if "qwen" in label or "our method" in label:
        return "ours"
    if "minimized" in label or "minimised" in label:
        return "minimized"
    if any(token in label for token in ("chembl", "loqi", "reference", "crystal")):
        return "reference"
    return "baseline"


def _method_color(method: str, frame: pd.DataFrame, config: ReportPlotConfig) -> str:
    return PALETTE.get(_category(method, frame, config), PALETTE["baseline"])


def _available_methods(
    frame: pd.DataFrame,
    metric: str,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    if metric not in frame.columns:
        return []
    methods = list(dict.fromkeys(frame.dropna(subset=[metric])["method"].tolist()))
    excluded = exclude or set()
    return [method for method in methods if method not in excluded]


def _panel_grid(count: int, max_columns: int) -> tuple[int, int]:
    columns = min(max_columns, max(1, count))
    return max(1, math.ceil(count / columns)), columns


def _hide_unused(axes: np.ndarray, used: int) -> None:
    for ax in np.asarray(axes).reshape(-1)[used:]:
        ax.set_visible(False)


def _robust_limits(values: pd.Series) -> tuple[float, float] | None:
    finite = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if finite.empty:
        return None
    low = float(finite.min())
    high = float(finite.max())
    pad = max((high - low) * 0.08, abs(high) * 0.01, 1.0)
    return low - pad, high + pad


def _finish(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _paired_with_reference(
    frame: pd.DataFrame,
    method: str,
    reference_method: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    reference = frame.loc[
        frame["method"] == reference_method, ["ligand_id", *columns]
    ]
    method_rows = frame.loc[frame["method"] == method, ["ligand_id", *columns]]
    return method_rows.merge(
        reference, on="ligand_id", suffixes=("_method", "_reference")
    )


def plot_energy_small_multiples(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Plot each method's per-ligand median energy with deviation bars."""

    methods = _available_methods(frame, "energy_median")
    rows, columns = _panel_grid(len(methods), 4)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.75 * columns, 3.45 * rows + 0.6),
        sharey=True,
        squeeze=False,
    )
    for ax, method in zip(axes.flat, methods):
        group = frame.loc[frame["method"] == method].dropna(
            subset=["energy_median"]
        )
        group = group.sort_values(["energy_median", "ligand_id"])
        errors = (
            group["energy_std"].fillna(0).clip(lower=0)
            if "energy_std" in group
            else None
        )
        color = _method_color(method, frame, config)
        ax.errorbar(
            np.arange(len(group)),
            group["energy_median"],
            yerr=errors,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=2.8,
            elinewidth=0.65,
            capsize=1.2,
            alpha=0.82,
        )
        ax.set_title(method, fontsize=10, color=PALETTE["text"])
        ax.set_xlabel("Ligands, sorted within method", fontsize=8)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    _hide_unused(axes, len(methods))
    axes.flat[0].set_ylabel("Median conformer energy")
    limits = config.energy_ylim or _robust_limits(frame["energy_median"])
    if limits:
        axes.flat[0].set_ylim(*limits)
    fig.suptitle("Per-ligand energy by method", fontsize=15, y=1.01)
    fig.tight_layout()
    return _finish(fig, path, 170)


def plot_energy_delta_boxplot(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Plot method energy minus matched crystal/reference energy."""

    methods = _available_methods(
        frame, "energy_median", exclude={config.reference_method}
    )
    deltas: dict[str, pd.Series] = {}
    for method in methods:
        paired = _paired_with_reference(
            frame, method, config.reference_method, ["energy_median"]
        )
        delta = (
            paired["energy_median_method"] - paired["energy_median_reference"]
        ).dropna()
        if not delta.empty:
            deltas[method] = delta
    if not deltas:
        raise ValueError(
            f"No energy rows overlap the reference method {config.reference_method!r}."
        )
    ordered = sorted(deltas, key=lambda method: float(deltas[method].median()))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(
        [deltas[method] for method in ordered],
        tick_labels=ordered,
        patch_artist=True,
        showfliers=False,
        widths=0.56,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    for box, method in zip(bp["boxes"], ordered):
        box.set_facecolor(_method_color(method, frame, config))
        box.set_alpha(0.78)

    rng = np.random.default_rng(config.seed)
    for position, method in enumerate(ordered, start=1):
        values = deltas[method].to_numpy()
        ax.scatter(
            position + rng.normal(0, 0.055, size=len(values)),
            values,
            s=10,
            color=PALETTE["text"],
            alpha=0.28,
            linewidths=0,
            zorder=3,
        )
    ax.axhline(0, color=PALETTE["crystal"], linestyle="--", linewidth=1.2)
    ax.set_ylabel(f"Median energy minus {config.reference_method}")
    ax.set_title("Energy deviation from the matched reference pose")
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _finish(fig, path, 180)


def plot_energy_vs_reference(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Pair each method and reference energy at a shared ligand position."""

    methods = _available_methods(
        frame, "energy_median", exclude={config.reference_method}
    )
    rows, columns = _panel_grid(len(methods), 5)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 3.65 * rows + 0.7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    reference = frame.loc[
        frame["method"] == config.reference_method,
        ["ligand_id", "energy_median"],
    ].dropna()
    ligand_order = reference.sort_values(
        ["energy_median", "ligand_id"]
    )["ligand_id"].tolist()
    order = {ligand_id: index for index, ligand_id in enumerate(ligand_order)}

    for ax, method in zip(axes.flat, methods):
        method_columns = ["ligand_id", "energy_median"]
        if "energy_std" in frame.columns:
            method_columns.append("energy_std")
        method_rows = frame.loc[
            frame["method"] == method, method_columns
        ].copy()
        if "energy_std" not in method_rows.columns:
            method_rows["energy_std"] = np.nan
        paired = method_rows.merge(
            reference,
            on="ligand_id",
            suffixes=("_method", "_reference"),
        )
        paired["_order"] = paired["ligand_id"].map(order)
        paired = paired.dropna(
            subset=[
                "_order",
                "energy_median_method",
                "energy_median_reference",
            ]
        ).sort_values("_order")
        x = paired["_order"].to_numpy()
        errors = pd.to_numeric(
            paired["energy_std"], errors="coerce"
        ).clip(lower=0)
        ax.errorbar(
            x,
            paired["energy_median_method"],
            yerr=errors,
            fmt="o",
            color=PALETTE["reference"],
            ecolor=PALETTE["reference"],
            markersize=2.5,
            elinewidth=0.6,
            capsize=1.0,
            alpha=0.75,
        )
        ax.scatter(
            x, paired["energy_median_reference"], s=8, color=PALETTE["crystal"]
        )
        ax.set_title(method, fontsize=10)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    _hide_unused(axes, len(methods))
    axes.flat[0].set_ylabel("Median conformer energy")
    limits = config.energy_ylim or _robust_limits(frame["energy_median"])
    if limits:
        axes.flat[0].set_ylim(*limits)
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=PALETTE["reference"], label="Method"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PALETTE["crystal"],
            label=config.reference_method,
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=2, frameon=False)
    fig.suptitle("Method energy versus matched reference energy", fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, 170)


def _rmsd_ligand_order(
    frame: pd.DataFrame, config: ReportPlotConfig
) -> list[str]:
    order_rows = frame.loc[
        frame["method"] == config.rmsd_order_method,
        ["ligand_id", "best_rmsd"],
    ].dropna()
    if order_rows.empty:
        order_rows = (
            frame.dropna(subset=["best_rmsd"])
            .groupby("ligand_id", as_index=False)["best_rmsd"]
            .median()
        )
    return order_rows.sort_values(
        ["best_rmsd", "ligand_id"]
    )["ligand_id"].tolist()


def plot_all_method_rmsd(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Overlay every method's per-ligand best RMSD on a shared order."""

    methods = _available_methods(frame, "best_rmsd")
    ligand_order = _rmsd_ligand_order(frame, config)
    order = {ligand_id: index for index, ligand_id in enumerate(ligand_order)}
    categorical = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for index, method in enumerate(methods):
        group = frame.loc[
            frame["method"] == method, ["ligand_id", "best_rmsd"]
        ].dropna()
        group["_x"] = group["ligand_id"].map(order)
        group = group.dropna(subset=["_x"])
        color = (
            _method_color(method, frame, config)
            if _category(method, frame, config) == "ours"
            else categorical(index % 10)
        )
        ax.scatter(
            group["_x"],
            group["best_rmsd"],
            s=14,
            alpha=0.68,
            color=color,
            label=method,
        )

    for threshold in (0.5, 1.0):
        ax.axhline(
            threshold,
            color=PALETTE["text"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
        )
        ax.text(
            len(ligand_order) - 1,
            threshold,
            f" {threshold:.2f} Å",
            va="bottom",
            ha="right",
            fontsize=8,
        )
    ax.set_xlabel(f"Ligands, ordered by {config.rmsd_order_method}")
    ax.set_ylabel("Best RMSD (Å)")
    ax.set_title("Per-ligand best RMSD, all methods")
    ax.set_xlim(-1, max(1, len(ligand_order)))
    if config.rmsd_ylim:
        ax.set_ylim(*config.rmsd_ylim)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncols=min(5, len(methods)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return _finish(fig, path, 175)


def plot_rmsd_small_multiples(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Show best and typical RMSD, joined per ligand, in method panels."""

    methods = _available_methods(frame, "best_rmsd")
    rows, columns = _panel_grid(len(methods), 5)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 3.65 * rows + 0.7),
        sharey=True,
        squeeze=False,
    )
    for ax, method in zip(axes.flat, methods):
        columns_present = ["ligand_id", "best_rmsd"]
        if "typical_rmsd" in frame.columns:
            columns_present.append("typical_rmsd")
        group = frame.loc[
            frame["method"] == method, columns_present
        ].dropna(subset=["best_rmsd"])
        if "typical_rmsd" not in group:
            group["typical_rmsd"] = np.nan
        group = group.sort_values(["best_rmsd", "ligand_id"])
        x = np.arange(len(group))
        paired = group.dropna(subset=["typical_rmsd"])
        paired_x = np.flatnonzero(group["typical_rmsd"].notna())
        ax.vlines(
            paired_x,
            paired["best_rmsd"],
            paired["typical_rmsd"],
            color=PALETTE["connector"],
            linewidth=0.55,
            alpha=0.7,
        )
        ax.scatter(
            x, group["best_rmsd"], s=10, color=PALETTE["reference"], zorder=3
        )
        ax.scatter(
            paired_x,
            paired["typical_rmsd"],
            s=10,
            color=PALETTE["crystal"],
            zorder=3,
        )
        for threshold in config.rmsd_thresholds:
            ax.axhline(
                threshold,
                color=PALETTE["text"],
                linestyle="--",
                linewidth=0.7,
                alpha=0.38,
            )
        ax.set_title(method, fontsize=10)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    _hide_unused(axes, len(methods))
    axes.flat[0].set_ylabel("RMSD (Å)")
    if config.rmsd_ylim:
        axes.flat[0].set_ylim(*config.rmsd_ylim)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PALETTE["reference"],
            label="Best conformer",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PALETTE["crystal"],
            label="Typical conformer",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=2, frameon=False)
    fig.suptitle("Best versus typical RMSD by ligand", fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, 170)


def plot_rmsd_boxplot(
    frame: pd.DataFrame,
    path: Path,
    config: ReportPlotConfig,
) -> Path:
    """Plot the distribution of per-ligand best RMSD for each method."""

    methods = _available_methods(frame, "best_rmsd")
    values = {
        method: frame.loc[frame["method"] == method, "best_rmsd"].dropna()
        for method in methods
    }
    ordered = sorted(methods, key=lambda method: float(values[method].median()))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(
        [values[method] for method in ordered],
        tick_labels=ordered,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "white", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    for box, method in zip(bp["boxes"], ordered):
        box.set_facecolor(_method_color(method, frame, config))
        box.set_alpha(0.8)
    rng = np.random.default_rng(config.seed)
    for position, method in enumerate(ordered, start=1):
        method_values = values[method].to_numpy()
        ax.scatter(
            position + rng.normal(0, 0.055, size=len(method_values)),
            method_values,
            s=9,
            color=PALETTE["text"],
            alpha=0.22,
            linewidths=0,
            zorder=3,
        )
    for threshold in (0.5, 1.0):
        ax.axhline(
            threshold,
            color=PALETTE["text"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.5,
        )
    ax.set_ylabel("Best RMSD (Å)")
    ax.set_title("Distribution of per-ligand best RMSD")
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    if config.rmsd_ylim:
        ax.set_ylim(*config.rmsd_ylim)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _finish(fig, path, 180)


def generate_report_plots(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: ReportPlotConfig | None = None,
) -> dict[str, Path]:
    """Generate every applicable plot and return output paths by plot name."""

    config = config or ReportPlotConfig()
    selected = _select_rows(frame, config)
    output_dir = Path(output_dir)
    outputs: dict[str, Path] = {}

    energy_plots = (
        ("energy_small_multiples", plot_energy_small_multiples),
        ("energy_delta_boxplot", plot_energy_delta_boxplot),
        ("energy_vs_reference", plot_energy_vs_reference),
    )
    if _has_metric(selected, "energy_median"):
        has_reference = (
            selected.loc[
                selected["method"] == config.reference_method, "energy_median"
            ].notna().any()
        )
        if not has_reference:
            energy_plots = energy_plots[:1]
        for name, function in energy_plots:
            outputs[name] = function(
                selected, output_dir / f"{name}.png", config
            )

    rmsd_plots = (
        ("rmsd_all_methods", plot_all_method_rmsd),
        ("rmsd_small_multiples", plot_rmsd_small_multiples),
        ("rmsd_boxplot", plot_rmsd_boxplot),
    )
    if _has_metric(selected, "best_rmsd"):
        for name, function in rmsd_plots:
            outputs[name] = function(
                selected, output_dir / f"{name}.png", config
            )

    if not outputs:
        raise ValueError(
            "No plots could be generated. Supply energy_median and/or best_rmsd values."
        )
    return outputs
