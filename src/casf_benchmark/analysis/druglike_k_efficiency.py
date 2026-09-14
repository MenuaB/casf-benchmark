"""Druglike COV/MAT vs attempted-K subsample curves.

Randomly subsample generated-conformer columns from a full ref×gen RMSD
matrix, recompute GEOM-style COV-R/P and MAT-R/P, then average over molecules
and seeds. Plot helpers and SQLite publishers are here so the dashboard can
reuse them later without re-running RMSD.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_K_VALUES = (10, 50, 100, 500, 1000)
PER_MOLECULE_METRIC_COLUMNS = ("cov_r_075", "cov_p_075", "mat_r", "mat_p")
DEFAULT_PLOT_LABELS = (
    "qwen_1p7b_fsq_bigdata_step47023",
    "loqi_druglike",
    "nextmol_dmt_l_druglike",
    "torsional_diffusion_druglike",
    "mcf_drugs_l_druglike",
    "flowr_druglike",
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0xFFFFFFFF


def resolve_generation_pickle(dirname: str, generation_results_root: Path) -> Path:
    path = Path(dirname)
    if path.is_absolute():
        return path / "generation_results.pickle"
    return generation_results_root / dirname / "generation_results.pickle"


def subsample_columns(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_gen = int(matrix.shape[1])
    if n_gen <= k:
        return matrix
    indices = rng.choice(n_gen, size=int(k), replace=False)
    return matrix[:, indices]


def k_rows_for_method(
    matrices: dict[str, np.ndarray],
    k_values: list[int],
    *,
    n_seeds: int,
    seed: int,
    threshold: float = 0.75,
    dmax: float = 3.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return seed-averaged per-molecule rows and molecule-mean summary rows."""
    per_molecule: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    smiles_order = list(matrices)
    from casf_benchmark.analysis.druglike_covmat import per_molecule_metrics

    for k in k_values:
        k = int(k)
        seed_metrics: dict[str, list[dict[str, float | int]]] = {smiles: [] for smiles in smiles_order}
        for replicate in range(int(n_seeds)):
            for smiles in smiles_order:
                rng = np.random.default_rng(stable_seed(seed, replicate, k, smiles))
                sampled = subsample_columns(matrices[smiles], k, rng)
                seed_metrics[smiles].append(per_molecule_metrics(sampled, threshold, dmax))
        mol_rows = []
        for smiles in smiles_order:
            replicates = seed_metrics[smiles]
            averaged = {
                column: float(np.nanmean([row[column] for row in replicates]))
                for column in PER_MOLECULE_METRIC_COLUMNS
            }
            n_gen = int(matrices[smiles].shape[1])
            mol_rows.append(
                {
                    "smiles": smiles,
                    "k": k,
                    "n_seeds": int(n_seeds),
                    "n_gen_confs": n_gen,
                    "n_sampled": int(min(k, n_gen)),
                    **averaged,
                }
            )
        per_molecule.extend(mol_rows)
        cov_r = [row["cov_r_075"] for row in mol_rows]
        cov_p = [row["cov_p_075"] for row in mol_rows]
        mat_r = [row["mat_r"] for row in mol_rows]
        mat_p = [row["mat_p"] for row in mol_rows]
        summary.append(
            {
                "k": k,
                "n_seeds": int(n_seeds),
                "n_molecules": len(mol_rows),
                "cov_r_mean": float(np.nanmean(cov_r)),
                "cov_p_mean": float(np.nanmean(cov_p)),
                "mat_r_mean": float(np.nanmean(mat_r)),
                "mat_p_mean": float(np.nanmean(mat_p)),
                "cov_r_median": float(np.nanmedian(cov_r)),
                "cov_p_median": float(np.nanmedian(cov_p)),
                "mat_r_median": float(np.nanmedian(mat_r)),
                "mat_p_median": float(np.nanmedian(mat_p)),
            }
        )
    return per_molecule, summary


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
    summary: pd.DataFrame, labels: tuple[str, ...] = DEFAULT_PLOT_LABELS
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


def _method_color_map(labels: list[str]) -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    return {label: cmap(index % 10) for index, label in enumerate(unique)}


def _plot_metric_pair(
    summary: pd.DataFrame,
    path: Path,
    *,
    recall_col: str,
    precision_col: str,
    ylabel: str,
    title: str,
    style_note: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if summary.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No druglike K-efficiency rows", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    frame = filter_plot_summary(summary) if "plot_label" not in summary.columns else summary.copy()
    if "plot_label" not in frame.columns and not frame.empty:
        frame["plot_label"] = frame.apply(plot_display_name, axis=1)
    if frame.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No druglike K-efficiency rows", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    label_col = "plot_label"
    colors = _method_color_map(frame[label_col].astype(str).tolist())
    for method in dict.fromkeys(frame[label_col].astype(str)):
        group = frame[frame[label_col].astype(str) == method].sort_values("k")
        color = colors[str(method)]
        ax.plot(
            group["k"],
            group[recall_col],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=str(method),
        )
        ax.plot(
            group["k"],
            group[precision_col],
            color=color,
            linestyle="--",
            marker="o",
            linewidth=1.6,
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("K (generated conformers per molecule)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.text(0.01, -0.14, style_note, transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(summary: pd.DataFrame, path: Path) -> None:
    _plot_metric_pair(
        summary,
        path,
        recall_col="cov_r_mean",
        precision_col="cov_p_mean",
        ylabel="Coverage (mean over 23 molecules)",
        title="Druglike coverage vs sample size (mean)",
        style_note="Solid: COV-R (recall)    Dashed: COV-P (precision)    Higher is better",
    )


def plot_mat(summary: pd.DataFrame, path: Path) -> None:
    _plot_metric_pair(
        summary,
        path,
        recall_col="mat_r_mean",
        precision_col="mat_p_mean",
        ylabel="Matching RMSD (Å, mean over 23 molecules)",
        title="Druglike matching vs sample size (mean)",
        style_note="Solid: MAT-R (recall)    Dashed: MAT-P (precision)    Lower is better",
    )


def plot_coverage_median(summary: pd.DataFrame, path: Path) -> None:
    _plot_metric_pair(
        summary,
        path,
        recall_col="cov_r_median",
        precision_col="cov_p_median",
        ylabel="Coverage (median over 23 molecules)",
        title="Druglike coverage vs sample size (median)",
        style_note="Solid: COV-R (recall)    Dashed: COV-P (precision)    Higher is better",
    )


def plot_mat_median(summary: pd.DataFrame, path: Path) -> None:
    _plot_metric_pair(
        summary,
        path,
        recall_col="mat_r_median",
        precision_col="mat_p_median",
        ylabel="Matching RMSD (Å, median over 23 molecules)",
        title="Druglike matching vs sample size (median)",
        style_note="Solid: MAT-R (recall)    Dashed: MAT-P (precision)    Lower is better",
    )


def publish_extended_tables(
    extended_db: Path,
    summary: pd.DataFrame,
    per_molecule: pd.DataFrame,
) -> None:
    extended_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(extended_db)) as con:
        summary.to_sql("extended_druglike_k_efficiency", con, if_exists="replace", index=False)
        per_molecule.to_sql(
            "extended_druglike_k_efficiency_per_molecule",
            con,
            if_exists="replace",
            index=False,
        )
        con.commit()
