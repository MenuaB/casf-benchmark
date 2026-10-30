#!/usr/bin/env python3
"""Crystal-selection sensitivity for ChEMBL-count paper metrics.

Uses existing per-ligand RMSDs. Does not regenerate conformers or replace the
frozen casf-ref-dedup-v1 dashboard cohort.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from casf_benchmark.analysis.ref_dedup import (
    SELECTION_SEED,
    SOURCE_INTERSECTION,
    read_intersection,
    select_crystals,
)
from casf_benchmark.paths import REPO_ROOT

DEFAULT_DB = REPO_ROOT / "data/results/staging/public-baseline/casf_analysis_dashboard.sqlite"
DEFAULT_OUT = REPO_ROOT / "docs/ref_dedup_sensitivity"
N_SEEDS = 1000


def selected_ids(intersection: pd.DataFrame, seed: str) -> list[str]:
    chosen = select_crystals(intersection, seed=seed)
    return chosen.loc[chosen["selected"], "ligand_id"].astype(str).tolist()


def load_rmsd(database: Path) -> tuple[pd.DataFrame, list[str]]:
    query = """
        SELECT method, display_label, mol_id, casf_best_rmsd
        FROM per_ligand_long
        WHERE ligand_set = 'ref'
          AND (
            tier = 'chembl_count'
            OR method = 'chembl3d_gt_pb'
          )
    """
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(query, connection)
    frame["mol_id"] = frame["mol_id"].astype(str)
    frame["casf_best_rmsd"] = pd.to_numeric(frame["casf_best_rmsd"], errors="coerce")
    order = frame["mol_id"].drop_duplicates().tolist()
    return frame, order


def metrics_for_ids(wide: pd.DataFrame, labels: pd.Series, mol_ids: list[str]) -> pd.DataFrame:
    aligned = wide.reindex(columns=mol_ids)
    rmsd = aligned.to_numpy(dtype=float)
    defined = np.isfinite(rmsd)
    rows = []
    for index, method in enumerate(aligned.index):
        values = rmsd[index]
        finite = values[defined[index]]
        rows.append(
            {
                "method": method,
                "label": labels.get(method, method),
                "n_ligands": int(len(mol_ids)),
                "n_defined_rmsd": int(defined[index].sum()),
                "hit_0p5": float(np.nanmean(np.where(defined[index], values <= 0.5, 0.0))),
                "hit_0p75": float(np.nanmean(np.where(defined[index], values <= 0.75, 0.0))),
                "mean_best_rmsd": float(finite.mean()) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_methods(frame: pd.DataFrame) -> pd.Series:
    generators = frame[frame["method"].astype(str) != "chembl3d_gt_pb"].copy()
    generators = generators.sort_values(
        ["hit_0p75", "hit_0p5", "mean_best_rmsd", "method"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return pd.Series(range(1, len(generators) + 1), index=generators["method"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    intersection = read_intersection(SOURCE_INTERSECTION)
    per_ligand, _order = load_rmsd(args.database.resolve())
    wide = per_ligand.pivot(index="method", columns="mol_id", values="casf_best_rmsd")
    labels = per_ligand.drop_duplicates("method").set_index("method")["display_label"]
    primary_ids = selected_ids(intersection, SELECTION_SEED)
    primary = metrics_for_ids(wide, labels, primary_ids)
    primary.insert(0, "selection", "primary")
    records = []
    ranks = []
    primary_rank = rank_methods(primary)
    for index in range(args.seeds):
        seed = f"casf-ref-dedup-sensitivity-v1-{index:04d}"
        metrics = metrics_for_ids(wide, labels, selected_ids(intersection, seed))
        metrics.insert(0, "seed", seed)
        records.append(metrics)
        alternative = rank_methods(metrics)
        aligned = primary_rank.align(alternative, join="outer")
        tau = kendalltau(aligned[0].fillna(len(primary_rank) + 1), aligned[1].fillna(len(primary_rank) + 1)).statistic
        ranks.append(
            {
                "seed": seed,
                "kendall_tau_vs_primary": float(tau),
                "identical_order": bool(list(primary_rank.index) == list(alternative.index)),
                "same_top_method": bool(primary_rank.index[0] == alternative.index[0]),
            }
        )
        if index and index % 100 == 0:
            print(f"sensitivity selections: {index}/{args.seeds}", flush=True)
    alternatives = pd.concat(records, ignore_index=True)
    summary_rows = []
    for method, group in alternatives.groupby("method", sort=False):
        primary_row = primary.loc[primary["method"].eq(method)].iloc[0]
        row = {
            "method": method,
            "label": primary_row["label"],
            "n_defined_rmsd_primary": int(primary_row["n_defined_rmsd"]),
        }
        for metric in ("hit_0p5", "hit_0p75", "mean_best_rmsd"):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_primary"] = float(primary_row[metric])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_p2.5"] = float(values.quantile(0.025))
            row[f"{metric}_p97.5"] = float(values.quantile(0.975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("hit_0p75_primary", ascending=False, kind="mergesort")
    rank_frame = pd.DataFrame(ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.output / "primary_chembl_count_metrics.csv", index=False)
    alternatives.to_csv(args.output / "alternative_chembl_count_metrics.csv", index=False)
    summary.to_csv(args.output / "sensitivity_summary.csv", index=False)
    rank_frame.to_csv(args.output / "rank_stability.csv", index=False)
    top = summary[summary["method"].astype(str) != "chembl3d_gt_pb"].iloc[0]
    lines = [
        "# CASF ref crystal-selection sensitivity",
        "",
        "The dashboard uses one frozen selection, `casf-ref-dedup-v1` "
        f"(seed `{SELECTION_SEED}`). This audit draws {args.seeds} alternative selections "
        "with the same hash procedure and different seeds. It does not change the dashboard row.",
        "",
        "Metrics are the ChEMBL-count paper definitions on each selection's 1,044 ligands. "
        "Hit@0.5 and Hit@0.75 count a missing RMSD as a miss. Mean Best RMSD averages defined RMSDs only.",
        "",
        f"Generator rank agreement with the frozen selection: mean Kendall tau "
        f"{rank_frame['kendall_tau_vs_primary'].mean():.4f}; "
        f"identical order in {100 * rank_frame['identical_order'].mean():.1f}% of selections; "
        f"same top generator in {100 * rank_frame['same_top_method'].mean():.1f}% of selections.",
        "",
        f"Highest frozen Hit@0.75 among generators: {top['label']} "
        f"({100 * top['hit_0p75_primary']:.2f}%; alternative mean {100 * top['hit_0p75_mean']:.2f}%, "
        f"2.5–97.5% interval {100 * top['hit_0p75_p2.5']:.2f}–{100 * top['hit_0p75_p97.5']:.2f}%).",
        "",
        "Files: `primary_chembl_count_metrics.csv`, `sensitivity_summary.csv`, "
        "`alternative_chembl_count_metrics.csv`, `rank_stability.csv`.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote sensitivity audit to {args.output}")


if __name__ == "__main__":
    main()
