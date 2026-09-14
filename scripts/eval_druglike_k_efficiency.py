#!/usr/bin/env python3
"""Subsample druglike generation pools and plot COV/MAT vs attempted K.

Walks the druglike entries in generation_runs.yaml, computes (or reuses cached)
ref×gen RMSD matrices, randomly samples K generated conformers without
replacement over a few seeds, and writes molecule-mean COV-R/P and MAT-R/P
curves. Missing pickles are skipped with a warning so a catalog entry without
a finished generation_results.pickle does not abort the rest.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from casf_benchmark.analysis.druglike_covmat import (
    compute_rmsd_matrices,
    load_rmsd_matrices,
    save_rmsd_matrices,
)
from casf_benchmark.analysis.druglike_k_efficiency import (
    DEFAULT_K_VALUES,
    k_rows_for_method,
    plot_coverage,
    plot_coverage_median,
    plot_mat,
    plot_mat_median,
    publish_extended_tables,
    resolve_generation_pickle,
)
from casf_benchmark.catalog import describe_run, load_generation_run_entries
from casf_benchmark.paths import DEFAULT_EXTENDED_DB, RESULTS_ROOT

DEFAULT_GENERATION_RESULTS_ROOT = Path("/mnt/weka/vtarasov/outputs/outputs/gen_results")
DEFAULT_DRUGLIKE_PICKLE = Path("/mnt/weka/vtarasov/druglike_smi.pickle")
DEFAULT_CACHE_DIR = RESULTS_ROOT / "cache" / "druglike_rmsd_matrices"
DEFAULT_TABLES_DIR = RESULTS_ROOT / "tables"
DEFAULT_FIGURES_DIR = RESULTS_ROOT / "figures"


def parse_k_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k-values must be a comma-separated list of positive integers")
    return values


def first_name(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    names = row.get("names")
    if isinstance(names, (list, tuple)) and names:
        return str(names[0])
    if names:
        return str(names)
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-pickle",
        type=Path,
        default=DEFAULT_DRUGLIKE_PICKLE,
        help="Druglike reference pickle (default: %(default)s)",
    )
    parser.add_argument(
        "--generation-results-root",
        type=Path,
        default=DEFAULT_GENERATION_RESULTS_ROOT,
        help="Root for relative YAML druglike dirs (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Per-label RMSD matrix cache (default: %(default)s)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="CSV output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="PNG output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--extended-db",
        type=Path,
        default=DEFAULT_EXTENDED_DB,
        help="Sidecar SQLite to publish extended_druglike_k_* tables (default: %(default)s)",
    )
    parser.add_argument("--k-values", type=parse_k_values, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--dmax", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=80)
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated catalog labels to restrict the run",
    )
    return parser


def selected_labels(raw: str) -> set[str] | None:
    labels = {part.strip() for part in raw.split(",") if part.strip()}
    return labels or None


def load_or_compute_matrices(
    *,
    label: str,
    gen_pickle: Path,
    ground_truth: dict,
    cache_dir: Path,
    workers: int,
) -> dict[str, object]:
    cache_path = cache_dir / f"{label}.npz"
    if cache_path.is_file():
        print(f"{label}: loading cached RMSD matrices from {cache_path}", flush=True)
        return load_rmsd_matrices(cache_path)

    print(f"{label}: computing RMSD matrices from {gen_pickle}", flush=True)
    with gen_pickle.open("rb") as handle:
        generated = pickle.load(handle)
    matrices, missing, all_nan = compute_rmsd_matrices(ground_truth, generated, workers)
    if missing:
        print(f"WARNING: {label} missing generated confs for {len(missing)} molecule(s)", flush=True)
    if all_nan:
        print(f"WARNING: {label} all-NaN RMSD matrices for {len(all_nan)} molecule(s)", flush=True)
    save_rmsd_matrices(cache_path, matrices)
    print(f"{label}: cached RMSD matrices -> {cache_path}", flush=True)
    return matrices


def main() -> None:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    wanted = selected_labels(args.labels)

    with args.ground_truth_pickle.open("rb") as handle:
        ground_truth = pickle.load(handle)

    summary_rows: list[dict[str, object]] = []
    per_mol_rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for entry in load_generation_run_entries():
        dirname = entry.cohorts.get("druglike")
        if not dirname:
            continue
        label = entry.label
        if wanted is not None and label not in wanted:
            continue
        gen_pickle = resolve_generation_pickle(dirname, args.generation_results_root)
        if not gen_pickle.is_file():
            print(f"WARNING: skipping {label} (no generation_results.pickle at {gen_pickle})", flush=True)
            skipped.append(label)
            continue

        matrices = load_or_compute_matrices(
            label=label,
            gen_pickle=gen_pickle,
            ground_truth=ground_truth,
            cache_dir=args.cache_dir,
            workers=args.num_workers,
        )
        per_mol, summary = k_rows_for_method(
            matrices,
            args.k_values,
            n_seeds=args.n_seeds,
            seed=args.seed,
            threshold=args.threshold,
            dmax=args.dmax,
        )
        descriptors = describe_run(label, entry.descriptors)
        for row in summary:
            summary_rows.append({"label": label, **descriptors, **row})
        for row in per_mol:
            per_mol_rows.append(
                {
                    "label": label,
                    "display_label": descriptors.get("display_label", label),
                    "name": first_name(ground_truth.get(str(row["smiles"]))),
                    **row,
                }
            )
        print(f"{label}: K-curves for {len(summary)} sample sizes", flush=True)

    summary = pd.DataFrame(summary_rows)
    per_molecule = pd.DataFrame(per_mol_rows)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.tables_dir / "druglike_k_efficiency.csv"
    per_mol_path = args.tables_dir / "druglike_k_efficiency_per_molecule.csv"
    coverage_path = args.figures_dir / "druglike_k_coverage.png"
    mat_path = args.figures_dir / "druglike_k_mat.png"
    coverage_median_path = args.figures_dir / "druglike_k_coverage_median.png"
    mat_median_path = args.figures_dir / "druglike_k_mat_median.png"
    summary.to_csv(summary_path, index=False)
    per_molecule.to_csv(per_mol_path, index=False)
    plot_coverage(summary, coverage_path)
    plot_mat(summary, mat_path)
    plot_coverage_median(summary, coverage_median_path)
    plot_mat_median(summary, mat_median_path)
    publish_extended_tables(args.extended_db, summary, per_molecule)

    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {per_mol_path}", flush=True)
    print(f"Wrote {coverage_path}", flush=True)
    print(f"Wrote {mat_path}", flush=True)
    print(f"Wrote {coverage_median_path}", flush=True)
    print(f"Wrote {mat_median_path}", flush=True)
    print(
        f"Published extended_druglike_k_efficiency / "
        f"extended_druglike_k_efficiency_per_molecule -> {args.extended_db}",
        flush=True,
    )
    if skipped:
        print(f"Skipped {len(skipped)} catalog method(s): {', '.join(skipped)}", flush=True)


if __name__ == "__main__":
    main()
