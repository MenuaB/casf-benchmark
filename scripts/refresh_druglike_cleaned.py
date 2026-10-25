#!/usr/bin/env python3
"""Rebuild the druglike tables on the cleaned PLINDER reference set.

Keeps PoseBusters, diversity, and energy from the existing per-molecule rows,
drops molecules that are not in the cleaned SDF directory, and replaces
COV/MAT from new reference-by-generated RMSD matrices. The coverage cutoff
stays the druglike table's strict RMSD < 0.75 Å.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from casf_benchmark.analysis.druglike_covmat import (
    aggregate,
    compute_rmsd_matrices,
    load_rmsd_matrices,
    per_molecule_metrics,
    save_rmsd_matrices,
)
from casf_benchmark.analysis.druglike_k_efficiency import (
    DEFAULT_K_VALUES,
    k_rows_for_method,
    publish_extended_tables,
    resolve_generation_pickle,
)
from casf_benchmark.analysis.plinder_panel import DEFAULT_REFERENCE_DIR, load_reference_sdf_dir
from casf_benchmark.catalog import describe_run, load_generation_run_entries
from casf_benchmark.paths import RESULTS_ROOT

DEFAULT_GENERATION_RESULTS_ROOT = Path("/mnt/weka/vtarasov/outputs/outputs/gen_results")
DEFAULT_CACHE_DIR = RESULTS_ROOT / "cache" / "druglike_cleaned_rmsd_matrices"
DEFAULT_DB = RESULTS_ROOT / "extended_casf_analysis.druglike_cleaned.sqlite"
COVERAGE_COLUMNS = (
    "min_rmsd",
    "max_rmsd",
    "avg_rmsd",
    "cov_r_075",
    "cov_p_075",
    "mat_r",
    "mat_p",
    "cmat_r",
    "cmat_p",
    "num_true_confs",
    "num_gen_confs",
    "num_valid_rmsd_pairs",
)


def first_name(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    names = row.get("names")
    if isinstance(names, (list, tuple)) and names:
        return str(names[0])
    if names:
        return str(names)
    return ""


def summed_fail_counts(values: pd.Series) -> str:
    total: Counter[str] = Counter()
    for raw in values:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        text = str(raw).strip()
        if not text or text == "nan":
            continue
        total.update({str(key): int(count) for key, count in json.loads(text).items()})
    return json.dumps(dict(total))


def matrices_for_label(
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


def overlay_coverage(per_molecule: pd.DataFrame, matrices: dict[str, object]) -> pd.DataFrame:
    """Replace reference-derived columns for rows whose SMILES is in `matrices`."""
    updated = per_molecule.copy()
    for index, smiles in updated["smiles"].astype(str).items():
        matrix = matrices.get(smiles)
        if matrix is None:
            continue
        metrics = per_molecule_metrics(matrix, 0.75, 3.0)
        for column in COVERAGE_COLUMNS:
            if column in updated.columns and column in metrics:
                updated.at[index, column] = metrics[column]
    return updated


def reaggregate_summary(
    summary: pd.DataFrame,
    per_molecule: pd.DataFrame,
    *,
    n_reference_confs: int,
    n_reference_molecules: int,
) -> pd.DataFrame:
    rows = []
    for record in summary.to_dict(orient="records"):
        label = str(record["label"])
        group = per_molecule[per_molecule["label"].astype(str) == label]
        if group.empty:
            continue
        record = dict(record)
        record.update(aggregate(group.to_dict(orient="records")))
        record["n_molecules"] = int(len(group))
        record["total_confs"] = int(group["n_generated"].sum())
        record["total_pb_pass"] = int(group["n_pb_pass"].sum())
        record["overall_pb_pass_rate"] = (
            record["total_pb_pass"] / record["total_confs"] if record["total_confs"] else float("nan")
        )
        record["mean_per_molecule_pb_pass_rate"] = float(group["pb_pass_rate"].mean())
        record["pb_check_fail_counts_json"] = summed_fail_counts(group["pb_check_fail_counts_json"])
        finite = pd.to_numeric(group["num_valid_rmsd_pairs"], errors="coerce").fillna(0)
        record["molecule_success_rate"] = float((finite > 0).mean()) if len(group) else float("nan")
        record["total_true_confs"] = int(n_reference_confs)
        record["total_molecules_ground_truth"] = int(n_reference_molecules)
        record["n_missing_molecules"] = int(n_reference_molecules - len(group))
        record["threshold"] = 0.75
        rows.append(record)
    return pd.DataFrame(rows)


def refresh_tables(
    *,
    extended_db: Path,
    ground_truth: dict,
    cache_dir: Path,
    generation_results_root: Path,
    workers: int,
) -> None:
    allowed = set(ground_truth)
    n_reference_confs = sum(len(row.get("confs", [])) for row in ground_truth.values())
    with sqlite3.connect(str(extended_db)) as connection:
        per_molecule = pd.read_sql("select * from extended_druglike_per_molecule", connection)
        summary = pd.read_sql("select * from extended_druglike_summary", connection)

    per_molecule = per_molecule[per_molecule["smiles"].astype(str).isin(allowed)].copy()
    present_labels = set(per_molecule["label"].astype(str))
    summary = summary[summary["label"].astype(str).isin(present_labels)].copy()

    by_label: dict[str, dict[str, object]] = {}
    for entry in load_generation_run_entries():
        dirname = entry.cohorts.get("druglike")
        if not dirname:
            continue
        label = entry.label
        if label not in present_labels:
            continue
        gen_pickle = resolve_generation_pickle(dirname, generation_results_root)
        if not gen_pickle.is_file():
            print(f"WARNING: skipping {label} (no generation_results.pickle at {gen_pickle})", flush=True)
            continue
        by_label[label] = matrices_for_label(
            label=label,
            gen_pickle=gen_pickle,
            ground_truth=ground_truth,
            cache_dir=cache_dir,
            workers=workers,
        )
        mask = per_molecule["label"].astype(str) == label
        covered = overlay_coverage(per_molecule.loc[mask], by_label[label])
        per_molecule.loc[covered.index, list(COVERAGE_COLUMNS)] = covered.loc[:, list(COVERAGE_COLUMNS)]

    per_molecule = per_molecule[per_molecule["label"].astype(str).isin(by_label)].copy()
    summary = summary[summary["label"].astype(str).isin(by_label)].copy()
    summary = reaggregate_summary(
        summary,
        per_molecule,
        n_reference_confs=n_reference_confs,
        n_reference_molecules=len(ground_truth),
    )

    k_summary_rows: list[dict[str, object]] = []
    k_per_rows: list[dict[str, object]] = []
    for record in summary.to_dict(orient="records"):
        label = str(record["label"])
        matrices = by_label.get(label)
        if not matrices:
            continue
        per_k, summary_k = k_rows_for_method(matrices, list(DEFAULT_K_VALUES), n_seeds=3, seed=0)
        descriptors = describe_run(label, {key: record.get(key) for key in ("generator", "display_label", "model_size", "tokenizer", "recipe", "step")})
        for row in summary_k:
            k_summary_rows.append({"label": label, **descriptors, **row})
        for row in per_k:
            k_per_rows.append(
                {
                    "label": label,
                    "display_label": descriptors.get("display_label", label),
                    "name": first_name(ground_truth.get(str(row["smiles"]))),
                    **row,
                }
            )
        print(f"{label}: druglike table updated for {len(ground_truth)} reference molecules", flush=True)

    with sqlite3.connect(str(extended_db)) as connection:
        summary.to_sql("extended_druglike_summary", connection, if_exists="replace", index=False)
        per_molecule.to_sql("extended_druglike_per_molecule", connection, if_exists="replace", index=False)
        connection.commit()
    publish_extended_tables(extended_db, pd.DataFrame(k_summary_rows), pd.DataFrame(k_per_rows))
    print(
        f"Published druglike tables for {len(summary)} methods "
        f"({len(per_molecule)} molecule rows) -> {extended_db}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--extended-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--generation-results-root", type=Path, default=DEFAULT_GENERATION_RESULTS_ROOT)
    parser.add_argument("--num-workers", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    ground_truth = load_reference_sdf_dir(args.ground_truth_dir)
    print(f"Reference molecules: {len(ground_truth)}", flush=True)
    refresh_tables(
        extended_db=args.extended_db,
        ground_truth=ground_truth,
        cache_dir=args.cache_dir,
        generation_results_root=args.generation_results_root,
        workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
