#!/usr/bin/env python3
"""Score PLINDER-23 generation pools and publish the dashboard tables.

Reads druglike cohorts from generation_runs.yaml. A catalog entry with no
generation_results.pickle is skipped. Completed method directories under the
output root are reused unless --force is set, and the SQLite tables are rebuilt
from every completed method so a one-label rerun does not drop the others.
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from casf_benchmark.analysis.druglike_k_efficiency import (
    DEFAULT_K_VALUES,
    k_rows_at_thresholds,
    resolve_generation_pickle,
)
from casf_benchmark.analysis.plinder_panel import (
    HEADLINE_THRESHOLDS,
    cached_rmsd_matrices,
    _empty_coverage,
    headline_coverage,
    load_panel_frames,
    method_dir,
    method_outputs_complete,
    publish_plinder_tables,
    recovery_curve,
    recovery_thresholds,
    score_conformer_pool,
    summarize_molecules,
    write_method_outputs,
)
from casf_benchmark.catalog import describe_run, load_generation_run_entries
from casf_benchmark.paths import DEFAULT_EXTENDED_DB, RESULTS_ROOT

DEFAULT_GENERATION_RESULTS_ROOT = Path("/mnt/weka/vtarasov/outputs/outputs/gen_results")
DEFAULT_DRUGLIKE_PICKLE = Path("/mnt/weka/vtarasov/druglike_smi.pickle")
DEFAULT_CACHE_DIR = RESULTS_ROOT / "cache" / "druglike_rmsd_matrices"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "plinder_panel"


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
    parser.add_argument("--ground-truth-pickle", type=Path, default=DEFAULT_DRUGLIKE_PICKLE)
    parser.add_argument("--generation-results-root", type=Path, default=DEFAULT_GENERATION_RESULTS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extended-db", type=Path, default=DEFAULT_EXTENDED_DB)
    parser.add_argument("--k-values", type=parse_k_values, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16, help="Threads for a missing RMSD matrix")
    parser.add_argument("--molecule-workers", type=int, default=8, help="Processes for clustering, PoseBusters, and energy")
    parser.add_argument("--labels", default="", help="Comma-separated catalog labels to score this run")
    parser.add_argument("--force", action="store_true", help="Recompute method directories that already exist")
    return parser


def selected_labels(raw: str) -> set[str] | None:
    labels = {part.strip() for part in raw.split(",") if part.strip()}
    return labels or None


def _init_worker() -> None:
    RDLogger.DisableLog("rdApp.*")


def score_pools(
    label: str,
    generated: dict,
    ground_truth: dict,
    *,
    molecule_workers: int,
) -> pd.DataFrame:
    tasks = []
    for smiles, row in ground_truth.items():
        mols = list(generated.get(smiles) or [])
        tasks.append((smiles, mols, first_name(row)))
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(molecule_workers), len(tasks) or 1))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(score_conformer_pool, task): task[0] for task in tasks}
        for future in as_completed(futures):
            smiles = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[{label}] {row.get('name') or smiles} n={row['n_generated']} "
                f"pb_pass={row['n_pb_pass']}",
                flush=True,
            )
    order = {smiles: index for index, smiles in enumerate(ground_truth)}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["smiles"].map(order)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def attach_coverage(per_molecule: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    coverage_rows = []
    for smiles in per_molecule["smiles"].astype(str):
        matrix = matrices.get(smiles)
        if matrix is None:
            coverage_rows.append(_empty_coverage())
        else:
            coverage_rows.append(headline_coverage(matrix, inclusive=True))
    coverage = pd.DataFrame(coverage_rows)
    return pd.concat([per_molecule.reset_index(drop=True), coverage], axis=1)


def _with_descriptors(frame: pd.DataFrame, label: str, descriptors: dict[str, object]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.insert(0, "label", label)
    for key, value in descriptors.items():
        out[key] = value
    return out


def score_label(
    *,
    label: str,
    gen_pickle: Path,
    ground_truth: dict,
    descriptors: dict[str, object],
    cache_dir: Path,
    output_dir: Path,
    k_values: list[int],
    n_seeds: int,
    seed: int,
    num_workers: int,
    molecule_workers: int,
    force: bool,
) -> None:
    destination = method_dir(output_dir, label)
    if method_outputs_complete(destination) and not force:
        print(f"{label}: reusing {destination}", flush=True)
        return

    matrices = cached_rmsd_matrices(
        label=label,
        gen_pickle=gen_pickle,
        ground_truth=ground_truth,
        cache_dir=cache_dir,
        workers=num_workers,
    )
    with gen_pickle.open("rb") as handle:
        generated = pickle.load(handle)
    per_molecule = score_pools(
        label,
        generated,
        ground_truth,
        molecule_workers=molecule_workers,
    )
    per_molecule = attach_coverage(per_molecule, matrices)
    summary = pd.DataFrame([{**{"label": label}, **descriptors, **summarize_molecules(per_molecule)}])
    per_molecule = _with_descriptors(per_molecule, label, descriptors)
    # summary already has label and descriptors
    curve = pd.DataFrame(recovery_curve(matrices, recovery_thresholds(), inclusive=True))
    curve = _with_descriptors(curve, label, descriptors)
    _per_k, k_summary = k_rows_at_thresholds(
        matrices,
        k_values,
        HEADLINE_THRESHOLDS,
        n_seeds=n_seeds,
        seed=seed,
        inclusive=True,
    )
    k_frame = _with_descriptors(pd.DataFrame(k_summary), label, descriptors)
    write_method_outputs(
        destination,
        summary=summary,
        per_molecule=per_molecule,
        recovery=curve,
        k_efficiency=k_frame,
    )
    print(f"{label}: wrote {destination}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    wanted = selected_labels(args.labels)
    with args.ground_truth_pickle.open("rb") as handle:
        ground_truth = pickle.load(handle)

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
        score_label(
            label=label,
            gen_pickle=gen_pickle,
            ground_truth=ground_truth,
            descriptors=describe_run(label, entry.descriptors),
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            k_values=args.k_values,
            n_seeds=args.n_seeds,
            seed=args.seed,
            num_workers=args.num_workers,
            molecule_workers=args.molecule_workers,
            force=args.force,
        )

    summary, per_molecule, recovery, k_efficiency = load_panel_frames(args.output_dir)
    if summary.empty:
        print("No completed PLINDER method outputs to publish.", flush=True)
    else:
        publish_plinder_tables(args.extended_db, summary, per_molecule, recovery, k_efficiency)
        print(
            f"Published {len(summary)} methods to {args.extended_db} "
            f"({', '.join(sorted(summary['label'].astype(str).unique()))})",
            flush=True,
        )
    if skipped:
        print("Skipped: " + ", ".join(skipped), flush=True)


if __name__ == "__main__":
    main()
