#!/usr/bin/env python3
"""Append the ref_dedup cohort to a copy of the public dashboard databases.

Existing core, ref, and Plinder rows are copied unchanged. Derived rows are
filtered from per-ligand results with the frozen selection manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from casf_benchmark.analysis.common_protocol import sampling_tables
from casf_benchmark.analysis.dataset import build_comparison_master
from casf_benchmark.analysis.ref_dedup import (
    DEFAULT_MANIFEST,
    DEFAULT_PROVENANCE,
    SELECTION_SEED,
    SELECTION_VERSION,
    DerivedLigandSet,
    apply_derived_cohort,
    load_selected_ids,
    sha256_file,
)
from casf_benchmark.cli.build_dashboard_db import build_comparison_strata
from casf_benchmark.paths import REPO_ROOT

BASELINE = REPO_ROOT / "data/results/staging/public-baseline"
STAGING = REPO_ROOT / "data/results/staging/ref-dedup-v1"
CANDIDATE_RMSD = Path(
    "/mnt/weka/mbedrosian/code/casf-benchmark-ref-protocol/data/results/ref_protocol/ref/candidate_rmsds.npz"
)
SAMPLING_PROTOCOL = REPO_ROOT / "docs/ref_protocol_analysis/sampling_protocol.json"
SPEC = DerivedLigandSet(
    id="ref_dedup",
    label="CASF ref dedup",
    source_ligand_set="ref",
    selection_manifest=DEFAULT_MANIFEST,
)


def append_aligned(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    dest_cols = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
    for column in frame.columns:
        if column not in dest_cols:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}"')
            dest_cols.append(column)
    out = frame.copy()
    for column in dest_cols:
        if column not in out.columns:
            out[column] = pd.NA
    out[dest_cols].to_sql(table, connection, if_exists="append", index=False)
    return len(out)


def _budgets(protocol: dict) -> np.ndarray:
    return np.unique(
        np.rint(
            np.r_[
                np.geomspace(protocol["min_budget"], protocol["max_budget"], protocol["budget_grid_points"]),
                protocol["explicit_budgets"],
            ]
        ).astype(int)
    )


def derive_sampling(selected_ids: list[str], published: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not CANDIDATE_RMSD.is_file():
        raise FileNotFoundError(CANDIDATE_RMSD)
    archive = np.load(CANDIDATE_RMSD, allow_pickle=True)
    mol_ids = [str(mol_id) for mol_id in archive["mol_ids"]]
    index = {mol_id: position for position, mol_id in enumerate(mol_ids)}
    missing = [mol_id for mol_id in selected_ids if mol_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} selected ligands are absent from candidate RMSDs")
    protocol = json.loads(SAMPLING_PROTOCOL.read_text(encoding="utf-8"))
    budgets = _budgets(protocol)
    methods = [str(method) for method in archive["methods"]]
    positions = np.array([index[mol_id] for mol_id in selected_ids])
    full_curves, _full_frontier = sampling_tables(
        archive["rmsd"],
        archive["n_candidates"],
        methods,
        budgets,
        protocol["rmsd_thresholds"],
        protocol["target_recovery"],
        protocol["bootstrap_replicates"],
        protocol["bootstrap_seed"],
    )
    ref_curves = published[published["ligand_set"].astype(str).eq("ref")]
    if set(full_curves["method"].astype(str)) != set(ref_curves["method"].astype(str)):
        raise AssertionError("sampling methods do not match the published ref curves")
    check = full_curves.merge(ref_curves, on=["method", "budget", "threshold"], suffixes=("_new", "_old"))
    recovery_gap = (check["recovery_new"] - check["recovery_old"]).abs().max()
    if not np.isfinite(recovery_gap) or recovery_gap > 1e-9:
        raise AssertionError(f"recomputed ref sampling recovery differs from the published curves by {recovery_gap}")
    curves, frontier = sampling_tables(
        archive["rmsd"][:, positions, :],
        archive["n_candidates"][:, positions],
        methods,
        budgets,
        protocol["rmsd_thresholds"],
        protocol["target_recovery"],
        protocol["bootstrap_replicates"],
        protocol["bootstrap_seed"],
    )
    labels = ref_curves.drop_duplicates("method").set_index("method")["label"]
    curves["label"] = curves["method"].map(labels)
    frontier["label"] = frontier["method"].map(labels)
    curves["ligand_set"] = SPEC.id
    frontier["ligand_set"] = SPEC.id
    refs = np.asarray(archive["reference_best_rmsd"], dtype=float)[positions]
    order = np.sort(refs)
    threshold_at_target = float(order[int(np.ceil(len(refs) * protocol["target_recovery"])) - 1])
    reference = pd.DataFrame(
        [
            {
                "threshold": float(threshold),
                "recovery": float(np.mean(refs <= threshold)),
                "rmsd_threshold": threshold_at_target,
                "target_recovery": float(protocol["target_recovery"]),
                "n_entries": int(len(refs)),
                "ligand_set": SPEC.id,
            }
            for threshold in protocol["rmsd_thresholds"]
        ]
    )
    return curves, frontier, reference


def derive_energy(per_ligand: pd.DataFrame, selected_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = per_ligand[per_ligand["ligand_set"].astype(str).eq("ref")].copy()
    pieces = []
    for method, group in ref.groupby("method", sort=False):
        group = group.drop_duplicates("mol_id").set_index("mol_id")
        aligned = group.reindex(selected_ids)
        aligned["method"] = method
        aligned["label"] = group["label"].iloc[0]
        aligned["mol_id"] = selected_ids
        aligned["ligand_set"] = SPEC.id
        for column in ("n_retained", "n_energy", "n_failed"):
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0)
        pieces.append(aligned.reset_index(drop=True))
    energy = pd.concat(pieces, ignore_index=True)
    reference = energy.loc[energy["method"].astype(str).eq("chembl3d_gt_pb")].set_index("mol_id")["mean_energy"]
    rows = []
    for method, all_rows in energy.groupby("method", sort=False):
        valid = all_rows[pd.to_numeric(all_rows["n_energy"], errors="coerce").gt(0)]
        delta = (valid.set_index("mol_id")["mean_energy"] - reference).median()
        rows.append(
            {
                "method": method,
                "label": all_rows["label"].iloc[0],
                "n_molecules": int(len(valid)),
                "n_conformers": int(pd.to_numeric(valid["n_energy"], errors="coerce").fillna(0).sum()),
                "n_failed": int(pd.to_numeric(all_rows["n_failed"], errors="coerce").fillna(0).sum()),
                "mean_clusters_0p5": float(pd.to_numeric(valid["clusters_0p5"], errors="coerce").mean()) if len(valid) else np.nan,
                "median_energy_std": float(pd.to_numeric(valid["std_energy"], errors="coerce").median()) if len(valid) else np.nan,
                "mean_energy_std": float(pd.to_numeric(valid["std_energy"], errors="coerce").mean()) if len(valid) else np.nan,
                "median_mean_energy": float(pd.to_numeric(valid["mean_energy"], errors="coerce").median()) if len(valid) else np.nan,
                "median_mean_energy_delta_chembl": float(delta) if pd.notna(delta) else np.nan,
                "median_archived_std": float(pd.to_numeric(valid["archived_std"], errors="coerce").median()) if len(valid) else np.nan,
                "ligand_set": SPEC.id,
            }
        )
    return energy, pd.DataFrame(rows)


def relabel(frame: pd.DataFrame, selected_ids: list[str]) -> pd.DataFrame:
    if frame.empty or "ligand_set" not in frame.columns or "mol_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    chosen = set(selected_ids)
    out = frame[frame["ligand_set"].astype(str).eq("ref") & frame["mol_id"].astype(str).isin(chosen)].copy()
    out["ligand_set"] = SPEC.id
    if "run_id" in out.columns:
        out["run_id"] = out["run_id"].astype(str) + "__ref_dedup"
    return out


def derive_extended(connection: sqlite3.Connection, per_ligand: pd.DataFrame, selected_ids: list[str]) -> list[str]:
    from casf_benchmark.cli.extended_analysis import (
        _aggregate_k_ligand_rows,
        aggregate_chembl_k_rows,
        aggregate_energy_window_rows,
        build_bound_pose_difficulty,
        build_energy_window_comparisons,
        build_frontier_summary,
        build_k_efficiency_comparisons,
        build_paired_delta_table,
        build_pb_failure_tables,
    )

    unavailable: list[str] = []
    derived_per_ligand = per_ligand  # already ref_dedup rows
    paired_chembl, extreme_chembl = build_paired_delta_table(derived_per_ligand, "chembl3d_gt_pb")
    paired_rdkit, extreme_rdkit = build_paired_delta_table(derived_per_ligand, "rdkit_random_raw_fixed")
    frontier = build_frontier_summary(derived_per_ligand)
    failures, tails = build_pb_failure_tables(derived_per_ligand)
    difficulty, rescue = build_bound_pose_difficulty(derived_per_ligand)
    tables = {
        "extended_paired_delta_vs_chembl3d_pb": paired_chembl,
        "extended_paired_delta_extreme_ligands_vs_chembl3d_pb": extreme_chembl,
        "extended_paired_delta_vs_rdkit_raw": paired_rdkit,
        "extended_paired_delta_extreme_ligands_vs_rdkit_raw": extreme_rdkit,
        "extended_frontier_summary": frontier,
        "extended_pb_failure_mechanisms": failures,
        "extended_pb_failure_tail_ligands": tails,
        "extended_bound_pose_difficulty": difficulty,
        "extended_rescue_cases": rescue,
    }
    names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "extended_k_efficiency_per_ligand" in names:
        k_rows = relabel(pd.read_sql_query("SELECT * FROM extended_k_efficiency_per_ligand WHERE ligand_set = 'ref'", connection), selected_ids)
        if k_rows.empty:
            unavailable.append("extended_k_efficiency")
        else:
            summary = _aggregate_k_ligand_rows(k_rows)
            summary["n_metric_ligands"] = summary["n_ligands"]
            strata_parts = []
            for column, stratum_type in (("rotatable_bond_bin", "rotatable_bond_bin"), ("heavy_atom_bin", "heavy_atom_bin")):
                if column not in k_rows.columns:
                    continue
                strata = _aggregate_k_ligand_rows(k_rows, [column]).rename(columns={column: "stratum"})
                strata.insert(0, "stratum_type", stratum_type)
                strata_parts.append(strata)
            tables["extended_k_efficiency_per_ligand"] = k_rows
            tables["extended_k_efficiency"] = summary
            if strata_parts:
                tables["extended_k_efficiency_strata"] = pd.concat(strata_parts, ignore_index=True, sort=False)
            if "extended_per_conformer_casf_rmsd_manifest" in names:
                manifest = relabel(
                    pd.read_sql_query(
                        "SELECT * FROM extended_per_conformer_casf_rmsd_manifest WHERE ligand_set = 'ref'",
                        connection,
                    ),
                    selected_ids,
                )
                tables["extended_per_conformer_casf_rmsd_manifest"] = manifest
    else:
        unavailable.append("extended_k_efficiency")
    if "extended_energy_window_per_ligand" in names:
        energy_rows = relabel(
            pd.read_sql_query("SELECT * FROM extended_energy_window_per_ligand WHERE ligand_set = 'ref'", connection),
            selected_ids,
        )
        tables["extended_energy_window_per_ligand"] = energy_rows
        tables["extended_energy_window_summary"] = aggregate_energy_window_rows(energy_rows)
        tables["extended_energy_window_comparisons"] = build_energy_window_comparisons(energy_rows)
    else:
        unavailable.append("extended_energy_window_summary")
    if "extended_chembl_k_efficiency_per_ligand__values" in names:
        connection.execute("CREATE TEMP TABLE ref_dedup_selected(mol_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO ref_dedup_selected(mol_id) VALUES (?)",
            [(mol_id,) for mol_id in selected_ids],
        )
        chembl = pd.read_sql_query(
            """
            SELECT identity.ligand_set, identity.run_id, identity.run_label, identity.source,
                   identity.family, identity.tier, identity.method, ligands.mol_id,
                   ligands.n_available_pb, values_table.sampling_mode, values_table.seed,
                   values_table.k, values_table.n_used, values_table.casf_best_rmsd_at_k,
                   values_table.casf_hit_0p25_at_k, values_table.casf_hit_0p5_at_k,
                   values_table.casf_hit_0p75_at_k, values_table.casf_hit_2p0_at_k
            FROM extended_chembl_k_efficiency_per_ligand__values AS values_table
            JOIN extended_chembl_k_efficiency_per_ligand__ligands AS ligands
              ON ligands.identity_id = values_table.identity_id
             AND ligands.ligand_id = values_table.ligand_id
            JOIN extended_chembl_k_efficiency_per_ligand__identity AS identity
              ON identity.identity_id = values_table.identity_id
            JOIN ref_dedup_selected AS selected
              ON selected.mol_id = ligands.mol_id
            WHERE identity.ligand_set = 'ref'
            """,
            connection,
        )
        chembl["ligand_set"] = SPEC.id
        chembl["run_id"] = chembl["run_id"].astype(str) + "__ref_dedup"
        chembl_summary = aggregate_chembl_k_rows(chembl)
        if not chembl_summary.empty:
            existing = tables.get("extended_k_efficiency", pd.DataFrame())
            tables["extended_k_efficiency"] = pd.concat([existing, chembl_summary], ignore_index=True, sort=False)
        generation_rows = tables.get("extended_k_efficiency_per_ligand", pd.DataFrame())
        if not generation_rows.empty and not chembl.empty:
            comparisons = build_k_efficiency_comparisons(generation_rows, chembl)
            if not comparisons.empty:
                tables["extended_k_efficiency_comparisons"] = comparisons
    else:
        unavailable.append("extended_k_efficiency_comparisons")
    for name, frame in tables.items():
        if name not in names or frame.empty:
            unavailable.append(name)
            continue
        append_aligned(connection, name, frame)
    # The dashboard looks for a flat chembl table; the release stores a split layout.
    unavailable.append("extended_chembl_k_efficiency_per_ligand")
    return sorted(set(unavailable))


def preserved_equal(old: pd.DataFrame, new: pd.DataFrame, ligand_sets: set[str] | None) -> None:
    columns = list(old.columns)
    left = old
    right = new
    if ligand_sets is not None and "ligand_set" in old.columns:
        left = old[old["ligand_set"].astype(str).isin(ligand_sets)]
        right = new[new["ligand_set"].astype(str).isin(ligand_sets)]
    if len(left) != len(right):
        raise AssertionError(f"row count changed: {len(left)} -> {len(right)}")
    left_hash = pd.util.hash_pandas_object(left[columns], index=False).sort_values().reset_index(drop=True)
    right_hash = pd.util.hash_pandas_object(right[columns], index=False).sort_values().reset_index(drop=True)
    if not left_hash.equals(right_hash):
        raise AssertionError("preserved rows changed")


def compare_databases(old_path: Path, new_path: Path, cohort_tables: set[str] | None = None) -> None:
    with sqlite3.connect(f"file:{old_path}?mode=ro", uri=True) as old_con, sqlite3.connect(
        f"file:{new_path}?mode=ro", uri=True
    ) as new_con:
        old_tables = {row[0] for row in old_con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in sorted(old_tables):
            if table.startswith("sqlite_"):
                continue
            old = pd.read_sql_query(f'SELECT * FROM "{table}"', old_con)
            new = pd.read_sql_query(f'SELECT * FROM "{table}"', new_con)
            ligand_sets = {"core", "ref"} if "ligand_set" in old.columns else None
            if table.startswith("plinder_") or table.startswith("extended_druglike_"):
                ligand_sets = None
            preserved_equal(old, new, ligand_sets)
            print(f"preserved {table}: {len(old)} rows")


def build(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = load_selected_ids(DEFAULT_MANIFEST)
    if len(set(selected_ids)) != 1044:
        raise AssertionError("selection manifest must contain 1,044 ids")
    dashboard_out = output_dir / "casf_analysis_dashboard.sqlite"
    extended_out = output_dir / "extended_casf_analysis.sqlite"
    shutil.copy2(BASELINE / "casf_analysis_dashboard.sqlite", dashboard_out)
    shutil.copy2(BASELINE / "extended_casf_analysis.sqlite", extended_out)
    with sqlite3.connect(dashboard_out) as connection:
        per_ligand = pd.read_sql_query("SELECT * FROM per_ligand_long WHERE ligand_set = 'ref'", connection)
        derived = apply_derived_cohort(per_ligand, SPEC, selected_ids)
        universes = {method: set(group["mol_id"].astype(str)) for method, group in derived.groupby("method")}
        if set(map(frozenset, universes.values())) != {frozenset(selected_ids)}:
            raise AssertionError("derived methods do not share the selected ligand universe")
        comparison = build_comparison_master(derived, missing_as_miss_ligand_sets=frozenset({SPEC.id}))
        strata = build_comparison_strata(derived, missing_as_miss_ligand_sets=frozenset({SPEC.id}))
        if set(comparison.loc[comparison["row_type"].eq("generation"), "ligands"]) != {1044}:
            raise AssertionError("generation aggregates are not on 1,044 ligands")
        energy_per, energy_summary = derive_energy(
            pd.read_sql_query("SELECT * FROM paper_energy_per_ligand", connection),
            selected_ids,
        )
        if energy_per.groupby("method")["mol_id"].nunique().ne(1044).any():
            raise AssertionError("paper energy rows are not on 1,044 ligands")
        curves, frontier, reference = derive_sampling(
            selected_ids,
            pd.read_sql_query("SELECT * FROM paper_sampling_curves", connection),
        )
        protocol = pd.DataFrame(
            [
                {
                    "ligand_set": SPEC.id,
                    "n_entries": 1044,
                    "protocol": "derived-casf-ref-dedup-v1; common-H-MMFF94s-v1; exact-pre-PB-finite-pool-v1",
                    "provenance_json": DEFAULT_PROVENANCE.read_text(encoding="utf-8"),
                    "source_hashes_json": json.dumps(
                        {
                            "selection_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
                            "candidate_rmsds_sha256": sha256_file(CANDIDATE_RMSD),
                            "selection_seed": SELECTION_SEED,
                            "selection_version": SELECTION_VERSION,
                        }
                    ),
                }
            ]
        )
        provenance = pd.DataFrame(
            [
                {
                    "id": SPEC.id,
                    "label": SPEC.label,
                    "source_ligand_set": SPEC.source_ligand_set,
                    "selection_manifest": str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
                    "selection_key": SPEC.selection_key,
                    "selected_column": SPEC.selected_column,
                    "selection_version": SELECTION_VERSION,
                    "selection_seed": SELECTION_SEED,
                }
            ]
        )
        counts = {
            "per_ligand_long": append_aligned(connection, "per_ligand_long", derived),
            "comparison_rows": append_aligned(connection, "comparison_rows", comparison),
            "comparison_strata": append_aligned(connection, "comparison_strata", strata),
            "paper_energy_per_ligand": append_aligned(connection, "paper_energy_per_ligand", energy_per),
            "paper_energy_summary": append_aligned(connection, "paper_energy_summary", energy_summary),
            "paper_sampling_curves": append_aligned(connection, "paper_sampling_curves", curves),
            "paper_sampling_frontier": append_aligned(connection, "paper_sampling_frontier", frontier),
            "paper_sampling_reference": append_aligned(connection, "paper_sampling_reference", reference),
            "paper_protocol_runs": append_aligned(connection, "paper_protocol_runs", protocol),
        }
        provenance.to_sql("derived_ligand_sets", connection, if_exists="replace", index=False)
        counts["derived_ligand_sets"] = len(provenance)
    with sqlite3.connect(extended_out) as connection:
        unavailable = derive_extended(connection, derived, selected_ids)
    print("Comparing preserved cohorts")
    compare_databases(BASELINE / "casf_analysis_dashboard.sqlite", dashboard_out)
    compare_databases(BASELINE / "extended_casf_analysis.sqlite", extended_out)
    return {"counts": counts, "unavailable": unavailable, "dashboard": str(dashboard_out), "extended": str(extended_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STAGING)
    args = parser.parse_args()
    result = build(args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
