#!/usr/bin/env python3
"""Deterministic extended CASF analyses from the dashboard SQLite database.

This script is intentionally read-only with respect to the existing dashboard
database. It writes new CSV, Markdown, PNG, and sidecar SQLite outputs under the
requested extended-analysis directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


from casf_benchmark.paths import DEFAULT_DASHBOARD_DB, DEFAULT_EXTENDED_DB, REPO_ROOT

DEFAULT_DB = DEFAULT_DASHBOARD_DB
DEFAULT_OUTPUT = REPO_ROOT / "data" / "results" / "extended_analysis"
PAIRING_BASELINES = {
    "chembl3d_pb": "chembl3d_gt_pb",
    "rdkit_raw": "rdkit_random_raw_fixed",
}
HIT_COLUMNS = ["casf_hit_0p5", "casf_hit_0p75", "casf_hit_2p0"]
EXTENDED_HIT_THRESHOLDS = (0.25, 0.5, 0.75, 2.0)
ENERGY_WINDOWS = (("all", math.inf), ("deltaE_5", 5.0), ("deltaE_10", 10.0), ("deltaE_20", 20.0), ("deltaE_50", 50.0))
COMPACT_CHEMBL_K_TABLE = "extended_chembl_k_efficiency_per_ligand"
K_EFFICIENCY_SOURCES = (
    "loqi_raw_fixed",
    "torsional_diffusion_raw_fixed",
    "rdkit_random_raw_fixed",
    "rdkit_random_minimized_fixed",
    "torsion_raw_fixed",
    "nextmol_dmt_l_raw_fixed",
    "mcf_drugs_l_raw_fixed",
    "qwen_0p6b_bigdata_fixed",
    "qwen_0p6b_bigdata_to_revisited_fixed",
    "qwen_0p6b_fsq_bigdata_pretrain_fixed",
    "qwen_0p6b_fsq_fixed",
    "qwen_1p7b_bigdata_fixed",
    "qwen_1p7b_bigdata_to_revisited_fixed",
    "qwen_1p7b_fsq_bigdata_pretrain_fixed",
    "qwen_1p7b_fsq_fixed",
    "qwen_1p7b_revisited_fixed",
    "qwen_4b_bigdata_fixed",
    "qwen_4b_revisited_fixed",
    "qwen_0p6b_4e_from_bigdata_step29600_fixed",
    "qwen_0p6b_fsq_6e_from_bigdata_step28320_fixed",
    "qwen_0p6b_fsq_bigdata_step70534_fixed",
    "qwen_1p7b_4e_step29600_fixed",
    "qwen_1p7b_4e_from_bigdata_step26400_fixed",
    "qwen_1p7b_bigdata_step74000_fixed",
    "qwen_1p7b_fsq_6e_from_bigdata_step28320_fixed",
    "qwen_1p7b_fsq_bigdata_step47023_fixed",
    "qwen_4b_4e_step20000_fixed",
    "qwen_4b_4e_from_bigdata_step20000_fixed",
    "qwen_4b_bigdata_step110000_fixed",
)
K_EFFICIENCY_CHEMBL_BASELINE = "chembl3d_gt_pb"
IDENTITY_COLUMNS = [
    "ligand_set",
    "run_id",
    "run_label",
    "source",
    "family",
    "tier",
    "method",
]
BOND_GEOMETRY_TESTS = {"bond_angles", "bond_lengths", "bonds"}


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def extended_db(self) -> Path:
        return self.root / "extended_casf_analysis.sqlite"

    @property
    def dashboard_copy_db(self) -> Path:
        return self.root / "casf_analysis_dashboard_extended.sqlite"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def per_conformer_rmsd_parts(self) -> Path:
        return self.cache / "per_conformer_casf_rmsd_parts"

    @property
    def energy_window_parts(self) -> Path:
        return self.cache / "energy_window_parts"

    def mkdirs(self) -> None:
        for path in (self.tables, self.figures, self.reports):
            path.mkdir(parents=True, exist_ok=True)


def apply_generation_root_overrides(analysis_sources: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Redirect the per-conformer readers to the run roots listed in `path`.

    The dashboard only needs `analysis/tables/` under a run root, so it can be built
    from roots that carry nothing else. K-efficiency and the energy windows also read
    `generation/`, and this points them at a fuller copy of each run without
    rebuilding the dashboard. Runs absent from the mapping keep their dashboard root.
    """
    payload = yaml.safe_load(path.read_text()) or {}
    overrides: dict[str, str] = {
        str(run_id): str(root) for run_id, root in (payload.get("generation_roots") or {}).items()
    }
    if not overrides:
        print(f"WARNING: {path} lists no generation_roots; run roots left unchanged.", flush=True)
        return analysis_sources

    updated = analysis_sources.copy()
    run_ids = updated["run_id"].astype(str)
    updated["root"] = [
        overrides.get(run_id, root) for run_id, root in zip(run_ids, updated["root"])
    ]

    known = set(run_ids)
    print(f"Applied {len(known & set(overrides))} generation-root override(s) from {path}", flush=True)
    unknown = sorted(set(overrides) - known)
    if unknown:
        print(f"WARNING: {len(unknown)} override(s) name run_id(s) absent from the dashboard: {', '.join(unknown)}", flush=True)
    missing = sorted(known - set(overrides))
    if missing:
        print(f"WARNING: {len(missing)} run(s) have no override and keep their dashboard root: {', '.join(missing)}", flush=True)
    return updated


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)


def load_inputs(db_path: Path) -> dict[str, pd.DataFrame]:
    return {
        "per_ligand_long": read_table(db_path, "per_ligand_long"),
        "comparison_rows": read_table(db_path, "comparison_rows"),
        "analysis_sources": read_table(db_path, "analysis_sources"),
    }



def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_report(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0xFFFFFFFF


def table_preview(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame if max_rows is None else frame.head(max_rows)
    return "```text\n" + view.to_string(index=False) + "\n```"


def threshold_tag(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def quiet_rdkit_logs() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def parse_fail_counts_json(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value).strip()
        if not text or text == "{}":
            return {}
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return {}
    out: dict[str, float] = {}
    for key, count in raw.items():
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            out[str(key)] = numeric
    return out


def assign_rotatable_bin(values: Iterable[object]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 5, 10, 15, math.inf],
        labels=["0-4", "5-9", "10-14", "15+"],
        right=False,
        include_lowest=True,
    ).astype("string")


def assign_heavy_atom_bin(values: Iterable[object]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 20, 30, 40, math.inf],
        labels=["<20", "20-29", "30-39", "40+"],
        right=False,
        include_lowest=True,
    ).astype("string")


def assign_chembl_count_bin(values: Iterable[object]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 10, 50, 100, math.inf],
        labels=["1-9", "10-49", "50-99", "100+"],
        right=False,
        include_lowest=True,
    ).astype("string")


def assign_difficulty_bin(values: Iterable[object]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        numeric,
        bins=[-math.inf, 0.5, 0.75, 2.0, math.inf],
        labels=["easy", "moderate", "hard", "failed"],
        right=True,
        include_lowest=True,
    ).astype("string")


def k_efficiency_ligand_rows(
    rmsds: Iterable[object],
    k_values: list[int],
    *,
    seed: int,
    subsamples: int,
) -> list[dict[str, float]]:
    values = pd.to_numeric(pd.Series(list(rmsds)), errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    available = int(len(values))
    rows: list[dict[str, float]] = []
    if available == 0:
        for k in k_values:
            rows.append(
                {
                    "k": int(k),
                    "ligand_best_rmsd_at_k": np.nan,
                    "ligand_hit_0p25_at_k": np.nan,
                    "ligand_hit_0p5_at_k": np.nan,
                    "ligand_hit_0p75_at_k": np.nan,
                    "ligand_hit_2p0_at_k": np.nan,
                    "conformers_available": 0,
                    "ligand_has_at_least_k": 0,
                    "subsamples_used": 0,
                }
            )
        return rows

    for k in k_values:
        k = int(k)
        if available <= k:
            sampled_best = np.asarray([values.min()], dtype=float)
        else:
            replicate_best = []
            for replicate in range(subsamples):
                rng = np.random.default_rng(stable_seed(seed, k, replicate, available))
                indices = rng.choice(available, size=k, replace=False)
                replicate_best.append(float(values[indices].min()))
            sampled_best = np.asarray(replicate_best, dtype=float)
        rows.append(
            {
                "k": k,
                "ligand_best_rmsd_at_k": float(sampled_best.mean()),
                "ligand_hit_0p25_at_k": float((sampled_best <= 0.25).mean()),
                "ligand_hit_0p5_at_k": float((sampled_best <= 0.5).mean()),
                "ligand_hit_0p75_at_k": float((sampled_best <= 0.75).mean()),
                "ligand_hit_2p0_at_k": float((sampled_best <= 2.0).mean()),
                "conformers_available": available,
                "ligand_has_at_least_k": int(available >= k),
                "subsamples_used": int(len(sampled_best)),
            }
        )
    return rows


def non_dominated_frontier_ranks(
    frame: pd.DataFrame, objective_columns: list[str], maximize: list[bool]
) -> pd.Series:
    if len(objective_columns) != len(maximize):
        raise ValueError("objective_columns and maximize must have the same length")
    if frame.empty:
        return pd.Series(dtype="Int64")

    values = frame[objective_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for idx, should_maximize in enumerate(maximize):
        if not should_maximize:
            values[:, idx] = -values[:, idx]
    values = np.where(np.isfinite(values), values, -np.inf)

    ranks = np.zeros(len(frame), dtype=int)
    remaining = set(range(len(frame)))
    rank = 1
    while remaining:
        current_front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                    dominated = True
                    break
            if not dominated:
                current_front.append(i)
        for i in current_front:
            ranks[i] = rank
        remaining.difference_update(current_front)
        rank += 1
    return pd.Series(ranks, index=frame.index, dtype="Int64")


def _ligand_dir_for_set(ligand_set: str) -> Path:
    from casf_benchmark.paths import (
        DEFAULT_CORE_LIGAND_DIR,
        DEFAULT_REF_INTERSECTION_LIGAND_DIR,
    )

    return DEFAULT_REF_INTERSECTION_LIGAND_DIR if str(ligand_set) == "ref" else DEFAULT_CORE_LIGAND_DIR


def _chembl_map_for_set(ligand_set: str) -> pd.DataFrame:
    from casf_benchmark.cli.analyze_conformer_sets import load_chembl_map
    from casf_benchmark.paths import DEFAULT_CHEMBL_MAP_CSV, DEFAULT_CHEMBL_REF_MAP_CSV

    path = DEFAULT_CHEMBL_REF_MAP_CSV if str(ligand_set) == "ref" else DEFAULT_CHEMBL_MAP_CSV
    frame = load_chembl_map(path)
    frame["ligand_id"] = frame["ligand_id"].astype(str)
    return frame


def _read_conformer_part(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["conformer_index", "casf_rmsd", "energy"])
    frame = pd.read_csv(path)
    if "casf_rmsd" not in frame.columns:
        return pd.DataFrame(columns=["conformer_index", "casf_rmsd", "energy"])
    out = frame.copy()
    if "conformer_index" not in out.columns:
        out["conformer_index"] = range(len(out))
    if "energy" not in out.columns:
        out["energy"] = np.nan
    out["conformer_index"] = pd.to_numeric(out["conformer_index"], errors="coerce")
    out["casf_rmsd"] = pd.to_numeric(out["casf_rmsd"], errors="coerce")
    out["energy"] = pd.to_numeric(out["energy"], errors="coerce")
    out = out[out["casf_rmsd"].notna()].copy()
    return out.sort_values("conformer_index", kind="stable").reset_index(drop=True)


def _read_rmsd_part(path: Path) -> list[float]:
    frame = _read_conformer_part(path)
    return frame["casf_rmsd"].dropna().astype(float).tolist()


def _identity_payload_columns(payload: dict[str, object]) -> dict[str, object]:
    return {
        "ligand_set": payload["ligand_set"],
        "run_id": payload["run_id"],
        "run_label": payload["run_label"],
        "source": payload["source"],
        "family": payload["family"],
        "tier": payload["tier"],
        "method": payload["method"],
        "mol_id": payload["mol_id"],
    }


def _write_conformer_part(path: Path, payload: dict[str, object], records: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = records.copy()
    if "conformer_index" not in frame.columns:
        frame["conformer_index"] = range(len(frame))
    if "energy" not in frame.columns:
        frame["energy"] = np.nan
    for column, value in _identity_payload_columns(payload).items():
        frame[column] = value
    ordered = [
        "ligand_set",
        "run_id",
        "run_label",
        "source",
        "family",
        "tier",
        "method",
        "mol_id",
        "conformer_index",
        "casf_rmsd",
        "energy",
    ]
    frame = frame[[column for column in ordered if column in frame.columns]]
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_rmsd_part(path: Path, payload: dict[str, object], rmsds: list[float]) -> None:
    _write_conformer_part(
        path,
        payload,
        pd.DataFrame({"conformer_index": list(range(len(rmsds))), "casf_rmsd": rmsds, "energy": np.nan}),
    )


def _conformer_records_to_reference(mols: Sequence[object], reference_mol: object, conformer_indices: Sequence[int] | None = None) -> pd.DataFrame:
    from casf_benchmark.cli.analyze_conformer_sets import best_aligned_rmsd, forcefield_energy

    if conformer_indices is None:
        conformer_indices = list(range(len(mols)))
    rows = []
    for conformer_index, mol in zip(conformer_indices, mols):
        rows.append(
            {
                "conformer_index": int(conformer_index),
                "casf_rmsd": float(best_aligned_rmsd(mol, reference_mol)),
                "energy": float(forcefield_energy(mol)),
            }
        )
    return pd.DataFrame(rows)


def _rmsds_to_reference(mols: list[object], reference_mol: object) -> list[float]:
    records = _conformer_records_to_reference(mols, reference_mol)
    return records["casf_rmsd"].astype(float).tolist()


def _compute_generation_rmsds(payload: dict[str, object]) -> list[float]:
    records, _mols = _compute_generation_conformer_data(payload)
    return records["casf_rmsd"].dropna().astype(float).tolist()


def _compute_generation_conformer_data(payload: dict[str, object]) -> tuple[pd.DataFrame, list[object]]:
    from casf_benchmark.cli.analyze_conformer_sets import load_casf_ligand, load_sdf

    source = str(payload["source"])
    mol_id = str(payload["mol_id"])
    root = Path(str(payload["root"]))
    ligand_dir = Path(str(payload["ligand_dir"]))
    sdf_path = root / "generation" / source / f"{mol_id}.sdf"
    mols = load_sdf(sdf_path, required=True)
    if not mols:
        return pd.DataFrame(columns=["conformer_index", "casf_rmsd", "energy"]), []
    casf_bound = load_casf_ligand(mol_id, ligand_dir)
    return _conformer_records_to_reference(mols, casf_bound), mols


def _compute_chembl_pb_rmsds(payload: dict[str, object]) -> list[float]:
    records, _mols = _compute_chembl_pb_conformer_data(payload)
    return records["casf_rmsd"].dropna().astype(float).tolist()


def _compute_chembl_pb_conformer_data(payload: dict[str, object]) -> tuple[pd.DataFrame, list[object]]:
    from casf_benchmark.cli.analyze_conformer_sets import (
        forcefield_energy,
        get_reference_chembl_mols,
        load_casf_ligand,
        posebusters_result,
    )
    from casf_benchmark.paths import DEFAULT_CHEMBL_DATASET_ROOT

    mol_id = str(payload["mol_id"])
    ligand_dir = Path(str(payload["ligand_dir"]))
    casf_bound = load_casf_ligand(mol_id, ligand_dir)
    chembl_row = pd.Series(payload["chembl_row"])
    target_energies = [
        float(value)
        for value in payload.get("chembl_pb_energies", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    all_energies = [
        float(value)
        for value in payload.get("chembl_all_energies", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]

    def _match_positions_from_energy_table(ndigits: int) -> list[int]:
        if not target_energies or not all_energies:
            return []
        remaining = Counter(round(value, ndigits) for value in target_energies)
        positions = []
        for conformer_index, energy in enumerate(all_energies):
            key = round(float(energy), ndigits)
            if remaining[key] <= 0:
                continue
            remaining[key] -= 1
            positions.append(conformer_index)
            if len(positions) == len(target_energies):
                break
        return positions if len(positions) == len(target_energies) else []

    def _load_matched_energy_table_mols() -> tuple[pd.DataFrame, list[object]]:
        mols, _loaded_count = get_reference_chembl_mols(
            chembl_row,
            mol_id,
            DEFAULT_CHEMBL_DATASET_ROOT / "topologies",
            DEFAULT_CHEMBL_DATASET_ROOT / "zarr_database",
        )
        if not mols or len(mols) != len(all_energies):
            return pd.DataFrame(), []
        for ndigits in (8, 7, 6, 5):
            matched_positions = _match_positions_from_energy_table(ndigits)
            if not matched_positions:
                continue
            if any(position >= len(mols) for position in matched_positions):
                continue
            matched_mols = [mols[position] for position in matched_positions]
            records = _conformer_records_to_reference(matched_mols, casf_bound, matched_positions)
            records["energy"] = [all_energies[position] for position in matched_positions]
            return records, matched_mols
        return pd.DataFrame(), []

    if target_energies and all_energies:
        records, matched_mols = _load_matched_energy_table_mols()
        if len(records) == len(target_energies):
            return records, matched_mols

    chembl_mols, _loaded_count = get_reference_chembl_mols(
        chembl_row,
        mol_id,
        DEFAULT_CHEMBL_DATASET_ROOT / "topologies",
        DEFAULT_CHEMBL_DATASET_ROOT / "zarr_database",
    )
    if not chembl_mols:
        return pd.DataFrame(columns=["conformer_index", "casf_rmsd", "energy"]), []

    def _match_from_energy_table(ndigits: int) -> tuple[pd.DataFrame, list[object]]:
        if len(all_energies) != len(chembl_mols):
            return pd.DataFrame(), []
        remaining = Counter(round(value, ndigits) for value in target_energies)
        matched_indices = []
        matched_mols = []
        matched_energies = []
        for conformer_index, (mol, energy) in enumerate(zip(chembl_mols, all_energies)):
            key = round(float(energy), ndigits)
            if remaining[key] <= 0:
                continue
            remaining[key] -= 1
            matched_indices.append(conformer_index)
            matched_mols.append(mol)
            matched_energies.append(float(energy))
            if len(matched_mols) == len(target_energies):
                break
        if len(matched_mols) != len(target_energies):
            return pd.DataFrame(), []
        records = _conformer_records_to_reference(matched_mols, casf_bound, matched_indices)
        records["energy"] = matched_energies
        return records, matched_mols

    def _match_by_rounding(ndigits: int) -> tuple[pd.DataFrame, list[object]]:
        remaining = Counter(round(value, ndigits) for value in target_energies)
        matched_mols = []
        matched_indices = []
        for conformer_index, mol in enumerate(chembl_mols):
            energy = forcefield_energy(mol)
            if not math.isfinite(float(energy)):
                continue
            key = round(float(energy), ndigits)
            if remaining[key] <= 0:
                continue
            remaining[key] -= 1
            matched_mols.append(mol)
            matched_indices.append(conformer_index)
            if len(matched_mols) == len(target_energies):
                break
        return _conformer_records_to_reference(matched_mols, casf_bound, matched_indices), matched_mols

    if target_energies:
        for ndigits in (8, 7, 6, 5):
            records, matched_mols = _match_from_energy_table(ndigits)
            if len(records) == len(target_energies):
                return records, matched_mols
        for ndigits in (8, 7, 6, 5):
            records, matched_mols = _match_by_rounding(ndigits)
            if len(records) == len(target_energies):
                return records, matched_mols
        raise RuntimeError(
            f"Matched fewer ChEMBL3D-PB conformers than expected for {mol_id}: "
            f"matched={len(records)}, expected={len(target_energies)}"
        )

    pb = posebusters_result(
        chembl_mols,
        casf_bound,
        max_workers=int(payload.get("posebusters_workers", 1)),
        energy_num_threads=int(payload.get("posebusters_energy_threads", 1)),
    )
    matched = [(idx, mol) for idx, (mol, passes) in enumerate(zip(chembl_mols, pb.passes)) if passes]
    if not matched:
        return pd.DataFrame(columns=["conformer_index", "casf_rmsd", "energy"]), []
    matched_indices = [idx for idx, _mol in matched]
    matched_mols = [mol for _idx, mol in matched]
    return _conformer_records_to_reference(matched_mols, casf_bound, matched_indices), matched_mols


def _k_efficiency_worker(payload: dict[str, object]) -> list[dict[str, object]]:
    quiet_rdkit_logs()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("RDKIT_NUM_THREADS", "1")
    part_path = Path(str(payload["part_path"]))
    if part_path.exists() and not bool(payload.get("recompute", False)):
        records = _read_conformer_part(part_path)
    elif str(payload["source"]) == "chembl3d_gt_pb":
        records, _mols = _compute_chembl_pb_conformer_data(payload)
        _write_conformer_part(part_path, payload, records)
    else:
        records, _mols = _compute_generation_conformer_data(payload)
        _write_conformer_part(part_path, payload, records)

    rows = []
    rmsds = records["casf_rmsd"].dropna().astype(float).tolist()
    seed = stable_seed(payload["seed"], payload["ligand_set"], payload["source"], payload["mol_id"])
    for row in k_efficiency_ligand_rows(
        rmsds,
        list(payload["k_values"]),
        seed=seed,
        subsamples=int(payload["subsamples"]),
    ):
        rows.append(
            {
                "ligand_set": payload["ligand_set"],
                "run_id": payload["run_id"],
                "run_label": payload["run_label"],
                "source": payload["source"],
                "family": payload["family"],
                "tier": payload["tier"],
                "method": payload["method"],
                "mol_id": payload["mol_id"],
                "rotatable_bonds": payload["rotatable_bonds"],
                "heavy_atoms": payload["heavy_atoms"],
                "pb_pass_rate": payload["pb_pass_rate"],
                **row,
            }
        )
    return rows


def _k_task_payloads(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    k_values: list[int],
    subsamples: int,
    seed: int,
    recompute: bool,
) -> list[dict[str, object]]:
    root_by_run = {
        str(row["run_id"]): str(row["root"])
        for _, row in analysis_sources.iterrows()
        if pd.notna(row.get("run_id")) and pd.notna(row.get("root"))
    }
    selected = per_ligand[per_ligand["source"].astype(str).isin(K_EFFICIENCY_SOURCES)].copy()
    selected = selected[pd.to_numeric(selected["conformer_count"], errors="coerce").fillna(0) > 0]
    selected = selected[pd.to_numeric(selected["casf_best_rmsd"], errors="coerce").notna()]

    chembl_maps = {
        ligand_set: _chembl_map_for_set(ligand_set).set_index("ligand_id")
        for ligand_set in sorted(selected["ligand_set"].dropna().astype(str).unique())
    }
    chembl_pb_energy_lookup: dict[str, dict[str, list[float]]] = {}
    chembl_all_energy_lookup: dict[str, dict[str, list[float]]] = {}
    for ligand_set, frame in selected[selected["source"].astype(str) == "chembl3d_gt_pb"].groupby("ligand_set"):
        run_ids = sorted(frame["run_id"].dropna().astype(str).unique())
        if not run_ids:
            continue
        root = root_by_run.get(run_ids[0], "")
        energy_path = Path(root) / "analysis" / "tables" / "geometric_energy_values.csv"
        if not energy_path.exists():
            continue
        energies = pd.read_csv(energy_path, usecols=["mol_id", "source", "energy"])
        energies = energies[energies["source"].astype(str) == "chembl3d_gt_pb"]
        chembl_pb_energy_lookup[str(ligand_set)] = {
            str(mol_id): pd.to_numeric(group["energy"], errors="coerce").dropna().astype(float).tolist()
            for mol_id, group in energies.groupby("mol_id", sort=False)
        }
        all_energies = pd.read_csv(energy_path, usecols=["mol_id", "source", "energy"])
        all_energies = all_energies[all_energies["source"].astype(str) == "chembl3d_gt"]
        chembl_all_energy_lookup[str(ligand_set)] = {
            str(mol_id): pd.to_numeric(group["energy"], errors="coerce").dropna().astype(float).tolist()
            for mol_id, group in all_energies.groupby("mol_id", sort=False)
        }
    payloads: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        ligand_set = str(row["ligand_set"])
        source = str(row["source"])
        run_id = str(row["run_id"])
        mol_id = str(row["mol_id"])
        root = root_by_run.get(run_id, "")
        if source != "chembl3d_gt_pb" and not root:
            continue
        payload: dict[str, object] = {
            "ligand_set": ligand_set,
            "run_id": run_id,
            "run_label": row.get("run_label", ""),
            "source": source,
            "family": row.get("family", source),
            "tier": row.get("tier", ""),
            "method": row.get("method", source),
            "mol_id": mol_id,
            "rotatable_bonds": row.get("rotatable_bonds", np.nan),
            "heavy_atoms": row.get("heavy_atoms", np.nan),
            "pb_pass_rate": row.get("pb_pass_rate", np.nan),
            "root": root,
            "ligand_dir": str(_ligand_dir_for_set(ligand_set)),
            "k_values": k_values,
            "subsamples": subsamples,
            "seed": seed,
            "recompute": recompute,
            "part_path": str(paths.per_conformer_rmsd_parts / ligand_set / source / f"{mol_id}.csv"),
        }
        if source == "chembl3d_gt_pb":
            chembl_map = chembl_maps.get(ligand_set)
            if chembl_map is None or mol_id not in chembl_map.index:
                continue
            energies = chembl_pb_energy_lookup.get(ligand_set, {}).get(mol_id, [])
            if not energies:
                continue
            payload["chembl_row"] = chembl_map.loc[mol_id].to_dict()
            payload["chembl_pb_energies"] = energies
            payload["chembl_all_energies"] = chembl_all_energy_lookup.get(ligand_set, {}).get(mol_id, [])
        payloads.append(payload)
    return payloads


def _chembl_pb_energy_lookup(
    baseline: pd.DataFrame,
    analysis_sources: pd.DataFrame,
) -> dict[str, dict[str, list[float]]]:
    root_by_run = {
        str(row["run_id"]): str(row["root"])
        for _, row in analysis_sources.iterrows()
        if pd.notna(row.get("run_id")) and pd.notna(row.get("root"))
    }
    lookup: dict[str, dict[str, list[float]]] = {}
    for ligand_set, frame in baseline.groupby("ligand_set"):
        run_ids = sorted(frame["run_id"].dropna().astype(str).unique())
        if not run_ids:
            continue
        root = root_by_run.get(run_ids[0], "")
        energy_path = Path(root) / "analysis" / "tables" / "geometric_energy_values.csv"
        if not energy_path.exists():
            continue
        energies = pd.read_csv(energy_path, usecols=["mol_id", "source", "energy"])
        energies = energies[energies["source"].astype(str) == "chembl3d_gt_pb"]
        lookup[str(ligand_set)] = {
            str(mol_id): pd.to_numeric(group["energy"], errors="coerce").dropna().astype(float).tolist()
            for mol_id, group in energies.groupby("mol_id", sort=False)
        }
    return lookup


def _chembl_all_energy_lookup(
    baseline: pd.DataFrame,
    analysis_sources: pd.DataFrame,
) -> dict[str, dict[str, list[float]]]:
    root_by_run = {
        str(row["run_id"]): str(row["root"])
        for _, row in analysis_sources.iterrows()
        if pd.notna(row.get("run_id")) and pd.notna(row.get("root"))
    }
    lookup: dict[str, dict[str, list[float]]] = {}
    for ligand_set, frame in baseline.groupby("ligand_set"):
        run_ids = sorted(frame["run_id"].dropna().astype(str).unique())
        if not run_ids:
            continue
        root = root_by_run.get(run_ids[0], "")
        energy_path = Path(root) / "analysis" / "tables" / "geometric_energy_values.csv"
        if not energy_path.exists():
            continue
        energies = pd.read_csv(energy_path, usecols=["mol_id", "source", "energy"])
        energies = energies[energies["source"].astype(str) == "chembl3d_gt"]
        lookup[str(ligand_set)] = {
            str(mol_id): pd.to_numeric(group["energy"], errors="coerce").dropna().astype(float).tolist()
            for mol_id, group in energies.groupby("mol_id", sort=False)
        }
    return lookup


def _chembl_k_payloads(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    *,
    recompute: bool,
) -> list[dict[str, object]]:
    baseline = per_ligand[per_ligand["source"].astype(str) == K_EFFICIENCY_CHEMBL_BASELINE].copy()
    if baseline.empty:
        return []
    chembl_maps = {
        ligand_set: _chembl_map_for_set(ligand_set).set_index("ligand_id")
        for ligand_set in sorted(baseline["ligand_set"].dropna().astype(str).unique())
    }
    energy_lookup = _chembl_pb_energy_lookup(baseline, analysis_sources)
    all_energy_lookup = _chembl_all_energy_lookup(baseline, analysis_sources)
    payloads: list[dict[str, object]] = []
    for _, row in baseline.iterrows():
        ligand_set = str(row["ligand_set"])
        mol_id = str(row["mol_id"])
        chembl_map = chembl_maps.get(ligand_set)
        if chembl_map is None or mol_id not in chembl_map.index:
            continue
        payload: dict[str, object] = {
            "ligand_set": ligand_set,
            "run_id": row.get("run_id", f"reference_{ligand_set}"),
            "run_label": row.get("run_label", ""),
            "source": K_EFFICIENCY_CHEMBL_BASELINE,
            "family": row.get("family", K_EFFICIENCY_CHEMBL_BASELINE),
            "tier": row.get("tier", "reference"),
            "method": row.get("method", K_EFFICIENCY_CHEMBL_BASELINE),
            "mol_id": mol_id,
            "rotatable_bonds": row.get("rotatable_bonds", np.nan),
            "heavy_atoms": row.get("heavy_atoms", np.nan),
            "pb_pass_rate": row.get("pb_pass_rate", np.nan),
            "ligand_dir": str(_ligand_dir_for_set(ligand_set)),
            "chembl_row": chembl_map.loc[mol_id].to_dict(),
            "chembl_pb_energies": energy_lookup.get(ligand_set, {}).get(mol_id, []),
            "chembl_all_energies": all_energy_lookup.get(ligand_set, {}).get(mol_id, []),
            "recompute": recompute,
            "part_path": str(paths.per_conformer_rmsd_parts / ligand_set / K_EFFICIENCY_CHEMBL_BASELINE / f"{mol_id}.csv"),
        }
        payloads.append(payload)
    return payloads


def _load_or_compute_conformer_records(payload: dict[str, object]) -> pd.DataFrame:
    part_path = Path(str(payload["part_path"]))
    if part_path.exists() and not bool(payload.get("recompute", False)):
        return _read_conformer_part(part_path)
    if str(payload["source"]) == K_EFFICIENCY_CHEMBL_BASELINE:
        records, _mols = _compute_chembl_pb_conformer_data(payload)
    else:
        records, _mols = _compute_generation_conformer_data(payload)
    _write_conformer_part(part_path, payload, records)
    return records


def _chembl_k_worker(payload: dict[str, object]) -> list[dict[str, object]]:
    quiet_rdkit_logs()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("RDKIT_NUM_THREADS", "1")
    records = _load_or_compute_conformer_records(payload)
    values = pd.to_numeric(records.get("casf_rmsd", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    available = int(len(values))
    rows: list[dict[str, object]] = []
    for k in payload["k_values"]:
        k = int(k)
        n_used = min(k, available)
        first_values = values[:n_used]
        first_best = float(np.min(first_values)) if n_used else np.nan
        rows.append(
            {
                **_identity_payload_columns(payload),
                "sampling_mode": "first_k",
                "seed": np.nan,
                "k": k,
                "mol_id": payload["mol_id"],
                "rotatable_bonds": payload.get("rotatable_bonds", np.nan),
                "heavy_atoms": payload.get("heavy_atoms", np.nan),
                "n_available_pb": available,
                "n_used": n_used,
                "casf_best_rmsd_at_k": first_best,
                **{f"casf_hit_{threshold_tag(threshold)}_at_k": float(first_best <= threshold) if math.isfinite(first_best) else 0.0 for threshold in EXTENDED_HIT_THRESHOLDS},
            }
        )
        for seed in payload["random_seeds"]:
            if n_used == 0:
                best = np.nan
            elif available <= k:
                best = float(np.min(values))
            else:
                rng = np.random.default_rng(stable_seed(payload["seed_base"], payload["ligand_set"], payload["mol_id"], k, seed))
                indices = rng.choice(available, size=k, replace=False)
                best = float(np.min(values[indices]))
            rows.append(
                {
                    **_identity_payload_columns(payload),
                    "sampling_mode": "random_k",
                    "seed": int(seed),
                    "k": k,
                    "mol_id": payload["mol_id"],
                    "rotatable_bonds": payload.get("rotatable_bonds", np.nan),
                    "heavy_atoms": payload.get("heavy_atoms", np.nan),
                    "n_available_pb": available,
                    "n_used": n_used,
                    "casf_best_rmsd_at_k": best,
                    **{f"casf_hit_{threshold_tag(threshold)}_at_k": float(best <= threshold) if math.isfinite(best) else 0.0 for threshold in EXTENDED_HIT_THRESHOLDS},
                }
            )
    return rows


def _quantile_interval(values: Sequence[object]) -> tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.nan, np.nan
    return float(np.quantile(numeric, 0.025)), float(np.quantile(numeric, 0.975))


def _aggregate_chembl_seed_group(group: pd.DataFrame, summary_mode: str) -> dict[str, object]:
    available = pd.to_numeric(group["n_available_pb"], errors="coerce").fillna(0)
    k = int(group["k"].iloc[0])
    if summary_mode == "strict_at_least_k":
        metric_group = group[available >= k]
    else:
        metric_group = group
    best = pd.to_numeric(metric_group["casf_best_rmsd_at_k"], errors="coerce")
    row = {
        "summary_mode": summary_mode,
        "n_ligands": int(group["mol_id"].nunique()),
        "n_metric_ligands": int(metric_group["mol_id"].nunique()),
        "n_ligands_with_pb": int((available > 0).sum()),
        "mean_n_used": pd.to_numeric(metric_group["n_used"], errors="coerce").mean(),
        "n_ligands_with_at_least_k": int((available >= k).sum()),
        "mean_best_rmsd_at_k": best.mean(),
        "median_best_rmsd_at_k": best.median(),
        "status": "computed",
        "skip_reason": "",
    }
    for threshold in EXTENDED_HIT_THRESHOLDS:
        tag = threshold_tag(threshold)
        hits = pd.to_numeric(metric_group[f"casf_hit_{tag}_at_k"], errors="coerce")
        denominator = row["n_metric_ligands"] if summary_mode == "strict_at_least_k" else row["n_ligands"]
        row[f"hit_{tag}_at_k"] = hits.sum() / denominator if denominator else np.nan
    return row


def aggregate_chembl_k_rows(chembl_ligand_rows: pd.DataFrame) -> pd.DataFrame:
    if chembl_ligand_rows.empty:
        return pd.DataFrame()
    seed_rows: list[dict[str, object]] = []
    group_cols = [*IDENTITY_COLUMNS, "sampling_mode", "k", "seed"]
    for key, group in chembl_ligand_rows.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(group_cols, key))
        for summary_mode in ("capped_at_available", "strict_at_least_k"):
            seed_rows.append({**identity, **_aggregate_chembl_seed_group(group, summary_mode)})
    seed_frame = pd.DataFrame(seed_rows)
    summary_rows: list[dict[str, object]] = []
    summary_group_cols = [*IDENTITY_COLUMNS, "sampling_mode", "k", "summary_mode"]
    for key, group in seed_frame.groupby(summary_group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(summary_group_cols, key))
        first = group.iloc[0]
        row.update(
            {
                "n_ligands": int(first["n_ligands"]),
                "n_metric_ligands": int(first["n_metric_ligands"]),
                "n_ligands_with_pb": int(first["n_ligands_with_pb"]),
                "mean_n_used": pd.to_numeric(group["mean_n_used"], errors="coerce").mean(),
                "n_ligands_with_at_least_k": int(first["n_ligands_with_at_least_k"]),
                "mean_best_rmsd_at_k": pd.to_numeric(group["mean_best_rmsd_at_k"], errors="coerce").mean(),
                "median_best_rmsd_at_k": pd.to_numeric(group["median_best_rmsd_at_k"], errors="coerce").mean(),
                "status": "computed",
                "skip_reason": "",
            }
        )
        for threshold in EXTENDED_HIT_THRESHOLDS:
            tag = threshold_tag(threshold)
            row[f"hit_{tag}_at_k"] = pd.to_numeric(group[f"hit_{tag}_at_k"], errors="coerce").mean()
        low, high = _quantile_interval(group["hit_0p75_at_k"])
        rmsd_low, rmsd_high = _quantile_interval(group["mean_best_rmsd_at_k"])
        row["ci_metric"] = "hit_0p75_at_k"
        row["ci_low"] = low
        row["ci_high"] = high
        row["mean_best_rmsd_ci_low"] = rmsd_low
        row["mean_best_rmsd_ci_high"] = rmsd_high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_chembl_k_efficiency(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    k_values: list[int],
    *,
    seed_base: int,
    random_repeats: int,
    workers: int,
    recompute_rmsd: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = _chembl_k_payloads(per_ligand, analysis_sources, paths, recompute=recompute_rmsd)
    for payload in payloads:
        payload["k_values"] = k_values
        payload["seed_base"] = seed_base
        payload["random_seeds"] = list(range(seed_base, seed_base + random_repeats))

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_chembl_k_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - keep other ligand tasks.
                failures.append(f"{type(exc).__name__}: {exc}")
            if index % 250 == 0:
                print(f"ChEMBL K-efficiency tasks complete: {index}/{len(futures)}", flush=True)

    if failures:
        write_csv(pd.DataFrame({"failure": failures}), paths.tables / "extended_chembl_k_efficiency_failures.csv")
    ligand_rows = pd.DataFrame(rows)
    summary = aggregate_chembl_k_rows(ligand_rows)
    return summary, ligand_rows


def _generation_k_for_comparison(k_ligand_rows: pd.DataFrame) -> pd.DataFrame:
    if k_ligand_rows.empty:
        return pd.DataFrame()
    out = k_ligand_rows.copy()
    out["sampling_mode"] = "random_subsample"
    out["seed"] = np.nan
    out["n_available_pb"] = pd.to_numeric(out["conformers_available"], errors="coerce")
    out["n_used"] = out[["k", "n_available_pb"]].min(axis=1)
    out["casf_best_rmsd_at_k"] = pd.to_numeric(out["ligand_best_rmsd_at_k"], errors="coerce")
    for threshold in EXTENDED_HIT_THRESHOLDS:
        tag = threshold_tag(threshold)
        source_col = f"ligand_hit_{tag}_at_k"
        out[f"casf_hit_{tag}_at_k"] = pd.to_numeric(out[source_col], errors="coerce") if source_col in out.columns else np.nan
    return out


def _mean_random_chembl_ligand_rows(chembl_ligand_rows: pd.DataFrame) -> pd.DataFrame:
    random_rows = chembl_ligand_rows[chembl_ligand_rows["sampling_mode"].astype(str) == "random_k"].copy()
    if random_rows.empty:
        return random_rows
    group_cols = [column for column in [*IDENTITY_COLUMNS, "k", "mol_id", "rotatable_bonds", "heavy_atoms", "n_available_pb", "n_used"] if column in random_rows.columns]
    agg = {"casf_best_rmsd_at_k": "mean"}
    for threshold in EXTENDED_HIT_THRESHOLDS:
        tag = threshold_tag(threshold)
        agg[f"casf_hit_{tag}_at_k"] = "mean"
    out = random_rows.groupby(group_cols, dropna=False, sort=True).agg(agg).reset_index()
    out["sampling_mode"] = "random_k_mean"
    out["seed"] = np.nan
    return out


def _paired_k_delta_row(
    comparison: str,
    method: pd.DataFrame,
    baseline: pd.DataFrame,
    identity: dict[str, object],
    *,
    k: int,
    baseline_sampling_mode: str,
) -> dict[str, object]:
    cols = ["mol_id", "casf_best_rmsd_at_k", *[f"casf_hit_{threshold_tag(t)}_at_k" for t in EXTENDED_HIT_THRESHOLDS]]
    paired = method[cols].merge(baseline[cols], on="mol_id", suffixes=("_method", "_baseline"), how="inner")
    method_best = pd.to_numeric(paired["casf_best_rmsd_at_k_method"], errors="coerce")
    baseline_best = pd.to_numeric(paired["casf_best_rmsd_at_k_baseline"], errors="coerce")
    delta = method_best - baseline_best
    valid = delta.dropna()
    row = {
        **identity,
        "comparison": comparison,
        "k": int(k),
        "baseline_source": K_EFFICIENCY_CHEMBL_BASELINE,
        "baseline_sampling_mode": baseline_sampling_mode,
        "n_common_ligands": int(len(valid)),
        "mean_delta_best_rmsd_at_k": float(valid.mean()) if len(valid) else np.nan,
        "median_delta_best_rmsd_at_k": float(valid.median()) if len(valid) else np.nan,
    }
    for threshold in EXTENDED_HIT_THRESHOLDS:
        tag = threshold_tag(threshold)
        method_hit = pd.to_numeric(paired[f"casf_hit_{tag}_at_k_method"], errors="coerce")
        baseline_hit = pd.to_numeric(paired[f"casf_hit_{tag}_at_k_baseline"], errors="coerce")
        row[f"delta_hit_{tag}_at_k"] = float((method_hit - baseline_hit).dropna().mean()) if len(paired) else np.nan
    return row


def build_k_efficiency_comparisons(
    generation_ligand_rows: pd.DataFrame,
    chembl_ligand_rows: pd.DataFrame,
) -> pd.DataFrame:
    generation = _generation_k_for_comparison(generation_ligand_rows)
    if generation.empty or chembl_ligand_rows.empty:
        return pd.DataFrame()
    chembl_first = chembl_ligand_rows[chembl_ligand_rows["sampling_mode"].astype(str) == "first_k"].copy()
    chembl_random_mean = _mean_random_chembl_ligand_rows(chembl_ligand_rows)
    rows: list[dict[str, object]] = []

    qwen = generation[generation["source"].astype(str).str.contains("qwen", case=False, na=False)].copy()
    for _, method in qwen.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(method)
        ligand_set = str(identity["ligand_set"])
        for k, method_k in method.groupby("k", sort=True):
            for baseline_mode, baseline_all in [("first_k", chembl_first), ("random_k_mean", chembl_random_mean)]:
                baseline = baseline_all[
                    (baseline_all["ligand_set"].astype(str) == ligand_set)
                    & (pd.to_numeric(baseline_all["k"], errors="coerce") == int(k))
                ]
                if baseline.empty:
                    continue
                rows.append(
                    _paired_k_delta_row(
                        f"qwen_vs_chembl_{baseline_mode}",
                        method_k,
                        baseline,
                        identity,
                        k=int(k),
                        baseline_sampling_mode=baseline_mode,
                    )
                )

    for ligand_set, frame in chembl_first.groupby("ligand_set", dropna=False, sort=True):
        full = frame[pd.to_numeric(frame["k"], errors="coerce") == pd.to_numeric(frame["k"], errors="coerce").max()]
        if full.empty:
            continue
        identity = method_identity(frame)
        for k in (1, 5, 10, 25, 50, 100):
            current = frame[pd.to_numeric(frame["k"], errors="coerce") == k]
            if current.empty:
                continue
            rows.append(
                _paired_k_delta_row(
                    "chembl_full_available_vs_chembl_k",
                    current,
                    full,
                    identity,
                    k=int(k),
                    baseline_sampling_mode="first_k_full_available",
                )
            )
    return pd.DataFrame(rows)


def _energy_window_payloads(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    *,
    recompute: bool,
) -> list[dict[str, object]]:
    root_by_run = {
        str(row["run_id"]): str(row["root"])
        for _, row in analysis_sources.iterrows()
        if pd.notna(row.get("run_id")) and pd.notna(row.get("root"))
    }
    selected = pd.concat(
        [
            generation_methods(per_ligand),
            per_ligand[per_ligand["source"].astype(str) == K_EFFICIENCY_CHEMBL_BASELINE],
        ],
        ignore_index=True,
        sort=False,
    )
    selected = selected[pd.to_numeric(selected["conformer_count"], errors="coerce").fillna(0) >= 0].copy()
    chembl_maps = {
        ligand_set: _chembl_map_for_set(ligand_set).set_index("ligand_id")
        for ligand_set in sorted(selected["ligand_set"].dropna().astype(str).unique())
    }
    energy_lookup = _chembl_pb_energy_lookup(
        selected[selected["source"].astype(str) == K_EFFICIENCY_CHEMBL_BASELINE],
        analysis_sources,
    )
    all_energy_lookup = _chembl_all_energy_lookup(
        selected[selected["source"].astype(str) == K_EFFICIENCY_CHEMBL_BASELINE],
        analysis_sources,
    )
    payloads: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        ligand_set = str(row["ligand_set"])
        source = str(row["source"])
        run_id = str(row["run_id"])
        mol_id = str(row["mol_id"])
        root = root_by_run.get(run_id, "")
        if source != K_EFFICIENCY_CHEMBL_BASELINE and not root:
            continue
        payload: dict[str, object] = {
            "ligand_set": ligand_set,
            "run_id": run_id,
            "run_label": row.get("run_label", ""),
            "source": source,
            "family": row.get("family", source),
            "tier": row.get("tier", ""),
            "method": row.get("method", source),
            "mol_id": mol_id,
            "rotatable_bonds": row.get("rotatable_bonds", np.nan),
            "heavy_atoms": row.get("heavy_atoms", np.nan),
            "root": root,
            "ligand_dir": str(_ligand_dir_for_set(ligand_set)),
            "recompute": recompute,
            "part_path": str(paths.energy_window_parts / ligand_set / source / f"{mol_id}.csv"),
        }
        if source == K_EFFICIENCY_CHEMBL_BASELINE:
            chembl_map = chembl_maps.get(ligand_set)
            if chembl_map is None or mol_id not in chembl_map.index:
                continue
            payload["chembl_row"] = chembl_map.loc[mol_id].to_dict()
            payload["chembl_pb_energies"] = energy_lookup.get(ligand_set, {}).get(mol_id, [])
            payload["chembl_all_energies"] = all_energy_lookup.get(ligand_set, {}).get(mol_id, [])
        payloads.append(payload)
    return payloads


def _useful_cluster_count(mols: Sequence[object], rmsds: Sequence[object], threshold: float = 1.0, hit_threshold: float = 0.75) -> float:
    if not mols:
        return 0.0
    from casf_benchmark.analysis.metrics import greedy_cluster_assignments

    assignments = greedy_cluster_assignments(mols, threshold)
    values = pd.to_numeric(pd.Series(list(rmsds)), errors="coerce").to_numpy(dtype=float)
    useful: set[int] = set()
    for assignment, rmsd in zip(assignments, values):
        if math.isfinite(float(rmsd)) and float(rmsd) <= hit_threshold:
            useful.add(int(assignment))
    return float(len(useful))


def _energy_window_metrics(payload: dict[str, object], records: pd.DataFrame, mols: list[object]) -> list[dict[str, object]]:
    from casf_benchmark.analysis.metrics import greedy_cluster_metrics, pairwise_rmsd_stats

    rows: list[dict[str, object]] = []
    frame = records.copy().reset_index(drop=True)
    if "energy" not in frame.columns:
        frame["energy"] = np.nan
    frame["energy"] = pd.to_numeric(frame["energy"], errors="coerce")
    frame["casf_rmsd"] = pd.to_numeric(frame["casf_rmsd"], errors="coerce")
    n_pb_confs = int(len(frame))
    finite_energy = frame["energy"].dropna()
    min_energy = float(finite_energy.min()) if not finite_energy.empty else np.nan
    delta = frame["energy"] - min_energy if math.isfinite(min_energy) else pd.Series(np.nan, index=frame.index)

    for window_name, cutoff in ENERGY_WINDOWS:
        if window_name == "all":
            mask = pd.Series(True, index=frame.index)
        elif math.isfinite(min_energy):
            mask = delta <= cutoff
        else:
            mask = pd.Series(False, index=frame.index)
        selected = frame[mask].copy()
        selected_positions = selected.index.tolist()
        selected_mols = [mols[pos] for pos in selected_positions if pos < len(mols)]
        n_window_confs = int(len(selected))
        best = pd.to_numeric(selected["casf_rmsd"], errors="coerce").dropna()
        best_rmsd = float(best.min()) if not best.empty else np.nan
        clusters = greedy_cluster_metrics(selected_mols, thresholds=(1.0,))
        pairwise = pairwise_rmsd_stats(selected_mols)
        selected_energy = pd.to_numeric(selected["energy"], errors="coerce").dropna()
        max_delta = float((selected_energy - min_energy).max()) if math.isfinite(min_energy) and not selected_energy.empty else np.nan
        row = {
            **_identity_payload_columns(payload),
            "mol_id": payload["mol_id"],
            "rotatable_bonds": payload.get("rotatable_bonds", np.nan),
            "heavy_atoms": payload.get("heavy_atoms", np.nan),
            "energy_window": window_name,
            "n_pb_confs": n_pb_confs,
            "n_window_confs": n_window_confs,
            "fraction_window_confs": float(n_window_confs / n_pb_confs) if n_pb_confs else 0.0,
            "window_energy_min": float(selected_energy.min()) if not selected_energy.empty else np.nan,
            "window_energy_max_delta": max_delta,
            "casf_best_rmsd_window": best_rmsd,
            "greedy_clusters_1p0_window": clusters.get("greedy_clusters_1p0", np.nan),
            "clusters_per_100_1p0_window": clusters.get("clusters_per_100_1p0", np.nan),
            "cluster_entropy_1p0_window": clusters.get("cluster_entropy_1p0", np.nan),
            "largest_cluster_fraction_1p0_window": clusters.get("largest_cluster_fraction_1p0", np.nan),
            "pairwise_mean_window": pairwise.get("pairwise_mean", np.nan),
            "pairwise_p90_window": pairwise.get("pairwise_p90", np.nan),
            "useful_low_energy_clusters": _useful_cluster_count(
                selected_mols,
                selected["casf_rmsd"].tolist(),
            )
            if window_name == "deltaE_10"
            else np.nan,
        }
        for threshold in EXTENDED_HIT_THRESHOLDS:
            tag = threshold_tag(threshold)
            row[f"casf_hit_{tag}_window"] = float(best_rmsd <= threshold) if math.isfinite(best_rmsd) else 0.0
        rows.append(row)
    return rows


def _energy_window_worker(payload: dict[str, object]) -> list[dict[str, object]]:
    quiet_rdkit_logs()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("RDKIT_NUM_THREADS", "1")
    part_path = Path(str(payload["part_path"]))
    if part_path.exists() and not bool(payload.get("recompute", False)):
        cached = pd.read_csv(part_path)
        if "energy_window" in cached.columns:
            return cached.to_dict("records")
    if str(payload["source"]) == K_EFFICIENCY_CHEMBL_BASELINE:
        records, mols = _compute_chembl_pb_conformer_data(payload)
    else:
        records, mols = _compute_generation_conformer_data(payload)
    rows = _energy_window_metrics(payload, records, mols)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(part_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(part_path)
    return rows


def aggregate_energy_window_rows(per_ligand_rows: pd.DataFrame) -> pd.DataFrame:
    if per_ligand_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = [*IDENTITY_COLUMNS, "energy_window"]
    for key, group in per_ligand_rows.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        n_ligands = int(group["mol_id"].nunique())
        row.update(
            {
                "n_ligands": n_ligands,
                "n_ligands_with_window_confs": int(pd.to_numeric(group["n_window_confs"], errors="coerce").fillna(0).gt(0).sum()),
                "mean_fraction_window_confs": pd.to_numeric(group["fraction_window_confs"], errors="coerce").mean(),
                "mean_n_window_confs": pd.to_numeric(group["n_window_confs"], errors="coerce").mean(),
                "mean_best_rmsd_window": pd.to_numeric(group["casf_best_rmsd_window"], errors="coerce").mean(),
                "mean_clusters_1p0_window": pd.to_numeric(group["greedy_clusters_1p0_window"], errors="coerce").mean(),
                "mean_clusters_per_100_1p0_window": pd.to_numeric(group["clusters_per_100_1p0_window"], errors="coerce").mean(),
                "mean_entropy_1p0_window": pd.to_numeric(group["cluster_entropy_1p0_window"], errors="coerce").mean(),
                "mean_largest_cluster_fraction_1p0_window": pd.to_numeric(group["largest_cluster_fraction_1p0_window"], errors="coerce").mean(),
                "mean_useful_low_energy_clusters": pd.to_numeric(group["useful_low_energy_clusters"], errors="coerce").mean(),
            }
        )
        for threshold in EXTENDED_HIT_THRESHOLDS:
            tag = threshold_tag(threshold)
            hits = pd.to_numeric(group[f"casf_hit_{tag}_window"], errors="coerce").fillna(0)
            row[f"hit_{tag}_window"] = float(hits.sum() / n_ligands) if n_ligands else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    pieces = []
    for _, method in summary.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        method = method.copy()
        all_row = method[method["energy_window"].astype(str) == "all"]
        de10_row = method[method["energy_window"].astype(str) == "deltaE_10"]
        if not all_row.empty and not de10_row.empty:
            all_hit = float(all_row.iloc[0].get("hit_0p75_window", np.nan))
            de10_hit = float(de10_row.iloc[0].get("hit_0p75_window", np.nan))
            all_clusters = float(all_row.iloc[0].get("mean_clusters_1p0_window", np.nan))
            de10_clusters = float(de10_row.iloc[0].get("mean_clusters_1p0_window", np.nan))
            method["hit_retention_deltaE_10"] = de10_hit / all_hit if all_hit else np.nan
            method["cluster_retention_deltaE_10"] = de10_clusters / all_clusters if all_clusters else np.nan
        pieces.append(method)
    return pd.concat(pieces, ignore_index=True, sort=False)


def build_energy_window_comparisons(per_ligand_rows: pd.DataFrame) -> pd.DataFrame:
    if per_ligand_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, method in per_ligand_rows.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(method)
        pivot = method.pivot_table(
            index="mol_id",
            columns="energy_window",
            values=["casf_best_rmsd_window", "casf_hit_0p75_window", "greedy_clusters_1p0_window"],
            aggfunc="first",
        )
        if ("casf_hit_0p75_window", "all") in pivot.columns and ("casf_hit_0p75_window", "deltaE_20") in pivot.columns:
            all_hit = pd.to_numeric(pivot[("casf_hit_0p75_window", "all")], errors="coerce").fillna(0)
            de20_hit = pd.to_numeric(pivot[("casf_hit_0p75_window", "deltaE_20")], errors="coerce").fillna(0)
            hit_ligands = all_hit > 0
            high_energy_fraction = float(((hit_ligands) & (de20_hit <= 0)).sum() / hit_ligands.sum()) if hit_ligands.sum() else np.nan
        else:
            high_energy_fraction = np.nan
        for left, right in [("all", "deltaE_20"), ("deltaE_20", "deltaE_10")]:
            left_col = ("casf_hit_0p75_window", left)
            right_col = ("casf_hit_0p75_window", right)
            if left_col not in pivot.columns or right_col not in pivot.columns:
                continue
            left_hit = pd.to_numeric(pivot[left_col], errors="coerce").fillna(0)
            right_hit = pd.to_numeric(pivot[right_col], errors="coerce").fillna(0)
            rows.append(
                {
                    **identity,
                    "comparison": f"{left}_vs_{right}",
                    "n_ligands": int(len(pivot)),
                    "delta_hit_0p75_window": float(right_hit.mean() - left_hit.mean()),
                    "high_energy_hit_fraction": high_energy_fraction,
                }
            )
    return pd.DataFrame(rows)


def build_energy_window_analysis(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    *,
    workers: int,
    recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = _energy_window_payloads(per_ligand, analysis_sources, paths, recompute=recompute)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_energy_window_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - keep other ligand/method tasks.
                failures.append(f"{type(exc).__name__}: {exc}")
            if index % 250 == 0:
                print(f"Energy-window tasks complete: {index}/{len(futures)}", flush=True)
    failure_path = paths.tables / "extended_energy_window_failures.csv"
    if failures:
        write_csv(pd.DataFrame({"failure": failures}), failure_path)
    elif failure_path.exists():
        failure_path.unlink()
    per_ligand_rows = pd.DataFrame(rows)
    summary = aggregate_energy_window_rows(per_ligand_rows)
    comparisons = build_energy_window_comparisons(per_ligand_rows)
    return per_ligand_rows, summary, comparisons


def _aggregate_k_ligand_rows(ligand_rows: pd.DataFrame, stratum_cols: list[str] | None = None) -> pd.DataFrame:
    if ligand_rows.empty:
        return pd.DataFrame()
    frame = ligand_rows.copy()
    group_cols = [*IDENTITY_COLUMNS, "k"]
    if stratum_cols:
        group_cols.extend(stratum_cols)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        best = pd.to_numeric(group["ligand_best_rmsd_at_k"], errors="coerce")
        row.update(
            {
                "n_ligands": int(group["mol_id"].nunique()),
                "sampling_mode": "random_subsample",
                "summary_mode": "capped_at_available",
                "n_ligands_with_pb": int(pd.to_numeric(group["conformers_available"], errors="coerce").fillna(0).gt(0).sum()),
                "mean_n_used": pd.to_numeric(group[["k", "conformers_available"]].min(axis=1), errors="coerce").mean(),
                "mean_best_rmsd_at_k": best.mean(),
                "median_best_rmsd_at_k": best.median(),
                "hit_0p25_at_k": pd.to_numeric(group["ligand_hit_0p25_at_k"], errors="coerce").mean(),
                "hit_0p5_at_k": pd.to_numeric(group["ligand_hit_0p5_at_k"], errors="coerce").mean(),
                "hit_0p75_at_k": pd.to_numeric(group["ligand_hit_0p75_at_k"], errors="coerce").mean(),
                "hit_2p0_at_k": pd.to_numeric(group["ligand_hit_2p0_at_k"], errors="coerce").mean(),
                "mean_pb_pass_rate_at_k": pd.to_numeric(group["pb_pass_rate"], errors="coerce").mean(),
                "mean_conformers_available": pd.to_numeric(group["conformers_available"], errors="coerce").mean(),
                "n_ligands_with_at_least_k": int(pd.to_numeric(group["ligand_has_at_least_k"], errors="coerce").fillna(0).sum()),
                "ci_metric": "",
                "ci_low": np.nan,
                "ci_high": np.nan,
                "mean_best_rmsd_ci_low": np.nan,
                "mean_best_rmsd_ci_high": np.nan,
                "status": "computed",
                "skip_reason": "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def chembl_pb_k_unavailable_rows(per_ligand: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    reason = (
        "ChEMBL3D-PB K-efficiency requires per-conformer PB-pass indices or per-conformer "
        "CASF RMSDs; the current dashboard artifacts contain only per-ligand PB-filtered best RMSD"
    )
    baseline = per_ligand[per_ligand["source"].astype(str) == K_EFFICIENCY_CHEMBL_BASELINE].copy()
    rows: list[dict[str, object]] = []
    for _, frame in baseline.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(frame)
        n_ligands = int(pd.to_numeric(frame["conformer_count"], errors="coerce").fillna(0).gt(0).sum())
        mean_confs = pd.to_numeric(frame["conformer_count"], errors="coerce").mean()
        for k in k_values:
            rows.append(
                {
                    **identity,
                    "k": int(k),
                    "n_ligands": n_ligands,
                    "mean_best_rmsd_at_k": np.nan,
                    "median_best_rmsd_at_k": np.nan,
                    "hit_0p5_at_k": np.nan,
                    "hit_0p75_at_k": np.nan,
                    "hit_2p0_at_k": np.nan,
                    "mean_pb_pass_rate_at_k": pd.to_numeric(frame["pb_pass_rate"], errors="coerce").mean(),
                    "mean_conformers_available": mean_confs,
                    "n_ligands_with_at_least_k": int(pd.to_numeric(frame["conformer_count"], errors="coerce").fillna(0).ge(k).sum()),
                    "status": "skipped",
                    "skip_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def build_k_efficiency(
    per_ligand: pd.DataFrame,
    analysis_sources: pd.DataFrame,
    paths: OutputPaths,
    k_values: list[int],
    *,
    subsamples: int,
    seed: int,
    workers: int,
    recompute_rmsd: bool,
    allow_failures: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = _k_task_payloads(
        per_ligand,
        analysis_sources,
        paths,
        k_values,
        subsamples,
        seed,
        recompute_rmsd,
    )
    all_rows: list[dict[str, object]] = []
    failures: list[str] = []
    max_workers = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_k_efficiency_worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - keep other ligand/method tasks.
                failures.append(f"{type(exc).__name__}: {exc}")
            if index % 250 == 0:
                print(f"K-efficiency RMSD tasks complete: {index}/{len(futures)}", flush=True)

    failure_path = paths.tables / "extended_k_efficiency_failures.csv"
    if failures:
        failure_report = pd.DataFrame({"failure": failures})
        write_csv(failure_report, failure_path)
        preview = "; ".join(failures[:5])
        suffix = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        message = f"K-efficiency failed for {len(failures)} task(s): {preview}{suffix}"
        if not allow_failures:
            raise RuntimeError(message)
        print(
            f"WARNING: {message}\n"
            f"WARNING: keeping the {len(all_rows)} task(s) that succeeded; the sources in "
            f"{failure_path} are absent from every extended_k_efficiency* table.",
            flush=True,
        )
    if failure_path.exists():
        failure_path.unlink()

    ligand_rows = pd.DataFrame(all_rows)
    if ligand_rows.empty:
        reason = "; ".join(failures[:5]) if failures else "no K-efficiency payloads could be computed"
        skipped, skipped_strata = skipped_k_efficiency_frames(k_values, reason)
        return skipped, skipped_strata, ligand_rows

    ligand_rows["rotatable_bond_bin"] = assign_rotatable_bin(ligand_rows["rotatable_bonds"])
    ligand_rows["heavy_atom_bin"] = assign_heavy_atom_bin(ligand_rows["heavy_atoms"])
    total = _aggregate_k_ligand_rows(ligand_rows)
    strata_parts = []
    for col in ["rotatable_bond_bin", "heavy_atom_bin"]:
        strata = _aggregate_k_ligand_rows(ligand_rows, [col])
        if strata.empty:
            continue
        strata = strata.rename(columns={col: "stratum"})
        strata.insert(0, "stratum_type", col)
        strata_parts.append(strata)
    strata_frame = pd.concat(strata_parts, ignore_index=True, sort=False) if strata_parts else pd.DataFrame()
    manifest = (
        ligand_rows.groupby([*IDENTITY_COLUMNS, "mol_id"], dropna=False, sort=True)
        .agg(
            conformers_available=("conformers_available", "max"),
            rotatable_bonds=("rotatable_bonds", "first"),
            heavy_atoms=("heavy_atoms", "first"),
        )
        .reset_index()
    )
    write_csv(manifest, paths.tables / "extended_per_conformer_casf_rmsd_manifest.csv")
    return total, strata_frame, ligand_rows


def generation_methods(per_ligand: pd.DataFrame) -> pd.DataFrame:
    return per_ligand[per_ligand["row_type"].astype(str) == "generation"].copy()


def paired_metric_summary(paired: pd.DataFrame, threshold: float = 0.10) -> dict[str, float]:
    delta = pd.to_numeric(paired["method_casf_best_rmsd"], errors="coerce") - pd.to_numeric(
        paired["baseline_casf_best_rmsd"], errors="coerce"
    )
    valid = delta.dropna()
    summary = {
        "n_common_ligands": int(len(valid)),
        "mean_delta_casf_best_rmsd": float(valid.mean()) if len(valid) else np.nan,
        "median_delta_casf_best_rmsd": float(valid.median()) if len(valid) else np.nan,
        "win_rate_0p1A": float((valid < -threshold).mean()) if len(valid) else np.nan,
        "loss_rate_0p1A": float((valid > threshold).mean()) if len(valid) else np.nan,
    }
    for hit_col, out_col in [
        ("casf_hit_0p5", "delta_hit_0p5"),
        ("casf_hit_0p75", "delta_hit_0p75"),
        ("casf_hit_2p0", "delta_hit_2p0"),
    ]:
        method_hit = pd.to_numeric(paired[f"method_{hit_col}"], errors="coerce")
        baseline_hit = pd.to_numeric(paired[f"baseline_{hit_col}"], errors="coerce")
        hit_delta = (method_hit - baseline_hit).dropna()
        summary[out_col] = float(hit_delta.mean()) if len(hit_delta) else np.nan
    return summary


def add_common_pair_columns(method: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = [
        "mol_id",
        "casf_best_rmsd",
        "casf_hit_0p5",
        "casf_hit_0p75",
        "casf_hit_2p0",
        "conformer_count",
        "rotatable_bonds",
        "heavy_atoms",
    ]
    method_cols = [
        "mol_id",
        "casf_best_rmsd",
        "casf_hit_0p5",
        "casf_hit_0p75",
        "casf_hit_2p0",
        "pb_pass_rate",
        "greedy_clusters_1p0",
        "rotatable_bonds",
        "heavy_atoms",
    ]
    paired = baseline[baseline_cols].merge(
        method[method_cols],
        on="mol_id",
        how="inner",
        suffixes=("_baseline", "_method"),
    )
    return paired.rename(
        columns={
            "casf_best_rmsd_baseline": "baseline_casf_best_rmsd",
            "casf_hit_0p5_baseline": "baseline_casf_hit_0p5",
            "casf_hit_0p75_baseline": "baseline_casf_hit_0p75",
            "casf_hit_2p0_baseline": "baseline_casf_hit_2p0",
            "conformer_count": "conformer_count_baseline",
            "casf_best_rmsd_method": "method_casf_best_rmsd",
            "casf_hit_0p5_method": "method_casf_hit_0p5",
            "casf_hit_0p75_method": "method_casf_hit_0p75",
            "casf_hit_2p0_method": "method_casf_hit_2p0",
            "rotatable_bonds_method": "rotatable_bonds",
            "heavy_atoms_method": "heavy_atoms",
        }
    )


def method_identity(method_frame: pd.DataFrame) -> dict[str, object]:
    first = method_frame.iloc[0]
    return {column: first.get(column) for column in IDENTITY_COLUMNS}


def comparison_row(
    identity: dict[str, object],
    method: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_source: str,
    paired: pd.DataFrame,
    stratum_type: str,
    stratum: str,
) -> dict[str, object]:
    row = {
        **identity,
        "n_ligands": int(paired["method_casf_best_rmsd"].notna().sum()),
        "baseline_source": baseline_source,
        "n_reference_ligands": int(baseline["casf_best_rmsd"].notna().sum()),
        "n_method_ligands": int(method["casf_best_rmsd"].notna().sum()),
        "n_reference_only": int(len(set(baseline["mol_id"]) - set(method["mol_id"]))),
        "n_method_only": int(len(set(method["mol_id"]) - set(baseline["mol_id"]))),
        "excluded_reason_summary": "",
        "stratum_type": stratum_type,
        "stratum": stratum,
    }
    row.update(paired_metric_summary(paired))
    delta = pd.to_numeric(paired["method_casf_best_rmsd"], errors="coerce") - pd.to_numeric(
        paired["baseline_casf_best_rmsd"], errors="coerce"
    )
    valid = delta.dropna()
    for threshold in (0.05, 0.10, 0.25):
        label = str(threshold).replace(".", "p")
        row[f"win_rate_{label}A"] = float((valid < -threshold).mean()) if len(valid) else np.nan
        row[f"loss_rate_{label}A"] = float((valid > threshold).mean()) if len(valid) else np.nan
        row[f"tie_rate_{label}A"] = float((valid.abs() <= threshold).mean()) if len(valid) else np.nan
    row["catastrophic_loss_count_gt1A"] = int((valid > 1.0).sum())
    row["large_rescue_count_gt1A"] = int((valid < -1.0).sum())
    return row


def build_paired_delta_table(
    per_ligand: pd.DataFrame, baseline_source: str, include_strata: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_all = per_ligand[per_ligand["source"].astype(str) == baseline_source].copy()
    methods = generation_methods(per_ligand)
    chembl_all = per_ligand[per_ligand["source"].astype(str) == "chembl3d_gt"].copy()
    methods = pd.concat([methods, chembl_all], ignore_index=True, sort=False)

    rows: list[dict[str, object]] = []
    rescue_rows: list[pd.DataFrame] = []
    group_cols = IDENTITY_COLUMNS
    for _, method in methods.groupby(group_cols, dropna=False, sort=True):
        identity = method_identity(method)
        baseline = baseline_all[baseline_all["ligand_set"].astype(str) == str(identity["ligand_set"])]
        if baseline.empty:
            continue
        paired = add_common_pair_columns(method, baseline)
        rows.append(comparison_row(identity, method, baseline, baseline_source, paired, "total", "all"))

        delta = pd.to_numeric(paired["method_casf_best_rmsd"], errors="coerce") - pd.to_numeric(
            paired["baseline_casf_best_rmsd"], errors="coerce"
        )
        detail = paired.copy()
        detail["delta_casf_best_rmsd"] = delta
        detail = detail.sort_values("delta_casf_best_rmsd", na_position="last")
        for direction, selected in [
            ("rescued", detail.head(20)),
            ("worsened", detail.sort_values("delta_casf_best_rmsd", ascending=False, na_position="last").head(20)),
        ]:
            selected = selected.copy()
            for key, value in identity.items():
                selected[key] = value
            selected["baseline_source"] = baseline_source
            selected["direction"] = direction
            rescue_rows.append(
                selected[
                    [
                        *IDENTITY_COLUMNS,
                        "baseline_source",
                        "direction",
                        "mol_id",
                        "baseline_casf_best_rmsd",
                        "method_casf_best_rmsd",
                        "delta_casf_best_rmsd",
                        "baseline_casf_hit_0p75",
                        "method_casf_hit_0p75",
                    ]
                ]
            )

        if not include_strata:
            continue
        paired = paired.copy()
        paired["rotatable_bond_bin"] = assign_rotatable_bin(paired["rotatable_bonds"])
        paired["heavy_atom_bin"] = assign_heavy_atom_bin(paired["heavy_atoms"])
        paired["chembl_conformer_count_bin"] = assign_chembl_count_bin(paired["conformer_count_baseline"])
        paired["baseline_casf_difficulty_bin"] = assign_difficulty_bin(paired["baseline_casf_best_rmsd"])
        for stratum_type in [
            "rotatable_bond_bin",
            "heavy_atom_bin",
            "chembl_conformer_count_bin",
            "baseline_casf_difficulty_bin",
        ]:
            for stratum, stratum_frame in paired.groupby(stratum_type, dropna=True, sort=True):
                rows.append(
                    comparison_row(
                        identity,
                        method,
                        baseline,
                        baseline_source,
                        stratum_frame,
                        stratum_type,
                        str(stratum),
                    )
                )
    detail_frame = pd.concat(rescue_rows, ignore_index=True, sort=False) if rescue_rows else pd.DataFrame()
    return pd.DataFrame(rows), detail_frame


def aggregate_generation_rows(per_ligand: pd.DataFrame) -> pd.DataFrame:
    methods = generation_methods(per_ligand)
    rows: list[dict[str, object]] = []
    numeric_means = [
        "conformer_count",
        "pb_pass_rate",
        "pb_fail_confs",
        "pb_input_confs",
        "greedy_clusters_1p0",
        "casf_best_rmsd",
        "casf_hit_0p75",
    ]
    for _, method in methods.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(method)
        row = {**identity, "n_ligands": int(method["mol_id"].nunique())}
        for column in numeric_means:
            row[f"mean_{column}"] = pd.to_numeric(method[column], errors="coerce").mean()
            row[f"median_{column}"] = pd.to_numeric(method[column], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows)


def build_frontier_summary(per_ligand: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_generation_rows(per_ligand)
    if summary.empty:
        return summary
    summary["useful_clusters_per_100_confs"] = (
        summary["mean_greedy_clusters_1p0"] / summary["mean_conformer_count"] * 100.0
    )
    summary["hit_0p75_per_100_confs"] = (
        summary["mean_casf_hit_0p75"] / summary["mean_conformer_count"] * 100.0
    )
    summary["valid_hit_score"] = summary["mean_casf_hit_0p75"] * summary["mean_pb_pass_rate"]
    pieces = []
    for _, ligand_set_frame in summary.groupby("ligand_set", dropna=False, sort=True):
        ligand_set_frame = ligand_set_frame.copy()
        ligand_set_frame["frontier_rank"] = non_dominated_frontier_ranks(
            ligand_set_frame,
            ["mean_casf_hit_0p75", "mean_pb_pass_rate", "mean_conformer_count"],
            [True, True, False],
        )
        pieces.append(ligand_set_frame)
    return pd.concat(pieces, ignore_index=True, sort=False)


def build_pb_failure_tables(per_ligand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = per_ligand[per_ligand["row_type"].astype(str) == "generation"].copy()
    rows: list[dict[str, object]] = []
    tails: list[pd.DataFrame] = []
    for _, method in methods.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(method)
        counts: dict[str, float] = {}
        parsed = method["pb_check_fail_counts_json"].map(parse_fail_counts_json)
        for item in parsed:
            for key, count in item.items():
                counts[key] = counts.get(key, 0.0) + count
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        total_check_failures = sum(counts.values())
        pb_fail_total = pd.to_numeric(method["pb_fail_confs"], errors="coerce").sum()
        pb_input_total = pd.to_numeric(method["pb_input_confs"], errors="coerce").sum()
        row = {
            **identity,
            "n_ligands": int(method["mol_id"].nunique()),
            "total_pb_fail_rate": float(pb_fail_total / pb_input_total) if pb_input_total else np.nan,
            "median_per_ligand_pb_pass_rate": pd.to_numeric(method["pb_pass_rate"], errors="coerce").median(),
            "ligands_pb_pass_rate_lt_0p9": int((pd.to_numeric(method["pb_pass_rate"], errors="coerce") < 0.9).sum()),
            "ligands_pb_pass_rate_lt_0p5": int((pd.to_numeric(method["pb_pass_rate"], errors="coerce") < 0.5).sum()),
            "dominant_pb_failure_test": sorted_counts[0][0] if sorted_counts else "",
            "second_dominant_pb_failure_test": sorted_counts[1][0] if len(sorted_counts) > 1 else "",
            "total_check_failure_counts": total_check_failures,
        }
        for key in ["tetrahedral_chirality", "internal_steric_clash", "energy_ratio"]:
            row[f"fraction_failures_{key}"] = counts.get(key, 0.0) / total_check_failures if total_check_failures else 0.0
        bond_total = sum(counts.get(key, 0.0) for key in BOND_GEOMETRY_TESTS)
        row["fraction_failures_bond_geometry_tests"] = bond_total / total_check_failures if total_check_failures else 0.0
        rows.append(row)

        tail = method.copy()
        tail["top_three_failed_pb_tests"] = parsed.map(
            lambda value: ", ".join(
                key for key, count in sorted(value.items(), key=lambda item: item[1], reverse=True)[:3] if count > 0
            )
        )
        tail = tail.sort_values(["pb_pass_rate", "pb_input_confs"], ascending=[True, False], na_position="last").head(25)
        tails.append(
            tail[
                [
                    *IDENTITY_COLUMNS,
                    "mol_id",
                    "rotatable_bonds",
                    "heavy_atoms",
                    "pb_pass_rate",
                    "pb_input_confs",
                    "top_three_failed_pb_tests",
                    "casf_best_rmsd",
                    "casf_hit_0p75",
                ]
            ].assign(n_ligands=len(tail))
        )
    tail_frame = pd.concat(tails, ignore_index=True, sort=False) if tails else pd.DataFrame()
    return pd.DataFrame(rows), tail_frame


def build_bound_pose_difficulty(per_ligand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_all = per_ligand[per_ligand["source"].astype(str) == "chembl3d_gt_pb"].copy()
    methods = generation_methods(per_ligand)
    rows: list[dict[str, object]] = []
    rescue_cases: list[pd.DataFrame] = []
    rescue_sources = {"loqi_raw_fixed", "torsional_diffusion_raw_fixed", "rdkit_random_raw_fixed"}
    for _, method in methods.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        identity = method_identity(method)
        baseline = baseline_all[baseline_all["ligand_set"].astype(str) == str(identity["ligand_set"])]
        if baseline.empty:
            continue
        paired = add_common_pair_columns(method, baseline)
        paired["baseline_difficulty_bin"] = assign_difficulty_bin(paired["baseline_casf_best_rmsd"])
        paired["delta_casf_best_rmsd"] = pd.to_numeric(paired["method_casf_best_rmsd"], errors="coerce") - pd.to_numeric(
            paired["baseline_casf_best_rmsd"], errors="coerce"
        )
        paired["rescue"] = (pd.to_numeric(paired["baseline_casf_hit_0p75"], errors="coerce") < 1.0) & (
            pd.to_numeric(paired["method_casf_hit_0p75"], errors="coerce") >= 1.0
        )
        paired["catastrophic_loss"] = (pd.to_numeric(paired["baseline_casf_hit_0p75"], errors="coerce") >= 1.0) & (
            pd.to_numeric(paired["method_casf_hit_2p0"], errors="coerce") < 1.0
        )
        for difficulty, group in paired.groupby("baseline_difficulty_bin", dropna=True, sort=True):
            rows.append(
                {
                    **identity,
                    "n_ligands": int(group["method_casf_best_rmsd"].notna().sum()),
                    "baseline_source": "chembl3d_gt_pb",
                    "difficulty_bin": str(difficulty),
                    "n_common_ligands": int(group["method_casf_best_rmsd"].notna().sum()),
                    "mean_delta_casf_best_rmsd": group["delta_casf_best_rmsd"].mean(),
                    "hit_0p75": pd.to_numeric(group["method_casf_hit_0p75"], errors="coerce").mean(),
                    "rescue_rate": group["rescue"].mean(),
                    "catastrophic_loss_rate": group["catastrophic_loss"].mean(),
                }
            )
        if identity["source"] in rescue_sources:
            candidates = paired[paired["rescue"]].copy()
            candidates = candidates.sort_values("delta_casf_best_rmsd").head(50)
            for key, value in identity.items():
                candidates[key] = value
            rescue_cases.append(
                candidates[
                    [
                        *IDENTITY_COLUMNS,
                        "mol_id",
                        "rotatable_bonds",
                        "heavy_atoms",
                        "baseline_casf_best_rmsd",
                        "method_casf_best_rmsd",
                        "delta_casf_best_rmsd",
                        "pb_pass_rate",
                        "greedy_clusters_1p0",
                    ]
                ].assign(n_ligands=len(candidates), baseline_source="chembl3d_gt_pb")
            )
    rescue_frame = pd.concat(rescue_cases, ignore_index=True, sort=False) if rescue_cases else pd.DataFrame()
    return pd.DataFrame(rows), rescue_frame


def build_sanity_checks(per_ligand: pd.DataFrame, comparison_rows: pd.DataFrame) -> pd.DataFrame:
    required = [
        "ligand_set",
        "run_id",
        "run_label",
        "source",
        "family",
        "tier",
        "method",
        "mol_id",
        "row_type",
        "casf_best_rmsd",
        "casf_hit_0p5",
        "casf_hit_0p75",
        "casf_hit_2p0",
        "pb_input_confs",
        "pb_pass_confs",
        "pb_fail_confs",
        "conformer_count",
    ]
    rows: list[dict[str, object]] = []

    def add(check: str, severity: str, affected: int, details: str) -> None:
        rows.append(
            {
                "check": check,
                "severity": severity,
                "status": "pass" if affected == 0 else ("fail" if severity == "error" else "warn"),
                "affected_rows": int(affected),
                "details": details,
            }
        )

    missing = [column for column in required if column not in per_ligand.columns]
    add("missing_required_columns", "error", len(missing), ", ".join(missing))
    if missing:
        return pd.DataFrame(rows)

    duplicate_cols = ["ligand_set", "mol_id", "source", "tier"]
    duplicates = per_ligand.duplicated(duplicate_cols, keep=False)
    add("duplicate_ligand_source_tier_rows", "error", int(duplicates.sum()), "key=" + "/".join(duplicate_cols))
    add(
        "negative_casf_best_rmsd",
        "error",
        int((pd.to_numeric(per_ligand["casf_best_rmsd"], errors="coerce") < 0).sum()),
        "CASF best RMSD must be non-negative",
    )
    for column in ["casf_hit_0p5", "casf_hit_0p75", "casf_hit_2p0"]:
        values = pd.to_numeric(per_ligand[column], errors="coerce")
        add(f"{column}_outside_0_1", "error", int(((values < 0) | (values > 1)).sum()), "hit values must be in [0, 1]")
    pb_input = pd.to_numeric(per_ligand["pb_input_confs"], errors="coerce")
    pb_pass = pd.to_numeric(per_ligand["pb_pass_confs"], errors="coerce")
    pb_fail = pd.to_numeric(per_ligand["pb_fail_confs"], errors="coerce")
    pb_present = pb_input.notna() & pb_pass.notna() & pb_fail.notna()
    add(
        "pb_pass_fail_count_mismatch",
        "error",
        int(((pb_pass + pb_fail) != pb_input)[pb_present].sum()),
        "pb_pass_confs + pb_fail_confs should equal pb_input_confs where all are present",
    )
    conformer_count = pd.to_numeric(per_ligand["conformer_count"], errors="coerce")
    add(
        "conformer_count_greater_than_pb_input",
        "warn",
        int(((conformer_count > pb_input) & pb_input.notna()).sum()),
        "warning only because pre-PB versus post-PB semantics vary by source",
    )
    qwen = per_ligand[per_ligand["source"].astype(str).str.contains("qwen", case=False, na=False)]
    add(
        "qwen_rotatable_bond_metadata_warning",
        "warn",
        int(qwen["rotatable_bonds"].isna().sum()) if not qwen.empty else 0,
        "flagged only; this task does not repair Qwen metadata",
    )
    add(
        "stale_analysis_manifest_not_checked",
        "warn",
        1,
        "Analysis 1 was explicitly skipped, so manifest/SDF stale-output checks were not run",
    )
    if not comparison_rows.empty:
        count_rows = comparison_rows.groupby(["ligand_set", "source", "tier"], dropna=False)["ligands"].first()
        add("ligand_counts_by_run_source_tier", "info", 0, f"{len(count_rows)} run/source/tier count rows available")
    return pd.DataFrame(rows)


def skipped_k_efficiency_frames(k_values: list[int], reason: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "ligand_set",
        "run_id",
        "run_label",
        "source",
        "family",
        "tier",
        "method",
        "n_ligands",
        "k",
        "status",
        "skip_reason",
        "mean_best_rmsd_at_k",
        "median_best_rmsd_at_k",
        "hit_0p5_at_k",
        "hit_0p75_at_k",
        "hit_2p0_at_k",
        "mean_pb_pass_rate_at_k",
        "mean_conformers_available",
        "n_ligands_with_at_least_k",
    ]
    frame = pd.DataFrame([{"k": k, "status": "skipped", "skip_reason": reason} for k in k_values], columns=columns)
    strata = frame.assign(stratum_type="", stratum="")
    return frame, strata


def write_k_efficiency_outputs(
    paths: OutputPaths,
    k_eff: pd.DataFrame,
    k_eff_strata: pd.DataFrame,
) -> None:
    write_csv(k_eff, paths.tables / "extended_k_efficiency.csv")
    write_csv(k_eff_strata, paths.tables / "extended_k_efficiency_strata.csv")
    if not k_eff.empty and set(k_eff["status"].dropna().astype(str)) == {"skipped"}:
        reason = str(k_eff["skip_reason"].dropna().iloc[0]) if k_eff["skip_reason"].notna().any() else "unknown"
        report = (
            "# K-Efficiency Curves\n\n"
            "Status: skipped.\n\n"
            f"Reason: {reason}.\n\n"
            "The script intentionally does not approximate K curves from best-per-ligand rows."
        )
    else:
        report = k_efficiency_report(k_eff, k_eff_strata)
    write_report(report, paths.reports / "extended_k_efficiency.md")
    plot_k_efficiency(k_eff, paths)


def k_efficiency_report(k_eff: pd.DataFrame, k_eff_strata: pd.DataFrame) -> str:
    if k_eff.empty:
        return "# K-Efficiency Curves\n\nNo rows were computed."

    ref = k_eff[k_eff["ligand_set"].astype(str) == "ref"].copy()
    baseline_mask = ref["source"].astype(str) == "chembl3d_gt_pb"
    if "sampling_mode" in ref.columns:
        baseline_mask &= ref["sampling_mode"].fillna("").astype(str).isin(["", "first_k"])
    if "summary_mode" in ref.columns:
        baseline_mask &= ref["summary_mode"].fillna("").astype(str).isin(["", "capped_at_available"])
    baseline = ref[baseline_mask][["k", "hit_0p75_at_k"]].drop_duplicates("k").rename(
        columns={"hit_0p75_at_k": "baseline_hit_0p75_at_k"}
    )
    crossover_rows = []
    for source, frame in ref[ref["source"].astype(str) != "chembl3d_gt_pb"].groupby("source", sort=True):
        merged = frame.merge(baseline, on="k", how="inner")
        wins = merged[pd.to_numeric(merged["hit_0p75_at_k"], errors="coerce") > pd.to_numeric(merged["baseline_hit_0p75_at_k"], errors="coerce")]
        crossover_rows.append(
            {
                "source": source,
                "crossover_k_hit_0p75": int(wins["k"].min()) if not wins.empty else np.nan,
            }
        )
    crossover = pd.DataFrame(crossover_rows)
    preview_cols = [
        "ligand_set",
        "source",
        "sampling_mode",
        "summary_mode",
        "k",
        "n_ligands",
        "hit_0p75_at_k",
        "mean_best_rmsd_at_k",
        "mean_conformers_available",
        "n_ligands_with_at_least_k",
    ]
    report = "# K-Efficiency Curves\n\n"
    report += "Computed from per-conformer CASF RMSDs with deterministic subsampling from PB-passing conformer sets.\n\n"
    report += "## Ref Hit@0.75 Crossover vs ChEMBL3D-PB\n\n"
    report += table_preview(crossover) + "\n\n"
    report += "## Ref Summary Preview\n\n"
    report += table_preview(ref[preview_cols].sort_values(["source", "k"]).head(40)) + "\n\n"
    if not k_eff_strata.empty:
        strata_preview = k_eff_strata[
            (k_eff_strata["ligand_set"].astype(str) == "ref")
            & (k_eff_strata["stratum_type"].astype(str) == "rotatable_bond_bin")
        ]
        report += "## Rotatable-Bond Strata Preview\n\n"
        report += table_preview(strata_preview.sort_values(["source", "stratum", "k"]).head(40))
    return report


def plot_k_efficiency(k_eff: pd.DataFrame, paths: OutputPaths) -> None:
    ref = k_eff[k_eff["ligand_set"].astype(str) == "ref"].copy() if not k_eff.empty else pd.DataFrame()
    if ref.empty or set(ref.get("status", pd.Series(dtype=str)).dropna().astype(str)) == {"skipped"}:
        for filename in ["k_efficiency_hit_0p75_ref.png", "k_efficiency_best_rmsd_ref.png"]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.axis("off")
            ax.text(0.5, 0.55, "K-efficiency not computed", ha="center", va="center", fontsize=16)
            ax.text(0.5, 0.42, "No computed K-efficiency rows are available", ha="center", va="center", fontsize=10)
            fig.tight_layout()
            fig.savefig(paths.figures / filename, dpi=180)
            plt.close(fig)
        return
    for y_col, filename, ylabel in [
        ("hit_0p75_at_k", "k_efficiency_hit_0p75_ref.png", "Hit@0.75 at K"),
        ("mean_best_rmsd_at_k", "k_efficiency_best_rmsd_ref.png", "Mean best RMSD at K"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        group_cols = ["source"]
        if "sampling_mode" in ref.columns:
            group_cols.append("sampling_mode")
        if "summary_mode" in ref.columns:
            group_cols.append("summary_mode")
        for key, frame in ref.groupby(group_cols, dropna=False, sort=True):
            frame = frame.sort_values("k")
            if not isinstance(key, tuple):
                key = (key,)
            label = " / ".join(str(value) for value in key if str(value) not in {"", "nan", "None"})
            ax.plot(frame["k"], frame[y_col], marker="o", linewidth=1.5, markersize=3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title("CASF K-efficiency (ref)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
        fig.tight_layout()
        fig.savefig(paths.figures / filename, dpi=180)
        plt.close(fig)


def plot_frontier(summary: pd.DataFrame, paths: OutputPaths) -> None:
    ref = summary[summary["ligand_set"].astype(str) == "ref"].copy()
    if ref.empty:
        ref = summary.copy()
    for x_col, y_col, filename, y_label in [
        ("mean_greedy_clusters_1p0", "mean_casf_hit_0p75", "frontier_hit_0p75_vs_clusters_ref.png", "CASF Hit@0.75"),
        ("mean_pb_pass_rate", "mean_casf_best_rmsd", "frontier_rmsd_vs_validity_ref.png", "Mean CASF best RMSD"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = pd.to_numeric(ref["mean_pb_pass_rate"], errors="coerce")
        sizes = pd.to_numeric(ref["mean_conformer_count"], errors="coerce").fillna(20).clip(lower=20, upper=500)
        scatter = ax.scatter(ref[x_col], ref[y_col], c=colors, s=sizes, cmap="viridis", alpha=0.85, edgecolor="black", linewidth=0.4)
        for _, row in ref.iterrows():
            if int(row.get("frontier_rank", 99)) == 1:
                ax.annotate(str(row["family"]), (row[x_col], row[y_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_label)
        ax.set_title("CASF recovery-validity frontier (ref)")
        fig.colorbar(scatter, ax=ax, label="Mean PB pass rate")
        fig.tight_layout()
        fig.savefig(paths.figures / filename, dpi=180)
        plt.close(fig)


def _write_chembl_k_per_ligand_sqlite(connection: sqlite3.Connection, name: str, frame: pd.DataFrame) -> None:
    identity_columns = ["ligand_set", "run_id", "run_label", "source", "family", "tier", "method"]
    ligand_columns = ["identity_id", "ligand_id", "mol_id", "n_available_pb"]
    value_columns = [
        "identity_id",
        "ligand_id",
        "sampling_mode",
        "seed",
        "k",
        "n_used",
        "casf_best_rmsd_at_k",
        "casf_hit_0p25_at_k",
        "casf_hit_0p5_at_k",
        "casf_hit_0p75_at_k",
        "casf_hit_2p0_at_k",
    ]
    if frame.empty or any(column not in frame.columns for column in identity_columns):
        frame.to_sql(name, connection, if_exists="replace", index=False)
        return

    identity = frame[identity_columns].drop_duplicates().reset_index(drop=True)
    identity.insert(0, "identity_id", range(len(identity)))
    merged = frame.merge(identity, on=identity_columns, how="left", validate="many_to_one")
    ligands = merged[["identity_id", "mol_id", "n_available_pb"]].drop_duplicates().reset_index(drop=True)
    ligands.insert(1, "ligand_id", range(len(ligands)))
    merged = merged.merge(ligands, on=["identity_id", "mol_id", "n_available_pb"], how="left", validate="many_to_one")
    values = merged[value_columns].copy()

    connection.execute(f'DROP VIEW IF EXISTS "{name}"')
    connection.execute(f'DROP TABLE IF EXISTS "{name}"')
    for internal in [f"{name}__identity", f"{name}__ligands", f"{name}__values"]:
        connection.execute(f'DROP TABLE IF EXISTS "{internal}"')

    identity.to_sql(f"{name}__identity", connection, if_exists="replace", index=False)
    ligands[ligand_columns].to_sql(f"{name}__ligands", connection, if_exists="replace", index=False)
    values.to_sql(f"{name}__values", connection, if_exists="replace", index=False)
    connection.execute(
        f'''
        CREATE VIEW "{name}" AS
        SELECT
            i.ligand_set,
            i.run_id,
            i.run_label,
            i.source,
            i.family,
            i.tier,
            i.method,
            v.sampling_mode,
            v.seed,
            v.k,
            l.mol_id,
            l.n_available_pb,
            v.n_used,
            v.casf_best_rmsd_at_k,
            v.casf_hit_0p25_at_k,
            v.casf_hit_0p5_at_k,
            v.casf_hit_0p75_at_k,
            v.casf_hit_2p0_at_k
        FROM "{name}__values" AS v
        JOIN "{name}__ligands" AS l
            ON v.identity_id = l.identity_id AND v.ligand_id = l.ligand_id
        JOIN "{name}__identity" AS i
            ON v.identity_id = i.identity_id
        '''
    )


def _write_sqlite_table(connection: sqlite3.Connection, name: str, frame: pd.DataFrame) -> None:
    if len(frame.columns) == 0:
        return
    if name == COMPACT_CHEMBL_K_TABLE:
        _write_chembl_k_per_ligand_sqlite(connection, name, frame)
        return
    frame.to_sql(name, connection, if_exists="replace", index=False)


def write_extended_sqlite(paths: OutputPaths, tables: dict[str, pd.DataFrame]) -> None:
    tmp = paths.extended_db.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    with sqlite3.connect(tmp) as connection:
        for name, frame in tables.items():
            _write_sqlite_table(connection, name, frame)
        connection.execute("PRAGMA user_version = 1")
        connection.execute("PRAGMA optimize")
    tmp.replace(paths.extended_db)


def write_dashboard_copy_with_extended_tables(
    source_db: Path, paths: OutputPaths, tables: dict[str, pd.DataFrame]
) -> None:
    tmp = paths.dashboard_copy_db.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(source_db, tmp)
    with sqlite3.connect(tmp) as connection:
        for name, frame in tables.items():
            _write_sqlite_table(connection, name, frame)
        connection.execute("PRAGMA optimize")
    tmp.replace(paths.dashboard_copy_db)


def write_readme(paths: OutputPaths, db_path: Path, skip_analysis_1: bool) -> None:
    text = f"""# Extended CASF Analysis

Generated from:

```bash
casf-extended-analysis --dashboard-db {db_path} --output-dir {paths.root}
```

This directory contains additive outputs only. The original dashboard database is read-only input.

- `tables/`: extended CSV tables
- `reports/`: Markdown summaries
- `figures/`: PNG figures
- `cache/per_conformer_casf_rmsd_parts/`: cached per-conformer CASF RMSD exports used for K-efficiency
- `extended_casf_analysis.sqlite`: sidecar SQLite database containing the new extended tables
- `casf_analysis_dashboard_extended.sqlite`: copy of the dashboard database with the new extended tables appended

Analysis 1 ligand-universe/missingness audit was skipped by request: {skip_analysis_1}.
"""
    write_report(text, paths.root / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-values", default="1,2,5,10,25,50,100,250,500,1000")
    parser.add_argument("--k-subsamples", type=int, default=20)
    parser.add_argument("--k-seed", type=int, default=1729)
    parser.add_argument("--chembl-k-random-repeats", type=int, default=100)
    parser.add_argument("--chembl-k-seed-base", type=int, default=91001)
    parser.add_argument("--k-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--recompute-k-rmsd", action="store_true")
    parser.add_argument("--skip-k-efficiency", action="store_true")
    parser.add_argument(
        "--allow-k-efficiency-failures",
        action="store_true",
        help="Keep the tasks that succeeded instead of aborting, reporting the rest in "
        "extended_k_efficiency_failures.csv.",
    )
    parser.add_argument("--energy-window-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--recompute-energy-window", action="store_true")
    parser.add_argument("--skip-energy-window", action="store_true")
    parser.add_argument(
        "--generation-root-overrides",
        type=Path,
        default=None,
        help="YAML {generation_roots: {run_id: root}} pointing the per-conformer analyses "
        "(K-efficiency, energy windows) at run roots that carry generation/. "
        "Generate with scripts/build_generation_root_overrides.py.",
    )
    parser.add_argument("--fail-on-sanity-error", action="store_true")
    parser.add_argument("--skip-analysis-1", action="store_true", default=True)
    args = parser.parse_args()

    paths = OutputPaths(args.output_dir.resolve())
    paths.mkdirs()
    data = load_inputs(args.dashboard_db.resolve())
    per_ligand = data["per_ligand_long"]
    comparison_rows = data["comparison_rows"]
    analysis_sources = data["analysis_sources"]
    if args.generation_root_overrides is not None:
        analysis_sources = apply_generation_root_overrides(
            analysis_sources, args.generation_root_overrides.resolve()
        )

    sanity = build_sanity_checks(per_ligand, comparison_rows)
    write_csv(sanity, paths.tables / "extended_sanity_checks.csv")
    failing = sanity[(sanity["severity"] == "error") & (sanity["status"] == "fail")]
    if args.fail_on_sanity_error and not failing.empty:
        write_report(
            "# Extended Sanity Checks\n\nSanity checks failed.\n\n" + table_preview(failing),
            paths.reports / "extended_sanity_checks.md",
        )
        raise SystemExit("Sanity checks failed; see extended_sanity_checks.csv")

    k_values = [int(value) for value in str(args.k_values).split(",") if value.strip()]
    if args.skip_k_efficiency:
        k_eff, k_eff_strata = skipped_k_efficiency_frames(k_values, "--skip-k-efficiency was set")
        k_ligand_rows = pd.DataFrame()
        chembl_k_eff = pd.DataFrame()
        chembl_k_ligand_rows = pd.DataFrame()
    else:
        k_eff, k_eff_strata, k_ligand_rows = build_k_efficiency(
            per_ligand,
            analysis_sources,
            paths,
            k_values,
            subsamples=args.k_subsamples,
            seed=args.k_seed,
            workers=args.k_workers,
            recompute_rmsd=args.recompute_k_rmsd,
            allow_failures=args.allow_k_efficiency_failures,
        )
        chembl_k_eff, chembl_k_ligand_rows = build_chembl_k_efficiency(
            per_ligand,
            analysis_sources,
            paths,
            k_values,
            seed_base=args.chembl_k_seed_base,
            random_repeats=args.chembl_k_random_repeats,
            workers=args.k_workers,
            recompute_rmsd=args.recompute_k_rmsd,
        )
        if not chembl_k_eff.empty:
            k_eff = pd.concat([k_eff, chembl_k_eff], ignore_index=True, sort=False)
    write_k_efficiency_outputs(paths, k_eff, k_eff_strata)
    if not k_ligand_rows.empty:
        write_csv(k_ligand_rows, paths.tables / "extended_k_efficiency_per_ligand.csv")
    if not chembl_k_ligand_rows.empty:
        write_csv(chembl_k_ligand_rows, paths.tables / "extended_chembl_k_efficiency_per_ligand.csv")
    k_comparisons = build_k_efficiency_comparisons(k_ligand_rows, chembl_k_ligand_rows)
    write_csv(k_comparisons, paths.tables / "extended_k_efficiency_comparisons.csv")
    k_manifest_path = paths.tables / "extended_per_conformer_casf_rmsd_manifest.csv"
    k_rmsd_manifest = pd.read_csv(k_manifest_path) if k_manifest_path.exists() else pd.DataFrame()

    if args.skip_energy_window:
        energy_window_per_ligand = pd.DataFrame()
        energy_window_summary = pd.DataFrame()
        energy_window_comparisons = pd.DataFrame()
    else:
        energy_window_per_ligand, energy_window_summary, energy_window_comparisons = build_energy_window_analysis(
            per_ligand,
            analysis_sources,
            paths,
            workers=args.energy_window_workers,
            recompute=args.recompute_energy_window,
        )
    write_csv(energy_window_per_ligand, paths.tables / "extended_energy_window_per_ligand.csv")
    write_csv(energy_window_summary, paths.tables / "extended_energy_window_summary.csv")
    write_csv(energy_window_comparisons, paths.tables / "extended_energy_window_comparisons.csv")

    paired_tables: dict[str, pd.DataFrame] = {}
    detail_tables: dict[str, pd.DataFrame] = {}
    for label, source in PAIRING_BASELINES.items():
        paired, details = build_paired_delta_table(per_ligand, source)
        paired_tables[label] = paired
        detail_tables[label] = details
        write_csv(paired, paths.tables / f"extended_paired_delta_vs_{label}.csv")
        write_csv(details, paths.tables / f"extended_paired_delta_extreme_ligands_vs_{label}.csv")
    paired_report = "# Paired Delta Summary\n\n"
    for label, frame in paired_tables.items():
        total = frame[frame["stratum_type"] == "total"].copy()
        preview_cols = [
            "ligand_set",
            "source",
            "n_common_ligands",
            "mean_delta_casf_best_rmsd",
            "win_rate_0p1A",
            "loss_rate_0p1A",
            "delta_hit_0p75",
        ]
        paired_report += f"## Baseline: {label}\n\n" + table_preview(total[preview_cols].sort_values(["ligand_set", "mean_delta_casf_best_rmsd"]).head(20)) + "\n\n"
        detail = detail_tables[label].copy()
        if detail.empty:
            paired_report += "No per-ligand extreme rows were generated.\n\n"
            continue
        detail_cols = [
            "ligand_set",
            "source",
            "direction",
            "mol_id",
            "baseline_casf_best_rmsd",
            "method_casf_best_rmsd",
            "delta_casf_best_rmsd",
            "baseline_casf_hit_0p75",
            "method_casf_hit_0p75",
        ]
        for (ligand_set_name, source_name, direction), group in detail.groupby(
            ["ligand_set", "source", "direction"], dropna=False, sort=True
        ):
            paired_report += f"### {ligand_set_name} / {source_name} / {direction}\n\n"
            paired_report += markdown_table(group[detail_cols], max_rows=20) + "\n\n"
    write_report(paired_report, paths.reports / "extended_paired_delta_summary.md")

    frontier = build_frontier_summary(per_ligand)
    write_csv(frontier, paths.tables / "extended_frontier_summary.csv")
    plot_frontier(frontier, paths)
    frontier_report = (
        "# Diversity-Validity-Recovery Frontier\n\n"
        "Frontier rank 1 rows are non-dominated when maximizing Hit@0.75 and PB pass rate while minimizing mean conformer count.\n\n"
        + table_preview(frontier[frontier["frontier_rank"] == 1].sort_values(["ligand_set", "source"]))
    )
    write_report(frontier_report, paths.reports / "extended_frontier_summary.md")

    pb_failures, pb_tail = build_pb_failure_tables(per_ligand)
    write_csv(pb_failures, paths.tables / "extended_pb_failure_mechanisms.csv")
    write_csv(pb_tail, paths.tables / "extended_pb_failure_tail_ligands.csv")
    pb_report = (
        "# PoseBusters Failure Mechanisms\n\n"
        "Automated PoseBusters labels are summarized directly; no manual chirality adjudication is performed.\n\n"
        + table_preview(pb_failures.sort_values("total_pb_fail_rate", ascending=False).head(25))
    )
    write_report(pb_report, paths.reports / "extended_pb_failure_mechanisms.md")

    difficulty, rescue_cases = build_bound_pose_difficulty(per_ligand)
    write_csv(difficulty, paths.tables / "extended_bound_pose_difficulty.csv")
    write_csv(rescue_cases, paths.tables / "extended_rescue_cases.csv")
    rescue_report = (
        "# Bound-Pose Difficulty And Rescue\n\n"
        "Difficulty bins are defined from ChEMBL3D-PB CASF best RMSD on common ligands.\n\n"
        + table_preview(difficulty.sort_values(["ligand_set", "source", "difficulty_bin"]).head(40))
    )
    write_report(rescue_report, paths.reports / "extended_bound_pose_rescue.md")

    sanity_report = (
        "# Extended Sanity Checks\n\n"
        "Manuscript-ready status: deterministic paired/frontier/PB summaries are usable for rows without sanity errors. "
        "K-efficiency is computed for generation methods and ChEMBL3D-PB from cached per-conformer CASF RMSDs when conformer artifacts are available. "
        "Energy-window tables use per-ligand relative MMFF94s energies and PB-passing conformer sets.\n\n"
        + table_preview(sanity)
    )
    write_report(sanity_report, paths.reports / "extended_sanity_checks.md")

    extended_tables = {
        "extended_k_efficiency": k_eff,
        "extended_k_efficiency_strata": k_eff_strata,
        "extended_k_efficiency_per_ligand": k_ligand_rows,
        "extended_chembl_k_efficiency_per_ligand": chembl_k_ligand_rows,
        "extended_k_efficiency_comparisons": k_comparisons,
        "extended_energy_window_per_ligand": energy_window_per_ligand,
        "extended_energy_window_summary": energy_window_summary,
        "extended_energy_window_comparisons": energy_window_comparisons,
        "extended_per_conformer_casf_rmsd_manifest": k_rmsd_manifest,
        "extended_paired_delta_vs_chembl3d_pb": paired_tables["chembl3d_pb"],
        "extended_paired_delta_vs_rdkit_raw": paired_tables["rdkit_raw"],
        "extended_paired_delta_extreme_ligands_vs_chembl3d_pb": detail_tables["chembl3d_pb"],
        "extended_paired_delta_extreme_ligands_vs_rdkit_raw": detail_tables["rdkit_raw"],
        "extended_frontier_summary": frontier,
        "extended_pb_failure_mechanisms": pb_failures,
        "extended_pb_failure_tail_ligands": pb_tail,
        "extended_bound_pose_difficulty": difficulty,
        "extended_rescue_cases": rescue_cases,
        "extended_sanity_checks": sanity,
    }
    write_extended_sqlite(paths, extended_tables)
    write_dashboard_copy_with_extended_tables(args.dashboard_db.resolve(), paths, extended_tables)
    write_readme(paths, args.dashboard_db.resolve(), args.skip_analysis_1)

    print(f"Wrote extended CASF outputs to {paths.root}")
    print(f"Wrote sidecar SQLite DB to {paths.extended_db}")
    print(f"Wrote dashboard DB copy with extended tables to {paths.dashboard_copy_db}")


if __name__ == "__main__":
    main()
