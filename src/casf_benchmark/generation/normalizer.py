"""Normalize external CASF conformer generation outputs into analyzer-ready sets."""

from __future__ import annotations

import csv
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from casf_benchmark.chembl3d.loader import load_torsion_ref
from casf_benchmark.generation.conformer_sets import (
    FilterStats,
    InputMolecule,
    MethodResult,
    chembl_count_target,
    dynamic_candidate_count,
    empty_result,
    finalize_pipeline_pair,
    indices_sidecar_path,
    load_intersection_molecules,
    safe_mol_id,
    select_dynamic_indices,
    write_manifest,
)

SAMPLED_TIERS = ("fixed", "dynamic", "chembl_count")
TIER_SUFFIXES = tuple(f"_{tier}" for tier in SAMPLED_TIERS)
MANIFEST_PARTS_DIRNAME = "manifest_parts_normalized"


@dataclass(frozen=True)
class SourcePool:
    method: str
    output_stem: str
    row: dict[str, object]


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path, sep="\t")


def restore_manifest_backup(generation_dir: Path, backup_name: str, manifest_name: str = "manifest.tsv") -> Path:
    backup_path = generation_dir / backup_name
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup manifest not found: {backup_path}")
    manifest_path = generation_dir / manifest_name
    shutil.copy2(backup_path, manifest_path)
    return manifest_path


def quarantine_paths(paths: Iterable[Path], quarantine_root: Path) -> list[Path]:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        target = quarantine_root / path.name
        if target.exists():
            raise FileExistsError(f"Quarantine target already exists: {target}")
        shutil.move(str(path), str(target))
        moved.append(target)
    return moved


def strip_sampled_tier(method: str) -> tuple[str, str | None]:
    for tier in SAMPLED_TIERS:
        suffix = f"_{tier}"
        if method.endswith(suffix):
            return method[: -len(suffix)], tier
    return method, None


def is_sampled_method(method: str, set_tier: object | None = None) -> bool:
    _stem, suffix_tier = strip_sampled_tier(method)
    if suffix_tier is not None:
        return True
    tier = str(set_tier or "").strip()
    return tier in SAMPLED_TIERS


def discover_fixed_source_pools(
    manifest_df: pd.DataFrame,
    methods: Sequence[str] | None = None,
) -> list[SourcePool]:
    if "generation_method" not in manifest_df.columns:
        raise ValueError("Manifest must contain generation_method")
    requested = set(methods or [])
    pools: list[SourcePool] = []
    for method, frame in manifest_df.groupby("generation_method", sort=False):
        method = str(method)
        if requested and method not in requested:
            continue
        first = frame.iloc[0].to_dict()
        if is_sampled_method(method, first.get("set_tier")):
            continue
        pools.append(SourcePool(method=method, output_stem=method, row=first))
    if requested:
        found = {pool.method for pool in pools}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested fixed source method(s) not found: {missing}")
    return pools


def load_multi_record_sdf(path: Path) -> Chem.Mol:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-pool SDF: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Fixed-pool SDF is empty: {path}")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    records = [mol for mol in supplier if mol is not None]
    if not records:
        raise ValueError(f"No conformers could be loaded from {path}")

    template = Chem.Mol(records[0])
    template.RemoveAllConformers()
    out = Chem.Mol(template)
    out.RemoveAllConformers()
    for conf_id, record in enumerate(records):
        if record.GetNumConformers() == 0:
            continue
        conf = Chem.Conformer(record.GetConformer(0))
        conf.SetId(conf_id)
        out.AddConformer(conf, assignId=True)
    if out.GetNumConformers() == 0:
        raise ValueError(f"No conformers with coordinates could be loaded from {path}")
    return out


def _int_from_row(row: dict[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool_from_row(row: dict[str, object], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return default


def _filter_stats_from_row(row: dict[str, object], pool_size: int) -> FilterStats:
    # External generators may carry stale attempted-count columns. The analyzer
    # needs the actual fixed-pool size available on disk before PB filtering.
    return FilterStats(
        generated_candidates=pool_size,
        finite_rejected=_int_from_row(row, "finite_rejected"),
        clash_rejected=_int_from_row(row, "clash_rejected"),
        bond_rejected=_int_from_row(row, "bond_rejected"),
        stereo_rejected=_int_from_row(row, "stereo_rejected"),
        rmsd_rejected=_int_from_row(row, "rmsd_rejected"),
        pre_clash_passed=pool_size,
        generation_batches=_int_from_row(row, "generation_batches", 1),
    )


def _rotatable_bonds(row: dict[str, object], reference_mol: Chem.Mol) -> int:
    value = _int_from_row(row, "rotatable_bonds", 0)
    if value > 0:
        return value
    base = Chem.Mol(reference_mol)
    base.RemoveAllConformers()
    return int(Descriptors.NumRotatableBonds(base))


def materialize_source_pool(
    input_mol: InputMolecule,
    pool: SourcePool,
    generation_dir: Path,
    topology_root: Path,
    fixed_set_size: int,
    seed: int,
    posebusters_workers: int,
    posebusters_energy_threads: int,
) -> list[MethodResult]:
    start = time.perf_counter()
    output_methods = {tier: f"{pool.output_stem}_{tier}" for tier in SAMPLED_TIERS}
    output_paths = {
        method: generation_dir / method / f"{input_mol.mol_id}.sdf"
        for method in output_methods.values()
    }

    try:
        fixed_pre_pb = load_multi_record_sdf(generation_dir / pool.method / f"{input_mol.mol_id}.sdf")
    except Exception as exc:  # noqa: BLE001 - manifest records the failed ligand/method.
        status = f"fixed_pool_load_failed:{type(exc).__name__}"
        return [
            empty_result(input_mol, method, status, output_paths[method], time.perf_counter() - start)
            for method in output_methods.values()
        ]

    reference_mol, _reference_source = load_torsion_ref(
        input_mol.chembl3d_group,
        input_mol.chembl3d_mol_id,
        topology_root,
        Path(input_mol.source_input),
        expected_smiles=input_mol.smiles,
        sdf_record_index=input_mol.chembl3d_sdf_record_index,
    )
    if reference_mol is None:
        return [
            empty_result(
                input_mol,
                method,
                "reference_load_failed",
                output_paths[method],
                time.perf_counter() - start,
            )
            for method in output_methods.values()
        ]

    pool_size = fixed_pre_pb.GetNumConformers()
    rotatable_bonds = _rotatable_bonds(pool.row, reference_mol)
    dynamic_target = dynamic_candidate_count(rotatable_bonds)
    chembl_target = chembl_count_target(input_mol)
    fixed_target = _int_from_row(pool.row, "num_target_confs", fixed_set_size) or fixed_set_size
    dynamic_indices = select_dynamic_indices(
        pool_size,
        dynamic_target,
        seed,
        f"{input_mol.mol_id}:{pool.output_stem}:dynamic_indices",
    )
    chembl_count_indices = select_dynamic_indices(
        pool_size,
        chembl_target,
        seed,
        f"{input_mol.mol_id}:{pool.output_stem}:chembl_count_indices",
    )
    template = Chem.Mol(fixed_pre_pb)
    template.RemoveAllConformers()
    stats = _filter_stats_from_row(pool.row, pool_size)
    pool_status = "ok" if pool_size >= fixed_target else "failed_to_fill_pool"

    return finalize_pipeline_pair(
        input_mol=input_mol,
        fixed_method=output_methods["fixed"],
        dynamic_method=output_methods["dynamic"],
        chembl_count_method=output_methods["chembl_count"],
        family=pool.output_stem,
        fixed_pre_pb=fixed_pre_pb,
        dynamic_indices=dynamic_indices,
        chembl_count_indices=chembl_count_indices,
        template=template,
        reference_mol=reference_mol,
        paths=output_paths,
        rotatable_bonds=rotatable_bonds,
        fixed_target=fixed_target,
        dynamic_target=dynamic_target,
        chembl_count_target=chembl_target,
        stats=stats,
        pool_status_value=pool_status,
        minimization_applied=_bool_from_row(pool.row, "minimization_applied"),
        minimization_input_confs=_int_from_row(pool.row, "minimization_input_confs"),
        minimization_failed=_int_from_row(pool.row, "minimization_failed"),
        minimization_invalid=_int_from_row(pool.row, "minimization_invalid"),
        minimization_error=_int_from_row(pool.row, "minimization_error"),
        post_min_clash_rejected=_int_from_row(pool.row, "post_min_clash_rejected"),
        posebusters_workers=posebusters_workers,
        posebusters_energy_threads=posebusters_energy_threads,
        pipeline_start_time=start,
    )


def materialize_generation_root(
    generation_dir: Path,
    manifest_path: Path,
    chembl_map_csv: Path,
    ligand_dir: Path,
    chembl_dataset_root: Path,
    methods: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    fixed_set_size: int = 1000,
    seed: int = 1729,
    posebusters_workers: int = 1,
    posebusters_energy_threads: int = 1,
) -> list[MethodResult]:
    manifest_df = read_manifest(manifest_path)
    pools = discover_fixed_source_pools(manifest_df, methods)
    molecules = load_intersection_molecules(chembl_map_csv, ligand_dir, limit=limit, offset=offset)
    topology_root = chembl_dataset_root / "topologies"

    rows: list[MethodResult] = []
    for pool in pools:
        for index, input_mol in enumerate(molecules, start=1):
            print(
                f"[{pool.method}] {index}/{len(molecules)} {input_mol.mol_id}",
                flush=True,
            )
            rows.extend(
                materialize_source_pool(
                    input_mol=input_mol,
                    pool=pool,
                    generation_dir=generation_dir,
                    topology_root=topology_root,
                    fixed_set_size=fixed_set_size,
                    seed=seed,
                    posebusters_workers=posebusters_workers,
                    posebusters_energy_threads=posebusters_energy_threads,
                )
            )
    return rows


def write_manifest_part(parts_dir: Path, rows: Sequence[MethodResult]) -> Path:
    mol_ids = sorted({row.mol_id for row in rows})
    if len(mol_ids) != 1:
        raise ValueError(f"Manifest part must contain exactly one mol_id, got {mol_ids}")
    path = parts_dir / f"{mol_ids[0]}.tsv"
    write_manifest(path, list(rows))
    return path


def merge_manifest_parts(parts_dir: Path, manifest_path: Path) -> int:
    part_paths = sorted(parts_dir.glob("*.tsv"))
    if not part_paths:
        raise FileNotFoundError(f"No manifest parts found under {parts_dir}")
    frames = [pd.read_csv(path, sep="\t") for path in part_paths]
    manifest = pd.concat(frames, ignore_index=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return len(manifest)


def validate_sampled_generation_root(generation_dir: Path, manifest_path: Path) -> list[ValidationIssue]:
    manifest = read_manifest(manifest_path)
    issues: list[ValidationIssue] = []
    required = {"mol_id", "generation_method", "set_tier", "pb_input_confs", "kept_confs"}
    missing = required - set(manifest.columns)
    if missing:
        return [ValidationIssue("error", f"Manifest missing required columns: {sorted(missing)}")]

    for row in manifest.itertuples(index=False):
        method = str(getattr(row, "generation_method"))
        mol_id = str(getattr(row, "mol_id"))
        tier = str(getattr(row, "set_tier"))
        sdf_path = generation_dir / method / f"{mol_id}.sdf"
        if not sdf_path.exists():
            issues.append(ValidationIssue("error", f"Missing SDF for {method}/{mol_id}: {sdf_path}"))
            continue
        if tier in SAMPLED_TIERS:
            sidecar = indices_sidecar_path(sdf_path)
            if not sidecar.exists():
                issues.append(ValidationIssue("error", f"Missing indices sidecar for {method}/{mol_id}: {sidecar}"))
    return issues


def qwen_wrong_artifact_paths(generation_dir: Path) -> list[Path]:
    paths = [generation_dir / MANIFEST_PARTS_DIRNAME]
    for method_dir in sorted(generation_dir.iterdir() if generation_dir.exists() else []):
        if not method_dir.is_dir():
            continue
        if not method_dir.name.startswith("qwen_"):
            continue
        _stem, tier = strip_sampled_tier(method_dir.name)
        if tier in SAMPLED_TIERS:
            paths.append(method_dir)
    manifest = generation_dir / "manifest.tsv"
    if manifest.exists():
        paths.append(manifest)
    return paths


def load_method_list_from_csv(source_csv: Path, column: str = "generation_method") -> list[str]:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Expected {column!r} column in {source_csv}")
        return [safe_mol_id(str(row[column]).strip()) for row in reader if str(row[column]).strip()]
