#!/usr/bin/env python3
"""Exact CASF16-to-ChEMBL3D matches via regenerated CASF SMILES only."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Geometry import Point3D
except ImportError as exc:  # pragma: no cover
    Chem = None
    Point3D = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None

from casf_benchmark.chembl3d.identity import (
    AmbiguousStereoIdentityError,
    StereoIdentity,
    identities_match,
    stereo_identity_from_mol,
    stereo_identity_from_smiles,
)
from casf_benchmark.chembl3d.loader import (
    count_matching_chembl3d_conformers,
    prepare_torsion_ref_mol,
)
from casf_benchmark.generation.conformer_sets import get_rotatable_torsions
from casf_benchmark.paths import (
    DEFAULT_CASF16_DATA,
    DEFAULT_CASF_LIGAND_DIR,
    DEFAULT_CHEMBL3D_INDEX_CSV,
    DEFAULT_CHEMBL_DATASET_ROOT,
    DEFAULT_CHEMBL_MAP_CSV,
    WEKA_DATA_ROOT,
)

DEFAULT_CASF16_DIR = DEFAULT_CASF16_DATA
DEFAULT_LIGAND_DIR = DEFAULT_CASF_LIGAND_DIR
DEFAULT_CHEMBL_INDEX = DEFAULT_CHEMBL3D_INDEX_CSV
DEFAULT_DATA_DIR = WEKA_DATA_ROOT / "casf16"
DEFAULT_OUTPUT_CSV = DEFAULT_CHEMBL_MAP_CSV
DEFAULT_TOPOLOGY_ROOT = DEFAULT_CHEMBL_DATASET_ROOT / "topologies"

GROUP_RE = re.compile(r"^\d{3}$")

OUTPUT_FIELDS = (
    "ligand_id",
    "source_file",
    "casf_explicit_isomeric_smiles",
    "casf_explicit_nonisomeric_smiles",
    "casf_heavy_isomeric_smiles",
    "casf_heavy_nonisomeric_smiles",
    "chembl3d_group",
    "chembl3d_mol_id",
    "chembl3d_isomeric_smiles",
    "chembl3d_sdf_record_index",
    "chembl3d_inchi_stereo",
    "conformer_count",
    "chembl3d_index_conformer_count",
)


@dataclass(frozen=True)
class ChemblIndexRow:
    group: str
    mol_id: str
    isomeric_canonical_smiles: str
    conformer_count: int


@dataclass(frozen=True)
class TopologyRecord:
    sdf_record_index: int
    mol_id: str
    mol: object
    identity: StereoIdentity | None


@dataclass(frozen=True)
class SelectedChemblHit:
    index_row: ChemblIndexRow
    sdf_record_index: int
    inchi_stereo: str
    topology_mol: object
    wrong_first_record: bool


@dataclass(frozen=True)
class CasfRegenSmiles:
    ligand_id: str
    source_file: str
    status: str
    explicit_isomeric_smiles: str = ""
    explicit_nonisomeric_smiles: str = ""
    heavy_isomeric_smiles: str = ""
    heavy_nonisomeric_smiles: str = ""
    error: str = ""


def require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RDKit is required. Run with a chemistry environment, "
            "for example `/home/mbedrosian/.conda/envs/chembl3d/bin/python`."
        ) from RDKIT_IMPORT_ERROR


def casf_regenerated_smiles(mol2_path: Path) -> CasfRegenSmiles:
    """RemoveHs -> AddHs (zero H coords) -> explicit and heavy canonical SMILES."""
    require_rdkit()
    ligand_id = mol2_path.stem
    source_file = mol2_path.name

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                return CasfRegenSmiles(
                    ligand_id=ligand_id,
                    source_file=source_file,
                    status="mol2_parse_failed",
                    error=f"sanitize_failed: {exc}",
                )
    if mol is None:
        return CasfRegenSmiles(
            ligand_id=ligand_id,
            source_file=source_file,
            status="mol2_parse_failed",
            error="MolFromMol2File returned None",
        )

    mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers():
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                conf.SetAtomPosition(atom.GetIdx(), Point3D(0.0, 0.0, 0.0))

    explicit_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    explicit_nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    heavy = Chem.RemoveHs(mol)
    heavy_isomeric = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    heavy_nonisomeric = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=False)

    return CasfRegenSmiles(
        ligand_id=ligand_id,
        source_file=source_file,
        status="ok",
        explicit_isomeric_smiles=explicit_isomeric,
        explicit_nonisomeric_smiles=explicit_nonisomeric,
        heavy_isomeric_smiles=heavy_isomeric,
        heavy_nonisomeric_smiles=heavy_nonisomeric,
    )


def discover_ligand_mol2_files(ligand_dir: Path) -> list[Path]:
    return sorted(ligand_dir.glob("*.mol2"))


def load_casf_regenerated(ligand_dir: Path) -> tuple[dict[str, CasfRegenSmiles], int, int]:
    by_ligand_id: dict[str, CasfRegenSmiles] = {}
    processed = 0
    failed = 0
    for mol2_path in discover_ligand_mol2_files(ligand_dir):
        processed += 1
        regen = casf_regenerated_smiles(mol2_path)
        by_ligand_id[regen.ligand_id] = regen
        if regen.status != "ok":
            failed += 1
    return by_ligand_id, processed, failed


def parse_index_row(fields: list[str]) -> ChemblIndexRow | None:
    if len(fields) == 7:
        group, mol_id, _original, isomeric, _heavy, _rot, raw_count = fields
        if not GROUP_RE.match(group):
            return None
        raw_count = raw_count.strip()
        return ChemblIndexRow(
            group=group,
            mol_id=mol_id,
            isomeric_canonical_smiles=isomeric,
            conformer_count=int(raw_count) if raw_count.isdigit() else 0,
        )

    if len(fields) >= 9:
        group, mol_id, isomeric, raw_count = fields[0], fields[2], fields[5], fields[8]
        if not GROUP_RE.match(group):
            return None
        raw_count = raw_count.strip()
        return ChemblIndexRow(
            group=group,
            mol_id=mol_id,
            isomeric_canonical_smiles=isomeric,
            conformer_count=int(raw_count) if raw_count.isdigit() else 0,
        )

    return None


def lookup_chembl_hits(
    path: Path,
    target_smiles: set[str],
) -> dict[str, list[ChemblIndexRow]]:
    hits: dict[str, list[ChemblIndexRow]] = {}
    dedupe: dict[tuple[str, str, str, str], ChemblIndexRow] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return hits
        for fields in reader:
            row = parse_index_row(fields)
            if row is None or row.isomeric_canonical_smiles not in target_smiles:
                continue
            key = (row.isomeric_canonical_smiles, row.group, row.mol_id)
            existing = dedupe.get(key)
            if existing is None or row.conformer_count > existing.conformer_count:
                dedupe[key] = row

    for row in dedupe.values():
        hits.setdefault(row.isomeric_canonical_smiles, []).append(row)
    for rows in hits.values():
        rows.sort(key=lambda item: (item.group, item.mol_id))
    return hits


def _record_mol_id(mol: Chem.Mol) -> str:
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    prop_mol_id = mol.GetProp("mol_id") if mol.HasProp("mol_id") else ""
    return prop_mol_id or name


def build_topology_index(
    chembl_hits: dict[str, list[ChemblIndexRow]],
    topology_root: Path,
) -> dict[tuple[str, str], list[TopologyRecord]]:
    """Index every SDF record for requested mol_ids.

    Flipper stereoisomers share a parent id, so later records must not be skipped.
    """
    needed_by_group: dict[str, set[str]] = {}
    for hits in chembl_hits.values():
        for hit in hits:
            needed_by_group.setdefault(hit.group, set()).add(hit.mol_id)
    groups = sorted(needed_by_group)
    print(f"indexing_topology_groups={len(groups)}", flush=True)
    topology_index: dict[tuple[str, str], list[TopologyRecord]] = {}
    for idx, group in enumerate(groups, start=1):
        needed = needed_by_group[group]
        sdf_path = topology_root / f"{int(group):03d}.sdf"
        if sdf_path.is_file():
            for record_index, mol in enumerate(
                Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            ):
                if mol is None:
                    continue
                mol_id = _record_mol_id(mol)
                if not mol_id or mol_id not in needed:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    identity = None
                else:
                    identity = stereo_identity_from_mol(mol, from_3d=True)
                topology_index.setdefault((group, mol_id), []).append(
                    TopologyRecord(
                        sdf_record_index=record_index,
                        mol_id=mol_id,
                        mol=Chem.Mol(mol),
                        identity=identity,
                    )
                )
        if idx % 10 == 0 or idx == len(groups):
            print(f"topology_groups_indexed={idx}/{len(groups)}", flush=True)
    return topology_index


def matching_topology_records(
    hit: ChemblIndexRow,
    topology_index: dict[tuple[str, str], list[TopologyRecord]],
) -> list[TopologyRecord]:
    expected = stereo_identity_from_smiles(hit.isomeric_canonical_smiles)
    records = topology_index.get((hit.group, hit.mol_id), [])
    return [record for record in records if identities_match(record.identity, expected)]


def pick_chembl_hit(
    hits: tuple[ChemblIndexRow, ...],
    topology_index: dict[tuple[str, str], list[TopologyRecord]] | None,
    *,
    require_topology_sdf: bool = True,
) -> SelectedChemblHit | ChemblIndexRow | None:
    if not hits:
        return None
    if topology_index is not None:
        for hit in hits:
            matches = matching_topology_records(hit, topology_index)
            if len(matches) > 1:
                indices = [record.sdf_record_index for record in matches]
                raise AmbiguousStereoIdentityError(
                    f"Multiple topology SDF records match {hit.mol_id} stereo "
                    f"{hit.isomeric_canonical_smiles!r}: record_indices={indices}"
                )
            if len(matches) != 1:
                continue
            record = matches[0]
            first_index = min(item.sdf_record_index for item in topology_index.get((hit.group, hit.mol_id), [record]))
            inchi_stereo = record.identity.inchi_stereo if record.identity is not None else ""
            return SelectedChemblHit(
                index_row=hit,
                sdf_record_index=record.sdf_record_index,
                inchi_stereo=inchi_stereo,
                topology_mol=record.mol,
                wrong_first_record=record.sdf_record_index != first_index,
            )
        if require_topology_sdf:
            return None
    if require_topology_sdf:
        return None
    return hits[0]


def chembl_hit_has_rotatable_bonds(selected: SelectedChemblHit) -> bool:
    if prepare_torsion_ref_mol is None or get_rotatable_torsions is None:
        return True
    ref = prepare_torsion_ref_mol(selected.topology_mol)
    return ref is not None and bool(get_rotatable_torsions(ref))


def resolve_eligible_chembl_hit(
    hits: tuple[ChemblIndexRow, ...],
    topology_index: dict[tuple[str, str], list[TopologyRecord]] | None,
) -> SelectedChemblHit | None | str:
    """Return a selected ChEMBL hit, None (missing SDF), or 'no_rotatable_bonds'."""
    if not hits:
        return None
    hit = pick_chembl_hit(hits, topology_index, require_topology_sdf=True)
    if hit is None or not isinstance(hit, SelectedChemblHit):
        return None
    if topology_index is not None and not chembl_hit_has_rotatable_bonds(hit):
        return "no_rotatable_bonds"
    return hit


def filtered_conformer_count(
    selected: SelectedChemblHit,
    topology_root: Path,
    zarr_root: Path | None,
) -> tuple[int, int]:
    index_count = selected.index_row.conformer_count
    if zarr_root is None or not zarr_root.exists():
        return index_count, index_count
    try:
        filtered = count_matching_chembl3d_conformers(
            selected.index_row.group,
            selected.index_row.mol_id,
            topology_root,
            zarr_root,
            expected_smiles=selected.index_row.isomeric_canonical_smiles,
            sdf_record_index=selected.sdf_record_index,
        )
    except Exception as exc:
        print(
            f"WARNING: could not recount filtered conformers for "
            f"{selected.index_row.group}/{selected.index_row.mol_id}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return index_count, index_count
    return filtered, index_count


def build_exact_match_rows(
    casf_by_ligand: dict[str, CasfRegenSmiles],
    chembl_hits: dict[str, list[ChemblIndexRow]],
    topology_index: dict[tuple[str, str], list[TopologyRecord]] | None = None,
    *,
    topology_root: Path | None = None,
    zarr_root: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    excluded: dict[str, list[str]] = {
        "no_smiles_match": [],
        "missing_topology_sdf": [],
        "no_rotatable_bonds": [],
    }
    eligible_by_smiles: dict[str, SelectedChemblHit | None | str] = {}
    hit_smiles = list(chembl_hits.items())
    for idx, (smiles, hits) in enumerate(hit_smiles, start=1):
        eligible_by_smiles[smiles] = resolve_eligible_chembl_hit(tuple(hits), topology_index)
        if idx % 200 == 0 or idx == len(hit_smiles):
            print(f"eligible_smiles_resolved={idx}/{len(hit_smiles)}", flush=True)

    filtered_count_cache: dict[tuple[str, str, str, int], tuple[int, int]] = {}
    for regen in casf_by_ligand.values():
        if regen.status != "ok":
            continue
        hits = chembl_hits.get(regen.heavy_isomeric_smiles)
        if not hits:
            excluded["no_smiles_match"].append(regen.ligand_id)
            continue
        eligible = eligible_by_smiles[regen.heavy_isomeric_smiles]
        if eligible == "no_rotatable_bonds":
            excluded["no_rotatable_bonds"].append(regen.ligand_id)
            continue
        if eligible is None or not isinstance(eligible, SelectedChemblHit):
            excluded["missing_topology_sdf"].append(regen.ligand_id)
            continue
        selected = eligible
        hit = selected.index_row
        count_key = (
            hit.group,
            hit.mol_id,
            hit.isomeric_canonical_smiles,
            selected.sdf_record_index,
        )
        if count_key not in filtered_count_cache:
            if topology_root is not None:
                filtered_count_cache[count_key] = filtered_conformer_count(
                    selected, topology_root, zarr_root
                )
            else:
                filtered_count_cache[count_key] = (hit.conformer_count, hit.conformer_count)
        conformer_count, index_count = filtered_count_cache[count_key]
        rows.append(
            {
                "ligand_id": regen.ligand_id,
                "source_file": regen.source_file,
                "casf_explicit_isomeric_smiles": regen.explicit_isomeric_smiles,
                "casf_explicit_nonisomeric_smiles": regen.explicit_nonisomeric_smiles,
                "casf_heavy_isomeric_smiles": regen.heavy_isomeric_smiles,
                "casf_heavy_nonisomeric_smiles": regen.heavy_nonisomeric_smiles,
                "chembl3d_group": hit.group,
                "chembl3d_mol_id": hit.mol_id,
                "chembl3d_isomeric_smiles": hit.isomeric_canonical_smiles,
                "chembl3d_sdf_record_index": selected.sdf_record_index,
                "chembl3d_inchi_stereo": selected.inchi_stereo,
                "conformer_count": conformer_count,
                "chembl3d_index_conformer_count": index_count,
                "wrong_first_topology": selected.wrong_first_record,
            }
        )
    rows.sort(key=lambda row: (str(row["ligand_id"]), str(row["chembl3d_mol_id"])))
    return rows, excluded


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate CASF16 SMILES (RemoveHs -> AddHs with zero H coords) and "
            "write exact matches against the ChEMBL3D topology index. Output rows "
            "require a ChEMBL3D topology SDF stereoisomer that matches the mapping "
            "SMILES (not merely the first record for that mol_id) and at least one "
            "rotatable bond."
        )
    )
    parser.add_argument("--ligand-dir", type=Path, default=DEFAULT_LIGAND_DIR)
    parser.add_argument("--chembl-index", type=Path, default=DEFAULT_CHEMBL_INDEX)
    parser.add_argument("--topology-root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=DEFAULT_CHEMBL_DATASET_ROOT / "zarr_database",
        help=(
            "ChEMBL3D zarr root used to recount conformer_count for the selected "
            "stereoisomer. If missing, conformer_count is copied from the index "
            "(chembl3d_index_conformer_count)."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.ligand_dir.is_dir():
        raise FileNotFoundError(f"CASF ligand directory not found: {args.ligand_dir}")
    if not args.chembl_index.is_file():
        raise FileNotFoundError(f"ChEMBL3D index not found: {args.chembl_index}")

    casf_by_ligand, processed, failed = load_casf_regenerated(args.ligand_dir)
    target_smiles = {
        regen.heavy_isomeric_smiles
        for regen in casf_by_ligand.values()
        if regen.status == "ok" and regen.heavy_isomeric_smiles
    }
    print(f"casf_ligands_processed={processed}", flush=True)
    print(f"casf_regen_failures={failed}", flush=True)
    print(f"target_heavy_isomeric_smiles={len(target_smiles)}", flush=True)

    print(f"Scanning ChEMBL3D index: {args.chembl_index}", flush=True)
    chembl_hits = lookup_chembl_hits(args.chembl_index, target_smiles)
    print(f"chembl_hit_smiles={len(chembl_hits)}", flush=True)

    topology_index = build_topology_index(chembl_hits, args.topology_root)
    print(f"topology_index_entries={sum(len(records) for records in topology_index.values())}", flush=True)

    zarr_root = args.zarr_root if args.zarr_root.exists() else None
    if zarr_root is None:
        print(
            f"zarr_root_missing={args.zarr_root}; "
            "conformer_count will use unfiltered index counts until rematch with zarr",
            flush=True,
        )

    rows, excluded = build_exact_match_rows(
        casf_by_ligand,
        chembl_hits,
        topology_index,
        topology_root=args.topology_root,
        zarr_root=zarr_root,
    )
    matched_ligands = {row["ligand_id"] for row in rows}
    write_csv(args.output_csv, rows)

    wrong_first = [str(row["ligand_id"]) for row in rows if row.get("wrong_first_topology")]
    missing_record_index = [
        str(row["ligand_id"]) for row in rows if row.get("chembl3d_sdf_record_index") in (None, "")
    ]
    print(f"exact_matched_ligands={len(matched_ligands)}", flush=True)
    print(f"exact_match_rows={len(rows)}", flush=True)
    print(f"wrong_first_topology_ligands={len(wrong_first)}", flush=True)
    if wrong_first:
        print(f"wrong_first_topology_ligand_ids={','.join(sorted(wrong_first))}", flush=True)
    print(f"rows_missing_sdf_record_index={len(missing_record_index)}", flush=True)
    for reason, ligand_ids in excluded.items():
        print(f"excluded_{reason}={len(ligand_ids)}", flush=True)
        if ligand_ids:
            print(f"excluded_{reason}_ligands={','.join(sorted(ligand_ids))}", flush=True)
    print(f"output_csv={args.output_csv.resolve()}", flush=True)
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
