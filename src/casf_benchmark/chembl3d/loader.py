"""Load ChEMBL3D conformer coordinates from zarr + topology SDF templates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    Chem = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None

try:
    import numpy as np
    import zarr
    from rdkit.Geometry import Point3D
except ImportError as exc:  # pragma: no cover
    np = None
    zarr = None
    Point3D = None
    ZARR_IMPORT_ERROR = exc
else:
    ZARR_IMPORT_ERROR = None

from casf_benchmark.chembl3d.identity import (
    AmbiguousStereoIdentityError,
    StereoIdentityMismatchError,
    expected_identity_from_mapping,
    mol_matches_expected_smiles,
    stereo_identity_from_mol,
    stereo_identity_from_smiles,
)

__all__ = [
    "AmbiguousStereoIdentityError",
    "StereoIdentityMismatchError",
    "count_matching_chembl3d_conformers",
    "decode_mol_id",
    "expected_identity_from_mapping",
    "find_mol_id_indices",
    "load_chembl3d_conformers",
    "load_topology_mol",
    "load_torsion_ref",
    "load_torsion_ref_from_chembl3d",
    "load_torsion_ref_from_mol2",
    "mol_matches_expected_smiles",
    "prepare_torsion_ref_mol",
    "require_dependencies",
    "stereo_identity_from_mol",
    "stereo_identity_from_smiles",
]


def require_dependencies() -> None:
    if ZARR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ChEMBL3D zarr loading requires numpy and zarr. Run with the ChEMBL3D "
            "environment, for example `/home/mbedrosian/.conda/envs/chembl3d/bin/python`."
        ) from ZARR_IMPORT_ERROR


def decode_mol_id(value: object) -> str:
    if isinstance(value, bytes):
        raw = value
    elif hasattr(value, "tobytes"):
        raw = value.tobytes()
    else:
        raw = bytes(value)
    return raw.decode("utf-8", errors="replace").rstrip(" \x00")


def find_mol_id_indices(mol_id_array, mol_id: str) -> list[int]:
    """Return every exact row index for a ChEMBL3D molecule id."""
    require_dependencies()
    encoded = mol_id.encode("utf-8")
    values = np.asarray(mol_id_array[:])
    matches = np.where(values == encoded)[0]
    return [int(index) for index in matches.tolist()]


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is required to load ChEMBL3D topology SDF files.") from RDKIT_IMPORT_ERROR


def _require_expected_smiles(expected_smiles: str) -> str:
    text = (expected_smiles or "").strip()
    if not text:
        raise ValueError("expected_smiles is required; (group, mol_id) is not a unique stereoisomer key")
    if stereo_identity_from_smiles(text) is None:
        raise ValueError(f"expected_smiles could not be parsed: {expected_smiles!r}")
    return text


def _topology_sdf_path(group: str, topology_root: Path) -> Path:
    return topology_root / f"{int(group):03d}.sdf"


def _record_mol_id(mol: Chem.Mol) -> str:
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    prop_mol_id = mol.GetProp("mol_id") if mol.HasProp("mol_id") else ""
    return prop_mol_id or name


def _sanitize_topology_mol(mol: Chem.Mol, mol_id: str, sdf_path: Path) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError(f"Failed to sanitize topology molecule {mol_id} in {sdf_path}") from exc
    return Chem.Mol(mol)


def _iter_sdf_records(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for record_index, mol in enumerate(supplier):
        yield record_index, mol


def load_topology_mol(
    group: str,
    mol_id: str,
    topology_root: Path,
    *,
    expected_smiles: str,
    sdf_record_index: int | None = None,
) -> Chem.Mol | None:
    """Load the unique topology SDF record for ``mol_id`` that matches ``expected_smiles``.

    ``(group, mol_id)`` is not unique: Flipper stereoisomers share a parent id.
    When ``sdf_record_index`` is set, that record is loaded and **validated**.
    Otherwise every record with that id is scanned. Zero matches return ``None``;
    duplicate records with the same stereo keep the lowest index.
    """
    _require_rdkit()
    expected = _require_expected_smiles(expected_smiles)
    sdf_path = _topology_sdf_path(group, topology_root)
    if not sdf_path.exists():
        return None

    if sdf_record_index is not None:
        selected: Chem.Mol | None = None
        for record_index, mol in _iter_sdf_records(sdf_path):
            if record_index != sdf_record_index:
                continue
            if mol is None:
                raise StereoIdentityMismatchError(
                    f"Stale chembl3d_sdf_record_index={sdf_record_index} for {mol_id} in {sdf_path}: "
                    "record could not be parsed"
                )
            record_id = _record_mol_id(mol)
            if record_id != mol_id:
                raise StereoIdentityMismatchError(
                    f"Stale chembl3d_sdf_record_index={sdf_record_index} for {mol_id} in {sdf_path}: "
                    f"record mol_id is {record_id!r}"
                )
            selected = _sanitize_topology_mol(mol, mol_id, sdf_path)
            break
        if selected is None:
            raise StereoIdentityMismatchError(
                f"Stale chembl3d_sdf_record_index={sdf_record_index} for {mol_id} in {sdf_path}: "
                "index is past the end of the SDF"
            )
        if not mol_matches_expected_smiles(selected, expected, from_3d=True):
            observed = stereo_identity_from_mol(selected, from_3d=True)
            raise StereoIdentityMismatchError(
                f"Stale chembl3d_sdf_record_index={sdf_record_index} for {mol_id} in {sdf_path}: "
                f"3D identity {observed} does not match expected SMILES {expected!r}"
            )
        selected.SetProp("chembl3d_sdf_record_index", str(sdf_record_index))
        return selected

    matches: list[tuple[int, Chem.Mol]] = []
    for record_index, mol in _iter_sdf_records(sdf_path):
        if mol is None:
            continue
        if _record_mol_id(mol) != mol_id:
            continue
        sanitized = _sanitize_topology_mol(mol, mol_id, sdf_path)
        if mol_matches_expected_smiles(sanitized, expected, from_3d=True):
            matches.append((record_index, sanitized))
    if not matches:
        return None
    if len(matches) > 1:
        indices = [index for index, _ in matches]
        print(
            f"WARNING: duplicate topology SDF records match {mol_id} stereo {expected!r} "
            f"in {sdf_path}: record_indices={indices}; using {min(indices)}",
            flush=True,
        )
        matches.sort(key=lambda item: item[0])
    record_index, selected = matches[0]
    selected.SetProp("chembl3d_sdf_record_index", str(record_index))
    return selected


def prepare_torsion_ref_mol(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None or mol.GetNumConformers() == 0:
        return None
    prepared = Chem.Mol(mol)
    if not any(atom.GetAtomicNum() == 1 for atom in prepared.GetAtoms()):
        prepared = Chem.AddHs(prepared, addCoords=True)
    return prepared


def load_torsion_ref_from_chembl3d(
    group: str,
    mol_id: str,
    topology_root: Path,
    *,
    expected_smiles: str,
    sdf_record_index: int | None = None,
) -> Chem.Mol | None:
    """Load torsion perturbation seed from ChEMBL3D topology SDF coordinates."""
    return prepare_torsion_ref_mol(
        load_topology_mol(
            group,
            mol_id,
            topology_root,
            expected_smiles=expected_smiles,
            sdf_record_index=sdf_record_index,
        )
    )


def load_torsion_ref_from_mol2(mol2_path: Path) -> Chem.Mol | None:
    """Load torsion seed from a CASF mol2 when ChEMBL3D topology SDF has no entry."""
    _require_rdkit()
    if not mol2_path.is_file():
        return None

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return prepare_torsion_ref_mol(mol)


def load_torsion_ref(
    group: str,
    mol_id: str,
    topology_root: Path,
    mol2_path: Path | None = None,
    *,
    expected_smiles: str,
    sdf_record_index: int | None = None,
) -> tuple[Chem.Mol | None, str]:
    """Prefer ChEMBL3D topology SDF; fall back to CASF mol2 only if stereo matches."""
    expected = _require_expected_smiles(expected_smiles)
    ref = load_torsion_ref_from_chembl3d(
        group,
        mol_id,
        topology_root,
        expected_smiles=expected,
        sdf_record_index=sdf_record_index,
    )
    if ref is not None:
        return ref, "chembl3d_topology_sdf"
    if mol2_path is not None:
        ref = load_torsion_ref_from_mol2(mol2_path)
        if ref is not None and mol_matches_expected_smiles(ref, expected, from_3d=True):
            return ref, "casf_mol2_fallback"
    return None, "unavailable"


def _set_coords_from_row(template: Chem.Mol, coord_row) -> Chem.Mol:
    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    n_atoms = mol.GetNumAtoms()
    if len(coord_row) != n_atoms:
        raise ValueError(
            f"Coordinate row length {len(coord_row)} != topology atom count {n_atoms}"
        )
    conf = Chem.Conformer(n_atoms)
    for atom_idx in range(n_atoms):
        point = Point3D(
            float(coord_row[atom_idx][0]),
            float(coord_row[atom_idx][1]),
            float(coord_row[atom_idx][2]),
        )
        conf.SetAtomPosition(atom_idx, point)
    mol.AddConformer(conf, assignId=True)
    return mol


def _topology_atomic_numbers(template: Chem.Mol) -> list[int]:
    return [atom.GetAtomicNum() for atom in template.GetAtoms()]


def _validate_atomic_numbers(numbers_row, expected: list[int], row_idx: int, mol_id: str) -> None:
    observed = [int(value) for value in np.asarray(numbers_row).tolist()]
    if observed != expected:
        raise ValueError(
            f"Atomic numbers row {row_idx} for {mol_id} does not match topology: "
            f"{observed} != {expected}"
        )


def load_chembl3d_conformers(
    group: str,
    mol_id: str,
    topology_root: Path,
    zarr_root: Path,
    *,
    expected_smiles: str,
    sdf_record_index: int | None = None,
    limit: int | None = None,
    row_indices: Sequence[int] | None = None,
) -> list[Chem.Mol]:
    require_dependencies()
    expected = _require_expected_smiles(expected_smiles)
    template = load_topology_mol(
        group,
        mol_id,
        topology_root,
        expected_smiles=expected,
        sdf_record_index=sdf_record_index,
    )
    if template is None:
        sdf_path = _topology_sdf_path(group, topology_root)
        raise FileNotFoundError(
            f"ChEMBL3D topology molecule {mol_id} not found in {sdf_path} "
            f"for expected SMILES {expected!r}"
        )

    group_path = zarr_root / f"{int(group):03d}"
    mol_id_path = group_path / "mol_id"
    coord_path = group_path / "coord"
    numbers_path = group_path / "numbers"
    if not mol_id_path.exists() or not coord_path.exists() or not numbers_path.exists():
        missing = [
            str(path)
            for path in (mol_id_path, coord_path, numbers_path)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing ChEMBL3D zarr array(s) for group {int(group):03d}: {missing}")

    mol_id_array = zarr.open_array(str(mol_id_path), mode="r")
    coord_array = zarr.open_array(str(coord_path), mode="r")
    numbers_array = zarr.open_array(str(numbers_path), mode="r")
    if row_indices is None:
        row_indices = find_mol_id_indices(mol_id_array, mol_id)
        if not row_indices:
            return []
    else:
        row_indices = [int(index) for index in row_indices]
        if not row_indices:
            return []

    mols: list[Chem.Mol] = []
    expected_numbers = _topology_atomic_numbers(template)
    template_record_index = (
        template.GetProp("chembl3d_sdf_record_index")
        if template.HasProp("chembl3d_sdf_record_index")
        else ""
    )
    for row_idx in row_indices:
        _validate_atomic_numbers(numbers_array[row_idx], expected_numbers, row_idx, mol_id)
        mol = _set_coords_from_row(template, coord_array[row_idx])
        if not mol_matches_expected_smiles(mol, expected, from_3d=True):
            continue
        mol.SetProp("_Name", mol_id)
        mol.SetProp("chembl3d_group", f"{int(group):03d}")
        mol.SetProp("chembl3d_mol_id", mol_id)
        mol.SetProp("chembl3d_isomeric_smiles", expected)
        if template_record_index:
            mol.SetProp("chembl3d_sdf_record_index", template_record_index)
        mols.append(mol)
        if limit is not None and len(mols) >= limit:
            break
    return mols


def count_matching_chembl3d_conformers(
    group: str,
    mol_id: str,
    topology_root: Path,
    zarr_root: Path,
    *,
    expected_smiles: str,
    sdf_record_index: int | None = None,
) -> int:
    """Count zarr rows whose reconstructed 3D stereo matches ``expected_smiles``."""
    return len(
        load_chembl3d_conformers(
            group,
            mol_id,
            topology_root,
            zarr_root,
            expected_smiles=expected_smiles,
            sdf_record_index=sdf_record_index,
        )
    )
