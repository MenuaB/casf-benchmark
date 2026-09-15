"""Stereoisomer identity for ChEMBL3D topology and zarr rows.

ChEMBL3D Flipper isomers share a parent ``mol_id``. Matching already selects one
isomeric SMILES; loaders must use that stereo (plus an optional SDF record index),
not ``(group, mol_id)``. Property tags on the SDF are not identity.

Identity is reconstructed from 3D coordinates the same way PoseBusters checks
molecular identity: assign stereo from the conformer, then compare canonical
heavy isomeric SMILES **and** InChI stereo layers (``/t``, ``/b``, ``/m``, ``/s``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import hashlib

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    Chem = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None

INCHI_STEREO_LAYER_PREFIXES = ("b", "t", "m", "s")


class StereoIdentityError(ValueError):
    """Base error for stereoisomer identity failures."""


class AmbiguousStereoIdentityError(StereoIdentityError):
    """More than one SDF record matches the requested stereoisomer."""


class StereoIdentityMismatchError(StereoIdentityError):
    """A pinned SDF record no longer matches the requested stereoisomer."""


@dataclass(frozen=True)
class StereoIdentity:
    smiles: str
    inchi_stereo: str

    def cache_token(self, sdf_record_index: int | None = None) -> str:
        index = "" if sdf_record_index is None else str(sdf_record_index)
        return f"{index}|{self.smiles}|{self.inchi_stereo}"


def require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is required for ChEMBL3D stereo identity.") from RDKIT_IMPORT_ERROR


def inchi_stereo_layers(inchi: str) -> str:
    """Return concatenated InChI stereo layers (``/t``, ``/b``, ``/m``, ``/s``)."""
    if not inchi:
        return ""
    layers: list[str] = []
    for part in inchi.split("/")[1:]:
        if part[:1] in INCHI_STEREO_LAYER_PREFIXES:
            layers.append(f"/{part}")
    return "".join(layers)


def _mol_to_inchi(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToInchi(mol) or ""
    except Exception:
        try:
            from rdkit.Chem import inchi as rd_inchi

            return rd_inchi.MolToInchi(mol) or ""
        except Exception:
            return ""


def stereo_identity_from_mol(mol: Chem.Mol, *, from_3d: bool) -> StereoIdentity | None:
    """Canonical heavy isomeric SMILES + InChI stereo from a molecule."""
    require_rdkit()
    if mol is None:
        return None
    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception:
        return None
    if from_3d:
        if work.GetNumConformers() == 0:
            return None
        Chem.AssignStereochemistryFrom3D(work)
    else:
        Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    heavy = Chem.RemoveHs(work)
    smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    return StereoIdentity(smiles=smiles, inchi_stereo=inchi_stereo_layers(_mol_to_inchi(heavy)))


def stereo_identity_from_smiles(smiles: str) -> StereoIdentity | None:
    """Re-canonicalize mapping SMILES the same way 3D identity is compared."""
    require_rdkit()
    text = (smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return stereo_identity_from_mol(mol, from_3d=False)


def identities_match(candidate: StereoIdentity | None, expected: StereoIdentity | None) -> bool:
    if candidate is None or expected is None:
        return False
    return candidate.smiles == expected.smiles and candidate.inchi_stereo == expected.inchi_stereo


def mol_matches_expected_smiles(mol: Chem.Mol, expected_smiles: str, *, from_3d: bool = True) -> bool:
    return identities_match(
        stereo_identity_from_mol(mol, from_3d=from_3d),
        stereo_identity_from_smiles(expected_smiles),
    )


def parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if text in {"", "nan", "NaN", "None", "<NA>"}:
        return None
    return int(float(text))


def expected_identity_from_mapping(row: Mapping[str, object]) -> tuple[str, int | None]:
    """Return ``(expected_smiles, sdf_record_index)`` from an intersection CSV row."""
    smiles = str(
        row.get("chembl3d_isomeric_smiles") or row.get("casf_heavy_isomeric_smiles") or ""
    ).strip()
    if not smiles:
        raise ValueError("mapping row is missing chembl3d_isomeric_smiles / casf_heavy_isomeric_smiles")
    return smiles, parse_optional_int(row.get("chembl3d_sdf_record_index"))


def stereo_cache_token(row: Mapping[str, object] | None) -> str:
    """Stable cache key so old ``(group, mol_id)`` pickles cannot be reused."""
    if not row:
        return ""
    smiles, sdf_record_index = expected_identity_from_mapping(row)
    identity = stereo_identity_from_smiles(smiles)
    if identity is None:
        raw = StereoIdentity(smiles=smiles, inchi_stereo="").cache_token(sdf_record_index)
    else:
        raw = identity.cache_token(sdf_record_index)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
