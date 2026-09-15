"""Stereoisomer identity join: matching, topology load, zarr filter, and counts.

The first SDF record for a ChEMBL3D mol_id is often a Flipper sibling of the
stereoisomer selected by isomeric SMILES. These tests put the wrong isomer first
and require every path to recover the requested one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from casf_benchmark.chembl3d.identity import (
    AmbiguousStereoIdentityError,
    StereoIdentityMismatchError,
    expected_identity_from_mapping,
    mol_matches_expected_smiles,
    stereo_cache_token,
    stereo_identity_from_mol,
    stereo_identity_from_smiles,
)
from casf_benchmark.chembl3d.loader import (
    count_matching_chembl3d_conformers,
    load_chembl3d_conformers,
    load_topology_mol,
    load_torsion_ref,
)
from casf_benchmark.cli.match_casf16_chembl3d import (
    CasfRegenSmiles,
    ChemblIndexRow,
    build_exact_match_rows,
    build_topology_index,
    pick_chembl_hit,
    write_csv,
)

GROUP = "001"
MOL_ID = "CHEMBL_STEREO_0"
A_SMILES = "CC[C@H](C)O"
B_SMILES = "CC[C@@H](C)O"


def _canonical(smiles: str) -> str:
    identity = stereo_identity_from_smiles(smiles)
    assert identity is not None
    return identity.smiles


def embed_stereoisomer(smiles: str, seed: int = 7) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    status = AllChem.EmbedMolecule(mol, params)
    assert status == 0, f"ETKDG failed for {smiles}"
    mol.SetProp("mol_id", MOL_ID)
    mol.SetProp("_Name", MOL_ID)
    identity = stereo_identity_from_mol(mol, from_3d=True)
    expected = stereo_identity_from_smiles(smiles)
    assert identity is not None and expected is not None
    assert identity == expected, f"3D identity {identity} != expected {expected}"
    return mol


def write_sdf(path: Path, mols: list[Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def stereo_pair(tmp_path: Path) -> tuple[str, str, Chem.Mol, Chem.Mol, Path]:
    a_smiles = _canonical(A_SMILES)
    b_smiles = _canonical(B_SMILES)
    assert a_smiles != b_smiles
    mol_b = embed_stereoisomer(b_smiles, seed=11)
    mol_a = embed_stereoisomer(a_smiles, seed=13)
    topology_root = tmp_path / "topologies"
    topology_root.mkdir()
    write_sdf(topology_root / f"{GROUP}.sdf", [mol_b, mol_a])
    return a_smiles, b_smiles, mol_a, mol_b, topology_root


def test_identity_helper_distinguishes_enantiomers():
    a = stereo_identity_from_smiles(A_SMILES)
    b = stereo_identity_from_smiles(B_SMILES)
    assert a is not None and b is not None
    assert a.smiles != b.smiles
    assert a.inchi_stereo != b.inchi_stereo
    mol_a = embed_stereoisomer(a.smiles)
    assert mol_matches_expected_smiles(mol_a, A_SMILES)
    assert not mol_matches_expected_smiles(mol_a, B_SMILES)


def test_load_topology_selects_second_record_when_first_is_wrong(tmp_path):
    a_smiles, b_smiles, mol_a, _mol_b, topology_root = stereo_pair(tmp_path)

    loaded = load_topology_mol(GROUP, MOL_ID, topology_root, expected_smiles=a_smiles)
    assert loaded is not None
    assert loaded.GetProp("chembl3d_sdf_record_index") == "1"
    assert mol_matches_expected_smiles(loaded, a_smiles)
    assert not mol_matches_expected_smiles(loaded, b_smiles)

    first_is_b = load_topology_mol(GROUP, MOL_ID, topology_root, expected_smiles=b_smiles)
    assert first_is_b is not None
    assert first_is_b.GetProp("chembl3d_sdf_record_index") == "0"


def test_load_topology_validates_pinned_record_index(tmp_path):
    a_smiles, b_smiles, _mol_a, _mol_b, topology_root = stereo_pair(tmp_path)

    pinned = load_topology_mol(
        GROUP, MOL_ID, topology_root, expected_smiles=a_smiles, sdf_record_index=1
    )
    assert pinned is not None
    assert pinned.GetProp("chembl3d_sdf_record_index") == "1"

    with pytest.raises(StereoIdentityMismatchError, match="Stale"):
        load_topology_mol(
            GROUP, MOL_ID, topology_root, expected_smiles=a_smiles, sdf_record_index=0
        )


def test_load_topology_no_match_returns_none(tmp_path):
    _a_smiles, _b_smiles, _mol_a, _mol_b, topology_root = stereo_pair(tmp_path)
    assert (
        load_topology_mol(GROUP, MOL_ID, topology_root, expected_smiles="C[C@H](O)C(=O)O")
        is None
    )


def test_load_topology_ambiguous_matches_raise(tmp_path):
    a_smiles, _b_smiles, mol_a, _mol_b, _topology_root = stereo_pair(tmp_path)
    topology_root = tmp_path / "dup"
    topology_root.mkdir()
    write_sdf(topology_root / f"{GROUP}.sdf", [mol_a, Chem.Mol(mol_a)])
    with pytest.raises(AmbiguousStereoIdentityError, match="Multiple topology SDF records"):
        load_topology_mol(GROUP, MOL_ID, topology_root, expected_smiles=a_smiles)


def test_first_record_already_requested_still_works(tmp_path):
    a_smiles = _canonical(A_SMILES)
    b_smiles = _canonical(B_SMILES)
    mol_a = embed_stereoisomer(a_smiles, seed=3)
    mol_b = embed_stereoisomer(b_smiles, seed=5)
    topology_root = tmp_path / "topologies"
    topology_root.mkdir()
    write_sdf(topology_root / f"{GROUP}.sdf", [mol_a, mol_b])
    loaded = load_topology_mol(GROUP, MOL_ID, topology_root, expected_smiles=a_smiles)
    assert loaded is not None
    assert loaded.GetProp("chembl3d_sdf_record_index") == "0"


def test_expected_smiles_is_required(tmp_path):
    with pytest.raises(TypeError):
        load_topology_mol(GROUP, MOL_ID, tmp_path)


def test_torsion_ref_uses_requested_isomer(tmp_path):
    a_smiles, _b_smiles, _mol_a, _mol_b, topology_root = stereo_pair(tmp_path)
    ref, source = load_torsion_ref(
        GROUP, MOL_ID, topology_root, expected_smiles=a_smiles
    )
    assert source == "chembl3d_topology_sdf"
    assert ref is not None
    assert mol_matches_expected_smiles(ref, a_smiles)


def test_mol2_fallback_rejected_when_stereo_disagrees(tmp_path):
    a_smiles, b_smiles, _mol_a, mol_b, topology_root = stereo_pair(tmp_path)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    mol2_path = tmp_path / "crystal.mol2"
    Chem.MolToMol2File(mol_b, str(mol2_path))
    ref, source = load_torsion_ref(
        GROUP,
        MOL_ID,
        empty_root,
        mol2_path,
        expected_smiles=a_smiles,
    )
    assert ref is None
    assert source == "unavailable"

    matching_ref, matching_source = load_torsion_ref(
        GROUP,
        MOL_ID,
        empty_root,
        mol2_path,
        expected_smiles=b_smiles,
    )
    assert matching_source == "casf_mol2_fallback"
    assert matching_ref is not None


def _write_fake_zarr(zarr_root: Path, mols: list[Chem.Mol]) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("numpy")
    import numpy as np
    import zarr

    group_path = zarr_root / GROUP
    n_atoms = mols[0].GetNumAtoms()
    mol_id_arr = zarr.open_array(str(group_path / "mol_id"), mode="w", shape=(len(mols),), dtype="S20", chunks=(1,))
    coord_arr = zarr.open_array(
        str(group_path / "coord"),
        mode="w",
        shape=(len(mols), n_atoms, 3),
        dtype="f4",
        chunks=(1, n_atoms, 3),
    )
    numbers_arr = zarr.open_array(
        str(group_path / "numbers"),
        mode="w",
        shape=(len(mols), n_atoms),
        dtype="i4",
        chunks=(1, n_atoms),
    )
    encoded = MOL_ID.encode("utf-8")
    for row, mol in enumerate(mols):
        mol_id_arr[row] = encoded
        numbers_arr[row] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        coords = []
        conf = mol.GetConformer()
        for idx in range(n_atoms):
            pos = conf.GetAtomPosition(idx)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
        coord_arr[row] = np.asarray(coords, dtype="f4")


def test_zarr_filter_keeps_requested_isomer_only(tmp_path):
    pytest.importorskip("zarr")
    a_smiles, _b_smiles, mol_a, mol_b, topology_root = stereo_pair(tmp_path)
    assert mol_a.GetNumAtoms() == mol_b.GetNumAtoms()
    zarr_root = tmp_path / "zarr_database"
    _write_fake_zarr(zarr_root, [mol_b, mol_a])

    loaded = load_chembl3d_conformers(
        GROUP,
        MOL_ID,
        topology_root,
        zarr_root,
        expected_smiles=a_smiles,
        sdf_record_index=1,
    )
    assert len(loaded) == 1
    assert mol_matches_expected_smiles(loaded[0], a_smiles)
    assert count_matching_chembl3d_conformers(
        GROUP,
        MOL_ID,
        topology_root,
        zarr_root,
        expected_smiles=a_smiles,
        sdf_record_index=1,
    ) == 1


def test_matching_selects_second_record_not_first(tmp_path):
    a_smiles, _b_smiles, _mol_a, _mol_b, topology_root = stereo_pair(tmp_path)
    hit = ChemblIndexRow(
        group=GROUP,
        mol_id=MOL_ID,
        isomeric_canonical_smiles=a_smiles,
        conformer_count=99,
    )
    chembl_hits = {a_smiles: [hit]}
    topology_index = build_topology_index(chembl_hits, topology_root)
    assert len(topology_index[(GROUP, MOL_ID)]) == 2

    selected = pick_chembl_hit((hit,), topology_index)
    assert selected is not None
    assert selected.sdf_record_index == 1
    assert selected.wrong_first_record is True

    regen = CasfRegenSmiles(
        ligand_id="lig_a",
        source_file="lig_a.mol2",
        status="ok",
        heavy_isomeric_smiles=a_smiles,
    )
    rows, excluded = build_exact_match_rows(
        {"lig_a": regen},
        chembl_hits,
        topology_index,
        topology_root=topology_root,
        zarr_root=None,
    )
    assert excluded == {"no_smiles_match": [], "missing_topology_sdf": [], "no_rotatable_bonds": []}
    assert len(rows) == 1
    row = rows[0]
    assert row["chembl3d_sdf_record_index"] == 1
    assert row["chembl3d_inchi_stereo"]
    assert row["conformer_count"] == 99
    assert row["chembl3d_index_conformer_count"] == 99
    assert row["wrong_first_topology"] is True

    csv_path = tmp_path / "map.csv"
    write_csv(csv_path, rows)
    text = csv_path.read_text()
    assert "chembl3d_sdf_record_index" in text
    assert "chembl3d_inchi_stereo" in text
    assert "chembl3d_index_conformer_count" in text


def test_matching_filtered_conformer_count_uses_stereo(tmp_path):
    pytest.importorskip("zarr")
    a_smiles, _b_smiles, mol_a, mol_b, topology_root = stereo_pair(tmp_path)
    zarr_root = tmp_path / "zarr_database"
    _write_fake_zarr(zarr_root, [mol_b, mol_a, mol_a])

    hit = ChemblIndexRow(
        group=GROUP,
        mol_id=MOL_ID,
        isomeric_canonical_smiles=a_smiles,
        conformer_count=99,
    )
    chembl_hits = {a_smiles: [hit]}
    topology_index = build_topology_index(chembl_hits, topology_root)
    regen = CasfRegenSmiles(
        ligand_id="lig_a",
        source_file="lig_a.mol2",
        status="ok",
        heavy_isomeric_smiles=a_smiles,
    )
    rows, _excluded = build_exact_match_rows(
        {"lig_a": regen},
        chembl_hits,
        topology_index,
        topology_root=topology_root,
        zarr_root=zarr_root,
    )
    assert rows[0]["conformer_count"] == 2
    assert rows[0]["chembl3d_index_conformer_count"] == 99


def test_mapping_identity_helper_reads_record_index():
    smiles, index = expected_identity_from_mapping(
        {
            "chembl3d_isomeric_smiles": A_SMILES,
            "chembl3d_sdf_record_index": "9428.0",
        }
    )
    assert smiles == A_SMILES
    assert index == 9428
    token = stereo_cache_token(
        {"chembl3d_isomeric_smiles": A_SMILES, "chembl3d_sdf_record_index": 9428}
    )
    other = stereo_cache_token(
        {"chembl3d_isomeric_smiles": A_SMILES, "chembl3d_sdf_record_index": 9422}
    )
    assert token and token != other
