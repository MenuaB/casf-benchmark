from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from casf_benchmark.generation import normalizer


def multi_conf_ethane(num_confs: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    mol.RemoveAllConformers()
    for conf_id in range(num_confs):
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(conf_id)
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, (float(atom_idx), float(conf_id), 0.0))
        mol.AddConformer(conf, assignId=True)
    return mol


def test_discover_fixed_source_pools_skips_existing_sampled_tiers():
    manifest = pd.DataFrame(
        [
            {"generation_method": "qwen_4b_revisited", "set_tier": "qwen"},
            {"generation_method": "qwen_4b_revisited_fixed", "set_tier": "fixed"},
            {"generation_method": "loqi_raw_dynamic", "set_tier": "dynamic"},
        ]
    )

    pools = normalizer.discover_fixed_source_pools(manifest)

    assert [pool.method for pool in pools] == ["qwen_4b_revisited"]
    assert pools[0].output_stem == "qwen_4b_revisited"


def test_materialize_source_pool_samples_from_full_pool_before_pb(monkeypatch, tmp_path):
    captured = {}
    input_mol = normalizer.InputMolecule(
        mol_id="lig_a",
        smiles="CC",
        source_input=str(tmp_path / "lig_a.mol2"),
        chembl3d_group="001",
        chembl3d_mol_id="CHEMBL1",
        chembl3d_conformer_count=3,
    )
    pool = normalizer.SourcePool(
        method="qwen_4b_revisited",
        output_stem="qwen_4b_revisited",
        row={"rotatable_bonds": 2, "num_target_confs": 1000},
    )

    monkeypatch.setattr(normalizer, "load_multi_record_sdf", lambda _path: multi_conf_ethane(10))
    monkeypatch.setattr(normalizer, "load_torsion_ref", lambda *_args, **_kwargs: (multi_conf_ethane(1), "fake"))

    def fake_finalize_pipeline_pair(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(normalizer, "finalize_pipeline_pair", fake_finalize_pipeline_pair)

    rows = normalizer.materialize_source_pool(
        input_mol=input_mol,
        pool=pool,
        generation_dir=tmp_path / "generation",
        topology_root=tmp_path / "topologies",
        fixed_set_size=1000,
        seed=1729,
        posebusters_workers=2,
        posebusters_energy_threads=1,
    )

    assert rows == []
    assert captured["fixed_method"] == "qwen_4b_revisited_fixed"
    assert captured["dynamic_method"] == "qwen_4b_revisited_dynamic"
    assert captured["chembl_count_method"] == "qwen_4b_revisited_chembl_count"
    assert captured["fixed_pre_pb"].GetNumConformers() == 10
    assert len(captured["dynamic_indices"]) == 10
    assert len(captured["chembl_count_indices"]) == 3
    assert captured["stats"].generated_candidates == 10
    assert captured["pool_status_value"] == "failed_to_fill_pool"


def test_validate_sampled_generation_root_accepts_empty_sdf_with_sidecar(tmp_path):
    generation_dir = tmp_path / "generation"
    method_dir = generation_dir / "qwen_4b_revisited_dynamic"
    method_dir.mkdir(parents=True)
    sdf_path = method_dir / "lig_a.sdf"
    sdf_path.write_text("")
    sdf_path.with_suffix(".indices.tsv").write_text(
        "mol_id\tlig_a\nfamily\tqwen_4b_revisited\ntier\tdynamic\nfixed_pool_index\n0\n"
    )
    manifest = pd.DataFrame(
        [
            {
                "mol_id": "lig_a",
                "generation_method": "qwen_4b_revisited_dynamic",
                "set_tier": "dynamic",
                "pb_input_confs": 1,
                "kept_confs": 0,
            }
        ]
    )
    manifest_path = generation_dir / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    assert normalizer.validate_sampled_generation_root(generation_dir, manifest_path) == []
