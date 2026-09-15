# Selective rerun after the stereoisomer identity join fix

The code change stops looking up ChEMBL3D by `(group, mol_id)`. Rematch, regen, rematerialize, and re-analysis still have to run **on Weka** (topologies + zarr + Slurm). This environment cannot mount `/mnt/weka/mbedrosian` or run the cluster generation pipeline.

**Do not** re-analyze current post-PB generation SDFs and call the join fixed. Generation-mode analysis copies `pb_*` from the manifest and does not re-run PoseBusters.

## What is wrong today

Matching already stores `chembl3d_isomeric_smiles` for the bound isomer (A). Loaders then took the **first** SDF record for that `mol_id`, which is often a Flipper sibling (B). PoseBusters identity fails because it rebuilds tetrahedral chirality from 3D and compares InChI stereo layers. RMSD can still look fine (example `CHEMBL273712_0`: requested record 9428 vs first record 9422, heavy-atom RMSD 0.64 Å).

Measured on the bundled maps + Weka index/SDFs before this fix:

- 27 / 1119 mapped `(group, mol_id)` keys have multiple isomeric SMILES
- 21 / 27 load the wrong first SDF record
- 4 / 94 core and 33 / 1236 ref ligand rows get the wrong topology
- 7 further ref ligands sit on a colliding key but the first record happened to be A (still need ensemble recount)

Known core wrong-first ligands: `1nc3`, `3g31`, `3syr`, `4f2w`.

Same parent id, **different** requested stereo (keep a separate ensemble per row):

- `4f2w` vs `4x24` on `CHEMBL188375_0` (`N@@H+` vs `N@H+`)
- `3ocp` / `3u10` vs `5btx` / `5k8s` on `CHEMBL9871_0` (`P@@` vs `P@`)

Same stereo, two crystals: `1nc3` and `2qtt` both want A of `CHEMBL273712_0`.

## 1. Rematch both CSVs on Weka

Needs topologies **and** zarr so `conformer_count` is the filtered row count for that identity, not the index value copied onto every isomer.

```bash
export CASF_BENCHMARK_DATA_ROOT=/mnt/weka/mbedrosian

casf-match-casf16-chembl3d \
  --ligand-dir $CASF_BENCHMARK_DATA_ROOT/data/casf16/CASF16/ligands \
  --topology-root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/topologies \
  --zarr-root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/zarr_database \
  --output-csv $CASF_BENCHMARK_DATA_ROOT/data/casf16/casf16_core_chembl3d_exact_intersection.csv

casf-match-casf16-chembl3d \
  --ligand-dir $CASF_BENCHMARK_DATA_ROOT/data/casf16/CASF16_REF/ligands \
  --topology-root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/topologies \
  --zarr-root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/zarr_database \
  --output-csv $CASF_BENCHMARK_DATA_ROOT/data/casf16/casf16_ref_chembl3d_exact_intersection.csv
```

Copy the rematched CSVs into `data/mapping/` if updating the bundled repo.

Console diagnostics to keep:

- `wrong_first_topology_ligands` / `wrong_first_topology_ligand_ids` — the 37 ligands that loaded B
- `rows_missing_sdf_record_index` must be 0
- `excluded_*` counts

Without zarr, rematch still writes `chembl3d_sdf_record_index` and `chembl3d_inchi_stereo` but `conformer_count` stays the unfiltered index value (`chembl3d_index_conformer_count`).

## 2. Bust caches

Delete `analysis/cache/` (including `geometric_reference_parts`, `geometric_generation_parts`) and `chembl3d_mols` under the affected run roots before re-analysis. New pickle parts embed a stereo identity token so old `(group, mol_id)` pickles cannot survive `--resume-parts`.

## 3. Selective work (only colliding keys)

Other ~90 core / ~1200 ref ligands are unchanged.

| Path | Those 37 wrong-first ligands (4 core + 33 ref) |
| --- | --- |
| RDKit / torsion | **Generate again** (seeded from B today), then analyze |
| Qwen / LOQI / DMT / MCF / torsional diffusion | **Do not infer again** if raw `{method}/{mol_id}.sdf` still exists. Rematerialize (PB vs A, retier), then analyze. Infer only if raw pools are gone |
| `chembl3d_sdf` / `chembl3d_gt` / `chembl3d_gt_pb` | **Re-analyze only** after the loader filter |

The extra **7** ref ligands whose first SDF record happened to be A: reference re-analysis (and `chembl_count` retier if the filtered count drops). Their RDKit/Qwen coordinates can stay.

Limit generation/materialization with `--molecule_offset` / `--limit_molecules` or a ligand-id allowlist built from `wrong_first_topology_ligand_ids`.

Example after rematch (core RDKit/torsion, one ligand):

```bash
casf-generate-conformer-sets \
  --chembl_map_csv $CASF_BENCHMARK_DATA_ROOT/data/casf16/casf16_core_chembl3d_exact_intersection.csv \
  --ligand_dir $CASF_BENCHMARK_DATA_ROOT/data/casf16/CASF16/core_chembl3d_exact_intersection_ligands \
  --chembl3d_topology_root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/topologies \
  --output_dir $OUTPUT \
  --molecule_offset N \
  --limit_molecules 1
```

Rematerialize learned models from existing raw SDFs (`casf-materialize-generation-sets`) so PoseBusters runs against topology A.

Then reference-mode analysis (`casf-analyze-conformer-sets --mode reference`) for the 37 + 7, and generation-mode analysis only for ligands that were regenerated or rematerialized.

## 4. What this playbook is not

- A full-panel regen
- A rename of SDF records / SMILES tags
- Best-RMSD-across-isomers scoring
- Prompting generators for every Flipper isomer
