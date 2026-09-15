# ChEMBL3D Data Loader

## Purpose

This document describes how the CASF benchmark reads ChEMBL3D structures from on-disk storage. Implementation lives in `src/casf_benchmark/chembl3d/loader.py` and `src/casf_benchmark/chembl3d/identity.py`. The loader is used at three distinct stages of the pipeline:

1. **Intersection mapping** — verify that a ChEMBL3D topology SDF **stereoisomer** exists for each CASF ligand candidate (`scripts/match_casf16_chembl3d_exact.py`).
2. **Conformer generation** — load a torsion-seed reference molecule for RDKit embedding and torsion perturbation (`load_torsion_ref`).
3. **Geometric analysis** — load topology SDF entries and full zarr conformer ensembles for reference baselines (`chembl3d_sdf`, `chembl3d_gt`, `chembl3d_gt_pb`).

Dataset provenance and how ChEMBL3D was assembled are in [generator_models_catalog.md](generator_models_catalog.md#chembl3d). Intersection mapping that produces the identity keys consumed here is in [data_preparation.md](data_preparation.md).

**Invariant:** requested stereo == selected SDF record == every loaded/counted zarr row == PoseBusters reference. This is pose recovery of the bound stereoisomer, not enumeration of Flipper siblings.

---

## On-disk layout

ChEMBL3D is stored as a sharded topology SDF tree plus a parallel zarr coordinate archive. Both are keyed by a three-digit **group** shard (000–999) and a string **mol_id** within that shard. **`mol_id` is not unique across stereoisomers.** OpenEye Flipper writes several SDF records (and zarr rows) under the same parent id. Looking up the first `mol_id` / `_Name` hit is a bug: that record is often a different stereoisomer than the mapping SMILES.

| Path | Format | Contents |
| --- | --- | --- |
| `{topology_root}/{group}.sdf` | Multi-record SDF | One **or more** topology molecules per `mol_id` (Flipper isomers). Each record includes at least one 3D conformer used as the torsion-seed template |
| `{zarr_root}/{group}/mol_id` | zarr array (bytes) | Encoded molecule identifiers; rows index into coordinate arrays. Sibling stereoisomers share the id |
| `{zarr_root}/{group}/coord` | zarr array (float32, N×A×3) | Per-conformer atom coordinates aligned to the **selected** topology SDF atom order |
| `{zarr_root}/{group}/numbers` | zarr array (int, N×A) | Per-conformer atomic numbers; validated against topology on load. Atomic numbers cannot tell ribose from xylose |
| `{index_csv}` | CSV | SMILES lookup table: `group`, `mol_id`, `isomeric_canonical_smiles`, `conformer_count`, … |

Default mount paths (override with `CASF_BENCHMARK_DATA_ROOT` or CLI flags):

```
/mnt/weka/mbedrosian/data/chembl3d/topologies/{000..999}.sdf
/mnt/weka/mbedrosian/data/chembl3d/zarr_database/{000..999}/mol_id|coord|numbers
/mnt/weka/mbedrosian/data/chembl3d_index/chembl3d_topology_smiles_index.csv
```

The intersection mapping CSV (`casf16_*_exact_intersection.csv`) stores the identity of the bound isomer: `(chembl3d_group, chembl3d_mol_id, chembl3d_isomeric_smiles, chembl3d_sdf_record_index, chembl3d_inchi_stereo, conformer_count)`. Downstream loaders require `expected_smiles` (the mapping SMILES) and optionally `sdf_record_index`. SDF property tags are not identity.

---

## Dependencies and environment

The loader requires:

- **RDKit** — SDF parsing, conformer construction, mol2 fallback
- **NumPy** — coordinate array handling
- **zarr** — reading the conformer archive

Call `require_dependencies()` before zarr access; it raises a clear error if NumPy or zarr is missing. The project documents the `chembl3d` conda environment as a known working configuration:

```bash
/home/mbedrosian/.conda/envs/chembl3d/bin/python
```

No GPU is required. All I/O is local filesystem reads.

---

## Stereo identity

From a mol with 3D coordinates: `AssignStereochemistryFrom3D`, then canonical heavy isomeric SMILES **and** InChI stereo layers `/t /b /m /s` (the same check PoseBusters uses for identity). A candidate matches the mapping SMILES only if **both** agree after the expected SMILES is re-canonicalized the same way.

Do not: match on non-isomeric SMILES; pick the isomer with lowest RMSD to the crystal; treat first-record load as a fallback; rewrite SDF names or SMILES tags (PoseBusters rebuilds tetrahedral chirality from 3D).

## API reference (step-by-step behavior)

### `load_topology_mol(group, mol_id, topology_root, *, expected_smiles, sdf_record_index=None)`

`expected_smiles` is required. Omitting it is an error: `(group, mol_id)` is not a stereoisomer key.

**Step 1.** Resolve the SDF path as `{topology_root}/{int(group):03d}.sdf`. Return `None` if the file does not exist.

**Step 2.** If `sdf_record_index` is set, load **that** record and **validate** it still matches `mol_id` and the expected 3D stereo. A stale CSV index raises `StereoIdentityMismatchError`.

**Step 3.** Otherwise iterate **every** record in the SDF supplier (`removeHs=False`, `sanitize=False`). Keep records whose `mol_id` property or `_Name` equals the requested id **and** whose reconstructed 3D stereo matches `expected_smiles`.

**Step 4.** Sanitize each kept molecule. Raise `ValueError` on sanitization failure (topology corruption).

**Step 5.** Zero stereo matches → `None`. One match → deep copy of that `Chem.Mol`. More than one remaining match → `AmbiguousStereoIdentityError`. Never return the first record as a default.

This function returns a single topology entry. It does **not** read the zarr archive.

### `prepare_torsion_ref_mol(mol)`

**Step 1.** Return `None` if the input is `None` or has zero conformers.

**Step 2.** Deep-copy the molecule.

**Step 3.** If no explicit hydrogens are present, call `Chem.AddHs(prepared, addCoords=True)`.

**Step 4.** Return the prepared molecule.

Explicit hydrogens are required for MMFF minimization and torsion SMARTS detection in the generation pipeline.

### `load_torsion_ref_from_chembl3d(group, mol_id, topology_root, *, expected_smiles, sdf_record_index=None)`

Calls `load_topology_mol` then `prepare_torsion_ref_mol`. This is the preferred torsion-seed source for generation.

### `load_torsion_ref_from_mol2(mol2_path)`

Fallback when no ChEMBL3D topology entry resolves:

**Step 1.** Parse MOL2 with `sanitize=True`; retry with `sanitize=False` + manual sanitize on failure.

**Step 2.** Require at least one conformer.

**Step 3.** Run through `prepare_torsion_ref_mol`.

Used when a mapped ligand's ChEMBL3D SDF entry is missing but the CASF crystal MOL2 has usable 3D coordinates.

### `load_torsion_ref(group, mol_id, topology_root, mol2_path=None, *, expected_smiles, sdf_record_index=None)`

**Step 1.** Try `load_torsion_ref_from_chembl3d` with the expected stereo.

**Step 2.** If that returns `None` and `mol2_path` is provided, try `load_torsion_ref_from_mol2`. Keep the MOL2 **only if** its 3D stereo matches `expected_smiles`. A crystal that disagrees with the mapping SMILES is rejected, not used as a silent fallback.

**Step 3.** Return `(mol, source_tag)` where `source_tag` is one of:
- `"chembl3d_topology_sdf"` — ChEMBL3D topology used
- `"casf_mol2_fallback"` — CASF MOL2 used
- `"unavailable"` — no reference loaded

The generation script (`conformer_sets.py`) treats `"unavailable"` as a hard failure for all twelve method outputs on that ligand.

### `find_mol_id_indices(mol_id_array, mol_id)`

**Step 1.** Open the zarr `mol_id` array for the group (read-only).

**Step 2.** Encode the requested `mol_id` as UTF-8 bytes.

**Step 3.** Return all row indices where the stored bytes match exactly (handles duplicate rows if present).

### `load_chembl3d_conformers(group, mol_id, topology_root, zarr_root, *, expected_smiles, sdf_record_index=None, limit=None, row_indices=None)`

This is the main ensemble loader used in reference-mode analysis.

**Step 1. Load topology template.**

Call `load_topology_mol` with `expected_smiles` / `sdf_record_index`. Raise `FileNotFoundError` if the topology SDF stereoisomer is missing — the zarr coordinates cannot be interpreted without a matching atom graph.

**Step 2. Open zarr arrays.**

For group `{group:03d}`, open:
- `mol_id` — byte-encoded identifiers
- `coord` — shape `(N, n_atoms, 3)`
- `numbers` — shape `(N, n_atoms)` atomic numbers

Raise `FileNotFoundError` listing any missing array paths.

**Step 3. Resolve row indices.**

If `row_indices` is not provided:
- Call `find_mol_id_indices` on the `mol_id` array (all sibling isomers share this id).
- Return an empty list if no rows match.

If `row_indices` is provided explicitly, use those rows directly.

**Step 4. Validate, reconstruct stereo, and materialize conformers.**

Paste each zarr row onto the **selected** topology. For each row index:
1. Read the `numbers` row and compare to the topology's atomic-number sequence. Raise `ValueError` on mismatch (data integrity guard — corruption, not a stereoisomer skip).
2. Read the `coord` row (length must equal topology atom count).
3. Clone the topology template, remove all conformers, attach a new conformer with the zarr coordinates.
4. Reconstruct stereo from those coords. **Skip** rows that do not match `expected_smiles` (sibling Flipper isomers). Atomic numbers cannot tell those isomers apart.
5. Set properties: `_Name`, `chembl3d_group`, `chembl3d_mol_id`, `chembl3d_isomeric_smiles`, `chembl3d_sdf_record_index`.

Apply `limit` **after** the stereo filter (first N matching rows).

**Step 5.** Return the list of RDKit molecules (one per matching zarr row).

Each returned molecule shares the same bond topology and atom ordering as the selected SDF template; only coordinates differ, and every row is the requested stereoisomer.

---

## Usage in the benchmark pipeline

### During intersection mapping

`match_casf16_chembl3d_exact.py` indexes **all** SDF records for each hit `mol_id` and confirms 3D identity equals the index SMILES before checking rotatable bonds. Ligands with zero rotatable torsions are excluded with reason `no_rotatable_bonds`. Console output includes `wrong_first_topology_ligands` for rows whose selected record is not the first `mol_id` hit.

### During RDKit/torsion generation

For each intersection CSV row:

```
load_torsion_ref(group, mol_id, topology_root, casf_mol2_path,
                 expected_smiles=chembl3d_isomeric_smiles,
                 sdf_record_index=chembl3d_sdf_record_index)
    → torsion reference (requested stereoisomer)
    → base_mol = copy with conformers removed (ETKDG embedding template)
    → PoseBusters reference molecule
```

The selected ChEMBL3D topology defines the molecular graph for embedding. CASF MOL2 coordinates are **not** used as the embedding template unless ChEMBL3D loading fails entirely **and** the crystal stereo matches the mapping SMILES.

### During external pool materialization

`materialize_casf_generation_sets.py` uses the same `load_torsion_ref` call to obtain the PoseBusters reference for tier validation of LOQI, DMT, MCF, Torsional Diffusion, and Qwen outputs. Learned models are prompted with the mapping SMILES (isomer A). Rematerialize against A; do not re-infer if the raw `{method}/{mol_id}.sdf` still exists.

### During reference-mode analysis

For each mapped ligand, `analyze_reference_ligand` in `scripts/analyze_casf_conformer_sets.py`:

| Reference source | Loader call | Result |
| --- | --- | --- |
| `chembl3d_sdf` | `load_topology_mol` with expected SMILES / record index | Selected topology SDF stereoisomer |
| `chembl3d_gt` | `load_chembl3d_conformers` (stereo-filtered ensemble) | Zarr rows whose reconstructed stereo matches the request |
| `chembl3d_gt_pb` | Same load, then PoseBusters filter | Subset passing validity vs CASF crystal |

**Special case:** ligand `1tlp_1tlp_conf0` is deterministically subsampled to 2000 conformers before PoseBusters and metrics (seed 42). The stereo filter runs **before** that cap. Sampling mixed isomers first can drop almost all of the requested isomer (`CHEMBL41289_0` has 128 stereoisomers).

Analysis pickle parts under `analysis/cache/` store a stereo identity token (record index + SMILES/InChI hash). Old `(group, mol_id)` pickles are not reused. Bust `analysis/cache/` / `chembl3d_mols` when rematching.

Generation-mode analysis of current post-PB SDFs is **not** a fix: it copies `pb_*` from the manifest and does not re-run PoseBusters.

---

## Data flow diagram

```
chembl3d_topology_smiles_index.csv
        │  (isomeric SMILES lookup during mapping)
        ▼
intersection CSV  ──►  (group, mol_id, isomeric SMILES, sdf_record_index, filtered conformer_count)
        │
        ├─► topologies/{group}.sdf  ──► load_topology_mol / load_torsion_ref
        │                                      │  (scan all records; fail closed)
        │                                      ▼
        │                              generation seed + PB reference
        │
        └─► zarr_database/{group}/     ──► load_chembl3d_conformers
                 mol_id / coord / numbers          │  (filter rows by reconstructed stereo)
                                                   ▼
                                         chembl3d_gt / chembl3d_gt_pb
                                         (reference-mode analysis)
```

---

## Error handling and validation

| Condition | Behavior |
| --- | --- |
| Missing SDF shard | `load_topology_mol` returns `None` |
| Zero stereo matches for `mol_id` | `load_topology_mol` returns `None` |
| Multiple stereo matches | `AmbiguousStereoIdentityError` |
| Stale `sdf_record_index` | `StereoIdentityMismatchError` |
| Missing zarr group directory | `load_chembl3d_conformers` raises `FileNotFoundError` |
| Atomic number mismatch | `ValueError` with row index and observed vs expected numbers |
| Coordinate length ≠ atom count | `ValueError` with lengths |
| Zarr row is a sibling stereoisomer | Skipped (not counted) |
| No matching zarr rows | Returns empty list (reference analysis raises `RuntimeError`) |
| MOL2 stereo disagrees with request | Fallback rejected (`unavailable`) |
| RDKit not installed | `RuntimeError` on any loader call requiring RDKit |

The loader does not cache open zarr arrays or SDF suppliers across calls. Each invocation opens files fresh. For large batch jobs (reference analysis on 1200+ ligands), this is acceptable because per-ligand work dominates I/O.

---

## Relationship to ChEMBL3D dataset construction

ChEMBL3D was built by Nikitin et al. (ChemRxiv 2025, [doi:10.26434/chemrxiv-2025-k4h7v](https://doi.org/10.26434/chemrxiv-2025-k4h7v)) from ChEMBL v34 drug-like structures:

1. Protomer enumeration (OpenEye FixpKa)
2. Initial 3D sampling (OpenEye Omega Classic)
3. Stereoisomer expansion (OpenEye Flipper)
4. AIMNet2 geometry optimization under implicit solvation
5. Filtering of broken topologies and high-energy outliers

The public release contains ~1.8M unique molecules and ~250M optimized conformers. The on-disk layout in this benchmark (SDF shards + zarr coordinates + SMILES index) is a project-specific materialization of that release for fast random access during Slurm array jobs.

The SMILES index CSV is a precomputed artifact on the shared mount. There is no index-build script in this repository; regenerating it requires the upstream ChEMBL3D release workflow.

---

## Preconditions checklist

Before calling the loader in production:

1. ChEMBL3D topology SDF tree mounted at `topology_root`.
2. Zarr archive mounted at `zarr_root` with matching group shards.
3. Intersection CSV with valid `(chembl3d_group, chembl3d_mol_id, chembl3d_isomeric_smiles)` and, after rematch, `chembl3d_sdf_record_index` for each ligand.
4. Python environment with RDKit, NumPy, and zarr.
5. For generation: CASF MOL2 paths available as fallback when topology SDF entries are missing.

---

## Related documentation

- [data_preparation.md](data_preparation.md) — produces the mapping CSV and validates topology stereo identity
- [generation_methods.md](generation_methods.md) — uses `load_torsion_ref` during generation
- [stereo_identity_rerun.md](stereo_identity_rerun.md) — selective rematch / regen / rematerialize after this join fix
- [analyzer.md](analyzer.md) — uses ensemble loading in reference mode
- [generator_models_catalog.md](generator_models_catalog.md) — ChEMBL3D dataset provenance and LOQI training context
- [extras.md](extras.md#weka-paths) — production mount paths
