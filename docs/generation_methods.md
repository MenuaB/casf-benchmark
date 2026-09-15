# Generation methods

Conformer generation for the CASF–ChEMBL3D intersection panel: in-repo RDKit/torsion baselines and external learned models.

**Generator:** [`src/casf_benchmark/generation/conformer_sets.py`](../src/casf_benchmark/generation/conformer_sets.py)  
**Slurm:** [`scripts/submit_casf.sh`](../scripts/submit_casf.sh) · [`scripts/run_casf_ref_conformer_molecule.sbatch`](../scripts/run_casf_ref_conformer_molecule.sbatch) · [`scripts/run_casf_ref_conformer_merge.sbatch`](../scripts/run_casf_ref_conformer_merge.sbatch)  
**Catalog:** [`src/casf_benchmark/config/casf_generation_families.yaml`](../src/casf_benchmark/config/casf_generation_families.yaml)

## Inputs

| Input | Default (see [`paths.py`](../src/casf_benchmark/paths.py)) |
| --- | --- |
| Mapping CSV | `data/mapping/casf16_{core,ref}_chembl3d_exact_intersection.csv` |
| CASF MOL2 | `.../core_chembl3d_exact_intersection_ligands/` (or ref) |
| ChEMBL3D topologies | `.../chembl3d/topologies/{group}.sdf` |

Required CSV columns: `ligand_id`, `source_file`, `chembl3d_group`, `chembl3d_mol_id`, `conformer_count`. SMILES: `chembl3d_isomeric_smiles` if present, else `casf_heavy_isomeric_smiles`. After rematch, also `chembl3d_sdf_record_index` (pinned SDF record for the bound stereoisomer). `(group, mol_id)` is not unique — Flipper isomers share a parent id.

CASF MOL2 path: `{ligand_dir}/{source_file}`. Used as PB reference fallback only when topology load fails **and** the crystal 3D stereo matches the mapping SMILES. This benchmark recovers the bound stereoisomer; generators are not prompted for Flipper siblings, and analysis does not take best RMSD across isomers.

## Per-ligand workflow

1. **Mapping validation** — empty `chembl3d_group`/`chembl3d_mol_id` → all 12 methods get `missing_chembl3d_mapping`.
2. **Reference load** — `load_torsion_ref` with `expected_smiles` (and `sdf_record_index` when present). ChEMBL3D topology first; CASF MOL2 fallback only if stereo matches. Failure → `chembl3d_ref_load_failed`. First-id topology load is a bug and is not used.
3. **Embedding template** — copy reference, strip conformers → `base_mol` (RDKit embed template). Rotatable bonds: `Descriptors.NumRotatableBonds`.
4. **Tier targets:**
   - **fixed** = `--fixed_set_size` (default 1000)
   - **dynamic** = `max(1, -20 + 22 × rotatable_bonds)`
   - **chembl_count** = `max(1, conformer_count)` from CSV  
   If dynamic/chembl targets exceed fixed pool size, subsets cap at fixed pool → manifest status `dynamic_target_capped` / `chembl_count_target_capped`.
5. **Torsion gate** — SMARTS `[!$(*#*)&!D1]-!@[!$(*#*)&!D1]`. No torsions: RDKit methods run; torsion methods empty with `no_rotatable_bonds`.
6. **Four families** → fixed pre-PB pool each → tier subsampling → **PoseBusters once** on fixed pool → write tier SDFs + manifest.

## Four in-repo families

Each family → three tiers → **12 methods** per ligand: `{family}_{fixed|dynamic|chembl_count}`.

### RDKit random raw (`rdkit_random_raw`)

- ETKDGv3: `useRandomCoords=True`, enforced chirality, `pruneRmsThresh=-1.0`
- Per-batch seed: deterministic from global `--seed` + ligand id
- Batches of `--generation_batch_size` (default 1000) until fixed pool filled
- Reject non-finite coordinates only (no DG clash filter)

### RDKit random minimized (`rdkit_random_minimized`)

- Same embedding as raw; MMFF94s (`--ff_variant`, default `MMFF94s`) up to `--max_minimize_iters` (500)
- Accept minimization status `{0, 1}` only
- Pool filling in tranches: embed up to `max(remaining, 2500)` per iteration, minimize in parallel (`--minimize_workers`, default = `--num_threads`)

### Torsion raw (`torsion_raw`)

- Copy torsion reference; perturb each rotatable torsion uniformly in `±max_torsion_delta_deg` (default ±120°); `--perturb_fraction` (default 1.0) controls fraction perturbed per trial
- Pre-pool DG steric clash filter at **0.7** via `GetMoleculeBoundsMatrix` (set15 bounds, scaled VdW, triangle smoothing, macrocycle-14 off; bonds and 1,3 pairs excluded; H ignored)
- Parallel torsion batches when `--minimize_workers` > 1

### Torsion minimized (`torsion_minimized`)

Tranche workflow:

1. Torsion perturb → pre-minimize clash filter (0.7)
2. First tranche targets `--torsion_min_pre_pool_size` (default 1500) passers; refills target 500
3. MMFF94s minimize (same acceptance as RDKit minimized)
4. Post-minimize clash filter (0.7)
5. Append to pool until fixed target met

Constants in code: `RDKIT_MIN_PRE_POOL_SIZE = 2500`, `TORSION_MIN_REFILL_POOL_SIZE = 500`.

## Sampling tiers

| Tier | Selection | Sidecar |
| --- | --- | --- |
| **fixed** | Full pre-PB pool | indices `0…N-1` |
| **dynamic** | `min(dynamic_target, pool_size)` without replacement, seeded before PB | `{method}/{mol_id}.indices.tsv` |
| **chembl_count** | `min(conformer_count, pool_size)`, same protocol | same sidecar format |

Failed conformers stay in the sampling frame; PB labels for dynamic/chembl tiers are **projected from fixed-tier PB results** (PB run once).

## Validation and PoseBusters

Filter order differs by family (see manifest funnel fields). PoseBusters is **mandatory** (import failure aborts). Reference for PB = torsion reference (ChEMBL3D or CASF fallback); if reference lacks coords, copy from first predicted conformer.

Active PB checks: file load, RDKit sanitization, InChI, connectivity, no radicals, identity vs reference (formula, bonds, tetrahedral chirality, double-bond stereo), DG geometry (bonds/angles threshold 0.25, clash 0.3), planar aromatic rings and double bonds (0.25 Å), energy ratio (100.0, ensemble 50). Batch failures fall back to divide-and-conquer per conformer.

## Output layout

[`resolve_generation_dir`](../src/casf_benchmark/paths.py): if `--output_dir` already has method subdirs, write there; else create `generation/`.

```
{run_root}/generation/
├── manifest.tsv
├── manifest_parts/{mol_id}.tsv
├── {method}/{mol_id}.sdf
└── {method}/{mol_id}.indices.tsv   # dynamic & chembl_count only
```

Default run roots: `pharma_generation_analysis/{core,ref}_pb_full_dynamic_chembl_count`.

SDF properties include: `mol_id`, `input_smiles`, `source_input`, `generation_method`, `set_tier`, `conf_id`, `num_rotatable_bonds`, `num_target_confs`, `minimization_applied`, minimization metadata. Empty SDFs are valid (zero survivors); manifest still records status.

### Manifest (12 rows/ligand on success)

Key fields: `mol_id`, `input_smiles`, `source_input`, `generation_method`, `set_tier`, `num_target_confs`, `rotatable_bonds`, funnel (`generated_candidates`, `finite_rejected`, `clash_rejected`, `pre_clash_passed`, `generation_batches`), minimization stats, `pb_*` (counts, rates, `pb_check_fail_counts_json`), `kept_confs`, `selected_confs`, `waste_ratio`, `status`, `walltime_seconds`, `sdf_path`.

**Status values:** `ok`, `failed_to_fill_pool`, `embedding_failed`, `all_candidates_invalid`, `empty_dynamic_subset`, `empty_chembl_count_subset`, `dynamic_target_capped`, `chembl_count_target_capped`, `missing_chembl3d_mapping`, `chembl3d_ref_load_failed`, `no_rotatable_bonds`.

## Commands

```bash
# Production Slurm (one ligand per array task + merge job)
./scripts/submit_casf.sh generate core   # or: ref

# Single-ligand test
python -m casf_benchmark.generation.conformer_sets \
  --chembl_map_csv data/mapping/casf16_core_chembl3d_exact_intersection.csv \
  --ligand_dir $CASF_BENCHMARK_DATA_ROOT/data/casf16/CASF16/core_chembl3d_exact_intersection_ligands \
  --chembl3d_topology_root $CASF_BENCHMARK_DATA_ROOT/data/chembl3d/topologies \
  --output_dir $OUTPUT \
  --molecule_offset 0 --limit_molecules 1 --num_threads 8 --minimize_workers 8

# Merge parts only
python -m casf_benchmark.generation.conformer_sets --output_dir $OUTPUT --merge_manifest
```

### CLI parameters

| Flag | Default / role |
| --- | --- |
| `--seed` | 42; combined with ligand key for reproducibility |
| `--fixed_set_size` | 1000 |
| `--generation_batch_size` | 1000 |
| `--torsion_min_pre_pool_size` | 1500 |
| `--max_torsion_delta_deg` | 120 |
| `--perturb_fraction` | 1.0 |
| `--ff_variant` | MMFF94s |
| `--max_minimize_iters` | 500 |
| `--num_threads` | RDKit embed + PB energy-ratio threads |
| `--minimize_workers` | MMFF + torsion batch workers (= `--num_threads`) |
| `--molecule_offset` / `--limit_molecules` | Array scheduling |
| `--merge_manifest` | Merge `manifest_parts/` → `manifest.tsv` |

`--pre_clash_cutoff` / `--post_clash_cutoff` exist but are overridden per family — do not rely on them.

### Slurm behavior

[`submit_casf.sh`](../scripts/submit_casf.sh): validates inputs; counts CSV rows; submits array in chunks ≤1000 (max 80 concurrent); dependent merge job. Each task: `--molecule_offset {base+task_id}`, `--limit_molecules 1`. **Skips** ligand if `manifest_parts/{mol_id}.tsv` already exists (idempotent restart).

Env overrides: `CHEMBL_MAP_CSV`, `CASF_LIGAND_DIR`, `OUTPUT_DIR`, `CHEMBL3D_TOPOLOGY_ROOT`, `PYTHON`, `PARTITION`, `NUM_THREADS`, `MINIMIZE_WORKERS`.

Worker processes set `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` to avoid oversubscription.

## External learned models

Inference runs **outside** this repo. Register in [`src/casf_benchmark/config/casf_generation_families.yaml`](../src/casf_benchmark/config/casf_generation_families.yaml).

### Raw output contract (before materialization)

```
{run_root}/generation/manifest.tsv
{run_root}/generation/{source_method}/{mol_id}.sdf
```

- `{source_method}` **untiered** (e.g. `loqi_raw`, `qwen_4b_bigdata`) — no `_fixed/_dynamic/_chembl_count` suffix
- Multi-record SDF, up to 1000 conformers, same topology as ChEMBL3D reference
- Manifest: tab-separated; one row per `(generation_method, mol_id)`

### Catalog (typical Weka run roots under `codex_dir/`)

| Model | Source method | Core | Ref |
| --- | --- | --- | --- |
| LOQI | `loqi_raw` | `loqi/generations/casf16_core_loqi_1k` | `.../casf16_ref_loqi_1k` |
| NExT-Mol DMT-L | `nextmol_dmt_l_raw` | `nextmol_dmt_l/generations/casf16_core_nextmol_dmt_l_1k` | `.../casf16_ref_nextmol_dmt_l_1k` |
| Torsional Diffusion | `torsional_diffusion_raw` | `torsional_diffusion/generations/casf16_core_torsional_diffusion_1k` | `.../casf16_ref_torsional_diffusion_1k` |
| MCF drugs-L | `mcf_drugs_l_raw` | `mcf_drugs_l/generations/casf16_core_mcf_drugs_l_1k` | `.../casf16_ref_mcf_drugs_l_1k` |
| Qwen | `qwen_*` | `qwen_gens` | `qwen/generations/casf16_ref_qwen_1k` |

After inference → [materialization.md](materialization.md) → [analyzer.md](analyzer.md).

**Per-model papers, training data, and benchmark numbers:** [generator_models_catalog.md](generator_models_catalog.md).

## Design rationale

- ChEMBL3D topology of the **requested stereoisomer** (conformers stripped) as `base_mol` aligns RDKit embedding with the ChEMBL3D graph.
- Single PB pass on fixed pool; subset tiers inherit labels.
- Seeded subsampling keyed by ligand + method for reproducibility.
- Manifest-part architecture enables parallel Slurm with idempotent restart.

Learned-model rematerialization must re-run PoseBusters against isomer A. Re-analyzing current post-PB generation files is not a stereo-identity fix (analysis copies `pb_*` from the manifest). See [stereo_identity_rerun.md](stereo_identity_rerun.md).

## Dependencies

RDKit, PoseBusters, pandas. CPU only. Install with `pip install -e ".[dev]"`.

See also: [data_preparation.md](data_preparation.md) · [materialization.md](materialization.md)
