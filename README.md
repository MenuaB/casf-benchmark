# CASF Benchmark

CASF–ChEMBL3D exact-intersection conformer benchmark: data prep → generation → analysis → Streamlit dashboard.

Bundled results under `data/results/` let you run the dashboard without regenerating anything.

## Quick start (dashboard)

```bash
git clone https://github.com/YerevaNN/casf-benchmark.git && cd casf-benchmark
conda env create -f environment-dashboard.yml && conda activate casf-benchmark-dashboard
streamlit run apps/dashboard/streamlit_app.py
```

Uses [`data/results/casf_analysis_dashboard.sqlite`](data/results/casf_analysis_dashboard.sqlite). Override with `CASF_DASHBOARD_DB`.

## Full pipeline (cluster)

```bash
conda env create -f environment-analysis.yml && conda activate casf-benchmark-analysis
export CASF_BENCHMARK_DATA_ROOT=/mnt/weka/mbedrosian
casf-analyze-conformer-sets --help
```

Installs the package editable via `pip install -e ".[dev]"` (see `environment-analysis.yml`). Console scripts such as `casf-analyze-conformer-sets`, `casf-build-dashboard-db`, and `casf-generate-conformer-sets` are available after install. Legacy `python scripts/...` wrappers remain for Slurm compatibility.

## Documentation

| Doc | Topic |
| --- | --- |
| [docs/data_preparation.md](docs/data_preparation.md) | CASF–ChEMBL3D intersection mapping |
| [docs/casf16_chembl3d_data_loader_method.md](docs/casf16_chembl3d_data_loader_method.md) | ChEMBL3D topology/zarr loader (detailed) |
| [docs/stereo_identity_rerun.md](docs/stereo_identity_rerun.md) | Rematch / regen playbook for Flipper identity join |
| [docs/generation_methods.md](docs/generation_methods.md) | RDKit/torsion + external model output contract |
| [docs/generator_models_catalog.md](docs/generator_models_catalog.md) | Per-model papers, training data, benchmarks |
| [docs/analyzer.md](docs/analyzer.md) | Geometric analysis pipeline |
| [docs/materialization.md](docs/materialization.md) | External pool → tiered layout |
| [docs/dashboard.md](docs/dashboard.md) | SQLite build + Streamlit |
| [docs/extras.md](docs/extras.md) | Install, Weka paths, deployment |
| [docs/qwen_checkpoints_and_druglike_report.md](docs/qwen_checkpoints_and_druglike_report.md) | Qwen checkpoint sweep + druglike set: runs, results, code status |
| [docs/casf16_hypothesis.md](docs/casf16_hypothesis.md) | Hypothesis (H0/H1, metrics, predictions) |
| [docs/casf16-core-hypothesis-assessment.md](docs/casf16-core-hypothesis-assessment.md) | Core 94-ligand results vs hypothesis |
| [docs/bioactive_conformer_benchmark_analysis.md](docs/bioactive_conformer_benchmark_analysis.md) | Paper-style analysis notes and figure plan |
| [docs/casf_geometric_report_core_94_ligands.md](docs/casf_geometric_report_core_94_ligands.md) | Snapshot geometric report (core set) |
| [docs/casf_geometric_report_ref_1219_ligands.md](docs/casf_geometric_report_ref_1219_ligands.md) | Snapshot geometric report (ref set) |
| [docs/generation_pipeline_design.md](docs/generation_pipeline_design.md) | Original SMILES conformer generation design notes |

## Common commands

```bash
./scripts/submit_casf.sh generate core          # RDKit/torsion Slurm generation
./scripts/submit_casf.sh analyze core           # geometric analysis
casf-build-dashboard-db
./scripts/push_with_mirror.sh                   # sync personal mirror for Streamlit Cloud
```

Public dashboard mirror: [MenuaB/casf-benchmark](https://github.com/MenuaB/casf-benchmark) (see [docs/extras.md](docs/extras.md#streamlit-cloud)).
