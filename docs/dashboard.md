# Dashboard

Read-only Streamlit UI over precomputed analysis SQLite.

**App:** [`apps/dashboard/streamlit_app.py`](../apps/dashboard/streamlit_app.py)  
**Build scripts:** [`scripts/build_casf_analysis_master_csv.py`](../scripts/build_casf_analysis_master_csv.py) · [`scripts/build_casf_analysis_dashboard_db.py`](../scripts/build_casf_analysis_dashboard_db.py)  
**Config:** [`src/casf_benchmark/config/casf_analysis_sources.yaml`](../src/casf_benchmark/config/casf_analysis_sources.yaml) (repo-relative) · [`src/casf_benchmark/config/casf_analysis_sources.weka.yaml`](../src/casf_benchmark/config/casf_analysis_sources.weka.yaml) (cluster)

## Data flow

```
data/results/runs/{run_id}/analysis/tables/geometric_per_ligand_long.csv
        →  build_casf_analysis_master_csv.py
        →  build_casf_analysis_dashboard_db.py
        →  data/results/casf_analysis_dashboard.sqlite
        →  streamlit run apps/dashboard/streamlit_app.py
```

Bundled DB for clone-and-run: [`data/results/casf_analysis_dashboard.sqlite`](../data/results/casf_analysis_dashboard.sqlite).

Override: `CASF_DASHBOARD_DB`, `CASF_EXTENDED_DB` (extended tab — [`data/results/extended_casf_analysis.sqlite`](../data/results/extended_casf_analysis.sqlite)).

## SQLite tables

| Table | Contents |
| --- | --- |
| `comparison_rows` | Cohort aggregates by family/tier/method |
| `comparison_strata` | Same, split by rotatable-bond / heavy-atom bins |
| `per_ligand_long` | Full per-ligand metrics |
| `analysis_sources` | Run roots from YAML |
| `catalog_families` | From [`casf_generation_families.yaml`](../src/casf_benchmark/config/casf_generation_families.yaml) |

## UI

Sidebar: ligand set, tier, family, stratum breakdown. Tabs: Overview, Clustering, Energy, CASF hits, Funnel. Extended section loads optional `extended_*` tables.

Per-table **Columns & rows** expander: hide columns; filter by method, stratum, or `mol_id` substring. `method` column stays pinned on horizontal scroll.

## Extended analysis tables

`casf-extended-analysis` appends additive tables to the sidecar extended DB and to
`casf_analysis_dashboard_extended.sqlite`. The dashboard shows all available
extended tables as tabs.

| Table | Contents |
| --- | --- |
| `extended_k_efficiency` | Generation K curves plus ChEMBL3D-PB `first_k` and `random_k` rows; ChEMBL rows include `summary_mode` (`capped_at_available`, `strict_at_least_k`), `n_metric_ligands`, `n_ligands_with_pb`, `mean_n_used`, and 95% intervals for random Hit@0.75. |
| `extended_chembl_k_efficiency_per_ligand` | ChEMBL3D-PB per-ligand K results with requested `k`, source-order `n_available_pb`, `n_used`, best CASF RMSD, and hits at 0.25/0.5/0.75/2.0 A. |
| `extended_k_efficiency_comparisons` | Matched-K Qwen-vs-ChEMBL paired deltas and ChEMBL internal saturation rows. |
| `extended_energy_window_per_ligand` | Per-method, per-ligand PB-passing conformer metrics inside `all`, `deltaE_5`, `deltaE_10`, `deltaE_20`, and `deltaE_50` windows. Energies are per-ligand relative MMFF94s values. |
| `extended_energy_window_summary` | Full-universe aggregate hit, RMSD, retained-fraction, and 1 A cluster metrics for each energy window. |
| `extended_energy_window_comparisons` | Window-to-window hit deltas and high-energy-hit fractions. |

Large repeated ChEMBL per-ligand rows may be stored compactly behind a SQLite
view with the same public table name, so dashboard queries and exports remain
unchanged.

Energy-window hit rates use the full ligand universe for the method row. Ligands
with zero conformers inside a window count as misses and are also visible through
`n_ligands_with_window_confs`.

## Commands

```bash
# View (minimal env — see extras.md)
streamlit run apps/dashboard/streamlit_app.py

# Rebuild DB from bundled per-run CSVs (after pip install -e ".[dev]")
casf-build-master-csv
casf-build-dashboard-db

# Extended sidecar DB with ChEMBL K and energy-window tables
casf-extended-analysis

# Full cluster rebuild (all sources + reference runs)
sbatch scripts/run_pb_once_casf_analysis.sbatch
```

Adding a generator: entry in `casf_generation_families.yaml` + analyzed run root in `casf_analysis_sources.yaml` → rebuild scripts above.

Public hosting: [extras.md](extras.md#streamlit-cloud).

See also: [analyzer.md](analyzer.md) · [extras.md](extras.md)

### Comparable paper energies and sampling curves

For CASF core and ref, **Energy → Common protocol (paper)** reads
`paper_energy_summary` and `paper_energy_per_ligand`. These cover the paper's ten
generators at the ChEMBL-count target plus stored ChEMBL3D-PB. Energies use MMFF94s
with explicit H removed and rebuilt, without coordinate optimization. The energy
spread plotted in the paper is the median across molecules of the population SD
within each retained ensemble. Counts and failures are provided alongside it.

**Sampling curves** reads `paper_sampling_curves`, `paper_sampling_frontier`, and
`paper_sampling_reference`. These are exact hypergeometric subset probabilities
for the saved pre-PoseBusters pools, capped at available candidates. Rejections
and failed RMSDs remain non-hits, and every dataset entry has equal weight.
The 80% threshold is inverted on observed RMSDs. Confidence intervals use 2,000
shared molecule bootstrap resamples with seed 20261006.

These complete-dataset paper views have their own method selection; sidebar
tier/family/stratum filters apply to the archived benchmark views. Older energies
remain explicitly labeled **Archived energies**, including energy charts in the
per-ligand report. They must not be mixed with the rebuilt-H energies.

`paper_protocol_runs` records dataset size, protocols, provenance, and source
hashes. The database builder adds these tables to a snapshot without replacing
any archived rows. Reproduction scripts are in `docs/ref_protocol_analysis/`:

```bash
source .venv/bin/activate
# Core used RDKit 2025.09.6. Zarr 2.18.3 and numcodecs<0.16 are needed to
# reconstruct the stored reference coordinates on Weka.
python docs/ref_protocol_analysis/calculate.py --database INPUT.sqlite --output RESULTS --stage energy --workers 8
python docs/ref_protocol_analysis/calculate.py --database INPUT.sqlite --output RESULTS --stage reference --workers 8
python docs/ref_protocol_analysis/calculate.py --database INPUT.sqlite --output RESULTS --stage sampling --workers 8
python docs/ref_protocol_analysis/calculate.py --database INPUT.sqlite --output RESULTS --stage aggregate
python docs/ref_protocol_analysis/build_database.py --database INPUT.sqlite --output OUTPUT.sqlite --core-energy CORE_ENERGY --core-sampling CORE_SAMPLING --ref-results RESULTS
```

The ref reference reconstruction verifies original energies pointwise and uses
unique archived energy identities to recover the exact ordered PB-retained
subset. Ambiguous identities stop the calculation. Re-running PB is insufficient:
its stochastic energy check can change subset membership while preserving count.
