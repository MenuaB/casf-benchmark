# CASF ref crystal-selection sensitivity

The dashboard uses one frozen selection, `casf-ref-dedup-v1` (seed `casf-ref-dedup-v1-20261030`). This audit draws 1000 alternative selections with the same hash procedure and different seeds. It does not change the dashboard row.

Metrics are the ChEMBL-count paper definitions on each selection's 1,044 ligands. Hit@0.5 and Hit@0.75 count a missing RMSD as a miss. Mean Best RMSD averages defined RMSDs only.

Generator rank agreement with the frozen selection: mean Kendall tau 0.9681; identical order in 1.5% of selections; same top generator in 100.0% of selections.

Highest frozen Hit@0.75 among generators: Qwen 1.7B FSQ bigdata (step 47023) (75.48%; alternative mean 75.30%, 2.5–97.5% interval 74.90–75.67%).

Files: `primary_chembl_count_metrics.csv`, `sensitivity_summary.csv`, `alternative_chembl_count_metrics.csv`, `rank_stability.csv`.
