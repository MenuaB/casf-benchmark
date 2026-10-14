"""Per-table help copy for the CASF Analysis Dashboard."""

from __future__ import annotations

import streamlit as st

# Keys match the `name` passed to display_table() in streamlit_app.py.
TABLE_HELP: dict[str, dict[str, str | dict[str, str]]] = {
    "overview": {
        "summary": (
            "Ensemble size and geometric diversity for each method at the selected "
            "ligand set, tier, and optional stratum. Cohort values are means of "
            "per-ligand stats (not conformer-pooled). Pairwise RMSD uses up to 120 "
            "sampled confs; torsion std averages rotatable-dihedral stds across confs."
        ),
        "columns": {
            "stratum": "Bin label when broken down by rotors / heavy atoms.",
            "row_type": "`generation` (method) or `reference` (baseline).",
            "display_label": "Human-readable method / reference name.",
            "family": "Generator family id or reference source name.",
            "tier": "`fixed` / `dynamic` / `chembl_count` / `reference`.",
            "method": "Full method key (includes tier) or reference source.",
            "ligands": "Unique mol_id count in the aggregate.",
            "ligands_scope": "`{n_ligands}/{mean_confs}` eligibility summary.",
            "total_confs": "Sum of per-ligand conformer counts.",
            "mean_confs_per_ligand": "Mean ensemble size per ligand.",
            "pairwise_mean": "Mean of per-ligand mean pairwise heavy-atom RMSD (Å).",
            "pairwise_p90": "Mean of per-ligand 90th-percentile pairwise RMSD (Å).",
            "mean_torsion_std_deg": "Mean rotatable-torsion std (°) — higher ⇒ more torsional diversity.",
        },
    },
    "clustering": {
        "summary": (
            "Greedy RMSD clustering of each ligand's ensemble — primary diversity "
            "axis of the benchmark (basin coverage). Centers assigned in conf order "
            "at 0.5 / 1.0 / 2.0 / 3.0 Å; cohort = mean of per-ligand cluster stats. "
            "`clusters_per_100_1p0` is the preferred size-normalized compare."
        ),
        "columns": {
            "stratum / row_type / display_label / family / tier / method / ligands / ligands_scope": (
                "Same identity fields as Overview."
            ),
            "mean_clusters_0p5": "Mean greedy cluster count @ 0.5 Å.",
            "mean_clusters_1p0": "Mean greedy cluster count @ 1.0 Å (main diversity summary).",
            "mean_clusters_2p0": "Mean greedy cluster count @ 2.0 Å.",
            "mean_clusters_3p0": "Mean greedy cluster count @ 3.0 Å.",
            "clusters_per_100_1p0": "Mean of `100 × clusters / n_confs` @ 1.0 Å.",
            "cluster_entropy_1p0": "Mean normalized Shannon entropy of 1.0 Å sizes (1 ≈ even).",
            "effective_clusters_1p0": "Mean of `exp(raw entropy)` — effective populated clusters.",
            "largest_cluster_fraction_1p0": "Mean fraction in the largest 1.0 Å cluster (high ⇒ collapse).",
            "singleton_fraction_1p0": "Mean fraction of 1.0 Å clusters of size 1.",
        },
    },
    "energy": {
        "summary": (
            "MMFF94s energy spread of evaluated conformers (secondary to geometry; "
            "not \"lower is better\"). Per ligand: min/max/median/std of finite "
            "MMFF94s energies. Cohort min/max/median/std are means of those; "
            "`median_energy_std` is the median of per-ligand stds. Generation / chembl3d_gt_pb "
            "use PB-kept sets; chembl3d_gt uses the full ChEMBL3D ensemble."
        ),
        "columns": {
            "stratum / row_type / display_label / family / tier / method / ligands / ligands_scope": (
                "Same identity fields as Overview."
            ),
            "energy_min": "Mean of per-ligand minimum MMFF94s energy.",
            "energy_max": "Mean of per-ligand maximum MMFF94s energy.",
            "energy_median": "Mean of per-ligand median MMFF94s energy.",
            "energy_std": "Mean per-ligand energy std — high ⇒ multi-basin; low + few clusters ⇒ collapse.",
            "median_energy_std": "Median of per-ligand energy std — typical ligand spread, less pulled by outlier molecules.",
        },
    },
    "casf_hits": {
        "summary": (
            "Recovery of the CASF crystal pose (bound-pose proxy) — primary success "
            "metrics. Per ligand: heavy-atom symmetry-aware best/median RMSD to "
            "crystal; Hit@X = 1 if best ≤ X Å. Cohort = mean of per-ligand values "
            "(hit means = ligand fractions). Thresholds: 0.25 / 0.5 / 0.75 / 2.0 Å."
        ),
        "columns": {
            "stratum / row_type / display_label / family / tier / method / ligands / ligands_scope": (
                "Same identity fields as Overview."
            ),
            "casf_best_rmsd": "Mean per-ligand best crystal RMSD (Å). Lower better.",
            "casf_median_rmsd": "Mean per-ligand median crystal RMSD (Å).",
            "casf_hit_0p25": "Fraction of ligands with best ≤ 0.25 Å.",
            "casf_hit_0p5": "Fraction with best ≤ 0.5 Å.",
            "casf_hit_0p75": "Fraction with best ≤ 0.75 Å (headline Hit@0.75).",
            "casf_hit_2p0": "Fraction with best ≤ 2.0 Å.",
        },
    },
    "casf_opt_hits": {
        "summary": (
            "Same recovery metrics against CASF optimized ligand poses (`ligands_opt`), "
            "not the crystal. Identical pipeline with `casf_opt_*` prefix; ligands "
            "without an opt pose contribute NaNs and are omitted from means."
        ),
        "columns": {
            "stratum / row_type / display_label / family / tier / method / ligands / ligands_scope": (
                "Same identity fields as Overview."
            ),
            "casf_opt_best_rmsd": "Mean best RMSD to optimized pose (Å).",
            "casf_opt_median_rmsd": "Mean median RMSD to optimized pose (Å).",
            "casf_opt_hit_0p25": "Fraction with best opt RMSD ≤ 0.25 Å.",
            "casf_opt_hit_0p5": "Fraction ≤ 0.5 Å.",
            "casf_opt_hit_0p75": "Fraction ≤ 0.75 Å.",
            "casf_opt_hit_2p0": "Fraction ≤ 2.0 Å.",
        },
    },
    "funnel": {
        "summary": (
            "Generation validity / yield funnel for generation rows only: selected "
            "pool → PoseBusters → kept. Not diversity or CASF recovery. Rates are "
            "per-ligand then averaged. Dynamic / chembl_count report the selected "
            "subset entering PB (PB run once on the fixed pool)."
        ),
        "columns": {
            "row_type / display_label / family / tier / method": "Method identity (always generation here).",
            "ligands": "Manifest ligands in the cohort.",
            "generated_ligands": "Ligands with ≥1 generated candidate before PB.",
            "selected_ligands": "Ligands with ≥1 conf in the PB input pool.",
            "kept_ligands": "Ligands with ≥1 conf kept after PB.",
            "target_confs_mean": "Mean target size (1000 / dynamic formula / ChEMBL count).",
            "selected_pool_total": "Sum of PB input confs.",
            "pb_fail_total": "Sum of PB failures.",
            "kept_confs_total": "Sum of kept confs after filters.",
            "pb_fail_rate_mean": "Mean of `pb_fail / pb_input` per ligand.",
            "kept_vs_target_rate_mean": "Mean of `kept / target_confs` per ligand (yield).",
        },
    },
    "extended_paired_delta_vs_chembl3d_pb": {
        "summary": (
            "Ligand-matched comparison of each fixed-tier generator (plus chembl3d_gt) "
            "vs chembl3d_gt_pb: does the method beat PB-filtered ChEMBL3D on crystal "
            "recovery? `delta = method − baseline` (negative best-RMSD = better). "
            "Win/loss use a 0.1 Å margin; strata cover rotors, heavy atoms, ChEMBL "
            "conf-count bins, and baseline difficulty."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "stratum_type": "`total` or bin kind.",
            "stratum": "Bin value (`all` for total).",
            "n_common_ligands": "Paired ligands with finite deltas.",
            "mean_delta_casf_best_rmsd": "Mean (method − baseline) best RMSD; <0 better.",
            "median_delta_casf_best_rmsd": "Median of those deltas.",
            "win_rate_0p1A": "Fraction with delta < −0.1 Å.",
            "loss_rate_0p1A": "Fraction with delta > +0.1 Å.",
            "delta_hit_0p5": "Mean change in Hit@0.5 (method − baseline).",
            "delta_hit_0p75": "Mean change in Hit@0.75.",
            "delta_hit_2p0": "Mean change in Hit@2.0.",
            "catastrophic_loss_count_gt1A": "Ligands with delta > +1.0 Å.",
            "large_rescue_count_gt1A": "Ligands with delta < −1.0 Å.",
        },
    },
    "extended_paired_delta_extreme_ligands_vs_chembl3d_pb": {
        "summary": (
            "Per-ligand extremes vs chembl3d_gt_pb: top-20 rescued (most negative "
            "delta) and top-20 worsened (most positive) per method. Use to inspect "
            "where generators help or hurt relative to PB-filtered ChEMBL3D."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "baseline_source": "Usually `chembl3d_gt_pb`.",
            "direction": "`rescued` or `worsened`.",
            "mol_id": "Ligand id.",
            "baseline_casf_best_rmsd": "Baseline best crystal RMSD (Å).",
            "method_casf_best_rmsd": "Method best crystal RMSD (Å).",
            "delta_casf_best_rmsd": "method − baseline (negative = rescue).",
            "baseline_casf_hit_0p75": "Baseline Hit@0.75 (0/1).",
            "method_casf_hit_0p75": "Method Hit@0.75 (0/1).",
        },
    },
    "extended_paired_delta_vs_rdkit_raw": {
        "summary": (
            "Same ligand-matched paired stats as vs ChEMBL3D-PB, but baseline is "
            "`rdkit_random_raw_fixed` (classical diverse RDKit). Same delta sign "
            "convention: negative best-RMSD delta = method closer to crystal."
        ),
        "columns": {
            "ligand_set / source / family / tier / method / stratum_type / stratum": "Method + stratum identity.",
            "n_common_ligands": "Paired ligands with finite deltas.",
            "mean_delta_casf_best_rmsd": "Mean (method − RDKit) best RMSD; <0 better.",
            "median_delta_casf_best_rmsd": "Median of those deltas.",
            "win_rate_0p1A": "Fraction with delta < −0.1 Å.",
            "loss_rate_0p1A": "Fraction with delta > +0.1 Å.",
            "delta_hit_0p5 / delta_hit_0p75 / delta_hit_2p0": "Mean Hit@X change vs RDKit.",
            "catastrophic_loss_count_gt1A": "Ligands with delta > +1.0 Å.",
            "large_rescue_count_gt1A": "Ligands with delta < −1.0 Å.",
        },
    },
    "extended_paired_delta_extreme_ligands_vs_rdkit_raw": {
        "summary": (
            "Extreme rescued / worsened ligands vs `rdkit_random_raw_fixed`. Same "
            "layout as Extremes vs ChEMBL3D-PB; baseline_source is RDKit raw fixed."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "baseline_source": "`rdkit_random_raw_fixed`.",
            "direction": "`rescued` or `worsened`.",
            "mol_id": "Ligand id.",
            "baseline_casf_best_rmsd": "RDKit best crystal RMSD (Å).",
            "method_casf_best_rmsd": "Method best crystal RMSD (Å).",
            "delta_casf_best_rmsd": "method − baseline.",
            "baseline_casf_hit_0p75 / method_casf_hit_0p75": "Hit@0.75 flags (0/1).",
        },
    },
    "extended_frontier_summary": {
        "summary": (
            "Fixed-tier generation tradeoff ranking: Hit@0.75 vs PB validity vs "
            "ensemble size. `valid_hit_score = hit_0p75 × pb_pass_rate`. "
            "`frontier_rank` is non-dominated rank maximizing hit & PB pass rate "
            "while minimizing mean conformer count (1 = front)."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Fixed-tier generation methods.",
            "n_ligands": "Ligands averaged.",
            "mean_casf_hit_0p75": "Mean Hit@0.75.",
            "mean_casf_best_rmsd": "Mean best crystal RMSD (Å).",
            "mean_pb_pass_rate": "Mean PoseBusters pass rate.",
            "mean_conformer_count": "Mean ensemble size.",
            "mean_greedy_clusters_1p0": "Mean 1.0 Å cluster count.",
            "useful_clusters_per_100_confs": "`mean_clusters_1p0 / mean_confs × 100`.",
            "hit_0p75_per_100_confs": "`mean_hit_0p75 / mean_confs × 100`.",
            "valid_hit_score": "Hit@0.75 × PB pass rate.",
            "frontier_rank": "1 = non-dominated front.",
        },
    },
    "extended_pb_failure_mechanisms": {
        "summary": (
            "Which PoseBusters checks dominate failures and how severe per-ligand "
            "yield tails are. Fail counts summed from per-ligand PB check JSON; "
            "bond-geometry fraction = bond_angles + bond_lengths + bonds."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Generation method identity.",
            "n_ligands": "Ligand count.",
            "total_pb_fail_rate": "Σ pb_fail / Σ pb_input.",
            "median_per_ligand_pb_pass_rate": "Median of per-ligand PB pass rates.",
            "ligands_pb_pass_rate_lt_0p9": "Count with pass rate < 0.9.",
            "ligands_pb_pass_rate_lt_0p5": "Count with pass rate < 0.5.",
            "dominant_pb_failure_test": "Most frequent failing PB test.",
            "second_dominant_pb_failure_test": "Second most frequent failing test.",
            "fraction_failures_tetrahedral_chirality": "Share of check failures from chirality.",
            "fraction_failures_internal_steric_clash": "Clash share.",
            "fraction_failures_energy_ratio": "Energy-ratio share.",
            "fraction_failures_bond_geometry_tests": "Combined bond-geometry share.",
        },
    },
    "extended_pb_failure_tail_ligands": {
        "summary": (
            "Worst 25 ligands by PB pass rate per method — concrete yield problem "
            "cases with top failing tests and crystal-recovery context."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "mol_id": "Ligand id.",
            "rotatable_bonds / heavy_atoms": "Size / flexibility context.",
            "pb_pass_rate": "Per-ligand PB pass rate.",
            "pb_input_confs": "PB input conformer count.",
            "top_three_failed_pb_tests": "Top-3 failing tests by count.",
            "casf_best_rmsd": "Best crystal RMSD for this ligand (Å).",
            "casf_hit_0p75": "Hit@0.75 for this ligand (0/1).",
        },
    },
    "extended_bound_pose_difficulty": {
        "summary": (
            "Method performance stratified by how hard the ligand is for "
            "chembl3d_gt_pb (baseline best RMSD bins: easy ≤0.5, moderate ≤0.75, "
            "hard ≤2.0, failed >2.0 Å). Rescue = baseline miss@0.75 & method "
            "hit@0.75; catastrophic loss = baseline hit@0.75 & method miss@2.0."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "baseline_source": "`chembl3d_gt_pb`.",
            "difficulty_bin": "easy / moderate / hard / failed.",
            "n_common_ligands": "Paired ligands in the bin.",
            "mean_delta_casf_best_rmsd": "Mean method − baseline best RMSD.",
            "hit_0p75": "Method Hit@0.75 within the bin.",
            "rescue_rate": "Fraction rescued (def. above).",
            "catastrophic_loss_rate": "Fraction catastrophic losses.",
        },
    },
    "extended_rescue_cases": {
        "summary": (
            "Concrete ligands where selected methods (LOQI raw fixed, torsional "
            "diffusion raw fixed, RDKit random raw fixed) rescue a chembl3d_gt_pb "
            "miss@0.75 — up to 50 per method, sorted by delta."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "mol_id": "Ligand id.",
            "rotatable_bonds / heavy_atoms": "Size / flexibility context.",
            "baseline_casf_best_rmsd": "chembl3d_gt_pb best RMSD (Å).",
            "method_casf_best_rmsd": "Method best RMSD (Å).",
            "delta_casf_best_rmsd": "method − baseline (large negative = strong rescue).",
            "pb_pass_rate": "Method PB pass rate.",
            "greedy_clusters_1p0": "Method 1.0 Å cluster count.",
        },
    },
    "extended_k_efficiency": {
        "summary": (
            "How crystal recovery scales with sample size K. Generation: random "
            "subsamples of per-conformer CASF RMSDs (K ∈ {1…1000}, up to 20 draws). "
            "ChEMBL3D-PB: first_k (source order) and random_k. "
            "`capped_at_available` uses min(K,n); `strict_at_least_k` keeps only "
            "ligands with ≥K PB confs."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method / ChEMBL-PB identity.",
            "sampling_mode": "`random_subsample`, `first_k`, or `random_k`.",
            "summary_mode": "`capped_at_available` or `strict_at_least_k`.",
            "k": "Sample size.",
            "status / skip_reason": "`computed` or `skipped` + reason.",
            "n_ligands": "Ligand universe size.",
            "n_metric_ligands": "Ligands used for metrics (strict mode).",
            "n_ligands_with_pb": "Ligands with ≥1 PB conf / RMSD.",
            "mean_n_used": "Mean confs used (`min(K, available)`).",
            "mean_best_rmsd_at_k": "Mean best RMSD at K (Å).",
            "hit_0p25_at_k / hit_0p75_at_k / hit_2p0_at_k": "Mean Hit@threshold at K.",
            "ci_low / ci_high": "2.5–97.5% quantiles of Hit@0.75 over random seeds.",
            "mean_pb_pass_rate_at_k": "Mean PB pass rate where available.",
            "mean_conformers_available": "Mean available confs.",
            "n_ligands_with_at_least_k": "Count with ≥K available.",
        },
    },
    "extended_chembl_k_efficiency_per_ligand": {
        "summary": (
            "ChEMBL3D-PB per-ligand K results — detail behind the K-efficiency "
            "curves (debugging / ligand-level plots)."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "sampling_mode / seed / k / mol_id": "Task key for one draw.",
            "n_available_pb": "PB-passing confs with RMSDs.",
            "n_used": "`min(K, available)`.",
            "casf_best_rmsd_at_k": "Best RMSD among used confs (Å).",
            "casf_hit_0p25_at_k / casf_hit_0p5_at_k / casf_hit_0p75_at_k / casf_hit_2p0_at_k": (
                "Hit flags at each threshold for that draw."
            ),
        },
    },
    "extended_k_efficiency_comparisons": {
        "summary": (
            "Matched-K paired deltas: Qwen vs ChEMBL3D-PB (first_k and mean "
            "random_k) plus ChEMBL saturation (`chembl_full_available_vs_chembl_k`). "
            "Same delta sign: negative best-RMSD = method closer to crystal."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Usually Qwen method or ChEMBL.",
            "comparison": "Comparison id string.",
            "baseline_source": "`chembl3d_gt_pb`.",
            "baseline_sampling_mode": "e.g. `first_k`, `random_k_mean`.",
            "k": "Matched sample size.",
            "n_common_ligands": "Paired ligands.",
            "mean_delta_best_rmsd_at_k": "Mean (method − baseline) best RMSD at K.",
            "median_delta_best_rmsd_at_k": "Median of those deltas.",
            "delta_hit_0p25_at_k / delta_hit_0p5_at_k / delta_hit_0p75_at_k / delta_hit_2p0_at_k": (
                "Mean hit-rate change at each threshold."
            ),
        },
    },
    "extended_energy_window_summary": {
        "summary": (
            "Crystal recovery and 1 Å clustering inside relative MMFF energy "
            "windows of PB-passing confs (ΔE = E − min_E): `all`, `deltaE_5/10/20/50`. "
            "Asks whether bioactive hits sit in low-energy basins or high-energy "
            "tails. Empty-window ligands count as misses; retention ratios compare "
            "`deltaE_10` vs `all`."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "energy_window": "Window name.",
            "n_ligands": "Full method universe.",
            "n_ligands_with_window_confs": "Ligands with ≥1 conf in the window.",
            "mean_fraction_window_confs": "Mean fraction of PB confs kept in window.",
            "mean_n_window_confs": "Mean confs in window.",
            "mean_best_rmsd_window": "Mean best crystal RMSD in window (Å).",
            "hit_0p25_window / hit_0p5_window / hit_0p75_window / hit_2p0_window": (
                "Hit rates in window (miss if empty)."
            ),
            "mean_clusters_1p0_window": "Mean 1.0 Å clusters in window.",
            "mean_clusters_per_100_1p0_window": "Size-normalized clusters in window.",
            "mean_entropy_1p0_window": "Mean normalized cluster entropy.",
            "mean_largest_cluster_fraction_1p0_window": "Mean largest-cluster fraction.",
            "hit_retention_deltaE_10": "Hit@0.75(deltaE_10) / Hit@0.75(all).",
            "cluster_retention_deltaE_10": "Clusters(deltaE_10) / Clusters(all).",
            "mean_useful_low_energy_clusters": (
                "Mean 1.0 Å clusters containing a ≤0.75 Å hit (deltaE_10 only)."
            ),
        },
    },
    "extended_energy_window_per_ligand": {
        "summary": (
            "Same energy-window metrics per ligand — detail / QA behind the "
            "summary table. Empty window ⇒ hit flags 0."
        ),
        "columns": {
            "ligand_set / source / family / tier / method / mol_id / energy_window": "Row key.",
            "n_pb_confs": "PB confs for the ligand.",
            "n_window_confs / fraction_window_confs": "Window occupancy.",
            "window_energy_min": "Min energy among window confs.",
            "window_energy_max_delta": "Max ΔE of window confs vs global min.",
            "casf_best_rmsd_window / casf_hit_*_window": "Recovery in window.",
            "greedy_clusters_1p0_window / clusters_per_100_1p0_window / cluster_entropy_1p0_window / largest_cluster_fraction_1p0_window": (
                "Clustering in window."
            ),
            "useful_low_energy_clusters": "Useful clusters (filled for deltaE_10 only).",
        },
    },
    "extended_energy_window_comparisons": {
        "summary": (
            "Window-to-window Hit@0.75 shifts (`all_vs_deltaE_20`, "
            "`deltaE_20_vs_deltaE_10`) and how often hits live outside a tight "
            "energy band. `high_energy_hit_fraction`: among ligands that hit@0.75 "
            "in `all`, fraction that miss in `deltaE_20`."
        ),
        "columns": {
            "ligand_set / source / family / tier / method": "Method identity.",
            "comparison": "e.g. `all_vs_deltaE_20`.",
            "n_ligands": "Ligands compared.",
            "delta_hit_0p75_window": "Hit@0.75 change (right − left window).",
            "high_energy_hit_fraction": "Fraction of all-window hits not recovered inside ΔE≤20.",
        },
    },
    "extended_per_conformer_casf_rmsd_manifest": {
        "summary": (
            "Inventory of ligands/methods with per-conformer CASF RMSD parts "
            "(supports K-efficiency) — provenance, not a metrics table."
        ),
        "columns": {
            "ligand_set / run_id / run_label": "Provenance.",
            "source / family / tier / method": "Method identity.",
            "mol_id": "Ligand id.",
            "conformers_available": "Conf count with RMSDs.",
            "rotatable_bonds / heavy_atoms": "Ligand descriptors.",
        },
    },
    "extended_sanity_checks": {
        "summary": (
            "Automated QA of dashboard inputs before trusting extended tables: "
            "required columns, duplicate rows, RMSD/hit ranges, PB count "
            "consistency, and related warnings."
        ),
        "columns": {
            "check": "Check id.",
            "severity": "`error` or `warn`.",
            "status": "`pass` / `fail` / `warn`.",
            "affected_rows": "Count of bad rows (or 1 for global flags).",
            "details": "Human-readable note.",
        },
    },
    "plinder_clustering": {
        "summary": (
            "Greedy heavy-atom RMSD clustering of the full generated pool "
            "(about 1,000 conformers), at 0.5 / 1.0 / 2.0 / 3.0 Å. "
            "Values are means over the 23 molecules. PoseBusters pass rate is separate "
            "and is not a filter on the clusters."
        ),
        "columns": {
            "display_label": "Method name.",
            "n_molecules": "Molecules with a generated pool.",
            "mean_n_generated": "Mean generated conformers per molecule.",
            "mean_pb_pass_rate": "Mean fraction of conformers that pass PoseBusters.",
            "mean_clusters_0p5": "Mean greedy cluster count at 0.5 Å.",
            "mean_clusters_1p0": "Mean greedy cluster count at 1.0 Å.",
            "largest_cluster_fraction_1p0": "Mean fraction of the pool in the largest 1.0 Å cluster.",
        },
    },
    "plinder_energy": {
        "summary": (
            "MMFF94s after stripping and rebuilding hydrogens, with no minimization, "
            "on PoseBusters-passing conformers. Each column is the mean of the "
            "per-molecule statistic. Molecules with no MMFF parameters are omitted "
            "from that molecule's energy and stay blank."
        ),
        "columns": {
            "energy_min": "Mean of per-molecule minimum energies.",
            "energy_median": "Mean of per-molecule median energies.",
            "energy_std": "Mean of per-molecule energy standard deviations.",
            "energy_count": "Mean number of conformers with a finite energy.",
        },
    },
    "plinder_recovery": {
        "summary": (
            "COV-R is the fraction of experimental references with a generated "
            "conformer at RMSD ≤ the cutoff. COV-P is the fraction of generated "
            "conformers within the cutoff of some reference. Both are averaged "
            "equally across molecules. MAT-R and MAT-P are the mean nearest distances "
            "and do not use a cutoff."
        ),
        "columns": {
            "cov_r_0p5_mean": "Mean COV-R at 0.5 Å.",
            "cov_r_0p75_mean": "Mean COV-R at 0.75 Å.",
            "cov_r_1p0_mean": "Mean COV-R at 1.0 Å.",
            "mat_r_mean": "Mean MAT-R (Å).",
            "mat_p_mean": "Mean MAT-P (Å).",
        },
    },
}


def render_table_help(name: str) -> None:
    """Render About / Columns help for a dashboard table. No-op if unknown."""
    help_block = TABLE_HELP.get(name)
    if not help_block:
        return

    summary = str(help_block["summary"])
    columns = help_block["columns"]
    assert isinstance(columns, dict)

    st.markdown("##### About this table")
    st.markdown(summary)

    lines = [f"- **`{col}`** — {gloss}" for col, gloss in columns.items()]
    with st.expander("Column glossary", expanded=False):
        st.markdown("\n".join(lines))
