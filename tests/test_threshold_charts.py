from __future__ import annotations

import importlib.util
from types import ModuleType

import pandas as pd

from casf_benchmark.paths import REPO_ROOT


def load_charts() -> ModuleType:
    path = REPO_ROOT / "apps" / "dashboard" / "threshold_charts.py"
    spec = importlib.util.spec_from_file_location("threshold_charts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_type": "generation",
                "ligand_set": "core",
                "tier": "fixed",
                "family": "loqi_raw",
                "display_label": "LOQI",
                "casf_hit_0p25": 0.5,
                "casf_hit_0p5": 0.8,
                "casf_hit_0p75": 0.9,
                "casf_hit_2p0": 1.0,
                "mean_clusters_0p5": 80.0,
                "mean_clusters_1p0": 20.0,
                "mean_clusters_2p0": 3.0,
                "mean_clusters_3p0": 1.2,
            },
            {
                "row_type": "generation",
                "ligand_set": "core",
                "tier": "dynamic",
                "family": "loqi_raw",
                "display_label": "LOQI",
                "casf_hit_0p25": 0.4,
                "casf_hit_0p5": 0.7,
                "casf_hit_0p75": 0.85,
                "casf_hit_2p0": 1.0,
                "mean_clusters_0p5": 40.0,
                "mean_clusters_1p0": 10.0,
                "mean_clusters_2p0": 2.0,
                "mean_clusters_3p0": 1.1,
            },
            {
                "row_type": "generation",
                "ligand_set": "core",
                "tier": "fixed",
                "family": "rdkit_random_raw",
                "display_label": "RDKit random (raw)",
                "casf_hit_0p25": 0.3,
                "casf_hit_0p5": 0.6,
                "casf_hit_0p75": 0.8,
                "casf_hit_2p0": 1.0,
                "mean_clusters_0p5": 100.0,
                "mean_clusters_1p0": 30.0,
                "mean_clusters_2p0": 4.0,
                "mean_clusters_3p0": 1.3,
            },
            {
                "row_type": "generation",
                "ligand_set": "core",
                "tier": "fixed",
                "family": "qwen_1p7b_fsq_bigdata_pretrain",
                "display_label": "Qwen 1.7B fsq+bigdata-pretrain",
                "casf_hit_0p75": 0.9,
                "mean_clusters_1p0": 70.0,
            },
            {
                "row_type": "generation",
                "ligand_set": "core",
                "tier": "fixed",
                "family": "qwen_0p6b_bigdata_step111000",
                "display_label": "other qwen",
                "casf_hit_0p75": 0.1,
                "mean_clusters_1p0": 5.0,
            },
            {
                "row_type": "reference",
                "ligand_set": "core",
                "tier": "reference",
                "family": "chembl3d_gt_pb",
                "display_label": "ChEMBL3D-PB",
                "casf_hit_0p75": 0.99,
                "mean_clusters_1p0": 50.0,
            },
        ]
    )


def test_panel_drops_other_families_and_references() -> None:
    charts = load_charts()
    filtered = charts.filter_panel(_rows(), ligand_set="core", tier="All", family="All")
    assert set(filtered["family"]) == {"loqi_raw", "rdkit_random_raw", "qwen_1p7b_fsq_bigdata_pretrain"}
    assert set(filtered["plot_label"]) == {"LOQI", "RDKit random (raw)", "Qwen 1.7B FSQ"}
    assert "dynamic" in set(filtered["tier"].astype(str))


def test_tier_dropdown_keeps_only_the_selected_tier() -> None:
    charts = load_charts()
    filtered = charts.filter_panel(_rows(), ligand_set="core", tier="fixed", family="All")
    assert set(filtered["tier"].astype(str)) == {"fixed"}
    assert set(filtered["family"]) == {"loqi_raw", "rdkit_random_raw", "qwen_1p7b_fsq_bigdata_pretrain"}


def test_available_thresholds_skip_missing_columns() -> None:
    charts = load_charts()
    frame = pd.DataFrame([{"casf_hit_0p75": 0.5, "casf_hit_2p0": float("nan")}])
    found = charts.available_thresholds(frame, charts.HIT_THRESHOLDS)
    assert found == [("casf_hit_0p75", 0.75)]


def test_melt_and_chart_use_numeric_thresholds() -> None:
    charts = load_charts()
    filtered = charts.filter_panel(_rows(), ligand_set="core", tier="fixed", family="loqi_raw")
    long = charts.melt_thresholds(filtered, charts.HIT_THRESHOLDS)
    assert set(long["threshold"]) == {0.25, 0.5, 0.75, 2.0}
    assert list(long.sort_values("threshold")["value"]) == [0.5, 0.8, 0.9, 1.0]
    chart = charts.threshold_chart(
        long, ylabel="Hit rate", title="Hit rate (fixed)", method_order=["LOQI"]
    )
    assert chart.to_dict()["mark"]["type"] == "line"
