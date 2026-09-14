from __future__ import annotations

import importlib.util
from types import ModuleType

import pandas as pd

from casf_benchmark.paths import REPO_ROOT


def load_charts() -> ModuleType:
    path = REPO_ROOT / "apps" / "dashboard" / "druglike_k_charts.py"
    spec = importlib.util.spec_from_file_location("druglike_k_charts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_k_charts_keep_the_six_method_panel() -> None:
    charts = load_charts()
    summary = pd.DataFrame(
        [
            {
                "label": "qwen_1p7b_fsq_bigdata_step47023",
                "generator": "Qwen",
                "model_size": "1.7B",
                "tokenizer": "FSQ",
                "k": 10,
                "cov_r_mean": 0.9,
                "cov_p_mean": 0.4,
            },
            {
                "label": "qwen_0p6b_bigdata_step111000",
                "generator": "Qwen",
                "model_size": "0.6B",
                "tokenizer": "Binned",
                "k": 10,
                "cov_r_mean": 0.1,
                "cov_p_mean": 0.1,
            },
            {
                "label": "loqi_druglike",
                "generator": "LOQI",
                "display_label": "LOQI",
                "k": 10,
                "cov_r_mean": 0.8,
                "cov_p_mean": 0.5,
            },
        ]
    )
    filtered = charts.filter_plot_summary(summary)
    assert set(filtered["label"]) == {"qwen_1p7b_fsq_bigdata_step47023", "loqi_druglike"}
    assert set(filtered["plot_label"]) == {"Qwen 1.7B FSQ", "LOQI"}
    assert set(filtered["label"]).issubset(charts.PLOT_LABELS)


def test_long_metric_pair_and_chart_cover_recall_and_precision() -> None:
    charts = load_charts()
    frame = pd.DataFrame(
        [
            {"plot_label": "LOQI", "k": 10, "cov_r_mean": 0.2, "cov_p_mean": 0.8},
            {"plot_label": "LOQI", "k": 100, "cov_r_mean": 0.5, "cov_p_mean": 0.6},
        ]
    )
    long = charts.long_metric_pair(frame, "cov_r_mean", "cov_p_mean")
    assert len(long) == 4
    assert set(long["series"]) == {"Recall (R)", "Precision (P)"}
    chart = charts.k_chart(
        frame,
        recall_col="cov_r_mean",
        precision_col="cov_p_mean",
        ylabel="Coverage",
        title="Coverage mean",
    )
    assert chart.to_dict()["mark"]["type"] == "line"
