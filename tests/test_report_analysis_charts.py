from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd

from casf_benchmark.paths import REPO_ROOT


def load_report_charts() -> ModuleType:
    path = REPO_ROOT / "apps" / "dashboard" / "report_analysis_charts.py"
    spec = importlib.util.spec_from_file_location("report_analysis_charts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, ligand in enumerate(("a", "b")):
        rows.extend(
            [
                {
                    "mol_id": ligand,
                    "display_label": "CASF crystal",
                    "row_type": "reference",
                    "tier": "reference",
                    "family": "reference",
                    "ligand_set": "core",
                    "energy_median": 10.0 + index,
                },
                {
                    "mol_id": ligand,
                    "display_label": "ChEMBL3D ground truth",
                    "row_type": "reference",
                    "tier": "reference",
                    "family": "reference",
                    "ligand_set": "core",
                    "energy_median": 11.0 + index,
                    "casf_best_rmsd": 0.2 + index * 0.1,
                    "casf_median_rmsd": 0.8 + index * 0.1,
                    "greedy_clusters_0p5": 4 + index,
                },
                {
                    "mol_id": ligand,
                    "display_label": "Qwen model",
                    "row_type": "generation",
                    "tier": "fixed",
                    "family": "qwen",
                    "ligand_set": "core",
                    "energy_median": 12.0 + index * 2,
                    "energy_std": 1.0,
                    "casf_best_rmsd": 0.25 + index * 0.1,
                    "casf_median_rmsd": 0.9 + index * 0.1,
                    "greedy_clusters_0p5": 8 + index,
                },
                {
                    "mol_id": ligand,
                    "display_label": "Qwen model",
                    "row_type": "generation",
                    "tier": "dynamic",
                    "family": "qwen",
                    "ligand_set": "core",
                    "energy_median": 30.0 + index,
                    "casf_best_rmsd": 1.0 + index,
                },
                {
                    "mol_id": ligand,
                    "display_label": "Other model",
                    "row_type": "generation",
                    "tier": "fixed",
                    "family": "other",
                    "ligand_set": "core",
                    "energy_median": 14.0 + index,
                    "casf_best_rmsd": 0.4 + index * 0.1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_filter_report_rows_selects_tier_and_keeps_references() -> None:
    charts = load_report_charts()
    view = charts.filter_report_rows(
        report_rows(),
        ligand_set="core",
        tier="fixed",
        family="qwen",
    )
    assert set(view["plot_method"]) == {
        "CASF crystal",
        "ChEMBL3D ground truth",
        "Qwen model",
    }
    assert not (
        (view["row_type"] == "generation") & (view["tier"] != "fixed")
    ).any()


def test_all_method_summaries_use_every_selected_ligand() -> None:
    charts = load_report_charts()
    view = charts.filter_report_rows(
        report_rows(),
        ligand_set="core",
        tier="fixed",
        family="All",
    )
    summary = charts.aggregate_method_summary(view)
    qwen = summary[summary["method"] == "Qwen model"].iloc[0]
    assert qwen["n_ligands"] == 2
    assert qwen["mean_best_rmsd"] == 0.3
    assert qwen["mean_clusters_0p5"] == 8.5

    delta = charts.energy_delta_rows(view)
    qwen_delta = delta[delta["plot_method"] == "Qwen model"]["energy_delta"]
    assert qwen_delta.tolist() == [2.0, 3.0]


def test_tukey_summary_is_exact_and_chartable() -> None:
    charts = load_report_charts()
    frame = pd.DataFrame(
        {
            "plot_method": ["model"] * 5,
            "plot_category": ["Qwen"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    summary = charts.tukey_box_summary(frame, metric="value")
    row = summary.iloc[0]
    assert row["count"] == 5
    assert row["median"] == 3.0
    assert row["whisker_high"] == 4.0
    assert charts.box_summary_chart(
        summary,
        title="Test",
        axis_title="Value",
    ).to_dict()["title"] == "Test"


def test_complete_detailed_plot_suite_includes_crystal_and_all_methods() -> None:
    charts = load_report_charts()
    view = charts.filter_report_rows(
        report_rows(),
        ligand_set="core",
        tier="fixed",
        family="All",
    )
    energy_domain = charts.energy_panel_domain(view)
    rmsd_domain = charts.rmsd_panel_domain(view)

    crystal = charts.energy_small_multiple_chart(
        view,
        "CASF crystal",
        y_domain=energy_domain,
    ).to_dict()
    method_energy = charts.energy_small_multiple_chart(
        view,
        "Qwen model",
        y_domain=energy_domain,
    ).to_dict()
    delta = charts.energy_delta_distribution_chart(view).to_dict()
    paired_energy = charts.energy_reference_chart(
        view,
        "Qwen model",
        y_domain=energy_domain,
    ).to_dict()
    all_rmsd = charts.rmsd_all_methods_chart(view).to_dict()
    method_rmsd = charts.rmsd_detail_chart(
        view,
        "Qwen model",
        y_domain=rmsd_domain,
    ).to_dict()
    rmsd_distribution = charts.rmsd_distribution_chart(view).to_dict()

    assert "layer" not in crystal
    assert len(method_energy["layer"]) == 2
    assert energy_domain == [-200.0, 420.0]
    assert rmsd_domain == [0.0, 2.6]
    assert delta["title"] == "Deviation from CASF crystal"
    assert paired_energy["layer"][0]["encoding"]["x"]["field"] == "ligand_rank"
    assert paired_energy["layer"][0]["encoding"]["y"]["field"] == "energy_low"
    assert paired_energy["layer"][0]["encoding"]["y2"]["field"] == "energy_high"
    assert all_rmsd["title"] == "Per ligand: best RMSD, all methods together"
    assert len(method_rmsd["layer"]) == 3
    assert len(rmsd_distribution["layer"]) == 5
    assert rmsd_distribution["layer"][0]["encoding"]["x"]["field"] == "threshold"
    assert rmsd_distribution["layer"][-1]["encoding"]["x"]["field"] == (
        "casf_best_rmsd"
    )


def test_reference_report_order_and_delta_sort_are_preserved() -> None:
    charts = load_report_charts()
    methods = [
        "CASF crystal",
        "Qwen 4B revisited",
        "LOQI",
        "Torsion perturb (raw)",
    ]
    assert charts.report_method_order(methods, reference_last=True) == [
        "Torsion perturb (raw)",
        "LOQI",
        "Qwen 4B revisited",
        "CASF crystal",
    ]

    view = charts.filter_report_rows(
        report_rows(),
        ligand_set="core",
        tier="fixed",
        family="All",
    )
    chart = charts.energy_delta_distribution_chart(view).to_dict()
    method_sort = chart["layer"][1]["encoding"]["y"]["sort"]
    assert method_sort == ["ChEMBL3D ground truth", "Qwen model", "Other model"]


def test_report_defaults_to_chembl_count_when_sidebar_tier_is_all() -> None:
    source = (
        REPO_ROOT / "apps" / "dashboard" / "report_analysis_charts.py"
    ).read_text(encoding="utf-8")
    assert 'tiers.index("chembl_count")' in source


def test_rmsd_order_can_come_from_hidden_reference_rows() -> None:
    charts = load_report_charts()
    view = charts.filter_report_rows(
        report_rows(),
        ligand_set="core",
        tier="fixed",
        family="All",
    )
    view.loc[
        view["plot_method"] == "ChEMBL3D ground truth",
        "casf_best_rmsd",
    ] = [0.9, 0.1]
    displayed = view[view["row_type"] == "generation"].copy()
    chart = charts.rmsd_all_methods_chart(
        displayed,
        order_frame=view,
    ).to_dict()
    dataset_name = chart["layer"][0]["data"]["name"]
    plotted = pd.DataFrame(chart["datasets"][dataset_name])
    ranks = dict(zip(plotted["mol_id"], plotted["ligand_rank"]))
    assert ranks == {"b": 1, "a": 2}


def test_report_section_is_immediately_before_extended_analysis() -> None:
    source = (
        REPO_ROOT / "apps" / "dashboard" / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    druglike = source.rindex("    render_druglike_tables(extended_db_path)")
    report = source.rindex("    render_report_analysis(")
    extended = source.rindex("    render_extended_analysis(extended_db_path)")
    assert druglike < report < extended


def test_dashboard_renders_all_six_reference_report_figures() -> None:
    source = (
        REPO_ROOT / "apps" / "dashboard" / "report_analysis_charts.py"
    ).read_text(encoding="utf-8")
    for heading in (
        "All methods on one chart: deviation from CASF crystal",
        "Each method individually against CASF crystal",
        "Per ligand: best RMSD, all methods together",
        "Overall distribution of best RMSD by method",
    ):
        assert heading in source
    assert source.count('st.subheader("One by one, side by side")') == 2
