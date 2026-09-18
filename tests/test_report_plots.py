from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from casf_benchmark.visualization.report_plots import (
    ReportPlotConfig,
    generate_report_plots,
)


def synthetic_plot_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(8):
        ligand = f"ligand-{index}"
        rows.extend(
            [
                {
                    "mol_id": ligand,
                    "display_label": "CASF crystal",
                    "tier": "reference",
                    "energy_median": 20.0 + index,
                },
                {
                    "mol_id": ligand,
                    "display_label": "ChEMBL3D ground truth",
                    "tier": "reference",
                    "energy_median": 21.0 + index,
                    "energy_std": 0.5,
                    "casf_best_rmsd": 0.1 + index * 0.06,
                    "casf_median_rmsd": 0.6 + index * 0.08,
                },
                {
                    "mol_id": ligand,
                    "display_label": "New model",
                    "tier": "fixed",
                    "energy_median": 22.0 + index * 1.2,
                    "energy_std": 1.0 + index * 0.1,
                    "casf_best_rmsd": 0.15 + index * 0.05,
                    "casf_median_rmsd": 0.7 + index * 0.09,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_generate_complete_plot_suite(tmp_path: Path) -> None:
    outputs = generate_report_plots(
        synthetic_plot_data(),
        tmp_path,
        config=ReportPlotConfig(tier="fixed"),
    )

    assert set(outputs) == {
        "energy_small_multiples",
        "energy_delta_boxplot",
        "energy_vs_reference",
        "rmsd_all_methods",
        "rmsd_small_multiples",
        "rmsd_boxplot",
    }
    assert all(
        path.exists() and path.stat().st_size > 1_000
        for path in outputs.values()
    )


def test_missing_identity_columns_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ligand_id"):
        generate_report_plots(pd.DataFrame({"energy_median": [1.0]}), tmp_path)


def test_display_label_takes_precedence_over_internal_method(tmp_path: Path) -> None:
    frame = synthetic_plot_data()
    frame["method"] = "internal_key"
    outputs = generate_report_plots(
        frame,
        tmp_path,
        config=ReportPlotConfig(
            tier="fixed",
            methods=("New model",),
        ),
    )
    assert "rmsd_boxplot" in outputs


def test_duplicate_ligand_method_rows_require_tier_selection(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "ligand_id": ["a", "a"],
            "method": ["model", "model"],
            "tier": ["fixed", "dynamic"],
            "best_rmsd": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="Select a tier"):
        generate_report_plots(frame, tmp_path)


def test_original_report_family_csv_shape_is_supported(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "family": ["torsion_raw", "casf_crystal"] * 3,
            "mol_id": ["a", "a", "b", "b", "c", "c"],
            "energy_median": [10.0, 12.0, 20.0, 21.0, 30.0, 28.0],
            "energy_std": [2.0, 0.0, 3.0, 0.0, 1.0, 0.0],
        }
    )
    outputs = generate_report_plots(frame, tmp_path)
    assert set(outputs) == {
        "energy_small_multiples",
        "energy_delta_boxplot",
        "energy_vs_reference",
    }


def test_reference_domains_match_original_html_figures() -> None:
    config = ReportPlotConfig()
    assert config.energy_ylim == (-200.0, 420.0)
    assert config.rmsd_ylim == (0.0, 2.6)
