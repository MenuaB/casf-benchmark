from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from casf_benchmark.analysis.druglike_covmat import per_molecule_metrics
from casf_benchmark.analysis.druglike_k_efficiency import k_rows_at_thresholds, k_rows_for_method
from casf_benchmark.analysis.plinder_panel import (
    HEADLINE_THRESHOLDS,
    common_hydrogen_mmff_energy,
    headline_coverage,
    load_panel_frames,
    load_reference_sdf_dir,
    pool_frame_for_reference,
    publish_plinder_tables,
    recovery_curve,
    recovery_thresholds,
    score_conformer_pool,
    summarize_molecules,
    write_method_outputs,
)
from casf_benchmark.paths import REPO_ROOT


def load_dashboard_module(name: str) -> ModuleType:
    path = REPO_ROOT / "apps" / "dashboard" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_coverage_stays_strict_and_exposes_generic_names() -> None:
    matrix = np.asarray([[0.75]], dtype=float)
    strict = per_molecule_metrics(matrix, 0.75, 3.0)
    inclusive = per_molecule_metrics(matrix, 0.75, 3.0, inclusive=True)
    assert strict["cov_r_075"] == 0.0
    assert strict["cov_p_075"] == 0.0
    assert strict["cov_r"] == 0.0
    assert strict["cov_p"] == 0.0
    assert inclusive["cov_r"] == 1.0
    assert inclusive["cov_r_075"] == 1.0


def test_headline_coverage_uses_inclusive_cutoffs() -> None:
    matrix = np.asarray([[0.5, 2.0], [1.0, 2.0]], dtype=float)
    row = headline_coverage(matrix, inclusive=True)
    assert row["cov_r_0p5"] == pytest.approx(0.5)
    assert row["cov_r_0p75"] == pytest.approx(0.5)
    assert row["cov_r_1p0"] == pytest.approx(1.0)
    assert row["mat_r"] == pytest.approx(0.75)
    assert set(HEADLINE_THRESHOLDS) == {0.5, 0.75, 1.0}


def test_recovery_grid_stops_at_one_angstrom() -> None:
    grid = recovery_thresholds()
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert 1.5 not in grid
    assert set(HEADLINE_THRESHOLDS).issubset(grid)
    curve = recovery_curve({"CC": np.asarray([[0.5, 2.0], [1.0, 2.0]], dtype=float)}, grid)
    by_threshold = {row["threshold"]: row["cov_r_mean"] for row in curve}
    assert by_threshold[0.5] == pytest.approx(0.5)
    assert by_threshold[1.0] == pytest.approx(1.0)
    assert max(by_threshold) == 1.0


def test_k_rows_share_mat_across_thresholds() -> None:
    matrix = np.asarray([[0.5, 2.0], [1.0, 2.0]], dtype=float)
    per_mol, summary = k_rows_at_thresholds(
        {"CC": matrix},
        [2],
        (0.5, 0.75, 1.0),
        n_seeds=1,
        seed=0,
        inclusive=True,
    )
    assert {row["threshold"] for row in summary} == {0.5, 0.75, 1.0}
    assert len({row["mat_r_mean"] for row in summary}) == 1
    assert summary[0]["mat_r_mean"] == pytest.approx(0.75)
    by_threshold = {row["threshold"]: row for row in summary}
    assert by_threshold[0.5]["cov_r_mean"] == pytest.approx(0.5)
    assert by_threshold[1.0]["cov_r_mean"] == pytest.approx(1.0)
    assert {row["threshold"] for row in per_mol} == {0.5, 0.75, 1.0}
    assert "cov_r_075" not in per_mol[0]


def test_single_threshold_k_rows_keep_legacy_columns() -> None:
    matrix = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    per_mol, summary = k_rows_for_method({"CC": matrix}, [2], n_seeds=1, seed=0)
    assert len(summary) == 1
    assert "threshold" not in summary[0]
    assert per_mol[0]["cov_r_075"] == pytest.approx(0.5)
    assert summary[0]["cov_r_mean"] == pytest.approx(0.5)


def _embedded(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    return mol


def _write_reference_sdf(path: Path, mols: list[Chem.Mol], name: str, ccd: str) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            record = Chem.Mol(mol)
            record.SetProp("molecule_name", name)
            record.SetProp("ccd", ccd)
            writer.write(record)
    finally:
        writer.close()


def test_reference_sdf_dir_keys_heavy_atom_smiles(tmp_path: Path) -> None:
    ethanol = Chem.RemoveHs(_embedded("CCO"))
    shifted = Chem.Mol(ethanol)
    conformer = Chem.Conformer(shifted.GetConformer())
    position = conformer.GetAtomPosition(0)
    conformer.SetAtomPosition(0, (position.x + 0.2, position.y, position.z))
    shifted.RemoveAllConformers()
    shifted.AddConformer(conformer, assignId=True)
    methane = Chem.RemoveHs(_embedded("C"))
    _write_reference_sdf(tmp_path / "NCT.sdf", [ethanol, shifted], "Nicotine", "NCT")
    _write_reference_sdf(tmp_path / "FOM.sdf", [methane], "Fosmidomycin", "FOM")

    loaded = load_reference_sdf_dir(tmp_path)

    assert [row["names"][0] for row in loaded.values()] == ["Fosmidomycin", "Nicotine"]
    nicotine = next(row for row in loaded.values() if row["ccd"] == "NCT")
    assert len(nicotine["confs"]) == 2
    assert Chem.MolToSmiles(Chem.MolFromSmiles("CCO")) in loaded


def test_pool_frame_keeps_only_reference_molecules() -> None:
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "B"],
            "name": ["ethanol", "boron"],
            "n_generated": [10, 10],
            "greedy_clusters_1p0": [2, 3],
            "cov_r_0p5": [0.1, 0.2],
            "mat_r": [0.4, 0.5],
            "label": ["x", "x"],
        }
    )
    kept = pool_frame_for_reference(frame, {"CCO": {"names": ["ethanol"]}})
    assert kept is not None
    assert kept["smiles"].tolist() == ["CCO"]
    assert "cov_r_0p5" not in kept.columns
    assert "mat_r" not in kept.columns
    assert "label" not in kept.columns
    assert kept.loc[0, "greedy_clusters_1p0"] == 2
    assert pool_frame_for_reference(frame, {"CCC": {}}) is None


def test_explicit_hydrogens_still_pass_posebusters() -> None:
    ethanol = _embedded("CCO")
    assert ethanol.GetNumAtoms() > Chem.RemoveHs(ethanol).GetNumAtoms()
    row = score_conformer_pool(("CCO", [ethanol], "ethanol"))
    assert row["n_generated"] == 1
    assert row["n_pb_pass"] == 1
    assert math.isfinite(row["energy_min"])


def test_common_hydrogen_energy_scores_ethanol_and_skips_boron() -> None:
    ethanol = common_hydrogen_mmff_energy(_embedded("CCO"))
    assert math.isfinite(ethanol)
    boron = Chem.AddHs(Chem.MolFromSmiles("B"))
    conformer = Chem.Conformer(boron.GetNumAtoms())
    boron.AddConformer(conformer, assignId=True)
    assert math.isnan(common_hydrogen_mmff_energy(boron))


def test_summarize_and_publish_round_trip(tmp_path: Path) -> None:
    per_molecule = pd.DataFrame(
        [
            {
                "name": "A",
                "n_generated": 4,
                "pb_pass_rate": 1.0,
                "greedy_clusters_0p5": 2,
                "greedy_clusters_1p0": 1,
                "largest_cluster_fraction_1p0": 1.0,
                "energy_min": 1.0,
                "energy_median": 2.0,
                "energy_std": 0.5,
                "cov_r_0p5": 0.25,
                "cov_r_0p75": 0.5,
                "cov_r_1p0": 1.0,
                "mat_r": 0.4,
                "mat_p": 0.8,
            },
            {
                "name": "B",
                "n_generated": 2,
                "pb_pass_rate": 0.5,
                "greedy_clusters_0p5": 4,
                "greedy_clusters_1p0": 3,
                "largest_cluster_fraction_1p0": 0.5,
                "energy_min": 3.0,
                "energy_median": 4.0,
                "energy_std": 1.5,
                "cov_r_0p5": 0.75,
                "cov_r_0p75": 1.0,
                "cov_r_1p0": 1.0,
                "mat_r": 0.2,
                "mat_p": 0.6,
            },
        ]
    )
    summary = pd.DataFrame([{**{"label": "loqi_druglike", "display_label": "LOQI"}, **summarize_molecules(per_molecule)}])
    assert summary.loc[0, "mean_clusters_0p5"] == pytest.approx(3.0)
    assert summary.loc[0, "largest_cluster_fraction_1p0"] == pytest.approx(0.75)
    assert summary.loc[0, "cov_r_0p75_mean"] == pytest.approx(0.75)
    assert summary.loc[0, "energy_min"] == pytest.approx(2.0)

    method = tmp_path / "methods" / "loqi_druglike"
    recovery = pd.DataFrame(
        [{"label": "loqi_druglike", "display_label": "LOQI", "threshold": 0.5, "cov_r_mean": 0.5, "cov_p_mean": 0.25}]
    )
    k_efficiency = pd.DataFrame(
        [
            {
                "label": "loqi_druglike",
                "display_label": "LOQI",
                "k": 10,
                "threshold": 0.5,
                "cov_r_mean": 0.2,
                "cov_p_mean": 0.1,
                "cov_r_median": 0.2,
                "cov_p_median": 0.1,
                "mat_r_mean": 0.4,
                "mat_p_mean": 0.8,
                "mat_r_median": 0.4,
                "mat_p_median": 0.8,
            }
        ]
    )
    per_molecule = per_molecule.copy()
    per_molecule.insert(0, "label", "loqi_druglike")
    per_molecule["display_label"] = "LOQI"
    write_method_outputs(
        method,
        summary=summary,
        per_molecule=per_molecule,
        recovery=recovery,
        k_efficiency=k_efficiency,
    )
    loaded = load_panel_frames(tmp_path / "methods")
    assert list(loaded[0]["label"]) == ["loqi_druglike"]
    db = tmp_path / "extended.sqlite"
    publish_plinder_tables(db, *loaded)
    import sqlite3

    with sqlite3.connect(db) as connection:
        published = pd.read_sql("select * from plinder_k_efficiency", connection)
    assert published.loc[0, "threshold"] == pytest.approx(0.5)
    assert published.loc[0, "mat_r_mean"] == pytest.approx(0.4)


def test_dashboard_chart_frame_accepts_threshold_rows() -> None:
    charts = load_dashboard_module("plinder_charts")
    k_charts = load_dashboard_module("druglike_k_charts")
    summary = pd.DataFrame(
        [
            {
                "label": "loqi_druglike",
                "display_label": "LOQI",
                "generator": "LOQI",
                "k": 10,
                "threshold": 0.5,
                "cov_r_mean": 0.2,
                "cov_p_mean": 0.1,
                "cov_r_median": 0.2,
                "cov_p_median": 0.1,
                "mat_r_mean": 0.4,
                "mat_p_mean": 0.9,
                "mat_r_median": 0.4,
                "mat_p_median": 0.9,
            },
            {
                "label": "loqi_druglike",
                "display_label": "LOQI",
                "generator": "LOQI",
                "k": 10,
                "threshold": 0.75,
                "cov_r_mean": 0.6,
                "cov_p_mean": 0.3,
                "cov_r_median": 0.6,
                "cov_p_median": 0.3,
                "mat_r_mean": 0.4,
                "mat_p_mean": 0.9,
                "mat_r_median": 0.4,
                "mat_p_median": 0.9,
            },
            {
                "label": "qwen_0p6b_bigdata_step111000",
                "generator": "Qwen",
                "model_size": "0.6B",
                "tokenizer": "Binned",
                "display_label": "qwen_0p6b_bigdata_step111000",
                "k": 10,
                "threshold": 0.75,
                "cov_r_mean": 0.1,
                "cov_p_mean": 0.1,
                "cov_r_median": 0.1,
                "cov_p_median": 0.1,
                "mat_r_mean": 0.5,
                "mat_p_mean": 1.0,
                "mat_r_median": 0.5,
                "mat_p_median": 1.0,
            },
        ]
    )
    choices = charts.method_choices(summary["label"].tolist())
    assert choices[0] == "loqi_druglike"
    selected = charts.with_plot_labels(charts.coverage_at_threshold(summary, 0.75), ["loqi_druglike"])
    assert set(selected["label"]) == {"loqi_druglike"}
    assert len(selected) == 1
    chart = k_charts.k_chart(
        selected,
        recall_col="cov_r_mean",
        precision_col="cov_p_mean",
        ylabel="Coverage",
        title="Coverage",
    )
    assert chart.to_dict()["mark"]["type"] == "line"
    recovery = charts.with_plot_labels(
        pd.DataFrame(
            [
                {
                    "label": "loqi_druglike",
                    "display_label": "LOQI",
                    "generator": "LOQI",
                    "threshold": 0.5,
                    "cov_r_mean": 0.4,
                    "cov_p_mean": 0.2,
                },
                {
                    "label": "loqi_druglike",
                    "display_label": "LOQI",
                    "generator": "LOQI",
                    "threshold": 1.0,
                    "cov_r_mean": 0.9,
                    "cov_p_mean": 0.5,
                },
            ]
        ),
        ["loqi_druglike"],
    )
    drawn = charts.recovery_chart(recovery).to_dict()
    assert drawn["mark"]["type"] == "line"
    assert drawn["encoding"]["x"]["scale"]["domain"] == [0, 1]
