from __future__ import annotations

import importlib.util
import pickle
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from casf_benchmark.analysis.druglike_covmat import per_molecule_metrics
from casf_benchmark.analysis.druglike_k_efficiency import (
    DEFAULT_PLOT_LABELS,
    filter_plot_summary,
    k_rows_for_method,
    plot_coverage,
    plot_coverage_median,
    plot_mat,
    plot_mat_median,
    resolve_generation_pickle,
    stable_seed,
    subsample_columns,
)
from casf_benchmark.paths import REPO_ROOT


def load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_parts_preserves_druglike_order(tmp_path: Path, monkeypatch) -> None:
    inference = load_script("run_off_the_shelf_druglike")
    monkeypatch.setattr(inference, "normalize_conformers", lambda mols: mols)
    druglike = tmp_path / "druglike.pickle"
    with druglike.open("wb") as handle:
        pickle.dump(OrderedDict([("CC", {}), ("CO", {})]), handle)

    for index, (smiles, conformers) in enumerate((("CC", ["a"]), ("CO", ["b", "c"]))):
        path = inference.part_path(tmp_path / "run", index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(OrderedDict([(smiles, conformers)]), handle)

    destination = inference.merge_parts(druglike, tmp_path / "run")
    with destination.open("rb") as handle:
        merged = pickle.load(handle)
    assert list(merged) == ["CC", "CO"]
    assert merged["CO"] == ["b", "c"]


def test_covmat_metrics_have_expected_recall_and_precision() -> None:
    covmat = load_script("eval_druglike_covmat")
    metrics = covmat.per_molecule_metrics(
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        threshold=0.75,
        dmax=3.0,
    )
    assert metrics["cov_r_075"] == 0.5
    assert metrics["cov_p_075"] == 0.5
    assert metrics["mat_r"] == 1.0
    assert metrics["mat_p"] == 0.5
    assert metrics["cmat_r"] == 1.0
    assert metrics["cmat_p"] == 0.5


def test_full_k_reproduces_per_molecule_metrics() -> None:
    matrix = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    expected = per_molecule_metrics(matrix, threshold=0.75, dmax=3.0)
    per_mol, summary = k_rows_for_method({"CC": matrix}, [2], n_seeds=3, seed=0)
    assert per_mol[0]["cov_r_075"] == expected["cov_r_075"]
    assert per_mol[0]["cov_p_075"] == expected["cov_p_075"]
    assert per_mol[0]["mat_r"] == expected["mat_r"]
    assert per_mol[0]["mat_p"] == expected["mat_p"]
    assert summary[0]["cov_r_mean"] == 0.5
    assert summary[0]["cov_p_mean"] == 0.5
    assert summary[0]["mat_r_mean"] == 1.0
    assert summary[0]["mat_p_mean"] == 0.5
    assert summary[0]["cov_r_median"] == 0.5
    assert summary[0]["mat_r_median"] == 1.0
    assert summary[0]["n_seeds"] == 3
    assert summary[0]["n_molecules"] == 1


def test_k1_and_k2_on_known_matrix() -> None:
    matrix = np.asarray([[0.0, 1.0, 0.4], [2.0, 3.0, 0.1]], dtype=float)
    col0 = matrix[:, :1]
    per_mol, summary = k_rows_for_method({"CC": col0}, [1, 2], n_seeds=3, seed=0)
    by_k = {int(row["k"]): row for row in summary}
    assert by_k[1]["cov_r_mean"] == 0.5
    assert by_k[1]["cov_p_mean"] == 1.0
    assert by_k[1]["mat_r_mean"] == 1.0
    assert by_k[1]["mat_p_mean"] == 0.0
    assert all(row["n_sampled"] == 1 for row in per_mol)

    k2 = k_rows_for_method({"CC": matrix}, [3], n_seeds=3, seed=0)[1][0]
    expected = per_molecule_metrics(matrix, threshold=0.75, dmax=3.0)
    assert k2["cov_r_mean"] == pytest.approx(expected["cov_r_075"])
    assert k2["cov_p_mean"] == pytest.approx(expected["cov_p_075"])
    assert k2["mat_r_mean"] == pytest.approx(expected["mat_r"])
    assert k2["mat_p_mean"] == pytest.approx(expected["mat_p"])


def test_three_seeds_are_averaged() -> None:
    matrix = np.asarray([[0.0, 1.0, 0.4], [2.0, 3.0, 0.1]], dtype=float)
    smiles = "CC"
    k = 1
    n_seeds = 3
    seed = 0
    expected = []
    for replicate in range(n_seeds):
        rng = np.random.default_rng(stable_seed(seed, replicate, k, smiles))
        sampled = subsample_columns(matrix, k, rng)
        expected.append(per_molecule_metrics(sampled, 0.75, 3.0)["cov_r_075"])
    _, summary = k_rows_for_method({smiles: matrix}, [k], n_seeds=n_seeds, seed=seed)
    assert summary[0]["cov_r_mean"] == pytest.approx(float(np.mean(expected)))


def test_k_larger_than_pool_uses_full_matrix() -> None:
    matrix = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    sampled = subsample_columns(matrix, 10, np.random.default_rng(0))
    np.testing.assert_array_equal(sampled, matrix)
    per_mol, _ = k_rows_for_method({"CC": matrix, "CO": matrix[:, :1]}, [2], n_seeds=3, seed=0)
    by_smiles = {row["smiles"]: row for row in per_mol}
    assert by_smiles["CC"]["n_sampled"] == 2
    assert by_smiles["CO"]["n_sampled"] == 1
    assert by_smiles["CO"]["cov_p_075"] == 1.0


def test_resolve_generation_pickle_accepts_absolute_and_relative(tmp_path: Path) -> None:
    relative = resolve_generation_pickle("run_a", tmp_path)
    assert relative == tmp_path / "run_a" / "generation_results.pickle"
    absolute = resolve_generation_pickle("/mnt/weka/example/loqi_druglike", tmp_path)
    assert absolute == Path("/mnt/weka/example/loqi_druglike") / "generation_results.pickle"


def test_molecule_median_differs_from_mean() -> None:
    zero = np.asarray([[1.0], [2.0]], dtype=float)  # COV-R = 0
    half = np.asarray([[0.0], [2.0]], dtype=float)  # COV-R = 0.5
    _, summary = k_rows_for_method(
        {"a": zero, "b": half, "c": half},
        [1],
        n_seeds=1,
        seed=0,
    )
    assert summary[0]["cov_r_mean"] == pytest.approx(1.0 / 3.0)
    assert summary[0]["cov_r_median"] == pytest.approx(0.5)


def test_filter_plot_summary_keeps_requested_methods() -> None:
    import pandas as pd

    summary = pd.DataFrame(
        [
            {"label": "qwen_1p7b_fsq_bigdata_step47023", "generator": "Qwen", "model_size": "1.7B", "tokenizer": "FSQ", "k": 10, "cov_r_mean": 0.9},
            {"label": "qwen_0p6b_bigdata_step111000", "generator": "Qwen", "model_size": "0.6B", "tokenizer": "Binned", "k": 10, "cov_r_mean": 0.1},
            {"label": "loqi_druglike", "generator": "LOQI", "display_label": "LOQI", "k": 10, "cov_r_mean": 0.8},
        ]
    )
    filtered = filter_plot_summary(summary)
    assert set(filtered["label"]) == {"qwen_1p7b_fsq_bigdata_step47023", "loqi_druglike"}
    assert "Qwen 1.7B FSQ" in set(filtered["plot_label"])
    assert set(DEFAULT_PLOT_LABELS) >= set(filtered["label"])


def test_k_plots_write_pngs(tmp_path: Path) -> None:
    import pandas as pd

    summary = pd.DataFrame(
        [
            {
                "label": "loqi_druglike",
                "display_label": "LOQI",
                "generator": "LOQI",
                "k": 10,
                "cov_r_mean": 0.2,
                "cov_p_mean": 0.8,
                "mat_r_mean": 1.2,
                "mat_p_mean": 0.4,
                "cov_r_median": 0.15,
                "cov_p_median": 0.75,
                "mat_r_median": 1.1,
                "mat_p_median": 0.35,
            },
            {
                "label": "loqi_druglike",
                "display_label": "LOQI",
                "generator": "LOQI",
                "k": 100,
                "cov_r_mean": 0.5,
                "cov_p_mean": 0.6,
                "mat_r_mean": 0.9,
                "mat_p_mean": 0.5,
                "cov_r_median": 0.45,
                "cov_p_median": 0.55,
                "mat_r_median": 0.85,
                "mat_p_median": 0.48,
            },
        ]
    )
    coverage = tmp_path / "coverage.png"
    mat = tmp_path / "mat.png"
    coverage_median = tmp_path / "coverage_median.png"
    mat_median = tmp_path / "mat_median.png"
    plot_coverage(summary, coverage)
    plot_mat(summary, mat)
    plot_coverage_median(summary, coverage_median)
    plot_mat_median(summary, mat_median)
    for path in (coverage, mat, coverage_median, mat_median):
        assert path.is_file() and path.stat().st_size > 0
