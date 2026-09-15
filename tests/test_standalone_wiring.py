"""Standalone wiring checks for repo-local assets and imports."""

from __future__ import annotations

import importlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from casf_benchmark.paths import (
    BUNDLED_CHEMBL_MAP_CSV,
    BUNDLED_CHEMBL_REF_MAP_CSV,
    CONFIG_ROOT,
    DEFAULT_ANALYSIS_SOURCES_CONFIG,
    DEFAULT_ANALYSIS_SOURCES_WEKA_CONFIG,
    DEFAULT_DASHBOARD_DB,
    DEFAULT_EXTENDED_DB,
    DEFAULT_GENERATION_FAMILIES_CONFIG,
    DEFAULT_MASTER_CSV,
    DEFAULT_PER_LIGAND_LONG_CSV,
    DEFAULT_RUNS_ROOT,
    GENERATION_SCRIPT,
    REPO_ROOT,
)

SUBMODULES = (
    "casf_benchmark.analysis.dataset",
    "casf_benchmark.analysis.metrics",
    "casf_benchmark.catalog",
    "casf_benchmark.chembl3d.identity",
    "casf_benchmark.chembl3d.loader",
    "casf_benchmark.cli.analyze_conformer_sets",
    "casf_benchmark.cli.build_dashboard_db",
    "casf_benchmark.cli.extended_analysis",
    "casf_benchmark.generation.conformer_sets",
    "casf_benchmark.generation.normalizer",
    "casf_benchmark.paths",
    "casf_benchmark.stratum_bins",
)

ENTRY_POINTS = (
    "casf-analyze-conformer-sets",
    "casf-build-dashboard-db",
    "casf-build-master-csv",
    "casf-generate-conformer-sets",
)


def test_all_submodules_import_cleanly() -> None:
    for name in SUBMODULES:
        importlib.import_module(name)


def test_packaged_config_exists() -> None:
    for path in (
        CONFIG_ROOT / "casf_analysis_sources.yaml",
        CONFIG_ROOT / "casf_analysis_sources.weka.yaml",
        CONFIG_ROOT / "casf_generation_families.yaml",
    ):
        assert path.is_file(), f"missing packaged config: {path}"


def test_bundled_assets_exist() -> None:
    required = (
        BUNDLED_CHEMBL_MAP_CSV,
        BUNDLED_CHEMBL_REF_MAP_CSV,
        DEFAULT_DASHBOARD_DB,
        DEFAULT_EXTENDED_DB,
        DEFAULT_MASTER_CSV,
        DEFAULT_PER_LIGAND_LONG_CSV,
        DEFAULT_ANALYSIS_SOURCES_CONFIG,
        DEFAULT_ANALYSIS_SOURCES_WEKA_CONFIG,
        DEFAULT_GENERATION_FAMILIES_CONFIG,
        GENERATION_SCRIPT,
    )
    for path in required:
        assert path.is_file(), f"missing bundled asset: {path}"


def test_bundled_analysis_sources_resolve() -> None:
    cfg = yaml.safe_load(DEFAULT_ANALYSIS_SOURCES_CONFIG.read_text(encoding="utf-8"))
    missing: list[str] = []
    for section in ("sources", "reference_sources"):
        for entry in cfg.get(section, []):
            root = REPO_ROOT / entry["root"]
            csv_path = root / "analysis/tables/geometric_per_ligand_long.csv"
            if not csv_path.is_file():
                missing.append(str(csv_path))
    assert not missing, "missing bundled run CSVs:\n" + "\n".join(missing)


def test_dashboard_sqlite_has_core_tables() -> None:
    with sqlite3.connect(DEFAULT_DASHBOARD_DB) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"comparison_rows", "comparison_strata", "per_ligand_long"}.issubset(tables)


def test_console_scripts_are_on_path() -> None:
    for command in ENTRY_POINTS:
        assert shutil.which(command), f"missing console script: {command}"


def test_console_scripts_respond_to_help() -> None:
    for command in ENTRY_POINTS:
        proc = subprocess.run([command, "--help"], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"{command} --help failed:\n{proc.stderr}"


def test_script_wrappers_delegate_without_pythonpath() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_casf_analysis_master_csv.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**dict(__import__("os").environ.items()), "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_no_molgen3d_imports_in_source() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "import molgen3D" in text or "from molgen3D" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"molgen3D imports found: {offenders}"


def test_bundled_run_count_matches_config() -> None:
    cfg = yaml.safe_load(DEFAULT_ANALYSIS_SOURCES_CONFIG.read_text(encoding="utf-8"))
    configured = len(cfg.get("sources", [])) + len(cfg.get("reference_sources", []))
    bundled = len(list(DEFAULT_RUNS_ROOT.iterdir()))
    assert bundled == configured
