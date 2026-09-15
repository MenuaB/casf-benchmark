#!/usr/bin/env python3
"""Streamlit dashboard for PB-once CASF analysis outputs."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from casf_benchmark.paths import DEFAULT_DASHBOARD_DB, DEFAULT_EXTENDED_DB as _DEFAULT_EXTENDED_DB


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sibling_module(module_name: str):
    """Load a .py sitting next to this Streamlit entrypoint (Cloud-safe)."""
    return _load_module_from_path(
        module_name, Path(__file__).resolve().parent / f"{module_name}.py"
    )


def _load_repo_release_data():
    """Load release_data.py from this checkout, not a stale Cloud wheel.

    Community Cloud often keeps an older `casf_benchmark` install after git pull.
    The checkout copy has the /tmp fetch and `latest` URL alias; the wheel may not.
    """
    path = Path(__file__).resolve().parents[2] / "src" / "casf_benchmark" / "release_data.py"
    if path.is_file():
        return _load_module_from_path("_casf_release_data_src", path)
    from casf_benchmark import release_data as packaged

    return packaged


release_data = _load_repo_release_data()


render_table_help = _load_sibling_module("table_help").render_table_help
render_report_analysis = _load_sibling_module("report_analysis_charts").render_report_analysis
render_druglike_k_charts = _load_sibling_module("druglike_k_charts").render_druglike_k_charts
render_threshold_charts = _load_sibling_module("threshold_charts").render_threshold_charts

_DEFAULT_DB = DEFAULT_DASHBOARD_DB
DEFAULT_DB = Path(os.environ.get("CASF_DASHBOARD_DB", str(_DEFAULT_DB)))
DEFAULT_EXTENDED_DB = Path(os.environ.get("CASF_EXTENDED_DB", str(_DEFAULT_EXTENDED_DB)))
TIERS = ("fixed", "dynamic", "chembl_count")
QWEN_DASHBOARD_LABELS = {
    "qwen_0p6b_bigdata": "Qwen 0.6B bigdata",
    "qwen_0p6b_bigdata_to_revisited": "Qwen 0.6B bigdata→revisited",
    "qwen_0p6b_fsq": "Qwen 0.6B fsq",
    "qwen_0p6b_fsq_bigdata_pretrain": "Qwen 0.6B fsq+bigdata-pretrain",
    "qwen_1p7b_bigdata": "Qwen 1.7B bigdata",
    "qwen_1p7b_bigdata_to_revisited": "Qwen 1.7B bigdata→revisited",
    "qwen_1p7b_fsq": "Qwen 1.7B fsq",
    "qwen_1p7b_fsq_bigdata_pretrain": "Qwen 1.7B fsq+bigdata-pretrain",
    "qwen_1p7b_revisited": "Qwen 1.7B revisited",
    "qwen_4b_bigdata": "Qwen 4B bigdata",
    "qwen_4b_revisited": "Qwen 4B revisited",
}
BREAKDOWN_LABELS = {
    "Total": "total",
    "Rotatable bonds": "rotatable_bonds",
    "Heavy atoms": "heavy_atoms",
}
EXTENDED_TABLES = {
    "Paired vs ChEMBL3D-PB": {
        "table": "extended_paired_delta_vs_chembl3d_pb",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "stratum_type",
            "stratum",
            "n_common_ligands",
            "mean_delta_casf_best_rmsd",
            "median_delta_casf_best_rmsd",
            "win_rate_0p1A",
            "loss_rate_0p1A",
            "delta_hit_0p5",
            "delta_hit_0p75",
            "delta_hit_2p0",
            "catastrophic_loss_count_gt1A",
            "large_rescue_count_gt1A",
        ],
    },
    "Extremes vs ChEMBL3D-PB": {
        "table": "extended_paired_delta_extreme_ligands_vs_chembl3d_pb",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "baseline_source",
            "direction",
            "mol_id",
            "baseline_casf_best_rmsd",
            "method_casf_best_rmsd",
            "delta_casf_best_rmsd",
            "baseline_casf_hit_0p75",
            "method_casf_hit_0p75",
        ],
    },
    "Paired vs RDKit raw": {
        "table": "extended_paired_delta_vs_rdkit_raw",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "stratum_type",
            "stratum",
            "n_common_ligands",
            "mean_delta_casf_best_rmsd",
            "median_delta_casf_best_rmsd",
            "win_rate_0p1A",
            "loss_rate_0p1A",
            "delta_hit_0p5",
            "delta_hit_0p75",
            "delta_hit_2p0",
            "catastrophic_loss_count_gt1A",
            "large_rescue_count_gt1A",
        ],
    },
    "Extremes vs RDKit raw": {
        "table": "extended_paired_delta_extreme_ligands_vs_rdkit_raw",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "baseline_source",
            "direction",
            "mol_id",
            "baseline_casf_best_rmsd",
            "method_casf_best_rmsd",
            "delta_casf_best_rmsd",
            "baseline_casf_hit_0p75",
            "method_casf_hit_0p75",
        ],
    },
    "Frontier": {
        "table": "extended_frontier_summary",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "n_ligands",
            "mean_casf_hit_0p75",
            "mean_casf_best_rmsd",
            "mean_pb_pass_rate",
            "mean_conformer_count",
            "mean_greedy_clusters_1p0",
            "useful_clusters_per_100_confs",
            "hit_0p75_per_100_confs",
            "valid_hit_score",
            "frontier_rank",
        ],
    },
    "PB failures": {
        "table": "extended_pb_failure_mechanisms",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "n_ligands",
            "total_pb_fail_rate",
            "median_per_ligand_pb_pass_rate",
            "ligands_pb_pass_rate_lt_0p9",
            "ligands_pb_pass_rate_lt_0p5",
            "dominant_pb_failure_test",
            "second_dominant_pb_failure_test",
            "fraction_failures_tetrahedral_chirality",
            "fraction_failures_internal_steric_clash",
            "fraction_failures_energy_ratio",
            "fraction_failures_bond_geometry_tests",
        ],
    },
    "PB tail ligands": {
        "table": "extended_pb_failure_tail_ligands",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "mol_id",
            "rotatable_bonds",
            "heavy_atoms",
            "pb_pass_rate",
            "pb_input_confs",
            "top_three_failed_pb_tests",
            "casf_best_rmsd",
            "casf_hit_0p75",
        ],
    },
    "Difficulty": {
        "table": "extended_bound_pose_difficulty",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "baseline_source",
            "difficulty_bin",
            "n_common_ligands",
            "mean_delta_casf_best_rmsd",
            "hit_0p75",
            "rescue_rate",
            "catastrophic_loss_rate",
        ],
    },
    "Rescue cases": {
        "table": "extended_rescue_cases",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "mol_id",
            "rotatable_bonds",
            "heavy_atoms",
            "baseline_casf_best_rmsd",
            "method_casf_best_rmsd",
            "delta_casf_best_rmsd",
            "pb_pass_rate",
            "greedy_clusters_1p0",
        ],
    },
    "K efficiency": {
        "table": "extended_k_efficiency",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "sampling_mode",
            "summary_mode",
            "k",
            "status",
            "skip_reason",
            "n_ligands",
            "n_metric_ligands",
            "n_ligands_with_pb",
            "mean_n_used",
            "mean_best_rmsd_at_k",
            "hit_0p25_at_k",
            "hit_0p75_at_k",
            "hit_2p0_at_k",
            "ci_low",
            "ci_high",
            "mean_pb_pass_rate_at_k",
            "mean_conformers_available",
            "n_ligands_with_at_least_k",
        ],
    },
    "K per ligand": {
        "table": "extended_chembl_k_efficiency_per_ligand",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "sampling_mode",
            "seed",
            "k",
            "mol_id",
            "n_available_pb",
            "n_used",
            "casf_best_rmsd_at_k",
            "casf_hit_0p25_at_k",
            "casf_hit_0p5_at_k",
            "casf_hit_0p75_at_k",
            "casf_hit_2p0_at_k",
        ],
    },
    "K comparisons": {
        "table": "extended_k_efficiency_comparisons",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "comparison",
            "baseline_source",
            "baseline_sampling_mode",
            "k",
            "n_common_ligands",
            "mean_delta_best_rmsd_at_k",
            "median_delta_best_rmsd_at_k",
            "delta_hit_0p25_at_k",
            "delta_hit_0p5_at_k",
            "delta_hit_0p75_at_k",
            "delta_hit_2p0_at_k",
        ],
    },
    "Energy windows": {
        "table": "extended_energy_window_summary",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "energy_window",
            "n_ligands",
            "n_ligands_with_window_confs",
            "mean_fraction_window_confs",
            "mean_n_window_confs",
            "mean_best_rmsd_window",
            "hit_0p25_window",
            "hit_0p5_window",
            "hit_0p75_window",
            "hit_2p0_window",
            "mean_clusters_1p0_window",
            "mean_clusters_per_100_1p0_window",
            "mean_entropy_1p0_window",
            "mean_largest_cluster_fraction_1p0_window",
            "hit_retention_deltaE_10",
            "cluster_retention_deltaE_10",
            "mean_useful_low_energy_clusters",
        ],
    },
    "Energy per ligand": {
        "table": "extended_energy_window_per_ligand",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "mol_id",
            "energy_window",
            "n_pb_confs",
            "n_window_confs",
            "fraction_window_confs",
            "window_energy_min",
            "window_energy_max_delta",
            "casf_best_rmsd_window",
            "casf_hit_0p25_window",
            "casf_hit_0p5_window",
            "casf_hit_0p75_window",
            "casf_hit_2p0_window",
            "greedy_clusters_1p0_window",
            "clusters_per_100_1p0_window",
            "cluster_entropy_1p0_window",
            "largest_cluster_fraction_1p0_window",
            "useful_low_energy_clusters",
        ],
    },
    "Energy comparisons": {
        "table": "extended_energy_window_comparisons",
        "columns": [
            "ligand_set",
            "source",
            "family",
            "tier",
            "method",
            "comparison",
            "n_ligands",
            "delta_hit_0p75_window",
            "high_energy_hit_fraction",
        ],
    },
    "K RMSD manifest": {
        "table": "extended_per_conformer_casf_rmsd_manifest",
        "columns": [
            "ligand_set",
            "run_id",
            "run_label",
            "source",
            "family",
            "tier",
            "method",
            "mol_id",
            "conformers_available",
            "rotatable_bonds",
            "heavy_atoms",
        ],
    },
    "Sanity": {
        "table": "extended_sanity_checks",
        "columns": ["check", "severity", "status", "affected_rows", "details"],
    },
}

# Separate from Extended Analysis: one row per checkpoint/molecule, not the
# ligand_set/tier/family stratification those filters target.
DRUGLIKE_TABLES: dict[str, dict[str, object]] = {
    "Druglike summary": {
        "table": "extended_druglike_summary",
        "columns": [
            "label",
            "display_label",
            # Present once the table is rebuilt for a non-Qwen generator; harmless
            # while absent, since display_table() keeps only the columns that exist.
            "generator",
            "model_size",
            "tokenizer",
            "recipe",
            "step",
            "n_molecules",
            "total_confs",
            "overall_pb_pass_rate",
            "mean_per_molecule_pb_pass_rate",
            "cov_r_mean",
            "cov_r_median",
            "cov_p_mean",
            "cov_p_median",
            "mat_r_mean",
            "mat_r_median",
            "mat_p_mean",
            "mat_p_median",
            "molecule_success_rate",
            "total_true_confs",
        ],
    },
    "Druglike per-molecule": {
        "table": "extended_druglike_per_molecule",
        "columns": [
            "label",
            "name",
            "category",
            "num_heavy_atoms",
            "rotatable_bonds",
            "mw",
            "n_generated",
            "n_pb_pass",
            "pb_pass_rate",
            "cov_r_075",
            "cov_p_075",
            "mat_r",
            "mat_p",
            "num_true_confs",
            "div_mean_torsion_std_deg",
            "div_greedy_clusters_1p0",
            "div_cluster_entropy_1p0",
            "energy_energy_median",
        ],
    },
}

TABLE_HEIGHT_PX = 520
# Shorter than Extended Analysis so the druglike block sits on the first screen
# instead of only its column headers peeking under the comparison grid.
COMPARISON_TABLE_HEIGHT_PX = 320
DRUGLIKE_TABLE_HEIGHT_PX = 420

# Serialises the release fetch: Streamlit may run several script threads at once
# (a browser refresh mid-download is enough), and two of them writing the same
# `.part` file would corrupt it.
_FETCH_LOCK = threading.Lock()

# Kept in the Streamlit entrypoint so Community Cloud picks it up even when an
# older editable install of casf-benchmark is still cached in the runtime image.
# Do not read new attributes off `release_data` here: Cloud crashed on
# `release_data.LATEST_TAG` while still serving the pre-latest package.
# Stale `release_tag()` still returns ots-v2; force a concrete tag that the
# old `.../releases/download/{tag}/...` URL can fetch (the `latest` alias
# exists only in a newer package).
_LEGACY_RELEASE_TAGS = frozenset(
    {
        "dashboard-data-qwen-druglike",
        "dashboard-data-druglike-ots-v1",
        "dashboard-data-druglike-ots-v2",
    }
)
_FORCED_RELEASE_TAG = "dashboard-data-druglike-k-v1"


def effective_release_tag() -> str:
    tag = release_data.release_tag()
    if tag in _LEGACY_RELEASE_TAGS:
        return _FORCED_RELEASE_TAG
    return tag


def release_pin_path() -> Path:
    return DEFAULT_DASHBOARD_DB.parent / ".dashboard_release_pin"


def read_release_pin() -> str | None:
    path = release_pin_path()
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def mark_release_assets_current() -> None:
    # Pin after the first asset lands so a rerun does not wipe it as stale.
    path = release_pin_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{effective_release_tag()}\n", encoding="utf-8")


def invalidate_stale_release_assets() -> None:
    tag = effective_release_tag()
    pin = read_release_pin()
    if pin is None or pin == tag:
        return
    for asset_path in release_data.RELEASE_ASSET_PATHS:
        if asset_path.exists() and release_data.is_release_asset(asset_path):
            asset_path.unlink()
    release_pin_path().unlink(missing_ok=True)


def ensure_db_available(path: Path) -> None:
    """Download a missing dashboard DB from the pinned GitHub Release.

    A no-op when the file is already on disk -- a Weka checkout or a local rebuild
    keeps working exactly as before -- and when `path` is not one of the default
    locations, so a DB the operator pointed us at is never overwritten. This is
    what lets a fresh Streamlit Cloud clone, which has no DB at all, serve the
    dashboard. See `casf_benchmark.release_data` for the env pins.
    """
    if not release_data.is_release_asset(path):
        return
    if path.exists():
        mark_release_assets_current()
        return

    tag = effective_release_tag()
    status = st.empty()
    bar = st.progress(0.0)

    def on_progress(downloaded: int, total: int) -> None:
        bar.progress(min(downloaded / total, 1.0) if total else 0.0)
        total_text = f" / {total / 1e6:.0f} MB" if total else ""
        status.caption(
            f"Fetching {path.name} from release `{tag}` "
            f"({downloaded / 1e6:.0f}{total_text})"
        )

    try:
        with _FETCH_LOCK:
            # Another script thread may have completed the download while we waited.
            if release_data.fetch_release_asset(
                path, tag=tag, repo=release_data.release_repo(), progress=on_progress
            ):
                mark_release_assets_current()
            elif path.exists():
                mark_release_assets_current()
    except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
        if path.exists():
            mark_release_assets_current()
        else:
            st.error(
                f"Could not fetch {path.name} from release `{tag}` "
                f"of {release_data.release_repo()}: {error}"
            )
    finally:
        bar.empty()
        status.empty()


def _families_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "casf_benchmark" / "config" / "casf_generation_families.yaml"


def hidden_on_core_family_ids() -> frozenset[str]:
    """Read hide_on_core flags from this checkout, not a stale Cloud wheel."""
    path = _families_config_path()
    if not path.is_file():
        return frozenset()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return frozenset(
        str(entry["id"])
        for entry in data.get("families", [])
        if isinstance(entry, dict) and entry.get("hide_on_core")
    )


def drop_hidden_core_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Omit hide_on_core families for core rows; sqlite/ref/druglike stay intact."""
    if frame.empty or "family" not in frame.columns or "ligand_set" not in frame.columns:
        return frame
    hidden = hidden_on_core_family_ids()
    if not hidden:
        return frame
    keep = ~(
        (frame["ligand_set"].astype(str) == "core")
        & frame["family"].astype(str).isin(hidden)
    )
    return frame.loc[keep].reset_index(drop=True)


def filter_public_methods(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the public Qwen shortlist while preserving every non-Qwen row."""

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    family = (
        out["family"].fillna("").astype(str)
        if "family" in out.columns
        else pd.Series("", index=out.index, dtype=str)
    )
    is_qwen = family.str.startswith("qwen_")
    for column in ("display_label", "method", "source"):
        if column in out.columns:
            is_qwen |= out[column].fillna("").astype(str).str.contains(
                r"qwen",
                case=False,
                regex=True,
            )
    out = out[~is_qwen | family.isin(QWEN_DASHBOARD_LABELS)].copy()
    if "display_label" in out.columns and "family" in out.columns:
        public_labels = out["family"].map(QWEN_DASHBOARD_LABELS)
        has_public_label = public_labels.notna()
        out.loc[has_public_label, "display_label"] = public_labels[has_public_label]
    return out


@st.cache_data(show_spinner=False)
def load_table(db_path: str, table: str, db_mtime_ns: int) -> pd.DataFrame:
    del db_mtime_ns
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    return filter_public_methods(frame)


@st.cache_data(show_spinner=False)
def load_table_names(db_path: str, db_mtime_ns: int) -> set[str]:
    del db_mtime_ns
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    return {str(row[0]) for row in rows}


def select_view(frame: pd.DataFrame, ligand_set: str, tier: str, family: str) -> pd.DataFrame:
    out = frame[frame["ligand_set"].astype(str) == ligand_set].copy()
    if tier != "All":
        if "view_tier" in out.columns:
            out = out[out["view_tier"].astype(str) == tier]
        else:
            out = out[
                ((out["row_type"].astype(str) == "generation") & (out["tier"].astype(str) == tier))
                | (out["row_type"].astype(str) == "reference")
            ]
    if family != "All":
        out = out[out["family"].astype(str) == family]
    return sort_rows(out)


def sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_row_type_order"] = out["row_type"].map({"generation": 0, "reference": 1}).fillna(9)
    out["_tier_order"] = out["tier"].map({"fixed": 0, "dynamic": 1, "chembl_count": 2, "reference": 3}).fillna(9)
    sort_cols = [col for col in ("stratum", "_row_type_order", "family", "_tier_order", "method") if col in out.columns]
    out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=["_row_type_order", "_tier_order"]).reset_index(drop=True)


def _build_column_config(frame: pd.DataFrame, columns: list[str]) -> dict[str, st.column_config.Column]:
    """Column display config: pin method, format numbers to three decimals."""
    config: dict[str, st.column_config.Column] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if column == "method":
            config[column] = st.column_config.TextColumn(column, pinned=True, width="large")
        elif pd.api.types.is_bool_dtype(series):
            config[column] = st.column_config.CheckboxColumn(column)
        elif pd.api.types.is_integer_dtype(series):
            config[column] = st.column_config.NumberColumn(column, format="%d")
        elif pd.api.types.is_float_dtype(series):
            config[column] = st.column_config.NumberColumn(column, format="%.3f")
    return config


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in frame.columns
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def _apply_color_scale(frame: pd.DataFrame, columns: list[str]):
    numeric = _numeric_columns(frame, columns)
    if not numeric:
        return frame
    return frame.style.background_gradient(cmap="viridis", subset=numeric, axis=0)


def _apply_table_view_controls(data: pd.DataFrame, cols: list[str], name: str) -> tuple[pd.DataFrame, list[str], bool]:
    """Let users hide columns and filter rows before rendering."""
    with st.expander("Columns & rows", expanded=False):
        visible = st.multiselect(
            "Visible columns",
            cols,
            default=cols,
            key=f"{name}_visible_cols",
        )
        visible = [column for column in cols if column in visible]
        if not visible:
            st.warning("Select at least one column.")
            visible = cols

        filtered = data
        if "method" in filtered.columns:
            methods = sorted(filtered["method"].dropna().astype(str).unique())
            if methods:
                selected_methods = st.multiselect(
                    "Methods (rows)",
                    methods,
                    default=[],
                    key=f"{name}_methods",
                    help="Leave empty to show every method.",
                )
                if selected_methods:
                    filtered = filtered[filtered["method"].astype(str).isin(selected_methods)]
        elif "display_label" in filtered.columns:
            labels = sorted(filtered["display_label"].dropna().astype(str).unique())
            if labels:
                selected_labels = st.multiselect(
                    "Labels (rows)",
                    labels,
                    default=[],
                    key=f"{name}_display_labels",
                    help="Leave empty to show every label.",
                )
                if selected_labels:
                    filtered = filtered[filtered["display_label"].astype(str).isin(selected_labels)]

        if "stratum" in filtered.columns and filtered["stratum"].nunique() > 1:
            strata = sorted(filtered["stratum"].dropna().astype(str).unique())
            selected_strata = st.multiselect(
                "Strata (rows)",
                strata,
                default=[],
                key=f"{name}_strata",
                help="Leave empty to show every stratum.",
            )
            if selected_strata:
                filtered = filtered[filtered["stratum"].astype(str).isin(selected_strata)]

        if "mol_id" in filtered.columns:
            mol_filter = st.text_input(
                "Filter mol_id (substring)",
                key=f"{name}_mol_id_filter",
            )
            if mol_filter.strip():
                filtered = filtered[
                    filtered["mol_id"].astype(str).str.contains(mol_filter.strip(), case=False, na=False)
                ]

        color_by_value = st.checkbox(
            "Color numeric columns by value",
            value=False,
            key=f"{name}_color_scale",
            help="Heatmap per column (low → high). Sorting, pinning, and CSV export are unchanged.",
        )

    return filtered.reset_index(drop=True), visible, color_by_value


def display_table(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
    *,
    height: int = TABLE_HEIGHT_PX,
) -> None:
    cols = [column for column in columns if column in frame.columns]
    color_by_value = False
    if not cols:
        data = frame
        visible_cols = list(data.columns)
    else:
        data, visible_cols, color_by_value = _apply_table_view_controls(frame[cols], cols, name)

    display_data = _apply_color_scale(data, visible_cols) if color_by_value else data
    st.dataframe(
        display_data,
        column_config=_build_column_config(data, visible_cols),
        column_order=visible_cols,
        use_container_width=True,
        height=height,
        hide_index=True,
    )
    st.download_button(
        f"Download {name} CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )
    render_table_help(name)


def filter_extended_table(
    frame: pd.DataFrame,
    ligand_set: str,
    tier: str,
    family: str,
    stratum_type: str,
) -> pd.DataFrame:
    out = drop_hidden_core_rows(frame.copy())
    if ligand_set != "All" and "ligand_set" in out.columns:
        out = out[out["ligand_set"].astype(str) == ligand_set]
    if tier != "All" and "tier" in out.columns:
        out = out[out["tier"].astype(str) == tier]
    if family != "All" and "family" in out.columns:
        out = out[out["family"].astype(str) == family]
    if stratum_type != "All" and "stratum_type" in out.columns:
        out = out[out["stratum_type"].astype(str) == stratum_type]
    return out.reset_index(drop=True)


def default_extended_db_path(db_path: Path, table_names: set[str]) -> Path:
    if any(name.startswith("extended_") for name in table_names):
        return db_path
    override = os.environ.get("CASF_EXTENDED_DB") or os.environ.get("CASF_EXTENDED_ANALYSIS_DB")
    return Path(override if override else str(DEFAULT_EXTENDED_DB))


def sidebar_extended_db_path(db_path: Path, table_names: set[str]) -> Path:
    default_path = default_extended_db_path(db_path, table_names)
    with st.sidebar.expander("Extended analysis", expanded=False):
        return Path(
            st.text_input("Extended DB", str(default_path), key="extended_db_path")
        ).expanduser()


def _preferred_extended_db_path(extended_db_path: Path, required_tables: set[str]) -> Path:
    """Fetch the sidecar if needed, then use the first DB that has the tables.

    Streamlit Cloud can land the main sqlite while the sidecar fetch fails or is
    interrupted. The sidebar then points at a file with no `extended_*` tables.
    """
    sidecar = Path(str(DEFAULT_EXTENDED_DB)).expanduser()
    ensure_db_available(extended_db_path)
    if sidecar != extended_db_path:
        ensure_db_available(sidecar)

    candidates = [extended_db_path]
    if sidecar not in candidates:
        candidates.append(sidecar)
    for path in candidates:
        if not path.exists():
            continue
        names = load_table_names(str(path), path.stat().st_mtime_ns)
        if any(name in names for name in required_tables):
            return path
    return extended_db_path


def render_druglike_tables(extended_db_path: Path) -> None:
    st.divider()
    st.header("Druglike conformer evaluation")

    required = {spec["table"] for spec in DRUGLIKE_TABLES.values()}
    extended_db_path = _preferred_extended_db_path(extended_db_path, required)
    if not extended_db_path.exists():
        st.info(f"Extended DB not found: {extended_db_path}")
        return

    extended_mtime_ns = extended_db_path.stat().st_mtime_ns
    extended_table_names = load_table_names(str(extended_db_path), extended_mtime_ns)
    available = [
        label
        for label, spec in DRUGLIKE_TABLES.items()
        if spec["table"] in extended_table_names
    ]
    if not available:
        st.info(f"No druglike tables found in {extended_db_path}")
        return

    tabs = st.tabs(available)
    for tab, label in zip(tabs, available):
        spec = DRUGLIKE_TABLES[label]
        with tab:
            frame = load_table(str(extended_db_path), spec["table"], extended_mtime_ns)
            display_table(
                frame,
                spec["columns"],
                spec["table"],
                height=DRUGLIKE_TABLE_HEIGHT_PX,
            )

    if "extended_druglike_k_efficiency" in extended_table_names:
        k_frame = load_table(
            str(extended_db_path),
            "extended_druglike_k_efficiency",
            extended_mtime_ns,
        )
        render_druglike_k_charts(k_frame)


def render_extended_analysis(extended_db_path: Path) -> None:
    st.divider()
    st.header("Extended Analysis")

    required = {spec["table"] for spec in EXTENDED_TABLES.values()}
    extended_db_path = _preferred_extended_db_path(extended_db_path, required)
    if not extended_db_path.exists():
        st.info(f"Extended DB not found: {extended_db_path}")
        return

    extended_mtime_ns = extended_db_path.stat().st_mtime_ns
    extended_table_names = load_table_names(str(extended_db_path), extended_mtime_ns)
    available = [
        label
        for label, spec in EXTENDED_TABLES.items()
        if spec["table"] in extended_table_names
    ]
    if not available:
        st.info(f"No extended_* tables found in {extended_db_path}")
        return

    first_available = drop_hidden_core_rows(
        load_table(
            str(extended_db_path),
            EXTENDED_TABLES[available[0]]["table"],
            extended_mtime_ns,
        )
    )
    ligand_sets = ["All"]
    if "ligand_set" in first_available.columns:
        ligand_sets.extend(sorted(first_available["ligand_set"].dropna().astype(str).unique()))

    filter_cols = st.columns(4)
    with filter_cols[0]:
        ligand_set = st.selectbox("Extended ligand set", ligand_sets, key="extended_ligand_set")
    with filter_cols[1]:
        tier = st.selectbox("Extended tier", ["All", *TIERS], key="extended_tier")
    with filter_cols[2]:
        family_source = first_available
        if ligand_set != "All" and "ligand_set" in family_source.columns:
            family_source = family_source[family_source["ligand_set"].astype(str) == ligand_set]
        families = ["All"]
        if "family" in family_source.columns:
            families.extend(sorted(family_source["family"].dropna().astype(str).unique()))
        family = st.selectbox("Extended family", families, key="extended_family")
    with filter_cols[3]:
        stratum_type = st.selectbox(
            "Extended stratum",
            ["All", "total", "rotatable_bond_bin", "heavy_atom_bin", "chembl_conformer_count_bin", "baseline_casf_difficulty_bin"],
            key="extended_stratum_type",
        )

    label = st.selectbox("Extended table", available, key="extended_table")
    spec = EXTENDED_TABLES[label]
    frame = load_table(str(extended_db_path), spec["table"], extended_mtime_ns)
    view = filter_extended_table(frame, ligand_set, tier, family, stratum_type)
    display_table(view, spec["columns"], spec["table"])


def main() -> None:
    st.set_page_config(page_title="CASF Analysis Dashboard", layout="wide")
    st.title("CASF Analysis Dashboard")

    db_default = Path(os.environ.get("CASF_DASHBOARD_DB", str(DEFAULT_DB)))
    db_path = Path(st.sidebar.text_input("Dashboard DB", str(db_default))).expanduser()
    invalidate_stale_release_assets()
    ensure_db_available(db_path)
    ensure_db_available(DEFAULT_EXTENDED_DB)
    if not db_path.exists():
        st.error(f"Dashboard DB not found: {db_path}")
        st.stop()
    st.sidebar.caption(f"Dashboard data release: `{effective_release_tag()}`")

    db_mtime_ns = db_path.stat().st_mtime_ns
    table_names = load_table_names(str(db_path), db_mtime_ns)
    comparison_rows = drop_hidden_core_rows(load_table(str(db_path), "comparison_rows", db_mtime_ns))
    comparison_strata = drop_hidden_core_rows(load_table(str(db_path), "comparison_strata", db_mtime_ns))
    per_ligand = drop_hidden_core_rows(load_table(str(db_path), "per_ligand_long", db_mtime_ns))

    ligand_sets = sorted(comparison_rows["ligand_set"].dropna().astype(str).unique())
    ligand_set = st.sidebar.selectbox("Ligand set", ligand_sets, index=0 if "core" not in ligand_sets else ligand_sets.index("core"))
    tier = st.sidebar.selectbox("Tier", ["All", *TIERS], index=0)
    family_source = comparison_rows[comparison_rows["ligand_set"].astype(str) == ligand_set]
    families = ["All", *sorted(family_source["family"].dropna().astype(str).unique())]
    family = st.sidebar.selectbox("Family", families)

    with st.sidebar.expander("Break down aggregates", expanded=False):
        breakdown_label = st.radio("View", list(BREAKDOWN_LABELS), index=0)
    breakdown = BREAKDOWN_LABELS[breakdown_label]

    if breakdown == "total":
        view = select_view(comparison_rows, ligand_set, tier, family)
    else:
        strata = comparison_strata[comparison_strata["breakdown"].astype(str) == breakdown]
        view = select_view(strata, ligand_set, tier, family)

    identity = [
        "stratum",
        "row_type",
        "display_label",
        "family",
        "tier",
        "method",
        "ligands",
        "ligands_scope",
    ]
    main_views = {
        "Overview": (
            "overview",
            [
                *identity,
                "total_confs",
                "mean_confs_per_ligand",
                "pairwise_mean",
                "pairwise_p90",
                "mean_torsion_std_deg",
            ],
        ),
        "Clustering": (
            "clustering",
            [
                *identity,
                "mean_clusters_0p5",
                "mean_clusters_1p0",
                "mean_clusters_2p0",
                "mean_clusters_3p0",
                "clusters_per_100_1p0",
                "cluster_entropy_1p0",
                "effective_clusters_1p0",
                "largest_cluster_fraction_1p0",
                "singleton_fraction_1p0",
            ],
        ),
        "Energy": (
            "energy",
            [*identity, "energy_min", "energy_max", "energy_median", "energy_std"],
        ),
        "CASF hits": (
            "casf_hits",
            [
                *identity,
                "casf_best_rmsd",
                "casf_median_rmsd",
                "casf_hit_0p25",
                "casf_hit_0p5",
                "casf_hit_0p75",
                "casf_hit_2p0",
            ],
        ),
        "CASF opt hits": (
            "casf_opt_hits",
            [
                *identity,
                "casf_opt_best_rmsd",
                "casf_opt_median_rmsd",
                "casf_opt_hit_0p25",
                "casf_opt_hit_0p5",
                "casf_opt_hit_0p75",
                "casf_opt_hit_2p0",
            ],
        ),
        "Funnel": (
            "funnel",
            [
                "row_type",
                "display_label",
                "family",
                "tier",
                "method",
                "ligands",
                "generated_ligands",
                "selected_ligands",
                "kept_ligands",
                "target_confs_mean",
                "selected_pool_total",
                "pb_fail_total",
                "kept_confs_total",
                "pb_fail_rate_mean",
                "kept_vs_target_rate_mean",
            ],
        ),
    }
    table_label = st.radio("Table", list(main_views), horizontal=True, key="main_table")
    table_name, table_columns = main_views[table_label]
    table_frame = view
    if table_name == "funnel":
        table_frame = select_view(comparison_rows, ligand_set, tier, family)
        table_frame = table_frame[table_frame["row_type"].astype(str) == "generation"]
    display_table(
        table_frame,
        table_columns,
        table_name,
        height=COMPARISON_TABLE_HEIGHT_PX,
    )
    render_threshold_charts(
        comparison_rows,
        ligand_set=ligand_set,
        tier=tier,
        family=family,
    )
    extended_db_path = sidebar_extended_db_path(db_path, table_names)
    render_druglike_tables(extended_db_path)
    render_report_analysis(
        per_ligand,
        ligand_set=ligand_set,
        tier=tier,
        family=family,
    )
    render_extended_analysis(extended_db_path)


if __name__ == "__main__":
    main()
