"""Build reusable CASF analysis datasets from analyzer table outputs."""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from casf_benchmark.analysis.metrics import (
    CASF_GEOMETRIC_CLUSTER_THRESHOLDS,
    safe_mean,
    safe_median,
    safe_sum,
    threshold_tag,
)
from casf_benchmark.analysis.ref_dedup import apply_derived_ligand_sets, parse_derived_ligand_sets
from casf_benchmark.catalog import (
    display_label_for_method,
    family_by_id,
    maybe_parse_method_name,
)
from casf_benchmark.paths import DEFAULT_PER_LIGAND_LONG_CSV, REPO_ROOT

SUMMARY_TABLES = {
    "generation_filter": "geometric_generation_filter_summary.csv",
    "reference_filter": "geometric_reference_filter_summary.csv",
    "cluster": "geometric_cluster_summary.csv",
    "energy": "geometric_energy_summary.csv",
    "casf_hit": "geometric_casf_hit_summary.csv",
    "casf_opt_hit": "geometric_casf_opt_hit_summary.csv",
}
PB_TABLES = {
    "generation": "geometric_generation_pb_failure_summary.csv",
    "reference": "geometric_reference_pb_failure_summary.csv",
}
IDENTITY_COLUMNS = [
    "run_id",
    "run_label",
    "ligand_set",
    "row_type",
    "family",
    "tier",
    "variant",
    "method",
    "display_label",
    "source",
]
MASTER_KEY = ["row_type", "family", "tier", "ligand_set"]
REFERENCE_SOURCES = {"casf_crystal", "casf_opt", "chembl3d_sdf", "chembl3d_gt", "chembl3d_gt_pb"}
REFERENCE_LABELS = {
    "casf_crystal": "CASF crystal",
    "casf_opt": "CASF optimized",
    "chembl3d_sdf": "ChEMBL3D topology SDF",
    "chembl3d_gt": "ChEMBL3D ground truth",
    "chembl3d_gt_pb": "ChEMBL3D ground truth PB",
}
DEFAULT_GLOBAL_LONG_OUTPUT = DEFAULT_PER_LIGAND_LONG_CSV


@dataclass(frozen=True)
class AnalysisSource:
    run_id: str
    label: str
    root: Path
    ligand_set: str


def parse_simple_sources_yaml(path: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"CASF analysis source config must be a mapping: {path}")
    rows: list[dict[str, str]] = []
    for section in ("sources", "reference_sources"):
        for entry in data.get(section, []) or []:
            if not isinstance(entry, dict):
                raise ValueError(f"CASF analysis source entry must be a mapping: {entry!r}")
            rows.append({str(key): str(value) for key, value in entry.items()})
    return rows


def infer_ligand_set(root: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    name = root.name.lower()
    if "casf16_ref" in name or name.startswith("ref") or "_ref" in name:
        return "ref"
    if "core" in name:
        return "core"
    return "unknown"


def _display_label_for_source(source: str) -> str:
    parsed = maybe_parse_method_name(source)
    if parsed is None:
        return source
    try:
        return family_by_id(parsed.family).display_label
    except KeyError:
        return source


def _resolve_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def expand_sources(config_path: Path) -> list[AnalysisSource]:
    configs = parse_simple_sources_yaml(config_path)
    sources: list[AnalysisSource] = []
    seen: set[str] = set()
    for entry in configs:
        roots: Iterable[Path]
        if "root_glob" in entry:
            roots = [_resolve_root(Path(path)) for path in sorted(glob.glob(entry["root_glob"]))]
        else:
            roots = [_resolve_root(Path(entry["root"]))]
        for root in roots:
            ligand_set = infer_ligand_set(root, entry.get("ligand_set"))
            base_id = entry.get("run_id", root.name)
            run_id = base_id if "root_glob" not in entry else f"{base_id}_{ligand_set}"
            if run_id in seen:
                run_id = f"{run_id}_{root.name}"
            seen.add(run_id)
            label = entry.get("label", run_id)
            if "root_glob" in entry and ligand_set != "unknown":
                label = f"{label} {ligand_set.title()}"
            sources.append(AnalysisSource(run_id=run_id, label=label, root=root, ligand_set=ligand_set))
    return sources


def resolve_tables_dir(root: Path) -> Path:
    candidates = [root / "analysis" / "tables", root / "analysis", root / "tables", root]
    for candidate in candidates:
        if any((candidate / filename).exists() for filename in SUMMARY_TABLES.values()):
            return candidate
    return root / "analysis" / "tables"


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def table_applies_to_source(table_name: str, source: AnalysisSource) -> bool:
    if table_name == "casf_opt_hit":
        return source.ligand_set == "core"
    return True


def parse_source_metadata(source: object) -> tuple[str, str, str]:
    text = str(source)
    if text in REFERENCE_SOURCES:
        return text, "", "reference"
    parsed = maybe_parse_method_name(text)
    if parsed is not None:
        return parsed.family, parsed.variant, parsed.tier
    return text, "generated", "unknown"


def ensure_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "source" not in frame.columns:
        return frame
    out = frame.copy()
    parsed = out["source"].map(parse_source_metadata)
    for index, column in enumerate(("family", "variant", "tier")):
        if column not in out.columns:
            out[column] = [value[index] for value in parsed]
        else:
            values = out[column].astype("string")
            missing = values.isna() | (values == "")
            replacements = pd.Series([value[index] for value in parsed], index=out.index, dtype="string")
            out[column] = values.mask(missing, replacements)
    if "method" not in out.columns:
        out["method"] = out["source"].astype(str)
    if "display_label" not in out.columns:
        out["display_label"] = [
            REFERENCE_LABELS.get(str(source), _display_label_for_source(str(source)))
            for source in out["source"]
        ]
    if "row_type" not in out.columns:
        out["row_type"] = [
            "reference" if str(source) in REFERENCE_SOURCES else "generation"
            for source in out["source"]
        ]
    return out


def add_run_columns(frame: pd.DataFrame, source: AnalysisSource) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = ensure_metadata(frame)
    out.insert(0, "run_id", source.run_id)
    out.insert(1, "run_label", source.label)
    out.insert(2, "ligand_set", source.ligand_set)
    return out


def _rename_metric_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in frame.columns:
        if column in IDENTITY_COLUMNS:
            continue
        if column == "ligands_scope":
            rename[column] = f"{prefix}_ligands_scope"
        elif column == "ligands":
            rename[column] = f"{prefix}_ligands"
    return frame.rename(columns=rename)


def merge_wide(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    merged = pd.DataFrame()
    for prefix, frame in frames:
        if frame.empty or "source" not in frame.columns:
            continue
        right = _rename_metric_columns(ensure_metadata(frame), prefix)
        for column in IDENTITY_COLUMNS:
            if column not in right.columns:
                right[column] = ""
        right = right[[*IDENTITY_COLUMNS, *[col for col in right.columns if col not in IDENTITY_COLUMNS]]]
        if merged.empty:
            merged = right.copy()
            continue
        overlap = (set(merged.columns) & set(right.columns)) - set(IDENTITY_COLUMNS)
        if not overlap:
            merged = merged.merge(right, on=IDENTITY_COLUMNS, how="outer")
            continue
        # Shared metric columns (e.g. total_confs, selected_pool_total) appear in multiple
        # summary tables for different sources; coalesce instead of accumulating _x/_y suffixes.
        left_suffix, right_suffix = "_merge_left", "_merge_right"
        merged = merged.merge(
            right,
            on=IDENTITY_COLUMNS,
            how="outer",
            suffixes=(left_suffix, right_suffix),
        )
        for column in sorted(overlap):
            left_col = f"{column}{left_suffix}"
            right_col = f"{column}{right_suffix}"
            if left_col in merged.columns and right_col in merged.columns:
                merged[column] = merged[left_col].combine_first(merged[right_col])
                merged = merged.drop(columns=[left_col, right_col])
    return merged


def _safe_pb_column(value: object) -> str:
    text = str(value).strip().lower()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else "_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unknown"


def pivot_pb_failures(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty or not {"source", "pb_test"}.issubset(frame.columns):
        return pd.DataFrame()
    base = ensure_metadata(frame)
    for column in IDENTITY_COLUMNS:
        if column not in base.columns:
            base[column] = ""
    rows = []
    for keys, group in base.groupby(IDENTITY_COLUMNS, sort=False, dropna=False):
        row = {column: value for column, value in zip(IDENTITY_COLUMNS, keys)}
        for item in group.itertuples(index=False):
            test_name = _safe_pb_column(getattr(item, "pb_test"))
            if hasattr(item, "pb_fail_count"):
                row[f"{prefix}_pb_fail_count_{test_name}"] = getattr(item, "pb_fail_count")
            if hasattr(item, "pb_fail_rate"):
                row[f"{prefix}_pb_fail_rate_{test_name}"] = getattr(item, "pb_fail_rate")
        rows.append(row)
    return pd.DataFrame(rows)


def load_source_master(source: AnalysisSource) -> pd.DataFrame:
    tables_dir = resolve_tables_dir(source.root)
    frames: list[tuple[str, pd.DataFrame]] = []
    for key, filename in SUMMARY_TABLES.items():
        if not table_applies_to_source(key, source):
            continue
        frame = read_optional_csv(tables_dir / filename)
        if not frame.empty:
            frames.append((key, add_run_columns(frame, source)))
    for key, filename in PB_TABLES.items():
        frame = read_optional_csv(tables_dir / filename)
        if frame.empty:
            continue
        pivoted = pivot_pb_failures(add_run_columns(frame, source), key)
        if not pivoted.empty:
            frames.append((f"{key}_pb", pivoted))
    return merge_wide(frames)


def coalesce_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, object]] = []
    for _keys, group in frame.groupby(IDENTITY_COLUMNS, sort=False, dropna=False):
        row: dict[str, object] = {}
        for column in frame.columns:
            values = group[column]
            selected = None
            for value in values.tolist():
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                if isinstance(value, str) and value == "":
                    continue
                selected = value
                break
            row[column] = selected
        rows.append(row)
    return pd.DataFrame(rows, columns=frame.columns)


def canonicalize_reference_runs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"run_id", "run_label", "ligand_set", "source"}.issubset(frame.columns):
        return frame
    out = frame.copy()
    source_values = out["source"].astype(str)
    tier_values = out["tier"].astype(str) if "tier" in out.columns else pd.Series("", index=out.index)
    reference_mask = tier_values.eq("reference") | source_values.isin(REFERENCE_SOURCES)
    if not reference_mask.any():
        return out
    ligand_sets = out.loc[reference_mask, "ligand_set"].astype(str)
    out.loc[reference_mask, "run_id"] = "reference_" + ligand_sets
    out.loc[reference_mask, "run_label"] = ligand_sets.str.title() + " Reference Baselines"
    return out


def coalesce_reference_detail_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "source" not in frame.columns:
        return frame
    canonical = canonicalize_reference_runs(frame)
    if "tier" not in canonical.columns:
        return canonical
    reference_mask = canonical["tier"].astype(str).eq("reference")
    if not reference_mask.any():
        return canonical
    generation = canonical[~reference_mask]
    references = canonical[reference_mask]
    keys = [column for column in IDENTITY_COLUMNS if column in references.columns]
    if "pb_test" in references.columns:
        keys.append("pb_test")
    references = references.groupby(keys, sort=False, dropna=False).first().reset_index()
    return pd.concat([generation, references], ignore_index=True, sort=False)


def resolve_long_table_path(root: Path) -> Path:
    tables_dir = resolve_tables_dir(root)
    preferred = tables_dir / "geometric_per_ligand_long.csv"
    if preferred.exists():
        return preferred
    return tables_dir / "geometric_per_ligand_metrics.csv"


def load_source_long(source: AnalysisSource) -> pd.DataFrame:
    path = resolve_long_table_path(source.root)
    frame = read_optional_csv(path)
    if frame.empty:
        return frame
    out = ensure_metadata(frame)
    if "run_id" not in out.columns:
        out.insert(0, "run_id", source.run_id)
    if "run_label" not in out.columns:
        out.insert(1, "run_label", source.label)
    if "ligand_set" not in out.columns:
        out.insert(2, "ligand_set", source.ligand_set)
    else:
        values = out["ligand_set"].astype("string")
        out["ligand_set"] = values.mask(values.isna() | (values == ""), source.ligand_set)
    out["_source_mtime"] = path.stat().st_mtime if path.exists() else 0.0
    return out


def build_global_per_ligand_long(
    config_path: Path,
    output_csv: Path | None = DEFAULT_GLOBAL_LONG_OUTPUT,
) -> pd.DataFrame:
    frames = [load_source_long(source) for source in expand_sources(config_path)]
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        global_long = pd.DataFrame()
    else:
        global_long = pd.concat(nonempty, ignore_index=True, sort=False)
        if {"mol_id", "method", "ligand_set"}.issubset(global_long.columns):
            if "_source_mtime" in global_long.columns:
                global_long = global_long.sort_values("_source_mtime", kind="stable")
            global_long = global_long.drop_duplicates(
                subset=["mol_id", "method", "ligand_set"],
                keep="last",
            )
        if "_source_mtime" in global_long.columns:
            global_long = global_long.drop(columns=["_source_mtime"])
    specs = parse_derived_ligand_sets(config_path)
    if specs and not global_long.empty:
        global_long = apply_derived_ligand_sets(global_long, specs)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        global_long.to_csv(output_csv, index=False)
    return global_long


def _mean_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return math.nan
    return safe_mean(pd.to_numeric(frame[column], errors="coerce").tolist())


def _hit_rate_column(frame: pd.DataFrame, column: str, *, missing_as_miss: bool) -> float:
    """Mean hit flag. Derived cohorts count a missing flag as a miss when any hit is defined."""
    if column not in frame.columns:
        return math.nan
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.empty or not numeric.notna().any():
        return math.nan
    if missing_as_miss:
        return float(numeric.fillna(0).mean())
    return safe_mean(numeric.tolist())


def _median_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return math.nan
    return safe_median(pd.to_numeric(frame[column], errors="coerce").tolist())


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return math.nan
    return safe_sum(pd.to_numeric(frame[column], errors="coerce").tolist())


def _is_compare_source(source: object) -> bool:
    return str(source) not in {"casf_crystal", "casf_opt", "chembl3d_sdf"}


def build_comparison_master(
    global_long: pd.DataFrame,
    *,
    missing_as_miss_ligand_sets: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    if global_long.empty:
        return pd.DataFrame()
    data = ensure_metadata(global_long)
    group_cols = [
        "ligand_set",
        "row_type",
        "family",
        "tier",
        "variant",
        "method",
        "display_label",
        "source",
    ]
    rows: list[dict[str, object]] = []
    for keys, frame in data.groupby(group_cols, sort=False, dropna=False):
        row = {column: value for column, value in zip(group_cols, keys)}
        row["run_id"] = frame["run_id"].dropna().astype(str).iloc[-1] if "run_id" in frame and frame["run_id"].notna().any() else ""
        row["run_label"] = frame["run_label"].dropna().astype(str).iloc[-1] if "run_label" in frame and frame["run_label"].notna().any() else ""
        row["ligands"] = int(frame["mol_id"].nunique()) if "mol_id" in frame.columns else len(frame)
        row["total_confs"] = _sum_column(frame, "conformer_count")
        row["mean_confs_per_ligand"] = _mean_column(frame, "conformer_count")
        if row["row_type"] == "generation":
            pb_input = pd.to_numeric(frame.get("pb_input_confs", pd.Series(index=frame.index)), errors="coerce").fillna(0)
            pb_fail = pd.to_numeric(frame.get("pb_fail_confs", pd.Series(index=frame.index)), errors="coerce").fillna(0)
            kept = pd.to_numeric(frame.get("kept_confs", pd.Series(index=frame.index)), errors="coerce").fillna(0)
            target = pd.to_numeric(
                frame.get("target_confs", frame.get("num_target_confs", pd.Series(index=frame.index))),
                errors="coerce",
            ).fillna(0)
            generated = pd.to_numeric(frame.get("generated_candidates", pd.Series(index=frame.index)), errors="coerce").fillna(0)
            row["generated_ligands"] = int((generated > 0).sum())
            row["selected_ligands"] = int((pb_input > 0).sum())
            row["kept_ligands"] = int((kept > 0).sum())
            row["selected_pool_total"] = float(pb_input.sum())
            row["kept_confs_total"] = float(kept.sum())
            row["pb_fail_total"] = float(pb_fail.sum())
            row["target_confs_mean"] = safe_mean(target.tolist())
            row["pb_fail_rate_mean"] = safe_mean(
                [float(n / d) for n, d in zip(pb_fail.tolist(), pb_input.tolist()) if d > 0]
            )
            row["kept_vs_target_rate_mean"] = safe_mean(
                [float(n / d) for n, d in zip(kept.tolist(), target.tolist()) if d > 0]
            )
            for column in (
                "generated_candidates",
                "finite_rejected",
                "clash_rejected",
                "bond_rejected",
                "stereo_rejected",
                "rmsd_rejected",
                "pre_clash_passed",
                "minimization_input_confs",
                "minimization_failed",
            ):
                row[f"{column}_total"] = _sum_column(frame, column)
        elif row["row_type"] == "reference":
            row["selected_pool_total"] = _sum_column(frame, "clash_input_confs")
            row["clash_fail_total"] = _sum_column(frame, "clash_fail_confs")
            row["pb_fail_total"] = _sum_column(frame, "pb_fail_confs")
            row["clash_fail_rate"] = (
                row["clash_fail_total"] / row["selected_pool_total"]
                if row["selected_pool_total"]
                else math.nan
            )
            pb_input_total = _sum_column(frame, "pb_input_confs")
            row["pb_fail_rate"] = row["pb_fail_total"] / pb_input_total if pb_input_total else math.nan

        for column in (
            "pairwise_mean",
            "pairwise_p90",
            "mean_torsion_std_deg",
            "energy_min",
            "energy_max",
            "energy_median",
            "energy_std",
            "casf_best_rmsd",
            "casf_median_rmsd",
            "casf_opt_best_rmsd",
            "casf_opt_median_rmsd",
        ):
            row[column] = _mean_column(frame, column)
        row["median_energy_std"] = _median_column(frame, "energy_std")
        for threshold in CASF_GEOMETRIC_CLUSTER_THRESHOLDS:
            tag = threshold_tag(threshold)
            cluster_col = f"greedy_clusters_{tag}"
            row[f"total_clusters_{tag}"] = _sum_column(frame, cluster_col)
            row[f"mean_clusters_{tag}"] = _mean_column(frame, cluster_col)
            for column in (
                f"clusters_per_100_{tag}",
                f"cluster_entropy_{tag}",
                f"largest_cluster_fraction_{tag}",
                f"effective_clusters_{tag}",
                f"simpson_concentration_{tag}",
                f"singleton_fraction_{tag}",
            ):
                row[column] = _mean_column(frame, column)
        missing_as_miss = str(row["ligand_set"]) in missing_as_miss_ligand_sets
        for prefix in ("casf", "casf_opt"):
            for tag in ("0p25", "0p5", "0p75", "2p0"):
                row[f"{prefix}_hit_{tag}"] = _hit_rate_column(
                    frame,
                    f"{prefix}_hit_{tag}",
                    missing_as_miss=missing_as_miss,
                )
        if missing_as_miss and "casf_best_rmsd" in frame.columns:
            row["defined_rmsd_ligands"] = int(pd.to_numeric(frame["casf_best_rmsd"], errors="coerce").notna().sum())
        rows.append(row)
    master = pd.DataFrame(rows)
    if master.empty:
        return master
    master = add_ligands_scope(master, "total_confs")
    ordered = [column for column in IDENTITY_COLUMNS if column in master.columns]
    ordered.extend(column for column in master.columns if column not in ordered)
    master = master[ordered]
    master["__row_type_order"] = master["row_type"].map({"generation": 0, "reference": 1}).fillna(9)
    master["__tier_order"] = master["tier"].map({"fixed": 0, "dynamic": 1, "chembl_count": 2, "reference": 3}).fillna(9)
    master = master.sort_values(["ligand_set", "__row_type_order", "family", "__tier_order"], kind="stable")
    return master.drop(columns=["__row_type_order", "__tier_order"]).reset_index(drop=True)


def add_ligands_scope(frame: pd.DataFrame, confs_col: str) -> pd.DataFrame:
    if frame.empty or "ligands" not in frame.columns:
        return frame
    out = frame.copy()
    if confs_col in out.columns:
        mean_confs = pd.to_numeric(out[confs_col], errors="coerce") / pd.to_numeric(out["ligands"], errors="coerce")
    else:
        mean_confs = pd.Series([math.nan] * len(out), index=out.index)
    out["ligands_scope"] = [
        f"{int(ligands)}/{int(round(float(mean_conf)))}"
        if pd.notna(mean_conf) and math.isfinite(float(mean_conf))
        else f"{int(ligands)}/-"
        for ligands, mean_conf in zip(out["ligands"], mean_confs)
    ]
    return out


def _missing_as_miss_sets(config_path: Path) -> frozenset[str]:
    return frozenset(spec.id for spec in parse_derived_ligand_sets(config_path))


def build_master_frame(config_path: Path) -> tuple[int, pd.DataFrame]:
    sources = expand_sources(config_path)
    global_long = build_global_per_ligand_long(config_path, output_csv=None)
    return len(sources), build_comparison_master(
        global_long,
        missing_as_miss_ligand_sets=_missing_as_miss_sets(config_path),
    )


def build_master_csv(config_path: Path, output_csv: Path) -> tuple[int, int]:
    global_long = build_global_per_ligand_long(config_path, DEFAULT_GLOBAL_LONG_OUTPUT)
    master = build_comparison_master(
        global_long,
        missing_as_miss_ligand_sets=_missing_as_miss_sets(config_path),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_csv, index=False)
    return len(expand_sources(config_path)), len(master)
