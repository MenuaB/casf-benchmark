#!/usr/bin/env python3
"""Build a SQLite dashboard database from CASF long-table outputs."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

from casf_benchmark.analysis.dataset import (
    build_comparison_master,
    build_global_per_ligand_long,
    expand_sources,
)
from casf_benchmark.analysis.ref_dedup import parse_derived_ligand_sets
from casf_benchmark.catalog import load_families
from casf_benchmark.paths import DEFAULT_DASHBOARD_DB, DEFAULT_ANALYSIS_SOURCES_CONFIG
from casf_benchmark.stratum_bins import STRATA, assign_stratum

DEFAULT_CONFIG = DEFAULT_ANALYSIS_SOURCES_CONFIG
DEFAULT_OUTPUT = DEFAULT_DASHBOARD_DB
TIERS = ("fixed", "dynamic", "chembl_count")


def _write_table(connection: sqlite3.Connection, name: str, frame: pd.DataFrame) -> int:
    frame.to_sql(name, connection, if_exists="replace", index=False)
    return len(frame)


def build_comparison_strata(
    per_ligand_long: pd.DataFrame,
    *,
    missing_as_miss_ligand_sets: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    if per_ligand_long.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for ligand_set in sorted(per_ligand_long["ligand_set"].dropna().astype(str).unique()):
        ligand_set_frame = per_ligand_long[per_ligand_long["ligand_set"].astype(str) == ligand_set].copy()
        for tier in TIERS:
            tier_frame = ligand_set_frame[
                ((ligand_set_frame["row_type"].astype(str) == "generation") & (ligand_set_frame["tier"].astype(str) == tier))
                | (ligand_set_frame["row_type"].astype(str) == "reference")
            ].copy()
            if tier_frame.empty:
                continue
            for breakdown, spec in STRATA.items():
                if spec.column not in tier_frame.columns:
                    continue
                tier_frame = tier_frame.copy()
                tier_frame["stratum"] = assign_stratum(tier_frame[spec.column], breakdown)
                for label in spec.labels:
                    stratum_frame = tier_frame[tier_frame["stratum"].astype(str) == label].drop(columns=["stratum"])
                    if stratum_frame.empty:
                        continue
                    aggregate = build_comparison_master(
                        stratum_frame,
                        missing_as_miss_ligand_sets=missing_as_miss_ligand_sets,
                    )
                    if aggregate.empty:
                        continue
                    aggregate.insert(0, "breakdown", breakdown)
                    aggregate.insert(1, "stratum", label)
                    aggregate.insert(2, "view_tier", tier)
                    rows.append(aggregate)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def validate_strata_ligands(comparison_rows: pd.DataFrame, comparison_strata: pd.DataFrame) -> None:
    if comparison_rows.empty or comparison_strata.empty:
        return
    total = comparison_rows.copy()
    total_key = ["ligand_set", "row_type", "family", "method"]
    for breakdown in sorted(comparison_strata["breakdown"].dropna().astype(str).unique()):
        for view_tier in TIERS:
            strata = comparison_strata[
                (comparison_strata["breakdown"].astype(str) == breakdown)
                & (comparison_strata["view_tier"].astype(str) == view_tier)
            ]
            if strata.empty:
                continue
            strata_counts = (
                strata.groupby(total_key, dropna=False)["ligands"].sum().reset_index(name="strata_ligands")
            )
            expected = total[
                ((total["row_type"].astype(str) == "generation") & (total["tier"].astype(str) == view_tier))
                | (total["row_type"].astype(str) == "reference")
            ][[*total_key, "ligands"]]
            merged = expected.merge(strata_counts, on=total_key, how="left")
            merged["strata_ligands"] = merged["strata_ligands"].fillna(0)
            bad = merged[merged["strata_ligands"] != merged["ligands"]]
            if not bad.empty:
                preview = bad.head(5).to_dict("records")
                raise ValueError(
                    f"Strata ligand counts do not sum to total for {breakdown}/{view_tier}: {preview}"
                )


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": family.id,
                "display_label": family.display_label,
                "variant": family.variant,
                "core_root": str(family.core_root) if family.core_root is not None else "",
                "ref_root": str(family.ref_root) if family.ref_root is not None else "",
            }
            for family in load_families()
        ]
    )


def build_dashboard_db(config_path: Path, output_db: Path) -> dict[str, int]:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = output_db.with_suffix(output_db.suffix + ".tmp")
    if tmp_db.exists():
        tmp_db.unlink()

    sources = expand_sources(config_path)
    derived_specs = parse_derived_ligand_sets(config_path)
    missing_as_miss = frozenset(spec.id for spec in derived_specs)
    per_ligand_long = build_global_per_ligand_long(config_path, output_csv=None)
    comparison_rows = build_comparison_master(per_ligand_long, missing_as_miss_ligand_sets=missing_as_miss)
    comparison_strata = build_comparison_strata(
        per_ligand_long,
        missing_as_miss_ligand_sets=missing_as_miss,
    )
    validate_strata_ligands(comparison_rows, comparison_strata)

    table_rows: dict[str, int] = {}
    with sqlite3.connect(tmp_db) as connection:
        table_rows["analysis_sources"] = _write_table(
            connection,
            "analysis_sources",
            pd.DataFrame([source.__dict__ | {"root": str(source.root)} for source in sources]),
        )
        table_rows["comparison_rows"] = _write_table(connection, "comparison_rows", comparison_rows)
        table_rows["comparison_strata"] = _write_table(connection, "comparison_strata", comparison_strata)
        table_rows["per_ligand_long"] = _write_table(connection, "per_ligand_long", per_ligand_long)
        table_rows["catalog_families"] = _write_table(connection, "catalog_families", catalog_frame())
        if derived_specs:
            table_rows["derived_ligand_sets"] = _write_table(
                connection,
                "derived_ligand_sets",
                pd.DataFrame(
                    [
                        {
                            "id": spec.id,
                            "label": spec.label,
                            "source_ligand_set": spec.source_ligand_set,
                            "selection_manifest": str(spec.selection_manifest),
                            "selection_key": spec.selection_key,
                            "selected_column": spec.selected_column,
                        }
                        for spec in derived_specs
                    ]
                ),
            )

        connection.execute("CREATE INDEX IF NOT EXISTS idx_comparison_rows_view ON comparison_rows(ligand_set, tier, row_type, family)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_comparison_strata_view ON comparison_strata(ligand_set, view_tier, breakdown, stratum, row_type, family)")
        connection.execute("PRAGMA user_version = 2")
        connection.execute("PRAGMA optimize")

    os.replace(tmp_db, output_db)
    table_rows["configured_runs"] = len(sources)
    return table_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table_rows = build_dashboard_db(args.config.resolve(), args.output_db.resolve())
    print(f"Wrote dashboard DB: {args.output_db}")
    for table_name, row_count in sorted(table_rows.items()):
        print(f"{table_name}: {row_count}")


if __name__ == "__main__":
    main()
