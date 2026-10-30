from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from casf_benchmark.analysis.dataset import (
    build_comparison_master,
    build_global_per_ligand_long,
    build_master_frame,
)
from casf_benchmark.analysis.ref_dedup import (
    DEFAULT_MANIFEST,
    DerivedLigandSet,
    apply_derived_cohort,
    load_selected_ids,
    select_crystals,
    write_manifest,
)
from casf_benchmark.cli.build_dashboard_db import build_dashboard_db
from casf_benchmark.paths import REPO_ROOT
from casf_benchmark.stratum_bins import assign_stratum


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(path: Path, sources: list[tuple[str, str, str, Path]], refs: list[tuple[str, str, str, Path]] | None = None) -> None:
    lines = ["sources:"]
    for run_id, label, ligand_set, root in sources:
        lines.extend(
            [
                f"  - run_id: {run_id}",
                f"    label: {label}",
                f"    ligand_set: {ligand_set}",
                f"    root: {root}",
            ]
        )
    if refs:
        lines.append("reference_sources:")
        for run_id, label, ligand_set, root in refs:
            lines.extend(
                [
                    f"  - run_id: {run_id}",
                    f"    label: {label}",
                    f"    ligand_set: {ligand_set}",
                    f"    root: {root}",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gen_row(mol_id: str, method: str, ligand_set: str, rot: int, heavy: int, clusters: int = 2) -> dict[str, object]:
    family = method.removesuffix("_fixed").removesuffix("_dynamic").removesuffix("_chembl_count")
    tier = method.rsplit("_", 1)[-1]
    if method.endswith("_chembl_count"):
        tier = "chembl_count"
        family = method[: -len("_chembl_count")]
    return {
        "mol_id": mol_id,
        "ligand_set": ligand_set,
        "row_type": "generation",
        "family": family,
        "tier": tier,
        "variant": "4b_revisited" if family == "qwen_4b_revisited" else "",
        "method": method,
        "source": method,
        "display_label": family,
        "rotatable_bonds": rot,
        "heavy_atoms": heavy,
        "conformer_count": 10,
        "pb_input_confs": 10,
        "pb_pass_confs": 9,
        "pb_fail_confs": 1,
        "pb_pass_rate": 0.9,
        "generated_candidates": 12,
        "kept_confs": 9,
        "target_confs": 10,
        "greedy_clusters_1p0": clusters,
        "clusters_per_100_1p0": clusters * 10,
        "energy_median": 2.0,
        "casf_hit_0p5": 0.5,
        "casf_opt_hit_0p5": 0.25 if ligand_set == "core" else float("nan"),
    }


def ref_row(mol_id: str, source: str, ligand_set: str, rot: int, heavy: int) -> dict[str, object]:
    return {
        "mol_id": mol_id,
        "ligand_set": ligand_set,
        "row_type": "reference",
        "family": source,
        "tier": "reference",
        "variant": "",
        "method": source,
        "source": source,
        "display_label": source,
        "rotatable_bonds": rot,
        "heavy_atoms": heavy,
        "conformer_count": 1,
        "clash_input_confs": 1,
        "clash_fail_confs": 0,
        "pb_input_confs": 1,
        "pb_pass_confs": 1,
        "pb_fail_confs": 0,
        "pb_pass_rate": 1.0,
        "greedy_clusters_1p0": 1,
        "clusters_per_100_1p0": 100.0,
        "energy_median": 1.0,
        "casf_hit_0p5": 1.0,
        "casf_opt_hit_0p5": 1.0 if ligand_set == "core" else float("nan"),
    }


def test_build_master_from_per_ligand_long(tmp_path):
    root = tmp_path / "qwen_core"
    rows = [
        gen_row("lig_a", "qwen_4b_revisited_fixed", "core", 2, 18),
        gen_row("lig_b", "qwen_4b_revisited_fixed", "core", 7, 25, clusters=4),
    ]
    write_csv(root / "analysis" / "tables" / "geometric_per_ligand_long.csv", rows)
    config = tmp_path / "sources.yaml"
    write_config(config, [("qwen_core", "Qwen Core", "core", root)])

    global_long = build_global_per_ligand_long(config, tmp_path / "long.csv")
    master = build_comparison_master(global_long)

    assert len(global_long) == 2
    assert len(master) == 1
    row = master.iloc[0]
    assert row["family"] == "qwen_4b_revisited"
    assert row["variant"] == "4b_revisited"
    assert row["ligands"] == 2
    assert row["mean_clusters_1p0"] == 3.0
    assert row["selected_pool_total"] == 20.0


def test_comparison_master_median_energy_std_is_not_the_mean():
    rows = [
        gen_row("lig_a", "loqi_raw_fixed", "core", 2, 18),
        gen_row("lig_b", "loqi_raw_fixed", "core", 7, 25),
        gen_row("lig_c", "loqi_raw_fixed", "core", 12, 35),
    ]
    rows[0]["energy_std"] = 1.0
    rows[1]["energy_std"] = 2.0
    rows[2]["energy_std"] = 100.0

    master = build_comparison_master(pd.DataFrame(rows))

    row = master.iloc[0]
    assert abs(float(row["energy_std"]) - (103.0 / 3.0)) < 1e-9
    assert float(row["median_energy_std"]) == 2.0


def test_build_master_frame_keeps_reference_rows_separate(tmp_path):
    gen_root = tmp_path / "loqi_core"
    ref_root = tmp_path / "reference_core"
    write_csv(
        gen_root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [gen_row("lig_a", "loqi_raw_fixed", "core", 2, 18)],
    )
    write_csv(
        ref_root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [ref_row("lig_a", source, "core", 2, 18) for source in ["casf_crystal", "chembl3d_gt"]],
    )
    config = tmp_path / "sources.yaml"
    write_config(
        config,
        [("loqi_core", "LOQI Core", "core", gen_root)],
        refs=[("reference_core", "Reference Core", "core", ref_root)],
    )

    _run_count, master = build_master_frame(config)

    assert set(master["row_type"]) == {"generation", "reference"}
    assert set(master["source"]) == {"loqi_raw_fixed", "casf_crystal", "chembl3d_gt"}
    assert len(master) == 3


def test_dashboard_db_writes_strata_and_omits_empty_bins(tmp_path):
    root = tmp_path / "loqi_core"
    ref_root = tmp_path / "reference_core"
    write_csv(
        root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [
            gen_row("lig_a", "loqi_raw_fixed", "core", 2, 18),
            gen_row("lig_b", "loqi_raw_fixed", "core", 7, 25),
        ],
    )
    write_csv(
        ref_root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [
            ref_row("lig_a", "chembl3d_gt", "core", 2, 18),
            ref_row("lig_b", "chembl3d_gt", "core", 7, 25),
        ],
    )
    config = tmp_path / "sources.yaml"
    write_config(
        config,
        [("loqi_core", "LOQI Core", "core", root)],
        refs=[("reference_core", "Reference Core", "core", ref_root)],
    )

    db_path = tmp_path / "dashboard.sqlite"
    build_dashboard_db(config, db_path)

    with sqlite3.connect(db_path) as connection:
        rows = pd.read_sql_query("SELECT * FROM comparison_rows", connection)
        strata = pd.read_sql_query("SELECT * FROM comparison_strata", connection)
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert user_version == 2
    assert set(rows["row_type"]) == {"generation", "reference"}
    rb = strata[(strata["breakdown"] == "rotatable_bonds") & (strata["view_tier"] == "fixed")]
    assert "15+" not in set(rb["stratum"])
    loqi_total = rows[rows["method"] == "loqi_raw_fixed"].iloc[0]["ligands"]
    loqi_strata = rb[rb["method"] == "loqi_raw_fixed"]["ligands"].sum()
    assert loqi_strata == loqi_total


def _tiny_intersection() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ligand_id": "lig_a", "casf_heavy_isomeric_smiles": "CCO", "source_file": "a.mol2"},
            {"ligand_id": "lig_b", "casf_heavy_isomeric_smiles": "CCC", "source_file": "b.mol2"},
            {"ligand_id": "lig_c", "casf_heavy_isomeric_smiles": "CCC", "source_file": "c.mol2"},
        ]
    )


def test_derived_ref_dedup_keeps_one_shared_crystal_universe(tmp_path):
    selected = select_crystals(_tiny_intersection())
    manifest = tmp_path / "selection.csv"
    write_manifest(selected, manifest)
    chosen = set(load_selected_ids(manifest))
    assert len(chosen) == 2
    dropped = {"lig_b", "lig_c"} - chosen
    assert len(dropped) == 1
    missing_selected = next(iter(chosen - {"lig_a"}))

    gen_root = tmp_path / "model_ref"
    ref_root = tmp_path / "reference_ref"
    core_root = tmp_path / "model_core"
    present = [mol_id for mol_id in ("lig_a", "lig_b", "lig_c") if mol_id != missing_selected]
    generation_rows = []
    for mol_id in present:
        row = gen_row(mol_id, "loqi_raw_fixed", "ref", 2, 18)
        row["run_id"] = "loqi_ref"
        row["casf_best_rmsd"] = 0.4 if mol_id == "lig_a" else float("nan")
        row["casf_hit_0p5"] = 1.0 if mol_id == "lig_a" else float("nan")
        generation_rows.append(row)
    write_csv(gen_root / "analysis" / "tables" / "geometric_per_ligand_long.csv", generation_rows)
    write_csv(
        ref_root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [ref_row(mol_id, "chembl3d_gt", "ref", 2, 18) | {"run_id": "reference_ref", "casf_best_rmsd": 1.0, "casf_hit_0p5": 0.0} for mol_id in ("lig_a", "lig_b", "lig_c")],
    )
    write_csv(
        core_root / "analysis" / "tables" / "geometric_per_ligand_long.csv",
        [gen_row("core_a", "loqi_raw_fixed", "core", 2, 18)],
    )
    config = tmp_path / "sources.yaml"
    write_config(
        config,
        [
            ("loqi_ref", "LOQI ref", "ref", gen_root),
            ("loqi_core", "LOQI core", "core", core_root),
        ],
        refs=[("reference_ref", "Reference ref", "ref", ref_root)],
    )
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n".join(
            [
                "derived_ligand_sets:",
                "  - id: ref_dedup",
                '    label: "CASF ref dedup"',
                "    source_ligand_set: ref",
                f"    selection_manifest: {manifest}",
                "    selection_key: ligand_id",
                "    selected_column: selected",
                "",
            ]
        ),
        encoding="utf-8",
    )

    global_long = build_global_per_ligand_long(config, output_csv=None)
    ref_rows = global_long[global_long["ligand_set"].astype(str) == "ref"]
    dedup = global_long[global_long["ligand_set"].astype(str) == "ref_dedup"]
    core = global_long[global_long["ligand_set"].astype(str) == "core"]

    assert set(ref_rows.loc[ref_rows["method"] == "loqi_raw_fixed", "mol_id"]) == set(present)
    assert set(core["mol_id"]) == {"core_a"}
    for method, group in dedup.groupby("method"):
        assert set(group["mol_id"]) == chosen, method
        assert group["run_id"].astype(str).str.endswith("__ref_dedup").all()
        assert set(group["ligand_set"]) == {"ref_dedup"}
    assert missing_selected in set(dedup.loc[dedup["method"] == "loqi_raw_fixed", "mol_id"])
    assert dropped.isdisjoint(set(dedup["mol_id"]))
    kept = dedup[(dedup["method"] == "loqi_raw_fixed") & (dedup["mol_id"] == "lig_a")].iloc[0]
    assert float(kept["casf_best_rmsd"]) == 0.4
    missing_row = dedup[(dedup["method"] == "loqi_raw_fixed") & (dedup["mol_id"] == missing_selected)].iloc[0]
    assert pd.isna(missing_row["casf_best_rmsd"])

    master = build_comparison_master(dedup, missing_as_miss_ligand_sets=frozenset({"ref_dedup"}))
    loqi = master[master["method"] == "loqi_raw_fixed"].iloc[0]
    assert int(loqi["ligands"]) == 2
    assert float(loqi["casf_hit_0p5"]) == 0.5
    assert float(loqi["casf_best_rmsd"]) == 0.4
    assert int(loqi["defined_rmsd_ligands"]) == 1
    reference = master[master["method"] == "chembl3d_gt"].iloc[0]
    assert int(reference["ligands"]) == 2


def test_public_ref_rows_derive_the_same_selected_universe():
    candidates = [
        REPO_ROOT / "data/results/staging/public-baseline/casf_analysis_dashboard.sqlite",
        REPO_ROOT / "data/results/casf_analysis_dashboard.sqlite",
    ]
    database = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
    if database is None:
        pytest.skip("no dashboard sqlite is available")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        per_ligand = pd.read_sql_query(
            "SELECT * FROM per_ligand_long WHERE ligand_set = 'ref'",
            connection,
        )
    selected_ids = load_selected_ids(DEFAULT_MANIFEST)
    assert len(selected_ids) == 1044
    assert set(per_ligand["mol_id"].astype(str)) == set(
        pd.read_csv(REPO_ROOT / "data/mapping/casf16_ref_chembl3d_exact_intersection.csv", usecols=["ligand_id"])[
            "ligand_id"
        ].astype(str)
    )
    derived = apply_derived_cohort(
        per_ligand,
        DerivedLigandSet(
            id="ref_dedup",
            label="CASF ref dedup",
            source_ligand_set="ref",
            selection_manifest=DEFAULT_MANIFEST,
        ),
        selected_ids,
    )
    assert set(per_ligand["mol_id"].astype(str)) == set(
        pd.read_csv(DEFAULT_MANIFEST, usecols=["ligand_id"])["ligand_id"].astype(str)
    )
    assert per_ligand["mol_id"].nunique() == 1236
    universes = {
        frozenset(group["mol_id"].astype(str))
        for _method, group in derived.groupby("method")
    }
    assert universes == {frozenset(selected_ids)}
    joined = per_ligand.merge(
        derived,
        on=["method", "mol_id"],
        suffixes=("_ref", "_dedup"),
    )
    assert len(joined) == 1044 * per_ligand["method"].nunique()
    for column in ("casf_best_rmsd", "casf_hit_0p5", "conformer_count"):
        left = pd.to_numeric(joined[f"{column}_ref"], errors="coerce").reset_index(drop=True)
        right = pd.to_numeric(joined[f"{column}_dedup"], errors="coerce").reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False)
    for column in ("run_label", "family", "tier"):
        left = joined[f"{column}_ref"].astype("string").reset_index(drop=True)
        right = joined[f"{column}_dedup"].astype("string").reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False)
    master = build_comparison_master(derived, missing_as_miss_ligand_sets=frozenset({"ref_dedup"}))
    generation = master[master["row_type"].astype(str) == "generation"]
    assert set(generation["ligands"]) == {1044}


def test_assign_stratum_preserves_series_index():
    values = pd.Series([2, 7], index=[100, 200])

    strata = assign_stratum(values, "rotatable_bonds")

    assert list(strata.index) == [100, 200]
    assert list(strata.astype(str)) == ["0-4", "5-9"]
