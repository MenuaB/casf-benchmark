from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from casf_benchmark.catalog import (
    describe_run,
    load_generation_run_entries,
    load_generation_runs,
)
from casf_benchmark.paths import DEFAULT_GENERATION_RUNS_CONFIG


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "runs.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_bundled_config_covers_every_cohort() -> None:
    assert DEFAULT_GENERATION_RUNS_CONFIG.is_file()
    for cohort in ("casf16_core", "casf16_ref", "druglike"):
        runs = load_generation_runs(cohort)
        assert runs, f"no runs configured for {cohort}"
        for label, dirname in runs:
            assert label
            assert dirname


def test_bundled_config_labels_are_unique_per_cohort() -> None:
    for cohort in ("casf16_core", "casf16_ref", "druglike"):
        labels = [label for label, _ in load_generation_runs(cohort)]
        assert len(labels) == len(set(labels))


def test_casf16_cohorts_are_a_subset_of_druglike() -> None:
    # Every checkpoint benchmarked on CASF16 was also run on the druglike set, so the
    # dashboard can join the two; the reverse does not hold.
    druglike = {label for label, _ in load_generation_runs("druglike")}
    for cohort in ("casf16_core", "casf16_ref"):
        assert {label for label, _ in load_generation_runs(cohort)} <= druglike


def test_preserves_config_order(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        runs:
          - label: b_second
            cohorts: {druglike: dir_b}
          - label: a_first
            cohorts: {druglike: dir_a}
        """,
    )
    assert load_generation_runs("druglike", config) == (
        ("b_second", "dir_b"),
        ("a_first", "dir_a"),
    )


def test_omits_checkpoints_without_the_requested_cohort(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        runs:
          - label: both
            cohorts: {casf16_core: core_dir, druglike: druglike_dir}
          - label: druglike_only
            cohorts: {druglike: only_dir}
        """,
    )
    assert load_generation_runs("casf16_core", config) == (("both", "core_dir"),)
    assert load_generation_runs("unknown_cohort", config) == ()


def test_rejects_duplicate_labels(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        """
        runs:
          - label: dupe
            cohorts: {druglike: first}
          - label: dupe
            cohorts: {druglike: second}
        """,
    )
    with pytest.raises(ValueError, match="duplicate label"):
        load_generation_runs("druglike", config)


def test_rejects_non_mapping_config(tmp_path: Path) -> None:
    config = write_config(tmp_path, "- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_generation_runs("druglike", config)


def test_accepts_arbitrary_non_qwen_labels(tmp_path: Path) -> None:
    # The next generators Menua evaluates are extra YAML entries, not a script fork:
    # nothing in the loader may assume a Qwen-shaped label.
    config = write_config(
        tmp_path,
        """
        runs:
          - label: rdkit_etkdg_druglike
            generator: RDKit
            display_label: "RDKit ETKDG"
            cohorts: {druglike: rdkit_dir}
          - label: loqi
            cohorts: {druglike: loqi_dir, casf16_core: loqi_core_dir}
        """,
    )
    assert load_generation_runs("druglike", config) == (
        ("rdkit_etkdg_druglike", "rdkit_dir"),
        ("loqi", "loqi_dir"),
    )
    entries = {entry.label: entry for entry in load_generation_run_entries(config)}
    assert entries["rdkit_etkdg_druglike"].descriptors == {
        "generator": "RDKit",
        "display_label": "RDKit ETKDG",
    }
    assert entries["loqi"].descriptors == {}


def test_describe_run_parses_qwen_labels() -> None:
    assert describe_run("qwen_1p7b_fsq_bigdata_step47023") == {
        "generator": "Qwen",
        "display_label": "qwen_1p7b_fsq_bigdata_step47023",
        "model_size": "1.7B",
        "tokenizer": "FSQ",
        "recipe": "bigdata",
        "step": 47023,
    }


def test_describe_run_degrades_for_other_generators() -> None:
    # A non-Qwen label must still land usable columns rather than an all-null row.
    described = describe_run("rdkit_etkdg_druglike")
    assert described["generator"] == "rdkit"
    assert described["display_label"] == "rdkit_etkdg_druglike"
    assert described["model_size"] is None
    assert described["step"] is None


def test_describe_run_prefers_explicit_descriptors() -> None:
    described = describe_run(
        "qwen_4b_4e_step20000", {"generator": "Qwen (rerun)", "recipe": "hand-labelled"}
    )
    assert described["generator"] == "Qwen (rerun)"
    assert described["recipe"] == "hand-labelled"
    # Fields the catalog did not override are still parsed from the label.
    assert described["model_size"] == "4B"
    assert described["step"] == 20000


def test_bundled_qwen_entries_all_describe_cleanly() -> None:
    for entry in load_generation_run_entries():
        if not entry.label.startswith("qwen_"):
            continue
        described = describe_run(entry.label, entry.descriptors)
        assert described["generator"] == "Qwen", entry.label
        assert described["model_size"], entry.label
        assert described["step"], entry.label


def test_per_checkpoint_families_claim_only_ligand_sets_that_have_results() -> None:
    """A family must not advertise a ligand set the analysis never produced.

    `qwen_0p6b_4e_from_bigdata_step29600` used to claim a `ref_root` whose analysis
    had aborted, so the dashboard's ref view looked like an accidental gap rather
    than a known one. Each per-checkpoint family keeps its runs at
    `.../casf_benchmark_runs/{id}_{core,ref}`, so the root it claims and the
    analysis source that feeds the dashboard have to agree exactly.
    """
    import yaml

    from casf_benchmark.catalog import load_families
    from casf_benchmark.paths import DEFAULT_ANALYSIS_SOURCES_CONFIG

    config = yaml.safe_load(DEFAULT_ANALYSIS_SOURCES_CONFIG.read_text(encoding="utf-8"))
    run_ids = {str(entry["run_id"]) for entry in config.get("sources", [])}

    mismatches = []
    for family in load_families():
        for ligand_set, root in (("core", family.core_root), ("ref", family.ref_root)):
            # Only the per-checkpoint layout has this 1:1 root/source correspondence;
            # the shared pharma and codex roots feed several families at once.
            if root is None or root.parent.name != "casf_benchmark_runs":
                continue
            expected = f"{family.id}_{ligand_set}"
            if expected not in run_ids:
                mismatches.append(f"{family.id} claims {ligand_set}_root but {expected} is not a source")
    assert not mismatches, "catalog claims results that do not exist:\n" + "\n".join(mismatches)


def test_pr1_qwen_families_are_hidden_on_core_only() -> None:
    import pandas as pd

    from casf_benchmark.catalog import (
        drop_hidden_core_rows,
        hidden_on_core_family_ids,
        is_hidden_on_core,
        load_families,
    )

    hidden = hidden_on_core_family_ids()
    step_named = {family.id for family in load_families() if "_step" in family.id}
    assert hidden == step_named
    assert "qwen_0p6b_fsq_bigdata_pretrain" not in hidden
    assert "qwen_0p6b_fsq_bigdata_step70534" in hidden
    assert is_hidden_on_core("qwen_0p6b_fsq_bigdata_step70534", "core")
    assert not is_hidden_on_core("qwen_0p6b_fsq_bigdata_step70534", "ref")
    assert not is_hidden_on_core("qwen_0p6b_fsq_bigdata_pretrain", "core")

    frame = pd.DataFrame(
        [
            {"ligand_set": "core", "family": "qwen_0p6b_fsq_bigdata_step70534"},
            {"ligand_set": "core", "family": "qwen_0p6b_fsq_bigdata_pretrain"},
            {"ligand_set": "ref", "family": "qwen_0p6b_fsq_bigdata_step70534"},
            {"ligand_set": "core", "family": "loqi_raw"},
        ]
    )
    out = drop_hidden_core_rows(frame)
    kept = set(zip(out["ligand_set"], out["family"]))
    assert kept == {
        ("core", "qwen_0p6b_fsq_bigdata_pretrain"),
        ("ref", "qwen_0p6b_fsq_bigdata_step70534"),
        ("core", "loqi_raw"),
    }
