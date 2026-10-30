from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from casf_benchmark.analysis.ref_dedup import (
    DEFAULT_MANIFEST,
    DEFAULT_PROVENANCE,
    EXPECTED_COUNTS,
    SELECTION_SEED,
    SOURCE_INTERSECTION,
    WEKA_INTERSECTION,
    assert_selection,
    manifest_frame,
    read_intersection,
    select_crystals,
    selection_counts,
    selection_digest,
    sha256_file,
    write_manifest,
    write_provenance,
    write_selection_artifacts,
)
from casf_benchmark.paths import REPO_ROOT


def test_committed_selection_matches_expected_counts() -> None:
    frame = pd.read_csv(DEFAULT_MANIFEST, dtype=str).fillna("")
    frame["selected"] = frame["selected"].map(lambda value: value == "true")
    frame["selection_rank"] = frame["selection_rank"].astype(int)
    assert_selection(frame)
    assert selection_counts(frame) == EXPECTED_COUNTS
    assert set(frame["selection_seed"]) == {SELECTION_SEED}
    chosen = frame[frame["selected"]]
    assert chosen["selection_rank"].eq(1).all()
    assert not any(column in frame.columns for column in ("casf_best_rmsd", "method", "run_id"))


def test_selection_is_byte_identical_and_order_independent(tmp_path: Path) -> None:
    source = read_intersection(SOURCE_INTERSECTION)
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    shuffled = source.sample(frac=1, random_state=7).reset_index(drop=True)
    write_manifest(select_crystals(source), first)
    write_manifest(select_crystals(shuffled), second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == DEFAULT_MANIFEST.read_bytes()


def test_rerunning_the_writer_matches_committed_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    provenance = tmp_path / "provenance.json"
    write_selection_artifacts(SOURCE_INTERSECTION, manifest, provenance)
    assert manifest.read_bytes() == DEFAULT_MANIFEST.read_bytes()
    assert provenance.read_bytes() == DEFAULT_PROVENANCE.read_bytes()


def test_selection_ignores_model_result_columns() -> None:
    source = read_intersection(SOURCE_INTERSECTION)
    mutated = source.copy()
    mutated["casf_best_rmsd"] = range(len(mutated))
    mutated["method"] = "pretend_model"
    left = manifest_frame(select_crystals(source))
    right = manifest_frame(select_crystals(mutated))
    pd.testing.assert_frame_equal(left, right)
    source_text = inspect.getsource(select_crystals) + inspect.getsource(selection_digest)
    assert "casf_best_rmsd" not in source_text
    assert "per_ligand" not in source_text


def test_digest_uses_only_seed_smiles_and_ligand_id() -> None:
    digest = selection_digest("seed", "CCO", "lig")
    assert digest == selection_digest("seed", "CCO", "lig")
    assert digest != selection_digest("other", "CCO", "lig")
    assert digest != selection_digest("seed", "CCO", "other")


def test_local_mapping_hash_is_recorded() -> None:
    payload = DEFAULT_PROVENANCE.read_text(encoding="utf-8")
    recorded = sha256_file(SOURCE_INTERSECTION)
    assert recorded in payload
    assert str(SOURCE_INTERSECTION.relative_to(REPO_ROOT)) in payload


@pytest.mark.skipif(not WEKA_INTERSECTION.is_file(), reason="Weka intersection CSV is not mounted")
def test_weka_mapping_matches_the_bundled_csv() -> None:
    assert sha256_file(WEKA_INTERSECTION) == sha256_file(SOURCE_INTERSECTION)
