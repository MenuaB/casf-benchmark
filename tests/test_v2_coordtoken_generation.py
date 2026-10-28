from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from casf_benchmark.paths import REPO_ROOT


def load_generation_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_v2_coordtoken_generation.py"
    spec = importlib.util.spec_from_file_location("v2_coordtoken_generation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shards_are_disjoint_and_reconstruct_stable_length_order() -> None:
    module = load_generation_script()
    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
    shards = [module.select_shard(smiles, index, 4) for index in range(4)]
    assert set().union(*map(set, shards)) == set(smiles)
    assert sum(map(len, shards)) == len(smiles)
    assert shards[0] == ["C", "CCCCC"]


def test_inference_bundle_repairs_stale_tokenizer_hash_without_mutating_export(
    tmp_path: Path,
) -> None:
    module = load_generation_script()
    source = tmp_path / "source"
    source.mkdir()
    tokenizer = b"exported tokenizer bytes"
    (source / "tokenizer.json").write_bytes(tokenizer)
    (source / "config.json").write_text('{"model_type": "qwen3"}\n', encoding="utf-8")
    (source / "weights.safetensors").write_bytes(b"weights")
    source_contract = {
        "format": module.COORDTOKEN_FORMAT,
        "checkpoint_sha256": "decoder-sha",
        "tokenizer_sha256": "stale-sha",
    }
    (source / "coordtoken_contract.json").write_text(
        json.dumps(source_contract), encoding="utf-8"
    )

    destination, contract = module.prepare_inference_model(source, tmp_path / "derived")
    actual_tokenizer_sha = hashlib.sha256(tokenizer).hexdigest()
    assert contract["tokenizer_sha256"] == actual_tokenizer_sha
    assert contract["source_tokenizer_sha256"] == "stale-sha"
    assert json.loads((destination / "coordtoken_contract.json").read_text()) == contract
    config = json.loads((destination / "config.json").read_text())
    assert config["coordtoken_format"] == module.COORDTOKEN_FORMAT
    assert config["coordtoken_checkpoint_sha256"] == "decoder-sha"
    assert config["coordtoken_tokenizer_sha256"] == actual_tokenizer_sha
    assert (destination / "tokenizer.json").is_symlink()
    assert (destination / "weights.safetensors").is_symlink()
    assert json.loads((source / "coordtoken_contract.json").read_text()) == source_contract
