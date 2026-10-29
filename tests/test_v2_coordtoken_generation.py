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


def test_default_decoder_and_contract_are_from_the_casf_release() -> None:
    module = load_generation_script()
    assert "coordtoken-atom12-reg1-casf-epoch2-20261020" in str(
        module.DEFAULT_COORDTOKEN_CHECKPOINT
    )
    assert (
        module.DEFAULT_COORDTOKEN_CONTRACT.parents[1]
        == module.DEFAULT_COORDTOKEN_CHECKPOINT.parent
    )
    assert module.DEFAULT_SOURCE_TOKENIZER.name == (
        "Qwen3_tokenizer_coordtoken_atom12_reg1"
    )


def test_inference_bundle_repairs_decoder_contract_and_stale_tokenizer_hash(
    tmp_path: Path,
) -> None:
    module = load_generation_script()
    source_tokenizer = tmp_path / "source-tokenizer"
    source_tokenizer.mkdir()
    source_tokenizer_payload = {
        "model": {"vocab": {"base": 0}},
        "added_tokens": [{"content": "<0>", "id": 147547}],
    }
    source_tokenizer_json = json.dumps(source_tokenizer_payload).encode()
    (source_tokenizer / "tokenizer.json").write_bytes(source_tokenizer_json)
    source_tokenizer_sha = hashlib.sha256(source_tokenizer_json).hexdigest()

    source = tmp_path / "source"
    source.mkdir()
    wrapper_payload = {
        **source_tokenizer_payload,
        "added_tokens": source_tokenizer_payload["added_tokens"]
        + [
            {"content": f"b{index:03d}", "id": 151673 + index}
            for index in range(256)
        ],
    }
    tokenizer = json.dumps(wrapper_payload).encode()
    (source / "tokenizer.json").write_bytes(tokenizer)
    (source / "config.json").write_text('{"model_type": "qwen3"}\n', encoding="utf-8")
    (source / "weights.safetensors").write_bytes(b"weights")
    source_contract = {
        "format": module.COORDTOKEN_FORMAT,
        "checkpoint_sha256": "wrong-decoder-sha",
        "tokenizer_sha256": "stale-sha",
    }
    (source / "coordtoken_contract.json").write_text(
        json.dumps(source_contract), encoding="utf-8"
    )
    (source / "molgen3d_tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "source_metadata": {
                    "coordtoken_checkpoint_sha256": "corpus-decoder-sha",
                    "dataset_tokenizer_json_sha256": source_tokenizer_sha,
                    "source_format": module.COORDTOKEN_FORMAT,
                },
                "source": str(source_tokenizer),
            }
        ),
        encoding="utf-8",
    )

    decoder_contract = {
        **source_contract,
        "checkpoint_sha256": "corpus-decoder-sha",
    }
    decoder_contract_path = tmp_path / "decoder-contract.json"
    decoder_contract_path.write_text(json.dumps(decoder_contract), encoding="utf-8")

    destination, contract = module.prepare_inference_model(
        source, tmp_path / "derived", decoder_contract_path, source_tokenizer
    )
    actual_tokenizer_sha = hashlib.sha256(tokenizer).hexdigest()
    assert contract["checkpoint_sha256"] == "corpus-decoder-sha"
    assert (
        contract["training_export_declared_checkpoint_sha256"]
        == "wrong-decoder-sha"
    )
    assert contract["training_manifest_checkpoint_sha256"] == "corpus-decoder-sha"
    assert contract["training_source_tokenizer"] == {
        "path": str(source_tokenizer),
        "sha256": source_tokenizer_sha,
        "num_tokens": 2,
        "wrapper_extra_tokens": 256,
    }
    assert contract["tokenizer_sha256"] == actual_tokenizer_sha
    assert contract["source_tokenizer_sha256"] == "stale-sha"
    assert json.loads((destination / "coordtoken_contract.json").read_text()) == contract
    config = json.loads((destination / "config.json").read_text())
    assert config["coordtoken_format"] == module.COORDTOKEN_FORMAT
    assert config["coordtoken_checkpoint_sha256"] == "corpus-decoder-sha"
    assert config["coordtoken_tokenizer_sha256"] == actual_tokenizer_sha
    assert (destination / "tokenizer.json").is_symlink()
    assert (destination / "weights.safetensors").is_symlink()
    assert json.loads((source / "coordtoken_contract.json").read_text()) == source_contract
