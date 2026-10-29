#!/usr/bin/env python3
"""Generate the paper cohorts from the final V2 CoordToken checkpoint.

The inference model directory is derived from the immutable training export.  The
export contains the right token IDs but retained the older mixed-data decoder hash.
The training corpus was encoded with the CASF-filtered decoder.  Bind the derived
inference directory to that corpus decoder contract, and add the three mirrored
provenance fields required by inference, without mutating the training artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_MODEL_EXPORT = Path(
    "/mnt/weka/mbedrosian/pretraining-runs/20261027-coordtoken-casf-v2/"
    "training/hf_exports/step-41054"
)
DEFAULT_COORDTOKEN_CHECKPOINT = Path(
    "/mnt/weka/fgeikyan/CoordToken/artifacts/releases/"
    "coordtoken-atom12-reg1-casf-epoch2-20261020/model.pt"
)
DEFAULT_COORDTOKEN_CONTRACT = Path(
    "/mnt/weka/fgeikyan/CoordToken/artifacts/releases/"
    "coordtoken-atom12-reg1-casf-epoch2-20261020/qwen_tokenizer/"
    "coordtoken_contract.json"
)
DEFAULT_SOURCE_TOKENIZER = Path(
    "/mnt/weka/fgeikyan/CoordToken/research/coordtoken_encoding_20261019/"
    "source/src/molgen3D/training/tokenizers/"
    "Qwen3_tokenizer_coordtoken_atom12_reg1"
)
DEFAULT_WORK_ROOT = Path(
    "/mnt/weka/mbedrosian/pharma_generation_analysis/v2_fsq_step41054_corrected"
)
DEFAULT_RUN_ID = "qwen_0p6b_v2_fsq_step41054"
DEFAULT_INPUTS = {
    "core": Path(
        "/mnt/weka/mbedrosian/data/casf16/"
        "casf16_core_chembl3d_exact_intersection.csv"
    ),
    "ref": Path(
        "/mnt/weka/mbedrosian/data/casf16/"
        "casf16_ref_chembl3d_exact_intersection.csv"
    ),
    "plinder": Path("/mnt/weka/vtarasov/druglike_smi.pickle"),
}
COORDTOKEN_FORMAT = "atom12_reg1_prefix_v1"

# The public paper panel is the established PLINDER-22 subset.  Bortezomib is the
# one molecule omitted by every existing Qwen row in that panel.
PLINDER22_EXCLUDED_SMILES = {
    "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_pickle_dump(path: Path, payload: object) -> None:
    import cloudpickle

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        cloudpickle.dump(payload, handle, protocol=4)
    os.replace(temporary, path)


def tokenizer_id_map(tokenizer_path: Path) -> dict[str, int]:
    """Return every token ID, rejecting contradictory tokenizer entries."""
    payload = read_json(tokenizer_path / "tokenizer.json")
    token_ids = dict(payload["model"]["vocab"])
    for entry in payload.get("added_tokens", []):
        token = entry["content"]
        token_id = entry["id"]
        if token in token_ids and token_ids[token] != token_id:
            raise ValueError(
                f"Tokenizer assigns two IDs to {token!r}: "
                f"{token_ids[token]} and {token_id}"
            )
        token_ids[token] = token_id
    return token_ids


def verify_source_tokenizer(
    model_export: Path,
    source_tokenizer: Path,
) -> dict:
    """Prove that the training wrapper is an append-only copy of the source.

    Training deliberately used a compatibility wrapper that appends the 256 old
    binned-coordinate tokens.  Inference must use that wrapper because it matches
    the model vocabulary, but every token from the requested CoordToken tokenizer
    must retain exactly the same ID.
    """
    model_export = model_export.resolve()
    source_tokenizer = source_tokenizer.resolve()
    manifest = read_json(model_export / "molgen3d_tokenizer_manifest.json")
    manifest_source = Path(manifest["source"]).resolve()
    if manifest_source != source_tokenizer:
        raise ValueError(
            "Training manifest names a different source tokenizer: "
            f"{manifest_source} != {source_tokenizer}"
        )

    source_sha256 = sha256(source_tokenizer / "tokenizer.json")
    source_metadata = manifest.get("source_metadata", {})
    if source_sha256 != source_metadata.get("dataset_tokenizer_json_sha256"):
        raise ValueError("Source tokenizer hash differs from the training manifest")
    if source_metadata.get("source_format") != COORDTOKEN_FORMAT:
        raise ValueError(
            f"Unexpected source tokenizer format: {source_metadata.get('source_format')}"
        )

    source_ids = tokenizer_id_map(source_tokenizer)
    wrapper_ids = tokenizer_id_map(model_export)
    changed = {
        token: (token_id, wrapper_ids.get(token))
        for token, token_id in source_ids.items()
        if wrapper_ids.get(token) != token_id
    }
    if changed:
        examples = list(changed.items())[:5]
        raise ValueError(f"Training wrapper changed source token IDs: {examples}")

    wrapper_only = {
        token: token_id for token, token_id in wrapper_ids.items() if token not in source_ids
    }
    expected_wrapper_only = {f"b{index:03d}": 151673 + index for index in range(256)}
    if wrapper_only != expected_wrapper_only:
        raise ValueError(
            "Training tokenizer is not the expected append-only binned-token wrapper"
        )

    return {
        "path": str(source_tokenizer),
        "sha256": source_sha256,
        "num_tokens": len(source_ids),
        "wrapper_extra_tokens": len(wrapper_only),
    }


def prepare_inference_model(
    source: Path,
    destination: Path,
    decoder_contract_path: Path,
    source_tokenizer_path: Path,
) -> tuple[Path, dict]:
    """Create a derived HF directory bound to the corpus' decoder contract."""
    source = source.resolve()
    source_tokenizer = verify_source_tokenizer(source, source_tokenizer_path)
    source_contract_path = source / "coordtoken_contract.json"
    source_contract = read_json(source_contract_path)
    contract = read_json(decoder_contract_path)
    if contract["format"] != COORDTOKEN_FORMAT:
        raise ValueError(f"Unexpected CoordToken format: {contract['format']}")
    # The two CoordToken releases deliberately share an LM alphabet.  Their sole
    # contract difference is the learned decoder checkpoint, whose code meanings
    # are not interchangeable.
    for key in source_contract:
        if key == "checkpoint_sha256":
            continue
        if source_contract[key] != contract.get(key):
            raise ValueError(f"LM and decoder contracts differ in {key}")
    contract["training_export_declared_checkpoint_sha256"] = source_contract[
        "checkpoint_sha256"
    ]
    tokenizer_manifest = read_json(source / "molgen3d_tokenizer_manifest.json")
    training_checkpoint_sha256 = tokenizer_manifest.get("source_metadata", {}).get(
        "coordtoken_checkpoint_sha256"
    )
    if training_checkpoint_sha256 != contract["checkpoint_sha256"]:
        raise ValueError(
            "Corpus decoder contract differs from the training tokenizer manifest"
        )
    contract["training_manifest_checkpoint_sha256"] = training_checkpoint_sha256
    contract["training_source_tokenizer"] = source_tokenizer
    tokenizer_sha256 = sha256(source / "tokenizer.json")
    if tokenizer_sha256 != contract["tokenizer_sha256"]:
        # The V2 exporter rewrote tokenizer.json after copying the contract.  Its
        # geometry IDs are unchanged (and are checked by the model vocabulary), but
        # the byte hash in the copied contract is stale.  Preserve that declaration
        # while binding the derived inference bundle to the actual exported bytes.
        contract["source_tokenizer_sha256"] = contract["tokenizer_sha256"]
        contract["tokenizer_sha256"] = tokenizer_sha256

    config = read_json(source / "config.json")
    for key in ("format", "checkpoint_sha256", "tokenizer_sha256"):
        config[f"coordtoken_{key}"] = contract[key]

    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"config.json", "coordtoken_contract.json"}:
            continue
        link = destination / item.name
        if link.is_symlink():
            if link.resolve() != item.resolve():
                raise ValueError(f"Wrong existing symlink: {link}")
        elif link.exists():
            if not link.samefile(item):
                raise ValueError(f"Unexpected file in derived model directory: {link}")
        else:
            link.symlink_to(item)

    config_path = destination / "config.json"
    if config_path.is_symlink():
        raise ValueError(f"Derived config must not be a symlink: {config_path}")
    atomic_json_dump(config_path, config)
    contract_path = destination / "coordtoken_contract.json"
    if contract_path.is_symlink():
        raise ValueError(f"Derived contract must not be a symlink: {contract_path}")
    atomic_json_dump(contract_path, contract)
    return destination, contract


def load_smiles(cohort: str, input_path: Path) -> list[str]:
    if cohort in {"core", "ref"}:
        with input_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            smiles = [row["chembl3d_isomeric_smiles"].strip() for row in rows]
    elif cohort == "plinder":
        import cloudpickle

        with input_path.open("rb") as handle:
            smiles = list(cloudpickle.load(handle))
        smiles = [smi for smi in smiles if smi not in PLINDER22_EXCLUDED_SMILES]
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    if any(not smi for smi in smiles):
        raise ValueError(f"Empty SMILES in {input_path}")
    # Stable length ordering exactly matches the previous V2 binned generation.
    return sorted(dict.fromkeys(smiles), key=len)


def select_shard(smiles: list[str], shard_index: int, num_shards: int) -> list[str]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    return smiles[shard_index::num_shards]


def output_dir(work_root: Path, run_id: str, cohort: str, shard_index: int, num_shards: int) -> Path:
    generation_root = work_root / "gen_results"
    if cohort == "ref" and num_shards > 1:
        return generation_root / "_ref_shards" / f"shard_{shard_index}"
    return generation_root / f"{run_id}_{cohort}"


def run_generation(args: argparse.Namespace) -> None:
    from loguru import logger
    from molgen3D.config.sampling_config import sampling_configs
    from molgen3D.evaluation.inference import generate_conformers, set_seed

    source_model = args.model_export.resolve()
    derived_model, contract = prepare_inference_model(
        source_model,
        args.work_root / "inference_model",
        args.coordtoken_contract.resolve(),
        args.source_tokenizer.resolve(),
    )
    if sha256(args.coordtoken_checkpoint) != contract["checkpoint_sha256"]:
        raise ValueError("CoordToken decoder checkpoint hash does not match the tokenizer")

    all_smiles = load_smiles(args.cohort, args.input)
    shard_smiles = select_shard(all_smiles, args.shard_index, args.num_shards)
    if args.limit_molecules is not None:
        shard_smiles = shard_smiles[: args.limit_molecules]
    prompts = [
        (smi, f"[SMILES]{smi}[/SMILES]")
        for smi in shard_smiles
        for _ in range(args.samples_per_molecule)
    ]
    destination = output_dir(
        args.work_root, args.run_id, args.cohort, args.shard_index, args.num_shards
    )
    metadata_path = destination / "generation_metadata.json"
    results_path = destination / "generation_results.pickle"
    if results_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if (
            metadata.get("num_molecules") == len(shard_smiles)
            and metadata.get("samples_per_molecule") == args.samples_per_molecule
        ):
            logger.info("Complete output already exists; skipping {}", destination)
            return
        raise ValueError(f"Existing output metadata differs: {destination}")

    destination.mkdir(parents=True, exist_ok=True)
    logger.add(destination / "logs.txt", rotation="50 MB")
    logger.info(
        "Generating cohort={} shard={}/{} molecules={} prompts={} output={}",
        args.cohort,
        args.shard_index,
        args.num_shards,
        len(shard_smiles),
        len(prompts),
        destination,
    )
    set_seed(args.seed + args.shard_index)
    inference_config = {
        "model_path": str(derived_model),
        "tokenizer_path": str(derived_model),
        "torch_dtype": "bfloat16",
        "attention_imp": "sdpa",
        "device": "cuda",
        "binned": False,
        "serialization_tag": "fsq",
        "coordtoken_format": contract["format"],
        "coordtoken_checkpoint": str(args.coordtoken_checkpoint),
        "coordtoken_sha256": contract["checkpoint_sha256"],
        "batch_size": args.batch_size,
        "gen_config": sampling_configs["top_p_sampling1"],
    }
    generations, stats = generate_conformers(inference_config, prompts)
    atomic_pickle_dump(results_path, dict(generations))
    with (destination / "generation_results.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"stats={stats}")

    metadata = {
        "batch_size": args.batch_size,
        "cohort": args.cohort,
        "coordtoken_checkpoint": str(args.coordtoken_checkpoint),
        "coordtoken_checkpoint_sha256": contract["checkpoint_sha256"],
        "coordtoken_contract": str(args.coordtoken_contract.resolve()),
        "coordtoken_format": contract["format"],
        "coordtoken_source_contract_sha256": sha256(
            source_model / "coordtoken_contract.json"
        ),
        "coordtoken_training_source_tokenizer": contract[
            "training_source_tokenizer"
        ],
        "coordtoken_source_tokenizer_sha256": contract.get(
            "source_tokenizer_sha256", contract["tokenizer_sha256"]
        ),
        "coordtoken_training_export_declared_checkpoint_sha256": contract[
            "training_export_declared_checkpoint_sha256"
        ],
        "coordtoken_training_manifest_checkpoint_sha256": contract[
            "training_manifest_checkpoint_sha256"
        ],
        "derived_model_path": str(derived_model),
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "model_path": str(source_model),
        "model_config_sha256": sha256(source_model / "config.json"),
        "model_tokenizer_sha256": contract["tokenizer_sha256"],
        "molgen_source": str(Path(__import__("molgen3D").__file__).resolve().parent.parent),
        "num_molecules": len(shard_smiles),
        "num_shards": args.num_shards,
        "run_id": args.run_id,
        "samples_per_molecule": args.samples_per_molecule,
        "seed": args.seed + args.shard_index,
        "serialization": "coordtoken",
        "shard_index": args.shard_index,
        "stats": dict(stats),
        "temperature": 1.0,
        "top_p": 1.0,
    }
    atomic_json_dump(metadata_path, metadata)


def merge_ref(args: argparse.Namespace) -> None:
    import cloudpickle

    merged: dict[str, list] = {}
    total_stats: Counter = Counter()
    shard_metadata = []
    for shard_index in range(args.num_shards):
        shard_dir = output_dir(args.work_root, args.run_id, "ref", shard_index, args.num_shards)
        metadata = read_json(shard_dir / "generation_metadata.json")
        with (shard_dir / "generation_results.pickle").open("rb") as handle:
            generations = cloudpickle.load(handle)
        overlap = set(merged).intersection(generations)
        if overlap:
            raise ValueError(f"Duplicate molecules across ref shards: {sorted(overlap)[:3]}")
        merged.update(generations)
        total_stats.update(metadata["stats"])
        shard_metadata.append(metadata)

    expected = load_smiles("ref", DEFAULT_INPUTS["ref"])
    if set(merged) != set(expected):
        raise ValueError(
            f"Merged ref population differs: expected={len(expected)}, actual={len(merged)}"
        )
    destination = args.work_root / "gen_results" / f"{args.run_id}_ref"
    atomic_pickle_dump(destination / "generation_results.pickle", merged)
    with (destination / "generation_results.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"stats={total_stats}")
    atomic_json_dump(
        destination / "generation_metadata.json",
        {
            "cohort": "ref",
            "num_molecules": len(merged),
            "num_shards": args.num_shards,
            "run_id": args.run_id,
            "samples_per_molecule": shard_metadata[0]["samples_per_molecule"],
            "shards": shard_metadata,
            "stats": dict(total_stats),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("cohort", choices=sorted(DEFAULT_INPUTS))
    generate.add_argument("--input", type=Path)
    generate.add_argument("--model-export", type=Path, default=DEFAULT_MODEL_EXPORT)
    generate.add_argument(
        "--coordtoken-checkpoint", type=Path, default=DEFAULT_COORDTOKEN_CHECKPOINT
    )
    generate.add_argument(
        "--coordtoken-contract", type=Path, default=DEFAULT_COORDTOKEN_CONTRACT
    )
    generate.add_argument(
        "--source-tokenizer", type=Path, default=DEFAULT_SOURCE_TOKENIZER
    )
    generate.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    generate.add_argument("--run-id", default=DEFAULT_RUN_ID)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--batch-size", type=int, default=256)
    generate.add_argument("--samples-per-molecule", type=int, default=1000)
    generate.add_argument("--limit-molecules", type=int)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=run_generation)

    merge = subparsers.add_parser("merge-ref")
    merge.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    merge.add_argument("--run-id", default=DEFAULT_RUN_ID)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.set_defaults(func=merge_ref)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate" and args.input is None:
        args.input = DEFAULT_INPUTS[args.cohort]
    args.func(args)


if __name__ == "__main__":
    main()
