"""Model-independent CASF ref crystal selection and derived-cohort rows.

One stereochemical SMILES keeps exactly one crystal. The choice is a SHA-256
rank of ``seed``, SMILES, and ``ligand_id``. It does not read model outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from casf_benchmark.paths import REPO_ROOT

SELECTION_VERSION = "casf-ref-dedup-v1"
SELECTION_SEED = "casf-ref-dedup-v1-20261030"
SELECTION_ALGORITHM = (
    "Within each casf_heavy_isomeric_smiles group, select the candidate with the "
    "smallest SHA256 hex digest of seed + tab + smiles + tab + ligand_id. "
    "Remaining ties break by ligand_id. Rank 1 is selected. "
    "The procedure does not read model outputs or RMSDs."
)
SOURCE_INTERSECTION = REPO_ROOT / "data" / "mapping" / "casf16_ref_chembl3d_exact_intersection.csv"
WEKA_INTERSECTION = Path("/mnt/weka/mbedrosian/data/casf16/casf16_ref_chembl3d_exact_intersection.csv")
DEFAULT_MANIFEST = REPO_ROOT / "data" / "mapping" / "casf16_ref_dedup_selection_v1.csv"
DEFAULT_PROVENANCE = REPO_ROOT / "data" / "mapping" / "casf16_ref_dedup_selection_v1.json"

EXPECTED_COUNTS = {
    "total_entries": 1236,
    "unique_smiles": 1044,
    "multi_crystal_compounds": 114,
    "entries_in_multi_crystal_compounds": 306,
    "selected_entries": 1044,
    "excluded_entries": 192,
}

MANIFEST_COLUMNS = [
    "ligand_id",
    "casf_heavy_isomeric_smiles",
    "source_file",
    "chembl3d_group",
    "chembl3d_mol_id",
    "chembl3d_isomeric_smiles",
    "chembl3d_sdf_record_index",
    "chembl3d_inchi_stereo",
    "conformer_count",
    "crystal_count_for_smiles",
    "selection_seed",
    "selection_digest",
    "selection_rank",
    "selected",
]

PRESERVED_MISSING_ROW_COLUMNS = (
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
    "mol_id",
    "rotatable_bonds",
    "heavy_atoms",
)


@dataclass(frozen=True)
class DerivedLigandSet:
    id: str
    label: str
    source_ligand_set: str
    selection_manifest: Path
    selection_key: str = "ligand_id"
    selected_column: str = "selected"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_digest(seed: str, smiles: str, ligand_id: str) -> str:
    payload = f"{seed}\t{smiles}\t{ligand_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def read_intersection(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"ligand_id", "casf_heavy_isomeric_smiles", "source_file"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["ligand_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ligand_id values")
    if (frame["ligand_id"] == "").any() or (frame["casf_heavy_isomeric_smiles"] == "").any():
        raise ValueError(f"{path} has blank ligand_id or SMILES values")
    return frame


def select_crystals(frame: pd.DataFrame, seed: str = SELECTION_SEED) -> pd.DataFrame:
    """Return one row per source crystal, with rank 1 marked selected."""
    if frame["ligand_id"].duplicated().any():
        raise ValueError("ligand_id values are not unique")
    out = frame.copy()
    out["selection_seed"] = seed
    out["selection_digest"] = [
        selection_digest(seed, smiles, ligand_id)
        for smiles, ligand_id in zip(out["casf_heavy_isomeric_smiles"], out["ligand_id"], strict=True)
    ]
    counts = out.groupby("casf_heavy_isomeric_smiles", sort=False)["ligand_id"].transform("size")
    out["crystal_count_for_smiles"] = counts.astype(int)
    out = out.sort_values(
        ["casf_heavy_isomeric_smiles", "selection_digest", "ligand_id"],
        kind="mergesort",
    )
    out["selection_rank"] = out.groupby("casf_heavy_isomeric_smiles", sort=False).cumcount() + 1
    out["selected"] = out["selection_rank"].eq(1)
    out = out.sort_values(
        ["casf_heavy_isomeric_smiles", "selection_rank", "ligand_id"],
        kind="mergesort",
    )
    return out.reset_index(drop=True)


def selection_counts(frame: pd.DataFrame) -> dict[str, int]:
    sizes = frame.groupby("casf_heavy_isomeric_smiles", sort=False)["ligand_id"].size()
    multi = sizes[sizes > 1]
    selected = int(frame["selected"].sum()) if "selected" in frame.columns else 0
    total = int(len(frame))
    return {
        "total_entries": total,
        "unique_smiles": int(sizes.shape[0]),
        "multi_crystal_compounds": int(multi.shape[0]),
        "entries_in_multi_crystal_compounds": int(multi.sum()),
        "selected_entries": selected,
        "excluded_entries": total - selected,
    }


def assert_selection(frame: pd.DataFrame, *, expected: dict[str, int] | None = EXPECTED_COUNTS) -> None:
    if frame["ligand_id"].duplicated().any():
        raise AssertionError("ligand_id values are not unique")
    counts = selection_counts(frame)
    if expected is not None:
        mismatches = {key: (counts[key], expected[key]) for key in expected if counts[key] != expected[key]}
        if mismatches:
            raise AssertionError(f"selection counts do not match: {mismatches}")
    chosen = frame[frame["selected"]]
    if chosen.groupby("casf_heavy_isomeric_smiles").size().ne(1).any():
        raise AssertionError("each SMILES must have exactly one selected crystal")
    if set(chosen["ligand_id"]) - set(frame["ligand_id"]):
        raise AssertionError("selected ligand_id values are not a subset of the source")
    if int(chosen["selection_rank"].eq(1).sum()) != len(chosen):
        raise AssertionError("selected rows must be rank 1")


def manifest_frame(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in MANIFEST_COLUMNS if column in selected.columns]
    out = selected[columns].copy()
    out["crystal_count_for_smiles"] = out["crystal_count_for_smiles"].astype(int)
    out["selection_rank"] = out["selection_rank"].astype(int)
    out["selected"] = out["selected"].map(lambda value: "true" if bool(value) else "false")
    return out


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame(frame).to_csv(path, index=False, lineterminator="\n")


def provenance_payload(source_csv: Path, frame: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {
        "selection_version": SELECTION_VERSION,
        "algorithm": SELECTION_ALGORITHM,
        "seed": SELECTION_SEED,
        "source_intersection_csv": str(source_csv.relative_to(REPO_ROOT) if source_csv.is_relative_to(REPO_ROOT) else source_csv),
        "source_intersection_sha256": sha256_file(source_csv),
        **selection_counts(frame),
    }
    if WEKA_INTERSECTION.is_file():
        weka_hash = sha256_file(WEKA_INTERSECTION)
        payload["weka_intersection_csv"] = str(WEKA_INTERSECTION)
        payload["weka_intersection_sha256"] = weka_hash
        payload["mappings_agree"] = weka_hash == payload["source_intersection_sha256"]
        if not payload["mappings_agree"]:
            raise AssertionError("local and Weka CASF ref intersection CSVs differ")
    return payload


def write_provenance(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_selection_artifacts(
    source_csv: Path = SOURCE_INTERSECTION,
    manifest_path: Path = DEFAULT_MANIFEST,
    provenance_path: Path = DEFAULT_PROVENANCE,
    seed: str = SELECTION_SEED,
) -> pd.DataFrame:
    selected = select_crystals(read_intersection(source_csv), seed=seed)
    assert_selection(selected)
    write_manifest(selected, manifest_path)
    write_provenance(provenance_payload(source_csv, selected), provenance_path)
    return selected


def _is_selected(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_selected_ids(path: Path, key: str = "ligand_id", selected_column: str = "selected") -> list[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if key not in frame.columns or selected_column not in frame.columns:
        raise ValueError(f"{path} must contain {key} and {selected_column}")
    chosen = frame.loc[frame[selected_column].map(_is_selected), key].astype(str)
    if chosen.empty:
        raise ValueError(f"{path} has no selected rows")
    return chosen.tolist()


def parse_derived_ligand_sets(config_path: Path) -> list[DerivedLigandSet]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"CASF analysis source config must be a mapping: {config_path}")
    specs: list[DerivedLigandSet] = []
    for entry in data.get("derived_ligand_sets") or []:
        if not isinstance(entry, dict):
            raise ValueError(f"derived ligand set must be a mapping: {entry!r}")
        manifest = Path(str(entry["selection_manifest"]))
        if not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        specs.append(
            DerivedLigandSet(
                id=str(entry["id"]),
                label=str(entry.get("label", entry["id"])),
                source_ligand_set=str(entry["source_ligand_set"]),
                selection_manifest=manifest,
                selection_key=str(entry.get("selection_key", "ligand_id")),
                selected_column=str(entry.get("selected_column", "selected")),
            )
        )
    return specs


def apply_derived_cohort(frame: pd.DataFrame, spec: DerivedLigandSet, selected_ids: list[str]) -> pd.DataFrame:
    """Copy selected source rows and keep every selected id in each method."""
    if frame.empty or "ligand_set" not in frame.columns or "mol_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    source = frame.loc[frame["ligand_set"].astype(str) == spec.source_ligand_set].copy()
    if source.empty:
        return source
    source["mol_id"] = source["mol_id"].astype(str)
    selected = list(dict.fromkeys(str(mol_id) for mol_id in selected_ids))
    selected_set = set(selected)
    kept = source.loc[source["mol_id"].isin(selected_set)].copy()
    group_cols = [column for column in ("run_id", "method", "tier", "row_type", "source") if column in source.columns]
    descriptor_cols = [column for column in ("rotatable_bonds", "heavy_atoms") if column in source.columns]
    donors = source.drop_duplicates("mol_id", keep="first").set_index("mol_id") if descriptor_cols else None
    identity_cols = [column for column in PRESERVED_MISSING_ROW_COLUMNS if column in source.columns and column not in descriptor_cols]
    missing_frames: list[pd.DataFrame] = []
    for _, group in source.groupby(group_cols, dropna=False, sort=False):
        present = set(group["mol_id"].astype(str))
        missing = [mol_id for mol_id in selected if mol_id not in present]
        if not missing:
            continue
        template = group.iloc[0]
        rows: list[dict[str, object]] = []
        for mol_id in missing:
            row: dict[str, object] = {column: pd.NA for column in source.columns}
            for column in identity_cols:
                row[column] = template[column]
            row["mol_id"] = mol_id
            if donors is not None and mol_id in donors.index:
                for column in descriptor_cols:
                    row[column] = donors.at[mol_id, column]
            rows.append(row)
        missing_frames.append(pd.DataFrame(rows, columns=source.columns))
    pieces = []
    if not kept.empty:
        pieces.append(kept)
    if missing_frames:
        extra = pd.concat(missing_frames, ignore_index=True).dropna(axis=1, how="all")
        if not extra.empty:
            pieces.append(extra)
    out = pd.concat(pieces, ignore_index=True, sort=False).reindex(columns=list(source.columns))
    out["ligand_set"] = spec.id
    if "run_id" in out.columns:
        suffix = f"__{spec.id}"
        run_ids = out["run_id"].astype("string")
        out["run_id"] = run_ids.where(run_ids.str.endswith(suffix), run_ids + suffix)
    return out


def apply_derived_ligand_sets(frame: pd.DataFrame, specs: list[DerivedLigandSet]) -> pd.DataFrame:
    if frame.empty or not specs:
        return frame
    derived: list[pd.DataFrame] = []
    for spec in specs:
        selected_ids = load_selected_ids(spec.selection_manifest, spec.selection_key, spec.selected_column)
        part = apply_derived_cohort(frame, spec, selected_ids)
        if not part.empty:
            derived.append(part)
    if not derived:
        return frame
    return pd.concat([frame, *derived], ignore_index=True, sort=False)
