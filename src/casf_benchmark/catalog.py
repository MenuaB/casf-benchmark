"""Canonical CASF generation family catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from casf_benchmark.paths import (
    DEFAULT_GENERATION_FAMILIES_CONFIG,
    DEFAULT_GENERATION_RUNS_CONFIG,
)

DEFAULT_CATALOG_PATH = DEFAULT_GENERATION_FAMILIES_CONFIG
TIERS = ("fixed", "dynamic", "chembl_count")
REFERENCE_TIER = "reference"


@dataclass(frozen=True)
class FamilySpec:
    id: str
    display_label: str
    core_root: Path | None
    ref_root: Path | None
    tiers: tuple[str, ...] = TIERS
    method_prefix: str | None = None
    generator: str = ""
    variant: str = ""
    hide_on_core: bool = False


@dataclass(frozen=True)
class ParsedMethod:
    family: str
    tier: str
    variant: str
    method: str


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(text)


def _load_catalog_data(catalog_path: Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"CASF catalog must be a mapping: {catalog_path}")
    return data


@lru_cache(maxsize=8)
def load_families(catalog_path: Path = DEFAULT_CATALOG_PATH) -> list[FamilySpec]:
    data = _load_catalog_data(Path(catalog_path))
    default_tiers = tuple(str(value) for value in data.get("tiers", TIERS))
    families = []
    for raw in data.get("families", []):
        if not isinstance(raw, dict):
            raise ValueError(f"CASF family entry must be a mapping: {raw!r}")
        family_id = str(raw["id"])
        families.append(
            FamilySpec(
                id=family_id,
                display_label=str(raw.get("display_label", family_id)),
                generator=str(raw.get("generator", "")),
                method_prefix=str(raw.get("method_prefix", family_id)),
                core_root=_path_or_none(raw.get("core_root")),
                ref_root=_path_or_none(raw.get("ref_root")),
                tiers=tuple(str(value) for value in raw.get("tiers", default_tiers)),
                variant=str(raw.get("variant") or (family_id.removeprefix("qwen_") if family_id.startswith("qwen_") else "")),
                hide_on_core=bool(raw.get("hide_on_core", False)),
            )
        )
    ids = [family.id for family in families]
    if len(ids) != len(set(ids)):
        raise ValueError("CASF catalog contains duplicate family id(s)")
    return families


@lru_cache(maxsize=8)
def load_reference_config(catalog_path: Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    data = _load_catalog_data(Path(catalog_path))
    reference = data.get("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("CASF catalog reference section must be a mapping")
    return reference


#: Descriptive fields a run entry may set explicitly, passed straight through to the
#: druglike result tables. Any generator can supply the ones that mean something for
#: it and omit the rest; nothing here is required.
RUN_DESCRIPTOR_FIELDS = ("generator", "display_label", "model_size", "tokenizer", "recipe", "step")

# Qwen checkpoint labels encode their own metadata, so the 12 bundled entries need no
# explicit descriptors. Labels that do not match simply fall back to the generic
# derivation in `describe_run`; they are not an error.
_QWEN_LABEL_PATTERN = re.compile(r"^qwen_(0p6b|1p7b|4b)_(fsq_)?(.+?)_step(\d+)$")
_QWEN_SIZE_NAMES = {"0p6b": "0.6B", "1p7b": "1.7B", "4b": "4B"}


@dataclass(frozen=True)
class GenerationRun:
    """One generation run: a label, its per-cohort output dirs, and how to describe it."""

    label: str
    cohorts: dict[str, str]
    descriptors: dict[str, object]


@lru_cache(maxsize=8)
def load_generation_run_entries(
    config_path: Path = DEFAULT_GENERATION_RUNS_CONFIG,
) -> tuple[GenerationRun, ...]:
    """Return every entry in the generation-runs catalog, in config order.

    The catalog is generator-agnostic: `label` is arbitrary text, so adding RDKit,
    loqi or anything else is a new YAML entry plus its CSVs, not a script change.
    """
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Generation runs config must be a mapping: {config_path}")
    entries = []
    for raw in data.get("runs", []):
        if not isinstance(raw, dict):
            raise ValueError(f"Generation run entry must be a mapping: {raw!r}")
        cohorts = raw.get("cohorts") or {}
        if not isinstance(cohorts, dict):
            raise ValueError(f"Generation run cohorts must be a mapping: {raw!r}")
        entries.append(
            GenerationRun(
                label=str(raw["label"]),
                cohorts={str(key): str(value) for key, value in cohorts.items() if value},
                descriptors={
                    field: raw[field] for field in RUN_DESCRIPTOR_FIELDS if raw.get(field) is not None
                },
            )
        )
    labels = [entry.label for entry in entries]
    if len(labels) != len(set(labels)):
        raise ValueError("Generation runs config has duplicate label(s)")
    return tuple(entries)


def load_generation_runs(
    cohort: str, config_path: Path = DEFAULT_GENERATION_RUNS_CONFIG
) -> tuple[tuple[str, str], ...]:
    """Return ordered (run label, inference output dirname) pairs for `cohort`.

    Runs that were not evaluated on `cohort` are omitted. Cohort names are the
    `cohorts` keys in the config, e.g. `casf16_core`, `casf16_ref`, `druglike`.
    """
    runs = [
        (entry.label, entry.cohorts[cohort])
        for entry in load_generation_run_entries(config_path)
        if cohort in entry.cohorts
    ]
    labels = [label for label, _ in runs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Generation runs config has duplicate label(s) for cohort {cohort!r}")
    return tuple(runs)


def describe_run(label: str, descriptors: dict[str, object] | None = None) -> dict[str, object]:
    """Return the descriptive columns for one run label.

    Explicit catalog descriptors win; otherwise Qwen checkpoint labels are parsed for
    the size/tokenizer/recipe/step they encode. A label that is neither -- any other
    generator -- still gets a usable `generator` and `display_label` and leaves the
    Qwen-specific fields empty, so mixed methods share one table.
    """
    descriptors = descriptors or {}
    match = _QWEN_LABEL_PATTERN.match(label)
    if match:
        derived: dict[str, object] = {
            "generator": "Qwen",
            "display_label": label,
            "model_size": _QWEN_SIZE_NAMES.get(match.group(1), match.group(1)),
            "tokenizer": "FSQ" if match.group(2) else "Binned",
            "recipe": match.group(3).replace("_", " "),
            "step": int(match.group(4)),
        }
    else:
        derived = {
            "generator": label.split("_", 1)[0],
            "display_label": label,
            "model_size": None,
            "tokenizer": None,
            "recipe": None,
            "step": None,
        }
    derived.update(descriptors)
    return derived


def hidden_on_core_family_ids(catalog_path: Path = DEFAULT_CATALOG_PATH) -> frozenset[str]:
    """Families the dashboard omits when ligand_set is core.

    Analysis still writes them; this is display-only so the PR #1 Qwen resamples
    do not sit next to the original core rows.
    """
    return frozenset(family.id for family in load_families(catalog_path) if family.hide_on_core)


def is_hidden_on_core(
    family_id: object,
    ligand_set: object,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> bool:
    return str(ligand_set) == "core" and str(family_id) in hidden_on_core_family_ids(catalog_path)


def drop_hidden_core_rows(frame: Any, catalog_path: Path = DEFAULT_CATALOG_PATH) -> Any:
    """Drop hide_on_core families from core rows. Ref and other sets are unchanged."""
    if getattr(frame, "empty", True):
        return frame
    columns = getattr(frame, "columns", [])
    if "family" not in columns or "ligand_set" not in columns:
        return frame
    hidden = hidden_on_core_family_ids(catalog_path)
    if not hidden:
        return frame
    keep = ~(
        (frame["ligand_set"].astype(str) == "core")
        & frame["family"].astype(str).isin(hidden)
    )
    return frame.loc[keep].reset_index(drop=True)


def family_by_id(family_id: str, catalog_path: Path = DEFAULT_CATALOG_PATH) -> FamilySpec:
    for family in load_families(catalog_path):
        if family.id == family_id:
            return family
    raise KeyError(f"Unknown CASF generation family: {family_id}")


def family_method_name(family_id: str, tier: str) -> str:
    if tier not in TIERS:
        raise ValueError(f"Unknown CASF generation tier {tier!r}; expected one of {TIERS}")
    return f"{family_id}_{tier}"


def parse_method_name(method: str) -> ParsedMethod:
    text = str(method)
    for tier in TIERS:
        suffix = f"_{tier}"
        if not text.endswith(suffix):
            continue
        family_id = text[: -len(suffix)]
        family = family_by_id(family_id)
        return ParsedMethod(
            family=family.id,
            tier=tier,
            variant=family.variant if family.id.startswith("qwen_") else "",
            method=text,
        )
    raise ValueError(f"Unknown or untiered CASF generation method: {method}")


def maybe_parse_method_name(method: object) -> ParsedMethod | None:
    try:
        return parse_method_name(str(method))
    except (KeyError, ValueError):
        return None


def families_for_ligand_set(ligand_set: str) -> list[FamilySpec]:
    if ligand_set not in {"core", "ref"}:
        raise ValueError(f"Unknown ligand_set {ligand_set!r}; expected 'core' or 'ref'")
    families = []
    for family in load_families():
        if ligand_set == "ref" and family.ref_root is None:
            continue
        families.append(family)
    return families


def resolve_generation_root(family_id: str, ligand_set: str) -> Path | None:
    family = family_by_id(family_id)
    if ligand_set == "core":
        return family.core_root
    if ligand_set == "ref":
        return family.ref_root
    raise ValueError(f"Unknown ligand_set {ligand_set!r}; expected 'core' or 'ref'")


def generation_method_order(ligand_set: str = "core") -> list[str]:
    return [
        family_method_name(family.id, tier)
        for family in families_for_ligand_set(ligand_set)
        for tier in family.tiers
    ]


def display_label_for_method(method: str) -> str:
    parsed = parse_method_name(method)
    family = family_by_id(parsed.family)
    return family.display_label
