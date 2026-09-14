"""GEOM-style COV/MAT helpers shared by the druglike evaluator and K-curves."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def best_rmsd(generated: Chem.Mol, reference: Chem.Mol) -> float:
    try:
        return float(rdMolAlign.GetBestRMS(generated, reference))
    except Exception:
        return float("nan")


def remove_hydrogens(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    cleaned = []
    for mol in mols:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        try:
            cleaned.append(Chem.RemoveHs(mol))
        except Exception:
            continue
    return cleaned


def compute_rmsd_matrices(
    ground_truth: dict, generated: dict, workers: int
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    matrices: dict[str, np.ndarray] = {}
    missing = []
    work = []
    for smiles, row in ground_truth.items():
        generated_mols = remove_hydrogens(generated.get(smiles, []))
        if not generated_mols:
            missing.append(smiles)
            continue
        references = remove_hydrogens(row.get("confs", []))
        matrix = np.full((len(references), len(generated_mols)), np.nan, dtype=np.float32)
        matrices[smiles] = matrix
        for reference_index, reference in enumerate(references):
            work.append((smiles, reference_index, reference, generated_mols))

    def compute_row(item):
        smiles, reference_index, reference, generated_mols = item
        values = np.asarray(
            [best_rmsd(mol, reference) for mol in generated_mols], dtype=np.float32
        )
        return smiles, reference_index, values

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(compute_row, item) for item in work]
        completed = 0
        for future in as_completed(futures):
            smiles, reference_index, values = future.result()
            matrices[smiles][reference_index] = values
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"RMSD rows: {completed}/{len(futures)}", flush=True)

    all_nan = [
        smiles
        for smiles, matrix in matrices.items()
        if matrix.size == 0 or np.isnan(matrix).all()
    ]
    return matrices, missing, all_nan


def finite_axis_min(matrix: np.ndarray, axis: int) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    finite = np.isfinite(matrix)
    replaced = np.where(finite, matrix, np.inf)
    values = replaced.min(axis=axis)
    values[~np.isfinite(values)] = np.nan
    return values


def per_molecule_metrics(
    matrix: np.ndarray, threshold: float, dmax: float
) -> dict[str, float | int]:
    valid = matrix[np.isfinite(matrix)]
    min_true = finite_axis_min(matrix, axis=1)
    min_generated = finite_axis_min(matrix, axis=0)
    finite_true = min_true[np.isfinite(min_true)]
    finite_generated = min_generated[np.isfinite(min_generated)]

    def coverage(values: np.ndarray) -> float:
        return float(np.mean(values < threshold)) if len(values) else float("nan")

    def matching(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    def censored_matching(values: np.ndarray, expected_size: int) -> float:
        censored = np.full(expected_size, dmax, dtype=float)
        if expected_size:
            finite_mask = np.isfinite(values)
            censored[finite_mask] = np.minimum(values[finite_mask], dmax)
        return float(np.mean(censored)) if expected_size else float("nan")

    return {
        "min_rmsd": float(valid.min()) if len(valid) else float("nan"),
        "max_rmsd": float(valid.max()) if len(valid) else float("nan"),
        "avg_rmsd": float(valid.mean()) if len(valid) else float("nan"),
        "cov_r_075": coverage(finite_true),
        "cov_p_075": coverage(finite_generated),
        "mat_r": matching(finite_true),
        "mat_p": matching(finite_generated),
        "cmat_r": censored_matching(min_true, matrix.shape[0]),
        "cmat_p": censored_matching(min_generated, matrix.shape[1]),
        "num_true_confs": int(matrix.shape[0]),
        "num_gen_confs": int(matrix.shape[1]),
        "num_valid_rmsd_pairs": int(len(valid)),
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    def stat(column: str, operation) -> float:
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce")
        return float(operation(values))

    return {
        "cov_r_mean": stat("cov_r_075", np.nanmean),
        "cov_r_median": stat("cov_r_075", np.nanmedian),
        "cov_p_mean": stat("cov_p_075", np.nanmean),
        "cov_p_median": stat("cov_p_075", np.nanmedian),
        "mat_r_mean": stat("mat_r", np.nanmean),
        "mat_r_median": stat("mat_r", np.nanmedian),
        "mat_p_mean": stat("mat_p", np.nanmean),
        "mat_p_median": stat("mat_p", np.nanmedian),
        "cmat_r_mean": stat("cmat_r", np.nanmean),
        "cmat_r_median": stat("cmat_r", np.nanmedian),
        "cmat_p_mean": stat("cmat_p", np.nanmean),
        "cmat_p_median": stat("cmat_p", np.nanmedian),
    }


def save_rmsd_matrices(path, matrices: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    smiles = np.asarray(list(matrices.keys()), dtype=object)
    payload = {f"m{index}": matrix for index, matrix in enumerate(matrices.values())}
    np.savez_compressed(path, smiles=smiles, **payload)


def load_rmsd_matrices(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        smiles = handle["smiles"].tolist()
        return {smi: handle[f"m{index}"] for index, smi in enumerate(smiles)}
