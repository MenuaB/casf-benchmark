"""Paper energy and finite-pool sampling protocols, shared by core and ref."""
from __future__ import annotations

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import hypergeom

ENERGY_PROTOCOL = {
    'force_field': 'MMFF94s', 'nonbonded_threshold': 100.0,
    'ignore_interfragment_interactions': True,
    'heavy_coordinate_tolerance_angstrom': 1e-10,
    'std_ddof': 0, 'coordinate_optimization': False,
    'hydrogens': 'RemoveHs then AddHs(addCoords=True); no coordinate optimization',
}


def common_energy(mol):
    """Single-point MMFF94s after rebuilding H; never reuse stored energies."""
    heavy = Chem.RemoveHs(mol)
    xyz = np.asarray(heavy.GetConformer().GetPositions()).copy()
    original_h = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    if not np.isfinite(xyz).all():
        return dict(energy=None, status='nonfinite_coordinates', original_hydrogens=original_h)
    prepared = Chem.AddHs(heavy, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(prepared, mmffVariant=ENERGY_PROTOCOL['force_field'])
    if props is None:
        return dict(energy=None, status='missing_parameters', original_hydrogens=original_h)
    ff = AllChem.MMFFGetMoleculeForceField(
        prepared, props, nonBondedThresh=ENERGY_PROTOCOL['nonbonded_threshold'],
        ignoreInterfragInteractions=ENERGY_PROTOCOL['ignore_interfragment_interactions'])
    ff.Initialize()
    change = float(np.max(np.abs(prepared.GetConformer().GetPositions()[:len(xyz)] - xyz)))
    if change > ENERGY_PROTOCOL['heavy_coordinate_tolerance_angstrom']:
        raise AssertionError(f'Heavy coordinates changed by {change} Å')
    energy = float(ff.CalcEnergy())
    return dict(energy=energy if math.isfinite(energy) else None,
                status='evaluated' if math.isfinite(energy) else 'nonfinite_energy',
                original_hydrogens=original_h,
                explicit_hydrogens=prepared.GetNumAtoms()-heavy.GetNumAtoms(),
                max_heavy_coordinate_change=change)


def sampling_tables(distances, counts, methods, budgets, thresholds, target,
                    bootstrap_replicates=2000, bootstrap_seed=20261006):
    """Uniform subsets without replacement; rejected candidates remain non-hits.

    distances has sorted finite retained RMSDs followed by infinity. counts is
    the size of the pre-PoseBusters pool, not the number of retained conformers.
    Molecules (including empty pools) have equal weight.
    """
    import pandas as pd
    n = distances.shape[1]
    if n == 0 or distances.shape[:2] != counts.shape:
        raise ValueError('Empty or inconsistent sampling population')
    if np.any(counts < 0) or np.any(np.isfinite(distances).sum(axis=2) > counts):
        raise ValueError('Retained RMSDs exceed candidate pool')
    weights = np.random.default_rng(bootstrap_seed).multinomial(n, np.full(n, 1/n),
                                                               size=bootstrap_replicates)/n
    lookup = {N: (hypergeom.sf(0, int(N), np.arange(N+1)[None, :],
                             np.minimum(budgets, N)[:, None]) if N else
                  np.zeros((len(budgets), 1))) for N in np.unique(counts)}
    curves, frontiers = [], []
    for mi, method in enumerate(methods):
        arrays = [np.sort(row[np.isfinite(row)]) for row in distances[mi]]
        for threshold in thresholds:
            p = np.stack([lookup[counts[mi, i]][:, np.searchsorted(a, threshold, side='right')]
                          for i, a in enumerate(arrays)], axis=1)
            mean = p.mean(axis=1)
            low, high = np.quantile(weights @ p.T, [.025, .975], axis=0)
            assert np.all(np.diff(mean) >= -1e-12)
            curves.extend(dict(method=str(method), budget=int(k), threshold=float(threshold),
                               recovery=float(mean[j]), ci_low=float(low[j]), ci_high=float(high[j]))
                          for j, k in enumerate(budgets))
        support = np.unique(np.concatenate(arrays))
        if not len(support):
            frontiers.extend(dict(method=str(method), budget=int(k), rmsd_threshold=np.nan,
                                  target_recovery=target, recovery_at_threshold=0., attainable=False)
                             for k in budgets)
            continue
        def coverage(indices):
            ts = support[indices]
            return np.stack([lookup[counts[mi, i]][np.arange(len(budgets)),
                             np.searchsorted(a, ts, side='right')]
                             for i, a in enumerate(arrays)]).mean(axis=0)
        lo = np.zeros(len(budgets), int)
        hi = np.full(len(budgets), len(support)-1, int)
        attainable = coverage(hi) >= target
        while np.any(lo < hi):
            mid = (lo+hi)//2
            above = coverage(mid) >= target
            hi, lo = np.where(above, mid, hi), np.where(above, lo, mid+1)
        frontier = support[lo].copy()
        frontier[~attainable] = np.nan
        actual, previous = coverage(lo), coverage(np.maximum(lo-1, 0))
        assert np.all(actual[attainable] >= target)
        assert np.all(previous[attainable & (lo > 0)] < target)
        assert np.all(np.diff(frontier[np.isfinite(frontier)]) <= 1e-12)
        frontiers.extend(dict(method=str(method), budget=int(k), rmsd_threshold=float(frontier[j]),
                              target_recovery=target, recovery_at_threshold=float(actual[j]),
                              attainable=bool(attainable[j])) for j, k in enumerate(budgets))
    return pd.DataFrame(curves), pd.DataFrame(frontiers)


def archived_retained_indices(all_energies, retained_energies, tolerance=1e-9):
    """Recover an archived ordered subset only when every identity is unique.

    Callers must first validate all energies pointwise against the source
    coordinates. Ambiguous equal-energy records are rejected, never guessed.
    """
    all_values = np.asarray(all_energies, dtype=float)
    indices = []
    for value in retained_energies:
        matches = np.flatnonzero(np.isclose(all_values, value, rtol=0, atol=tolerance))
        if len(matches) != 1:
            raise ValueError(f'Archived energy {value} has {len(matches)} possible source conformers')
        indices.append(int(matches[0]))
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError('Archived retained sequence is not a unique ordered subset')
    return indices
