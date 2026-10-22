import itertools
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from casf_benchmark.analysis.common_protocol import common_energy, sampling_tables


def test_common_energy_rebuilds_h_without_moving_heavy_atoms_or_using_stored_energy():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    xyz = mol.GetConformer().GetPositions().copy()
    a = common_energy(mol)
    mol.SetProp('minimization_applied', 'true')
    mol.SetProp('post_min_energy', '999999')
    b = common_energy(mol)
    c = common_energy(Chem.RemoveHs(mol))
    assert a['status'] == 'evaluated'
    assert a['energy'] == b['energy'] == c['energy']
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), xyz)
    assert a['max_heavy_coordinate_change'] == 0


def test_sampling_matches_enumeration_including_rejections_empty_pools_and_short_pools():
    # Three retained candidates, one rejection; an empty entry; a short pool.
    pools = [[.1, .4, .9, np.inf], [], [.3, np.inf]]
    distances = np.array([[p + [np.inf]*(4-len(p)) for p in pools]])
    budgets = np.array([1, 2, 4])
    curves, frontier = sampling_tables(distances, np.array([[4, 0, 2]]), ['m'], budgets,
                                      [.25, .5, 1.], .5, bootstrap_replicates=20)
    for row in curves.itertuples():
        probabilities = []
        for p in pools:
            selections = list(itertools.combinations(p, min(row.budget, len(p))))
            probabilities.append(np.mean([any(x <= row.threshold for x in s) for s in selections]))
        assert row.recovery == pytest.approx(np.mean(probabilities))
    # At budget 1, even infinite tolerance cannot achieve 50% recovery.
    assert not frontier.iloc[0].attainable
    assert np.isnan(frontier.iloc[0].rmsd_threshold)
    assert frontier.iloc[-1].rmsd_threshold == .3


def test_empty_sampling_support_remains_unattainable():
    curves, frontier = sampling_tables(np.full((1, 2, 3), np.inf), np.array([[0, 3]]), ['m'],
                                      np.array([1, 3]), [.75], .8, bootstrap_replicates=20)
    assert curves.recovery.eq(0).all()
    assert frontier.rmsd_threshold.isna().all()
    assert not frontier.attainable.any()


def test_archived_subset_requires_unique_ordered_identities():
    from casf_benchmark.analysis.common_protocol import archived_retained_indices
    assert archived_retained_indices([1., 4., 2., 9.], [1., 2.]) == [0, 2]
    with pytest.raises(ValueError, match='possible source conformers'):
        archived_retained_indices([1., 1., 2.], [1.])
    with pytest.raises(ValueError, match='ordered subset'):
        archived_retained_indices([1., 2.], [2., 1.])
