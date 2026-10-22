"""Attach validated protocol-specific tables to a snapshot, atomically.

Archived per_ligand_long energies intentionally retain their original protocol.
The public Energy view reads paper_energy_* by default; archived energy remains
an explicitly separate view. Never mix the two hydrogen treatments in a mean.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sqlite3
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def build_database(database, output, core_energy, core_sampling, ref_results):
    tables = {name: [] for name in ['paper_energy_per_ligand', 'paper_energy_summary',
              'paper_sampling_curves', 'paper_sampling_frontier', 'paper_sampling_reference', 'paper_protocol_runs']}
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as con:
        archived = pd.read_sql_query('SELECT ligand_set,method,mol_id,post_pb_confs,greedy_clusters_0p5,casf_best_rmsd,pb_input_confs FROM per_ligand_long', con)
    for cohort, energy_dir, sampling_dir, n in [('core', core_energy, core_sampling, 94), ('ref', ref_results, ref_results, 1236)]:
        energy = pd.read_csv(energy_dir/'per_molecule.csv')
        summary = pd.read_csv(energy_dir/'summary.csv')
        curves = pd.read_csv(sampling_dir/'recovery_curves.csv')
        frontier = pd.read_csv(sampling_dir/'recovery80_frontier.csv')
        reference = json.loads((sampling_dir/'reference.json').read_text())
        assert len(summary) == 11 and len(energy) == 11*n
        assert energy.groupby('method').mol_id.nunique().eq(n).all()
        assert not energy.duplicated(['method', 'mol_id']).any()
        paired = energy.merge(archived[archived.ligand_set.eq(cohort)], on=['method', 'mol_id'], validate='one_to_one')
        assert len(paired) == len(energy)
        np.testing.assert_array_equal(paired.n_retained, paired.post_pb_confs)
        np.testing.assert_allclose(paired.clusters_0p5, paired.greedy_clusters_0p5, equal_nan=True)
        assert (energy.n_energy + energy.n_failed).eq(energy.n_retained).all()
        labels = summary.set_index('method').label.to_dict()
        fixed_labels = {k.replace('_chembl_count', '_fixed'): v for k, v in labels.items() if k != 'chembl3d_gt_pb'}
        assert set(curves.method) == set(fixed_labels) == set(frontier.method)
        for row in summary.itertuples():
            per = energy[energy.method.eq(row.method)]
            valid = per[per.n_energy.gt(0)]
            assert row.n_molecules == len(valid)
            assert row.n_conformers == valid.n_energy.sum()
            assert row.n_failed == per.n_failed.sum()
            assert np.isclose(row.median_energy_std, valid.std_energy.median())
        for method in curves.method.unique():
            rows = archived[archived.ligand_set.eq(cohort) & archived.method.eq(method)]
            assert len(rows) == n and rows.mol_id.nunique() == n
            for threshold in curves.threshold.unique():
                endpoint = curves[curves.method.eq(method) & curves.threshold.eq(threshold) & curves.budget.eq(1000)].recovery.item()
                assert np.isclose(endpoint, rows.casf_best_rmsd.fillna(np.inf).le(threshold).mean(), atol=1e-12, rtol=0)
            endpoint = frontier[frontier.method.eq(method) & frontier.budget.eq(1000)].rmsd_threshold.item()
            expected = np.sort(rows.casf_best_rmsd.fillna(np.inf))[int(np.ceil(n*.8))-1]
            assert np.isclose(endpoint, expected, atol=1e-12, rtol=0)
        energy['label'] = energy.method.map(labels)
        curves['label'] = curves.method.map(fixed_labels)
        frontier['label'] = frontier.method.map(fixed_labels)
        reference_rows = pd.DataFrame([dict(threshold=float(t), recovery=v, rmsd_threshold=reference['rmsd_threshold'],
                                 target_recovery=reference['target_recovery'], n_entries=reference['n_entries'])
                                 for t, v in reference['hit_rates'].items()])
        for name, frame in [('paper_energy_per_ligand', energy), ('paper_energy_summary', summary),
                            ('paper_sampling_curves', curves), ('paper_sampling_frontier', frontier),
                            ('paper_sampling_reference', reference_rows)]:
            frame['ligand_set'] = cohort
            tables[name].append(frame)
        provenance_paths = [energy_dir/'provenance.json', sampling_dir/'extraction-provenance.json'] if cohort == 'core' else [ref_results/'provenance.json']
        provenance = {str(p): json.loads(p.read_text()) for p in provenance_paths}
        input_files = [energy_dir/'per_molecule.csv', energy_dir/'summary.csv', sampling_dir/'recovery_curves.csv', sampling_dir/'recovery80_frontier.csv', sampling_dir/'reference.json']
        tables['paper_protocol_runs'].append(pd.DataFrame([dict(ligand_set=cohort, n_entries=n,
               protocol='common-H-MMFF94s-v1; exact-pre-PB-finite-pool-v1',
               provenance_json=json.dumps(provenance), source_hashes_json=json.dumps({str(p):sha(p) for p in input_files}))]))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix('.tmp.sqlite')
    staging.unlink(missing_ok=True)
    try:
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as source, sqlite3.connect(staging) as dest:
            source.backup(dest)
            for table, frames in tables.items():
                pd.concat(frames, ignore_index=True).to_sql(table, dest, if_exists='replace', index=False)
            dest.execute('CREATE UNIQUE INDEX IF NOT EXISTS paper_energy_identity ON paper_energy_per_ligand(ligand_set,method,mol_id)')
            dest.execute('CREATE UNIQUE INDEX IF NOT EXISTS paper_sampling_identity ON paper_sampling_curves(ligand_set,method,budget,threshold)')
            dest.execute('CREATE UNIQUE INDEX IF NOT EXISTS paper_frontier_identity ON paper_sampling_frontier(ligand_set,method,budget)')
            assert dest.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            for (table,) in source.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'paper_%'"):
                assert source.execute(f'SELECT count(*) FROM "{table}"').fetchone() == dest.execute(f'SELECT count(*) FROM "{table}"').fetchone()
        os.replace(staging, output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return {table: sum(len(f) for f in frames) for table, frames in tables.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ['database', 'output', 'core-energy', 'core-sampling', 'ref-results']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build_database(a.database, a.output, a.core_energy, a.core_sampling, a.ref_results), indent=2))
