"""Reproduce core's protocols on exact archived ref ensembles. Resume by hashes."""
from pathlib import Path
import argparse
import csv
import gzip
import concurrent.futures as futures
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from casf_benchmark.analysis.common_protocol import common_energy, ENERGY_PROTOCOL, sampling_tables, archived_retained_indices
from casf_benchmark.analysis.metrics import best_aligned_rmsd, forcefield_energy
from casf_benchmark.cli import analyze_conformer_sets as analysis
from casf_benchmark.paths import (DEFAULT_CHEMBL_DATASET_ROOT, DEFAULT_CHEMBL_REF_MAP_CSV,
                                 DEFAULT_REF_INTERSECTION_LIGAND_DIR)
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False)+'\n')
    tmp.replace(path)


def reference_mols(job):
    """Batch the original loader's coordinates; preserve stereo/order/capping."""
    import zarr
    from casf_benchmark.chembl3d import loader
    from casf_benchmark.chembl3d.identity import expected_identity_from_mapping
    row = pd.Series(job['mapping'])
    group = str(row.chembl3d_group).zfill(3)
    expected, index = expected_identity_from_mapping(row)
    topology = DEFAULT_CHEMBL_DATASET_ROOT/'topologies'
    template = loader.load_topology_mol(group, str(row.chembl3d_mol_id), topology,
                                       expected_smiles=expected, sdf_record_index=index)
    assert template is not None
    base = DEFAULT_CHEMBL_DATASET_ROOT/'zarr_database'/group
    indices = loader.find_mol_id_indices(zarr.open_array(str(base/'mol_id'), mode='r'), str(row.chembl3d_mol_id))
    coords = zarr.open_array(str(base/'coord'), mode='r').oindex[indices, :, :]
    numbers = zarr.open_array(str(base/'numbers'), mode='r').oindex[indices, :]
    expected_numbers = loader._topology_atomic_numbers(template)
    mols = []
    for idx, xyz, atoms in zip(indices, coords, numbers):
        loader._validate_atomic_numbers(atoms, expected_numbers, idx, str(row.chembl3d_mol_id))
        mol = loader._set_coords_from_row(template, xyz)
        if loader.mol_matches_expected_smiles(mol, expected, from_3d=True): mols.append(mol)
    mols, _ = analysis.sample_reference_chembl_mols(mols, job['mol_id'])
    old = np.array([forcefield_energy(m) for m in mols])
    assert len(old) == len(job['all_energies']), (job['mol_id'], len(old), len(job['all_energies']))
    np.testing.assert_allclose(old, job['all_energies'], rtol=1e-5, atol=1e-4)
    if len(mols) != job['n_retained']:
        # PB's stochastic energy check can change membership on replay. Recover
        # the exact saved ordered subset only from uniquely matching records,
        # after checking every source energy against its original coordinates.
        positions = archived_retained_indices(job['all_energies'], job['pb_energies'])
        mols = [mols[i] for i in positions]
        old = old[positions]
    np.testing.assert_allclose(old, job['pb_energies'], rtol=1e-5, atol=1e-4)
    assert len(mols) == job['n_retained']
    return mols


def task(job):
    RDLogger.DisableLog('rdApp.*')
    cache = Path(job['cache'])
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if cache.exists():
        old = json.loads(cache.read_text())
        if old['fingerprint'] == fingerprint: return str(cache)
    if job['method'] == 'chembl3d_gt_pb':
        mols = reference_mols(job)
    else:
        path = Path(job['sdf'])
        mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False)] if path.exists() and path.stat().st_size else []
    assert len(mols) == job['n_retained'] and all(m is not None for m in mols), (job['method'], job['mol_id'], len(mols), job['n_retained'])
    result = dict(fingerprint=fingerprint, method=job['method'], mol_id=job['mol_id'],
                  n_retained=len(mols), n_candidates=job['n_candidates'], source_sha256=job.get('sdf_sha256'),
                  reference_sha256=job.get('reference_sha256'), rdkit=rdBase.rdkitVersion)
    if job['stage'] == 'sampling':
        ref = analysis.load_casf_ligand(job['mol_id'], DEFAULT_REF_INTERSECTION_LIGAND_DIR)
        assert ref is not None
        values = np.array([best_aligned_rmsd(m, ref) for m in mols])
        finite = values[np.isfinite(values)]
        if job['best'] is None: assert not len(finite)
        else:
            assert len(finite)
            np.testing.assert_allclose([finite.min(), np.median(finite)], [job['best'], job['median']], rtol=1e-5, atol=1e-5)
        result['rmsd'] = [float(x) if np.isfinite(x) else None for x in values]
    else:
        records = []
        for i, mol in enumerate(mols):
            try: record = common_energy(mol)
            except AssertionError: raise
            except Exception as exc: record = dict(energy=None, status=type(exc).__name__+': '+str(exc))
            records.append(dict(conformer_index=i, **record))
        result['records'] = records
    atomic_json(cache, result)
    return str(cache)


def build_jobs(args, per, sources, methods):
    stage = args.stage
    selected = [m.replace('_chembl_count', '_fixed') for m in methods if m != 'chembl3d_gt_pb'] if stage == 'sampling' else methods
    if stage == 'energy': selected = [m for m in selected if m != 'chembl3d_gt_pb']
    if stage == 'reference': selected = ['chembl3d_gt_pb']
    cache = args.output/('sampling_cache' if stage == 'sampling' else 'energy_cache')
    cache.mkdir(parents=True, exist_ok=True)
    mapping, energies = {}, {}
    if stage == 'reference':
        mapping = pd.read_csv(DEFAULT_CHEMBL_REF_MAP_CSV, dtype=str).fillna('').set_index('ligand_id').to_dict('index')
        energy_path = Path(sources['reference_ref'])/'analysis/tables/geometric_energy_values.csv'
        e = pd.read_csv(energy_path)
        energies = {(m, ident): g.sort_values('conformer_index').energy.tolist() for (m, ident), g in e.groupby(['method', 'mol_id'])}
    fingerprint = dict(script=sha(__file__), metric=sha(Path(analysis.__file__).parents[1]/'analysis/metrics.py'),
                       common_protocol=sha(Path(analysis.__file__).parents[1]/'analysis/common_protocol.py'),
                       rdkit=rdBase.rdkitVersion, database=sha(args.database))
    if stage == 'reference': fingerprint.update(mapping=sha(DEFAULT_CHEMBL_REF_MAP_CSV), archived_energies=sha(energy_path))
    jobs = []
    for method in selected:
        rows = per[per.method.eq(method)].sort_values('mol_id')
        assert len(rows) == 1236 and rows.mol_id.nunique() == 1236, (method, len(rows))
        for row in rows.to_dict('records'):
            job = dict(method=method, mol_id=row['mol_id'], n_retained=int(row['post_pb_confs']),
                       n_candidates=int(row['pb_input_confs']), stage=stage, fingerprint=fingerprint,
                       cache=str(cache/(method+'__'+row['mol_id']+'.json')))
            if method == 'chembl3d_gt_pb':
                job.update(mapping=mapping[row['mol_id']], all_energies=energies['chembl3d_gt', row['mol_id']], pb_energies=energies[method, row['mol_id']])
            else:
                path = Path(sources[row['run_id']])/'generation'/method/(row['mol_id']+'.sdf')
                if job['n_retained']: assert path.exists(), path
                job.update(sdf=str(path), sdf_sha256=sha(path) if path.exists() else None)
            if stage == 'sampling':
                assert 0 <= job['n_retained'] <= job['n_candidates'] <= 1000
                job.update(best=float(row['casf_best_rmsd']) if pd.notna(row['casf_best_rmsd']) else None,
                           median=float(row['casf_median_rmsd']) if pd.notna(row['casf_median_rmsd']) else None)
                ref = analysis.load_casf_ligand(row['mol_id'], DEFAULT_REF_INTERSECTION_LIGAND_DIR)
                assert ref is not None
                job['reference_sha256'] = hashlib.sha256(ref.ToBinary()).hexdigest()
            jobs.append(job)
    return jobs


def validated_result(job):
    result = json.loads(Path(job['cache']).read_text())
    expected = hashlib.sha256(json.dumps(job, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert result['fingerprint'] == expected, (job['method'], job['mol_id'], 'stale calculation')
    assert (result['method'], result['mol_id'], result['n_retained'], result['n_candidates']) == (
        job['method'], job['mol_id'], job['n_retained'], job['n_candidates'])
    return result


def aggregate(args, per, labels):
    jobs = {}
    for stage in ['energy', 'reference']:
        for job in json.loads((args.output/(stage+'-jobs.json')).read_text()):
            jobs[job['method'], job['mol_id']] = job
    rows = []
    for label in labels.to_dict('records'):
        for old in per[per.method.eq(label['method'])].to_dict('records'):
            p = args.output/'energy_cache'/(label['method']+'__'+old['mol_id']+'.json')
            r = validated_result(jobs[label['method'], old['mol_id']])
            assert r['n_retained'] == old['post_pb_confs']
            values = np.array([x['energy'] for x in r['records'] if x['energy'] is not None])
            rows.append(dict(method=label['method'], mol_id=old['mol_id'], n_retained=r['n_retained'],
                             n_energy=len(values), n_failed=len(r['records'])-len(values),
                             mean_energy=values.mean() if len(values) else np.nan,
                             std_energy=values.std(ddof=0) if len(values) else np.nan,
                             min_energy=values.min() if len(values) else np.nan,
                             max_energy=values.max() if len(values) else np.nan,
                             median_energy=np.median(values) if len(values) else np.nan,
                             clusters_0p5=old['greedy_clusters_0p5'], archived_std=old['energy_std']))
    # Keep the numerical audit trail as a portable, compressed release artifact.
    with gzip.open(args.output/'per_conformer.csv.gz', 'wt', newline='') as stream:
        fields = ['method', 'mol_id', 'conformer_index', 'energy', 'status',
                  'original_hydrogens', 'explicit_hydrogens', 'max_heavy_coordinate_change']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            p = args.output/'energy_cache'/(row['method']+'__'+row['mol_id']+'.json')
            for record in json.loads(p.read_text())['records']:
                writer.writerow(dict(method=row['method'], mol_id=row['mol_id'], **record))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'per_molecule.csv', index=False)
    reference = frame[frame.method.eq('chembl3d_gt_pb')].set_index('mol_id').mean_energy
    summary = []
    for label in labels.to_dict('records'):
        all_rows = frame[frame.method.eq(label['method'])]
        r = all_rows[all_rows.n_energy.gt(0)]
        summary.append(dict(method=label['method'], label=label['display_label'], n_molecules=len(r),
                            n_conformers=int(r.n_energy.sum()), n_failed=int(all_rows.n_failed.sum()),
                            mean_clusters_0p5=r.clusters_0p5.mean(), median_energy_std=r.std_energy.median(),
                            mean_energy_std=r.std_energy.mean(), median_mean_energy=r.mean_energy.median(),
                            median_mean_energy_delta_chembl=(r.set_index('mol_id').mean_energy-reference).median(),
                            median_archived_std=r.archived_std.median()))
    pd.DataFrame(summary).to_csv(args.output/'summary.csv', index=False)
    sampling_stats = summarize_sampling(args, per, labels)
    cfg = json.loads((HERE/'sampling_protocol.json').read_text())
    atomic_json(args.output/'provenance.json', dict(database=str(args.database), database_sha256=sha(args.database), rdkit=rdBase.rdkitVersion,
                script_sha256=sha(__file__), energy_protocol=ENERGY_PROTOCOL,
                sampling_protocol={k:v for k,v in cfg.items() if k not in ['database_sha256', 'workers']},
                input_job_manifests={name:sha(args.output/(name+'-jobs.json')) for name in ['energy', 'reference', 'sampling']},
                n_energy=int(frame.n_energy.sum()), n_energy_failed=int(frame.n_failed.sum()),
                **sampling_stats,
                reference_validation='Original stereo-matched Zarr reconstruction; all archived energies checked pointwise; six filtered entries reconstructed by unique ordered archived energy identities at absolute tolerance 1e-9; retained energies checked pointwise. PB replay is not used because its stochastic energy check can change membership.'))


def summarize_sampling(args, per, labels):
    jobs = {(j['method'], j['mol_id']): j for j in json.loads((args.output/'sampling-jobs.json').read_text())}
    methods = [m.replace('_chembl_count', '_fixed') for m in labels.method if m != 'chembl3d_gt_pb']
    ids = sorted(per[per.method.eq('chembl3d_gt_pb')].mol_id)
    distances = np.full((len(methods), len(ids), 1000), np.inf)
    counts = np.zeros((len(methods), len(ids)), int)
    audit = []
    for mi, method in enumerate(methods):
        for i, ident in enumerate(ids):
            r = validated_result(jobs[method, ident])
            counts[mi, i] = r['n_candidates']
            distances[mi, i, :r['n_retained']] = [x if x is not None else np.inf for x in r['rmsd']]
            audit.append({k: v for k, v in r.items() if k != 'rmsd'})
    distances.sort(axis=2)
    refs = per[per.method.eq('chembl3d_gt_pb')].set_index('mol_id').reindex(ids).casf_best_rmsd.fillna(np.inf).to_numpy()
    np.savez_compressed(args.output/'candidate_rmsds.npz', rmsd=distances, n_candidates=counts,
                        methods=np.array(methods), mol_ids=np.array(ids), reference_best_rmsd=refs)
    pd.DataFrame(audit).to_csv(args.output/'pool_audit.csv', index=False)
    cfg = json.loads((HERE/'sampling_protocol.json').read_text())
    budgets = np.unique(np.rint(np.r_[np.geomspace(cfg['min_budget'], cfg['max_budget'], cfg['budget_grid_points']), cfg['explicit_budgets']]).astype(int))
    curves, frontier = sampling_tables(distances, counts, methods, budgets, cfg['rmsd_thresholds'], cfg['target_recovery'], cfg['bootstrap_replicates'], cfg['bootstrap_seed'])
    curves.to_csv(args.output/'recovery_curves.csv', index=False)
    frontier.to_csv(args.output/'recovery80_frontier.csv', index=False)
    for method in methods:
        old = per[per.method.eq(method)].casf_best_rmsd.fillna(np.inf).to_numpy()
        for threshold in cfg['rmsd_thresholds']:
            measured = curves[curves.method.eq(method) & curves.budget.eq(1000) & curves.threshold.eq(threshold)].recovery.item()
            assert np.isclose(measured, (old <= threshold).mean(), rtol=0, atol=1e-12)
    atomic_json(args.output/'reference.json', dict(hit_rates={str(t):float(np.mean(refs <= t)) for t in cfg['rmsd_thresholds']}, target_recovery=cfg['target_recovery'], rmsd_threshold=float(np.sort(refs)[int(np.ceil(len(ids)*cfg['target_recovery']))-1]), n_entries=len(ids)))
    return dict(n_entries=len(ids), n_sampling_retained=sum(r['n_retained'] for r in audit),
                n_rmsd_failed=sum(r['n_retained'] for r in audit)-int(np.isfinite(distances).sum()),
                n_candidates=int(counts.sum()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['energy', 'reference', 'sampling', 'sampling-summary', 'aggregate'], required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.database = args.database.resolve(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    assert rdBase.rdkitVersion == '2025.09.6', rdBase.rdkitVersion
    with sqlite3.connect(f'file:{args.database}?mode=ro', uri=True) as c:
        per = pd.read_sql_query("SELECT * FROM per_ligand_long WHERE ligand_set='ref'", c)
        sources = pd.read_sql_query('SELECT * FROM analysis_sources', c).set_index('run_id').root.to_dict()
    labels = pd.read_csv(HERE/'methods.csv')
    if args.stage == 'aggregate': aggregate(args, per, labels); return
    if args.stage == 'sampling-summary':
        print(summarize_sampling(args, per, labels), flush=True)
        return
    jobs = build_jobs(args, per, sources, labels.method.tolist())
    atomic_json(args.output/(args.stage+'-jobs.json'), jobs)
    print(f'{args.stage}: {len(jobs)} ensembles, {sum(j["n_retained"] for j in jobs)} retained conformers', flush=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(task, j): j for j in jobs}
        for i, f in enumerate(futures.as_completed(pending), 1):
            try: f.result()
            except Exception:
                print('FAILED', pending[f]['method'], pending[f]['mol_id'], flush=True)
                raise
            if i % 100 == 0: print(f'{args.stage}: {i}/{len(jobs)} completed', flush=True)
    print(f'{args.stage}: done', flush=True)

if __name__ == '__main__': main()
