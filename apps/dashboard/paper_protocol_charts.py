"""Read-only views of the paper's comparable energy and exact sampling results."""
from __future__ import annotations
import sqlite3
import pandas as pd
import streamlit as st

METHOD_COLORS = {'ChEMBL3D-PB': '#333A43', 'RDKit raw': '#6F7782', 'RDKit minimized': '#6F7782', 'Torsion raw': '#987049', 'Torsion minimized': '#987049', 'LoQI': '#CE6428', 'Torsional Diffusion': '#AA589B', 'MCF drugs-L': '#998000', 'NExT-Mol DMT-L': '#399BBB', 'FlowR': '#00866B', 'Qwen 1.7B FSQ': '#0072B2'}

DEFAULT_CURVES = ['Qwen 1.7B FSQ', 'LoQI', 'FlowR', 'NExT-Mol DMT-L']


@st.cache_data(show_spinner=False)
def load_protocol_table(path: str, table: str, mtime_ns: int) -> pd.DataFrame:
    allowed = {'paper_energy_summary', 'paper_energy_per_ligand', 'paper_sampling_curves',
               'paper_sampling_frontier', 'paper_sampling_reference', 'paper_protocol_runs'}
    if table not in allowed:
        raise ValueError(table)
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        exists = con.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?', (table,)).fetchone()
        return pd.read_sql_query(f'SELECT * FROM {table}', con) if exists else pd.DataFrame()


def cohort_table(path, table, cohort):
    frame = load_protocol_table(str(path), table, path.stat().st_mtime_ns)
    return frame[frame.ligand_set.eq(cohort)].copy() if not frame.empty else frame


def render_energy(path, cohort):
    import altair as alt
    summary = cohort_table(path, 'paper_energy_summary', cohort)
    if summary.empty:
        st.info('Common-protocol energy results are not available for this cohort.')
        return
    st.subheader('Common-protocol energy · ChEMBL-count target')
    st.caption('MMFF94s single-point energies after removing and rebuilding hydrogens; '
               'no coordinate optimization. The paper’s ten generators and stored ChEMBL3D-PB '
               'are evaluated with the same protocol. These panels use the complete selected '
               'dataset; the sidebar tier, family, and breakdown filters apply to the archived views.')
    columns = st.columns(2)
    for column, metric, title in zip(columns, ['mean_clusters_0p5', 'median_energy_std'],
                                    ['Mean clusters at 0.5 Å', 'Median energy SD (kcal/mol)']):
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X(f'{metric}:Q', title=title), y=alt.Y('label:N', sort=summary.label.tolist(), title=None),
            color=alt.Color('label:N', legend=None, scale=alt.Scale(domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values()))),
            tooltip=['label:N', alt.Tooltip(f'{metric}:Q', format='.3f'), 'n_molecules:Q', 'n_conformers:Q', 'n_failed:Q']
        ).properties(height=340)
        column.altair_chart(chart, use_container_width=True)
    st.caption('Energy dispersion is the median across molecules of the population SD within each ensemble. '
               'Empty ensembles and failed energy evaluations are excluded from energy summaries and counted explicitly.')
    st.dataframe(summary[['label', 'n_molecules', 'n_conformers', 'n_failed', 'median_energy_std',
                          'mean_energy_std', 'median_mean_energy_delta_chembl']].rename(columns={
        'label': 'Method', 'n_molecules': 'Molecules', 'n_conformers': 'Conformers',
        'n_failed': 'Failed energies', 'median_energy_std': 'Median energy SD',
        'mean_energy_std': 'Mean energy SD', 'median_mean_energy_delta_chembl': 'Median paired Δmean energy'
    }), hide_index=True, use_container_width=True)
    st.download_button('Download energy summary', summary.to_csv(index=False), f'{cohort}-common-energy-summary.csv', 'text/csv')
    per = cohort_table(path, 'paper_energy_per_ligand', cohort)
    st.download_button('Download per-molecule energies', per.to_csv(index=False), f'{cohort}-common-energy-per-molecule.csv', 'text/csv')


def render_sampling(path, cohort):
    import altair as alt
    curves = cohort_table(path, 'paper_sampling_curves', cohort)
    frontier = cohort_table(path, 'paper_sampling_frontier', cohort)
    reference = cohort_table(path, 'paper_sampling_reference', cohort)
    if curves.empty or frontier.empty:
        st.info('Exact sampling curves are not available for this cohort.')
        return
    st.subheader('Exact sampling curves · 1,000-candidate target')
    st.caption('Uniform sampling without replacement from each saved pre-PoseBusters pool. '
               'Rejected candidates and failed RMSDs consume budget and count as non-hits. '
               'Budgets are capped at each pool’s available candidates. All molecules, including '
               'empty pools, receive equal weight. These are complete-dataset curves; sidebar '
               'tier, family, and breakdown filters apply to the archived views.')
    labels = curves.label.drop_duplicates().tolist()
    selected = st.multiselect('Methods', labels, default=[x for x in DEFAULT_CURVES if x in labels], key='paper_sampling_methods')
    threshold = st.select_slider('Recovery RMSD threshold (Å)', options=sorted(curves.threshold.unique()), value=.75)
    show_ci = st.checkbox('Show 95% molecule-bootstrap intervals', value=False)
    data = curves[curves.label.isin(selected) & curves.threshold.eq(threshold)].copy()
    data[['recovery', 'ci_low', 'ci_high']] *= 100
    front = frontier[frontier.label.isin(selected)].copy()
    color = alt.Color('label:N', title='Method', scale=alt.Scale(domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values())))
    left = alt.Chart(data).mark_line().encode(x=alt.X('budget:Q', title='Candidate budget'),
            y=alt.Y('recovery:Q', title='Expected recovery (%)', scale=alt.Scale(domain=[0, 100])), color=color,
            tooltip=['label:N', 'budget:Q', alt.Tooltip('recovery:Q', format='.2f')])
    if show_ci:
        band = alt.Chart(data).mark_area(opacity=.12).encode(x='budget:Q', y='ci_low:Q', y2='ci_high:Q', color=color)
        left = band + left
    right = alt.Chart(front).mark_line().encode(x=alt.X('budget:Q', title='Candidate budget'),
            y=alt.Y('rmsd_threshold:Q', title='RMSD threshold for 80% recovery (Å)'), color=color,
            tooltip=['label:N', 'budget:Q', alt.Tooltip('rmsd_threshold:Q', format='.3f'), 'attainable:N'])
    if not reference.empty:
        r = reference[reference.threshold.eq(threshold)].iloc[0]
        for side, value in [('left', 100*r.recovery), ('right', r.rmsd_threshold)]:
            line = alt.Chart(pd.DataFrame({'value': [value]})).mark_rule(strokeDash=[5, 4], color='#444444').encode(y='value:Q')
            if side == 'left': left = left + line
            else: right = right + line
    a, b = st.columns(2)
    a.altair_chart(left.properties(height=330), use_container_width=True)
    b.altair_chart(right.properties(height=330), use_container_width=True)
    st.caption('Dashed lines: each dataset’s full stored ChEMBL3D-PB ensemble. '
               'The 80% threshold is inverted on the observed RMSDs; unattainable thresholds remain missing. '
               'Confidence intervals use 2,000 shared molecule bootstrap resamples, seed 20261006.')
    st.download_button('Download recovery curves', curves.to_csv(index=False), f'{cohort}-sampling-recovery.csv', 'text/csv')
    st.download_button('Download 80% thresholds', frontier.to_csv(index=False), f'{cohort}-sampling-thresholds.csv', 'text/csv')
