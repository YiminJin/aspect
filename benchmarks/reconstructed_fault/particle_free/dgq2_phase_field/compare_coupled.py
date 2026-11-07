#!/usr/bin/env python3
"""Compare the four native DG phase cases with preserved Q1 and each other.

The FE families intentionally have different initial representations. No assertion
of identical Q1/DG initialization is made. Pair norms use the existing union-cell
bulk and mesh-crossing surface quadratures, with central/end regions retained.
Run only after the sequential simulations/reductions have completed.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-dg-phase-mpl')
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SUPPORT = Path(__file__).resolve().parents[1] / 'transported_objective'
sys.path.insert(0, str(SUPPORT))
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled'
os.environ['LAYERED_EVIDENCE'] = str(OUT)
from assess_layered_campaign import (load_case, state_at, compare_records,
                                     compare_final, common_bulk_quadrature)
from layered_fields import powers, basis, dot3, localization
from assess_sheared_moving_fault import metric
OLD = OUT.parent.parent / 'transported_objective/layered_bulk_campaign'
MATCHES = [('plus', 'plus'), ('plus-half', 'plus-half'), ('minus', 'minus'),
           ('plus-fine', 'plus-fine-cartesian')]


def read_case(name):
    directory = OUT if name.startswith('layered-dgphase-') else OLD
    return json.loads((directory / (name + '.json')).read_text())


def phase_log(path):
    """Native SSPRK stage-weighted material balance, including divergence."""
    records = []
    for line in (path / 'run.log').read_text().splitlines():
        if 'DG phase transport:' in line:
            fields = line.split('DG phase transport:', 1)[1].strip().split(', ')
            records.append({k: float(v) for k, v in (f.split('=') for f in fields)})
    if not records:
        return dict(records=[], available=False, signed_balance=None,
                    absolute_balance=None, source=None, boundary_flux=None,
                    substeps=None, limited_cell_stages=None, minimum_alpha=None)
    return dict(records=records, available=True,
                signed_balance=sum(r['balance error'] for r in records),
                absolute_balance=sum(abs(r['balance error']) for r in records),
                source=sum(r['source'] for r in records),
                boundary_flux=sum(r['boundary flux'] for r in records),
                substeps=sum(r['substeps'] for r in records),
                limited_cell_stages=sum(r['limited cells'] for r in records),
                minimum_alpha=min((r['minimum alpha'] for r in records), default=1.))


def timers(path):
    result = {}
    for line in (path / 'run.log').read_text().splitlines():
        m = re.match(r'\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*([\d.e+\-]+)s\s*\|\s*([\d.e+\-]+)%', line)
        if m:
            result[m[1]] = dict(calls=int(m[2]), seconds=float(m[3]), percent=float(m[4]))
    return result


def memory_samples(name):
    path = OUT / 'runtime-memory.jsonl'
    samples = []
    if path.exists():
        for line in path.read_text().splitlines():
            ranks = [r for r in json.loads(line)['ranks']
                     if '/'+name+'/input.prm' in r['command']]
            if ranks:
                samples.append(ranks)
    return dict(samples=len(samples),
                maximum_concurrent_sampled_rss_MiB=max((sum(r['rss_kib'] for r in s)/1024 for s in samples), default=None),
                maximum_sum_rank_high_water_MiB=max((sum(r['hwm_kib'] for r in s)/1024 for s in samples), default=None),
                caveat='RSS counts shared pages per rank; rank high-water marks need not coincide. Old Q1 memory was not sampled.')


def initial_reference(s, order):
    x, w = s.quadrature(order)
    p = .35 * np.maximum(0, 1-(x[:, 0]/1.4)**2)**4 * np.maximum(0, 1-((x[:, 1]-.6*x[:, 0])/.6)**2)**4
    phi = s.at(x)[:, 0]
    return dict(phase=metric((phi-p)**2, p*p, w), exact_integral=float(w@p),
                analytic_integral=.35*1.4*.6*(256/315)**2,
                represented_integral=float(w@phi))


def sampled_extrema(s):
    # Evaluate each cell polynomial, including both traces at a common face.
    # These are samples; the native Bernstein limiter supplies the guarantee.
    g = np.linspace(-1, 1, 9)
    q = np.stack(np.meshgrid(g, g, indexing='ij'), axis=-1).reshape(-1, 2)
    vals = s.coeff[..., 0].reshape(-1, 9) @ basis(q).T
    return dict(minimum=float(vals.min()), maximum=float(vals.max()),
                points_per_cell=len(q))


def summarize(case):
    path, cmd, rows = load_case(case['case'])
    initial = state_at(path, cmd, int(rows[0]['step']))
    final = state_at(path, cmd, int(rows[-1]['step']))
    e0, ef = case['records'][0]['energy'], case['records'][-1]['energy']
    source_trap = flux_trap = 0.
    rem = []
    for a, b in zip(case['records'][:-1], case['records'][1:]):
        dt = b['dt']; ea, eb = a['energy'], b['energy']
        source_trap += .5*dt*(ea['phase_dilatation']+eb['phase_dilatation'])
        flux_trap += .5*dt*(ea['phase_flux']+eb['phase_flux'])
        rem.append(eb['phase_transport_remainder'])
    native = phase_log(path)
    assert all(r['retries'] == 0 for r in case['records']), 'Account for retried diagnostics separately'
    if native['records']:
        assert len(native['records']) == len(rows)-1
    entries = dict(case=case['case'], arguments=cmd['arguments'], wall_s=case['wall_s'],
        accepted_steps=len(rows)-1, initial_q12=initial_reference(initial, 12),
        initial_q16=initial_reference(initial, 16), initial=sampled_extrema(initial),
        final=sampled_extrema(final), initial_energy=e0, final_energy=ef,
        phase_integral_relative=ef['profile_mass']/e0['profile_mass']-1,
        phase_trapezoid_source=source_trap, phase_trapezoid_flux=flux_trap,
        phase_trapezoid_signed_remainder=sum(rem),
        phase_trapezoid_absolute_remainder=sum(abs(v) for v in rem),
        native_stage_balance=native, energy=case['energy'], cost=case['cost'], timers=timers(path),
        memory=memory_samples(case['case']),
        phase_dofs=int(np.prod(final.shape)*9 if native['records'] else np.prod(final.shape+1)),
        cells=int(np.prod(final.shape)),
        initial_phase_mean=float(np.mean(initial.coeff[..., 0, 0])),
        final_I=final.surface[:, 5].tolist(), final_slip_rate=final.surface[:, 6].tolist(),
        checks=case['checks'])
    entries['reaction_projection_loss'] = sum(r['loss'] for r in case['stage_energy']['reaction'])
    entries['reaction_projection_absolute_loss'] = sum(abs(r['loss']) for r in case['stage_energy']['reaction'])
    entries['transport_stage_energy'] = case['stage_energy']['transport']
    return entries


def compare(a, b, same_initial=False):
    shift = a['arguments']['translation'] != b['arguments']['translation']
    pa, ca, ra = load_case(a['case']); pb, cb, rb = load_case(b['case'])
    sa = state_at(pa, ca, 0); sb = state_at(pb, cb, 0)
    x, w = common_bulk_quadrature(sa, sb, 0.)
    va, vb = sa.at(x), sb.at(x)
    initial = dict(phase=metric((va[:, 0]-vb[:, 0])**2, vb[:, 0]**2, w),
                   stress=metric(dot3(va[:, 3:6]-vb[:, 3:6], va[:, 3:6]-vb[:, 3:6]),
                                 dot3(vb[:, 3:6], vb[:, 3:6]), w),
                   surface_Linf=float(np.max(abs(sa.surface-sb.surface))))
    initial['surface_fields'] = {}
    for key, columns in [('geometry', [1, 2]), ('age', [3]),
                         ('normalization', [5]), ('slip_rate', [6])]:
        difference = sa.surface[:, columns]-sb.surface[:, columns]
        reference = sb.surface[:, columns]
        initial['surface_fields'][key] = dict(
            absolute_Linf=float(np.max(abs(difference))),
            relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(reference))
            if np.linalg.norm(reference) else None)
    same_mesh = (sa.coeff.shape==sb.coeff.shape
                 and np.array_equal(sa.lo,sb.lo) and np.array_equal(sa.hi,sb.hi))
    initial['bulk_coefficients_Linf'] = float(np.max(abs(sa.coeff-sb.coeff))) if same_mesh else None
    if same_initial:
        assert same_mesh and initial['bulk_coefficients_Linf'] < 1e-11 and initial['surface_Linf'] < 1e-11
    final = compare_final(a, b, shift)
    # Compare nonlinear localization on the same union-cell quadrature too.
    sa = state_at(pa, ca, int(ra[-1]['step'])); sb = state_at(pb, cb, int(rb[-1]['step']))
    offset = ((a['arguments']['translation']-b['arguments']['translation'])*b['records'][-1]['time']) if shift else 0.
    x, w = common_bulk_quadrature(sa, sb, offset)
    xa=x.copy();xa[:, 0]+=offset
    chia=localization(sa, xa, sa.at(xa))[2];chib=localization(sb, x, sb.at(x))[2]
    final['localization']=metric((chia-chib)**2, chib**2, w)
    result = dict(a=a['case'], b=b['case'], translation_removed=shift,
                  initialization=initial, final=final, history=compare_records(a, b, shift),
                  quadrature='Gauss8 on union Cartesian cells; surface split at labels, region and mesh crossings')
    (OUT/('pair-'+a['case']+'--'+b['case']+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    return result


def write_summary_csv(summaries, directory):
    rows=[]
    for s in summaries.values():
        row={k:s[k] for k in ['case','phase_dofs','wall_s','phase_integral_relative',
                              'phase_trapezoid_signed_remainder','phase_trapezoid_absolute_remainder',
                              'reaction_projection_loss']}
        row.update(initial_phase_relative_L2=s['initial_q16']['phase']['relative_L2'],
                   final_sampled_peak=s['final']['maximum'],
                   final_normal_width=np.sqrt(s['final_energy']['profile_normal_variance']),
                   final_localization_relative=s['final_energy']['normalization_relative'],
                   native_phase_balance_available=s['native_stage_balance']['available'],
                   native_phase_absolute_balance=s['native_stage_balance']['absolute_balance'],
                   endpoint_signed=s['energy']['signed'],endpoint_absolute=s['energy']['absolute'],
                   endpoint_relative=s['energy']['relative'],
                   trapezoid_signed=s['energy']['trapezoid_signed'],
                   trapezoid_absolute=s['energy']['trapezoid_absolute'],
                   stokes_iterations=s['cost']['stokes_iterations'])
        rows.append(row)
    with (directory/'comparison-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available-only',action='store_true',help='Reduce completed cases without claiming the matrix is finished')
    args=parser.parse_args()
    cases={}; summaries={}; pairs=[]
    for new, old in MATCHES:
        nn, on='layered-dgphase-'+new,'layered-bulk-'+old
        if not (OUT/(nn+'.json')).exists():
            if args.available_only:continue
            raise FileNotFoundError(nn)
        for name in (nn,on):
            cases[name]=read_case(name)
            print('Summarize',name,flush=True)
            summaries[name]=summarize(cases[name])
        print('Compare',nn,on,flush=True)
        pairs.append(compare(cases[nn],cases[on]))
    for a,b,same in [('plus','plus-half',True),('plus-half','plus-fine',False),('plus','minus',False)]:
        an,bn='layered-dgphase-'+a,'layered-dgphase-'+b
        if an in cases and bn in cases:
            print('Compare',an,bn,flush=True);pairs.append(compare(cases[an],cases[bn],same))
    retained=[]
    for a,b in [('plus','plus-half'),('plus-half','plus-fine-cartesian'),('plus','minus')]:
        path=OLD/('pair-layered-bulk-'+a+'--layered-bulk-'+b+'.json')
        previous=json.loads(path.read_text())
        retained.append(dict(a=previous['a'],b=previous['b'],final=previous['final'],
                             source=str(path.relative_to(ROOT))))
    result=dict(complete=len(cases)==8,cases=summaries,
                pairs=[dict(a=p['a'],b=p['b'],final=p['final']) for p in pairs],
                retained_q1_comparisons=retained)
    (OUT/'comparison-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_summary_csv(summaries,OUT)
    from plot_coupled import plot
    plot(cases,result,OUT)

if __name__=='__main__':main()
