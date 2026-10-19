#!/usr/bin/env python3
"""Independent fixed-local-mesh constrained phase and lifecycle assessment."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from assess_pf6a import Ref, ledger
from assess_phase_energy import csv, scaled_error
from assess_pf6_local import LocalState, quadrature_control
from assess_pf6_transfer import rows, norm2
from assess_pf6_domain_free import initialization


def geometric_C(ref, native):
    """Derive hanging interpolation from physical coarse edges, not exported C."""
    masters = native['physical'].astype(int)
    n, m = ref.n, len(masters)
    matrix = np.zeros((n, m))
    matrix[masters, np.arange(m)] = 1
    points = np.column_stack([ref.nodes['x'], ref.nodes['y']])
    positions = {i: j for j, i in enumerate(masters)}
    for i in sorted(set(range(n))-set(masters)):
        candidates = []
        p = points[i]
        for cell in ref.indices:
            for aa, bb in [(0, 1), (0, 2), (1, 3), (2, 3)]:
                a, b = cell[aa], cell[bb]
                if a not in positions or b not in positions:
                    continue
                edge = points[b]-points[a]
                t = (p-points[a])@edge/(edge@edge)
                if 0 < t < 1 and np.max(abs(p-(points[a]+t*edge))) < 1e-14:
                    v = np.zeros(m)
                    v[positions[a]], v[positions[b]] = 1-t, t
                    candidates.append(v)
        assert candidates, ('missing coarse physical edge', i)
        assert all(np.array_equal(candidates[0], v) for v in candidates)
        matrix[i] = candidates[0]
    return matrix


def operators(folder):
    ref = Ref(folder)
    native = np.atleast_1d(csv(folder/'native.csv'))
    matrix = geometric_C(ref, native)
    exported = np.zeros_like(matrix)
    for row in np.atleast_1d(csv(folder/'C.csv')):
        exported[int(row['physical']), int(row['master'])] = row['weight']
    assert np.array_equal(matrix, exported)
    p = ref.nodes['phase']
    assert np.array_equal(matrix@native['phase'], p)
    actual = ref.evaluate(p)
    g = matrix.T@actual['gradient']
    result = dict(physical_nodes=ref.n, independent_masters=len(native),
                  geometric_C_exact=True,
                  gradient_scaled_error=scaled_error(g, native['gradient']),
                  actions=[])
    data = np.genfromtxt(folder/'actions.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    for j in range(2):
        d = ref.nodes[f'd{j+1}']
        assert np.array_equal(matrix@d[native['physical'].astype(int)], d)
        independent = matrix.T@ref.evaluate(p+1e-30j*d)['gradient'].imag/1e-30
        fixed = data[(data['kind']=='fixed') & (data['direction']==j)]['value']
        result['actions'].append(dict(direction=j, fixed_action_scaled_error=scaled_error(independent, fixed)))
    elements = np.atleast_1d(csv(folder/'free-hessian.csv'))
    nf = 1+max(elements['i'].astype(int))
    h = np.zeros((nf, nf))
    h[elements['i'].astype(int), elements['j'].astype(int)] = elements['value']
    eig = np.linalg.eigvalsh(.5*(h+h.T))
    result['independent_free_eigenvalues'] = eig.tolist()
    stability = csv(folder/'stability.csv')
    assert abs(eig[0]-float(stability['minimum_free_eigenvalue'])) < 1e-10*max(1., float(max(abs(eig))))
    result['native_free_stability'] = {k: float(stability[k]) for k in stability.dtype.names}
    fd = np.atleast_1d(csv(folder/'fd.csv'))
    result['native_re_equilibrated_FD'] = [dict(direction=j,
        best_energy_relative=float(min(fd['energy_relative'][fd['direction']==j])),
        best_action_relative=float(min(fd['action_relative'][fd['direction']==j]))) for j in range(2)]
    assert result['gradient_scaled_error'] < 1e-9
    assert max(a['fixed_action_scaled_error'] for a in result['actions']) < 1e-9
    return result


def state(run, step):
    folder = run/'run'
    ref = Ref(folder/f'state-{step}-reference')
    p = ref.nodes['phase']
    data = np.atleast_1d(csv(folder/'local-evolution.csv'))
    row = data[data['step']==step][-1]
    # Independently identify masters from hanging coarse-edge geometry by the
    # actual fixed mesh map, also at initialization/restart.
    # For runs without operator diagnostics, derive dependent nodes directly.
    dependent = set()
    points = np.column_stack([ref.nodes['x'], ref.nodes['y']])
    for i, point in enumerate(points):
        for cell in ref.indices:
            for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
                edge = points[cell[b]]-points[cell[a]]
                t = (point-points[cell[a]])@edge/(edge@edge)
                if 0 < t < 1 and np.max(abs(point-points[cell[a]]-t*edge)) < 1e-14:
                    dependent.add(i)
    master = np.array(sorted(set(range(ref.n))-dependent))
    template = np.zeros(len(master), dtype=[('physical', int)])
    template['physical'] = master
    C = geometric_C(ref, template)
    assert np.array_equal(C@p[master], p)
    e = ref.evaluate(p)
    q24, q32 = ref.evaluate(p, 24), ref.evaluate(p, 32)
    lower = ref.nodes['accepted']
    # Native accepted exporter stores the current accepted obstacle. The true
    # incoming obstacle is separately exported by the before-step diagnostic.
    before = folder/f'before-step-{step}-phase.csv'
    if step and before.exists():
        import csv as csv_module
        with before.open() as f:
            incoming = {(float(r['x']), float(r['y'])):float(r['lower']) for r in csv_module.DictReader(f)}
        lower = np.array([incoming[tuple(x)] for x in points])
    gradient = C.T@e['gradient']
    projected = np.where(p[master]==lower[master], np.minimum(gradient, 0), gradient)
    result = dict(step=step, phase_expansion_exact=True,
        energy_scaled_error=max(abs(e[k]-row[k])/max(1., abs(row[k])) for k in ['bulk','cohesive','fracture']),
        normalization_scaled_error=scaled_error(e['I'], ref.surface['I_accepted']),
        independent_master_KKT=float(max(abs(projected))) if step else None,
        high_rule_master_KKT=float(max(abs(np.where(p[master]==lower[master], np.minimum(C.T@q32['gradient'], 0), C.T@q32['gradient'])))) if step else None,
        q24_q32_energy_uncertainty=abs(q24['energy']-q32['energy']),
        native_vs_q32_energy=abs(e['energy']-q32['energy']),
        native_vs_q32_load_scaled=scaled_error(e['loads'], q32['loads']))
    loads=folder/f'state-{step}-reference/native-loads.csv'
    if loads.exists():
        actual=csv(loads)
        cohesive=ref.sf.T@(ref.profiles['weight']*(ref.sf@ref.surface['Z'])/(ref.sf@e['I']))
        minimum=float(re.search(r'set Minimum slip rate\s*=\s*(\S+)',(folder/'parameters.prm').read_text()).group(1))
        rates=csv(folder/f'state-{step}-surface.csv')['V']
        active=(rates<=minimum) & (actual['residual']<=0)
        result['fault_contact_active']=int(sum(active))
        result['fault_KKT_weak_Linf_Pa_m']=float(max(abs(np.where(active,0.,actual['residual']))))
        assert result['fault_KKT_weak_Linf_Pa_m']<1e-8
        result['weak_shear_scaled_error']=scaled_error(e['loads'],actual['shear'])
        result['weak_cohesive_scaled_error']=scaled_error(cohesive,actual['cohesive'])
    samples=rows(folder/f'state-{step}-physical.csv')
    points=np.array([[float(row[k]) for k in ['x','y']] for row in samples])
    evaluated=LocalState(folder,f'state-{step}').evaluate(points)
    for name,keys in [('T',['Txx','Tyy','Txy']),('tau',['xx','yy','xy'])]:
        observed=np.array([[float(row[k]) for k in keys] for row in samples])
        difference=evaluated[name]-observed
        error=float(np.sqrt(sum(norm2(difference))/max(1.,sum(norm2(observed)))))
        result[name+'_full_tensor_sample_error']=error
        assert error<1e-9
    assert result['energy_scaled_error']<1e-9
    assert result['normalization_scaled_error']<1e-9
    if step: assert result['independent_master_KKT']<1e-9
    return result


def work_ledger(run, step, dt):
    """Reuse the established PF6a accounting path, not a new time integrator."""
    folder=run/'run'
    ref=Ref(folder/f'state-{step}-reference')
    old=Ref(folder/f'state-{step-1}-reference')
    fields=rows(folder/f'state-{step}-native-fields.csv')
    cells=list(dict.fromkeys(row['CellId'] for row in fields))
    lookup={name:i for i,name in enumerate(cells)}
    boxes={row['CellId']:row for row in rows(folder/f'state-{step}-cells.csv')}
    for cell,name in zip(ref.cells,cells):
        box=boxes[name]
        assert cell['x0']==float(box['x0']) and cell['y0']==float(box['y0'])
        assert cell['dx']==float(box['x1'])-float(box['x0'])
        assert cell['dy']==float(box['y1'])-float(box['y0'])
    points=rows(folder/'dg-support.csv')
    velocity=np.zeros((len(cells),3,3,2))
    pressure=np.zeros((len(cells),3,3,1))
    for row in fields:
        cell=lookup[row['CellId']]
        point=points[int(row['node'])]
        i,j=[round(2*float(point[k])) for k in ['r','s']]
        component=int(row['component'])
        if component<2:velocity[cell,i,j,component]=float(row['solution'])
        if component==2:pressure[cell,i,j,0]=float(row['solution'])
    rates=csv(folder/f'state-{step}-surface.csv')['V']
    low=ledger(ref,old,velocity,pressure,rates,dt,12)
    high=ledger(ref,old,velocity,pressure,rates,dt,24)
    assert abs(high['identity_remainder'])<1e-7
    return dict(accounting=high,time_quadrature_uncertainty=max(abs(high[k]-low[k]) for k in high),
        interpretation='Linear physical-phase / exact Maxwell T,Z path at fixed endpoint rates; endpoint discretization remainder is retained')


def assess(run):
    record=json.loads((run/'record.json').read_text())
    assert record['exit_code']==0
    data=np.atleast_1d(csv(run/'run/local-evolution.csv'))
    result=dict(label=run.name,seconds=record['seconds'],native_gates=len(csv(run/'run/transfer-gates.csv')),
        states=[state(run,int(row['step'])) for row in data],
        trajectory=[{k:float(row[k]) for k in data.dtype.names} for row in data])
    assert all(csv(run/'run/transfer-gates.csv')['passed'])
    if (run/'run/constrained-operators').exists(): result['operators']=operators(run/'run/constrained-operators')
    result['work_ledger']={str(int(row['step'])):work_ledger(run,int(row['step']),float(row['dt']))
        for row in data if row['step']>0 and (run/'run'/f'state-{int(row["step"])-1}-reference').exists()}
    if (run/'run/prescribed-initial-phase.csv').exists(): result['initialization']=initialization(run)
    result['physical_quadrature']=quadrature_control(run,[f'state-{int(r["step"])}' for r in data])
    assert any(row['interface_growth'] > 1e-10 and row['interface_active'] > 0 for row in data if row['step'] > 0)
    result['growing_and_obstacle_active_interface_exercised']=True
    result['passed']=True
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=dict(runs=[assess(run) for run in a.runs],strict_reference_match_passed=False,
        scope='Constrained-solver/lifecycle qualification only; previous branch/spatial/temporal failures retained')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'runs':[r['label'] for r in result['runs']],'passed':True}))

if __name__=='__main__':main()
