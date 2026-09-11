#!/usr/bin/env python3
"""Bounded true-pressure pilot checks; no scalar or exact-reference claim."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

base = Path(__file__).resolve().parent
sys.path.insert(0, str(base.parent))
from measure_case import profile_primitive, summarize


def read(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=','))


def stats(x, values):
    h = np.diff(x)
    mean = np.sum(h*(values[:-1]+values[1:])/2)/sum(h)
    rms = lambda v: np.sqrt(np.sum(h*(v[:-1]**2+v[:-1]*v[1:]+v[1:]**2)/3)/sum(h))
    return dict(mean=float(mean), minimum=float(min(values)), maximum=float(max(values)),
                rms=float(rms(values)), anomaly_rms=float(rms(values-mean)))


def boundaries(bulk, loading):
    error, top = 0., []
    for x in np.unique(bulk['x']):
        column = np.sort(bulk[bulk['x']==x], order='y')
        for rows, y in ((column[:3], -.5), (column[-3:], .5)):
            ys = rows['y']
            basis = np.array([np.prod([(y-ys[j])/(ys[i]-ys[j])
                                      for j in range(3) if i!=j]) for i in range(3)])
            error = max(error, abs(basis@rows['ux']-y*loading))
            if y<0: error = max(error, abs(basis@rows['uy']))
            else: top.append(float(basis@rows['uy']))
    # Three tangential Gauss nodes in each cell. The free top has zero net
    # flux, not pointwise zero normal velocity, under incompressibility.
    flux = np.mean(np.array(top).reshape(-1,3)@np.array([5,8,5])/18)*.25
    assert error<1e-12
    return dict(prescribed_velocity_error_m_s=float(error), top_uy_min_m_s=min(top),
                top_uy_max_m_s=max(top), top_flux_m2_s=float(flux))


def check(case, runtime_limit=120):
    directory = base/case
    get = lambda name,k: read(directory/f'{name}_{k}.csv')
    log = summarize(base/f'{case}.log')
    assert [s['time'] for s in log]==[0., .5, 1.]
    resources = json.loads((base/f'{case}.resources.json').read_text())
    # Execution budget only; nonlinear, linear and physical checks below are
    # unchanged for the separately approved 64x256 refinement pair.
    assert resources['exit_status']==0 and resources['wall_seconds']<runtime_limit
    phase = get('phase',0)
    surface0 = get('surface',0)
    x = surface0['x']
    arclength_weight = np.r_[np.diff(x)/2,0]+np.r_[0,np.diff(x)/2]
    segments = get('segments',0)
    assert max(abs(segments['nx']))<1e-12
    width = min(np.r_[segments['half_width_minus'],segments['half_width_plus']])
    ih, omitted = [], []
    for coordinate in np.unique(phase['x']):
        column = np.sort(np.unique(phase[phase['x']==coordinate][['y','phi']]),order='y')
        panel, total = profile_primitive(column['y'],column['phi'],np.array([-width,width]))
        ih.append(total); omitted.append(1-(panel[1]-panel[0])/total)
    result = dict(case=case, resources=resources, original_omitted_target=1e-6,
                  pilot_omitted_allowance=1e-4, actual_normalization_requirement=1e-4,
                  omitted_fraction_max=float(max(omitted)), profile_locations=len(ih),
                  independent_Ih_range_m=[min(ih),max(ih)],support_half_width_m=float(width),
                  steps=[], initial_Theta=stats(x,surface0['Theta']))
    assert max(omitted)<=1e-4
    old_theta = surface0['Theta'].copy()
    initial_particles = np.sort(get('particles',0),order='id')
    initial_stress = np.column_stack([initial_particles[n] for n in ('tau_xx','tau_yy','tau_xy')])
    assert np.array_equal(initial_stress,np.tile([0.,0.,1500.],(len(initial_stress),1)))
    previous_stress = initial_stress
    slip = np.zeros(len(x))
    for k,logged in enumerate(log):
        last = logged['nonlinear'][-1]
        assert last['bulk']<last['bulk target'] and last['surface']<1e-8*last['surface scale']
        assert all(r['fresh']<=r['target'] and r['pressure quotient']==0 for r in logged['linear'])
        assert not logged.get('armijo_exhausted',False)
        assert np.array_equal(get('phase',k),phase)
        s,weak,bulk,particles = (get(n,k) for n in ('surface','surface_weak','bulk','particles'))
        assert np.array_equal(s['x'],x) and np.all(s['fault']==0)
        particles = np.sort(particles,order='id')
        assert np.array_equal(particles['id'],initial_particles['id'])
        assert np.array_equal(particles['H'],initial_particles['H'])
        stress = np.column_stack([particles[n] for n in ('tau_xx','tau_yy','tau_xy')])
        change = float(np.max(abs(stress-previous_stress)))
        if k: assert change>1.
        previous_stress = stress
        dt = float(get('time',k)['dt'][0])
        if k:
            slip += dt*s['V']
            decay = np.exp(-dt*s['V']/.001)
            expected = old_theta*decay + .001/s['V']*(-np.expm1(-dt*s['V']/.001))
        else: expected = old_theta
        theta_error = float(max(abs(s['Theta']-expected)))
        assert theta_error<1e-10
        old_theta = s['Theta'].copy()
        at_bound = (s['V']-1e-12 <= 100*np.finfo(float).eps*np.maximum(1e-12,abs(s['V'])))
        # An interior node cannot enter this solver's lower-bound active set.
        # Do not infer active status from a nonzero residual if a bound is hit.
        assert not np.any(at_bound), 'A bound was reached: inspect the actual active partition before interpreting.'
        mass = np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
        rhs = np.column_stack([weak[n] for n in ('q','C','friction','damping','F')])
        assert np.max(abs(rhs[:,0]-rhs[:,1]-rhs[:,2]-rhs[:,3]-rhs[:,4]))<1e-10
        represented = np.linalg.solve(mass,rhs)
        rms = np.sqrt(max(0,rhs[:,-1]@represented[:,-1])/mass.sum())
        assert abs(rms-last['surface'])<1e-9
        assert abs(mass.sum()-sum(particles['volume'][particles['active']==1]))<1e-11
        normal = read(directory/f'constitutive_normal_{k}_rank0.csv')
        assert np.all(normal['step']==k) and np.all(normal['time']==logged['time'])
        assert np.max(abs(normal['weight']-mass.sum(axis=1)))<1e-12
        loads = np.column_stack([normal[n+'_load'] for n in ('p','sigma','tauN')])
        assert max(abs(loads[:,0]-loads[:,1]-loads[:,2]))<1e-10
        fields = np.linalg.solve(mass,loads)
        bx,ix = np.unique(bulk['x'],return_inverse=True)
        column_slip = np.bincount(ix,weights=bulk['weight']*(bulk['chi']*bulk['V']+bulk['history']))
        column_slip /= np.bincount(ix,weights=bulk['weight'])
        ratios = column_slip/np.interp(bx,x,s['V'])
        global_ratio = np.dot(bulk['weight'],bulk['chi']*bulk['V']+bulk['history'])/np.dot(arclength_weight,s['V'])
        assert max(abs(ratios-1))<=1e-4 and abs(global_ratio-1)<=1e-4
        row = dict(time_s=logged['time'], nonlinear=last, fresh_linear_checks=len(logged['linear']),
                   maximum_fresh_over_target=max(r['fresh']/r['target'] for r in logged['linear']),
                   active_nodes=0,free_nodes=len(x),theta_update_error_s=theta_error,
                   particle_stress_commit_change_max_Pa=change,
                   constitutive={},surface_balance_rms_Pa=float(rms),
                   surface_balance_weak_max=float(max(abs(weak['F']))),
                   normalization_max_error=float(max(abs(ratios-1))),global_normalization=float(global_ratio),
                   measured_slip_locations=len(bx),boundaries=boundaries(bulk,float(get('time',k)['U'][0])),
                   bulk_QP_pressure_mean_Pa=float(np.average(bulk['p'],weights=bulk['weight'])),
                   response={name:stats(x,value) for name,value in
                             [('V',s['V']),('Theta',s['Theta']),('C',s['C']),('slip',slip),
                              ('q',represented[:,0]),('friction',represented[:,2])]})
        for c,name in enumerate(('p','sigma','tauN')):
            row['constitutive'][name] = dict(domain_weighted_mean_Pa=float(sum(loads[:,c])/mass.sum()),
                raw_min_Pa=float(min(normal[name+'_min'])),raw_max_Pa=float(max(normal[name+'_max'])),
                surface_Q1=stats(x,fields[:,c]))
        data = np.column_stack((x,fields,s['V'],slip,s['Theta'],s['C'],represented))
        np.savetxt(base/f'{case}-surface-{k}.csv',data,delimiter=',',comments='',
                   header='s,p,sigma,tauN,V,slip,Theta,C,q,C_evaluated,friction,damping,F')
        result['steps'].append(row)
    (base/f'{case}-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare():
    result = []
    for k,t in enumerate((0.,.5,1.)):
        a,b = (read(base/f'{name}-surface-{k}.csv') for name in ('pilot','homogeneous'))
        assert np.array_equal(a['s'],b['s'])
        fields = [name for name in a.dtype.names if name!='s']
        difference = np.column_stack([a[name]-b[name] for name in fields])
        np.savetxt(base/f'bump-minus-homogeneous-{k}.csv',np.column_stack((a['s'],difference)),
                   delimiter=',',comments='',header='s,'+','.join(fields))
        result.append(dict(time_s=t,fields={name:stats(a['s'],difference[:,i]) for i,name in enumerate(fields)}))
    (base/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('pilot','homogeneous','pilot64','homogeneous64','compare'))
    parser.add_argument('--runtime-limit',type=float,default=120,
                        help='Approved wall-time limit in seconds, not a solver tolerance.')
    args=parser.parse_args()
    start=time.monotonic()
    output=compare() if args.case=='compare' else check(args.case,args.runtime_limit)
    print(json.dumps(output,indent=2))
    print('Analysis seconds:',time.monotonic()-start)
