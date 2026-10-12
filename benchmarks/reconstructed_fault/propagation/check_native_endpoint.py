#!/usr/bin/env python3
"""Read-only native operator checks and unaligned schedule accuracy gate."""
import argparse
from pathlib import Path

import numpy as np
import native_endpoint as native
import asymmetric_branch as old

E = native.E
STRESS, STORAGE = 1332.521975, .887807407


def schedule(prefix, onset=False):
    path = E / (f'prefix-{prefix}-fine-1024'+('-onset' if onset else ''))
    f = dict(np.load(path/'fields.npz'))
    fine = dict(np.load(E/'dt-1-2048/fields.npz'))
    h, hf = [old.read(p/'history.json') for p in [path, E/'dt-1-2048']]
    assert old.read(path/'summary.json')['completed']
    for key, value in dict(np.load(path/'initialization.npz')).items():
        assert np.array_equal(value, np.load(E/'dt-1-2048/initialization.npz')[key])
    z, w = np.polynomial.legendre.leggauss(24)
    s, weight = (z+1)/2, np.tile(w/1024, 512)
    def stress(fields, j, I):
        p, R = fields['phase'][j], fields['R'][j]
        phi = (1-s)*p[:-1, None]+s*p[1:, None]
        derivative = 512*((-3+4*s)*R[:-2:2, None] +
                          (4-8*s)*R[1::2, None]+(-1+4*s)*R[2::2, None])
        chi = old.core.h(phi).ravel()/I
        return 1e6*derivative.ravel()-fields['Z'][j]*chi, chi
    rows = []
    peak = max(b['V'] for b in hf)
    for j, a in enumerate(h, 1):
        k = round(a['time']*2048)
        b = hf[k-1]
        ta, _ = stress(f, j, a['I']); tb, chi = stress(fine, k, b['I'])
        delta = ta-tb
        previous_time = h[j-2]['time'] if j>1 else 0.
        k0 = round(previous_time*2048)
        previous_slip = hf[k0-1]['slip'] if k0 else 0.
        common_rate = (b['slip']-previous_slip)/(a['time']-previous_time)
        rows.append(dict(time=a['time'], phase=float(max(abs(f['phase'][j]-fine['phase'][k]))),
          q=abs(a['q']-b['q']), slip_fraction=abs(a['slip']-b['slip'])/hf[-1]['slip'],
          interval_rate_fraction=abs(a['V']-common_rate)/peak,
          whole_stress_fraction=float(np.sqrt(weight@(delta*delta)))/STRESS,
          localized_stress_fraction=float(np.sqrt((weight*chi)@(delta*delta)/(weight@chi)))/STRESS,
          bulk_storage_fraction=abs(a['bulk']-b['bulk'])/STORAGE))
    targets = old.read(E/'temporal-comparison.json')['targets']
    maximum = {key:max(row[key] for row in rows) for key in targets}
    result = dict(prefix=prefix, targets=targets, maximum=maximum,
                  passes={k:maximum[k]<=targets[k] for k in targets}, by_time=rows,
                  initial_fields_identical=True, aligned=False, global_reflection=False)
    old.write(E/(f'prefix-{prefix}-gate'+('-onset' if onset else '')+'.json'), result)
    print(old.json.dumps({k:result[k] for k in ['prefix','maximum','passes']}))
    assert all(result['passes'].values()), 'Do not launch an inaccurate native schedule'


def operators(output, report=None):
    a = np.genfromtxt(output/'operators.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    g = np.genfromtxt(output/'operator-gradients.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    adj = np.genfromtxt(output/'operator-adjoint.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    assert len(a)==15 and len(adj)==6
    actions=[]
    for label in ['unloaded','growing','asymmetric']:
        reference=np.loadtxt(E/'operator-input'/f'{label}-action.txt')
        approximate=[]
        for eps in [1e-5,5e-6]:
            plus=g[g['case']==f'{label}-{eps:+g}.txt']['native']
            minus=g[g['case']==f'{label}-{-eps:+g}.txt']['native']
            action=(plus-minus)/(2*eps)
            error=float(max(abs(action-reference)))
            approximate.append(action)
            actions.append(dict(case=label,epsilon=eps,maximum_action_error=error,
                                reference_scale=float(max(abs(reference)))))
            assert error<1e-5
        actions[-1]['half_spacing_uncertainty']=float(max(abs(approximate[0]-approximate[1])))
    maximum={key:float(max(abs(a[key]))) for key in a.dtype.names if key!='case'}
    assert max(maximum[k] for k in ['bulk_error','cohesive_error','fracture_error','potential_error'])<1e-8
    assert maximum['gradient_max_error']<1e-8 and maximum['I_relative_error']<1e-9
    if 'full_gradient_error' in maximum:assert maximum['full_gradient_error']<1e-10
    assert maximum['weak_force']<1e-7 and maximum['contact']<1e-6
    result=dict(maximum=maximum, reduced_tangent=actions,
       energy_adjoint_maximum_error=float(max(adj['error'])),
       normalization_action_relative_error=float(max(adj['I_action_relative_error'])),
       explicit_state_poisoning=True, accepted_history_restored=True,
       scope='native energy and point constitutive law with native FE weak quadrature; fresh evolution separate')
    quadratic=output/'operator-fracture-quadratic.csv'
    if quadratic.exists():
        q=np.genfromtxt(quadratic,delimiter=',',names=True,dtype=None,encoding='utf-8')
        assert len(q)==6
        result['CPDI_quadratic_reuse']={key:float(max(q[key])) for key in ['energy_error','gradient_error']}
        assert max(result['CPDI_quadratic_reuse'].values())<1e-10
    linear=output/'operator-linear-tangent.csv'
    if linear.exists():
        values=np.genfromtxt(linear,delimiter=',',names=True,dtype=None,encoding='utf8')
        result['linearized_tangent']={}
        for direction in np.unique(values['direction']):
            rows=values[values['direction']==direction]
            if direction=='tangential-reduced':
                # CSV zero is a placeholder: this row records a magnitude,
                # not an independent finite-difference error estimate.
                result['linearized_tangent'][str(direction)]=dict(diagnostic_only=True,
                    maximum_scale=float(max(rows['action_scale'])))
                continue
            error=float(max(rows['maximum_error']))
            result['linearized_tangent'][str(direction)]=dict(maximum_error=error,
                maximum_scale=float(max(rows['action_scale'])))
            # These are the existing plugin checks, in their native scales.
            if direction=='normal-prototype-exact':assert error<1e-7
            else:assert np.all(rows['maximum_error']<1e-6*np.maximum(1.,rows['action_scale']))
    old.write(report or E/'operator-checks.json',result);print(old.json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=int)
    p.add_argument('--onset',action='store_true')
    p.add_argument('--operators',type=Path)
    p.add_argument('--report',type=Path,help='Write a new result without replacing historical evidence.')
    args=p.parse_args()
    if args.prefix is not None:schedule(args.prefix,args.onset)
    if args.operators is not None:operators(args.operators,args.report)
