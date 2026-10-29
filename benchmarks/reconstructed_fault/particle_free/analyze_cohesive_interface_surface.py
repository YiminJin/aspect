#!/usr/bin/env python3
"""Surface-space errors, with bulk, time and representation effects separated."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

from run_cohesive_interface_surface import (OUT, WORK, PREVIOUS, ROOT, L, H, SHIFT,
    operators, initial, front, table, save_json)

TIMES = [2.30, 2.45, 2.50, 2.55, 2.58]
CASES = ['surface-256', 'surface-512', 'surface-512-half']


def history(name):
    with (OUT/(name+'-history.csv')).open() as f:
        data = list(csv.DictReader(f))
    if name.endswith('-half'):
        data = [r for r in history(name.removesuffix('-half')) if float(r['time_s']) <= 2.42375+1e-10]+data
    return [{k:float(v) for k,v in r.items()} for r in data]


def state(name, t):
    path = WORK/name/f'state-{round(t*1e6)}.npz'
    if not path.exists() and name.endswith('-half') and t <= 2.42375+1e-10:
        path = WORK/name.removesuffix('-half')/path.name
    with np.load(path) as f:
        return {k:f[k].copy() for k in f}


def contour(x, j, level, side):
    # Only used after verifying that current Q1 jump reproduces actual stored
    # Gauss damage. No histories are interpolated into a new point return.
    if side == 'left':
        mask = x <= 2.5
        xx, yy = x[mask], abs(j[mask])
        ids = np.flatnonzero(yy >= level*front.ref.DF)
        if not len(ids):
            return 2.5
        k = ids[0]
        assert k > 0
        return float(xx[k-1]+(xx[k]-xx[k-1])*(level*front.ref.DF-yy[k-1])/(yy[k]-yy[k-1]))
    mask = x >= 5.5
    xx, yy = x[mask], abs(j[mask])
    ids = np.flatnonzero(yy >= level*front.ref.DF)
    if not len(ids):
        return 5.5
    k = ids[-1]
    assert k < len(xx)-1
    return float(xx[k]+(xx[k+1]-xx[k])*(yy[k]-level*front.ref.DF)/(yy[k]-yy[k+1]))


def box(z, n):
    # Actual native Gauss3 integral means on common .125 m physical boxes.
    return (z.reshape(n, 3)@np.array([5/18, 4/9, 5/18])).reshape(128, n//128).mean(axis=1)


def fields(name, t, op=None):
    s = state(name, t)
    n = len(s['mem'])
    K, N, q = operators(n) if op is None else op
    P = q['P']
    shear = P@(front.ref.G/H*s['wall']-K@s['mem'])
    pn = front.ref.P0+q['PN']@s['mem']
    a0 = initial(n, q)['a']
    representation = float(max(abs(s['a']-np.maximum(a0, np.minimum(abs(s['jump'])/front.ref.DF, 1)))))
    assert representation < 1e-12, (name, t, representation)
    return dict(s=s, n=n, layout=q, x=q['x'], a0=a0, growth=s['a']-a0,
                shear=shear, pn=pn, V=q['Q']@s['rate'], damage_identity=representation)


def main():
    ops = {n:operators(n) for n in [256, 512]}
    histories = {name:history(name) for name in CASES}
    summary, profiles, comparisons, native_differences, energy, onset = [], [], [], [], [], []
    cache = {}
    for name in CASES:
        n = int(name.split('-')[1])
        if not name.endswith('-half'):
            # Diagnostic event definitions, not phase masks or solver gates.
            for threshold in [0., 1e-4, .001, .01]:
                r = next(r for r in histories[name] if r['integrated_new_damage_m'] > threshold)
                onset.append(dict(case=name, integrated_growth_threshold_m=threshold,
                                  lower_s=r['time_s']-r['dt_s'], upper_s=r['time_s'],
                                  phase_increment_max=r['max_damage_increment']))
        for t in TIMES:
            f = fields(name, t, ops[n])
            cache[name,t] = f
            s, q = f['s'], f['layout']
            row = min(histories[name], key=lambda r:abs(r['time_s']-t)).copy()
            assert abs(row['time_s']-t) < 1e-10
            row.update(case=name, dx_surface_m=L/n)
            for level, label in [(.01, '001'), (.05, '005'), (.5, '050'), (1., '100')]:
                for side in ['left', 'right']:
                    row[f'a{label}_{side}_m'] = contour(q['node_x'], s['nodal_jump'], level, side)
            row['left_full_advance_m'] = 2.5-row['a100_left_m']
            row['right_full_advance_m'] = row['a100_right_m']-5.5
            row['left_cohesive_width_m'] = row['a100_left_m']-row['a001_left_m']
            row['left_integrated_growth_m'] = float(q['w']@(f['growth']*(f['x']<2.5)))
            row['right_integrated_growth_m'] = float(q['w']@(f['growth']*(f['x']>5.5)))
            row['jump_damage_identity_error'] = f['damage_identity']
            summary.append(row)
            for i,x in enumerate(f['x']):
                profiles.append(dict(case=name, time_s=t, x_m=x, weight_m=q['w'][i],
                                     initial_damage=f['a0'][i], damage=s['a'][i], damage_increment=f['growth'][i],
                                     jump_m=s['jump'][i], relaxed_jump_m=s['gauss_memory'][i],
                                     absolute_slip_m=s['absolute_slip'][i], Theta=s['age'][i],
                                     velocity_m_s=f['V'][i], projected_normal_Pa=f['pn'][i],
                                     work_adjoint_shear_load_Pa=f['shear'][i]))
        for start, label in [(2.45, 'event'), (2.55, 'hold')]:
            data = [r for r in histories[name] if r['time_s'] > start+1e-10]
            absolute = sum(abs(r['physical_remainder_J_m']) for r in data)
            fracture = sum(r['fracture_increment_J_m'] for r in data)
            sliding = sum(r['friction_J_m']+r['radiation_J_m'] for r in data)
            energy.append(dict(case=name, interval=label, start_s=start, end_s=2.58,
                               absolute_remainder_J_m=absolute,
                               signed_remainder_J_m=sum(r['physical_remainder_J_m'] for r in data),
                               new_fracture_J_m=fracture, sliding_loss_J_m=sliding,
                               relative_fracture=absolute/fracture, relative_sliding=absolute/sliding,
                               external_work_J_m=sum(r['trapezoid_work_J_m'] for r in data)))
    lookup = {(r['case'],round(r['time_s'],8)):r for r in summary}
    for name,label in [('surface-256','surface refinement'), ('surface-512-half','fine temporal')]:
        for t in TIMES:
            if name.endswith('-half') and t < 2.42375:
                continue
            a, b = cache[name,t], cache['surface-512',t]
            ra, rb = lookup[name,round(t,8)], lookup['surface-512',round(t,8)]
            g, h = box(a['growth'],a['n']), box(b['growth'],b['n'])
            pn, pnb = box(a['pn'],a['n']), box(b['pn'],b['n'])
            shear, shearb = box(a['shear'],a['n']), box(b['shear'],b['n'])
            row = dict(kind=label, time_s=t,
                       growth_absolute_m=abs(ra['integrated_new_damage_m']-rb['integrated_new_damage_m']),
                       growth_relative=abs(ra['integrated_new_damage_m']-rb['integrated_new_damage_m'])/rb['integrated_new_damage_m'],
                       growth_box_L2_relative=float(np.linalg.norm(g-h)/np.linalg.norm(h)),
                       slip_absolute_m2=abs(ra['integrated_abs_jump_m2']-rb['integrated_abs_jump_m2']),
                       slip_relative=abs(ra['integrated_abs_jump_m2']-rb['integrated_abs_jump_m2'])/rb['integrated_abs_jump_m2'],
                       left_front_absolute_m=abs(ra['a100_left_m']-rb['a100_left_m']),
                       left_front_relative_to_advance=abs(ra['a100_left_m']-rb['a100_left_m'])/max(rb['left_full_advance_m'],1e-30),
                       right_front_absolute_m=abs(ra['a100_right_m']-rb['a100_right_m']),
                       normal_box_RMS_Pa=float(np.sqrt(np.mean((pn-pnb)**2))),
                       shear_load_box_RMS_Pa=float(np.sqrt(np.mean((shear-shearb)**2))),
                       reaction_absolute_N_m=abs(ra['reaction_N_m']-rb['reaction_N_m']))
            comparisons.append(row)
    with (PREVIOUS/'event-half-history.csv').open() as f:
        native = list(csv.DictReader(f))
    for t in TIMES:
        a = min(native, key=lambda r:abs(float(r['time_s'])-t))
        b = lookup['surface-256',round(t,8)]
        native_differences.append(dict(time_s=t,
            growth_absolute_m=abs(b['integrated_new_damage_m']-float(a['integrated_new_damage_m'])),
            growth_relative=abs(b['integrated_new_damage_m']-float(a['integrated_new_damage_m']))/float(a['integrated_new_damage_m']),
            left_front_absolute_m=abs(b['left_full_advance_m']-float(a['full_advance_m'])),
            slip_relative=abs(b['integrated_abs_jump_m2']-float(a['integrated_abs_jump_m2']))/float(a['integrated_abs_jump_m2'])))
    # Best L2 representation of the fine linear jump, independent of nonlinear
    # Gauss histories. Never use this projection to initialize another evolution.
    qfine, w = ops[512][2]['Q'], ops[512][2]['w']
    x = (ops[512][2]['x']+SHIFT)/(L/256)
    cell = np.floor(x).astype(int)%256
    xi = x-np.floor(x)
    Qc = np.zeros((len(x),256))
    Qc[np.arange(len(x)),cell] = 1-xi
    Qc[np.arange(len(x)),(cell+1)%256] = xi
    mass = Qc.T@(w[:,None]*Qc)
    representation = []
    for t in TIMES:
        f = cache['surface-512',t]
        z = f['s']['jump']
        projected = Qc@np.linalg.solve(mass,Qc.T@(w*z))
        tip = lookup['surface-512',round(t,8)]['a100_left_m']
        local = abs(f['x']-tip) <= .5
        representation.append(dict(time_s=t, fine_to_coarse_best_Q1_jump_relative_L2=float(np.sqrt(w@((z-projected)**2)/(w@(z*z)))),
                                   left_tip_RMS_error_m=float(np.sqrt(w[local]@((z[local]-projected[local])**2)/sum(w[local]))),
                                   maximum_error_m=float(max(abs(z-projected))),
                                   initial_jump_error_m=0., initial_seed_inventory_error_J_m=0.))
    table(OUT/'summary.csv',summary)
    table(OUT/'profiles.csv',profiles)
    table(OUT/'comparisons.csv',comparisons)
    table(OUT/'native-mechanics-differences.csv',native_differences)
    table(OUT/'representation.csv',representation)
    table(OUT/'energy.csv',energy)
    table(OUT/'onset-brackets.csv',onset)
    spatial = [r for r in comparisons if r['kind']=='surface refinement' and r['time_s'] >= 2.45]
    temporal = [r for r in comparisons if r['kind']=='fine temporal']
    result = dict(summary=summary, comparisons=comparisons, native_mechanics=native_differences,
                  representation=representation, energy=energy, onset_brackets=onset,
                  gates=dict(spatial_integral_5_percent=all(max(r['growth_relative'],r['slip_relative']) <= .05 for r in spatial),
                             spatial_full_front_003125m=all(r['left_front_absolute_m'] <= .03125 for r in spatial),
                             fine_temporal_integral_1_percent=all(max(r['growth_relative'],r['slip_relative']) <= .01 for r in temporal),
                             physical_work_1_percent=all(max(r['relative_fracture'],r['relative_sliding']) <= .01 for r in energy)),
                  averaging='Actual Gauss3 integral means in identical .125 m boxes, no constitutive history interpolation.',
                  raw_point_trace_accuracy_qualified=False, bulk_inertia_included=False,
                  convergence_order_established=False, historical_early_spatial_failure_retained=True)
    save_json(OUT/'verification.json',result)
    print(json.dumps(dict(gates=result['gates'], final_spatial=spatial[-1],final_temporal=temporal[-1], native_final=native_differences[-1]),indent=2))


def check_operators():
    from analyze_cohesive_interface_boundary import q1_symbols
    checks=[]
    for n in [256,512]:
        K,N,layout=operators(n)
        with np.load(WORK/f'operators-{n}.npz') as f:
            coarse=f['symbols']
        fine=q1_symbols(16.,n,.6,.4,600)
        relative=float(np.linalg.norm(fine-coarse)/np.linalg.norm(fine))
        assert relative<1e-6
        s=state(f'surface-{n}',2.58)
        change=layout['P']@np.fft.ifft((fine-coarse)*np.fft.fft(s['mem'])[None,:],axis=1).real.T
        checks.append(dict(n=n,alias_300_to_600_relative=relative,
            final_shear_load_max_change_Pa=float(max(abs(change[:,0]))),
            final_normal_max_change_Pa=float(max(abs(change[:,1]))),
            K_uniform_max_error_Pa_m=float(max(abs(K@np.ones(n)-1e6))),
            N_uniform_max_error_Pa_m=float(max(abs(N@np.ones(n))))))
    save_json(OUT/'operator-verification.json',dict(checks=checks,pressure_mean_shifted_to_match=False))


if __name__ == '__main__':
    check_operators() if len(sys.argv)>1 and sys.argv[1]=='operators' else main()
