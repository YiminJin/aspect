#!/usr/bin/env python3
"""Fresh Q1/Gauss surface refinements with the existing analytical mixed strip."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import zipfile

import numpy as np
from scipy.linalg import circulant

import cohesive_interface_front as front
from analyze_cohesive_interface_boundary import q1_symbols

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface'
WORK = ROOT / 'build-particle-free/cohesive-interface-surface'
PREVIOUS = OUT.parent / 'cohesive_interface_boundary'
L, H, SHIFT, RATE, LOAD_END, END = 16., 1., 4., .001, 2.55, 2.58
SELECTED = [2.30, 2.42375, 2.45, 2.50, 2.55, 2.58]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def table(path, data):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def operators(n):
    path = WORK / f'operators-{n}.npz'
    if path.exists():
        with np.load(path) as f:
            K, N = f['K'], f['N']
    else:
        symbols = q1_symbols(L, n, .6, .4)
        K = circulant(np.fft.ifft(symbols[0]).real)
        N = circulant(np.fft.ifft(symbols[1]).real)
        # Exact reciprocal/odd identities, as in the retained Layers reference.
        assert np.max(abs(K-K.T)) < 1e-6
        assert np.max(abs(N+N.T)) < 1e-6
        K = .5*(K+K.T)
        N = .5*(N-N.T)
        np.savez_compressed(path, K=K, N=N, symbols=symbols)
    x = np.arange(n)*L/n-SHIFT
    movable = (x > -3.5) & (x < 11.5)
    layout = front.front_layout(K, N, length=L, height=H, movable=movable)
    layout['x'] -= SHIFT
    layout['node_x'] = x
    return K, N, layout


def initial(n, layout):
    nq = len(layout['w'])
    return dict(wall=np.array(0.), mem=np.zeros(n), jump=np.zeros(nq),
                a=((layout['x'] > 2.5) & (layout['x'] < 5.5)).astype(float),
                age=np.ones(nq), rate=np.zeros(n), nodal_jump=np.zeros(n),
                gauss_memory=np.zeros(nq), absolute_slip=np.zeros(nq),
                reference=np.zeros(nq), clock_s=np.array(0.), publications=np.array(0))


def encoded(s):
    return b''.join(k.encode()+np.asarray(s[k]).tobytes() for k in sorted(s))


def bulk(K, s):
    dx = L/len(K)
    return .5*(front.ref.G*L/H*s['wall']**2
               -2*front.ref.G/H*dx*s['wall']*sum(s['mem'])+dx*s['mem']@K@s['mem'])


def bond(s, w):
    a = s['a']
    return w@(.5*(1-a)/(front.ref.C0*np.where(a == 0, 1, a))*s['jump']**2)


def advance(K, N, layout, old, target):
    dt = target-float(old['clock_s'])
    du = RATE*(min(target, LOAD_END)-min(float(old['clock_s']), LOAD_END))
    fallback = False
    try:
        new, d = front.solve_front(K, N, old, dt, du, layout=layout)
    except RuntimeError as e:
        if 'positive linesearch' not in str(e):
            raise
        # Only a new initial guess. Every accepted history, age and clock is
        # identical; signed box feasibility and admissible tangent still gate it.
        new, d = front.solve_front(K, N, old, dt, du, layout=layout,predictor=np.zeros(len(K)))
        fallback = True
    d['predictor_fallback'] = fallback
    dq = layout['Q']@d['increment']
    beta = 1/(1+front.ref.RELAX*dt)
    new.update(nodal_jump=old['nodal_jump']+d['increment'],
               gauss_memory=beta*(old['gauss_memory']+dq),
               absolute_slip=old['absolute_slip']+abs(dq), reference=old['reference'].copy(),
               clock_s=np.array(target), publications=np.array(int(old['publications'])+1))
    assert np.all(new['a'] >= old['a']) and max(new['a']) <= 1
    assert np.max(abs(layout['Q']@new['nodal_jump']-new['jump'])) < 1e-15
    assert np.max(abs(layout['Q']@new['mem']-new['gauss_memory'])) < 1e-15
    assert np.max(abs(new['jump'][d['tied']]), initial=0) == 0
    assert d['curvature'] > 0
    w = layout['w']
    E, Eold, B, Bold = bulk(K, new), bulk(K, old), bond(new, w), bond(old, w)
    fracture = front.ref.GC*(w@(new['a']-old['a']))
    friction = dt*(w@(new['a']*d['pn']*d['mu']*abs(d['V'])))
    radiation = dt*(w@(new['a']*front.ref.ETA_RAD*d['V']**2))
    maxwell = 2*front.ref.RELAX*dt*E
    reaction = L/len(K)*sum(d['traction'])
    oldreaction = front.ref.G*L/H*old['wall']-front.ref.G/H*L/len(K)*sum(old['mem'])
    endpoint, trap = reaction*du, .5*(reaction+oldreaction)*du
    difference = dict(wall=new['wall']-old['wall'], mem=new['mem']-old['mem'])
    be = bulk(K, difference)
    surface = w@(d['cohesive_q']*dq)-(B-Bold)-fracture
    ledger = endpoint-(E-Eold+B-Bold+fracture+friction+radiation+maxwell+be+surface)
    assert abs(ledger) < 1e-10*max(1, abs(endpoint)), (target, ledger)
    assert max(abs(d['traction'][~layout['movable']])) < 3000
    r = dict(time_s=target, dt_s=dt, publications=int(new['publications']),
             imposed_displacement_m=RATE*min(target, LOAD_END),
             integrated_new_damage_m=float(w@(new['a']-initial(len(K), layout)['a'])),
             integrated_abs_jump_m2=float(w@abs(new['jump'])), peak_jump_m=float(max(abs(new['jump']))),
             reaction_N_m=float(reaction), bulk_J_m=float(E), bond_J_m=float(B),
             fracture_increment_J_m=float(fracture), friction_J_m=float(friction),
             radiation_J_m=float(radiation), Maxwell_J_m=float(maxwell),
             endpoint_work_J_m=float(endpoint), trapezoid_work_J_m=float(trap),
             ledger_error_J_m=float(ledger), physical_remainder_J_m=float(trap-(E-Eold+B-Bold+fracture+friction+radiation+maxwell)),
             minimum_normal_Pa=float(min(d['pn'])), weak_error_Pa=float(d['weak_error_Pa']),
             minimum_symmetric_tangent_Pa_m=float(d['curvature']), iterations=d['iterations'],
             max_damage_increment=float(max(new['a']-old['a'])), peak_velocity_m_s=float(max(abs(d['V']))))
    return new, d, r


def qualify():
    # Preserve the existing unit-domain reference's default behavior.
    with zipfile.ZipFile(WORK/'entry-source.zip') as z:
        source = z.read('benchmarks/reconstructed_fault/particle_free/cohesive_interface_front.py').decode()
    entry = {}
    exec(compile(source, 'task-entry-reference', 'exec'), entry)
    K, N = np.eye(8)*8e6, np.zeros((8, 8))
    a, b, identity = entry['initialize'](8), front.initialize(8), 0.
    for _ in range(20):
        a, _ = entry['solve_front'](K, N, a, .002, .00002)
        b, _ = front.solve_front(K, N, b, .002, .00002)
        identity = max(identity, max(float(np.max(abs(np.asarray(a[k])-np.asarray(b[k])))) for k in a))
    assert identity < 1e-12
    # Native frozen incoming histories, actual FE operators and timestep; no
    # history remapping. This isolates the independent surface solver.
    from assess_cohesive_interface_boundary import baseline, load, at, read
    from check_cohesive_interface_kink import point
    c = load('event-half', baseline())
    _, op = read(c['p']/'operators.csv')
    K, N = op[:, 2].reshape(256, 256), op[:, 3].reshape(256, 256)
    _, _, layout = operators(256)
    layout = front.front_layout(K, N, length=L, height=H, movable=layout['movable'])
    comparisons = []
    for target in [2.45, 2.55, 2.5796875]:
        i = at(c, target)
        s = dict(wall=c['h']['wall_memory_m'][i-1], mem=c['n']['relaxed_jump_m'][i-1].copy(),
                 jump=c['s']['jump_m'][i-1].copy(), a=c['s']['damage'][i-1].copy(),
                 age=c['s']['Theta'][i-1].copy(), rate=c['n']['signed_rate_m_s'][i-1].copy())
        dt = c['h']['dt_s'][i]
        du = RATE*(min(target, LOAD_END)-min(float(c['h']['time_s'][i-1]), LOAD_END))
        new, d = front.solve_front(K, N, s, dt, du, layout=layout)
        error = {key:float(max(abs(new[k]-c['s'][key][i]))) for key,k in [('jump_m','jump'),('damage','a'),('Theta','age')]}
        error['memory_m'] = float(max(abs(new['mem']-c['n']['relaxed_jump_m'][i])))
        Q, _, _, P = front.previous.surface_rule(256)
        error['shear_Pa'] = float(max(abs(P@d['traction']-c['s']['shear_Pa'][i])))
        error['normal_Pa'] = float(max(abs(d['pn']-c['s']['pn_Pa'][i])))
        assert error['shear_Pa'] < 1e-7 and error['normal_Pa'] < 1e-7, error
        assert error['jump_m'] < 1e-10 and max(error['damage'], error['Theta']) < 1e-9, error
        # Independently evaluate the full pressure-coupled point return for FD.
        free = layout['movable'] & ~(np.asarray(layout['Q'][d['tied']].sum(axis=0)).ravel()>0)
        # Scale directions with the computed increment: an absolute random
        # direction can jump across the RSF zero-rate transition in cells whose
        # increment is many orders smaller than the front maximum.
        v = np.random.default_rng(42).normal(size=256)*free*d['increment']
        v /= np.linalg.norm(v)
        h = np.linalg.norm(d['increment'])*1e-4
        old = dict(jump=s['jump'], damage=s['a'], theta=s['age'], reference=np.zeros(len(s['jump'])))
        def force(inc):
            z = (s['mem']+inc)/(1+front.ref.RELAX*dt)
            pn = front.ref.P0+layout['PN']@z
            r = point(old, Q@inc, pn, dt)
            return front.ref.G/H*new['wall']-K@z-layout['R']@r['resistance']
        fd = (force(d['increment']+h*v)-force(d['increment']-h*v))/(2*h)
        exact = d['jacobian']@v
        derivative = float(np.linalg.norm((fd-exact)[free])/np.linalg.norm(exact[free]))
        assert derivative < 1e-6, (target, h, derivative)
        comparisons.append(dict(time_s=target, errors=error, derivative_relative=derivative,
                                weak_Pa=d['weak_error_Pa']))
    result = dict(default_reference_history_error=identity, native_frozen_steps=comparisons,
                  all_passed=True, physical_histories_interpolated=False)
    save_json(OUT/'qualification.json', result)
    print(json.dumps(result, indent=2), flush=True)


def run(n, half=False, resume=False):
    assert json.loads((OUT/'qualification.json').read_text())['all_passed']
    name = f'surface-{n}'+('-half' if half else '')
    folder = WORK/name
    folder.mkdir(exist_ok=resume)
    start = time.perf_counter()
    K, N, layout = operators(n)
    setup_seconds = time.perf_counter()-start
    with (PREVIOUS/'event-half-history.csv').open() as f:
        times = np.array([float(r['time_s']) for r in csv.DictReader(f)])
    prior_seconds = 0.
    if resume:
        stopped = json.loads((folder/'failure.json').read_text())
        prior_seconds = stopped['seconds']
        with np.load(folder/'last-accepted.npz') as f:
            s = {k:f[k].copy() for k in f}
        times = times[times > float(s['clock_s'])+1e-10]
        save_json(folder/'resume-info.json',dict(incoming_s=float(s['clock_s']),
                  failed_trial=stopped,history_conversion=False,initial_guess_retry=True))
    elif half:
        with np.load(WORK/f'surface-{n}'/'state-2423750.npz') as f:
            s = {k:f[k].copy() for k in f}
        times = times[times > float(s['clock_s'])+1e-10]
        previous = np.r_[float(s['clock_s']), times[:-1]]
        times = np.c_[(previous+times)/2, times].ravel()
    else:
        s = initial(n, layout)
    selected = set(np.argmin(abs(times-t)) for t in SELECTED if t >= float(s['clock_s']))
    selected.update(np.flatnonzero(times >= 2.3)[::20])
    checkpoints = []
    rows, lifecycle, fallbacks = [], None, []
    if resume:
        with (OUT/(name+'-history.csv')).open() as f:
            rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        assert abs(rows[-1]['time_s']-float(s['clock_s'])) < 1e-12
    history = (OUT/(name+'-history.csv')).open('a' if resume else 'w')
    writer = csv.DictWriter(history,fieldnames=list(rows[0])) if resume else None
    try:
        for i, target in enumerate(times):
            before = encoded(s)
            tic = time.perf_counter()
            trial, d, r = advance(K, N, layout, s, target)
            if d['predictor_fallback']:
                fallbacks.append(dict(time_s=target, incoming_s=float(s['clock_s']),
                                      accepted_weak_Pa=d['weak_error_Pa'],
                                      curvature_Pa_m=d['curvature'],
                                      maximum_damage_increment=r['max_damage_increment']))
            assert encoded(s) == before
            if lifecycle is None and r['max_damage_increment'] > 0:
                retry, _, rr = advance(K, N, layout, s, target)
                assert encoded(retry) == encoded(trial) and rr == r
                np.savez_compressed(folder/'lifecycle-incoming.npz', **s)
                with np.load(folder/'lifecycle-incoming.npz') as f:
                    restored = {k:f[k].copy() for k in f}
                restarted, _, _ = advance(K, N, layout, restored, target)
                assert encoded(restarted) == encoded(trial) and encoded(s) == before
                lifecycle = dict(time_s=target, rejected_accepted_unchanged=True,
                                 retry_bitwise=True, restart_bitwise=True, logical_counter_once=True)
            assert int(trial['publications']) == int(s['publications'])+1
            s = trial
            r['step_seconds'] = time.perf_counter()-tic
            if writer is None:
                writer = csv.DictWriter(history, fieldnames=list(r))
                writer.writeheader()
            writer.writerow(r)
            rows.append(r)
            if i in selected or i == len(times)-1:
                path = folder/f"state-{round(target*1e6)}.npz"
                np.savez_compressed(path, **s)
                checkpoints.append(dict(time_s=target, path=str(path.relative_to(ROOT))))
            if i % 20 == 0:
                history.flush()
                np.savez_compressed(folder/'last-accepted.npz', **s)
                progress = dict(case=name, accepted_time_s=target, steps=len(rows),
                                seconds=time.perf_counter()-start,
                                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                save_json(WORK/(name+'-progress.json'), progress)
                assert progress['seconds'] < 900, '900 s computational budget'
                assert progress['peak_RSS_MiB'] < 3072, '3 GiB RSS budget'
                print(json.dumps(progress), flush=True)
    except Exception as e:
        np.savez_compressed(folder/'last-accepted.npz', **s)
        save_json(folder/'failure.json', dict(target_s=float(target), error=repr(e), seconds=time.perf_counter()-start))
        raise
    finally:
        history.close()
    result = dict(case=name, n=n, dx_m=L/n, accepted_steps=len(rows), final_time_s=float(s['clock_s']),
                  seconds=time.perf_counter()-start+prior_seconds, setup_seconds=setup_seconds,
                  prior_stopped_command_seconds=prior_seconds, predictor_recoveries=fallbacks,
                  peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  lifecycle=lifecycle, checkpoints=checkpoints, fresh_initialization=not half,
                  incoming_checkpoint_time_s=2.42375 if half else 0.,
                  minimum_normal_Pa=min(r['minimum_normal_Pa'] for r in rows),
                  minimum_symmetric_tangent_Pa_m=min(r['minimum_symmetric_tangent_Pa_m'] for r in rows),
                  maximum_weak_Pa=max(r['weak_error_Pa'] for r in rows),
                  maximum_ledger_J_m=max(abs(r['ledger_error_J_m']) for r in rows),
                  final=rows[-1])
    save_json(OUT/(name+'-run.json'), result)
    print(json.dumps(result, indent=2), flush=True)


def check_lifecycle():
    checks = []
    for n in [256, 512]:
        K, N, layout = operators(n)
        with np.load(WORK/f'surface-{n}'/'state-2423750.npz') as f:
            old = {k:f[k].copy() for k in f}
        target = float(old['clock_s'])+.000625
        before = encoded(old)
        trial, _, r = advance(K,N,layout,old,target)
        assert encoded(old) == before
        path = WORK/f'lifecycle-{n}.npz'
        np.savez_compressed(path, **old)
        with np.load(path) as f:
            restored = {k:f[k].copy() for k in f}
        retry, _, _ = advance(K,N,layout,restored,target)
        assert encoded(retry) == encoded(trial) and encoded(old) == before
        checks.append(dict(n=n, incoming_s=float(old['clock_s']), target_s=target,
                           serialization_npz=True, all_state_arrays_bitwise=True,
                           rejection_accepted_unchanged=True,
                           counter_increment=int(trial['publications'])-int(old['publications'])))
    # Reproduce the preserved hold predictor stop, then change only its guess.
    K,N,layout=operators(512)
    with np.load(WORK/'surface-512/state-2550000.npz') as f:
        old={k:f[k].copy() for k in f}
    target=json.loads((WORK/'surface-512/failure.json').read_text())['target_s']
    before=encoded(old)
    try:
        front.solve_front(K,N,old,target-float(old['clock_s']),0.,layout=layout)
    except RuntimeError as e:
        original_error=str(e)
        assert 'positive linesearch' in original_error
    else:
        raise AssertionError('Preserved predictor stop was not reproduced')
    trial,d,r=advance(K,N,layout,old,target)
    assert d['predictor_fallback'] and encoded(old)==before
    save_json(OUT/'lifecycle-verification.json',dict(serialization=checks,
              hold_predictor_recovery=dict(incoming_s=float(old['clock_s']),target_s=target,
              original_error=original_error,accepted_history_sha256=hashlib.sha256(before).hexdigest(),
              accepted_state_unchanged=True,only_initial_guess_changed=True,
              weak_error_Pa=d['weak_error_Pa'],curvature_Pa_m=d['curvature'],
              maximum_damage_increment=r['max_damage_increment'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['qualify', 'check-lifecycle', '256', '512', '512-resume', '512-half'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    WORK.mkdir(exist_ok=True, parents=True)
    if args.stage == 'qualify':
        qualify()
    elif args.stage == 'check-lifecycle':
        check_lifecycle()
    else:
        run(int(args.stage.split('-')[0]), args.stage.endswith('-half'),args.stage.endswith('-resume'))
