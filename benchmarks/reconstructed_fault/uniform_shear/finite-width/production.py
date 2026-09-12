"""Bounded K4.2 execution and read-only guards; no reference-history resetting."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path, name, step):
    file = path/f'{name}_{step}.csv'
    with file.open() as stream:
        names = stream.readline().strip().split(',')
    return np.atleast_1d(np.loadtxt(file, delimiter=',', skiprows=1,
                                   dtype=[(n, float) for n in names]))


def check(path, step):
    start = time.monotonic()
    s = read(path, 'surface', step)
    s0 = read(path, 'surface', 0)
    phase = read(path, 'phase', step)
    bulk = read(path, 'bulk', step)
    weak = read(path, 'surface_weak', step)
    seg = read(path, 'segments', step)
    t = read(path, 'time', step)[0]
    particles = read(path, 'particles', step)
    checks = dict(finite=all(np.isfinite(a[n]).all() for a in (s,phase,bulk,weak,particles)
                            for n in a.dtype.names))
    checks['geometry_fixed'] = all(np.array_equal(s[n],s0[n]) for n in ('fault','node','x','y'))
    checks['geometry_horizontal'] = max(abs(s['y'])) < 1e-8 and max(abs(seg['ny']-1)) < 1e-8
    checks['fault32'] = len(s)==33 and abs(s['x'][0])<1e-8 and abs(s['x'][-1]-.25)<1e-8
    checks['phase_admissible'] = min(phase['phi'])>=-1e-4 and max(phase['phi'])<.8
    checks['sequence'] = abs(t['time']-step*.125)<1e-12 and abs(t['dt']-(.125 if step else 2))<1e-12
    lengths = np.hypot(np.diff(s['x']),np.diff(s['y']))
    weights = .5*(np.r_[lengths,0]+np.r_[0,lengths])
    means = {n:float(weights@s[n]/sum(lengths)) for n in ('V','C','Theta','Ih')}
    checks['interior_positive_V'] = min(s['V'])>1e-12
    # Match existing benchmark validation: positive domains, measured total.
    # Do not invent a tighter conservation gate than the accepted fixtures.
    checks['positive_particle_measure'] = min(particles['volume'])>0

    # Independent integration of the frozen Q1 FE profile: every transverse
    # interval is split at the actual support boundary, never tail-renormalized.
    xs, ys = np.unique(phase['x']), np.unique(phase['y'])
    grid = np.empty((len(xs),len(ys)))
    grid[np.searchsorted(xs,phase['x']),np.searchsorted(ys,phase['y'])] = phase['phi']
    ell = .15625 if path.name=='k42_A' else .078125
    m = 128*.15625/ell
    absc, gauss = np.polynomial.legendre.leggauss(8)
    lower, upper = float(min(seg['half_width_minus'])), float(min(seg['half_width_plus']))
    breaks = np.unique(np.r_[ys,-lower,upper])
    y = breaks[:-1,None]+np.diff(breaks)[:,None]*(absc+1)/2
    w = np.diff(breaks)[:,None]*gauss/2
    integrals, omitted = [], []
    for column in grid:
        p = np.maximum(np.interp(y,ys,column),0)
        h = m*p*(1+p)/(1-p)**2
        integral = float(np.sum(w*h))
        integrals.append(integral)
        omitted.append(float(np.sum(w*h*((y < -lower)|(y > upper)))/integral))
    checks['containment'] = max(omitted)<=1e-4
    # Every actual bulk-QP x-column, not just its bulk average.
    x, inverse = np.unique(bulk['x'],return_inverse=True)
    wx = np.bincount(inverse, weights=bulk['weight'])
    instantaneous = np.bincount(inverse,weights=bulk['weight']*bulk['chi']*bulk['V'])/wx
    history = np.bincount(inverse,weights=bulk['weight']*bulk['history'])/wx
    local_V = np.interp(x,s['x'],s['V'])
    norm_error = abs((instantaneous+history)/local_V-1)
    checks['normalization'] = max(norm_error)<=1e-4
    checks['homogeneity'] = np.ptp(s['V'])/1e-4<=1e-4
    mass = np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
    traction = np.linalg.solve(mass,weak['q'])
    balance = np.linalg.solve(mass,weak['F'])
    balance_rms = float(np.sqrt(balance@mass@balance/np.sum(mass)))
    checks['weak_balance'] = balance_rms/1500<=1e-8
    theta_error = C_error = 0.
    if step:
        old = read(path,'surface',step-1)
        decay = np.exp(-s['V']*.125/.001)
        expected = old['Theta']*decay+.001/s['V']*(-np.expm1(-s['V']*.125/.001))
        theta_error = float(max(abs(s['Theta']-expected)))
        kappa = -1e8*math.expm1(-.125/100)
        C_error = float(max(abs(s['C']-(math.exp(-.125/100)*old['C']+kappa*s['V']/s['Ih']))))
        checks['theta_update'] = theta_error<1e-10
        checks['cohesive_update'] = C_error<1e-5
    else:
        checks['retained_initial_histories'] = max(abs(s['Theta']-200))<1e-9 and max(abs(particles['tau_xy']-1500))<1e-10
    report = dict(step=step,time_s=float(t['time']),checks={k:bool(v) for k,v in checks.items()},
                  passed=bool(all(checks.values())),means=means,
                  independent_Ih_range=[min(integrals),max(integrals)],
                  max_omitted_fraction=max(omitted),max_normalization_error=float(max(norm_error)),
                  instantaneous_range=[float(min(instantaneous)),float(max(instantaneous))],
                  history_range=[float(min(history)),float(max(history))],
                  actual_weak_q_mean=float(weights@traction/sum(lengths)),
                  actual_weak_q_range=[float(min(traction)),float(max(traction))],
                  balance_rms_Pa=balance_rms,theta_update_error=theta_error,C_update_error=C_error,
                  phi_range=[float(min(phase['phi'])),float(max(phase['phi']))],
                  phi_along_fault_range=float(np.ptp(grid,axis=0).max()),
                  particle_volume_sum=float(np.sum(particles['volume'])),
                  particle_volume_relative_error=float(np.sum(particles['volume'])/.25-1),
                  normal_half_widths=[lower,upper],seconds=time.monotonic()-start)
    (path/f'k4_guard_{step}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('K4 accepted-state guard: '+json.dumps(report),flush=True)
    return 0 if report['passed'] else 2


def run(case):
    path = HERE/case
    assert not path.exists(), 'Preserve evidence; no automatic retry.'
    binary = ROOT/'build-pf-cpdi/aspect-release'
    plugin = HERE/'build/libuniform_shear.release.so'
    command = [str(binary),str(path.with_suffix('.prm'))]
    cap = 2400 if case=='k42_C' else 1200
    start = time.monotonic()
    with path.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(command,cwd=HERE/'build',stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                               'ASPECT_K4_STATE_GUARD':'1'})
        try:
            status = process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            status = 124
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(case=case,status=status,wall_seconds=time.monotonic()-start,cap_s=cap,
                  peak_rss_KiB=usage.ru_maxrss,command=command,
                  sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (binary,plugin,path.with_suffix('.prm'))},
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    path.with_suffix('.resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run'])
    parser.add_argument('path')
    parser.add_argument('--step',type=int)
    args=parser.parse_args()
    raise SystemExit(check(Path(args.path),args.step) if args.action=='check' else run(args.path))
