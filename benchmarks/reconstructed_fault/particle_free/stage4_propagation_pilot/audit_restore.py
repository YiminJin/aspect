#!/usr/bin/env python3
"""Audit this pilot's native restored parent without changing its state."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import numpy as np
from phase import coefficients, value, read

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
parser = argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--parent', default='pilot-adaptive')
parser.add_argument('--step', type=int, default=9)
args = parser.parse_args()
out = RUNS / ('stage4-propagation-' + args.label) / 'output'
parent = RUNS / ('stage4-propagation-' + args.parent) / 'output'
find = lambda name: next(out.glob(name + '*'))
clock = read(find('restore-clock.csv'))[0]
assert clock['accepted_step'] == args.step and clock['publications'] == args.step
assert clock['pending_step'] == args.step + 1
assert abs(clock['pending_time']-clock['accepted_time']-clock['pending_dt']) < 1e-14
local = read(parent / f'tip-sections-{args.step:04d}.csv')
restored = read(find('restore-sections.csv'))
errors = {key: float(np.max(abs(restored[key]-local[key])))
          for key in ('s', 'ds', 'age', 'slip', 'V', 'force')}
assert max(errors.values()) < 1e-12, errors
planes = read(parent / f'tip-planes-{args.step:04d}.csv')
restored_planes = read(find('restore-planes.csv'))
plane_errors = {key: float(np.max(abs(restored_planes[key]-planes[key])))
                for key in ('x', 'ds', 'angle', 'formed')}
assert max(plane_errors.values()) == 0., plane_errors
native_phase = np.concatenate([np.loadtxt(p, delimiter=',', skiprows=1, usecols=(2, 3))
                               for p in sorted(out.glob('restore-phase-r*.csv*'))])
assert len(native_phase) == 256*128*9
assert np.array_equal(native_phase[:, 0], native_phase[:, 1])
b = coefficients(parent, args.step)
gauss, weights = np.polynomial.legendre.leggauss(6)
gauss = (gauss+1)/2; weights = weights/2
positive = np.zeros(len(local)); negative = positive.copy()
phase_error = trace_tau = trace_A = bulk = volume = 0.
for path in sorted(out.glob('restore-points-r*.csv*')):
    points = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, 17))
    q = points[:, 0].astype(int); x, y, phi = points[:, 1:4].T
    assert np.max(abs(x*32-np.floor(x*32)-gauss[q % 6])) < 1e-12
    assert np.max(abs(y*32-np.floor(y*32)-gauss[q // 6])) < 1e-12
    measure = weights[q % 6]*weights[q // 6]/32**2
    volume += measure.sum()
    phase_error = max(phase_error, float(np.max(abs(phi-value(b, x, y)))))
    tau, strain = points[:, 4:8], points[:, 8:12]
    trace_tau = max(trace_tau, float(np.max(abs(tau[:, :3].sum(axis=1)))))
    trace_A = max(trace_A, float(np.max(abs(strain[:, :3].sum(axis=1)))))
    bulk += np.dot(measure, (np.sum(tau[:, :3]**2, axis=1)+2*tau[:, 3]**2)/(4e6))
    right = np.minimum(np.searchsorted(local['s'], x), len(local)-1)
    left = np.maximum(right-1, 0)
    index = np.where(abs(local['s'][right]-x) <= abs(local['s'][left]-x), right, left)
    assert np.max(abs(local['s'][index]-x)) < 1e-12
    positive += np.bincount(index, weights=measure*np.maximum(points[:, 13], 0), minlength=len(local))
    negative += np.bincount(index, weights=measure*np.minimum(points[:, 13], 0), minlength=len(local))
assert abs(volume-32) < 1e-12 and phase_error < 1e-13
energy = read(parent / 'bulk-accepted.csv')
energy = energy[energy['step'] == args.step][0]
bulk_error = float(bulk-energy['bulk'])
assert abs(bulk_error) < 1e-11*abs(energy['bulk'])
positive_error = float(np.max(abs(positive-local['ds']*local['positive_w'])))
negative_error = float(np.max(abs(negative-local['ds']*local['negative_w'])))
assert max(positive_error, negative_error) < 1e-11*np.sum(positive-negative)
record = dict(clock={k: float(clock[k]) for k in clock.dtype.names},
              section_errors=errors, plane_errors=plane_errors,
              native_phase_solution_difference=0., maximum_phase_value_error=phase_error,
              volume=float(volume), bulk_energy_error=bulk_error,
              stress_trace=trace_tau, cohesive_strain_trace=trace_A,
              positive_measure_error=positive_error, negative_measure_error=negative_error,
              limitation='Full pointwise tau/A predecessor exports were sparse; tensor payload identity is protected by checkpoint hashes. Bulk stress energy and full 3D traces are independently checked here. Stored rate is the last-stage fault rate, not ordinary bulk deformation.')
(EVIDENCE / (args.label+'-restore-audit.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
