#!/usr/bin/env python3
"""Read-only native restart and signed-deformation audit at the fine parent.

The stored rate/reaction are last-stage observations; stress, A, phase, age and
slip are accepted endpoint history. Reconstruct the actual affine Gauss measure
and compare the volume integrals with the native section observations.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-preparation-mpl')
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import H, coefficients, read, value

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
OUT = HERE.parent / 'evidence/stage4_preparation'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--coarse-view', type=Path, required=True)
parser.add_argument('--coarse-family', default='free76')
parser.add_argument('--fine-family', default='half')
parser.add_argument('--parent-step', type=int, default=60)
args = parser.parse_args()
gauss, weights = np.polynomial.legendre.leggauss(6)
gauss = (gauss+1)/2
weights = weights/2
records = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for law in ('weakening', 'neutral'):
    coarse = args.coarse_view / f'stage4-preparation-{args.coarse_family}-{law}' / 'output'
    run = RUNS / f'stage4-preparation-{args.fine_family}-{law}'
    output = run / 'output'
    paths = sorted(output.glob('restore-points-r*.csv*'))
    assert len(paths) == 4
    point = np.concatenate([np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, 17)) for path in paths])
    q = point[:, 0].astype(int)
    x, y, phi = point[:, 1:4].T
    assert len(point) == 256*128*36 and q.max() == 35
    coordinate_error = max(float(max(abs(x/H-np.floor(x/H)-gauss[q % 6]))),
                           float(max(abs(y/H-np.floor(y/H)-gauss[q // 6]))))
    assert coordinate_error < 1e-12
    measure = H**2*weights[q % 6]*weights[q // 6]
    assert abs(measure.sum()-32) < 1e-12
    parent_phase = coefficients(coarse, args.parent_step)
    phase_error = float(max(abs(phi-value(parent_phase, x, y))))
    assert phase_error < 1e-13
    tau_trace = float(max(abs(point[:, 4]+point[:, 5]+point[:, 6])))
    a_trace = float(max(abs(point[:, 8]+point[:, 9]+point[:, 10])))
    assert tau_trace < 1e-12*max(abs(point[:, 4:8]).max(), 1e-30)
    assert a_trace < 1e-12*max(abs(point[:, 8:12]).max(), 1e-30)
    # Native h.rate is the last accepted-stage fault deformation only.
    w = point[:, 13]
    signed, absolute = point[:, 14:16].T
    assert max(abs(signed)-absolute) < 1e-12*max(absolute.max(), 1e-30)
    local = read(coarse / f'tip-sections-{args.parent_step:04d}.csv')
    right = np.minimum(np.searchsorted(local['s'], x), len(local)-1)
    left = np.maximum(right-1, 0)
    index = np.where(abs(local['s'][right]-x) <= abs(local['s'][left]-x), right, left)
    assert max(abs(local['s'][index]-x)) < 1e-12
    positive = np.bincount(index, weights=measure*np.maximum(w, 0), minlength=len(local))
    negative = np.bincount(index, weights=measure*np.minimum(w, 0), minlength=len(local))
    positive_error = float(max(abs(positive-local['ds']*local['positive_w'])))
    negative_error = float(max(abs(negative-local['ds']*local['negative_w'])))
    rate_scale = float(sum(positive-negative))
    assert max(positive_error, negative_error) < 1e-11*max(rate_scale, 1e-30)
    restored = read(next(output.glob('restore-sections.csv*')))
    memory_errors = {key:float(max(abs(restored[key]-local[key]))) for key in ['s', 'ds', 'age', 'slip', 'V', 'force']}
    assert max(memory_errors.values()) < 1e-12
    command = json.loads((run / 'command.json').read_text())
    snapshot = command['arguments']['snapshot_id']
    hashes = {}
    for path in sorted((output / 'restart' / f'{snapshot:02d}').glob('*')):
        if path.is_file():
            name = f'restart/{snapshot:02d}/{path.name}'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == command['restart_source_sha256'][name]
            hashes[name] = digest
    assert hashes
    regional = {}
    for side, lower, upper in [('left', 2., 2.5), ('right', 5.5, 6.)]:
        mask = (local['s'] > lower) & (local['s'] < upper)
        pos, neg = float(sum(positive[mask])), float(sum(negative[mask]))
        regional[side] = dict(positive=pos, negative=neg,
                             cancellation_fraction=1-abs(pos+neg)/(pos-neg) if pos > neg else 0.,
                             positive_force_power=float(sum(positive[mask]*local['force'][mask])),
                             negative_force_power=float(sum(negative[mask]*local['force'][mask])))
    records[law] = dict(coordinate_quadrature_error=coordinate_error, maximum_restored_phase_error=phase_error,
                        maximum_deviator_stress_trace=tau_trace, maximum_cohesive_strain_trace=a_trace,
                        positive_measure_identity_error=positive_error, negative_measure_identity_error=negative_error,
                        section_memory_errors=memory_errors, copied_parent_checkpoint_sha256=hashes,
                        regional_signed_deformation=regional)
    for side, target, ax in zip(['left', 'right'], [2., 6.], axes):
        chosen = local['s'][np.argmin(abs(local['s']-target))]
        mask = abs(x-chosen) < 1e-12
        order = np.argsort(y[mask])
        ax.plot(y[mask][order], w[mask][order], color='tab:red' if law=='weakening' else 'tab:blue', label=law+f', x={chosen:.6f} m')
        ax.set(xlim=(1.65, 2.35), xlabel='y (m)', ylabel='Stored last-stage fault rate w (1/s)', title=side+' section: signed native deformation')
        ax.axhline(0, color='k', lw=.5); ax.legend(fontsize=7)
result = dict(laws=records, parent_step=args.parent_step,
              measure='Native 6x6 Gauss rule and affine cell Jacobian; volume positive/negative integrals independently match ds times native section observations.',
              history_meaning='tau/A/phase and section age/slip are endpoints; w/reaction are stored last-stage observations. Force times signed w is a nonlocal work contribution, not pointwise local friction dissipation.')
(OUT / f'{args.fine_family}-restore-audit.json').write_text(json.dumps(result, indent=2)+'\n')
for extension in ('png', 'pdf'):
    fig.savefig(OUT / f'{args.fine_family}-signed-deformation.{extension}', dpi=170)
print(json.dumps(result, indent=2))
