#!/usr/bin/env python3
"""Compare retained accepted states after changing only diagnostic cadence."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import json
import numpy as np
from phase import coefficients, read

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
parser = argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--steps', type=int, default=7)
args = parser.parse_args()
old = RUNS / 'stage4-propagation-free08/output'
new = RUNS / ('stage4-propagation-' + args.label) / 'output'
results = {}
for step in range(args.steps + 1):
    delta = coefficients(new, step) - coefficients(old, step)
    row = {'phase_max_absolute': float(np.max(np.abs(delta)))}
    for name in ['pilot-sections'] + (['tip-planes', 'tip-sections'] if step else []):
        before = read(old / f'{name}-{step:04d}.csv')
        after = read(new / f'{name}-{step:04d}.csv')
        assert before.shape == after.shape and before.dtype.names == after.dtype.names
        row[name] = {key: float(np.max(np.abs(after[key] - before[key])))
                     for key in before.dtype.names}
    results[str(step)] = row
for name in ['bulk-accepted', 'pilot-summary']:
    before, after = (read(path / (name + '.csv')) for path in [old, new])
    before = before[before['step'] <= args.steps]
    after = after[after['step'] <= args.steps]
    assert before.shape == after.shape
    results[name] = {key: float(np.max(np.abs(after[key] - before[key])))
                     for key in before.dtype.names}
(EVIDENCE / (args.label + '-prefix-comparison.json')).write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
