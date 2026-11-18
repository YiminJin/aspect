"""Bounded search-order comparison from the retained fine-grid step 88."""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('order', choices=['positive', 'default'])
a = p.parse_args()
parent = Path('build-particle-free/bulk-tip/runs/fine-growth-warm01')
env = os.environ.copy()
env.pop('ASPECT_TIP_POSITIVE_METRIC_FIRST', None)
env['ASPECT_TIP_TRACE_SEARCH'] = '1'
env['ASPECT_TIP_RESTORE_AUDIT'] = '1'
if a.order == 'positive':
    env['ASPECT_TIP_POSITIVE_METRIC_FIRST'] = '1'
command = ['python3', 'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py',
           'step89-search-' + a.order,
           '--executable', 'build-particle-free/material-overnight/native/aspect-release',
           '--checker', 'build-particle-free/material-overnight/checks/libbulk_tip_checks.release.so',
           '--fixture', str(parent / 'input.prm'), '--time-update', 'midpoint',
           '--ranks', '2', '--refinement', '1', '--dt', '.000625', '--end', '.06',
           '--end-step', '89', '--checkpoint-steps', '1', '--phase-balance',
           '--integrity-stop', '--resume-from', str(parent / 'output'),
           '--snapshot-id', '2', '--seconds', '2400']
subprocess.run(command, env=env, check=True)
