#!/usr/bin/env python3
"""Short provider rejection followed by a fresh, identical accepted-clock control.

Run through run_logged.py. This is benchmark orchestration only; the existing
ASPECT rejection and Function timestep plugins perform the two trajectories.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STAGE = ROOT / 'build-particle-free/transported-objective/stage-c'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign'


def main():
    env = os.environ.copy()
    env['LAYERED_EVIDENCE'] = str(OUT)
    common = ['--candidate', str(STAGE / 'layered-bulk-cartesian'),
              '--spin', 'bulk slip subtracted', '--background-shear', '3.7',
              '--translation', '2', '--halfwidth', '6', '--level', '4',
              '--dt', '.0125', '--end', '.05', '--ranks', '4', '--seconds', '900']

    def run(script, *args):
        subprocess.run([sys.executable, str(HERE / script), *args],
                       cwd=ROOT, env=env, check=True)

    rejected = 'layered-bulk-provider-reject'
    control = 'layered-bulk-provider-control'
    run('run_layered_campaign.py', 'bulk-provider-reject', *common, '--reject')
    output = STAGE / 'material-runs' / rejected / 'output'
    with (output / 'material-summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert int(rows[-1]['retries']) == 1, rows[-1]
    with (output / 'layered-background.csv').open() as f:
        background = list(csv.DictReader(f))
    assert max(int(r['retries']) for r in background) == 1

    # The Function plugin evaluates at the accepted parent time. Midpoints
    # between consecutive parents avoid roundoff-sensitive endpoint tests.
    steps = rows[1:]
    expression = format(float(steps[-1]['dt']), '.17g')
    for i in range(len(steps)-2, -1, -1):
        parent = float(rows[i]['time'])
        next_parent = float(rows[i+1]['time'])
        threshold = .5 * (parent + next_parent)
        expression = f"if(time<{threshold:.17g},{float(steps[i]['dt']):.17g},{expression})"
    (OUT / 'provider-rejection-schedule.json').write_text(json.dumps(
        dict(rejected=rejected, control=control, accepted_steps=steps,
             function=expression), indent=2) + '\n')
    run('run_layered_campaign.py', 'bulk-provider-control', *common,
        '--function', expression)
    run('check_layered_reference.py', rejected, control)
    run('check_layered_lifecycle.py', rejected, control)


if __name__ == '__main__':
    main()
