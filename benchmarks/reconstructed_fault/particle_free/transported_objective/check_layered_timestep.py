#!/usr/bin/env python3
"""Audit the existing pre-publication age predictor against accepted CSV history.

This is a benchmark diagnostic, not a new timestep controller. The mature DG
publication gate runs after TimeStepping::Manager::update(). Consequently step
n's proposal uses committed surface n-2 (zero for the first two steps), whereas
its actual age update uses surface n-1 and endpoint V_n.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/transported-objective/stage-c/material-runs'
OUT = Path(os.environ.get('LAYERED_EVIDENCE', str(ROOT /
    'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign')))


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def predicted_change(surface, dt):
    changes = []
    for row in surface:
        theta, velocity = float(row['Theta_s']), abs(float(row['V_m_s']))
        k = velocity / .008
        predicted = theta * math.exp(-k*dt) - math.expm1(-k*dt)/k if k else theta+dt
        changes.append(abs(math.log(predicted/theta)))
    return max(changes)


def audit(case):
    output = RUNS / case / 'output'
    arguments = json.loads((output.parent / 'command.json').read_text())['arguments']
    rows = read(output / 'material-summary.csv')
    assert int(rows[0]['step']) == 0, 'Use the uninterrupted trajectory for this audit'
    surfaces = {int(r['step']): read(output / f"material-{r['step']}-surface.csv") for r in rows}
    records = []
    for row in rows[1:]:
        n, dt = int(row['step']), float(row['dt'])
        parent, endpoint = surfaces[n-1], surfaces[n]
        records.append(dict(step=n, time=float(row['time']), dt=dt,
            predictor_surface_step=max(0,n-2),
            prepublication_prediction=predicted_change(surfaces[max(0,n-2)], dt),
            latest_parent_prediction=predicted_change(parent, dt),
            actual=max(abs(math.log(float(b['Theta_s'])/float(a['Theta_s'])))
                       for a,b in zip(parent,endpoint))))
    result = dict(case=case, records=records, steps=len(records),
                  dt_min=min(r['dt'] for r in records),dt_max=max(r['dt'] for r in records),
                  maxima={k:max(r[k] for r in records) for k in
                          ('prepublication_prediction','latest_parent_prediction','actual')},
                  interpretation='Existing controller prediction precedes mature DG publication; this is not an actual endpoint age-change acceptance bound.')
    if arguments.get('adaptive'):
        bound = arguments['max_log_state_change']
        excess = max(r['prepublication_prediction']-bound for r in records)
        result.update(requested_prediction_bound=bound, maximum_prediction_excess=excess,
                      actual_steps_above_prediction_bound=sum(r['actual']>bound+1e-12 for r in records))
        assert excess < 1e-12, ('Pre-publication prediction does not explain schedule',excess)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / ('timestep-'+case+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+')
    for name in parser.parse_args().cases:
        audit(name)
