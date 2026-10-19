#!/usr/bin/env python3
"""Read-only P2 physical-response replay; launches no native or server job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO/'benchmarks/reconstructed_fault/particle_free'))
from assess_p2_response import compare, PhysicalCase
from assess_phase_energy import csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protected',type=Path)
    a=p.parse_args();work=a.root/'build-particle-free/p2-response'
    checks={};identities=0
    for label in ['unloaded','primary','half']:
        run=a.root/'build-particle-free/runs'/f'p2-response-{label}'
        record=json.loads((run/'record.json').read_text());assert record['gate_passed']
        for name,digest in record['outputs'].items():
            assert sha(run/'run'/name)==digest,name
            identities+=1
        m=np.atleast_1d(csv(run/'run/metrics.csv'));g=np.atleast_1d(csv(run/'run/gates.csv'))
        assert np.array_equal(m['publications'],m['step']) and all(g['passed'])
        assert np.array_equal(m['time'],np.arange(len(m))/record['divisor'])
        assert np.all(m['dt'][1:]==1/record['divisor']) and m[-1]['time']==record['end_time']
        assert max(m['healing'])==0.
        case=PhysicalCase(work/f'native-{label}',True)
        assert np.min(case.f['phase'])>=0. and np.max(case.f['phase'])<1.
        assert np.min(case.f['I'])>0.
        checks[label]=dict(accepted_steps=int(m[-1]['step']),native_gates=len(g),
            healing=float(max(m['healing'])),phase_growth=case.report['phase_growth'],
            maximum_x_spread=case.report['max_x_spread'])
    assert checks['unloaded']['phase_growth']==0.
    assert checks['primary']['phase_growth']>.001 and checks['half']['phase_growth']>.001
    # Recompute unaligned common-field comparisons and physical work summaries.
    comparison_path=a.output.with_name(a.output.stem+'-comparison.json')
    result=compare(work,comparison_path)
    assert result['all_practical_comparisons_passed']
    assert not result['strict_reference_match_passed']
    init=result['initialization']['meshes'][0]
    assert init['relative_L2']<=1e-4 and init['relative_gradient_L2']<=.02 and init['unloaded_KKT']<=1e-11
    snapshots=[]
    for label in ['primary','half']:
        rows=json.loads((work/f'native-{label}-snapshots.json').read_text())
        for row in rows:
            assert row['physical_stress_error']<1e-9 and row['gradient_error']<1e-9
            assert max(row['native_weak_load_error'].values())<1e-9
            assert row['native_source_recurrence_error']<1e-9
            assert row['projected_contact_load_Linf']<1e-8
        snapshots.extend(rows)
    protected=0
    if a.protected:
        for name,data in json.loads(a.protected.read_text())['files'].items():
            assert sha(REPO/name)==data['sha256'],name
            protected+=1
    output=dict(passed=True,native=checks,raw_csv_identities=identities,
        protected_bp3_identities=protected,independent_snapshots=len(snapshots),
        practical_comparisons=result['passes'],strict_reference_match_passed=False,
        scope='bounded serial fixed-geometry cohesive physical response; no P3 or general propagation')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
