#!/usr/bin/env python3
"""Apply retained equivalence criteria to stress-consolidation controls."""
import json
from pathlib import Path
import re
import check_code_readability as controls
from check_aspect_dispatch import compare_csv
from check_stress_history_consolidation import match
from run_stress_consolidation import ROOT, OUT, RUNS


def main():
    controls.OUT = OUT
    controls.CASES = {k: 'readability-'+k for k in ('dc','sc','dl','sl','default','reject')}
    controls.CASES['restart4'] = 'readability-restart4-control'
    controls.CANDIDATES = {k: 'consolidation-'+k for k in controls.CASES}
    controls.main()
    lifecycle = [match('consolidation-dl','consolidation-restart_new'),
                 match('true-cutback-control','consolidation-reject'),
                 match('consolidation-dl','consolidation-midpoint')]
    assert len(lifecycle[0]['steps']) == 2
    for result in lifecycle:
        for step in result['steps']:
            assert step['tensor_L2'] < 5e-12 and step['work_traction_Linf'] < 5e-12
    stationary = ROOT/'build-particle-free/transported-objective/stage-c/runs'
    before, after = [stationary/name/'output' for name in ('readability-stationary','consolidation-stationary')]
    files = ['accepted_steps.csv','objective-dg-summary.csv','profiles/fault_0.csv',
             'profiles/fault_1.csv','objective-dg-final-rank0.csv','objective-dg-final-rank1.csv']
    rows = [compare_csv(before/f, after/f) for f in files]
    pattern = r'Fault linear solve: iterations=(\d+)'
    assert re.findall(pattern,(before.parent/'run.log').read_text()) == re.findall(pattern,(after.parent/'run.log').read_text())
    old = ROOT/'build-particle-free/code-readability/nonfault-after/output/statistics'
    new = ROOT/'build-particle-free/stress-consolidation-guide/nonfault/output/statistics'
    assert old.read_bytes() == new.read_bytes()
    data = dict(lifecycle=lifecycle,stationary=rows,nonfault_statistics_byte_identical=True)
    target = OUT/'auxiliary-equivalence.json'
    assert not target.exists()
    target.write_text(json.dumps(data,indent=2)+'\n')
    # Material source/rotation energy observations must retain their exact labels
    # and numerical values at accepted physical steps (initial checker probes are
    # separate nonpublishing calls).
    energy = {}
    for case in ('dc','dl'):
        logs = [(RUNS/(prefix+case)/'run.log').read_text() for prefix in ('readability-','consolidation-')]
        pattern = r'^\s*(Direct source energy: step=[1-9].*)$'
        a,b = [re.findall(pattern,log,re.M) for log in logs]
        assert a and a == b, case
        energy[case] = dict(observations=len(a),byte_identical=True)
    (OUT/'energy-equivalence.json').write_text(json.dumps(energy,indent=2)+'\n')
    print('Lifecycle, stationary, non-fault and physical-step energy observations pass.')


if __name__ == '__main__':
    main()
