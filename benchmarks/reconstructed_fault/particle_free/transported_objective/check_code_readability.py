#!/usr/bin/env python3
"""Compare ordinary timestep/component refactoring against the actual incoming source controls."""
import json
import re
from pathlib import Path
from check_stress_history_consolidation import match
from compare_bounded_profile import load, ROOT, RUNS
from run_code_readability import CASES, CANDIDATES

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/code_readability'


def main():
    target = OUT/'equivalence.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite completed evidence.')
    rows = []
    for case, reference in CASES.items():
        if case in ('midpoint', 'restart_new', 'source'):
            continue
        candidate = CANDIDATES[case]
        before = load(reference)[1]
        after = load(candidate)[1]
        assert [float(x['time']) for x in before] == [float(x['time']) for x in after], case
        result = match(reference,candidate)
        assert result['before_iterations'] == result['after_iterations'], (case,'linear iterations')
        logs = [(RUNS/name/'run.log').read_text() for name in (reference,candidate)]
        for pattern in [r'Material projected reaction: outer=(\d+)',
                        r'Fault linear solve: iterations=(\d+)']:
            assert re.findall(pattern,logs[0]) == re.findall(pattern,logs[1]), (case,pattern)
        # Do not require textual equality for floating-point line-search alpha.
        rejected = [re.findall(r'after (\d+) rejected candidates',text) for text in logs]
        assert rejected[0] == rejected[1], (case,'line-search decisions')
        alphas = [[float(x.rstrip('.')) for x in re.findall(r'candidates; alpha=([0-9.eE+\-]+)',text)] for text in logs]
        assert len(alphas[0]) == len(alphas[1]), case
        result['alpha_max_absolute_difference'] = max((abs(a-b) for a,b in zip(*alphas)),default=0.)
        assert result['alpha_max_absolute_difference'] < 5e-13, (case,'line-search alpha')
        result['rejected_candidates'] = rejected[1]
        for step in result['steps']:
            assert step['tensor_L2'] < 5e-12 and step['work_traction_Linf'] < 5e-12, (case,step)
        rows.append(result)
    target.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_bulk=max(s['bulk_Linf'] for r in rows for s in r['steps']),
                          max_surface=max(s['surface_Linf'] for r in rows for s in r['steps']),
                          max_tensor=max(s['tensor_L2'] for r in rows for s in r['steps']),
                          max_work_traction=max(s['work_traction_Linf'] for r in rows for s in r['steps']))))


if __name__ == '__main__':
    main()
