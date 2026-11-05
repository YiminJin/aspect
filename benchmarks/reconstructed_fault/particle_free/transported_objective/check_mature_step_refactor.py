#!/usr/bin/env python3
"""Compare the mature driver against preserved consolidation controls."""
import json
from pathlib import Path
from check_stress_history_consolidation import match
from compare_bounded_profile import load, ROOT
from run_mature_step_refactor import CASES

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/mature_step_refactor'


def main():
    target = OUT/'equivalence.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite completed evidence.')
    rows = []
    for case, reference in CASES.items():
        candidate = 'mature-step-'+case
        before = load(reference)[1]
        after = load(candidate)[1]
        assert [float(x['time']) for x in before] == [float(x['time']) for x in after], case
        result = match(reference,candidate)
        assert result['before_iterations'] == result['after_iterations'], (case,'linear iterations')
        rows.append(result)
    target.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_bulk=max(s['bulk_Linf'] for r in rows for s in r['steps']),
                          max_surface=max(s['surface_Linf'] for r in rows for s in r['steps']),
                          max_tensor=max(s['tensor_L2'] for r in rows for s in r['steps']),
                          max_work_traction=max(s['work_traction_Linf'] for r in rows for s in r['steps']))))


if __name__ == '__main__':
    main()
