#!/usr/bin/env python3
"""Compare native PF6a lifecycle exports without reimplementing the integrator.

The native observer separately loads/resaves complete manager archives, since
native load materializes the omitted default shear-sense value. This script
writes those pairs for the native archive check; it does not canonicalize them.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def pair(first, second, raw_manager=True, historical=False):
    result = {}
    aa, bb = [{p.name for p in directory.iterdir()} for directory in [first, second]]
    assert aa <= bb if historical else aa == bb
    if historical: result['new_exports_not_present_in_primary'] = sorted(bb-aa)
    for a in sorted(first.iterdir()):
        b = second/a.name
        exact = a.read_bytes() == b.read_bytes()
        result[a.name] = exact
        if a.name == 'manager.bin' and not raw_manager:
            result['raw_manager_size_difference'] = b.stat().st_size-a.stat().st_size
            continue
        assert exact, (a, b)
    return result


def states(first, second, raw_manager=True, historical=False):
    found = sorted(second.glob('state-*'), key=lambda p: int(p.name[6:]))
    assert found
    return {p.name: pair(first/p.name, p, raw_manager, historical) for p in found}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('build-particle-free/runs'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive-pairs', type=Path, required=True)
    a = p.parse_args()
    def run(name):
        path = a.runs/name
        assert json.loads((path/'record.json').read_text())['gate_passed']
        return path/'run'
    cut, matched = map(run, ['pf6a-outer-cutback', 'pf6a-outer-matched'])
    attempts = np.atleast_1d(np.genfromtxt(cut/'cutback.csv', delimiter=',', names=True))
    assert len(attempts) == 1
    row = attempts[0]
    assert tuple(row) == (16, .5, .03125, .484375, .015625, 15)
    result = {'cutback': states(matched, cut), 'cutback_attempts': [dict(zip(attempts.dtype.names, map(float, row)))]}
    coarse = run('pf6a-accepted-final-coarse')
    result['final_source_vs_primary_coarse'] = states(run('pf6a-final-32-1'), coarse, historical=True)
    parent = run('pf6a-accepted-final-parent')
    reference = run('pf6a-accepted-final-uninterrupted')
    resumed = run('pf6a-accepted-final-resumed')
    result['final_source_vs_primary_dt2'] = states(run('pf6a-final-32-2'), reference, historical=True)
    result['branch_prefix'] = states(reference, parent)
    result['restart'] = states(reference, resumed, False)
    result['terminal_error_path_coarse'] = states(coarse, run('pf6a-terminal-final-coarse'))
    result['terminal_error_path_cutback'] = states(matched, run('pf6a-terminal-final-cutback'))
    terminal_resumed = run('pf6a-terminal-final-resumed')
    result['terminal_error_path_restart'] = states(reference, terminal_resumed, False)
    # Both restart observers must retain identical raw manager representations.
    result['terminal_error_path_restart_vs_previous'] = states(resumed, terminal_resumed)
    archives = []
    for state in result['restart']:
        archives.append(f'{(reference/state/"manager.bin").resolve()} {(resumed/state/"manager.bin").resolve()}')
    for state in result['terminal_error_path_restart']:
        archives.append(f'{(reference/state/"manager.bin").resolve()} {(terminal_resumed/state/"manager.bin").resolve()}')
    a.archive_pairs.write_text('\n'.join(archives)+'\n')
    result['manager_archive_check'] = 'Native load/resave required; pairs written to '+str(a.archive_pairs)
    result['passed_except_pending_native_archive_check'] = True
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'comparisons': {k: len(v) for k, v in result.items() if isinstance(v, dict)},
                      'archive_pairs': len(archives)}))


if __name__ == '__main__':
    main()
