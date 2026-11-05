#!/usr/bin/env python3
"""Replay bounded refactoring controls sequentially, never overwrite evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name('run_material_native.py')
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/mature_step_refactor'
CASES = {key: 'consolidate-after-'+key for key in ('dc','sc','dl','sl')}
CASES.update(default='consolidate-default', reject='consolidate-reject', restart4='consolidate-restart4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', choices=list(CASES))
    args = parser.parse_args()
    for case in args.cases:
        reference = CASES[case]
        metadata = json.loads((RUNS/reference/'command.json').read_text())['arguments']
        assert 1 <= metadata['ranks'] <= 4
        label = 'mature-step-'+case
        assert not (RUNS/label).exists(), f'Preserve existing {label}; choose another output label explicitly.'
        command = [sys.executable, str(RUNNER), label]
        for key, value in metadata.items():
            if key in ('label', 'requested_end') or value is None or value == 'None':
                continue
            flag = '--'+key.replace('_','-')
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        print('Running',label,'against',reference,flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        record = dict(reference=reference, candidate=label, command=command, exit_code=completed.returncode)
        (OUT/(case+'-launch.json')).write_text(json.dumps(record,indent=2)+'\n')
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
