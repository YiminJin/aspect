#!/usr/bin/env python3
"""Replay the bounded reviewed controls; fresh outputs, sequential MPI, at most four ranks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence'
OUT = EVIDENCE/'stress_consolidation_guide'
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
REFERENCES = {k: 'readability-'+k for k in ('dc', 'sc', 'dl', 'sl', 'default', 'reject', 'midpoint', 'source')}
REFERENCES.update(restart4='readability-restart4-control', restart_new='readability-restart-new',
                  source_layered='readability-source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', choices=list(REFERENCES))
    args = parser.parse_args()
    for case in args.cases:
        reference = REFERENCES[case]
        metadata = json.loads((EVIDENCE/'code_readability/runs'/reference/'command.json').read_text())['arguments']
        if case == 'restart_new':
            metadata['restart'] = str(RUNS/'consolidation-midpoint')
        if case == 'source_layered':
            metadata['eta'] = '1+.5*tanh(4*x)'
        label = 'consolidation-'+case
        assert not (RUNS/label).exists(), 'Preserve existing run; choose a new label.'
        assert 1 <= metadata['ranks'] <= 4
        command = [sys.executable, str(HERE/'run_material_native.py'), label]
        for key, value in metadata.items():
            if key in ('label', 'requested_end') or value is None or value == 'None':
                continue
            flag = '--'+key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        print('Running', label, 'against', reference, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        (OUT/(label+'-launch.json')).write_text(json.dumps(dict(reference=reference,
            command=command, exit_code=result.returncode), indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
