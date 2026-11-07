#!/usr/bin/env python3
"""Replay failed fine-mesh initialization with a read-only vertex diagnostic."""
import json
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT/'build-particle-free/transported-objective/stage-c'
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--distort', action='store_true', help='Verify that the corrected guard still rejects a sheared cell')
    args=parser.parse_args()
    source = STAGE/'material-runs/layered-bulk-plus-fine'
    target = STAGE/('material-runs/mesh-audit-distorted' if args.distort else 'material-runs/mesh-audit-nondyadic')
    target.mkdir(exist_ok=False)
    text = (source/'input.prm').read_text().replace(str(source), str(target))
    library = STAGE/'layered-bulk/layered-provider/liblayered_mesh_audit.release.so'
    text = text.replace('set Additional shared libraries = ',
                        f'set Additional shared libraries = {library}, ')
    (target/'input.prm').write_text(text)
    (target/'fault.txt').write_bytes((source/'fault.txt').read_bytes())
    candidate='layered-bulk-cartesian' if args.distort else 'layered-bulk'
    command = ['/opt/openmpi/5.0.6/bin/mpirun', '-n', '4',
               str(STAGE/candidate/'aspect-release'), str(target/'input.prm')]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/openmpi/5.0.6/lib'
    env['ASPECT_TRUE_STRESS_ENERGY'] = '1'
    if args.distort:env['ASPECT_CARTESIAN_AUDIT_DISTORT']='1'
    with (target/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120)
    lines = [line for line in (target/'run.log').read_text().splitlines()
             if 'Cartesian audit rank=' in line]
    record = dict(command=command, solver_exit_code=result.returncode, distorted=args.distort, lines=lines)
    (OUT/('mesh-audit-distorted.json' if args.distort else 'mesh-audit.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    assert len(lines)==4 and result.returncode==1
    assert 'Mature DG source requires affine Cartesian cells' in (target/'run.log').read_text()


if __name__ == '__main__':
    main()
