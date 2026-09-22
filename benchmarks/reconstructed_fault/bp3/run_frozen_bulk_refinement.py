"""Prepare/run the single frozen bulk-refinement comparison; never a loading prefix."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE/'first_long_run/mechanical-modes-preflight-fixed'
OUT = HERE/'first_long_run/mechanical-bulk-refinement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'coarse', 'fine'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--decompose', action='store_true', help='Repeat only existing probes, exporting signed gradients and Q2 velocities')
    args = parser.parse_args()
    assert args.stage != 'coarse' or args.decompose
    assert not args.decompose or args.stage != 'capture'
    out = (HERE/'first_long_run/mechanical-velocity-decomposition' if args.decompose else OUT)/args.stage
    if args.decompose and not args.execute:
        source = OUT/('capture' if args.stage == 'coarse' else 'fine')
        out.mkdir(parents=True, exist_ok=True)
        assert not (out/'run.prm').exists(), 'Preserve existing evidence'
        text = re.sub(r'^set Output directory =.*$', 'set Output directory = '+str(out),
                      (source/'run.prm').read_text(), flags=re.MULTILINE)
        prm = out/'run.prm';prm.write_text(text)
        record = json.loads((source/'launch.json').read_text())
        record['command'][-1] = str(prm)
        record['environment'].pop('ASPECT_MECHANICAL_EXPORT_PROFILE', None)
        record['environment']['ASPECT_MECHANICAL_DECOMPOSITION'] = '1'
        files = [Path(path) for path in record['hashes'] if path != str(source/'run.prm')]+[prm]
        record['hashes'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
        print('Prepared unchanged frozen response with velocity export:', out);return
    if not args.execute:
        out.mkdir(parents=True, exist_ok=True)
        assert not (out/'run.prm').exists(), 'Preserve existing inputs/results'
        text = (BASE/'fresh.prm').read_text().replace(str(BASE), str(out))
        # Use the ordinary production controllers. These runs stop within t=0;
        # neither a saved-clock controller nor a real timestep is invoked.
        text = text.replace('convection time step, reconstructed fault time step, mechanical probe clock',
                            'convection time step, reconstructed fault time step')
        launch = json.loads((BASE/'mechanical_launch.json').read_text())
        env = launch['environment']
        env.pop('ASPECT_BP3_TIMESTEP_SEQUENCE', None)
        if args.stage == 'capture':
            env['ASPECT_MECHANICAL_EXPORT_PROFILE'] = '1'
        else:
            capture = OUT/'capture'
            assert 'FROZEN PROFILE SNAPSHOT VERIFIED' in (capture/'run.log').read_text()
            assert (out/'target_cells.txt').exists(), 'Generate the graded fine tree first'
            rows = []
            for path in sorted(capture.glob('phase_cells_rank*.csv')):
                with path.open() as stream:
                    rows.extend(csv.DictReader(stream))
            assert len({row['cell'] for row in rows}) == len(rows)
            with (capture/'phase_cells.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(sorted(rows, key=lambda row: row['cell']))
            text = text.replace('set Initial adaptive refinement = 8', 'set Initial adaptive refinement = 9')
            text = text.replace(str(HERE/'fixtures/modified_bp3_long_run_300km/target_cells.txt'), str(out/'target_cells.txt'))
            env['ASPECT_MECHANICAL_FROZEN_PROFILE'] = str(capture)
        prm = out/'run.prm';prm.write_text(text)
        inputs = [prm, ROOT/'build-pf-cpdi/aspect-release', HERE/'build/libbp3.release.so',
                  ROOT/'benchmarks/reconstructed_fault/performance/build-gmg/libfault_mechanical_modes.release.so']
        if args.stage == 'fine': inputs += [out/'target_cells.txt', OUT/'capture/phase_cells.csv', OUT/'capture/surface.csv']
        record = dict(command=['mpirun', '-np', '4', '--bind-to', 'core', '--map-by', 'core', str(inputs[1]), str(prm)],
                      environment=env, hard_wall_cap_s=600,
                      hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
        (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
        print('Prepared', out);return
    record = json.loads((out/'launch.json').read_text())
    assert not (out/'run.log').exists(), 'No automatic retry or evidence overwrite'
    for path, digest in record['hashes'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    with (out/'run.log').open('w') as log:
        process = subprocess.run(['timeout', str(record['hard_wall_cap_s'])]+record['command'],
                                 env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    marker = 'FROZEN PROFILE SNAPSHOT VERIFIED' if args.stage == 'capture' else 'MECHANICAL MODES VERIFIED'
    verified = marker in (out/'run.log').read_text()
    result = dict(seconds=time.monotonic()-start, returncode=process.returncode,
                  expected_noncommitting_stop=verified, peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2));assert verified, 'Preserved failure; no automatic retry'


if __name__ == '__main__': main()
