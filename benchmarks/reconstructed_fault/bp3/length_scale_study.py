"""Bounded modified-BP3 Dc/ell study. Old fixtures and results are read-only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from run_mechanical_modes import parameters
from run_mechanical_width import VirtualProfile, NORMAL, G
from prepare_300km_fixture import coordinate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'length-scale-study'
OLD = HERE/'fixtures/modified_bp3_long_run_300km'
LIB = ROOT/'benchmarks/reconstructed_fault/performance/build-gmg/libfault_mechanical_modes.release.so'
BIN = ROOT/'build-pf-cpdi/aspect-release'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(values, prefix=()):
    lines, children = [], []
    indent = '  '*len(prefix)
    for path, value in values.items():
        if path[:len(prefix)] != prefix:
            continue
        if len(path) == len(prefix)+1:
            lines.append(f'{indent}set {path[-1]} = {value}\n')
        elif path[len(prefix)] not in children:
            children.append(path[len(prefix)])
    for name in children:
        lines.extend([f'\n{indent}subsection {name}\n', render(values, prefix+(name,)), f'{indent}end\n'])
    return ''.join(lines)


def fixture(ell, h):
    out = HERE/f'fixtures/modified_bp3_dc024_ell{ell}'
    assert (out/'target_cells.txt').exists()
    assert not (out/'manifest.json').exists(), 'Preserve prepared fixtures'
    # These coefficients define the already accepted physical background
    # function, not the new material normalization. Never replace its stored
    # rational denominator by current I_h or recalibrate from an accepted V.
    for name in ('fault.txt', 'prestress.txt'):
        shutil.copy2(OLD/name, out/name)
    profile = VirtualProfile(ell, h)
    nodes = np.loadtxt(out/'fault.txt')[:, :2]
    rows, max_error = [], 0.
    for j, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        for q, z in enumerate((leggauss(3)[0]+1)/2):
            p = (1-z)*a+z*b
            e = profile.extent
            intervals = [(-e, min(-p[1]/NORMAL[1], e)),
                         (max((100000-p[1])/NORMAL[1], -e), e)]
            missing = sum(profile.integrate(p, lo, hi) for lo, hi in intervals)
            if missing:
                check = sum(profile.integrate(p, lo, hi, 16) for lo, hi in intervals)
                max_error = max(max_error, abs(check-missing))
            rows.append([3*j+q, *p, missing])
    assert max_error < 1e-6
    with (out/'completion.txt').open('x') as stream:
        stream.write(str(len(rows))+'\n')
        np.savetxt(stream, rows, fmt=['%d', '%.17g', '%.17g', '%.17g'])
    values = parameters((HERE/'bp3_modified_long_run.prm').read_text())
    values['Material model', 'Phase field fault', 'Characteristic slip distance'] = '0.024'
    values['Phase field model', 'Length scale'] = str(ell)
    values['Solver parameters', 'Stokes solver parameters', 'Stokes solver type'] = 'block AMG'
    values['Fault reconstruction', 'Prescribed faults file'] = str(out/'fault.txt')
    values['Mesh refinement', 'BP3 saved mesh', 'Target cells file'] = str(out/'target_cells.txt')
    level = max(len(s.split(':')[1]) for s in (out/'target_cells.txt').read_text().split())
    values['Mesh refinement', 'Initial adaptive refinement'] = str(level-1)
    values['Postprocess', 'BP3', 'Mature prestress file'] = str(out/'prestress.txt')
    values['Postprocess', 'BP3', 'Bottom normalization completion file'] = str(out/'completion.txt')
    values['Postprocess', 'BP3', 'Last accepted step'] = '8'
    values['Postprocess', 'BP3', 'Graceful wall seconds'] = '3000'
    values['Output directory',] = str(OUT/f'qualification-ell{ell}')
    prm = HERE/f'bp3_dc024_ell{ell}.prm'
    assert not prm.exists()
    prm.write_text('# Modified BP3 research candidate, not official BP3. No automatic long run.\n'+render(values))
    files = [out/name for name in ('target_cells.txt', 'fault.txt', 'prestress.txt', 'completion.txt')]+[prm]
    record = dict(Dc=.024, ell=ell, finest_h=h, profile_radius=profile.r[-1],
                  completion_order_error=max_error, completion_profiles=int(sum(row[3]>0 for row in rows)),
                  physical_prestress_unchanged=digest(out/'prestress.txt') == digest(OLD/'prestress.txt'),
                  hashes={str(p): digest(p) for p in files})
    (out/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


def prepare(label, reference=False, qualification=False, ranks=4):
    if qualification:
        result = json.loads((OUT/'comparison.json').read_text())
        assert result['profile_gate'] and result['mechanical_gate'], \
            'Do not launch evolution after a failed profile or mechanical prerequisite'
    out = OUT/label
    out.mkdir(parents=True)
    values = parameters((HERE/'bp3_dc024_ell50.prm').read_text())
    values['Output directory',] = str(out)
    values['Additional shared libraries',] = str(HERE/'build/libbp3.release.so')+', '+str(LIB)
    values['Resume computation',] = 'false'
    values['Postprocess', 'BP3', 'Profile time interval'] = '1'
    values['Postprocess', 'BP3', 'Audit full state every step'] = 'false'
    values['Checkpointing', 'Time between checkpoint'] = '0'
    values['Checkpointing', 'Steps between checkpoint'] = '6' if qualification else '0'
    values['Postprocess', 'BP3', 'Graceful wall seconds'] = '2700'
    if reference:
        values['Mesh refinement', 'BP3 saved mesh', 'Target cells file'] = str(OUT/'target_cells_reference.txt')
        values['Mesh refinement', 'Initial adaptive refinement'] = '12'
    (out/'run.prm').write_text(render(values))
    env = dict(ASPECT_SOURCE_DIR=str(ROOT), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_MECHANICAL_WIDTH_PROBE='1',
               ASPECT_MECHANICAL_PROBE_STEP='0', ASPECT_MECHANICAL_PROBE_NEWTON='0',
               ASPECT_MECHANICAL_DECOMPOSITION='1', ASPECT_BP3_LENGTH_STUDY='1')
    if qualification:
        env['ASPECT_BP3_LENGTH_QUALIFICATION'] = '1'
    paths = [Path(__file__), out/'run.prm', BIN, LIB, HERE/'build/libbp3.release.so']
    for section, key in [(('Fault reconstruction',), 'Prescribed faults file'),
                         (('Mesh refinement', 'BP3 saved mesh'), 'Target cells file'),
                         (('Postprocess', 'BP3'), 'Mature prestress file'),
                         (('Postprocess', 'BP3'), 'Bottom normalization completion file')]:
        paths.append(Path(values[section+(key,)]))
    record = dict(command=['mpirun', '-np', str(ranks), '--bind-to', 'core', '--map-by', 'core', str(BIN), str(out/'run.prm')],
                  environment=env, cap_seconds=3000 if qualification else 1200,
                  qualification=qualification, hashes={str(p): digest(p) for p in paths})
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared', out)


def run(label):
    out = OUT/label
    record = json.loads((out/'launch.json').read_text())
    used = sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('*/execution.json'))
    remaining = 7200-used
    assert remaining > 0, 'Aggregate two-hour simulation budget reached'
    cap = min(record['cap_seconds'], int(remaining))
    for name, value in record['hashes'].items():
        assert digest(Path(name)) == value, name
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    with (out/'run.log').open('x') as log:
        result = subprocess.run(['timeout', '--kill-after=20', str(cap)]+record['command'],
                                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (out/'run.log').read_text()
    if record['qualification']:
        accepted = out/'accepted_steps.csv'
        rows = list(csv.DictReader(accepted.open())) if accepted.exists() else []
        passed = (result.returncode == 0 and len(rows) == 9
                  and [int(r['step']) for r in rows] == list(range(9))
                  and all(int(r['fresh_linear_checks_passed']) == 1 for r in rows))
    else:
        passed = 'MECHANICAL MODES VERIFIED' in text
    execution = dict(seconds=time.monotonic()-start, status=result.returncode, expected_completion=passed,
                     peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                     previous_simulation_seconds=used, cap_seconds=cap)
    (out/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    print(json.dumps(execution, indent=2))
    assert passed, 'Preserved failure: no automatic retry or tolerance change'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fixture', 'prepare', 'run'))
    parser.add_argument('--label', default='probe-candidate')
    parser.add_argument('--ell', type=int, choices=(25, 50), default=50)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--ranks', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'fixture':
        fixture(args.ell, 12.20703125 if args.ell == 50 else 6.103515625)
    elif args.action == 'prepare':
        prepare(args.label, args.reference, args.qualification, args.ranks)
    else:
        run(args.label)
