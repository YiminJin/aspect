#!/usr/bin/env python3
"""Prepare fresh axial refinement and same-mesh native schedule restart.

No accepted field is transferred from the 96x128 comparison. The split at
accepted step95 changes only the pending interval through the existing native
post_resume_time_step hook. It does not change saved clocks or histories.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-particle-free/p5c-axial'
BASE = ROOT / 'build-particle-free/p5c-native'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh():
    assert not WORK.exists(), 'Preserve an existing task directory'
    WORK.mkdir()
    previous = json.loads((ROOT / 'build-particle-free/p5b-native/entry.json').read_text())
    protected = {name: sha(ROOT / name) for name in previous['protected']}
    assert protected == previous['protected']
    paths = [BASE / 'aspect-release', BASE / 'plugin/libbulk_p5c_checks.release.so',
             ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/p5c_native.zip']
    entry = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'protected': protected, 'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in paths},
             'schedule': [[.11875, .00125], [.145, .000625]], 'load_end_s': .140,
             'mesh': [192, 128], 'ranks': 4,
             'initialization': 'fresh continuum prescription, zero physical histories; no coarse transfer'}
    (WORK / 'entry.json').write_text(json.dumps(entry, indent=2) + '\n')
    for mode in ['fixed', 'radial']:
        prm = f'''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/bulk_cohesive/p5c_seed_{mode}.prm
set Output directory = $ASPECT_SOURCE_DIR/build-particle-free/p5c-axial/runs/{mode}-prefix
subsection Geometry model
 subsection Box
  set X repetitions = 192
 end
end
subsection Termination criteria
 set End step = 95
end
'''
        (WORK / f'{mode}-prefix.prm').write_text(prm)


def resume(mode):
    run = WORK / 'runs' / f'{mode}-prefix'
    found = []
    tag = f'bulk cohesive {mode} tensor cell history v1'.encode()
    for path in (run / 'restart').glob('[0-9][0-9]'):
        raw = zlib.decompress((path / 'resume.z').read_bytes()[16:])
        q, step, pub, time = struct.unpack_from('<IIQd', raw, raw.index(tag) + len(tag))
        if step == 95:
            found.append((path, q, pub, time))
    assert len(found) == 1, found
    source, q, pub, time = found[0]
    assert q == 8 and pub == 95 and abs(time - .11875) < 1e-12
    target = WORK / 'runs' / f'{mode}-active'
    assert not target.exists()
    (target / 'restart').mkdir(parents=True)
    shutil.copytree(source, target / 'restart' / source.name)
    (target / 'restart/last_good_checkpoint.txt').write_text(str(int(source.name)) + '\n')
    prm = f'''include $ASPECT_SOURCE_DIR/build-particle-free/p5c-axial/{mode}-prefix.prm
set Output directory = $ASPECT_SOURCE_DIR/build-particle-free/p5c-axial/runs/{mode}-active
set Resume computation = true
set Maximum time step = 0.000625
set Maximum first time step = 0.000625
subsection Termination criteria
 set End step = 137
end
'''
    (WORK / f'{mode}-active.prm').write_text(prm)
    record = {'mode': mode, 'q': q, 'step': 95, 'publications': pub, 'accepted_time_s': time,
              'saved_clock_changes': 'none', 'checkpoint': str(source.relative_to(ROOT)),
              'files': {p.name: sha(p) for p in source.iterdir() if p.is_file()}}
    (WORK / f'{mode}-active-input.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fresh', 'fixed', 'radial'])
    args = parser.parse_args()
    fresh() if args.stage == 'fresh' else resume(args.stage)
