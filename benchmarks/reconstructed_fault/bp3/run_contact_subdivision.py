"""Halve the last two matched intervals from the unchanged refined step-11 state.

ASPECT checkpoints after advance_time: only the pending time/dt in a disposable
archive are retimed. Mesh, solution vectors, histories, old_dt, step index and
all plugin serialization remain byte-identical. No production API is changed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zlib

here = Path(__file__).resolve().parent
repo = here.parents[2]
base = here / 'junction-matched-qualified-local4/refined'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retime(data, old_clock, new_time, new_dt):
    header = struct.unpack('<4I', data[:16])
    raw = zlib.decompress(data[16:])
    assert header == (1, len(raw), len(raw), len(data)-16)
    # serialize() begins with time, time_step, old_time_step, timestep_number.
    key = struct.pack('<dddI', *old_clock)
    assert raw.count(key) == 1, 'Unknown archive layout; do not guess.'
    offset = raw.index(key)
    assert offset == 237, 'Qualified BP3 archive layout changed.'
    changed = raw[:offset] + struct.pack('<dd', new_time, new_dt) + raw[offset+16:]
    assert changed[:offset] == raw[:offset] and changed[offset+16:] == raw[offset+16:]
    compressed = zlib.compress(changed, 9)
    return struct.pack('<4I', 1, len(changed), len(changed), len(compressed))+compressed, offset


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['prepare', 'run'])
parser.add_argument('--tag', default='')
args = parser.parse_args()
out = here / ('junction-contact-half'+('-'+args.tag if args.tag else '')+'-local4')
checkpoint = base / 'restart/03'
if args.mode == 'prepare':
    out.mkdir()
    with (base/'accepted_steps.csv').open() as f:
        accepted = list(csv.DictReader(f))
    assert len(accepted) == 14
    sequence = [[int(r['step']), float(r['time']), float(r['dt'])] for r in accepted[:12]]
    for row in accepted[12:]:
        end = float(row['time']); start = sequence[-1][1]
        for target in [start+(end-start)/2, end]:
            sequence.append([len(sequence), target, target-sequence[-1][1]])
    assert sequence[-1][0] == 15
    with (out/'sequence.csv').open('x') as f:
        writer = csv.writer(f); writer.writerow(['step', 'time', 'dt']); writer.writerows(sequence)
    hashes = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
    shutil.copytree(checkpoint, out/'restart/03')
    (out/'restart/last_good_checkpoint.txt').write_text('3\n')
    clock = (float(accepted[12]['time']), float(accepted[12]['dt']), float(accepted[11]['dt']), 12)
    new_data, offset = retime((checkpoint/'resume.z').read_bytes(), clock, sequence[12][1], sequence[12][2])
    (out/'restart/03/resume.z').write_bytes(new_data)
    # Independent round-trip: restoring the original two doubles reproduces
    # every uncompressed byte. All other checkpoint files are unchanged.
    original = zlib.decompress((checkpoint/'resume.z').read_bytes()[16:])
    altered = zlib.decompress(new_data[16:])
    assert altered[:offset]+struct.pack('<dd', *clock[:2])+altered[offset+16:] == original
    assert all(digest(out/'restart/03'/name)==value for name,value in hashes.items() if name!='resume.z')
    (out/'checkpoint_retime.json').write_text(json.dumps(dict(source=str(checkpoint), source_sha256=hashes,
        old_clock=clock, new_time=sequence[12][1], new_dt=sequence[12][2], offset=offset,
        uncompressed_bytes_outside_time_dt_identical=True, all_mesh_history_files_identical=True), indent=2)+'\n')
    (out/'run.prm').write_text(f'''# Fixed-mesh timestep subdivision only; stop at the previous final time.
include {base}/run.prm
set Output directory = {out}
set Resume computation = true
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, BP3 replay cap
end
subsection Termination criteria
  set Termination criteria = end time, end step
  set End step = 15
end
''')
    print(json.dumps(sequence[11:], indent=2))
else:
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(repo), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_FAULT_HISTORY_AUDIT='1',
        ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1', ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'sequence.csv'),
        ASPECT_BP3_TARGET_MESH=str(base/'target_cells.txt'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    cmd = ['mpirun', '-np', '4', '--bind-to', 'core', '--map-by', 'core',
           str(repo/'build-pf-cpdi/aspect-release'), str(out/'run.prm')]
    record = dict(command=cmd, environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):digest(p) for p in [repo/'build-pf-cpdi/aspect-release', here/'build/libbp3.release.so',
            here/'bp3.cc', here/'replay_time_step.h', out/'run.prm', out/'sequence.csv']})
    with (out/'provenance.json').open('x') as f: json.dump(record, f, indent=2)
    start = time.monotonic()
    with (out/'run.log').open('x') as log:
        result = subprocess.run(cmd, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)
    saved = json.loads((out/'checkpoint_retime.json').read_text())['source_sha256']
    record = dict(status=result.returncode, seconds=time.monotonic()-start,
        source_checkpoint_unchanged=saved=={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()})
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    assert result.returncode == 0 and record['source_checkpoint_unchanged'], 'Preserve failure; no automatic retry.'
