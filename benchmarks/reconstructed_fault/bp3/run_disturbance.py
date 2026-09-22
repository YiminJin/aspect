"""Matched short disturbance branches of the accepted local step-11 checkpoint.

Only pending time/dt in a disposable checkpoint are retimed, using the
previously qualified archive layout. Every history/mesh byte is preserved.
No loading prefix and no retry/overwrite of failed evidence.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import time
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SAVED = HERE/'first_long_run/mechanical-discrimination-roundoff-clock'
ROOT = HERE/'first_long_run/state-disturbance'
PLUGIN = REPO/'benchmarks/reconstructed_fault/performance/build-gmg/libfault_disturbance.release.so'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    assert math.isfinite(args.wavelength) and args.wavelength > 0
    out = ROOT/args.name
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = SAVED/'restart/01'
    step, start = (checkpoint/'bp3_accepted_state.txt').read_text().split()
    start = float(start)
    assert int(step) == 11
    dt = 31557600.*args.years/args.steps
    plugin = PLUGIN if args.control=='full' else PLUGIN.with_name('libfault_disturbance_controls.release.so')
    shutil.copytree(checkpoint, out/'restart/01')
    (out/'restart/last_good_checkpoint.txt').write_text('1\n')
    # Checkpoint follows advance_time; reset only the pending endpoint and dt.
    data = (checkpoint/'resume.z').read_bytes()
    raw = zlib.decompress(data[16:])
    header = struct.unpack('<4I', data[:16])
    assert header == (1, len(raw), len(raw), len(data)-16)
    clock = struct.unpack_from('<dddI', raw, 237)
    assert clock[3] == 12 and abs(clock[0]-clock[1]-start) < 1e-6
    changed = raw[:237]+struct.pack('<dd', start+dt, dt)+raw[253:]
    assert changed[:237]+struct.pack('<dd', *clock[:2])+changed[253:] == raw
    compressed = zlib.compress(changed, 9)
    (out/'restart/01/resume.z').write_bytes(struct.pack('<4I', 1, len(changed), len(changed), len(compressed))+compressed)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name != 'resume.z':
            assert sha(path) == sha(out/'restart/01'/path.name)
    for path in (checkpoint/'bp3_output_metadata').iterdir():
        shutil.copy2(path, out/path.name)
    text = (SAVED/'fresh.prm').read_text().replace('mechanical probe clock', 'disturbance clock')
    text += f'''
set Additional shared libraries = {plugin}
set Output directory = {out}
set Resume computation = true
set End time = {start+(args.steps+2)*dt:.17g}
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, disturbance clock
end
subsection Postprocess
  subsection BP3
    set Last accepted step = {11+args.steps}
    set Graceful wall seconds = 7000
    set Profile time interval = 1e100
    set Audit full state every step = false
  end
end
subsection Checkpointing
  set Steps between checkpoint = 0
  set Time between checkpoint = 0
end
'''
    if args.control=='normal':
        text+='''subsection Adiabatic conditions model
  set Model name = disturbance normal reference
end
'''
    (out/'run.prm').write_text(text)
    env = json.loads((SAVED/'mechanical_launch.json').read_text())['environment']
    env.pop('ASPECT_BP3_TIMESTEP_SEQUENCE', None)
    env.update(ASPECT_DISTURBANCE_EPS=str(args.epsilon), ASPECT_DISTURBANCE_DT=str(dt),
               ASPECT_DISTURBANCE_RATIO_LIMIT=str(args.limit),
               ASPECT_DISTURBANCE_WAVELENGTH=str(args.wavelength))
    if args.control!='full':
        env.update(ASPECT_DISTURBANCE_CONTROL=args.control,
                   ASPECT_DISTURBANCE_REFERENCE=str(ROOT/args.reference))
    record = dict(start_s=start, final_s=start+args.steps*dt, steps=args.steps, dt=dt,
                  epsilon=args.epsilon, wavelength_m=args.wavelength, ratio_limit=args.limit, environment=env, control=args.control,
                  command=['mpirun', '-np', '4', '--bind-to', 'core', '--map-by', 'core',
                           str(REPO/'build-pf-cpdi/aspect-release'), str(out/'run.prm')],
                  checkpoint_source=str(checkpoint), old_clock=clock,
                  unchanged_outside_pending_time_dt=True,
                  source_hashes={str(p): sha(p) for p in (checkpoint/'resume.z', plugin, out/'run.prm',
                                                        REPO/'build-pf-cpdi/aspect-release')})
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


def execute(args):
    out = ROOT/args.name
    record = json.loads((out/'launch.json').read_text())
    if record.get('control','full')!='full':
        ref=Path(record['environment']['ASPECT_DISTURBANCE_REFERENCE'])
        with (ref/'accepted_steps.csv').open() as stream:
            accepted={int(r['step']):r for r in csv.DictReader(stream)}
        for step in range(12,12+record['steps']):
            r=accepted[step]
            assert float(r['dt'])==record['dt'] and r['fresh_linear_checks_passed']=='1'
            assert float(r['Theta_relative_error'])<1e-12
    for path, digest in record['source_hashes'].items():
        assert sha(Path(path)) == digest, path
    assert not (out/'run.log').exists(), 'No automatic retry/overwrite'
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    command=list(record['command'])
    if args.cores:
        assert len(args.cores.split(','))==4
        command[command.index('--map-by')+1]='pe-list='+args.cores+':ordered'
    with (out/'run.log').open('w') as log:
        result = subprocess.run(['timeout', '7200']+command, cwd=HERE, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    execution = dict(seconds=time.monotonic()-start, returncode=result.returncode, command=command,
                     peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    print(json.dumps(execution, indent=2), flush=True)
    assert result.returncode == 0, 'Failure preserved; no automatic retry'
    with (out/'accepted_steps.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['step']) > 11]
    assert len(rows) == record['steps'] and abs(float(rows[-1]['time'])-record['final_s']) < 1e-5
    assert all(float(r['max_step_slip_over_Dc']) <= record['ratio_limit'] for r in rows)
    assert all(float(r['Theta_relative_error']) < 1e-12 and r['fresh_linear_checks_passed'] == '1' for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('name')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--years', type=float, default=1.)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--wavelength', type=float, default=200., help='Disturbance wavelength in metres; taper is unchanged')
    parser.add_argument('--limit', type=float, default=.25)
    parser.add_argument('--control', choices=['full','state','normal'],default='full')
    parser.add_argument('--reference',default='reference32')
    parser.add_argument('--cores',help='Four disjoint logical core IDs; execution placement only')
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else execute)(args)
