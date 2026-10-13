#!/usr/bin/env python3
"""One sequential, bound-core native endpoint MPI run with preserved provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent / 'native_mpi_evidence'
BUILD = ROOT / 'build-propagation-integration'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    out = BUILD / 'runs' / ('output-native-mpi-' + args.tag)
    prm = E / (args.tag + '.prm')
    assert not out.exists() and not prm.exists(), 'Preserve prior accepted/failed runs'
    text = args.input.read_text()
    # Last parameter assignment wins; retain normal refinement and 64 segments.
    text += f'\nset Output directory = {out}\n'
    # The reference uses one by 32 repetitions followed by four global refinements.
    # For 8/4 tangential cells, begin with 1x64/1x128 and use 3/2 refinements.
    level = {16: 4, 8: 3, 4: 2}[args.tangential]
    text += ('subsection Geometry model\n subsection Box\n'
             f'  set X repetitions = 1\n  set Y repetitions = {512 // (2**level)}\n'
             ' end\nend\nsubsection Mesh refinement\n'
             f' set Initial global refinement = {level}\nend\n')
    # Preserve the original physical particle lattice while changing only FE cells.
    text += ('subsection Particles\n subsection Generator\n subsection Reference cell\n'
             f'  set Number of particles per cell per direction = {48 // args.tangential}, 3\n'
             ' end\nend\nend\n')
    if args.end_time is not None:
        text += f'set End time = {args.end_time:.17g}\n'
    if args.failure_step is not None:
        text += ('subsection Postprocess\n subsection P2 native endpoint\n'
                 '  set Force complete step failure and retry = true\n'
                 f'  set Complete failure timestep = {args.failure_step}\n end\nend\n')
    prm.write_text(text)
    executable = BUILD / 'aspect-release'
    plugin = BUILD / 'plugin/libnative_endpoint_verification.release.so'
    command = ['mpirun', '--map-by', 'core', '--bind-to', 'core', '--rank-by', 'slot',
               '--report-bindings', '-np', str(args.ranks), str(executable), str(prm)]
    env = dict(os.environ, ASPECT_SOURCE_DIR=str(ROOT),
               ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.profile:
        profiler = BUILD / 'libnative_mpi_profile.so'
        assert profiler.exists(), 'Build the benchmark-only PMPI profiler first'
        env['LD_PRELOAD'] = str(profiler)
        env['ASPECT_BENCHMARK_MPI_PROFILE'] = str(E / args.tag)
    record = dict(command=command, output=str(out), input=str(prm),
                  source_input=str(args.input), ranks=args.ranks, threads_per_rank=1,
                  tangential_cells=args.tangential, normal_cells=512,
                  surface_segments=64, wall_cap_seconds=args.wall_cap,
                  executable_sha256=sha(executable), plugin_sha256=sha(plugin),
                  input_sha256=sha(prm), physical_history='fresh driver; retained fixed-entry diagnostics when selected; no checkpoint conversion')
    if args.profile:
        record['mpi_profiler_sha256'] = sha(profiler)
    (E / (args.tag + '-start.json')).write_text(json.dumps(record, indent=2)+'\n')
    started = time.perf_counter()
    done = threading.Event()
    with (E / (args.tag + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)

        def watch():
            if not done.wait(args.wall_cap):
                record['wall_cap_reached'] = True
                os.killpg(child.pid, signal.SIGINT)
                if not done.wait(30):
                    os.killpg(child.pid, signal.SIGTERM)

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
        try:
            status = child.wait()
        except BaseException:
            # The MPI launcher starts a new session. Interrupting this Python
            # driver alone must not leave its benchmark running unattended.
            record['interrupted'] = True
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            record.update(returncode=child.returncode,
                          wall_seconds=time.perf_counter()-started)
            (E / (args.tag + '-run.json')).write_text(json.dumps(record, indent=2)+'\n')
            raise
        finally:
            done.set()
            monitor.join()
    record.update(returncode=status, wall_seconds=time.perf_counter()-started)
    (E / (args.tag + '-run.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--ranks', type=int, choices=[1, 2, 4])
    parser.add_argument('--task3-suite', action='store_true',
                        help='Sequential original-mesh operators, growth/recovery and saddle at 1/2/4 ranks; stop on failure')
    parser.add_argument('--tangential', type=int, choices=[16, 8, 4], default=16)
    parser.add_argument('--wall-cap', type=float, default=3600)
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--failure-step', type=int)
    parser.add_argument('--profile', action='store_true',
                        help='Time PMPI calls; changes no solver settings')
    args = parser.parse_args()
    assert args.wall_cap > 0
    if args.task3_suite:
        assert not any([args.input,args.tag,args.ranks,args.end_time,args.failure_step])
        executable_sha = sha(BUILD/'aspect-release')
        plugin_sha = sha(BUILD/'plugin/libnative_endpoint_verification.release.so')
        for kind in ['operators','core','saddle']:
            for ranks in [1,2,4]:
                assert sha(BUILD/'aspect-release') == executable_sha
                assert sha(BUILD/'plugin/libnative_endpoint_verification.release.so') == plugin_sha
                case = argparse.Namespace(**vars(args))
                case.input = (E.parent/'native_endpoint_evidence/operators-solver-summation.prm'
                              if kind == 'operators' else E/(kind+'.prm'))
                case.tag = f'{kind}-final-r{ranks}'
                case.ranks = ranks
                case.tangential = 16
                case.wall_cap = 1800 if kind == 'operators' else 3600
                case.profile = True
                case.failure_step = 1 if kind == 'core' and ranks == 4 else None
                print('Starting sequential task-3 control:',case.tag,flush=True)
                status = run(case)
                if status:
                    raise SystemExit(status)
        raise SystemExit(0)
    assert args.input and args.tag and args.ranks, 'Individual run requires input, tag and ranks'
    raise SystemExit(run(args))
