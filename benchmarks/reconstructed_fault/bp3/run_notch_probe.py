"""One frozen-history prescribed-node impulse; intentional rollback on success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SAVED = HERE / 'work-replay-50-local4'
CONTROL = HERE / 'within-step-50-local4/A'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('amplitude', type=float, choices=[.01, .005], nargs='?')
    parser.add_argument('--free-trace', action='store_true',
                        help='Release only the free-side 40-km trace; keep the deep trace at Vp.')
    parser.add_argument('--label', help='Fresh evidence directory; existing attempts are never reused.')
    parser.add_argument('--location', type=float, choices=[40000., 25000.], default=40000.)
    parser.add_argument('--clamp-others', action='store_true',
                        help='Hold other rates at their saved converged values to isolate the bulk-relaxed operator column.')
    args = parser.parse_args()
    assert (args.amplitude is None) == args.free_trace
    assert not args.free_trace or (args.location == 40000. and not args.clamp_others)
    assert json.loads((CONTROL / 'analysis.json').read_text())['baseline_reproduced']
    label = args.label or ('independent-local4' if args.free_trace else 'impulse-' + str(args.amplitude))
    assert re.fullmatch(r'[a-zA-Z0-9_.-]+', label)
    out = HERE / ('free-trace' if args.free_trace else 'notch-mechanism') / label
    out.mkdir(parents=True)  # Preserve every attempt; never overwrite.
    checkpoint = SAVED / 'restart/01'
    hashes = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
    shutil.copytree(checkpoint, out / 'restart/01')
    (out / 'restart/last_good_checkpoint.txt').write_text('1\n')
    prm = out / 'run.prm'
    prm.write_text(f'''include {SAVED}/run.prm
set Output directory = {out}
set Resume computation = true
subsection Postprocess
  subsection BP3
    set Committing work-measure replay = false
  end
end
''')
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_FAULT_COMPARE_SURFACE_INVERSE='1',
               ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1', ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC=str(SAVED),
               ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
               ASPECT_BP3_TIMESTEP_SEQUENCE=str(SAVED / 'accepted_steps.csv'),
               ASPECT_BP3_TARGET_MESH=str(HERE / 'junction-matched-qualified-local4/refined/target_cells.txt'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    if args.free_trace:
        env.update(ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC='795', ASPECT_FAULT_COMPARE_COUPLING='1')
    else:
        env.update(ASPECT_BP3_NOTCH_BOUNDARY_PROBE=str(args.amplitude),
                   ASPECT_BP3_NOTCH_PROBE_XD=str(args.location))
    if args.clamp_others:
        env['ASPECT_BP3_NOTCH_CLAMP_OTHER_RATES'] = '1'
    command = ['timeout', '--signal=TERM', '--kill-after=15', '600', 'mpirun', '-np', '4',
               '--bind-to', 'core', '--map-by', 'core', str(REPO / 'build-pf-cpdi/aspect-release'), str(prm)]
    record = dict(command=command, checkpoint_sha256=hashes,
                  environment={k: v for k, v in env.items() if k.startswith('ASPECT_')},
                  executable_sha256=digest(REPO / 'build-pf-cpdi/aspect-release'),
                  plugin_sha256=digest(HERE / 'build/libbp3.release.so'), amplitude=args.amplitude,
                  location=args.location, clamp_others=args.clamp_others, free_trace=args.free_trace)
    (out / 'source.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--', 'source/reconstructed_fault/manager.cc',
         'source/reconstructed_fault/surface_system.cc', 'include/aspect/reconstructed_fault/manager.h',
         'benchmarks/reconstructed_fault/bp3/bp3.cc',
         'benchmarks/reconstructed_fault/bp3/within_step_diagnostic.h'], cwd=REPO))
    (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    with (out / 'run.log').open('x') as stream:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = (out / 'run.log').read_text()
    linear = re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)', log)
    record.update(status=result.returncode, seconds=time.monotonic()-start,
                  peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  converged='Noncommitting fault diagnostic converged:' in log,
                  rollback='BP3 noncommitting rollback verified:' in log,
                  derivatives='Within-step input/derivative checks passed;' in log,
                  trace_checks=(not args.free_trace or 'Independent trace checks passed:' in log),
                  fresh_linear_passed=bool(linear) and all(float(a) <= float(b) for _, a, b in linear),
                  fresh_linear_checks=len(linear), krylov=sum(int(i) for i, _, _ in linear),
                  incoming_identical=all((out / f'incoming_rank{rank}.txt').exists() and
                                        (out / f'incoming_rank{rank}.txt').read_bytes() ==
                                        (CONTROL / f'incoming_rank{rank}.txt').read_bytes() for rank in range(4)),
                  original_unchanged=hashes == {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()},
                  copy_unchanged=hashes == {p.name: digest(p) for p in (out / 'restart/01').iterdir() if p.is_file()},
                  accepted_output_written=(out / 'accepted_steps.csv').exists())
    (out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('checkpoint_sha256', 'environment')}, indent=2))
    assert all(record[k] for k in ('converged', 'rollback', 'derivatives', 'trace_checks', 'fresh_linear_passed',
                                  'incoming_identical', 'original_unchanged', 'copy_unchanged'))
    assert not record['accepted_output_written']


if __name__ == '__main__':
    main()
