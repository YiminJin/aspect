#!/usr/bin/env python3
"""Bounded local native-ASPECT runner. No constitutive or evolution computation.

The .prm files and native owner are authoritative. Each new run gets an isolated
folder; a restart branches by copying only the valid native restart directory.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, signal, subprocess, time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / 'build-particle-free/cohesive-rsf-native'
BINARY = WORK / 'native/aspect-release'
MPI = Path('/opt/openmpi/5.0.6/bin/mpiexec')


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('case', choices=['conversion', 'general', 'short', 'pilot', 'history', 'validate'])
    ap.add_argument('--label', required=True)
    ap.add_argument('--ranks', type=int, choices=[1, 2, 4], default=2)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--end', type=float)
    ap.add_argument('--dt', type=float)
    ap.add_argument('--resume-from', type=Path)
    ap.add_argument('--resume-slot', type=int)
    ap.add_argument('--force-reject', type=int)
    ap.add_argument('--history-records', action='store_true')
    ap.add_argument('--missing-reference', action='store_true')
    ap.add_argument('--affine', action='store_true')
    ap.add_argument('--cutback-control', action='store_true')
    ap.add_argument('--cutback-time', type=float)
    args = ap.parse_args()
    if Path(args.label).name != args.label:
        raise SystemExit('Label must be one directory name')
    folder = WORK / 'runs' / args.label
    folder.mkdir(parents=True, exist_ok=False)
    if args.case == 'validate':
        # Syntax validation only: no state conversion, mesh setup or advancement.
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
        results = []
        for prm in sorted(HERE.glob('*.prm')):
            cmd = [str(MPI), '--bind-to', 'none', '-n', '1', str(BINARY), '--validate', str(prm)]
            start = time.monotonic()
            with (folder / (prm.stem + '.log')).open('w') as log:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout)
            results.append(dict(input=str(prm.relative_to(ROOT)), command=cmd,
                                returncode=completed.returncode, elapsed_s=time.monotonic()-start))
        (folder / 'execution.json').write_text(json.dumps(dict(
            binary_sha256=hashlib.file_digest(BINARY.open('rb'), 'sha256').hexdigest(),
            parameter_validation=results), indent=2) + '\n')
        print(json.dumps(results, indent=2), flush=True)
        raise SystemExit(0 if all(r['returncode'] == 0 for r in results) else 1)
    output = folder / 'output'
    if args.resume_from:
        source = args.resume_from.resolve()
        shutil.copytree(source / 'restart', output / 'restart')
        if args.resume_slot is not None:
            slot = output / 'restart' / f'{args.resume_slot:02d}' / 'resume.z'
            if not slot.is_file():
                raise SystemExit('Requested checkpoint slot is absent')
            (output / 'restart/last_good_checkpoint.txt').write_text(str(args.resume_slot))
    base = HERE / f'{args.case}.prm'
    text = f'include {base}\nset Output directory = {output}\n'
    if args.case == 'history':
        text += 'set Additional shared libraries = ' + str(WORK / 'checks-native/libcohesive_rsf_native_checks.release.so') + '\n'
    if args.end is not None:
        text += f'set End time = {args.end:.17g}\n'
    if args.resume_from:
        text += 'set Resume computation = true\n'
    if args.missing_reference:
        if not args.resume_from:
            raise SystemExit('Missing-reference control requires a native restart')
        text += f'subsection Cohesive RSF pilot\n set Initial state file = {folder}/deliberately-absent-prototype.json\nend\n'
    if args.dt is not None:
        if args.dt <= 0:
            raise SystemExit('A positive timestep is required')
        text += f'set Maximum time step = {args.dt:.17g}\nset Maximum first time step = {args.dt:.17g}\nsubsection Time stepping\n subsection Function\n  set Function expression = {args.dt:.17g}\n end\nend\n'
    if args.cutback_control or args.cutback_time is not None:
        when = 0.000125 if args.cutback_time is None else args.cutback_time
        text += f'subsection Time stepping\n subsection Function\n  set Function expression = if(abs(time-{when:.17g})<1e-12,0.0000625,0.000125)\n end\nend\n'
    prm = folder / 'parameters.prm'
    prm.write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', OMPI_MCA_mpi_yield_when_idle='1')
    if args.force_reject is not None:
        env['ASPECT_COHESIVE_FORCE_REJECT_STEP'] = str(args.force_reject)
    if args.history_records:
        env['ASPECT_COHESIVE_WRITE_HISTORY_RECORDS'] = '1'
    if args.affine:
        env['ASPECT_COHESIVE_HISTORY_AFFINE'] = '1'
    cmd = [str(MPI), '--bind-to', 'none', '-n', str(args.ranks), str(BINARY), str(prm)]
    metadata = dict(command=cmd, cwd=str(ROOT), timeout_s=args.timeout, start_unix_s=time.time(), ranks=args.ranks,
                    binary_sha256=hashlib.file_digest(BINARY.open('rb'), 'sha256').hexdigest(),
                    input=text, source_checkpoint=str(args.resume_from) if args.resume_from else None,
                    test_environment={k: v for k, v in env.items() if k.startswith('ASPECT_COHESIVE_')})
    (folder / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    start = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    expected = output / ('maxwell-history-checks.csv' if args.case == 'history' else 'cohesive-index.csv')
    if code == 0 and not expected.exists():
        code = 125  # A help/syntax-only exit is not a completed native test.
    metadata.update(elapsed_s=time.monotonic()-start, returncode=code)
    (folder / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(folder=str(folder), elapsed_s=metadata['elapsed_s'], returncode=code)), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
