#!/usr/bin/env python3
"""Bounded local native-ASPECT runner. No constitutive or evolution computation.

The .prm files and native owner are authoritative. Each new run gets an isolated
folder; a restart branches by copying only the valid native restart directory.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, signal, subprocess, time, shlex, csv, tempfile, math

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / 'build-particle-free/cohesive-rsf-native'



def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('case', choices=['conversion', 'general', 'short', 'pilot', 'history', 'active', 'validate', 'portable-check'])
    ap.add_argument('--label', required=True)
    ap.add_argument('--ranks', type=int, choices=[1, 2, 4], default=2)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--graceful-stop-after', type=float, help='Request accepted-state termination after this many wall seconds; allow terminal output before --timeout.')
    ap.add_argument('--end', type=float)
    ap.add_argument('--dt', type=float)
    ap.add_argument('--resume-from', type=Path)
    ap.add_argument('--resume-slot', type=int)
    ap.add_argument('--force-reject', type=int)
    ap.add_argument('--history-records', action='store_true')
    ap.add_argument('--operators', action='store_true', help='Export small native K/Q/N for independent stability diagnostics.')
    ap.add_argument('--save-at-speed', default='', help='Comma-separated accepted speed thresholds (m/s) at which to preserve the latest complete native checkpoint.')
    ap.add_argument('--missing-reference', action='store_true')
    ap.add_argument('--affine', action='store_true')
    ap.add_argument('--cutback-control', action='store_true')
    ap.add_argument('--cutback-time', type=float)
    ap.add_argument('--work-dir', type=Path, default=Path(os.environ.get('ASPECT_COHESIVE_WORK_DIR', WORK)))
    ap.add_argument('--aspect', type=Path, default=os.environ.get('ASPECT_BINARY'))
    ap.add_argument('--plugin', type=Path, default=os.environ.get('ASPECT_COHESIVE_PLUGIN'))
    ap.add_argument('--mpiexec', default=os.environ.get('MPIEXEC', 'mpiexec'))
    ap.add_argument('--mpi-args', default=os.environ.get('MPIEXEC_PREFLAGS', ''))
    ap.add_argument('--np-flag', default=os.environ.get('MPIEXEC_NUMPROC_FLAG', '-n'))
    ap.add_argument('--input', type=Path)
    ap.add_argument('--cwd', type=Path, default=ROOT)
    ap.add_argument('--regression-from', type=Path, help='Optional retained pilot output/checkpoints for compatibility checks.')
    args = ap.parse_args(argv)
    if args.graceful_stop_after is not None and not 0 < args.graceful_stop_after < args.timeout:
        ap.error('--graceful-stop-after must be positive and smaller than --timeout')
    work = args.work_dir.expanduser().resolve()
    binary = (args.aspect or work/'native/aspect-release').expanduser().resolve()
    plugin = (args.plugin or work/'checks-native/libcohesive_rsf_native_checks.release.so').expanduser().resolve()
    cwd = args.cwd.expanduser().resolve()
    def command(prm, ranks, validate=False):
        launch = shlex.split(args.mpiexec) + shlex.split(args.mpi_args)
        if args.np_flag: launch += [args.np_flag, str(ranks)]
        return launch + [str(binary)] + (['--validate'] if validate else []) + [str(prm)]
    def execute(cmd, env, log):
        start = time.monotonic()
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            thresholds=sorted(float(x) for x in args.save_at_speed.split(',') if x)
            saved=set()
            while process.poll() is None:
                elapsed=time.monotonic()-start
                if elapsed>=args.timeout: raise subprocess.TimeoutExpired(cmd,args.timeout)
                if args.graceful_stop_after is not None and elapsed >= args.graceful_stop_after:
                    terminal = folder/'output/terminate-aspect'
                    terminal.parent.mkdir(parents=True, exist_ok=True)
                    terminal.touch(exist_ok=True)
                if thresholds:
                    # Only copy a published, completed native checkpoint. It may
                    # precede the newest accepted diagnostic by one step.
                    try:
                        rows=list(csv.DictReader((folder/'output/cohesive-index.csv').open()))
                        row=rows[-1];stem=Path(row['stem'])
                        d={r['quantity']:float(r['value']) for r in csv.DictReader(Path(str(stem)+'-diagnostics.csv').open())}
                        speed=d.get('maximum_speed_m_s',0.)
                        for threshold in thresholds:
                            if threshold in saved or speed<threshold: continue
                            restart=folder/'output/restart';marker=restart/'last_good_checkpoint.txt'
                            slot=marker.read_text().strip();source=restart/f'{int(slot):02d}'
                            signatures={f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in source.iterdir() if f.is_file()}
                            destination=folder/'preserved'/f'speed-{threshold:.6g}'
                            temporary=destination.with_name(destination.name+'.copying')
                            temporary.mkdir(parents=True,exist_ok=True)
                            shutil.copytree(source,temporary/'restart'/source.name,dirs_exist_ok=True)
                            current={f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in source.iterdir() if f.is_file()}
                            if marker.read_text().strip()!=slot or current!=signatures:
                                continue
                            (temporary/'restart/last_good_checkpoint.txt').write_text(slot)
                            (temporary/'observation.json').write_text(json.dumps(dict(threshold_m_s=threshold,observed_accepted=row,speed_m_s=speed,checkpoint_slot=slot,files=signatures),indent=2)+'\n')
                            temporary.rename(destination);saved.add(threshold)
                    except (FileNotFoundError,IndexError,KeyError,ValueError):
                        pass # Output/checkpoint publication has not completed yet.
                try: process.wait(timeout=min(.5,args.timeout-elapsed))
                except subprocess.TimeoutExpired: pass
            code=process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            code = 124
        return code, time.monotonic()-start
    if Path(args.label).name != args.label:
        raise SystemExit('Label must be one directory name')
    folder = work / 'runs' / args.label
    folder.mkdir(parents=True, exist_ok=False)
    if args.case == 'portable-check':
        # A fixed set of small integration controls; no parameter search or
        # independent evolution model. Every state is computed by ASPECT.
        from verify import snapshot
        launch_dir=Path(tempfile.mkdtemp(prefix='cohesive-portable-cwd-'))
        inputs=folder/'inputs';inputs.mkdir()
        cases=[]
        common=['--work-dir',str(work),'--aspect',str(binary),'--plugin',str(plugin),
                '--mpiexec',args.mpiexec,'--mpi-args='+args.mpi_args,'--np-flag='+args.np_flag,
                '--cwd',str(launch_dir),'--timeout',str(args.timeout),'--history-records']
        def case(name, text='', extra=(), failure=None):
            label=args.label+'-'+name
            path=inputs/(name+'.prm')
            path.write_text(f'include {HERE/"active.prm"}\n'+text)
            code=0
            try: main(['active','--label',label,'--input',str(path)]+common+list(extra))
            except SystemExit as error: code=error.code
            run=work/'runs'/label
            log=(run/'run.log').read_text()
            assert (code!=0 and failure in log) if failure else code==0, (name,code,log[-2000:])
            cases.append(dict(case=name,returncode=code,expected_failure=failure,
                              execution=json.loads((run/'execution.json').read_text())))
            return run/'output'
        def stems(output):
            return {round(float(r['time_s']),12):Path(r['stem']) for r in csv.DictReader((output/'cohesive-index.csv').open())}
        fresh=case('fresh')
        custom='subsection Cohesive RSF pilot\n set Seed x = 3.2\n set Seed y = .1\n set Seed angle = 5\n set Virgin normalized age = .9\n subsection Initial surface histories\n  set Function expression = .0004*4*z*(1-z); .75+.05*4*z*(1-z); 1.2+.1*4*z*(1-z); .0004*4*z*(1-z)-C0*(.75+.05*4*z*(1-z))/(1-(.75+.05*4*z*(1-z)))*q; 1e-8*min(z,1-z); .0004*4*z*(1-z)\n end\n subsection Initial native displacement memory\n  set Function expression = 1e-7*x*x*y; -1e-7*x*y*y\n end\nend\nsubsection Boundary velocity model\n subsection Cohesive pilot loading\n  subsection Strain history\n   set Function expression = .0003+.004*min(time,.000375); 5*min(time/.00025,1)\n  end\n end\nend\n'
        control=case('custom-control',custom,extra=['--end','.000375'])
        split=case('custom-split',custom,extra=['--end','.00025'])
        resumed=case('custom-resumed',custom,extra=['--resume-from',str(split),'--end','.000375'])
        x,y=stems(control),stems(resumed)
        checks=[]
        for t in sorted(x.keys() & y.keys()):
            result=snapshot(x[t],y[t]);checks.append(dict(time_s=t,**result))
            assert result['bulk_records_bitwise'] and result['surface_history_max_abs']==0. and result['geometry_bitwise']
            assert result['publications'][0]==result['publications'][1]
        # Independently check the configured seed and nonzero trace memory.
        initial=list(csv.DictReader(Path(str(stems(control)[0.])+'-surface.csv').open()))
        for row in initial:
            x0=float(row['x_m']);z=(x0-3.2)/3
            assert abs(float(row['y_m'])-(.1+math.tan(5*math.pi/180)*(x0-3.2)))<1e-13
            assert abs(float(row['relaxed_jump_m'])-1e-8*min(z,1-z))<1e-13
        assert any(float(row['relaxed_jump_m'])!=0 for row in initial)
        case('changed-loading',custom.replace('5*min(time/.00025,1)','6*min(time/.00025,1)'),
             extra=['--resume-from',str(split),'--end','.000375'],failure='loading schedule/axial ratio changed')
        case('nonrepresentable-memory',custom.replace('1e-8*min(z,1-z)','1e-5*z*(1-z)'),
             failure='Physical Gauss jump/Maxwell memory differs')
        case('changed-domain',custom+'\nsubsection Geometry model\n subsection Box\n  set X extent = 13\n end\nend\n',
             extra=['--resume-from',str(split),'--end','.000375'],failure='checkpoint version/material/geometry settings')
        case('sampling-outside','subsection Cohesive RSF pilot\n set Seed x = 7.6\n set Maximum formed length = 100\nend\n',failure='boundary protection failure')
        case('lifting-outside','subsection Cohesive RSF pilot\n set Seed y = 5.4\n set Seed angle = -20\nend\n',failure='boundary protection failure')
        candidate=case('candidate-outside','subsection Cohesive RSF pilot\n set Seed x = 7.4\n set Candidate extent = 1.8\n set Maximum candidate extent = 2\nend\n',failure='boundary protection failure')
        index=stems(candidate);assert list(index)==[0.]
        recovery=case('candidate-recovery','subsection Cohesive RSF pilot\n set Seed x = 7.4\n set Candidate extent = 1.8\n set Maximum candidate extent = 2\nend\n',
                      extra=['--resume-from',str(candidate),'--end','0'])
        unchanged=snapshot(index[0.],stems(recovery)[0.]);assert unchanged['bulk_records_bitwise'] and unchanged['surface_history_max_abs']==0.
        near=case('boundary-approach','subsection Cohesive RSF pilot\n set Seed x = 7.3\nend\n',extra=['--end','.000125'])
        assert (near/'cohesive-domain.csv').exists()
        regression=[]
        if args.regression_from:
            for name,slot,end in [('retained-step',1,'.462'),('retained-completed',2,'.462375')]:
                label=args.label+'-'+name
                try: main(['pilot','--label',label]+common+['--resume-from',str(args.regression_from),
                         '--resume-slot',str(slot),'--end',end,'--missing-reference'])
                except SystemExit as error: assert error.code==0
                output=work/'runs'/label/'output';current=stems(output)
                target=max(current)
                original=stems(args.regression_from)[target]
                result=snapshot(original,current[target]);assert result['surface_bitwise'] and result['bulk_center_files_bitwise'] and result['geometry_bitwise']
                regression.append(dict(case=name,time_s=target,execution=json.loads((work/'runs'/label/'execution.json').read_text()),**result))
        (folder/'verification.json').write_text(json.dumps(dict(cases=cases,restart=checks,
             candidate_recovery=unchanged,regression=regression,empty_launch_directory=str(launch_dir)),indent=2)+'\n')
        print(json.dumps(dict(portable_cases=len(cases),regression_cases=len(regression),passed=True)))
        return
    if args.case == 'validate':
        # Syntax validation only: no state conversion, mesh setup or advancement.
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
        results = []
        for prm in sorted(HERE.glob('*.prm')):
            wrapper=folder/(prm.stem+'.prm')
            wrapper.write_text(f'include {prm}\nset Additional shared libraries = {plugin}\n')
            cmd = command(wrapper, 1, validate=True)
            start = time.monotonic()
            with (folder / (prm.stem + '.log')).open('w') as log:
                code, elapsed = execute(cmd,env,log)
            results.append(dict(input=str(prm.relative_to(ROOT)), command=cmd,
                                returncode=code, elapsed_s=elapsed))
        (folder / 'execution.json').write_text(json.dumps(dict(
            binary_sha256=hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
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
    base = args.input.resolve() if args.input else HERE / f'{args.case}.prm'
    text = f'include {base}\nset Output directory = {output}\nset Additional shared libraries = {plugin}\n'
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
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1',
               OMPI_MCA_mpi_yield_when_idle=os.environ.get('OMPI_MCA_mpi_yield_when_idle','1'))
    if args.force_reject is not None:
        env['ASPECT_COHESIVE_FORCE_REJECT_STEP'] = str(args.force_reject)
    if args.history_records:
        env['ASPECT_COHESIVE_WRITE_HISTORY_RECORDS'] = '1'
    if args.operators:
        env['ASPECT_COHESIVE_EXPORT_OPERATORS'] = '1'
    if args.affine:
        env['ASPECT_COHESIVE_HISTORY_AFFINE'] = '1'
    cmd = command(prm, args.ranks)
    metadata = dict(command=cmd, cwd=str(cwd), timeout_s=args.timeout,
                    graceful_stop_after_s=args.graceful_stop_after,
                    start_unix_s=time.time(), ranks=args.ranks,
                    binary_sha256=hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
                    input=text, source_checkpoint=str(args.resume_from) if args.resume_from else None,
                    test_environment={k: v for k, v in env.items() if k.startswith('ASPECT_COHESIVE_')})
    metadata['run_environment']={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS','OMPI_MCA_mpi_yield_when_idle']}
    (folder / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (folder / 'run.log').open('w') as log:
        code, elapsed = execute(cmd,env,log)
    expected = output / ('maxwell-history-checks.csv' if args.case == 'history' else 'cohesive-index.csv')
    if code == 0 and not expected.exists():
        code = 125  # A help/syntax-only exit is not a completed native test.
    if code == 0:
        if args.case == 'history':
            actual_ranks = int(next(csv.DictReader(expected.open()))['ranks'])
        else:
            row = next(csv.DictReader(expected.open()))
            diagnostic = output/(Path(row['stem']).name+'-diagnostics.csv')
            values = {r['quantity']:r['value'] for r in csv.DictReader(diagnostic.open())}
            actual_ranks = int(float(values['ranks']))
        metadata['actual_ranks'] = actual_ranks
        if actual_ranks != args.ranks:
            code = 126  # Detect mismatched launcher/library or scheduler allocation.
    metadata.update(elapsed_s=elapsed, returncode=code)
    (folder / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(folder=str(folder), elapsed_s=metadata['elapsed_s'], returncode=code)), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
