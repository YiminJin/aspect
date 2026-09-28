"""Run one staged branch inside an allocation, once. No retry or staging."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from stage_normal_stress_diagnostic import sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('binary',type=Path)
    p.add_argument('launcher',nargs=argparse.REMAINDER,help='e.g. ibrun, or mpirun -np 32')
    args=p.parse_args();root=args.directory.resolve();binary=args.binary.resolve()
    if not args.launcher:raise ValueError('Supply the original MPI launcher/rank count explicitly')
    manifest=json.loads((root/'experiment.json').read_text())
    if (root/'experiment_run.json').exists() or (root/'experiment.log').exists():
        raise ValueError('Existing execution preserved; no automatic retry')
    if sha(root/'normal_stress_diagnostic_restart.prm')!=manifest['run_prm_sha256']:
        raise ValueError('The prepared input changed')
    staging=json.loads((root/'staging.json').read_text())
    for name,digest in {**staging['staged_input_sha256'],**staging['libraries']}.items():
        if sha(root/name)!=digest:raise ValueError(f'Prepared dependency changed: {name}')
    if sha(root/'production_input.prm')!=staging['original_input_sha256']:
        raise ValueError('The original physical input changed')
    for name,digest in staging['checkpoint_sha256'].items():
        expected=manifest['resume_after_sha256'] if name=='resume.z' and manifest['branch']=='B' else digest
        if sha(root/'output-normal-diagnostic/restart/01'/name)!=expected:
            raise ValueError(f'Prepared checkpoint changed: {name}')
    environment={k:v for k,v in os.environ.items()
                 if k.startswith(('ASPECT_','OMP_','OPENBLAS_','DEAL_II_')) or
                 k in ('LD_LIBRARY_PATH','SLURM_NTASKS','SLURM_NPROCS','SLURM_CPUS_PER_TASK')}
    record=dict(binary=str(binary),binary_sha256=sha(binary),launcher=args.launcher,environment=environment)
    if manifest['branch']=='B':
        if sha(root/'expected_clock.txt')!=manifest['expected_clock_sha256']:
            raise ValueError('Prepared half-step clock changed')
        control=json.loads((Path(manifest['A_directory'])/'experiment_run.json').read_text())
        if not control['passed']:raise ValueError('A did not complete')
        for k in ('binary_sha256','launcher','environment'):
            if record[k]!=control[k]:raise ValueError(f'A/B execution differs: {k}')
    start=time.monotonic()
    with (root/'experiment.log').open('x') as log:
        status=subprocess.run(['timeout','--kill-after=15','900']+args.launcher+[str(binary),'normal_stress_diagnostic_restart.prm'],
                              cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
    summary=root/'output-normal-diagnostic/normal_summary.csv'
    rows=list(csv.DictReader(summary.open())) if summary.exists() else []
    passed=status==0 and len(rows)==manifest['solves']
    if passed:
        passed=[int(r['step']) for r in rows]==list(range(5613,5613+manifest['solves']))
    if passed and manifest['branch']=='B':
        passed=all(float(r['time_s'])==s['time_s'] and float(r['dt'])==s['dt'] for r,s in zip(rows,manifest['schedule']))
    record.update(status=status,seconds=time.monotonic()-start,passed=passed,
                  accepted=[dict(step=int(r['step']),time_s=float(r['time_s']),dt=float(r['dt'])) for r in rows])
    (root/'experiment_run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not passed:raise SystemExit('Incomplete branch preserved. No retry or additional solves.')
