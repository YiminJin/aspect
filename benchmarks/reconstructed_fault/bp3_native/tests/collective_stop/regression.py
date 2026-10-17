#!/usr/bin/env python3
"""Bounded LOCAL MPI regression, not a server run controller. Never overwrites runs."""
import argparse
import csv
import json
import hashlib
import signal
import os
from pathlib import Path
import shutil
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'benchmarks/reconstructed_fault/particle_free'))
from run_pf6_mature import flatten


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def prepare(work, case):
    out = work/'runs'/case
    out.mkdir(parents=True, exist_ok=False)
    injected = case in ('wall', 'wall-resume')
    build = work/('injected' if injected else 'control')
    text = flatten(ROOT/'benchmarks/reconstructed_fault/bp3_native/section7b-fe-perturbed.prm', ROOT)
    final = 10 if case == 'wall' else (3 if case == 'scheduled-split' else 5)
    cadence = 500000 if case.startswith('scheduled') else 1000000000
    text += f'''
set Additional shared libraries = {build/'native/libbp3_restore_150x50.release.so'}, {build/'libbp3_coupled_checks.release.so'}
set Output directory = {out/'run'}
set End time = 2000000
set Maximum time step = 200000
set Maximum first time step = 200000
subsection Material model
 subsection Phase field fault
  set Bulk history representation = DG trial stress
  set FE bulk normal quadrature points = 6
  set Initial time step = 200000
  set Particle stress strain convention = three dimensional deviator
  set Use homogeneous mesh data = true
  set Enable prescribed DG accepted history = true
 end
end
subsection Compositional fields
 set Compositional field methods = static, static, static, static, static
 set Mapped particle properties =
end
subsection Particles
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
end
subsection Postprocess
 subsection BP3
  set Last accepted step = {final}
  set Graceful wall seconds = 1000000000
  set Stop after first event = false
  set Heavy output slip interval = 1
  set Heavy output time interval = {cadence}
  set Profile slip interval = 1
  set Profile time interval = 1000000000
  set Audit full state every step = true
 end
 subsection BP3 coupled checks
  set Mature lifecycle checks = true
  set Prescribed schedule check = false
  set Write authoritative stress = true
  set Write FE coefficients = true
  set Scheduled field output = true
 end
end
subsection Termination criteria
 set Checkpoint on termination = true
end
subsection Checkpointing
 set Steps between checkpoint = 0
 set Time between checkpoint = 0
end
'''
    if case.endswith('resume'):
        parent = 'wall' if injected else 'scheduled-split'
        shutil.copytree(work/'runs'/parent/'run/restart', out/'run/restart')
        text += '\nset Resume computation = true\n'
    (out/'input.prm').write_text(text)
    return out


def check(work):
    run = lambda case: work/'runs'/case/'run'
    traces = [rows(run('wall')/f'wall-stop-rank-{rank}.csv') for rank in range(2)]
    for rank, trace in enumerate(traces):
        assert [int(r['step']) for r in trace] == [0, 1, 2]
        last = trace[-1]
        assert int(last['wall_request']) == (rank == 1)
        assert all(int(last[k]) == 0 for k in ('other_request','ordinary_heavy','ordinary_profile'))
        assert all(int(last[k]) == 1 for k in ('collective_stop','heavy','profile'))
        assert all(int(r['collective_stop']) == 0 for r in trace[:-1])
    for filename in ('field-outputs.csv','heavy_outputs.csv','profiles.csv'):
        assert [int(r['step']) for r in rows(run('wall')/filename)] == [0,2],filename
    assert (run('wall')/'profiles/fault_2.csv').is_file()
    cp = run('wall')/'restart'
    tag = int((cp/'last_good_checkpoint.txt').read_text())
    assert (cp/f'{tag:02d}'/'bp3_accepted_state.txt').read_text().split() == ['2','400000']
    for case in ('control','wall','wall-resume','scheduled-split','scheduled-resume','scheduled-control'):
        for path in run(case).glob('rank-*-coupled-checks.csv'):
            data = rows(path)
            assert all(int(r['publications']) == 1 for r in data)
            assert len({r['step'] for r in data}) == len(data)
    # Same timestep sequence, then compare all persistent fields, stress,
    # surface state/loads and complete particle property bytes by stable ID.
    from analyze_pf6_mature_pilot import physical_states, merged
    import numpy as np
    comparisons = {}
    for split, reference in [('wall-resume','control'),('scheduled-resume','scheduled-control')]:
        result = physical_states(run(split),5,run(reference),5)
        for k, values in result.items():
            assert max(values.values()) == 0.,(split,k,values)
        for kind, keys in [('authoritative-stress',('cell','qp'))]:
            a=merged(run(split),f'{kind}-5.csv',keys); b=merged(run(reference),f'{kind}-5.csv',keys)
            for key in a: assert np.array_equal(a[key],b[key]),(split,kind,key)
        for rank in range(2):
            for kind in ('bulk','particles'):
                name=f'audit_{kind}_5_rank{rank}.csv'
                assert (run(split)/name).read_bytes()==(run(reference)/name).read_bytes(),(split,name)
        comparisons[split]=dict(max_persistent_difference=0.,stress_difference=0.,particle_and_bulk_audit_byte_identical=True)
    seq=lambda case:[(r['step'],r['time'],r['dt']) for r in rows(run(case)/'accepted_steps.csv')]
    assert seq('control')[3:]==seq('wall-resume')
    assert seq('scheduled-control')[4:]==seq('scheduled-resume')
    assert [int(r['step']) for r in rows(run('scheduled-control')/'field-outputs.csv')]==[0,3,5]
    assert [int(r['step']) for r in rows(run('scheduled-resume')/'field-outputs.csv')]==[5]
    result=dict(nonroot_only_wall_request=True,collective_stop_all_ranks=True,ordinary_outputs_not_due=True,
                terminal_diagnostics_and_checkpoint=True,once_only_publication=True,
                accepted_timestep_sequence_identical=True,comparisons=comparisons,
                scheduled_output_restart_retained=True,strict_reference_match_passed=False)
    (work/'regression.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--rank':
        # Linux ru_maxrss measures the native child, not this Python wrapper.
        rank=os.environ['OMPI_COMM_WORLD_RANK']; out=Path(sys.argv[2]); binary=sys.argv[3]
        begin=time.monotonic()
        result=subprocess.run([binary,str(out/'input.prm')])
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        (out/f'resources-rank-{rank}.json').write_text(json.dumps(dict(
            native_max_rss_KiB=usage.ru_maxrss,seconds=time.monotonic()-begin,
            user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,exit_code=result.returncode),indent=2)+'\n')
        raise SystemExit(result.returncode)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','check'])
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--case',choices=['control','wall','wall-resume','scheduled-split','scheduled-resume','scheduled-control'])
    p.add_argument('--binary',type=Path,default=ROOT/'build-particle-free/native/aspect-release')
    a=p.parse_args(); work=a.work.resolve()
    if a.action=='check': return check(work)
    assert a.case
    out=prepare(work,a.case)
    command=['mpirun','--map-by','core','--bind-to','core','-n','2',sys.executable,str(Path(__file__).resolve()),'--rank',str(out),str(a.binary.resolve())]
    env=os.environ.copy();env.update(OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1')
    begin=time.monotonic()
    artifacts=[a.binary.resolve(),out/'input.prm']+list((work/('injected' if a.case.startswith('wall') else 'control')).rglob('*.so'))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    with (out/'stdout.log').open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    record=dict(command=command,seconds=time.monotonic()-begin,exit_code=code,timeout_seconds=240,sha256=hashes)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));raise SystemExit(code)

if __name__=='__main__':main()
