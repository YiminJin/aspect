"""One committing mature-work BP3 replay; fresh start, four ranks, 40-minute cap."""
import argparse
import csv
import hashlib
import json
import os
import resource
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'mature-fault-50-local4'
OUT=HERE/'work-replay-50-local4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run'])
    parser.add_argument('--half-timesteps',action='store_true',
                        help='Fresh revised-work replay splitting each of the ten accepted real steps into two.')
    args=parser.parse_args()
    global OUT
    if args.half_timesteps:
        reference=OUT
        OUT=HERE/'work-replay-halfdt-50-local4'
    if args.action=='prepare':
        OUT.mkdir()  # Never overwrite an attempt.
        if args.half_timesteps:
            with (reference/'accepted_steps.csv').open() as stream:
                original=[r for r in csv.DictReader(stream) if int(r['step'])<=10]
            assert len(original)==11 and original[0]['step']=='0'
            rows=[dict(step=0,time=0.,dt=0.)];mapping=[]
            for old in original[1:]:
                half=float(old['dt'])/2;end=float(old['time'])
                start=rows[-1]['time']
                rows.append(dict(step=len(rows),time=start+half,dt=half))
                rows.append(dict(step=len(rows),time=end,dt=half))
                assert abs((end-start)-2*half)<=1e-12*2*half
                mapping.append(dict(baseline_step=int(old['step']),half_step=rows[-1]['step'],time=end))
            with (OUT/'clock.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['step','time','dt']);writer.writeheader();writer.writerows(rows)
            (OUT/'shared_times.json').write_text(json.dumps(mapping,indent=2)+'\n')
            include=reference
        else:
            (OUT/'clock.csv').write_bytes((BASE/'clock.csv').read_bytes())
            include=BASE
        (OUT/'run.prm').write_text(f'''include {include}/run.prm
set Output directory = {OUT}
set Resume computation = false
set End time = 922804465.5975173
subsection Postprocess
  subsection BP3
    set Committing work-measure replay = true
    set Bottom normalization completion file = {HERE}/top-source-paired-50-local4/completion.txt
  end
end
subsection Termination criteria
  set Termination criteria = end time, BP3 replay complete
end
''')
        print('Prepared one fresh four-rank replay; hard cap 2400 s. '+
              ('20 real half-steps; expected 25–35 min, approximately 1.5–2 GiB per rank. No controller bypass.'
               if args.half_timesteps else '10 real steps; expected 15–25 min.'))
        return
    assert (OUT/'run.prm').exists() and not (OUT/'run.log').exists()
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
        ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(OUT/'clock.csv'),
        ASPECT_BP3_ADAPTIVE_REPLAY='1',
        ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
        ASPECT_BP3_EXACT_TARGET='1',ASPECT_BP3_EXPECTED_FAULT=str(HERE/'fault-grid-50-local4/fault.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    if args.half_timesteps:
        # A temporal comparison must actually use the requested halves. The
        # existing strict replay cap stops if another controller requires less.
        env.pop('ASPECT_BP3_ADAPTIVE_REPLAY',None)
    command=['timeout','--signal=TERM','--kill-after=15','2400',
             'mpirun','-np','4','--bind-to','core','--map-by','core',
             str(REPO/'build-pf-cpdi/aspect-release'),str(OUT/'run.prm')]
    paths=[Path(__file__),OUT/'run.prm',OUT/'clock.csv',BASE/'prestress.txt',
           HERE/'top-source-paired-50-local4/completion.txt',HERE/'bp3.cc',HERE/'work_replay.h',
           HERE/'replay_stop.h',HERE/'replay_time_step.h',
           HERE/'build/libbp3.release.so',REPO/'build-pf-cpdi/aspect-release']
    if args.half_timesteps: paths.extend([reference/'accepted_steps.csv',OUT/'shared_times.json'])
    (OUT/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    start=time.monotonic()
    with (OUT/'run.log').open('x') as stream:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
    text=(OUT/'run.log').read_text()
    record=dict(status=result.returncode,seconds=time.monotonic()-start,
                child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                first_update_passed='BP3 WORK REPLAY FIRST UPDATE PASSED' in text)
    (OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert result.returncode==0 and record['first_update_passed'],'Preserve failure; no automatic retry or solver retuning.'


if __name__=='__main__':main()
