#!/usr/bin/env python3
"""Run one bounded native case, preserving individual cost/input provenance."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import threading
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent/'native_endpoint_evidence'
BUILD=ROOT/'build-propagation-integration'


def sha(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def run(case,tag,max_wall_seconds=None,force_complete_failure=False,failure_step=None,input_file=None):
    original=input_file or E/('operators.prm' if case=='operators' else 'native-onset.prm' if case=='onset' else 'native-prefix.prm')
    assert case!='fixed' or input_file is not None,'Fixed-entry diagnostics require an explicit fixture'
    out=BUILD/'runs'/f'output-native-endpoint-{case}-{tag}'
    assert not out.exists(),'Preserve earlier native states and failures'
    text=original.read_text()
    lines=[]
    for line in text.splitlines():
        if line.strip().startswith('set Output directory ='):
            line=f'set Output directory = {out}'
        lines.append(line)
    if force_complete_failure:
        assert case!='operators'
        failure_step=failure_step or (3 if case=='onset' else 1)
        assert failure_step>=1
        lines.extend(['subsection Postprocess',' subsection P2 native endpoint',
                      '  set Force complete step failure and retry = true',
                      f'  set Complete failure timestep = {failure_step}',' end','end'])
    prm=E/f'{case}-{tag}.prm';prm.write_text('\n'.join(lines)+'\n')
    env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    executable=BUILD/'aspect-release';plugin=BUILD/'plugin/libnative_endpoint_verification.release.so'
    provenance=dict(executable_sha=sha(executable),plugin_sha=sha(plugin),input_sha=sha(prm),
         initial_phi_sha=sha(E/'shared-initial-phase.txt'),initial_H_sha=sha(E/'shared-initial-H.txt'),
         configuration='fresh shared initialization, fixed 16x512, full-normal association, frozen particles',
         output=str(out),threads=1,started_UTC=datetime.now(timezone.utc).isoformat(),
         max_wall_seconds=max_wall_seconds)
    provenance['force_complete_failure']=force_complete_failure
    provenance['complete_failure_timestep']=failure_step
    provenance['source_input']=str(original)
    if case=='fixed':provenance['configuration']+='; scoped noncommitting fixed-entry references, not physical evolution'
    started=time.perf_counter()
    (E/f'{case}-{tag}-start.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (E/f'{case}-{tag}.log').open('w') as log:
        child=subprocess.Popen([str(executable),str(prm)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        stop_file=E/f'{case}-{tag}.stop'
        assert not stop_file.exists(),'Preserve prior stop requests'
        finished=threading.Event()
        def watch_budget():
            while not finished.wait(1.):
                expired=max_wall_seconds is not None and time.perf_counter()-started>=max_wall_seconds
                if expired or stop_file.exists():
                    provenance['wall_budget_reached']=expired
                    provenance['stop_reason']='wall budget' if expired else stop_file.read_text().strip()
                    try:child.send_signal(signal.SIGINT)
                    except ProcessLookupError:pass
                    return
        watcher=threading.Thread(target=watch_budget,daemon=True);watcher.start()
        try:
            _,status,usage=os.wait4(child.pid,0)
        except KeyboardInterrupt:
            child.send_signal(signal.SIGINT)
            _,status,usage=os.wait4(child.pid,0)
            provenance['interrupted']=True
        finally:
            finished.set();watcher.join()
        child.returncode=os.waitstatus_to_exitcode(status)
    provenance.update(returncode=child.returncode,wall_seconds=time.perf_counter()-started,
                      user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,
                      peak_RSS_KiB=usage.ru_maxrss)
    (E/f'{case}-{tag}-run.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)
    if child.returncode:raise SystemExit(child.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['operators','prefix','onset','fixed'])
    parser.add_argument('--tag',default='initial')
    parser.add_argument('--max-wall-seconds',type=float)
    parser.add_argument('--force-complete-failure',action='store_true')
    parser.add_argument('--failure-step',type=int,help='Explicit controlled-failure physical timestep (onset default: 3).')
    parser.add_argument('--input',type=Path,help='Explicit benchmark fixture; its exact hash is retained.')
    args=parser.parse_args()
    if args.max_wall_seconds is not None:assert args.max_wall_seconds>0
    run(args.case,args.tag,args.max_wall_seconds,args.force_complete_failure,args.failure_step,args.input)
