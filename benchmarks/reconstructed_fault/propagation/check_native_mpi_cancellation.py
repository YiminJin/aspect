#!/usr/bin/env python3
"""Process-control regression only; this does not run ASPECT or MPI."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import run_native_mpi as runner


def check(out):
    out.mkdir(exist_ok=False)
    runner.E=out;runner.BUILD=out/'fake-build'
    (runner.BUILD/'plugin').mkdir(parents=True)
    (runner.BUILD/'aspect-release').write_text('process-control fixture only\n')
    (runner.BUILD/'plugin/libnative_endpoint_verification.release.so').write_text('fixture\n')
    inp=out/'input.prm';inp.write_text('# process-control regression\n')
    real_popen=subprocess.Popen;children=[]

    class InterruptedWait:
        def __init__(self,*args,**kwargs):
            self.child=real_popen([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
            children.append(self.child);self.first=True
        def __getattr__(self,key):return getattr(self.child,key)
        def wait(self,*args,**kwargs):
            if self.first:self.first=False;raise KeyboardInterrupt
            return self.child.wait(*args,**kwargs)

    subprocess.Popen=InterruptedWait
    try:
        try:
            runner.run(argparse.Namespace(input=inp,tag='interrupted',ranks=1,tangential=16,
                       end_time=None,failure_step=None,profile=False,wall_cap=120))
        except KeyboardInterrupt:pass
        else:raise AssertionError('Interrupt was not propagated')
    finally:subprocess.Popen=real_popen
    assert len(children)==1 and children[0].poll() is not None
    record=json.loads((out/'interrupted-run.json').read_text())
    assert record['interrupted'] and record['returncode'] is not None
    print('PASS: interrupt stopped the isolated child and archived its outcome; no ASPECT/MPI run')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    check(p.parse_args().output)
