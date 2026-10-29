#!/usr/bin/env python3
"""Bounded exact-jump experiment, separate outputs and executable from baseline."""
import os,json,sys,shutil,subprocess,hashlib,time
from pathlib import Path
import run_cohesive_interface_native as native
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-exact'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_exact'

def setup():
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    text=native.SOURCE.read_text().replace('/cohesive-interface-native/plugin/','/cohesive-interface-exact/plugin/')
    prm=WORK/'template.prm';prm.write_text(text)
    native.SOURCE=prm;native.WORK=WORK;native.OUT=OUT
    os.environ['CI_EXACT_JUMP']='1'

def main():
    setup();stage=sys.argv[1] if len(sys.argv)>1 else 'operators'
    if stage=='operators':
        cases=[dict(name='operator-4'),dict(name='operator-8',nx=8,ny=80),dict(name='operator-16',nx=16,ny=160),dict(name='operator-32',nx=32,ny=160),dict(name='width-half',nx=32,ny=160,width=.0125),dict(name='quadrature6',quad=6)]
    elif stage=='histories':
        cases=[dict(name='formation',end=.42),dict(name='formation-half',dt=.001,end=.42),dict(name='mature',mode='mature',dt=.01,end=6),dict(name='nonuniform',mode='nonuniform',nx=8,ny=80,dt=.001,end=.15),dict(name='split',mode='nonuniform',nx=8,ny=80,dt=.001,end=.07500000000000005),dict(name='resume',mode='nonuniform',nx=8,ny=80,dt=.001,end=.15,resume=True,folder='split')]
    elif stage=='reject-mode-change':
        destination=WORK/'negative-restart'
        if destination.exists():destination.rename(WORK/('prior-negative-'+str(time.time_ns())))
        shutil.copytree(WORK/'runs/split/output',destination)
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.csv')}
        text=(WORK/'resume.prm').read_text().replace(str(WORK/'runs/split/output'),str(destination))
        prm=WORK/'negative-restart.prm';prm.write_text(text)
        env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_NX='8',CI_MODE='nonuniform',CI_WIDTH='.025',CI_EXACT_JUMP='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (WORK/'negative-restart.log').open('w') as log:
            result=subprocess.run([str(native.EXE),str(prm)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.csv')}
        # load emits a diagnostic row; no accepted history output may change.
        unchanged=all(after.get(k)==v for k,v in before.items() if k!='restart_loaded.csv')
        message='Interface restart configuration changed.'
        assert result.returncode!=0 and message in (WORK/'negative-restart.log').read_text() and unchanged
        (OUT/'restart_negative.json').write_text(json.dumps(dict(exit_code=result.returncode,diagnostic=message,accepted_outputs_unchanged=unchanged),indent=2)+'\n')
        return
    elif stage=='baseline':
        os.environ['CI_EXACT_JUMP']='0';cases=[dict(name='finite-width-control')]
    else:raise ValueError(stage)
    records=[native.run(**case) for case in cases]
    (OUT/('runs-'+stage+'.json')).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
