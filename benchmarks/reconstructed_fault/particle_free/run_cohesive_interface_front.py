#!/usr/bin/env python3
"""Sequential bounded front/admission and conditional growth cases."""
import os,sys,json,subprocess,shutil,hashlib,time
from pathlib import Path
import run_cohesive_interface_native as native
ROOT=Path(__file__).resolve().parents[3];WORK=ROOT/'build-particle-free/cohesive-interface-front';OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_front'

def setup():
    WORK.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    text=native.SOURCE.read_text().replace('/cohesive-interface-native/plugin/','/cohesive-interface-front/plugin/').replace('set Steps between checkpoint = 1','set Steps between checkpoint = 2')
    prm=WORK/'template.prm';prm.write_text(text);native.SOURCE=prm;native.WORK=WORK;native.OUT=OUT
    os.environ.update(CI_EXACT_JUMP='1',CI_LOAD_END='.84')

def main():
    setup();stage=sys.argv[1] if len(sys.argv)>1 else 'qualification'
    if stage=='qualification':
        split=0.
        for _ in range(110):split+=.002
        cases=[dict(name='qualification',nx=32,ny=160,dt=.002,end=.3,mode='front'),dict(name='split',nx=32,ny=160,dt=.002,end=split,mode='front'),dict(name='resume',nx=32,ny=160,dt=.002,end=.3,mode='front',resume=True,folder='split')]
    elif stage=='growth':
        certificate=OUT/'qualification_summary.json'
        assert certificate.exists() and json.loads(certificate.read_text())['passed']
        cases=[dict(name='growth-32',nx=32,ny=160,dt=.002,end=.86,mode='front'),dict(name='growth-64',nx=64,ny=160,dt=.002,end=.86,mode='front'),dict(name='growth-64-half',nx=64,ny=160,dt=.001,end=.86,mode='front')]
    elif stage=='identity':cases=[dict(name='uniform-control',nx=4,ny=40,dt=.002,end=.42,mode='formation')]
    else:raise ValueError(stage)
    records=[native.run(**case) for case in cases]
    (OUT/('runs-'+stage+'.json')).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
