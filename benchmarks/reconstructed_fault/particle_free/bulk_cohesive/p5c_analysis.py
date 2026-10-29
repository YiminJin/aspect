#!/usr/bin/env python3
"""Independent P5c ownership/intact controls; cell records include every tensor mode."""
from pathlib import Path
import argparse,json,hashlib,struct,zlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
def read(p):return np.atleast_1d(np.genfromtxt(p,names=True,delimiter=',',dtype=None,encoding=None))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def prepare_restart(work,mode,ranks):
    name=f'{mode}-r{ranks}';run=work/'runs'/name;choices=[]
    for p in (run/'restart').glob('[0-9][0-9]'):
        raw=zlib.decompress((p/'resume.z').read_bytes()[16:]);tag=f'bulk cohesive {mode} tensor cell history v1'.encode();off=raw.index(tag)+len(tag)
        q,step,pub,t=struct.unpack_from('<IIQd',raw,off)
        if step<64:choices.append((step,p))
    step,source=max(choices);target=work/'runs'/f'{name}-resume'
    if target.exists():return
    (target/'restart').mkdir(parents=True);shutil.copytree(source,target/'restart'/source.name)
    (target/'restart/last_good_checkpoint.txt').write_text(str(int(source.name))+'\n')
    (work/f'{name}-resume.prm').write_text((work/f'{name}.prm').read_text().replace(f'runs/{name}',f'runs/{name}-resume')+'set Resume computation = true\n')
def small(work,out,finish=False):
    result={'baseline_m':4.,'comparison_tolerance':1e-12,'MPI':{},'intact':{},'restart':{}}
    floor=np.ones(1992)*.001;floor[:54]=1000.
    floor[72:]=np.tile([.001,.001,.001,1000,.01]+[.001]*6+[.001]*6+[1000]*6+[.01]*6+[.001],64)
    for mode in ['fixed','radial']:
        maximum=0.;nonuniform={k:0. for k in ['Axx','Axy','Kxx','Kxy','Wxx','Wxy','L']}
        for i in range(65):
            a,b=[np.fromfile(work/'runs'/f'{mode}-r{n}'/f'record-{i:04}.bin').reshape(64,1992) for n in (1,4)]
            maximum=max(maximum,float(np.max(abs(a-b)/np.maximum(floor,np.max(abs(a),axis=0)))))
            x=read(work/'runs'/f'{mode}-r1'/f'means-{i:04}.csv')
            for k in nonuniform:nonuniform[k]=max(nonuniform[k],float(np.ptp(x[k])))
            for n in (1,4):
                a=read(work/'runs'/f'{mode}-r{n}'/'gates.csv');assert all(a['passed']==1)
        assert maximum<=1e-12,maximum
        assert nonuniform['Axy']>1e-6 and nonuniform['Kxy']>1e-6,nonuniform
        result['MPI'][mode]={'all_tensor_records_maximum_scaled_defect':maximum,'states':65,'cell_owners':64,'nonuniform_peak_to_peak':nonuniform}
        for n in (1,4):prepare_restart(work,mode,n)
        run=work/'runs'/f'intact-{mode}';initial=read(run/'phase-0000.csv');assert np.count_nonzero(initial['phi'])==0
        defect=0.;onset=None
        for i in range(25):
            a=read(run/f'means-{i:04}.csv')
            if a['phi'].max()>1e-10:
                if onset is None:onset=[(i-1)*.0125,i*.0125]
            else:
                expected=4/(2e6)*np.maximum(abs(a['tauxy'])-1000,0)**2
                defect=max(defect,float(np.max(abs(expected-a['Y']))/8))
        assert onset and defect<1e-12
        result['intact'][mode]={'initial_phi_exactly_zero':True,'onset_bracket_s':onset,'intact_driving_relative_defect':defect,'threshold_stress_Pa':3000.,'fracture_resistance_Pa':8.,'last_phase':float(read(run/'means-0024.csv')['phi'].mean())}
        if finish:
            for n in (1,4):
                base=work/'runs'/f'{mode}-r{n}';res=work/'runs'/f'{mode}-r{n}-resume';matches=[]
                for name in ['record-0064.bin','phase-0064.csv','means-0064.csv','boundary-reactions-0064.csv','phase-forces-0064.csv']:
                    assert sha(base/name)==sha(res/name),(mode,n,name)
                    matches.append(name)
                a=read(res/'bulk-accepted.csv');assert len(a)==1 and a['publications'][0]==64 and a['dt'][0]==.0125
                result['restart'][f'{mode}-r{n}']={'parent_publications':63,'accepted_publications':64,'bitwise_files':matches}
    save(out/('small-checks.json' if finish else 'small-checks-progress.json'),result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);p.add_argument('--finish',action='store_true');a=p.parse_args();small(a.work,a.out,a.finish)
