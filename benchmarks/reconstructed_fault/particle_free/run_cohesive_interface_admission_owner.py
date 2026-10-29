#!/usr/bin/env python3
"""Verify the activation window through native history ownership and ASPECT restart."""
import os,json,time,subprocess,shutil
import numpy as np
from cohesive_interface_admission import ROOT,OUT,WORK,Kernel,Admission,save
HOST=ROOT/'build-particle-free/coulomb-native/aspect-release'
def main():
    k=Kernel('native-16-axial');s=json.loads((OUT/'coarse-preactivation.json').read_text());solver=Admission(k)
    states=json.loads((OUT/'coarse-states.json').read_text());states=[x for x in states if x['clock_s']>s['clock_s']][:3]
    response=np.loadtxt(k.path/'response.csv',delimiter=',');c=np.r_[s['wall_memory'],s['memory']];bulk=response@c
    from cohesive_interface_admission import bounded_multiplier
    lam,err,peak=bounded_multiplier(k,(k.K@c)[1:],np.ones(len(k.s),bool));assert peak<=3000
    def history(state):return np.column_stack([state['histories'][x] for x in ['jump','damage','theta','reference','relaxed_jump','absolute_slip']])
    def emit(f,a):f.write(' '.join(format(float(x),'.17g') for x in np.array(a).ravel())+'\n')
    inp=WORK/'owner-input.txt'
    with inp.open('w') as f:
        f.write(f'{k.meta["nx"]} {len(k.nodes)} {len(bulk)} {len(k.s)}\n')
        for a in [c,bulk,s['delta'],s['memory'],history(s),k.K,k.Q,k.w,k.Nm,k.Np,lam[k.nq]]:emit(f,a)
    replay=WORK/'admission-input.txt'
    with replay.open('w') as f:
        f.write(f'{len(states)} {s["clock_s"]:.17g} {s["gamma"]:.17g}\n');emit(f,response)
        old=s
        for target in states:
            p=solver.solve(old,target['clock_s'],verify=True);new=solver.publish(old,p)
            for key in ['histories','delta','memory','gamma','wall_memory']:assert new[key]==target[key]
            f.write(f'{new["clock_s"]:.17g} {new["gamma"]:.17g}\n')
            cc=np.r_[new['wall_memory'],new['memory']];multipliers=np.zeros(len(k.s));multipliers[np.array(p['tied'])]=p['multipliers']
            for a in [cc,new['delta'],new['memory'],history(new),response@cc,p['tied'],multipliers]:emit(f,a)
            old=new
    template=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
    template=template.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so',str(WORK/'build/libcohesive_interface_kink_owner.release.so')).replace('cohesive interface native checks','cohesive kink owner checks').replace('set Y repetitions = 40','set Y repetitions = 4').replace('set Maximum time step = 0.002','set Maximum time step = 0.001').replace('set Maximum first time step = 0.002','set Maximum first time step = 0.001')
    env=dict(os.environ,CI_KINK_INPUT=str(inp),CI_ADMISSION_INPUT=str(replay),ASPECT_SOURCE_DIR=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    records=[]
    for name,folder,end,resume in [('native-control','native-control',.004,False),('native-split','native-split',.002,False),('native-resume','native-split',.004,True)]:
        directory=WORK/folder/'output';directory.parent.mkdir(exist_ok=True)
        if directory.exists() and not resume:directory.rename(directory.parent/('prior-'+str(time.time_ns())))
        text=template.replace('set End time = 0.004',f'set End time = {end}').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(directory))
        if resume:text+='\nset Resume computation = true\n'
        prm=WORK/(name+'.prm');prm.write_text(text);start=time.perf_counter()
        with (WORK/(name+'.log')).open('w') as log:r=subprocess.run([str(HOST),str(prm)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        records.append(dict(name=name,seconds=time.perf_counter()-start,exit_code=r.returncode));assert r.returncode==0,(name,r.returncode)
        dest=OUT/name;dest.mkdir(exist_ok=True)
        for p in directory.glob('admission-*'):shutil.copy2(p,dest/p.name)
        for p in directory.glob('kink-*.csv'):shutil.copy2(p,dest/p.name)
    assert (OUT/'native-control/admission-final.archive').read_bytes()==(OUT/'native-resume/admission-final.archive').read_bytes()
    a=np.genfromtxt(OUT/'native-control/admission-native.csv',delimiter=',',names=True);b=np.genfromtxt(OUT/'native-resume/admission-native.csv',delimiter=',',names=True);assert np.array_equal(a,b)
    save('native-owner.json',dict(runs=records,restart_bitwise=True,publications=len(a),geometry_publications=1,maximum_source_scaled_error=float(max(a['source_scaled_error'])),maximum_weak_Pa=float(max(a['weak_error_Pa'])),maximum_ledger_J_m=float(max(abs(a['ledger_error_J_m'])))))
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
