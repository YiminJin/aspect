#!/usr/bin/env python3
"""Sequential bounded native kernels and native owner/restart control."""
import argparse,os,subprocess,time,json,shutil
import numpy as np
from check_cohesive_interface_kink import ROOT,WORK,OUT,Kernel,write
EXE=ROOT/'build-particle-free/coulomb-native/aspect-release'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',ASPECT_SOURCE_DIR=str(ROOT),MPLCONFIGDIR='/tmp/kink-mpl')
def command(args,log,env=ENV,timeout=180):
    start=time.perf_counter()
    with (WORK/log).open('w') as f:p=subprocess.run([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT,timeout=timeout)
    r=dict(command=[str(a) for a in args],seconds=time.perf_counter()-start,exit_code=p.returncode)
    if p.returncode:raise RuntimeError((r,log))
    return r

def kernels():
    records=[]
    for name,nx,ny,q in [('native-16',16,8,3),('native-32',32,16,3),('native-16-q6',16,8,6),('native-16-axial',16,8,3)]:
        out=WORK/name
        if out.exists():out.rename(WORK/(name+'-prior-'+str(time.time_ns())))
        out.mkdir();records.append(command([WORK/'build/cohesive_interface_kink',nx,ny,out,q]+(['axial'] if 'axial' in name else []),name+'.log'))
    write('kernel-runs.json',records)

def owner():
    def make_input(k,s,filename):
        _,lam=k.certificate(s);coeff=k.coefficients(s);bulk=np.loadtxt(k.path/'response.csv',delimiter=',')@coeff
        hist=np.column_stack([s['histories'][key] for key in ['jump','damage','theta','reference','relaxed_jump','absolute_slip']])
        inp=WORK/filename
        with inp.open('w') as f:
            f.write(f'{k.meta["nx"]} {len(k.nodes)} {len(bulk)} {len(k.s)}\n')
            for a in [coeff,bulk,s['delta'],s['memory'],hist,k.K,k.Q,k.w,k.Nm,k.Np,lam]:f.write(' '.join(format(float(x),'.17g') for x in np.array(a).ravel())+'\n')
        return inp
    k=Kernel('native-16');s=k.initial();inp=make_input(k,s,'owner-input.txt')
    zero=k.initial(0.)
    for key in ['jump','damage','reference','relaxed_jump','absolute_slip']:zero['histories'][key]=np.zeros(len(k.s)).tolist()
    zero['delta']=np.zeros(len(k.nodes)).tolist();zero['memory']=zero['delta'].copy()
    unloaded_input=make_input(k,zero,'owner-unloaded-input.txt')
    ak=Kernel('native-16-axial');zero['wall_memory']=zero['gamma']=.0025
    overloaded_input=make_input(ak,zero,'owner-overloaded-input.txt')
    base=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
    base=base.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so',str(WORK/'build/libcohesive_interface_kink_owner.release.so')).replace('cohesive interface native checks','cohesive kink owner checks').replace('set Y repetitions = 40','set Y repetitions = 4').replace('set Maximum time step = 0.002','set Maximum time step = 0.001').replace('set Maximum first time step = 0.002','set Maximum first time step = 0.001')
    env=dict(ENV,CI_KINK_INPUT=str(inp));records=[]
    for name,folder,end,resume in [('owner-control','owner-control',.002,False),('owner-split','owner-split',.001,False),('owner-resume','owner-split',.002,True),('owner-unloaded','owner-unloaded',.002,False),('owner-overloaded','owner-overloaded',.002,False)]:
        out=WORK/folder/'output';out.parent.mkdir(exist_ok=True)
        if out.exists() and not resume:out.rename(out.parent/('prior-'+str(time.time_ns())))
        text=base.replace('set End time = 0.004',f'set End time = {end}').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(out))
        if resume:text+='\nset Resume computation = true\n'
        prm=WORK/(name+'.prm');prm.write_text(text)
        run_env=dict(env)
        if name=='owner-unloaded':run_env['CI_KINK_INPUT']=str(unloaded_input)
        if name=='owner-overloaded':run_env.update(CI_KINK_INPUT=str(overloaded_input),CI_EXPECT_KINK_REJECTION='1')
        records.append(command([EXE,prm],name+'.log',run_env))
        dest=OUT/name;dest.mkdir(exist_ok=True)
        for f in out.glob('kink-*'):shutil.copy2(f,dest/f.name)
    a=(OUT/'owner-control/kink-owner-final.archive').read_bytes();b=(OUT/'owner-resume/kink-owner-final.archive').read_bytes()
    assert a==b
    write('native-owner.json',dict(runs=records,restart_bitwise=True,scope='real native manager swap and ASPECT checkpoint; mechanical kernel state imported, physical time fixed'))
    print('Native geometry owner and ASPECT restart identity passed.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['kernels','owner']);args=p.parse_args();WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (kernels if args.stage=='kernels' else owner)()
