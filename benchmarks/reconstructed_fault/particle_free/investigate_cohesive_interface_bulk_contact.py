#!/usr/bin/env python3
"""Frozen common-field mixed mechanics; never converts accepted checkpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

from cohesive_interface_path import ROOT, WORK as PATH_WORK, PathKernel, save, table, encode, sha

WORK=ROOT/'build-particle-free/cohesive-interface-bulk-contact'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_bulk_contact'
EXE=WORK/'build/cohesive_interface_path'
SAVED=json.loads((PATH_WORK/'work96/final.json').read_text())
ARCS=SAVED['geometry']
COEFFICIENTS=np.r_[SAVED['wall_memory'],SAVED['memory']]


def build(label,n,degree=2,width=.75,mesh='cartesian',order=6,arcs=ARCS,coefficients=COEFFICIENTS,purpose='Frozen common-field reconstruction'):
    folder=WORK/label
    if not (folder/'native.json').exists():
        folder.mkdir(exist_ok=False)
        table(folder/'geometry.csv',arcs)
        np.savetxt(folder/'physical-coefficients.txt',coefficients,fmt='%.17g')
        save(folder/'input.json',dict(arcs=arcs,n=n,degree=degree,width_m=width,normal_test_m=.75,mesh=mesh,order=order,
            incoming=purpose,accepted_history_conversion=False,executable_sha256=sha(EXE)))
        command=[str(EXE),str(n),str(degree),str(folder/'geometry.csv'),str(folder),str(order),'compress',str(width),'.75',mesh,str(folder/'physical-coefficients.txt')]
        with (WORK/(label+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    k=PathKernel(folder,arcs,'work')
    return k


def identity():
    original=PATH_WORK/'kernel-d6f7f210663962e8'
    arcs=json.loads((original/'input.json').read_text())['arcs'];folder=WORK/'legacy-identity'
    if not (folder/'native.json').exists():
        folder.mkdir();table(folder/'geometry.csv',arcs)
        with (WORK/'legacy-identity.log').open('w') as log:
            subprocess.run([str(EXE),'16','2',str(folder/'geometry.csv'),str(folder),'6','compress'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    files=['stiffness.csv','Q.csv','normal-work.csv','normal-minus.csv','normal-plus.csv','shear.csv','columns.csv']
    matches={name:(folder/name).read_bytes()==(original/name).read_bytes() for name in files}
    assert all(matches.values()),matches
    k=build('fitted-small',16,mesh='fitted',arcs=arcs,coefficients=np.r_[.001,np.zeros(19)])
    assert k.meta['energy_relative']<1e-4
    k.close();save(OUT/'legacy-identity.json',dict(bitwise=matches,fitted_small_independent_checks=True))


def frozen(label,k):
    c=COEFFICIENTS;w=k.w;Q=k.Q
    mass=Q.T@(w[:,None]*Q)
    force=-(k.K@c)[1:];weak=np.linalg.solve(mass,force);shear=Q@weak
    raw=np.loadtxt(k.path/'shear.csv',delimiter=',')@c;normal=k.N@c
    pm=k.raw_Nm@c;pp=k.raw_Np@c
    rows=[];profiles=[];x=k.s['x_m'];xn=k.nodes['x_m']
    for name,lo,hi in [('old',3.75,5.5),('developed',6.,6.5),('zone',6.5,7.75),('ahead',7.75,8.4)]:
        test=np.where((xn>lo)&(xn<hi),np.sin(np.pi*(xn-lo)/(hi-lo))**2,0.)
        tq=Q@test;measure=w@tq;weight=w*tq
        rows.append(dict(label=label,region=name,measure_m=float(measure),
            tangential_work_conjugate_Pa=float(test@force/measure),normal_perturbation_Pa=float(weight@normal/measure),
            raw_shear_Pa=float(weight@raw/measure),raw_normal_mean_Pa=float(weight@(.5*(pm+pp))/measure),
            raw_normal_jump_RMS_Pa=float(np.sqrt(weight@((pm-pp)**2)/measure))))
    for q in range(len(w)):
        profiles.append(dict(label=label,x_m=float(x[q]),y_m=float(k.s['y_m'][q]),weight_m=float(w[q]),
            physical_jump_m=float(SAVED['histories']['jump'][q]),physical_gamma=SAVED['gamma'],relaxed_affine_coefficient=SAVED['wall_memory'],
            memory_m=float((Q@c[1:])[q]),weak_shear_projection_Pa=float(shear[q]),raw_shear_Pa=float(raw[q]),
            normal_Pa=float(10000+normal[q]),raw_normal_minus_Pa=float(10000+pm[q]),raw_normal_plus_Pa=float(10000+pp[q])))
    split=c.copy();split[0]=0
    info=dict(label=label,**k.meta,bulk_J_m=float(.5*c@k.K@c),jump_induced_J_m=float(.5*split@k.K@split),
        background_J_m=float(.5*c[0]**2*k.K[0,0]),cross_J_m=float(c[0]*(k.K[0,1:]@c[1:])))
    return rows,profiles,info


CASES=[('narrow64',64,2,.1875,'cartesian'),('narrow96',96,2,.125,'cartesian'),
       ('fixed96',96,2,.1875,'cartesian'),('wide64',64,2,.75,'cartesian'),('wide96',96,2,.75,'cartesian'),
       ('wide-Q3',64,3,.75,'cartesian'),('fitted64',64,2,.75,'fitted'),('fitted96',96,2,.75,'fitted')]
EXTRA=[('fitted-Q3',64,3,.75,'fitted'),('offset96',96,2,.6875,'cartesian'),('narrow-Q3',64,3,.1875,'cartesian')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--identity',action='store_true');parser.add_argument('--case');args=parser.parse_args()
    if args.identity:identity();return
    rows=[];profiles=[];summaries=[]
    for label,n,degree,width,mesh in (CASES+EXTRA if args.case else CASES):
        if args.case and label!=args.case:continue
        print('start',label,flush=True);tic=time.perf_counter()
        k=build(label,n,degree,width,mesh);r,p,s=frozen(label,k);k.close()
        save(OUT/(label+'-summary.json'),s);table(OUT/(label+'-loads.csv'),r);table(OUT/(label+'-profiles.csv'),p)
        rows+=r;profiles+=p;summaries.append(s);print('done',label,round(time.perf_counter()-tic,2),s['jump_induced_J_m'],flush=True)
    if not args.case:
        table(OUT/'frozen-loads.csv',rows);table(OUT/'frozen-profiles.csv',profiles);table(OUT/'frozen-summary.csv',summaries)


if __name__=='__main__':main()
