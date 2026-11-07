#!/usr/bin/env python3
"""Independent real-tensor recurrence and pressure/velocity balance from provider exports."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,json
import numpy as np
from assess_layered_campaign import OUT,load_case
from layered_fields import read

def check(name):
    path,cmd,rows=load_case(name);provider=read(path/'output/layered-background.csv')
    assert len(rows)==len(provider)
    g,w=np.polynomial.legendre.leggauss(8);weight=np.tile(w*.03125,64);tau=np.zeros((512,3));gamma=np.zeros(512)
    maxima=dict(parent=0.,endpoint=0.,gamma=0.,q=0.,normal_balance=0.,velocity_difference=0.,clock=0.)
    for row,pr in zip(rows,provider):
        step=int(row['step']);dt=float(row['dt']) if step else .1
        data=read(path/'output'/f'background-{step}.csv');y=np.array([float(r['y']) for r in data]);eta=5.5+4.5*np.tanh(y/.35)
        field=lambda key:np.array([float(r[key]) for r in data])
        c=np.cos(-.5*gamma*dt);s=np.sin(-.5*gamma*dt);xx,yy,xy=tau.T
        rotated=np.stack([c*c*xx+s*s*yy-2*c*s*xy,s*s*xx+c*c*yy+2*c*s*xy,c*s*(xx-yy)+(c*c-s*s)*xy],axis=1)
        beta=np.exp(-10*dt/eta);kappa=-eta*np.expm1(-10*dt/eta);next_tau=beta[:,None]*rotated
        q=(2+weight@(next_tau[:,2]/kappa))/(weight@(1/kappa));next_gamma=(q-next_tau[:,2])/kappa;next_tau[:,2]=q
        maxima['parent']=max(maxima['parent'],float(np.max(abs(field('parent_xx')-tau[:,0]))),float(np.max(abs(field('parent_xy')-tau[:,2]))))
        maxima['endpoint']=max(maxima['endpoint'],float(np.max(abs(field('xx')-next_tau[:,0]))),float(np.max(abs(field('xy')-next_tau[:,2]))))
        maxima['gamma']=max(maxima['gamma'],float(np.max(abs(field('gamma')-next_gamma))))
        maxima['q']=max(maxima['q'],abs(float(pr['q'])-q))
        maxima['normal_balance']=max(maxima['normal_balance'],float(np.max(abs(-field('xx')-field('pressure')))))
        maxima['velocity_difference']=max(maxima['velocity_difference'],abs(weight@field('gamma')-2))
        maxima['clock']=max(maxima['clock'],abs(float(pr['time'])-float(row['time'])))
        gamma=next_gamma
        if step:tau=next_tau
    assert max(maxima.values())<2e-11,maxima
    result=dict(case=name,maxima=maxima,steps=len(rows)-1,zero_initial_stored_stress=True,actual_accepted_schedule=True)
    (OUT/('reference-'+name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');a=p.parse_args()
    for name in a.cases:check(name)
