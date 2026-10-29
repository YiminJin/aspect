#!/usr/bin/env python3
"""Common physical profiles; native means and compatible weak-force averages."""
import argparse
import numpy as np
from fracture_analysis import kernels,grid_value,save_csv
from continuation_analysis import BASE,OUT

def export(name):
    with np.load(BASE/(name+'-plot-data.npz')) as source:d={k:source[k] for k in source.files}
    x,y=d['x'],d['y'];xx,yy=np.meshgrid(x,y);X,Y=np.meshgrid(.5*(x[:-1]+x[1:]),.5*(y[:-1]+y[1:]));dx=x[1]-x[0];dy=y[1]-y[0]
    cx=np.linspace(.9,min(1.46875,x[-1]-.03125),147);centers=np.column_stack([cx,np.full_like(cx,.5)]);radius=.03125
    kc=kernels(np.column_stack([X.ravel(),Y.ravel()]),np.full(X.size,dx*dy),centers,radius)
    rows=[]
    for t in [.15875,.16875,.17375,.175,.17625,.17875]:
        k=np.argmin(abs(d['times']-t))
        if abs(d['times'][k]-t)>1e-10:continue
        f=d['forces'][k].reshape(-1,4);kn=kernels(np.column_stack([xx.ravel(),yy.ravel()]),f[:,0],centers,radius)
        phase=grid_value(x,y,d['phase'][k],cx,np.full_like(cx,.5));initial=grid_value(x,y,d['initial'],cx,np.full_like(cx,.5))
        stress=kc@d['stress'][k].ravel();drive=kc@d['Y'][k].ravel();rate=kc@d['rate'][k].ravel()
        fracture=kn@(f[:,1]/f[:,0]);mechanical=kn@(-f[:,2]/f[:,0]);balance=kn@(f[:,3]/f[:,0])
        assert np.max(abs(fracture-mechanical-balance))<1e-10
        rows.extend(zip(np.full(len(cx),d['times'][k]),cx,np.full_like(cx,.5),phase,initial,phase-initial,stress,drive,rate,fracture,mechanical,balance))
    save_csv(OUT/(name+'-profiles.csv'),['time_s','x_m','y_m','phi','initial_phi','phase_increment','kernel_tau_xy_Pa','kernel_Y_Pa','kernel_abs_w_s_inv','compatible_weak_fracture_Pa','compatible_weak_mechanical_Pa','compatible_weak_residual_Pa'],rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:export(name)
