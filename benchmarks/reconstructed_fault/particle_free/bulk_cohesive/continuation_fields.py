#!/usr/bin/env python3
"""Selected native field exports with nodal and cell-mean measures kept apart."""
import argparse
import numpy as np
from continuation_analysis import BASE, OUT, save_csv


def export(name):
    with np.load(BASE/(name+'-plot-data.npz')) as z:
        d={k:z[k] for k in z.files}
    destination=OUT/'fields';destination.mkdir(exist_ok=True)
    X,Y=np.meshgrid(d['x'],d['y'])
    CX,CY=np.meshgrid((d['x'][:-1]+d['x'][1:])/2,(d['y'][:-1]+d['y'][1:])/2)
    for t in [.15875,.175,.17625,.17875]:
        k=np.argmin(abs(d['times']-t))
        if abs(d['times'][k]-t)>1e-10:
            continue
        mass,F,D,g=d['forces'][k].transpose(2,0,1)
        nodal=[X,Y,d['initial'],d['phase'][k],d['growth'][k],d['delta'][k],mass,F/mass,-D/mass,g/mass]
        cell=[CX,CY,d['tauxx'][k],d['tauyy'][k],d['stress'][k],d['Y'][k],d['rate'][k]]
        stem=name+f'-{t:.5f}'
        save_csv(destination/(stem+'-nodes.csv'),['x_m','y_m','initial_phi','phi','phase_growth','accepted_step_phase_change','lumped_mass_m2','fracture_force_density_Pa','mechanical_force_density_Pa','total_weak_residual_density_Pa'],zip(*(a.ravel() for a in nodal)))
        save_csv(destination/(stem+'-cells.csv'),['x_m','y_m','mean_tau_xx_Pa','mean_tau_yy_Pa','mean_tau_xy_Pa','mean_Y_Pa','mean_abs_w_s_inv'],zip(*(a.ravel() for a in cell)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:export(name)
