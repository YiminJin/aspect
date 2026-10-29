#!/usr/bin/env python3
"""Remove the known added-volume response when comparing domain sizes."""
import argparse
import numpy as np
from continuation_analysis import BASE,OUT,read,save_csv
G=1e6;ETA=1e8

def parent_stress(time,dt):
    # Exact same unloaded zero histories. The temporal branches share the
    # original dt=.00125 accepted prefix through .15875, then their stated dt.
    tau=0.
    for _ in range(87):tau=(tau+G*.018*.00125)/(1+.00125*G/ETA)
    for _ in range(round((min(time,.175)-.15875)/dt)):tau=(tau+G*.018*dt)/(1+dt*G/ETA)
    for _ in range(max(0,round((time-.175)/dt))):tau/=1+dt*G/ETA
    return tau

def export(name):
    h=read(OUT/(name+'-history.csv'));d=np.load(BASE/(name+'-plot-data.npz'));width=float(d['x'][-1]-d['x'][0]);height=float(d['y'][-1]-d['y'][0]);volume=width*height
    tau=0. if h['dt_s'][0]==0 else parent_stress(h['time_s'][0]-h['dt_s'][0],h['dt_s'][0]);rows=[]
    for a in h:
        previous=tau;dt=a['dt_s'];rate=.018 if .05+1e-12<a['time_s']<=.175+1e-12 else 0.
        du=a['native_prescribed_increment_m'];tau=(tau+G*du/height)/(1+G*dt/ETA);reaction=tau*width;Wbe=reaction*du;Wtr=.5*(previous+tau)*width*du
        rows.append([a['time_s'],a['imposed_relative_displacement_m'],width,height,tau,reaction,a['shear_reaction_N_m'],a['shear_reaction_N_m']/width,a['shear_reaction_N_m']-reaction,volume*tau*tau/(2*G),a['bulk_J_m'],a['bulk_J_m']-volume*tau*tau/(2*G),Wbe,Wtr,volume*dt*tau*tau/ETA])
    save_csv(OUT/(name+'-background.csv'),['time_s','imposed_relative_displacement_m','width_m','height_m','homogeneous_intact_tau_Pa','homogeneous_grip_reaction_N_m','actual_grip_reaction_N_m','actual_reaction_per_width_Pa','seed_excess_reaction_N_m','homogeneous_bulk_J_m','actual_bulk_J_m','seed_excess_bulk_J_m','homogeneous_endpoint_work_J_m','homogeneous_trapezoidal_work_J_m','homogeneous_viscous_J_m'],rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:export(name)
