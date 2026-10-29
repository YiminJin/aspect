#!/usr/bin/env python3
"""Physical work/storage and discrete remainders, with explicit branch labels."""
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from continuation_analysis import BASE,OUT,read
from continuation_plots import LABELS,COLORS

def main():
    names=['reference-continuation','half-continuation','quarter-continuation'];fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');parent=read(OUT/'original-history.csv')[127]
    for j,name in enumerate(names):
        h=read(OUT/(name+'-history.csv'));h=h[(h['time_s']>.15875+1e-12)&(h['time_s']<=.17875+1e-12)];label=LABELS[name];color=COLORS[j]
        axs[0,0].plot(1000*h['imposed_relative_displacement_m'],h['shear_reaction_N_m'],'.-',color=color,label=label,ms=2)
        axs[0,1].plot(h['time_s'],np.cumsum(h['trapezoidal_work_J_m']),color=color,label=label)
        axs[0,2].plot(h['time_s'],np.cumsum(h['trapezoidal_remainder_J_m']),color=color,label=label)
    h=read(OUT/'quarter-continuation-history.csv');h=h[(h['time_s']>.15875+1e-12)&(h['time_s']<=.17875+1e-12)]
    for key,label in [('bulk_J_m','Bulk'),('cohesive_J_m','Cohesive'),('fracture_J_m','Fracture')]:axs[1,0].plot(h['time_s'],h[key]-parent[key],label=label)
    for key,label in [('Db_J_m','Bulk viscosity'),('Dc_J_m','Cohesive viscosity'),('friction_J_m','Residual friction')]:axs[1,1].plot(h['time_s'],np.cumsum(h[key]),label=label)
    for key,label in [('stress_loss_J_m','Stress increment'),('traction_loss_J_m','Traction increment'),('phase_remainder_J_m','Signed compliance remainder'),('fracture_loss_J_m','Fracture gradient increment'),('phase_work_J_m','Phase residual work'),('mechanical_work_J_m','Mechanical residual work')]:axs[1,2].plot(h['time_s'],np.cumsum(h[key]),label=label)
    labels=[('Imposed relative grip displacement [mm]','Grip reaction [N/m]'),('Time [s]','Cumulative trapezoidal work [J/m]'),('Time [s]','Cumulative physical work remainder [J/m]'),('Time [s]','Storage change since .15875 s [J/m]'),('Time [s]','Cumulative physical dissipation [J/m]'),('Time [s]','Cumulative BE/residual terms [J/m]')]
    for ax,(x,y) in zip(axs.flat,labels):ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in [axs[0,1],axs[0,2],*axs[1]]:ax.axvline(.175,color='.6',ls=':')
    for ax in [axs[0,1],axs[0,2],*axs[1]]:
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[1,0].set_title('Quarter step: physical storage');axs[1,1].set_title('Quarter step: physical losses');axs[1,2].set_title('Quarter step: separate numerical/residual terms')
    fig.savefig(OUT/'physical-energy-work.png',dpi=150);plt.close(fig)
    dt=np.array([.00125,.000625,.0003125]);load=[];hold=[];frozen=[];increment=[];potential=[]
    for i,(lp,hp,cp) in enumerate([('loaded-stability','predictor-hold-stability','continuation-hold-stability'),('half-loaded-stability','half-predictor-hold-stability','half-continuation-hold-stability'),('quarter-loaded-stability','quarter-predictor-hold-stability','quarter-continuation-hold-stability')]):
        load.append(json.load(open(OUT/(lp+'.json')))['interior']['eigenvalues'][0]);r=json.load(open(OUT/(hp+'.json')))['interior'];hold.append(r['eigenvalues'][0]);frozen.append(r['frozen_mechanics_minimum_eigenvalue'])
        prefix=['reference-hold-critical','half-hold','quarter-hold'][i];continuation=['reference-relax','half-relax','quarter-relax'][i];a=read(OUT/(prefix+'-history.csv'))[0];b=read(OUT/(continuation+'-history.csv'))[0];increment.append(a['max_step_phase_increment']);potential.append(1e6*(a['incremental_potential_J_m']-b['incremental_potential_J_m']))
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained');xx=dt*1000
    axs[0].plot(xx,load,'o-',label='Loaded, mechanics relaxed');axs[0].plot(xx,hold,'o-',label='Predictor hold, mechanics relaxed');axs[0].plot(xx,frozen,'s--',label='Predictor hold, mechanics frozen');axs[0].axhline(0,color='.5');axs[0].set_ylabel('Minimum normalized curvature');axs[0].legend(fontsize=8)
    axs[1].plot(xx,increment,'o-',label='Unstable predictor root');axs[1].plot(xx,np.zeros(3),'s-',label='Stable continuation');axs[1].set_ylabel('First-hold phase increment');axs[1].legend(fontsize=8)
    axs[2].plot(xx,potential,'o-');axs[2].set_ylabel('Predictor minus continuation potential [micro J/m]')
    for ax in axs:ax.set_xlabel(r'$\Delta t$ [ms]');ax.grid(alpha=.2);ax.invert_xaxis()
    fig.savefig(OUT/'stability-and-branch.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
