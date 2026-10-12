#!/usr/bin/env python3
"""Physical histories and separate storage/release/work diagnostics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time_resolution import E,load,Reference,G,h
from assess_time_resolution import trajectory,stress_fields

def save(fig,name):
    fig.savefig(E/(name+'.png'),dpi=170);fig.savefig(E/(name+'.pdf'));plt.close(fig)

def main():
    ds=[64,128,256,512,1024];paths={d:trajectory(d) for d in ds};r=Reference(256,1.,24)
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for d,a in paths.items():
        t=np.arange(a['steps']+1)/d;label=f'dt=1/{d}'
        ax[0,0].plot(t,np.max(a['phi'],axis=1),label=label);ax[0,1].plot(t,a['q'],label=label);ax[0,2].plot(t,a['slip'],label=label)
        for j,time in enumerate([.546875,.625]):ax[1,j].plot(r.y,a['phi'][round(time*d)],label=label)
        indices=np.arange(41)*(d//64);rate=np.diff(a['slip'][indices])*64
        ax[1,2].stairs(rate,np.arange(41)/64,label=label)
    for j,(x,y) in enumerate([('time [s]','maximum phase'),('time [s]','cohesive traction [Pa]'),('time [s]','cumulative slip [m]')]):ax[0,j].set(xlabel=x,ylabel=y);ax[0,j].legend(fontsize=8)
    ax[0,0].set_xlim(.515625,.625)
    for j,time in enumerate([.546875,.625]):ax[1,j].set(xlabel='y [m]',ylabel='phase',title=f't={time:g} s');ax[1,j].legend(fontsize=8)
    ax[1,2].set(xlabel='interval [s]',ylabel='common-interval mean slip rate [m/s]');ax[1,2].legend(fontsize=8);save(fig,'physical-histories')
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for d,a in paths.items():
        t=np.arange(a['steps']+1)/d;label=f'dt=1/{d}'
        for j,name in enumerate(['bulk','cohesive','fracture']):ax[0,j].plot(t,a[name]-a[name][0],label=label)
        for j,name in enumerate(['D_relax','bulk_work_error','boundary_work_error']):ax[1,j].plot(t,np.r_[0.,np.cumsum([v[name] for v in a['energy']])],label=label)
    for j,name in enumerate(['bulk','cohesive','fracture']):ax[0,j].set(xlabel='time [s]',ylabel=f'{name} storage change [J/m]');ax[0,j].legend(fontsize=8)
    for j,name in enumerate(['phase-relaxation release','bulk-rate work residual','boundary-lifting work residual']):ax[1,j].set(xlabel='time [s]',ylabel=f'cumulative {name} [J/m]');ax[1,j].legend(fontsize=8)
    save(fig,'energy-trajectories')
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for d,a in paths.items():
        label=f'dt=1/{d}'
        for j,t in enumerate([.546875,.625]):
            stress,_=stress_fields(a,[round(t*d)],r);ax[j].plot(r.yq,stress[0],label=label)
        old=E/f'frozen-dt-1-{d}/summary.json'
        if old.exists():
            v=load(old)['energy_totals'];ax[2].scatter([1/d],[v['bulk_work_error']['absolute']],color='k',marker='x')
    for j,t in enumerate([.546875,.625]):ax[j].set(xlabel='y [m]',ylabel='shear stress [Pa]',title=f't={t:g} s');ax[j].legend(fontsize=8)
    for name,label in [('D_relax','evolving phase release'),('bulk_work_error','evolving bulk-rate work'),('boundary_work_error','evolving boundary work')]:ax[2].loglog([1/d for d in ds],[sum(abs(v[name]) for v in paths[d]['energy']) for d in ds],'o-',label=label)
    ax[2].scatter([],[],color='k',marker='x',label='frozen work error');ax[2].set(xlabel='dt [s]',ylabel='sum of absolute step amounts [J/m]');ax[2].legend(fontsize=8);save(fig,'stress-and-work')

if __name__=='__main__':main()
