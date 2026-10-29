#!/usr/bin/env python3
"""Figures only from exported, unit-labelled reference tables."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-cohesive-interface')
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.collections import LineCollection
from cohesive_interface import OUT,ELL,DC


def read(name):
    with (OUT/name).open() as f:
        return [{k:float(v) if k!='mode' else v for k,v in r.items()} for r in csv.DictReader(f)]
def values(rows,key):return np.array([r[key] for r in rows])
def finish(fig,name,title):
    fig.suptitle(title,fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/name,dpi=160);plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    point=read('point_history.csv');summary=json.loads((OUT/'point_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for dt,color in zip([.002,.001,.0005],['.6','#e39826','#176ca4']):
        r=[x for x in point if x['mode']=='formation' and x['dt_s']==dt];t=values(r,'time_s')
        onset=next(v['onset'] for v in summary['runs'] if v['mode']=='formation' and v['dt_s']==dt)
        axes[0,0].plot(onset,[dt,dt],color=color,lw=5,label=f'Δt={dt:g} s')
        axes[0,1].plot(1e3*values(r,'jump_m'),values(r,'shear_Pa')/1e3,color=color)
    axes[0,0].axvline(summary['analytic_onset_s'],color='k',ls=':',label='analytic intact threshold')
    axes[0,0].set(xlim=(.3738,.3762),ylim=(.00025,.0023),xlabel='First-activation bracket (s)',ylabel='Δt (s)');axes[0,0].ticklabel_format(useOffset=False,axis='x');axes[0,0].legend(fontsize=8,loc='center left')
    axes[0,1].set(xlabel='Geometric jump δ (mm)',ylabel='Physical shear traction (kPa)')
    r=[x for x in point if x['mode']=='formation' and x['dt_s']==.0005];t=values(r,'time_s')
    for key,label in [('cohesive_J_m2','Recoverable bond'),('fracture_increment_J_m2','Fracture inventory'),('friction_J_m2','Friction loss'),('radiation_J_m2','Radiation loss')]:axes[1,0].plot(t,values(r,key),label=label)
    axes[1,0].set(xlabel='Time (s)',ylabel='Energy / area (J/m²)');axes[1,0].legend(fontsize=8)
    runs=[r for r in summary['runs'] if r['mode']=='formation']
    axes[1,1].loglog([r['dt_s'] for r in runs],[r['absolute_remainders_over_scale'] for r in runs],'o-',label='sum |physical work remainder| / Gc')
    axes[1,1].set(xlabel='Δt (s)',ylabel='Dimensionless error');axes[1,1].legend(fontsize=8)
    finish(fig,'intact_formation.png','Exact intact admission: fracture compliance is separate from contact sliding')
    fig,axes=plt.subplots(2,2,figsize=(11,7));r=[x for x in point if x['mode']=='mature' and x['dt_s']==.001];t=values(r,'time_s')
    axes[0,0].plot(t,values(r,'jump_m')/DC,label='signed slip / Dc');axes[0,0].set(xlabel='Time (s)',ylabel='Geometric slip / Dc');axes[0,0].legend()
    axes[0,1].plot(t,values(r,'shear_Pa')/1e3,label='fully mature interface');axes[0,1].set(xlabel='Time (s)',ylabel='Physical shear traction (kPa)')
    axes[1,0].semilogy(t,values(r,'Theta'),label='one interface age Θ');axes[1,0].set(xlabel='Time (s)',ylabel='Dimensionless state Θ');axes[1,0].legend()
    for key,label in [('bulk_J_m2','Bulk storage'),('cohesive_J_m2','Bond storage (exactly zero)'),('friction_J_m2','Friction loss'),('radiation_J_m2','Radiation loss'),('Maxwell_J_m2','Maxwell loss')]:axes[1,1].plot(t,values(r,key),label=label)
    axes[1,1].set(xlabel='Time (s)',ylabel='Energy / area (J/m²)');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        for stop in [2,4,4.5]:ax.axvline(stop,color='.65',lw=.7,ls=':')
    finish(fig,'mature_interface.png','Mature RSF identity: velocity step, hold, reversal; no residual bond hardening')
    fig,axes=plt.subplots(2,2,figsize=(11,7));profiles={}
    for n,color in [(128,'.6'),(256,'#c28125'),(512,'#175fa6')]:
        dt=.002 if n==128 else .001;r=read(f'tip_profiles_n{n}_dt{dt:g}.csv');profiles[n]=r
        end=[x for x in r if abs(x['time_s']-.32)<1e-12];x=values(end,'x_m');new=values(end,'damage')-values(end,'initial_damage')
        axes[0,0].plot(x,new,label=f'N={n}, Δt={dt:g} s',color=color)
        axes[0,1].plot(x,values(end,'physical_shear_Pa')/1e3,color=color,label=f'N={n}')
        axes[1,0].plot(x,values(end,'normal_pressure_Pa')/1e3,color=color)
        axes[1,1].plot(x,values(end,'jump_m')*1e3,color=color)
    for ax,label in zip(axes.flat,['New surface damage Δa','Physical shear traction (kPa)','Actual normal pressure pn (kPa)','Geometric jump δ (mm)']):
        ax.set(xlim=(-.26,.26),xlabel='Along-fault x (m)',ylabel=label)
        for tip in [-.125,.125]:ax.axvline(tip,color='k',lw=.8,ls=':')
        ax.axvspan(-.125,.125,color='grey',alpha=.08)
    axes[0,0].legend();finish(fig,'seeded_tip_profiles.png','At 0.32 s: new cohesive fronts; shaded interval is the original mature seed')
    r=read('tip_history_n512_dt0.001.csv');t=values(r,'time_s');fig,axes=plt.subplots(2,2,figsize=(11,7))
    for level in [.01,.05,.5,.95]:axes[0,0].plot(t,values(r,f'right_a{level:g}_m'),label=f'a={level:g}')
    axes[0,0].axhline(.125,color='k',ls=':');axes[0,0].set(xlabel='Time (s)',ylabel='Right seed-connected front (m)');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(t,values(r,'integrated_new_damage_m')*1e3);axes[0,1].set(xlabel='Time (s)',ylabel='∫ new a ds (mm)')
    for key,label in [('bulk_J_m','Bulk storage'),('cohesive_J_m','Bond storage'),('fracture_increment_J_m','Added fracture'),('friction_J_m','Friction loss'),('radiation_J_m','Radiation loss')]:axes[1,0].plot(t,values(r,key),label=label)
    axes[1,0].set(xlabel='Time (s)',ylabel='Energy / out-of-plane length (J/m)');axes[1,0].legend(fontsize=8)
    for n,dt,color in [(128,.002,'.6'),(256,.002,'#e49d32'),(256,.001,'#ae4b34'),(512,.001,'#175fa6')]:
        rr=read(f'tip_history_n{n}_dt{dt:g}.csv');axes[1,1].plot(values(rr,'time_s'),values(rr,'integrated_new_damage_m')*1e3,label=f'N={n}, Δt={dt:g}',color=color)
    axes[1,1].set(xlim=(.18,.32),xlabel='Time (s)',ylabel='∫ new a ds (mm)');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.axvline(.30,color='.6',ls=':')
    finish(fig,'seeded_tip_history.png','Finite periodic specimen: load to 0.30 s, hold to 0.32 s; initial fracture 0.25 J/m excluded')
    fig,axes=plt.subplots(1,2,figsize=(10,4));r=read('mature_mixed_symbol.csv')
    axes[0].loglog(values(r,'wavelength_m'),-values(r,'growth_max_s_inv'),label='slow branch');axes[0].loglog(values(r,'wavelength_m'),-values(r,'growth_min_s_inv'),label='fast branch')
    axes[0].set(xlabel='Along-interface wavelength (m)',ylabel='Decay rate −Re(λ) (1/s)');axes[0].legend()
    axes[1].plot(values(r,'wavelength_m'),values(r,'normal_symbol_imag_Pa_m')/1e6);axes[1].set(xlabel='Along-interface wavelength (m)',ylabel='Im normal-pressure symbol (MPa/m)')
    finish(fig,'mixed_interface_symbol.png','Mature fixed-plane symbol: actual pressure and state retained, V=0.001 m/s')
    # Compact animation, all scales held fixed; no invented bulk phase field.
    rr=profiles[512];times=sorted(set(r['time_s'] for r in rr));fig,axes=plt.subplots(3,1,figsize=(10,7),gridspec_kw={'height_ratios':[.8,1,1]})
    def frame(k):
        for ax in axes:ax.clear()
        t=times[k];r=[x for x in rr if x['time_s']==t];x=values(r,'x_m');a=values(r,'damage')
        axes[0].fill_between([-1,1],0,.075,color='#dce6ee');axes[0].fill_between([-1,1],-.05,0,color='#e4e8df')
        lines=np.stack([np.column_stack([x-1/512,np.zeros_like(x)]),np.column_stack([x+1/512,np.zeros_like(x)])],axis=1)
        lc=LineCollection(lines,cmap='magma',norm=plt.Normalize(0,1),linewidth=4);lc.set_array(a);axes[0].add_collection(lc)
        axes[0].set(xlim=(-1,1),ylim=(-.055,.08),ylabel='y (m)',aspect='equal');axes[0].text(-.95,.03,'Clamped upper wall; periodic in x',fontsize=8)
        axes[1].plot(x,a-values(r,'initial_damage'),color='#ad2938');axes[1].set(xlim=(-.25,.25),ylim=(0,.07),ylabel='New surface damage Δa')
        axes[2].plot(x,values(r,'physical_shear_Pa')/1e3,label='physical shear');axes[2].plot(x,(values(r,'normal_pressure_Pa')-10000)/1e3,label='pn − p0');axes[2].set(xlim=(-.25,.25),ylim=(-1.1,3.3),ylabel='Traction (kPa)',xlabel='x (m)');axes[2].legend(loc='lower right',fontsize=8)
        for ax in axes:
            for tip in [-.125,.125]:ax.axvline(tip,color='k',ls=':',lw=.8)
        fig.suptitle(f't={t:.3f} s; imposed relative displacement={min(t,.30):.3f} mm\nSurface damage a, not a reconstructed bulk phase; dotted lines = original seed tips',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.92))
    animation=FuncAnimation(fig,frame,frames=len(times),interval=180);animation.save(OUT/'seeded_tip.gif',writer=PillowWriter(fps=6));plt.close(fig)

if __name__=='__main__':main()
