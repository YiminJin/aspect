#!/usr/bin/env python3
"""Physical surface histories, region classification and one bounded animation."""
import os,json,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-front')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.animation import FuncAnimation,PillowWriter
from analyze_cohesive_interface_front import OUT,load,CASES

def save(fig,name,title):
    fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/name,dpi=160);plt.close(fig)

def metrics():
    with (OUT/'physical_histories.csv').open() as f:
        return [{k:(v if k=='case' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
def col(r,k):return np.array([a[k] for a in r])

def main():
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    h,q,w,x,n=load('growth-64-half');times=h['time_s'];a=q['damage'];initial=((x>.375)&(x<.625)).astype(float)
    regions=np.where(a==0,0,np.where(a==1,2,1));colors=['#e5e5e5','#e99b38','#4e3d80'];cmap=ListedColormap(colors)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    im=axes[0].pcolormesh(x,times,regions,cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),shading='nearest',rasterized=True)
    cb=fig.colorbar(im,ax=axes[0],ticks=[0,1,2]);cb.ax.set_yticklabels(['tied','cohesive','decohered'])
    levels=[.01,.05,.1,.5,.9];masked=np.ma.array(a,mask=np.broadcast_to(initial==1,a.shape));cs=axes[0].contour(x,times,masked,levels=levels,colors='k',linewidths=.5);axes[0].clabel(cs,fmt='%g',fontsize=8,manual=[(.15,.37),(.15,.39),(.15,.415),(.15,.6),(.15,.79)])
    for ax,key,label,cm in [(axes[1],'jump_m','Geometric jump (mm)','viridis'),(axes[2],'shear_Pa','Weak physical shear (kPa)','coolwarm')]:
        factor=1000 if key=='jump_m' else .001;im=ax.pcolormesh(x,times,q[key]*factor,shading='nearest',cmap=cm);fig.colorbar(im,ax=ax,label=label)
    for ax in axes:
        ax.axvline(.375,color='k',ls='--',lw=.7);ax.axvline(.625,color='k',ls='--',lw=.7);ax.axhline(.84,color='k',ls=':',lw=.8);ax.set(ylabel='Time (s)',xlim=(0,1),ylim=(0,.86))
    axes[-1].set_xlabel('Along fixed candidate Γ, x (m)')
    save(fig,'space_time.png','Fixed-surface growth: early cohesive-zone expansion, late full decohesion; dashed lines = initial seed')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for t in [.24,.3,.36,.4,.6,.8,.824,.826]:
        k=int(np.argmin(abs(times-t)));label=f'{times[k]:.3f} s'
        axes[0,0].plot(x,a[k],label=label);axes[0,1].plot(x,q['jump_m'][k]*1000,label=label)
        axes[1,0].plot(x,q['shear_Pa'][k]/1000,label=label)
    axes[0,0].plot(x,initial,'k--',lw=1,label='initial a');axes[0,0].set(ylabel='Surface damage a',ylim=(-.02,1.04));axes[0,0].legend(fontsize=7,ncol=3)
    axes[0,1].set(ylabel='Geometric jump (mm)');axes[1,0].set(ylabel='Weak physical shear (kPa)')
    k=int(np.argmin(abs(times-.36)));damaged=a[k]>0
    axes[1,1].plot(x,q['shear_Pa'][k]/1000,label='physical weak traction')
    axes[1,1].plot(x[damaged],q['q_Pa'][k,damaged]/1000,'.',label='bond q, damaged')
    axes[1,1].plot(x[~damaged],q['q_Pa'][k,~damaged]/1000,'.',color='grey',label='one admissible tied multiplier')
    axes[1,1].set(ylabel='Traction at 0.360 s (kPa)');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(.375,color='.3',ls=':',lw=.7);ax.axvline(.625,color='.3',ls=':',lw=.7);ax.set(xlim=(0,1),xlabel='x (m)')
    save(fig,'profiles.png','Damage, slip and traction are different fields; tied Gauss multipliers need not be unique')
    hist=metrics();fine=[r for r in hist if r['case']=='growth-64-half'];fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name,label in zip(CASES,['32, Δt=.002 s','64, Δt=.002 s','64, Δt=.001 s']):
        r=[v for v in hist if v['case']==name]
        axes[0,0].plot(col(r,'time_s'),col(r,'integrated_growth_m'),label=label)
        axes[0,1].plot(col(r,'time_s'),col(r,'new_decohered_length_m'),label=label)
    axes[0,0].set(xlabel='Time (s)',ylabel='Integrated new damage (m)');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Time (s)',ylabel='New a=1 length (m)',xlim=(.82,.83));axes[0,1].legend(fontsize=8)
    for v in ['0p05','0p5','1p0']:
        axes[1,0].plot(col(fine,'U_m')*1000,col(fine,'right_'+v+'_m'),label='right a='+v.replace('p','.'))
        axes[1,0].plot(col(fine,'U_m')*1000,col(fine,'left_'+v+'_m'),'--',label='left a='+v.replace('p','.'))
    axes[1,0].set(xlabel='Imposed displacement U (mm)',ylabel='Seed-connected contour x (m)');axes[1,0].legend(fontsize=7,ncol=2)
    for key,label in [('min_curvature_Pa_m','admissible tangent'),('release_curvature_Pa_m','one-sided release tangent')]:axes[1,1].plot(col(fine,'time_s'),col(fine,key)/1e6,label=label)
    axes[1,1].set(xlabel='Time (s)',ylabel='Minimum curvature (MPa/m)',yscale='log');axes[1,1].legend(fontsize=8)
    save(fig,'comparison.png','Spatial onset is sensitive; later integrated growth agrees; full-decohesion timing is bracketed')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(col(fine,'U_m')*1000,col(fine,'reaction_Pa')/1000);axes[0,0].set(xlabel='Imposed displacement U (mm)',ylabel='Wall shear reaction (kPa)')
    for key,label in [('bulk_J_m','bulk storage'),('bond_J_m','bond storage'),('added_fracture_J_m','new fracture'),('physical_loss_J_m','cumulative physical loss')]:axes[0,1].plot(col(fine,'time_s'),col(fine,key),label=label)
    axes[0,1].set(xlabel='Time (s)',ylabel='Energy per thickness (J/m)');axes[0,1].legend(fontsize=8)
    for name in ['growth-64','growth-64-half']:
        r=[v for v in hist if v['case']==name];axes[1,0].plot(col(r,'time_s'),col(r,'absolute_physical_remainder_J_m')*1000,label=name)
    axes[1,0].set(xlabel='Time (s)',ylabel='Σ |physical work remainder| (mJ/m)');axes[1,0].legend(fontsize=8)
    for key in ['Maxwell_J_m','friction_J_m','radiation_J_m','bulk_BE_J_m','surface_remainder_J_m']:
        axes[1,1].plot(times,np.cumsum(h[key]),label=key.replace('_J_m',''))
    axes[1,1].set(xlabel='Time (s)',ylabel='Cumulative term (J/m)');axes[1,1].legend(fontsize=8)
    save(fig,'energy.png','Initial fracture inventory = 0.25 J/m; new fracture counted once; BE remainders are not dissipation')
    # Fewer frames early, all saved steps around the maturity transition.
    frame_times=np.unique(np.r_[np.linspace(0,.8,48),np.arange(.801,.832,.001),[.84,.85,.86]])
    frames=np.unique([int(np.argmin(abs(times-t))) for t in frame_times])
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    def update(k):
        for ax in axes:ax.clear();ax.grid(alpha=.2);ax.axvline(.375,color='k',ls=':',lw=.8);ax.axvline(.625,color='k',ls=':',lw=.8);ax.set_xlim(0,1)
        axes[0].plot(x,a[k],color='k',lw=1)
        for value,color,label in [(0,colors[0],'tied'),(1,colors[1],'cohesive'),(2,colors[2],'fully decohered')]:axes[0].scatter(x[regions[k]==value],a[k,regions[k]==value],s=12,color=color,label=label,edgecolors='.4',linewidths=.2)
        for level in [.05,.5,.9]:axes[0].axhline(level,color='.5',ls=':',lw=.5)
        axes[0].set(ylim=(-.04,1.06),ylabel='Surface a');axes[0].legend(loc='upper left',fontsize=8,ncol=3)
        axes[1].plot(x,q['jump_m'][k]*1000,color='tab:blue');axes[1].set(ylim=(-.02,.76),ylabel='Jump (mm)')
        axes[2].plot(x,q['shear_Pa'][k]/1000,label='physical weak shear');axes[2].plot(x[a[k]>0],q['q_Pa'][k,a[k]>0]/1000,'.',label='bond q (damaged)');axes[2].set(ylim=(-.1,3.5),ylabel='Traction (kPa)',xlabel='Along fixed Γ, x (m)');axes[2].legend(loc='upper right',fontsize=8)
        fig.suptitle(f"t={times[k]:.3f} s, U={.001*min(times[k],.84)*1000:.3f} mm — "+('loading' if times[k]<=.84 else 'hold')+'\nFixed candidate line; no bulk φ evolution or new geometry',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.93))
    animation=FuncAnimation(fig,update,frames=frames,interval=120);animation.save(OUT/'front_evolution.gif',writer=PillowWriter(fps=8),dpi=100);plt.close(fig)

if __name__=='__main__':main()
