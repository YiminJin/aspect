#!/usr/bin/env python3
"""PNG/GIF presentation of accepted P4 states, with physical axes/units."""
from pathlib import Path
import argparse,functools
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from continuation_analysis import ROOT,BASE,OUT,read,LEVELS,displacement
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
LABELS={'reference-continuation':r'$\Delta t=0.00125$ s, continuation','quarter-continuation':r'$\Delta t=0.0003125$ s, continuation','original':r'$\Delta t=0.00125$ s, predictor','half-complete':r'$\Delta t=0.000625$ s, predictor','half-continuation':r'$\Delta t=0.000625$ s, continuation','quarter-complete':r'$\Delta t=0.0003125$ s, predictor'}
COLORS=['#222222','#0072b2','#d55e00','#009e73','#cc79a7']

@functools.lru_cache(None)
def load(name):
    with np.load(BASE/(name+'-plot-data.npz')) as a:return {k:a[k] for k in a.files}
def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def curves(names):
    fig,axs=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for j,name in enumerate(names):
        c=read(OUT/(name+'-contours.csv'));color=COLORS[j]
        for row,levels in enumerate(((.001,.01),(.05,.1),(.3,.5))):
            for k,lev in enumerate(levels):
                a=c[(c['ray_angle_deg']==0)&(c['phi_contour']==lev)&(c['time_s']>=.15875-1e-10)&(c['time_s']<=.17875+1e-10)]
                label=LABELS.get(name,name)+f', phi={lev:g}'
                axs[row,0].plot(a['time_s'],a['x_m'],label=label,color=color,ls='-' if k else '--',marker='.',ms=2)
                axs[row,1].plot(1000*a['imposed_relative_displacement_m'],a['x_m'],label=label,color=color,ls='-' if k else '--',marker='.',ms=2)
    for ax in axs[:,0]:ax.set_xlabel('Time [s]');ax.axvline(.175,color='.6',ls=':');ax.set_xlim(.1585,.179)
    for ax in axs[:,1]:ax.set_xlabel('Imposed relative grip displacement [mm]')
    for ax in axs.flat:ax.set_ylabel('Seed-connected centerline first exit x [m]');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'front-continuation')
    fig,axs=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    fields=[('max_step_phase_increment','Maximum phase change per accepted step'),('integrated_abs_w_m2_s','Integral of |w| [m2/s]'),('shear_reaction_N_m','Shear reaction [N/m]')]
    for j,name in enumerate(names):
        h=read(OUT/(name+'-history.csv'));h=h[(h['time_s']>.15875+1e-12)&(h['time_s']<=.17875+1e-10)];color=COLORS[j]
        for ax,(field,label) in zip(axs.flat,fields):ax.plot(h['time_s'],h[field],'.-',color=color,label=LABELS.get(name,name),ms=3);ax.set_ylabel(label)
        axs[1,1].plot(h['time_s'],np.cumsum(h['trapezoidal_work_J_m']-h['energy_change_J_m']-h['Db_J_m']-h['Dc_J_m']-h['friction_J_m']),color=color,label=LABELS.get(name,name));axs[1,1].set_ylabel('Cumulative physical work remainder [J/m]')
        axs[2,0].plot(h['time_s'],np.cumsum(h['stress_loss_J_m']+h['traction_loss_J_m']+h['phase_remainder_J_m']+h['fracture_loss_J_m']),color=color,label=LABELS.get(name,name));axs[2,0].set_ylabel('Cumulative BE numerical remainder [J/m]')
        axs[2,1].plot(h['time_s'],h['fracture_J_m']-read(OUT/'original-history.csv')['fracture_J_m'][127],color=color,label=LABELS.get(name,name));axs[2,1].set_ylabel('Fracture storage change since checkpoint [J/m]')
    for ax in axs.flat:ax.axvline(.175,color='.6',ls='--');ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'response-and-energy')

def maps(names):
    data=[load(n) for n in names];t=.17875
    stressmax=max(float(d['stress'][np.argmin(abs(d['times']-t))].max()) for d in data)
    stressmin=min(0.,min(float(d['stress'][np.argmin(abs(d['times']-t))].min()) for d in data))
    fig,axs=plt.subplots(2,len(names),figsize=(5*len(names),7),squeeze=False,layout='constrained')
    growthmax=max(float(d['growth'][np.argmin(abs(d['times']-t))].max()) for d in data)
    for j,(name,d) in enumerate(zip(names,data)):
        i=np.argmin(abs(d['times']-t));x,y=d['x'],d['y']
        for row,key,maximum,cmap,label in [(0,'growth',growthmax,'magma','Phase increment from continuum initialization'),(1,'stress',stressmax,'viridis','Physical shear stress tau_xy [Pa]')]:
            ax=axs[row,j];value=d[key][i];m=ax.pcolormesh(x,y,value,shading=('gouraud' if key=='growth' else 'flat'),vmin=(0 if key=='growth' else stressmin),vmax=maximum,cmap=cmap)
            ax.contour(x,y,d['phase'][i],levels=[.01,.1,.3,.5],colors='white',linewidths=.6);ax.plot([0,1],[.5,.5],color='cyan',lw=1);ax.plot(1,.5,'co',ms=3)
            ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(name+f'; t={d["times"][i]:.5f} s');fig.colorbar(m,ax=ax,shrink=.8,label=label)
    save(fig,'common-scale-fields')

def movie(names):
    ds=[load(n) for n in names];times=(np.r_[.15875,ds[0]['times'][(ds[0]['times']>.15875+1e-12)&(ds[0]['times']<=.17875+1e-12)]] if len(names)==1 else np.arange(.15875,.1787501,.00125));fig,axs=plt.subplots(3,len(names),figsize=(5*len(names),9),squeeze=False,layout='constrained');m=[];contours=[]
    stressmax=max(float(d['stress'].max()) for d in ds);stressmin=min(0.,min(float(d['stress'].min()) for d in ds));maximum=max(float(d['growth'].max()) for d in ds);wmax=max(float(d['rate'].max()) for d in ds)
    for row,(key,lo,hi,cmap,label) in enumerate([('growth',0,maximum,'magma','Phase minus fresh represented seed'),('stress',stressmin,stressmax,'viridis','Physical tau_xy [Pa]'),('rate',0,wmax,'plasma','Native cell mean |w| [1/s]')]):
        for j,(name,d) in enumerate(zip(names,ds)):
            ax=axs[row,j];im=ax.pcolormesh(d['x'],d['y'],d[key][0],shading=('gouraud' if key=='growth' else 'flat'),cmap=cmap,vmin=lo,vmax=hi);m.append((im,key,d));ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(LABELS.get(name,name));ax.plot([0,1],[.5,.5],color='cyan',lw=1);fig.colorbar(im,ax=ax,shrink=.75,label=label)
    def frame(k):
        nonlocal contours
        parent=load('original')
        for c in contours:c.remove()
        contours=[];t=times[k]
        for ax,(im,key,d) in zip(axs.flat,m):
            i=np.argmin(abs(d['times']-t));used=d
            if t<d['times'][0]-1e-12:used=parent;i=np.argmin(abs(parent['times']-t))
            im.set_array(used[key][i]);contours.append(ax.contour(used['x'],used['y'],used['phase'][i],levels=[.01,.1,.3,.5],colors='white',linewidths=.5))
        note='Shared parent; rate uses preceding '+r'$\Delta t=.00125$ s' if k==0 else 'Native accepted frame; loading ends at .175 s'
        fig.suptitle(f'P4: t={t:.5f} s; relative grip displacement={1000*displacement(t):.4f} mm\n'+note,fontsize=10)
        return []
    FuncAnimation(fig,frame,frames=len(times),interval=250).save(OUT/'continuation.gif',writer=PillowWriter(fps=4));plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--movie',action='store_true');a=p.parse_args();curves(a.names);maps(a.names)
    if a.movie:movie(['quarter-continuation'])
