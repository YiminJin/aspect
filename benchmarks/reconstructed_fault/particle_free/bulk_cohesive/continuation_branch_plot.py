#!/usr/bin/env python3
"""Actual first-hold candidates, stress redistribution and phase-mode evidence."""
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuation_analysis import BASE,OUT,read,save_csv

def main():
    names=['reference-hold-critical','reference-relax'];data=[]
    for name in names:
        with np.load(BASE/(name+'-plot-data.npz')) as a:data.append({k:a[k] for k in a.files})
    with np.load(BASE/'original-plot-data.npz') as a:parentstress=a['stress'][140]
    beta=1/(1+.00125*.01);taumax=max(float(np.max(abs(d['stress'][0]-beta*parentstress))) for d in data)
    gmax=max(float(np.max((d['forces'][0,np.argmin(abs(d['y']-.5)),:,3]/d['forces'][0,np.argmin(abs(d['y']-.5)),:,0])[d['x']>=1.05])) for d in data)
    fig,axs=plt.subplots(3,2,figsize=(12,9),layout='constrained');maximum=max(float(d['delta'][0].max()) for d in data);rows=[]
    for j,(name,d) in enumerate(zip(names,data)):
        x,y=d['x'],d['y'];meta=json.load(open(BASE/'runs'/name/'continuation-state.json'));h=read(OUT/(name+'-history.csv'))[0]
        change=d['delta'][0];tauchange=d['stress'][0]-meta['beta']*parentstress
        im=axs[0,j].pcolormesh(x,y,change,shading='gouraud',cmap='magma',vmin=0,vmax=maximum);fig.colorbar(im,ax=axs[0,j],label='First-hold phase increment')
        im=axs[1,j].pcolormesh(x,y,tauchange,shading='flat',cmap='RdBu_r',vmin=-taumax,vmax=taumax);fig.colorbar(im,ax=axs[1,j],label='tau_xy minus Maxwell-scaled parent [Pa]')
        for row,ax in enumerate(axs[:2,j]):
            ax.contour(x,y,d['phase'][0],levels=[.01,.05,.1,.3],colors='white' if row==0 else 'black',linewidths=.7);ax.axvline(1.09264647817,color='gray',ls='--',lw=.8);ax.set_xlim(1.05,1.5);ax.set_ylim(.39,.61);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
        axs[0,j].set_title(('Existing predictor' if j==0 else 'Accepted-state continuation')+f'; t={d["times"][0]:.5f} s')
        center=np.argmin(abs(y-.5));force=d['forces'][0];g=force[center,:,3]/force[center,:,0]
        keep=x>=1.05;axs[2,j].plot(x[keep],g[keep],label='Native weak phase residual / mass');axs[2,j].axhline(0,color='.5',lw=.6);axs[2,j].set_xlim(1.05,1.5);axs[2,j].set_ylim(-.02*gmax,1.05*gmax);axs[2,j].set_xlabel('x [m]');axs[2,j].set_ylabel('Weak residual density [Pa]');axs[2,j].grid(alpha=.2)
        rows.append([name,d['times'][0],change.max(),h['integrated_abs_w_m2_s'],h['bulk_J_m'],h['cohesive_J_m'],h['fracture_J_m'],h['friction_J_m'],h['incremental_potential_J_m'],h['endpoint_work_J_m']])
    fig.savefig(OUT/'first-hold-branches.png',dpi=160);plt.close(fig)
    save_csv(OUT/'first-hold-branches.csv',['candidate','time_s','max_phase_increment','integrated_abs_w_m2_s','bulk_J_m','cohesive_J_m','fracture_J_m','friction_J_m','incremental_potential_J_m','external_work_J_m'],rows)
    mode=np.load(BASE/'stability/predictor-hold-stability-interior-mode.npz');c=mode['coordinates'];ids=c['phase_dof'].astype(int);v=np.zeros(data[0]['initial'].size);v[ids[ids>=0]]=mode['mode'][ids>=0];v/=max(abs(v));nodes=read(BASE/'runs/reference-hold-critical/phase-0141.csv');nodes.sort(order='dof');order=np.lexsort((nodes['x'],nodes['y']));shape=data[0]['initial'].shape
    vv=v[order].reshape(shape);xx,yy=np.meshgrid(data[0]['x'],data[0]['y']);keep=(xx>1.05)&(abs(yy-.5)<.12)
    save_csv(OUT/'unstable-mode-profile.csv',['x_m','y_m','normalized_phase_mode'],zip(xx[keep],yy[keep],vv[keep]))
    fig,ax=plt.subplots(figsize=(10,3),layout='constrained');im=ax.pcolormesh(data[0]['x'],data[0]['y'],vv,shading='gouraud',cmap='RdBu_r',vmin=-1,vmax=1);ax.contour(data[0]['x'],data[0]['y'],data[0]['phase'][0],levels=[.01,.05,.1,.3],colors='black',linewidths=.6);ax.set_xlim(1.05,1.5);ax.set_ylim(.39,.61);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title('Admissible negative-curvature phase mode; mechanics/return response eliminated');fig.colorbar(im,ax=ax,label='Phase mode / peak magnitude');fig.savefig(OUT/'unstable-mode.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
