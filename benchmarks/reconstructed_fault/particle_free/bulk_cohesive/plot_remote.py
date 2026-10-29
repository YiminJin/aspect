#!/usr/bin/env python3
"""Original-state physical stress/driving and discrete phase-force views."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_spatial import q1


def main(out):
    d=dict(np.load(out/'remote-fields.npz'));L={k:d[k] for k in ['x','y','grid_ids']};L.update(dx=L['x'][1]-L['x'][0],dy=L['y'][1]-L['y'][0])
    p0=d['phase0'];X,Y=np.meshgrid(np.linspace(0,1.5,385),np.linspace(0,1,257));xy=np.stack([X,Y],axis=-1);initial=q1(L,p0,xy)[0]
    centers=d['points'].mean(axis=1);ix=np.rint(centers[:,0]/L['dx']-.5).astype(int);iy=np.rint(centers[:,1]/L['dy']-.5).astype(int)
    weights=d['weights'];steps=[120,121,124,140];levels=[.001,.01,.05,.1,.3,.5]
    def grid(a):
        value=np.empty((len(L['y'])-1,len(L['x'])-1));value[iy,ix]=np.sum(weights*a,axis=1)/weights.sum(axis=1);return value
    def marks(ax,step):
        phi=q1(L,d[f'{step}_phase'],xy)[0]
        ax.contour(X,Y,initial,levels=levels,colors='white',linestyles='--',linewidths=.45)
        ax.contour(X,Y,phi,levels=levels,colors='cyan',linewidths=.55)
        support=np.hypot(np.maximum(X-.5,0),Y-.5)
        ax.contour(X,Y,support,levels=[.30882159],colors='lime',linestyles=':',linewidths=.85)
        newly=(p0==0)&(d[f'{step}_increment']>1e-12)
        act=q1(L,newly.astype(float),xy)[0];ax.contour(X,Y,act,levels=[.5],colors='magenta',linewidths=.65)
        ax.plot([0,.5],[.5,.5],color='red',lw=.7);ax.plot(.5,.5,'rx',ms=4)
        ax.set(xlim=(0,1.5),ylim=(0,1),aspect='equal',xlabel='x [m]',ylabel='y [m]')
    fig,axes=plt.subplots(4,4,figsize=(16,10),layout='constrained')
    columns=[('tau','Physical shear stress tau_xy [Pa]',0,2200,'viridis'),('Y','Local phase driving Y [Pa]',0,80,'magma'),('fracture','Weak fracture resistance / mass [Pa]',0,float(max(d[f'{s}_fracture'].max() for s in steps)),'viridis'),('balance','Weak (driving - resistance) / mass [Pa]',float(min(d[f'{s}_balance'].min() for s in steps)),0,'coolwarm')]
    artists=[]
    for j,(field,title,lo,hi,cmap) in enumerate(columns):
        for i,step in enumerate(steps):
            a=d[f'{step}_{field}'];ax=axes[i,j]
            if field in ['tau','Y']:artist=ax.pcolormesh(L['x'],L['y'],grid(a),vmin=lo,vmax=hi,cmap=cmap,rasterized=True)
            else:artist=ax.pcolormesh(L['x'],L['y'],a[L['grid_ids']],shading='gouraud',vmin=lo,vmax=hi,cmap=cmap,rasterized=True)
            marks(ax,step)
            if j==0:ax.set_ylabel(f't={step*.00125:.5f} s\ny [m]')
            if i==0:ax.set_title(title,fontsize=10)
        fig.colorbar(artist,ax=axes[:,j],shrink=.75,location='bottom',pad=.03)
    fig.suptitle('Retained affine loading: physical stress, local driving and native nodal force balance',fontsize=13)
    fig.supxlabel('White dashed: initial phase; cyan: current phase; lime dotted: original compact support; magenta: initially-zero nodes now damaged.\nRed: original seed/endpoint. QP maps are positive-weight cell averages; nodal force maps are visual interpolation only.\nAt accepted growing nodes, driving balances resistance within the retained KKT tolerance; negative balance is compatible with an active obstacle.',fontsize=9)
    fig.savefig(out/'remote-stress-balance.png',dpi=160);fig.savefig(out/'remote-stress-balance.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    slices=[slice(0,len(d['xx'])),slice(len(d['xx']),len(d['xx'])+len(d['rr'])),slice(len(d['xx'])+len(d['rr']),None)]
    colors=plt.cm.viridis(np.linspace(0,1,len(steps)))
    for row,(sl,name) in enumerate(zip(slices,['Centerline y=.5 m','Upper 45 degree ray','Lower 45 degree ray'])):
        xx=d['xx'] if row==0 else d['rr']
        for step,color in zip(steps,colors):
            i=list(d['steps']).index(step);p=d['profiles'][i,sl]
            axes[row,0].plot(xx,p[:,0],color=color,label=f'{step*.00125:.5f} s')
            axes[row,1].plot(xx,p[:,1],color=color)
            axes[row,2].plot(xx,p[:,2],color=color,label='mechanical' if step==steps[0] else None)
            axes[row,2].plot(xx,p[:,3],color=color,ls='--',label='fracture' if step==steps[0] else None)
        for ax in axes[row]:ax.grid(alpha=.25);ax.set_xlabel('x [m]' if row==0 else 'Radius from endpoint [m]')
        axes[row,0].set_ylabel(name+'\nPhysical tau_xy [Pa]');axes[row,1].set_ylabel('Local Y [Pa]');axes[row,2].set_ylabel('Weak force / mass [Pa]')
        for ax in axes[row]:ax.axvline(.5 if row==0 else 0,color='red',ls=':',lw=.8);ax.axvline(.80882159 if row==0 else .30882159,color='green',ls=':',lw=.8)
    axes[0,0].legend(fontsize=8);axes[0,2].legend(fontsize=8)
    fig.suptitle('Same physical compact averaging kernel, radius .0625 m, on original native points/nodes')
    fig.supxlabel('Force curves average native integrated nodal forces with matching nodal mass; Y/stress use positive Gauss8 weights.\nNo local history is moved to a new constitutive state. Dashed force curves are resistance, not a phase contour.',fontsize=9)
    fig.savefig(out/'remote-profiles.png',dpi=160);fig.savefig(out/'remote-profiles.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
