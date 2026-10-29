#!/usr/bin/env python3
"""Three principal figures, exclusively from the exported native trajectory."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_native import csv,gauss
from compare_native import grid

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.constrained_layout.use':True})


def save(fig,path):
    for suffix in ['.png','.pdf']:fig.savefig(path.with_suffix(suffix),dpi=190)
    plt.close(fig)


def endpoint(ax):
    ax.axvline(.5,color='black',ls=':',lw=1)
    ax.set_xlabel('x [m]')


def phase_contours(ax,g,**kwargs):
    x,y,p=g
    return ax.contour(x,y,p,levels=[.001,.01,.1,.3,.5],**kwargs)


def geometry(path,out,step):
    first=grid(path,0);last=grid(path,step);x,y,p=first
    fig,axes=plt.subplots(2,3,figsize=(13,7.3))
    for ax,g,title in zip(axes[0,:2],[first,last],['Prescribed physical seed, t = 0','Calculated phase, t = 0.16 s']):
        image=ax.pcolormesh(x,y,g[2],shading='gouraud',vmin=0,vmax=.6,cmap='viridis')
        phase_contours(ax,g,colors='white',linewidths=.6)
        ax.plot([0,.5],[.5,.5],color='tomato',lw=2,label='Original diagnostic surface')
        ax.scatter([.5],[.5],s=35,c='white',edgecolors='black',zorder=5)
        for yy,direction in [(.03,-1),(.97,1)]:
            ax.annotate('',xy=(.5+.16*direction,yy),xytext=(.5-.16*direction,yy),arrowprops={'arrowstyle':'->','color':'white','lw':1.5})
        ax.set_title(title);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_aspect('equal')
    fig.colorbar(image,ax=axes[0,:2],label='Physical phase φ [1]',shrink=.8)
    ax=axes[0,2]
    delta=last[2]-p
    im=ax.pcolormesh(x,y,delta,shading='gouraud',vmin=0,vmax=max(delta.max(),1e-6),cmap='magma')
    phase_contours(ax,first,colors='cyan',linewidths=.5)
    ax.plot([0,.5],[.5,.5],color='white',lw=1);ax.set_aspect('equal');ax.set_title('Actual phase increment; boundary nucleation')
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');fig.colorbar(im,ax=ax,label='Δφ [1]',shrink=.8)
    ax=axes[1,0]
    for value in x:ax.axvline(value,color='.7',lw=.4)
    for value in y:ax.axhline(value,color='.7',lw=.4)
    f=csv(path/f'bulk-{step:04d}.csv');mask=(f['x']>.4)&(f['x']<.9)&(abs(f['y']-.5)<.2)
    ax.scatter(f['x'][mask],f['y'][mask],s=2,color='steelblue',alpha=.5,label='Actual Gauss6 points')
    phase_contours(ax,first,colors='black',linestyles='--',linewidths=1)
    phase_contours(ax,last,colors='tomato',linewidths=.7)
    ax.scatter([.5],[.5],c='black',marker='x',s=50);ax.set(xlim=(.4,.9),ylim=(.3,.7),title='Tip/cap: initial and final contours coincide',xlabel='x [m]',ylabel='y [m]');ax.set_aspect('equal')
    ax.legend(fontsize=8,loc='upper right')
    ax=axes[1,1];j=np.argmin(abs(y-.5))
    ax.plot(x,p[j],color='black',ls='--',label='Initial')
    ax.plot(x,last[2][j],color='tomato',label='Final');ax.set_yscale('symlog',linthresh=1e-4)
    ax.set(title='Centerline: disconnected right-hand damage',ylabel='φ(x, 0.5 m) [1]');endpoint(ax);ax.legend()
    ax=axes[1,2];fine=path.parent/'spatial';f0=grid(fine,0);f1=grid(fine,step)
    im=ax.pcolormesh(f0[0],f0[1],f1[2]-f0[2],shading='gouraud',cmap='magma',vmin=0,vmax=.0013)
    phase_contours(ax,f0,colors='cyan',linestyles='--',linewidths=.7)
    phase_contours(ax,f1,colors='white',linewidths=.6)
    ax.scatter([.5],[.5],color='white',marker='x',s=35)
    ax.set(xlim=(.4,.9),ylim=(.3,.7),title='32 × 64 control: pre-existing cap flank thickens',xlabel='x [m]',ylabel='y [m]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,label='Fine-mesh Δφ [1]',shrink=.8)
    fig.suptitle('Bulk cohesive specimen: imposed seed and loading; computed boundary response')
    save(fig,out/'specimen')


def mechanism(path,out,step):
    f=csv(path/f'bulk-{step:04d}.csv');nodes=csv(path/f'phase-{step:04d}.csv');forces=csv(path/f'phase-forces-{step:04d}.csv')
    fig,axes=plt.subplots(2,3,figsize=(13,7.2))
    for ax,key,title,limits in zip(axes[0],['tauxy','q','driving'],['Physical shear τxy [Pa]','Bulk cohesive traction q [Pa]','Mechanical driving Y [Pa]'],[(900,2100),(0,1100),(0,80)]):
        im=ax.scatter(f['x'],f['y'],c=f[key],s=3,cmap='viridis',vmin=limits[0],vmax=limits[1])
        ax.set(xlim=(.4,1),ylim=(.25,.75),title=title,ylabel='y [m]');endpoint(ax)
        phase_contours(ax,grid(path,step),colors='white',linewidths=.5)
        fig.colorbar(im,ax=ax,shrink=.85)
    # A native Gauss row nearest the centerline. Do not interpolate local a/q.
    yy=np.unique(f['y']);line=np.sort(f[f['y']==yy[np.argmin(abs(yy-.5))]],order='x')
    ax=axes[1,0]
    ax.plot(line['x'],line['Txy'],label='Auxiliary Txy',ls='--')
    ax.plot(line['x'],line['tauxy'],label='Physical τxy');ax.plot(line['x'],line['q'],label='q')
    ax.set(title=f'Actual Gauss row y = {line["y"][0]:.5f} m',ylabel='Stress [Pa]');endpoint(ax);ax.legend(fontsize=8)
    line=np.sort(nodes[nodes['y']==.5],order='x');byid={int(a['dof']):a for a in forces};g=np.array([byid[int(i)] for i in line['dof']],dtype=forces.dtype)
    ax=axes[1,1]
    ax.plot(line['x'],g['fracture_gradient']/g['mass'],label='Fracture g_f / mass')
    ax.plot(line['x'],-g['mechanical_gradient']/g['mass'],label='Mechanical −g_m / mass')
    ax.plot(line['x'],g['residual']/g['mass'],label='Total g / mass',ls='--')
    ax.set(title='Weak nodal densities; distinct from pointwise Y',ylabel='Density [Pa]');endpoint(ax);ax.legend(fontsize=8)
    ax=axes[1,2]
    active=(g['phi']==g['obstacle'])
    ax.step(line['x'],active.astype(float),where='mid',label='At physical lower obstacle')
    ax.scatter(line['x'][~active],np.zeros(np.sum(~active)),c='tomato',s=20,label='Growing/free node')
    ax.set(ylim=(-.15,1.3),yticks=[0,1],title='Centerline KKT: g ≥ 0 at obstacle',ylabel='Active [1]');endpoint(ax);ax.legend(fontsize=8)
    fig.suptitle('At ramp end t = 0.15 s: finite driving ahead of the old endpoint; tip still held by obstacle')
    save(fig,out/'driving')


def response(path,half,out):
    a=csv(path/'bulk-accepted.csv');b=csv(half/'bulk-accepted.csv')
    ja=json.loads((out/'native-primary.json').read_text());jb=json.loads((out/'native-half.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(13,7.3))
    for data,name in [(a,'dt'),(b,'dt/2')]:
        mask=(data['time']>.05+1e-12)&(data['time']<=.15+1e-12)
        axes[0,0].plot(data['time'][mask],data['Wext'][mask]/(.018*data['dt'][mask]),label=name)
    axes[0,0].set(title='Boundary shear reaction',xlabel='Time [s]',ylabel='Generalized reaction [N/m]');axes[0,0].legend()
    for result,name,style in [(ja,'dt','-'),(jb,'dt/2','--')]:
        h=result['history'];t=[v['time'] for v in h]
        axes[0,1].plot(t,[v['ahead'] for v in h],style,label=f'Max ahead ({name})')
    axes[0,1].plot([v['time'] for v in ja['history']],[v['tip'] for v in ja['history']],color='black',label='Tip-window increment (zero)')
    axes[0,1].set(title='Boundary change versus connected tip',xlabel='Time [s]',ylabel='Δφ [1]');axes[0,1].legend(fontsize=8)
    for level in ['0.001','0.01','0.1','0.3','0.5']:
        h=ja['history'];axes[0,2].plot([v['time'] for v in h],[v['fronts'][level] for v in h],label=level)
    axes[0,2].set(title='Seed-connected contour positions: no advance',xlabel='Time [s]',ylabel='Centerline x [m]');axes[0,2].legend(fontsize=8,ncol=3)
    ax=axes[1,0];ax.plot(a['time'],np.cumsum(a['Wext']),label='Endpoint Wext',color='black')
    for key,label in [('bulk','Δ bulk storage'),('cohesive','Cohesive storage'),('fracture','Δ fracture storage')]:
        value=a[key]-a[key][0] if key!='cohesive' else a[key]
        ax.plot(a['time'],value,label=label)
    for key,label in [('Db','Bulk viscosity'),('Dc','Cohesive viscosity'),('Df','Friction')]:ax.plot(a['time'],np.cumsum(a[key]),label=label,ls='--')
    ax.set(title='Separate physical storage and dissipation',xlabel='Time [s]',ylabel='Energy per out-of-plane length [J/m]');ax.legend(fontsize=7,ncol=2)
    ax=axes[1,1]
    for data,label in [(a,'dt'),(b,'dt/2')]:
        remainder=np.cumsum(data['Wext']-data['energy_change']-data['Db']-data['Dc']-data['Df'])
        ax.plot(data['time'],remainder,label=f'Endpoint remainder ({label})')
    ax.set(title='Declared 1% endpoint-work screen fails',xlabel='Time [s]',ylabel='Energy remainder [J/m]');ax.legend(fontsize=8)
    ax=axes[1,2]
    for key,label in [('stress_loss','Stress discretization'),('traction_loss','Traction discretization'),('phase_remainder','Phase compliance'),('fracture_loss','Fracture discretization'),('phase_work','Phase residual work'),('mechanical_work','Mechanical residual work')]:
        ax.plot(a['time'],np.cumsum(a[key]),label=label)
    ax.set(title='Exact ledger closure is not physical qualification',xlabel='Time [s]',ylabel='Cumulative contribution [J/m]');ax.legend(fontsize=7)
    fig.suptitle('Bounded response: converged fields, first-order endpoint-work uncertainty, no tip propagation')
    save(fig,out/'response')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('primary',type=Path);p.add_argument('half',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();geometry(a.primary,a.output,64);mechanism(a.primary,a.output,60);response(a.primary,a.half,a.output)
