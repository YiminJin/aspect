#!/usr/bin/env python3
"""Two physical-coordinate P3 figures from exported native fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from assess_p3_extension import csv, load_state, lagrange, h


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    metrics=json.loads(a.metrics.read_text());material_m=metrics.get('constitutive_m',128.)
    old,cells,T0=load_state(a.data,'old');new,_,T1=load_state(a.data,'new')
    phase0=csv(a.data/'old-phase.csv');phase1=csv(a.data/'new-phase.csv')
    assert np.array_equal(phase0,phase1)
    px=np.unique(phase0['x']);py=np.unique(phase0['y'])
    values=phase0['phi'].reshape(len(py),len(px))
    def phi(x,y):
        i=np.clip(np.searchsorted(px,x,side='right')-1,0,len(px)-2)
        j=np.clip(np.searchsorted(py,y,side='right')-1,0,len(py)-2)
        u=(x-px[i])/(px[i+1]-px[i]);v=(y-py[j])/(py[j+1]-py[j])
        return (1-u)*(1-v)*values[j,i]+u*(1-v)*values[j,i+1]+(1-u)*v*values[j+1,i]+u*v*values[j+1,i+1]
    grid=np.linspace(0,1,701);X,Y=np.meshgrid(grid,grid);P=phi(X,Y)
    nodes=csv(a.data/'basis-0.csv')
    def tensor(T,x,y):
        i=np.clip(np.searchsorted(px,x,side='right')-1,0,len(px)-2)
        j=np.clip(np.searchsorted(py,y,side='right')-1,0,len(py)-2)
        u=(x-px[i])/(px[i+1]-px[i]);v=(y-py[j])/(py[j+1]-py[j])
        # load_state sorts cells by x then y.
        shape=lagrange(u)[...,np.rint(nodes['u']*2).astype(int)]*lagrange(v)[...,np.rint(nodes['v']*2).astype(int)]
        return np.sum(shape*T[i*(len(py)-1)+j,:,2],axis=-1)
    def correction(s,x,y):
        z=np.interp(x,s['x'],s['Z']);I=np.interp(x,s['x'],s['I'])
        return np.where((x>=s['x'][0])&(x<=s['x'][-1]),z*h(phi(x,y))*(material_m/128.)/I,0.)
    def decorate(ax,after,zoom=False):
        ax.plot(old['x'],old['y'],color='#8c2d04',lw=2.4)
        if after:ax.plot(new['x'][len(old)-1:],new['y'][len(old)-1:],color='#d7191c',lw=2.4)
        if not zoom:ax.plot(0,.5,'^',color='black',ms=6,zorder=6)
        ax.plot(.5,.5,'o',mfc='white',mec='black',ms=6,zorder=6)
        if after:ax.plot(.75,.5,'s',mfc='#d7191c',mec='black',ms=6,zorder=6)
        ax.set(xlim=(.40,.86) if zoom else (0,1),ylim=(.27,.73) if zoom else (0,1),xlabel='x [m]',ylabel='y [m]')
        ax.set_aspect('equal')
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'savefig.dpi':180})
    fig=plt.figure(figsize=(12,9),layout='constrained');gs=fig.add_gridspec(4,6,height_ratios=[.18,1,1,.8])
    for row,zoom in enumerate([False,True]):
        for k,after in enumerate([False,True]):
            ax=fig.add_subplot(gs[row+1,3*k:3*k+3])
            im=ax.pcolormesh(X,Y,P,cmap='cividis',vmin=0,vmax=.6,rasterized=True,shading='auto')
            cs=ax.contour(X,Y,P,levels=[.01,.1,.3,.5],colors='white',linewidths=.65)
            if zoom:ax.clabel(cs,fmt='%g',fontsize=8)
            decorate(ax,after,zoom);ax.set_title(('After prescribed append' if after else 'Before append')+(' — tip view' if zoom else ' — full domain'))
            if zoom:ax.annotate('Pre-existing cap',(.65,.6),xytext=(.43,.70),arrowprops={'arrowstyle':'->','lw':.8})
    fig.colorbar(im,ax=fig.axes[:4],label=r'Archived phase $\phi$ (unchanged)',fraction=.025)
    for k,(field,label) in enumerate([('I',r'$I$ [m]'),('Z',r'$Z=G s_{elas}$ [Pa m]'),('cumulative_slip','Accumulated slip [m]')]):
        ax=fig.add_subplot(gs[3,2*k:2*k+2]);ax.plot(old['x'],old[field],color='#2166ac',lw=2,label='Before')
        ax.plot(new['x'],new[field],'--',color='#d6604d',label='After')
        ax.scatter(new['x'][len(old):],new[field][len(old):],s=12,color='#d6604d',zorder=5)
        ax.axvspan(.5,.75,color='#d6604d',alpha=.1);ax.set(xlabel='Fault coordinate x [m]',ylabel=label,xlim=(0,.78));ax.grid(alpha=.2)
        if k==0:ax.legend(fontsize=8)
    handles=[Line2D([],[],color='#8c2d04',lw=2,label='Original segment'),Line2D([],[],color='#d7191c',lw=2,label='Prescribed append / new records'),Line2D([],[],marker='o',mfc='white',mec='black',ls='',label='Old physical tip'),Line2D([],[],marker='s',color='#d7191c',mec='black',ls='',label='New physical tip')]
    handles.append(Line2D([],[],marker='^',color='black',ls='',label='Unchanged boundary endpoint'))
    legend=fig.add_subplot(gs[0,:]);legend.axis('off');legend.legend(handles=handles,loc='center',ncol=3,fontsize=8)
    fig.suptitle(r'Imposed geometry change at fixed physical time: $\max|\Delta\phi|=0$',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(a.output/f'geometry_histories.{ext}',bbox_inches='tight')
    plt.close(fig)

    before=tensor(T0,X,Y)-correction(old,X,Y);after=tensor(T1,X,Y)-correction(new,X,Y)
    unchanged=tensor(T0,X,Y)-correction(new,X,Y)
    (a.output/'figure-sampling.json').write_text(json.dumps(dict(grid=[len(grid),len(grid)],
        tensor_trace='right cell at grid interfaces; final cell at physical boundary',
        physical_shear_range_before_Pa=[float(before.min()),float(before.max())],
        dense_grid_maximum_full_tensor_transfer_difference_Pa=float(np.sqrt(2)*np.max(np.abs(after-before))),
        dense_grid_maximum_full_tensor_unchanged_T_difference_Pa=float(np.sqrt(2)*np.max(np.abs(unchanged-before)))),indent=2)+'\n')
    lo=min(before.min(),after.min(),unchanged.min())/1000;hi=max(before.max(),after.max(),unchanged.max())/1000
    fig=plt.figure(figsize=(13,8),layout='constrained');gs=fig.add_gridspec(2,3);stress_axes=[];difference_axes=[]
    for k,(field,title) in enumerate([(before,'Before'),(after,'After transferred T'),(unchanged,'After: unchanged-T control')]):
        ax=fig.add_subplot(gs[0,k]);stress_axes.append(ax);im=ax.pcolormesh(X,Y,field/1000,cmap='RdBu_r',vmin=lo,vmax=hi,rasterized=True,shading='auto');decorate(ax,k>0);ax.set_title(title)
    fig.colorbar(im,ax=stress_axes,label=r'Physical reconstructed $\tau_{xy}$ [kPa]',fraction=.025)
    lim=max(np.max(np.abs(after-before)),np.max(np.abs(unchanged-before)))*1000
    for k,(field,title) in enumerate([(after-before,'Transfer projection error'),(unchanged-before,'Error if auxiliary T is unchanged')]):
        ax=fig.add_subplot(gs[1,k]);difference_axes.append(ax);im=ax.pcolormesh(X,Y,field*1000,cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True,shading='auto')
        decorate(ax,True);ax.set(xlim=(.35,.57),ylim=(.35,.65));ax.set_title(title+'\nOld-tip view')
    fig.colorbar(im,ax=difference_axes,label=r'$\Delta\tau_{xy}$ [mPa], common scale',fraction=.035)
    ax=fig.add_subplot(gs[1,2]);ax.axis('off');q=json.loads(a.metrics.read_text())
    text=('STORAGE [J/m]\n'
          f"Bulk: {q['bulk_energy_old']:.6f} → {q['bulk_energy_new']:.6f}\n"
          f"  Projection defect: {q['bulk_energy_transfer_defect']:+.3e}\n"
          f"Cohesive: {q['cohesive_energy_old']:.6g}\n"
          f"  Geometry / I change: {q['prescribed_geometry_cohesive_energy_change']:+.3e}\n"
          '  New records: 0; fracture change: 0\n\n'
          'SHEAR WEAK LOAD [Pa m]\n'
          f"Overlap measure change: {q['intended_overlap_measure_change']:+.6g}\n"
          f"Added surface: {q['intended_new_surface_shear']:+.6g}\n"
          f"Projection defect: {q['weak_shear_transfer_defect']:+.3e}\n"
          'Added friction load: +250 Pa m\n'
          +(f"Cohesive: {q['native_cohesive_weak_change']:+.2e}; residual: {q['native_residual_weak_change']:+.3f}\n\n" if 'native_cohesive_weak_change' in q else '\n')
          +
          r'$\tau_{xy}=T_{xy}-Z h(\phi)/I$'+'\nT is auxiliary history; τ is physical stress.\nNo time or phase evolution; no equilibrium claim.')
    ax.text(0,1,text,va='top',fontsize=10,linespacing=1.35)
    fig.suptitle('Physical consequences of the prescribed append',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(a.output/f'physical_consequences.{ext}',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
