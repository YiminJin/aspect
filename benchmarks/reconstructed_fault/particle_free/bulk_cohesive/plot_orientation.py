#!/usr/bin/env python3
"""Common physical axes/scales; original QP colors, actual nodal increments."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,LogNorm
from matplotlib.tri import Triangulation
from assess_native import csv


def mesh_lines(s):
    return [v[[0,1,3,2,0]] for v in s['initial_vertices']]


def contours(ax,s):
    # Explicit visualization of the physical bilinear Q1 field, not a remapping
    # of local constitutive histories. Small subdivisions reduce contour kinks.
    z=np.linspace(0,1,6);u,v=np.meshgrid(z,z);u=u.ravel();v=v.ravel()
    N=np.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],axis=-1)
    vertices=s['initial_vertices'];ids=s['initial_indices'];phi=s['common_phase_nodes'][ids]
    points=np.einsum('qi,cid->cqd',N,vertices).reshape(-1,2)
    values=np.einsum('qi,ci->cq',N,phi).ravel()
    triangles=[]
    for c in range(len(vertices)):
        for j in range(5):
            for i in range(5):
                a=c*36+j*6+i;triangles.extend([[a,a+1,a+7],[a,a+7,a+6]])
    t=Triangulation(points[:,0],points[:,1],np.array(triangles))
    ax.tricontour(t,values,levels=[.001,.01,.1,.3,.5],colors='white',linewidths=.8)


def main(out):
    report=json.loads((out/'assessment.json').read_text());t=report['last_common_time']
    data={name:dict(np.load(out/f'{name}-fields.npz')) for name in ['reference','rotated']}
    ymax=max(float(np.max(s['common_Y'])) for s in data.values());ymax=max(ymax,1.)
    dmax=max(float(np.max(s['common_phase_nodes']-s['initial_phase_nodes'])) for s in data.values());dmax=max(dmax,1e-12)
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for row,(name,s) in enumerate(data.items()):
        p=s['common_points'];xy=s['common_node_xy'];delta=s['common_phase_nodes']-s['initial_phase_nodes']
        for col,(values,norm,label) in enumerate([(s['initial_phi'],Normalize(0,.6),'Initial physical phase [1]'),(s['common_Y'],Normalize(0,ymax),f'Accepted driving Y at {t:g} s [Pa]'),(delta,Normalize(0,dmax),'Actual nodal damage increment [1]')]):
            ax=axes[row,col]
            if col==2:ax.set_facecolor('.25')
            ax.add_collection(LineCollection(mesh_lines(s),colors='.45',linewidths=.32,alpha=.7))
            points=p if col<2 else xy
            im=ax.scatter(points[:,0],points[:,1],c=values,cmap='viridis',norm=norm,s=1.2 if col<2 else 7,rasterized=True)
            contours(ax,s);ax.plot(.5,.5,'o',mfc='white',mec='red',ms=5)
            ax.set(xlim=(.4,.87),ylim=(.16,.84),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f'{name}: {label}')
            fig.colorbar(im,ax=ax,shrink=.8)
            if col==2:ax.text(.5,.03,'No cap growth before source guard' if report['cases'][name]['max_cap_increment']<1e-12 else 'Accepted increment only',transform=ax.transAxes,ha='center',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    fig.suptitle('Same physical seed/loading; white phase contours .001, .01, .1, .3, .5\nRotated transition fails unchanged DG source guard before flank activation; no post-onset comparison',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(out/f'physical-overlays.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for j,r in enumerate([.225,.2881107642904027,.29851520585223124]):
        for name,style in [('reference','-'),('rotated','--')]:
            a=np.genfromtxt(out/f'{name}-common-arcs.csv',delimiter=',',names=True);a=a[abs(a['radius']-r)<1e-12]
            axes[0,j].plot(a['angle'],a['fracture'],style,label=name)
            axes[1,j].plot(a['angle'],a['Y'],style,label=name)
        axes[0,j].set(title=f'r={r:.6f} m',ylabel='Common-kernel fracture action [Pa]')
        axes[1,j].set(ylabel=f'Common-kernel Y at {t:g} s [Pa]',xlabel='Angle from physical +x [degrees]')
        for ax in axes[:,j]:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Common physical radial kernel (radius .0625 m), distinct from native nodal KKT\nPre-onset only; absence of driving lobes here does not diagnose their orientation',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(out/f'common-kernel.{ext}',dpi=180)
    plt.close(fig)
    m=csv(Path('build-particle-free/bulk-orientation/runs/rotated/mesh-qualification.csv'))
    verts=np.stack([np.stack([m[f'x{i}'],m[f'y{i}']],axis=-1) for i in range(4)],axis=1)
    center=verts.mean(axis=1)
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    ax=axes[0];ax.add_collection(LineCollection([v[[0,1,3,2,0]] for v in verts],linewidths=.35,colors='.4'))
    im=ax.scatter(center[:,0],center[:,1],c=np.maximum(m['source3_error'],1e-13),norm=LogNorm(1e-13,100),s=8,cmap='magma')
    ax.plot(.5,.5,'o',mfc='white',mec='red');ax.set(xlim=(0,1.5),ylim=(0,1),aspect='equal',xlabel='x [m]',ylabel='y [m]',title='All-nine-mode Q2 gradient projection error [1/m]')
    fig.colorbar(im,ax=ax,shrink=.6)
    i=int(np.argmax(m['source3_error']));v=verts[i];c=v.mean(axis=0)
    ax=axes[1];ax.add_collection(LineCollection([q[[0,1,3,2,0]] for q in verts],linewidths=.6,colors='.5'));ax.plot(v[[0,1,3,2,0],0],v[[0,1,3,2,0],1],'r-',lw=2)
    ax.set(xlim=(c[0]-.06,c[0]+.06),ylim=(c[1]-.06,c[1]+.06),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f"Worst synthetic transition cell {m['cell'][i]}\nGauss3 and Gauss8 errors agree")
    for ext in ['png','pdf']:fig.savefig(out/f'projection-blocker.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);main(p.parse_args().output)
