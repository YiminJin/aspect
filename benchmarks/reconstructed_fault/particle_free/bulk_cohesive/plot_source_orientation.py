#!/usr/bin/env python3
"""Equal-scale native-QP maps and compatible physical-kernel angle cuts."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from plot_orientation import mesh_lines,contours


def main(output):
    report=json.loads((output/'assessment.json').read_text());time=report['common_final_time']
    data={name:dict(np.load(output/f'{name}-fields.npz')) for name in ['aligned','rotated']}
    ymax=max(float(max(s['common_Y'])) for s in data.values())
    delta={name:s['common_phase_nodes']-s['initial_phase_nodes'] for name,s in data.items()}
    dmax=max(float(max(delta[name][(s['common_node_xy'][:,0]>.4)&(s['common_node_xy'][:,0]<.88)&(abs(s['common_node_xy'][:,1]-.5)<.34)])) for name,s in data.items())
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for row,(name,s) in enumerate(data.items()):
        for col,(values,norm,label) in enumerate([(s['initial_phi'],Normalize(0,.6),'Initial phase [1]'),
            (s['common_Y'],Normalize(0,ymax),f'Mechanical driving Y, t={time:g} s [Pa]'),
            (delta[name],Normalize(0,dmax),'Actual nodal phase increment [1]')]):
            ax=axes[row,col];ax.add_collection(LineCollection(mesh_lines(s),colors='.5',linewidths=.35))
            if col==2:ax.set_facecolor('.18')
            points=s['common_points'] if col<2 else s['common_node_xy']
            im=ax.scatter(points[:,0],points[:,1],c=values,norm=norm,cmap='viridis',s=1.4 if col<2 else 10,rasterized=True)
            contour_state=dict(s)
            if col==0:contour_state['common_phase_nodes']=s['initial_phase_nodes']
            contours(ax,contour_state);ax.plot(.5,.5,'+',color='black',ms=8,mew=2)
            first=report['cases'][name]['first_growing_nodes']
            ax.scatter([n['x'] for n in first],[n['y'] for n in first],s=70,facecolors='none',edgecolors='red',linewidths=1.2)
            ax.set(xlim=(.4,.88),ylim=(.16,.84),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=name+': '+label)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Same physical seed and loading, fresh histories; native gradient remainder retained\nWhite phase contours: .001, .01, .1, .3, .5. Red circles: first-growing nodes. Black +: original endpoint.',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(output/f'physical-overlays.{ext}',dpi=175)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for column,radius in enumerate([.225,.2881107642904027,.29851520585223124]):
        for name,color,style in [('aligned','tab:blue','-'),('rotated','tab:orange','--')]:
            for stage,alpha in [('initial',.45),('common',1.)]:
                a=np.genfromtxt(output/f'{name}-{stage}-arcs.csv',delimiter=',',names=True);a=a[abs(a['radius']-radius)<1e-12]
                if stage=='initial':axes[0,column].plot(a['angle'],a['fracture'],style,color=color,alpha=alpha,label=name+' initial F')
                else:
                    axes[0,column].plot(a['angle'],a['fracture'],style,color=color,label=name+' final F')
                    axes[1,column].plot(a['angle'],a['Y'],style,color=color,label=name)
                    axes[2,column].plot(a['angle'],a['total'],style,color=color,label=name)
            pre=np.genfromtxt(output/f'{name}-pre-arcs.csv',delimiter=',',names=True);pre=pre[abs(pre['radius']-radius)<1e-12]
            axes[1,column].plot(pre['angle'],pre['Y'],style,color=color,alpha=.4,label=name+' pre-onset')
            for node in report['cases'][name]['first_growing_nodes']:
                if abs(node['radius']-radius)<.012:
                    for row in [1,2]:axes[row,column].axvline(node['angle'],color=color,alpha=.3,lw=.8)
        for row,label in enumerate(['Fracture action F [Pa]','Mechanical driving Y [Pa]','Total F − Y [Pa]']):
            axes[row,column].set(xlim=(-90,90),ylabel=label);axes[row,column].grid(alpha=.2)
        axes[0,column].set_title(f'Physical radius {radius:.6f} m')
        axes[2,column].axhline(0,color='.3',lw=.7);axes[2,column].set_xlabel('Angle from physical +x [degrees]')
    for row in range(3):
        limits=[ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:ax.set_ylim(min(a[0] for a in limits),max(a[1] for a in limits))
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=8)
    fig.suptitle(f'Common physical kernel, radius .0625 m; final fields at {time:g} s\nQP traction/driving averaged after native evaluation; curves are distinct from nodal KKT.',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(output/f'common-kernel.{ext}',dpi=175)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);main(parser.parse_args().output)
