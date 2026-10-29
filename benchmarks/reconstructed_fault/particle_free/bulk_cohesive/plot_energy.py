#!/usr/bin/env python3
"""Physical work, BE remainders and temporal response on common scales."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from assess_native import csv
from plot_orientation import contours

COLORS={'h':'tab:blue','half':'tab:orange','quarter':'tab:green'}
LABEL={'h':'h = .0025 s','half':'h/2','quarter':'h/4'}


def switches(ax):
    for t in [.05,.15]:ax.axvline(t,color='.5',lw=.8,ls=':')
    ax.set(xlabel='Physical time [s]',xlim=(0,.16));ax.grid(alpha=.18)


def main(output):
    report=json.loads((output/'assessment.json').read_text());data={n:csv(output/f'{n}-energy.csv') for n in COLORS}
    fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
    for name,d in data.items():
        color=COLORS[name];label=LABEL[name];t=d['time']
        axes[0,0].plot(t,np.cumsum(d['endpoint_work']),':',color=color)
        axes[0,0].plot(t,np.cumsum(d['trapezoidal_work']),color=color,label=label)
        axes[1,1].plot(t,np.cumsum(d['endpoint_remainder']),color=color,label=label)
        axes[2,0].plot(t,np.cumsum(d['trapezoidal_remainder']),color=color,label=label+' signed')
        axes[2,0].plot(t,np.cumsum(abs(d['trapezoidal_remainder'])),'--',color=color,label=label+' variation')
        axes[2,1].plot(t,d['Q'],color=color,label=label)
    d=data['quarter'];t=d['time']
    for key,label,color in [('bulk','bulk storage','tab:blue'),('cohesive','cohesive storage','tab:orange'),('fracture','fracture storage','tab:green')]:
        initial=0 if key!='fracture' else d['fracture'][0]
        axes[0,1].plot(t,d[key]-initial,color=color,label='Δ '+label)
    for key,label,color in [('Db','bulk','tab:blue'),('Dc','cohesive','tab:orange'),('Df','friction','tab:red')]:
        axes[0,1].plot(t,np.cumsum(d[key]),'--',color=color,label=label+' dissipation')
    axes[0,1].set_yscale('symlog',linthresh=1e-7)
    for key,label in [('stress_loss','stress BE'),('traction_loss','traction BE'),('phase_remainder','phase Taylor'),('fracture_loss','fracture BE')]:
        axes[1,0].plot(t,np.cumsum(d[key]),label=label)
    axes[1,0].plot(t,np.cumsum(d['phase_work']+d['mechanical_work']),'k:',label='residual work')
    axes[1,0].set_yscale('symlog',linthresh=1e-10)
    titles=['External work: trapezoidal solid, endpoint dotted','Storage changes and physical dissipation (h/4)',
            'Separate numerical remainders (h/4)','Endpoint residual = numerical terms − residual work',
            'Trapezoidal residual: signed sum and total variation','Generalized shear reaction remains nonzero on hold']
    labels=['Cumulative work [J/m]','Change / dissipation [J/m]','Cumulative remainder [J/m]','Cumulative endpoint residual [J/m]',
            'Cumulative physical residual [J/m]','Strain-conjugate reaction Q [J/m]']
    for ax,title,label in zip(axes.ravel(),titles,labels):
        ax.set(title=title,ylabel=label);switches(ax);ax.legend(fontsize=7)
    axes[2,1].text(.02,.93,'Loading: .05–.15 s\nHold: zero motion, retained stress',transform=axes[2,1].transAxes,va='top',fontsize=9)
    fig.suptitle('Same aligned wide specimen and backward Euler; only the timestep changes\nVertical markers: loading start and stop. Initial fracture storage is subtracted.',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(output/f'work-ledger.{ext}',dpi=175)
    plt.close(fig)

    geo=dict(np.load(output/'geometry.npz'));fields={n:dict(np.load(output/f'{n}-fields.npz')) for n in COLORS}
    vmax=max(float(max(v['phase'][-1]-geo['initial_phase'])) for v in fields.values())
    # Use the cap-window maximum, excluding the separate left boundary patch.
    vmax=max(float(max((v['phase'][-1]-geo['initial_phase'])[geo['cap_nodes']])) for v in fields.values())
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for column,(name,s) in enumerate(fields.items()):
        ax=axes[0,column];ax.set_facecolor('.18');xy=geo['xy'];delta=s['phase'][-1]-geo['initial_phase']
        lines=[v[[0,1,3,2,0]] for v in geo['vertices']];ax.add_collection(LineCollection(lines,colors='.4',linewidths=.4))
        im=ax.scatter(xy[:,0],xy[:,1],c=delta,norm=Normalize(0,vmax),s=15,cmap='viridis')
        contour_state=dict(initial_vertices=geo['vertices'],initial_indices=geo['indices'],common_phase_nodes=s['phase'][-1]);contours(ax,contour_state)
        first=report['cases'][name]['first_nodes'];ax.scatter([p['x'] for p in first],[p['y'] for p in first],s=60,facecolors='none',edgecolors='red')
        ax.set(xlim=(.6,.84),ylim=(.18,.82),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=LABEL[name]+': actual cap increment')
        fig.colorbar(im,ax=ax,shrink=.8,label='Δφ [1]')
        axes[1,0].plot(s['time'],s['history'][:,2],color=COLORS[name],label=LABEL[name])
        angle=np.tile(np.arange(-90.,91.),3);radius=np.repeat([.225,.2881107642904027,.29851520585223124],181)
        mask=abs(radius-.29851520585223124)<1e-12
        axes[1,1].plot(angle[mask],s['arc_0.16'][mask,2],color=COLORS[name],label=LABEL[name]+' at .16 s')
        axes[1,1].plot(angle[mask],s['arc_0.135'][mask,2],':',color=COLORS[name],alpha=.6)
    for name,style in [('h_quarter','-'),('half_quarter','--')]:
        rows=report['comparisons']['pairs'][name];rows=[r for r in rows if r['growth_L2_relative'] is not None]
        axes[1,2].plot([r['time'] for r in rows],[100*r['growth_L2_relative'] for r in rows],style,marker='o',label=name.replace('_',' vs '))
    axes[1,0].set(title='Growth history: initial seed subtracted',ylabel='Peak cap Δφ [1]');switches(axes[1,0]);axes[1,0].legend()
    axes[1,1].set(title='Common physical-kernel driving, r=.298515 m',xlabel='Angle from physical +x [degrees]',ylabel='Y [Pa]',xlim=(-90,90));axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=8)
    axes[1,2].set(title='Difference relative to the measured h/4 growth',xlabel='Physical time [s]',ylabel='Cap growth relative L2 difference [%]');axes[1,2].grid(alpha=.2);axes[1,2].legend()
    fig.suptitle('Identical initial seed on the same mesh; no phase-history interpolation\nWhite phase contours; red circles mark first cap activation. Dotted driving: .135 s.',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(output/f'temporal-response.{ext}',dpi=175)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);main(p.parse_args().output)
