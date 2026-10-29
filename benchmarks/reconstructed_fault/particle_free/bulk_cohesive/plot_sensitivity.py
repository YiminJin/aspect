#!/usr/bin/env python3
"""Common-scale plots of the three selected controls; no history interpolation."""
import argparse,json,csv as csv_module
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_native import csv
from assess_sensitivity import fields

plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.constrained_layout.use':True})
NAMES=['16×32 G6','32×32 G6','32×64 G6','32×64 G8','48×64 G8, Lx=1.5 m']
COLORS=['tab:blue','tab:orange','black','tab:red','tab:green']
STYLES=['-', '-', '-', '--', '-.']
KEYS=['initial','delta','Yweak','Fweak','Rweak']
LABELS=['Initial φ [1]','Phase increment Δφ [1]','N-weighted Y [Pa]','Fracture g_f / mass [Pa]','Total weak g / mass [Pa]']


def save(fig,path):
    for suffix in ['.png','.pdf']:fig.savefig(path.with_suffix(suffix),dpi=190)
    plt.close(fig)


def maps(paths,out):
    data=[fields(p,60) for p in paths]
    fig,ax=plt.subplots(5,5,figsize=(16,13))
    limits={key:(float(min(0,min(np.min(d[key]) for d in data))),float(max(np.max(d[key]) for d in data))) for key in KEYS}
    limits['initial']=(0,.6)
    for row,key in enumerate(KEYS):
        for col,(path,d) in enumerate(zip(paths,data)):
            a=ax[row,col]
            if key=='Yweak':
                f=csv(path/'bulk-0060.csv');lo=0;hi=max(float(np.max(csv(p/'bulk-0060.csv')['driving'])) for p in paths) if col==0 else hi
                im=a.scatter(f['x'],f['y'],c=f['driving'],s=.7,vmin=lo,vmax=hi,cmap='viridis',rasterized=True)
            else:
                im=a.pcolormesh(d['x'],d['y'],d[key],shading='gouraud',vmin=limits[key][0],vmax=limits[key][1],cmap='magma' if key=='delta' else 'viridis')
            a.plot([0,.5],[.5,.5],color='white',lw=1)
            a.axvline(d['x'][-1],color='black',lw=.8);a.plot([.5],[.5],marker='x',c='white',ms=4)
            if key=='delta':
                for y in [.3125,.5,.6875]:a.axhline(y,color='cyan',ls=':',lw=.5)
            a.set(xlim=(0,1.5),ylim=(0,1),aspect='equal')
            if row==0:a.set_title(NAMES[col])
            if col==0:a.set_ylabel(('Actual Gauss Y [Pa]' if key=='Yweak' else LABELS[row])+'\ny [m]')
            if row==4:a.set_xlabel('x [m]')
        fig.colorbar(im,ax=list(ax[row,:]),shrink=.8,pad=.01)
    fig.suptitle('At ramp end t = 0.15 s: identical physical scales; actual Gauss driving and native weak densities')
    save(fig,out/'maps')


def cuts(paths,out):
    data=[fields(p,60) for p in paths];fig,ax=plt.subplots(5,4,figsize=(15,13))
    records=[]
    for c,cut in enumerate([.5,.3125,.6875,'right']):
        for case,d in enumerate(data):
            if cut=='right':coordinate=d['y'];select=(slice(None),-1)
            else:
                j=int(np.argmin(abs(d['y']-cut)));assert d['y'][j]==cut
                coordinate=d['x'];select=(j,slice(None))
            for row,key in enumerate(KEYS):
                ax[row,c].plot(coordinate,d[key][select],c=COLORS[case],ls=STYLES[case],lw=1.2,label=NAMES[case])
            for k,s in enumerate(coordinate):records.append([NAMES[case],cut,float(s)]+[float(d[key][select][k]) for key in KEYS])
        for row in range(5):
            if c==0:ax[row,c].set_ylabel(LABELS[row])
            if row==0:ax[row,c].set_title(f'y = {cut} m' if cut!='right' else 'Actual right face x = Lx')
            if row==4:ax[row,c].set_xlabel('x [m]' if cut!='right' else 'y [m]')
            if cut!='right':ax[row,c].axvline(.5,color='.5',ls=':',lw=.7)
            ax[row,c].grid(alpha=.2)
        if cut!='right':
            for row in range(5):ax[row,c].set_xlim(0 if cut==.5 else .5,1.5)
    for row in range(5):
        lo=min(a.get_ylim()[0] for a in ax[row]);hi=max(a.get_ylim()[1] for a in ax[row])
        for a in ax[row]:a.set_ylim(lo,hi)
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('Common physical cuts: native nodal φ and weak densities (Y averaged with each nodal basis)')
    save(fig,out/'linecuts')
    with (out/'linecuts.csv').open('w') as f:
        w=csv_module.writer(f);w.writerow(['case','cut','coordinate_m']+KEYS);w.writerows(records)


def diagnosis(paths,evidence):
    stats=[json.loads((evidence/(key+'.json')).read_text()) for key in ['coarse','square','fine-gauss6','fine-gauss8','right-distance']]
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for i,s in enumerate(stats):
        t=[h['time'] for h in s['history']]
        for j,region in enumerate(['lower_flank','upper_flank','right_face']):
            ax[0,j].plot(t,[h[region] for h in s['history']],c=COLORS[i],ls=STYLES[i],label=NAMES[i])
        ax[1,2].plot([s['contours'][level]['final'] for level in ['0.001','0.01','0.1','0.3','0.5']],range(5),c=COLORS[i],ls=STYLES[i],marker='.',label=NAMES[i])
    for j,title in enumerate(['Lower cap flank','Upper cap flank','Right-boundary damage']):
        ax[0,j].set(title=title,xlabel='Time [s]',ylabel='Maximum Δφ [1]');ax[0,j].grid(alpha=.2)
    # A shared physical node isolates differing weak contributions despite
    # exactly identical sampled initial phi. No QP history is interpolated.
    for i in [1,2,3,4]:
        d0=fields(paths[i],60);jx=int(np.argmin(abs(d0['x']-.71875)));jy=int(np.argmin(abs(d0['y']-.3125)))
        rows=[]
        for step in range(1,65):
            d=fields(paths[i],step);rows.append([step*.0025,d['Fweak'][jy,jx],d['Yweak'][jy,jx],d['Rweak'][jy,jx]])
        r=np.array(rows)
        ax[1,0].plot(r[:,0],r[:,1],c=COLORS[i],ls=STYLES[i],label=f'g_f/m: {NAMES[i]}')
        ax[1,0].plot(r[:,0],r[:,2],c=COLORS[i],ls=':',label=f'Y_N: {NAMES[i]}')
        ax[1,1].plot(r[:,0],r[:,3],c=COLORS[i],ls=STYLES[i],label=NAMES[i])
    ax[1,0].set(title='Shared flank node (.71875, .3125) m',xlabel='Time [s]',ylabel='Weak contribution density [Pa]');ax[1,0].legend(fontsize=7)
    ax[1,1].set(title='Same node: phase inequality / activation',xlabel='Time [s]',ylabel='Total weak residual / mass [Pa]');ax[1,1].grid(alpha=.2)
    ax[1,2].set(title='Connected centerline fronts: unchanged in time',xlabel='Final front x [m]',yticks=range(5),yticklabels=['.001','.01','.1','.3','.5'],ylabel='Contour level [1]')
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('Separate flank growth, boundary nucleation and connected front position')
    save(fig,evidence/'diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);a=p.parse_args()
    paths=[Path('build-particle-free/bulk-cohesive/runs/primary-qualified'),Path('build-particle-free/bulk-sensitivity/runs/square'),
           Path('build-particle-free/bulk-cohesive/runs/spatial'),Path('build-particle-free/bulk-sensitivity/runs/fine-gauss8'),Path('build-particle-free/bulk-sensitivity/runs/right-distance')]
    maps(paths,a.evidence);cuts(paths,a.evidence);diagnosis(paths,a.evidence)
