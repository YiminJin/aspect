#!/usr/bin/env python3
"""Concise physical profiles and observed evolution, with failed targets visible."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from investigate_cohesive_interface_bulk_contact import WORK, OUT, PATH_WORK
from cohesive_interface_path import PathKernel, graph, functionals

plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.grid':True,'grid.alpha':.2})


def csv(name):return np.genfromtxt(OUT/name,delimiter=',',names=True,dtype=None,encoding=None)


def main():
    r=csv('lifting-reference-profiles.csv');mechanism=json.loads((OUT/'lifting-mechanism.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for n,width,label in [(64,.1875,'aligned support, 64 cells'),(96,.1875,'cut support, 96 cells'),(96,.75,'resolved wider support, 96 cells')]:
        p=r[(r['n']==n)&(r['width_m']==width)&(r['degree']==2)];p=np.sort(p,order='y_m')
        axs[0].plot(p['y_m'],p['shear_Pa'],label=label)
    axs[0].axhline(1e6*(.002-.001/12),color='k',ls=':',label='exact physical shear')
    axs[0].axvline(-.1875,color='.6',ls='--');axs[0].axvline(.1875,color='.6',ls='--')
    axs[0].set(xlim=(-.5,.5),xlabel='cross-fault y [m]',ylabel='physical shear [Pa]',title='Same imposed jump; artificial edges can add stiffness');axs[0].legend(fontsize=8)
    vals=[mechanism['predicted_load_change_Pa'],mechanism['actual_2D_load_change_Pa']]
    axs[1].bar(['independent 1D prediction','actual 2D old-region test'],vals,color=['C0','C1'])
    axs[1].set(ylabel='tangential-load change [Pa]',title='Pure-shear explanation needs no pressure or friction')
    axs[1].text(.02,.95,f'Prediction error {mechanism["prediction_error_Pa"]:.3f} Pa\n({100*mechanism["relative_error"]:.3f}%)',transform=axs[1].transAxes,va='top')
    fig.suptitle('Numerical lifting is a representation, not a physical diffuse band\nIts gradient kink must be cancelled by the continuous bulk approximation')
    fig.savefig(OUT/'lifting-mechanism.png');plt.close(fig)

    r=csv('frozen-profiles.csv');fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for label,text,color in [('narrow64','Q2 64, w=.1875','C0'),('narrow96','Q2 96, w=.125','C1'),
        ('wide96','Q2 96, w=.75','C2'),('wide-Q3','Q3 64, w=.75','C4'),('fitted96','fitted Q2 96','k')]:
        p=r[r['label']==label]
        p=p[(p['x_m']>=5.8)&(p['x_m']<=8.4)]
        axs[0,0].plot(p['x_m'],p['weak_shear_projection_Pa']/1000,label=text,color=color)
        axs[0,1].plot(p['x_m'],p['normal_Pa']/1000,label=text,color=color)
    axs[0,0].set(xlim=(5.8,8.4),xlabel='x along the same curved Γ [m]',ylabel='Q1 dual shear [kPa]',title='Work-conjugate projection; common physical test space');axs[0,0].legend(fontsize=8)
    axs[0,1].set(xlim=(5.8,8.4),xlabel='x [m]',ylabel='closed-contact reaction [kPa]',title='Same normal virtual bubble, .75 m support')
    for label,text,color in [('narrow96','cut Q2 96','C1'),('fitted96','fitted Q2 96','k')]:
        p=r[r['label']==label]
        p=p[(p['x_m']>=5.8)&(p['x_m']<=8.4)]
        axs[1,0].plot(p['x_m'],p['raw_shear_Pa']/1000,label=text,color=color)
        axs[1,1].plot(p['x_m'],p['raw_normal_minus_Pa']/1000,color=color,label=text+' raw −')
        axs[1,1].plot(p['x_m'],p['raw_normal_plus_Pa']/1000,color=color,ls='--',label=text+' raw +')
    axs[1,0].set(xlim=(5.8,8.4),xlabel='x [m]',ylabel='raw physical shear [kPa]',title='Unfiltered sided-mean Cauchy shear');axs[1,0].legend(fontsize=8)
    axs[1,1].set(xlim=(5.8,8.4),xlabel='x [m]',ylabel='raw normal compression [kPa]',title='Contact multiplier does not repair raw trace space');axs[1,1].legend(fontsize=8)
    fig.suptitle('Frozen response: identical geometry, Q1 Maxwell-memory field and loading\nEvery bulk approximation is solved independently; no accepted state is converted')
    fig.savefig(OUT/'frozen-shear-contact.png');plt.close(fig)

    errors=csv('frozen-errors.csv');div=csv('incompressibility.csv')
    labels=['narrow64','narrow96','narrow-Q3','fixed96','wide64','wide96','wide-Q3','offset96','fitted64','fitted96','fitted-Q3']
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    storage=[];d=[]
    for label in labels:
        p=errors[errors['label']==label][0];storage.append(100*p['jump_storage_relative'])
        p=div[(div['label']==label)&(div['region']==2)][0];d.append(p['divergence_RMS'])
    axs[0].bar(labels,storage);axs[0].axhline(2,color='C3',ls='--',label='declared 2% target');axs[0].legend(fontsize=8)
    axs[0].set(ylabel='jump-induced storage difference [%]',title='Large background energy is excluded')
    axs[1].bar(labels,d);axs[1].set_yscale('log');axs[1].set(ylabel='relaxed-strain divergence RMS [1]',title='Same physical tip region; weak constraints pass separately')
    for ax in axs:ax.tick_params(axis='x',rotation=60)
    fig.suptitle('Dominant lifting artifact and remaining cut-interface defect are different\nFitted96 is a checked reference, not exact continuum truth; region area uncertainty ≤.1%')
    fig.savefig(OUT/'bulk-storage-incompressibility.png');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    labels=['short-narrow64','short-narrow96','short-wide64','short-wide96','short-fitted64','short-fitted96']
    for label in labels:
        p=csv(label+'-history.csv');initial=json.loads((OUT/(label+'-evolution.json')).read_text())['initial_functionals']['new_damage_integral_m']
        axs[0,0].plot(p['time_s'],p['new_damage_integral_m']-initial,label=label.removeprefix('short-'))
        axs[0,1].plot(p['time_s'],p['full_front_arc_m']-3,label=label.removeprefix('short-'))
    for ax in axs[0]:ax.axvline(.44,color='.5',ls='--');ax.set_xlabel('time [s]')
    axs[0,0].set(ylabel='growth of new ∫a ds [m]',title='Increments relative to the common incoming state');axs[0,0].legend(fontsize=8)
    axs[0,1].set(ylabel='fully decohered advance along Γ [m]',title='Load ends at .44 s; stress/history remain during hold')
    r=csv('short-profiles.csv')
    for label in ['short-narrow96','short-wide96','short-fitted96','short-fitted96-half']:
        p=r[r['label']==label];axs[1,0].plot(p['x_m'],p['damage'],label=label.removeprefix('short-'))
    for a in [.05,.5,1.]:axs[1,0].axhline(a,color='.75',lw=.5)
    axs[1,0].set(xlim=(6,8.6),ylim=(-.03,1.03),xlabel='x [m]',ylabel='surface damage a',title='Process-zone shape at .445 s');axs[1,0].legend(fontsize=8)
    for label in ['short-wide96','short-wide96-half','short-fitted96','short-fitted96-half']:
        p=csv(label+'-history.csv');axs[1,1].plot(p['time_s'],np.cumsum(abs(p['physical_remainder_J_m']))*1000,label=label.removeprefix('short-'))
    axs[1,1].set(xlabel='time [s]',ylabel='Σ|physical work remainder| [mJ/m]',title='Independent time halving; exact BE ledger kept separately');axs[1,1].legend(fontsize=8)
    fig.suptitle('Short fixed-path evolution: common surface histories; independently reconstructed bulk state\nInitialization storage/force differences are reported separately from physical work')
    fig.savefig(OUT/'short-growth-fronts.png');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for ax,label,root in zip(axs[0],['causal-wide64','causal-wide96'],[WORK,WORK]):
        s=json.loads((root/label/'final.json').read_text());k=PathKernel(root/s['kernel_folder'],s['geometry'],'work')
        xy=np.c_[k.s['x_m'],k.s['y_m']];segments=np.stack([xy[:-1],xy[1:]],axis=1);a=np.array(s['histories']['damage'])
        for arc in s['geometry']:
            z=np.linspace(0,1,60);y,_=graph(arc,z);ax.plot(arc['x']+(arc['end']-arc['x'])*z,y,'--',color='.6',lw=1)
        line=LineCollection(segments,array=.5*(a[:-1]+a[1:]),cmap='viridis',clim=(0,1),linewidths=3);ax.add_collection(line)
        f=functionals(k,s)
        for name,m,color in [('full','*','k'),('half','s','C1'),('process','o','C3')]:ax.plot(f[name+'_front_x_m'],f[name+'_front_y_m'],m,color=color)
        ax.plot(6,0,'ko',ms=3);ax.set(xlim=(5.8,9.1),ylim=(-.15,1),xlabel='x [m]',ylabel='y [m]',title=label+' at .445 s');ax.set_aspect('equal');k.close()
    fig.colorbar(line,ax=list(axs[0]),label='actual surface damage a',shrink=.75)
    for label,root,color in [('work64',PATH_WORK,'C0'),('work96',PATH_WORK,'C1'),('causal-wide64',WORK,'C2'),('causal-wide96',WORK,'C3')]:
        folder=OUT if root==WORK else OUT.parent/'cohesive_interface_path';r=np.genfromtxt(folder/(label+'-history.csv'),delimiter=',',names=True)
        axs[1,0].plot(r['time_s'],r['new_damage_integral_m'],label=label,color=color)
        axs[1,1].plot(r['time_s'],r['full_front_arc_m']-3,label=label,color=color)
    axs[1,0].set(xlabel='time [s]',ylabel='new ∫a ds [m]');axs[1,0].legend(fontsize=8)
    axs[1,1].set(xlabel='time [s]',ylabel='fully decohered advance along Γ [m]')
    fig.suptitle('Fresh causal workflow with fixed numerical supports; R and constitutive law unchanged\n★ developed front, □ a=.5, ○ process front; dashed tied reserve is not fracture advance')
    fig.savefig(OUT/'causal-fronts.png');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');profiles=csv('fresh-profiles.csv')
    for label in ['fresh-wide64','fresh-wide96','fresh-fitted64','fresh-fitted96']:
        r=csv(label+'-history.csv');p=profiles[profiles['label']==label];text=label.removeprefix('fresh-')
        axs[0,0].plot(r['time_s'],r['new_damage_integral_m'],label=text)
        axs[0,1].plot(r['time_s'],r['full_front_arc_m']-3,label=text)
        axs[1,0].plot(p['x_m'],p['damage'],label=text)
        axs[1,1].plot(p['x_m'],p['slip_m']*1000,label=text)
    for name,color in [('wide96','C1'),('fitted96','C3')]:
        r=csv('own-time-'+name+'-history.csv')
        axs[0,0].plot(r['time_s'],r['new_damage_integral_m'],'--',color=color,lw=1,label=name+' half Δt')
        axs[0,1].plot(r['time_s'],r['full_front_arc_m']-3,'--',color=color,lw=1)
    axs[0,0].set(xlim=(.3,.445),xlabel='time [s]',ylabel='new ∫a ds [m]',title='Fresh physical evolution, without imported imbalance');axs[0,0].legend(fontsize=8)
    axs[0,1].set(xlim=(.42,.445),xlabel='time [s]',ylabel='fully decohered advance along Γ [m]',title='Loading / hold transition remains .44 s')
    for ax in axs[0]:ax.axvline(.44,color='.5',ls='--')
    axs[1,0].set(xlim=(6,8.5),xlabel='x [m]',ylabel='surface damage a',title='Actual process zone at .445 s')
    for a in [.05,.5,1.]:axs[1,0].axhline(a,color='.7',ls=':',lw=.5)
    axs[1,1].set(xlim=(3,8.5),xlabel='x [m]',ylabel='signed physical slip [mm]',title='Same histories and physical calibration from initialization')
    fig.suptitle('Independent fresh fixed-path confirmation\nIdentical continuum seed, homogeneous initial stress and zero Maxwell jump memory; all initial weak checks pass')
    fig.savefig(OUT/'fresh-fixed-path.png');plt.close(fig)
    print('Six PNG figures written from indexed CSV/accepted states')


if __name__=='__main__':main()
