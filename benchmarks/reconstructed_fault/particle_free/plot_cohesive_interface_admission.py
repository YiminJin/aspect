#!/usr/bin/env python3
"""Only computed states are shown; surface a is not bulk phase phi."""
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from cohesive_interface_admission import OUT,Kernel
plt.rcParams.update({'font.size':10,'figure.dpi':150})
def save(fig,name):fig.savefig(OUT/name,bbox_inches='tight');plt.close(fig)
def records(name):return np.genfromtxt(OUT/name,delimiter=',',names=True,dtype=None,encoding='utf-8')
def geometry(ax,installed=True):
    ax.plot([.25,.5],[0,0],color='0.3',lw=3,label='old tied horizontal segment')
    ax.plot([.5,.75],[0,.125],color='#0072b2',lw=3,ls='-' if installed else '--',label='prespecified append')
    ax.scatter([.25,.5,.75],[0,0,.125],c='k',s=12,zorder=5);ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')

def main():
    k=Kernel('native-16-axial');h=records('histories.csv');p=records('profiles.csv');mesh=np.genfromtxt(k.path/'mesh.csv',delimiter=',',names=True);polys=np.c_[mesh['x_m'],mesh['y_m']].reshape(-1,4,2)[:,[0,1,3,2]]
    fig,ax=plt.subplots(figsize=(7,6));ax.add_collection(PolyCollection(polys,facecolors='none',edgecolors='0.8',linewidths=.5));geometry(ax)
    ax.set(xlim=(-.08,1.12),ylim=(-.58,.58),title='Prepared confined block: prescribed-path admission')
    ax.annotate('',(1.09,0),(1.01,0),arrowprops={'arrowstyle':'->','color':'#d55e00'});ax.annotate('',(.5,.42),(.5,.54),arrowprops={'arrowstyle':'->','color':'#d55e00'});ax.annotate('',(.5,-.42),(.5,-.54),arrowprops={'arrowstyle':'->','color':'#d55e00'})
    ax.text(.02,.42,'u=(γx, −γy) on all boundaries\np₀=10 kPa; γ̇=.001/s to .65 s\nthen displacement hold to .66 s',fontsize=9)
    ax.text(.03,-.42,'Insert tied append at .25 s; zero new storage.\nNo free path; no opening at the closed corner.',fontsize=9);ax.legend(loc='upper right',fontsize=8);save(fig,'specimen.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');fine=p[p['case']=='fine'];times=[.58,.585,.6,.625,.65,.66]
    for i,t in enumerate(times):
        z=fine[abs(fine['time_s']-t)<1e-10];c=plt.cm.viridis(i/(len(times)-1));label=f'{t:.3f} s'
        for ax,key,scale in [(axes[0,0],'damage',1),(axes[0,1],'jump_m',1e6),(axes[1,0],'velocity_m_s',1e6),(axes[1,1],'normal_pressure_Pa',.001)]:ax.plot(z['x_m'],z[key]*scale,'o-',ms=3,color=c,label=label)
    axes[0,0].set(ylabel='surface damage a',title='Process-zone activation; a=1 never reached',ylim=(-.001,.03));axes[0,0].axhline(.01,color='gray',ls=':');axes[0,0].axhline(.02,color='gray',ls=':')
    axes[0,1].set(ylabel='signed geometric jump (μm)',title='Closed corner stays at zero jump')
    axes[1,0].set(ylabel='signed slip velocity (μm/s)',title='Hold can retain slip and damage growth')
    axes[1,1].set(ylabel='current normal pressure (kPa)',title='Full current mixed pressure retained')
    for ax in axes.flat:ax.set_xlabel('physical x (m)');ax.axvline(.5,color='gray',ls='--');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);save(fig,'profiles.png')
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
    for t,color in [(.58,'#0072b2'),(.66,'#d55e00')]:
        z=fine[(abs(fine['time_s']-t)<1e-10)&(fine['branch']==1)]
        axes[0].plot(z['x_m'],z['physical_plane_shear_Pa']/1000,'o-',ms=3,color=color,label=f'physical FE plane shear, {t:.2f} s')
        axes[0].plot(z['x_m'],z['total_resistance_Pa']/1000,'--',color=color,label=f'weak Gauss resistance, {t:.2f} s')
        axes[1].plot(z['x_m'],z['cohesive_traction_Pa']/1000,'o-',ms=3,color=color,label=f'bond traction, {t:.2f} s')
    axes[0].set(ylabel='t·τn / resistance (kPa)',title='Oblique-plane shear is nonzero even when Cartesian τxy=0')
    axes[1].set(ylabel='recoverable cohesive q (kPa)',xlabel='physical x on append (m)',title='Intact traction is a bounded reaction; its storage is zero')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'plane_traction.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for case,style in [('coarse','--'),('fine','-')]:
        z=h[h['case']==case];label='Δt=.005 s' if case=='coarse' else 'Δt=.0025 s'
        axes[0,0].plot(z['time_s'],z['integrated_damage_m'],style,label=label);axes[0,1].plot(z['time_s'],z['peak_damage'],style,label=label)
        axes[1,0].plot(z['time_s'],z['cumulative_absolute_remainder_J_m'],style,label=label+' sum |remainder|');axes[1,0].plot(z['time_s'],z['cumulative_signed_remainder_J_m'],style,alpha=.5,label=label+' signed')
        axes[1,1].plot(z['time_s'],z['cumulative_fracture_J_m'],style,label=label+' new fracture');axes[1,1].plot(z['time_s'],z['cumulative_sliding_J_m'],style,label=label+' sliding loss')
    axes[0,0].set(ylabel='∫a ds (m)',title='Temporal response; no spatial-accuracy claim');axes[0,1].set(ylabel='peak surface damage a',title='A narrow event bracket is not exact physical onset')
    inset=axes[0,1].inset_axes([.14,.43,.45,.47])
    for case,style in [('coarse','--'),('fine','-')]:
        z=h[h['case']==case];inset.plot(z['time_s'],z['peak_damage'],style)
    inset.set(xlim=(.58,.5875),ylim=(-.0001,.0025));inset.tick_params(labelsize=7)
    axes[1,0].set(ylabel='physical-work remainder (J/m)',title='Signed and absolute errors: no cancellation hidden')
    axes[1,1].set(ylabel='energy (J/m)',title='Sliding loss is small; its 1% screen FAILS');axes[1,1].set_yscale('symlog',linthresh=1e-6)
    for ax in axes.flat:ax.set_xlabel('physical time (s)');ax.grid(alpha=.2);ax.axvline(.65,color='gray',ls=':');ax.legend(fontsize=7)
    axes[0,1].legend(loc='lower right',fontsize=7)
    save(fig,'temporal_energy.png')
    fields=records('physical_fields.csv');fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');means=[]
    for t in [.58,.66]:
        z=fields[abs(fields['time_s']-t)<1e-10];w=z['weight_m2'].reshape(-1,36);means.append((w*z['tau_xy_Pa'].reshape(-1,36)).sum(axis=1)/w.sum(axis=1))
    limit=max(max(abs(x)) for x in means)
    for ax,t,m in zip(axes,[.58,.66],means):
        pc=PolyCollection(polys,array=m,cmap='RdBu_r',clim=(-limit,limit));ax.add_collection(pc);geometry(ax);ax.set(xlim=(0,1),ylim=(-.5,.5),title=f'Physical τxy, t={t:.2f} s')
    fig.colorbar(pc,ax=axes,label='τxy (Pa)',shrink=.7);save(fig,'physical_stress.png')
    # Frame states are exact accepted fine histories (not interpolation).
    states=json.loads((OUT/'fine-states.json').read_text());states=[s for s in states if s['clock_s']>=.575]
    raw=np.genfromtxt(k.path/'field-response.csv',delimiter=',',names=True).reshape(-1,len(k.nodes)+1);weights=raw['weight_m2'][:,0].reshape(-1,36);maps=[]
    for s in states:
        tau=raw['tau_xy_Pa']@np.r_[s['wall_memory'],s['memory']];maps.append((weights*tau.reshape(-1,36)).sum(axis=1)/weights.sum(axis=1))
    limit=max(max(abs(m)) for m in maps);fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');pc=PolyCollection(polys,array=maps[0],cmap='RdBu_r',clim=(-limit,limit));axes[0].add_collection(pc);geometry(axes[0]);axes[0].set(xlim=(0,1),ylim=(-.5,.5));fig.colorbar(pc,ax=axes[0],label='physical τxy (Pa)',shrink=.7)
    line,=axes[1].plot(k.s['x_m'],states[0]['histories']['damage'],'o-',ms=4);axes[1].set(xlabel='x (m)',ylabel='surface damage a',ylim=(-.001,.03));axes[1].axhline(.01,color='gray',ls=':');axes[1].axhline(.02,color='gray',ls=':');axes[1].axvline(.5,color='gray',ls='--');axes[1].grid(alpha=.2);title=fig.suptitle('')
    def update(i):
        s=states[i];pc.set_array(maps[i]);line.set_ydata(s['histories']['damage']);title.set_text(f't={s["clock_s"]:.8f} s; γ={s["gamma"]:.6f}; '+('displacement hold' if s['clock_s']>.65 else 'axial loading')+' — no full decohesion');return pc,line,title
    FuncAnimation(fig,update,frames=len(states)).save(OUT/'admission.gif',writer=PillowWriter(fps=8));plt.close(fig)
if __name__=='__main__':main()
