#!/usr/bin/env python3
"""Compact physical evidence for temporary views and persistent formation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from cohesive_interface_candidate_admission import WORK,OUT,SOURCE,Kernel,open_state,counts,zero_tail
from cohesive_interface_load_turn import functionals
from cohesive_interface_path import graph,save,table,axes

LABELS={'candidates2':'2 temporary cells','candidates8':'8 temporary cells',
    'half-step':'Half timestep','cadence-half':'Half refresh cadence (suffix)'}
COLORS=['#0072b2','#d55e00','#009e73','#cc79a7']


def rows(label):return np.genfromtxt(OUT/(label+'-history.csv'),delimiter=',',names=True)


def states(label):
    folder=WORK/label;values=json.loads((folder/'snapshots.json').read_text())
    for p in folder.glob('after-formation-*.json'):values.append(json.loads(p.read_text()))
    values.append(json.loads((folder/'final.json').read_text()))
    return {s['clock_s']:s for s in values}


def load(s):
    k=Kernel(Path(s['kernel_path']),s['geometry']);k.anchor(s['formed_arcs']);return k


def arclength(k):
    z,w=np.polynomial.legendre.leggauss(16);z=(z+1)/2;w=w/2;result=[]
    for x in k.s['x_m']:
        total=0.
        for a in k.arcs:
            right=min(x,a['end'])
            if right<=a['x']:break
            u=(right-a['x'])/(a['end']-a['x']);_,p=graph(a,u*z);total+=(right-a['x'])*(w@np.sqrt(1+p*p))
            if x<=a['end']:break
        result.append(total)
    return np.array(result)


def evidence():
    summary=[];profiles=[];loads=[];initials={};finals={}
    for label in LABELS:
        folder=WORK/label;ik,initial=open_state(folder/'initial.json');s=json.loads((folder/'final.json').read_text());k=load(s)
        h=rows(label);fi=functionals(ik,initial);f=functionals(k,s);q0,n0=counts(s);c=k.coefficients(s);a=np.array(s['histories']['damage'])
        initials[label]=fi;finals[label]=(s,k)
        d=json.loads((OUT/(label+'-summary.json')).read_text())
        summary.append(dict(label=label,initial_s=initial['clock_s'],final_s=s['clock_s'],steps=len(h),
            persistent_cells_added=s['formed_arcs']-initial['formed_arcs'],physical_publications=d['physical_publications'],
            new_damage_integral_m=f['new_damage_integral_m'],growth_m=f['new_damage_integral_m']-fi['new_damage_integral_m'],
            signed_slip_increment_m2=float(h['integrated_slip_m2'][-1]-ik.w@np.array(initial['histories']['jump'])),
            process_front_arc_m=f['process_front_arc_m'],half_front_arc_m=f['half_front_arc_m'],
            developed_front_available=f['new_full_front_available'],formed_length_m=float(sum(k.w[:q0])),
            physical_absolute_error_J_m=d['physical_error_J_m'],physical_vs_sliding=d['physical_vs_sliding'],physical_vs_fracture=d['physical_vs_fracture'],
            maximum_weak_Pa=float(max(h['weak_error_Pa'])),maximum_ledger_J_m=float(max(abs(h['ledger_error_J_m']))),
            activation_retries=int(sum(h['activation_retries'])),completion_retries=int(sum(h['extension_retries'])),
            maximum_transfer_stress_Pa=float(max(h['transfer_stress_Pa'])),maximum_transfer_storage_J_m=float(max(abs(h['transfer_storage_J_m']))),seconds=d['seconds']))
        arc_s=arclength(k);velocity=k.Q@np.array(s['nodal_rate']);normal=10000+k.N@c
        for q in range(len(k.s)):
            profiles.append(dict(label=label,time_s=s['clock_s'],s_arc_m=float(arc_s[q]),x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),
                weight_m=float(k.w[q]),persistent=q<q0,damage=float(a[q]),jump_m=s['histories']['jump'][q],theta=s['histories']['theta'][q],
                velocity_m_s=float(velocity[q]),variational_normal_Pa=float(normal[q]),
                tangent_deg=float(np.rad2deg(np.arctan2(k.s['ty'][q],k.s['tx'][q])))))
        for name,centre,width in [('old',4.5,.5),('near_original_tip',6.25,.15),('new_material',6.8,.2)]:
            v=np.exp(-.5*((k.nodes['x_m']-centre)/width)**2);v[n0:]=0;test=k.Q@v;mass=k.w@test
            loads.append(dict(label=label,test=name,time_s=s['clock_s'],centre_x_m=centre,sigma_x_m=width,test_mass_m=float(mass),
                weak_shear_Pa=float(-(k.K@c)[2:]@v/mass),normal_perturbation_Pa=float(k.w@(test*(k.N@c))/mass)))
        ik.close()
    table(OUT/'comparison.csv',summary);table(OUT/'surface-profiles.csv',profiles);table(OUT/'weak-loads.csv',loads)
    a,ka=finals['candidates2'];b,kb=finals['candidates8'];q0,n0=counts(a)
    assert a['geometry'][:a['formed_arcs']]==b['geometry'][:b['formed_arcs']]
    field_a=np.einsum('ijk,j->ik',ka.field,ka.coefficients(a));field_b=np.einsum('ijk,j->ik',kb.field,kb.coefficients(b))
    delta=field_a-field_b;metric=np.array([1.,1.,1.,2.]);tau_norm=np.sqrt(delta[:,:4]**2@metric)
    sigma_delta=delta[:,:4].copy();sigma_delta[:,:3]-=delta[:,4,None]
    sigma_norm=np.sqrt(sigma_delta**2@metric);volume=ka.centers['area_m2']
    relative=float(np.sqrt(volume@(tau_norm**2)/(volume@(field_b[:,:4]**2@metric))))
    history={key:float(max(abs(np.array(a['histories'][key][:q0])-np.array(b['histories'][key][:q0])))) for key in a['histories']}
    save(OUT/'reserve-independence.json',dict(persistent_geometry_bitwise=True,persistent_record_ids_identical=a['record_ids']==b['record_ids'],
        maximum_physical_stress_or_pressure_difference_Pa=float(max(abs(field_a-field_b).ravel())),Gauss_history_differences=history,
        maximum_full_plane_strain_deviator_difference_Pa=float(max(tau_norm)),maximum_total_Cauchy_difference_Pa=float(max(sigma_norm)),
        full_tensor_volume_L2_relative=relative,maximum_pressure_perturbation_difference_Pa=float(max(abs(delta[:,4]))),
        bulk_memory_difference_m=float(max(abs(np.array(a['memory'][:n0])-np.array(b['memory'][:n0])))),
        scope='Same incoming evolved state, fixed physical/numerical lengths and timestep; only temporary extent differs.'))
    diffs=[];baseline=summary[0]
    for d in summary[1:]:
        base_rows=rows('candidates2');h=rows(d['label']);start=d['initial_s'];prefix=base_rows[base_rows['time_s']<=start+1e-12]
        init_growth=initials[d['label']]['new_damage_integral_m'];base_growth=baseline['new_damage_integral_m']-init_growth
        diffs.append(dict(comparison=d['label'],growth_relative=float(d['growth_m']/base_growth-1),
            process_front_difference_m=d['process_front_arc_m']-baseline['process_front_arc_m'],
            half_front_difference_m=d['half_front_arc_m']-baseline['half_front_arc_m'],
            final_new_damage_relative=d['new_damage_integral_m']/baseline['new_damage_integral_m']-1))
    table(OUT/'uncertainty.csv',diffs)
    for s,k in finals.values():k.close()
    return summary


def new_material_storage():
    values=[]
    from check_cohesive_interface_kink import GC,C0
    for label in LABELS:
        for time,s in sorted(states(label).items()):
            k=load(s);start=sum(identifier.startswith('retained:') for identifier in s['record_ids'])
            a=np.array(s['histories']['damage']);e=np.array(s['histories']['jump'])-np.array(s['histories']['reference'])
            B=.5*(1-a)/(C0*np.where(a==0,1,a))*e*e
            values.append(dict(label=label,time_s=time,new_material_fracture_J_m=float(GC*k.w[start:]@a[start:]),
                new_material_cohesive_J_m=float(k.w[start:]@B[start:]),additional_metadata_charge_J_m=0.))
            k.close()
    table(OUT/'new-material-storage.csv',values)
    return values


def onset_control():
    if not (WORK/'cadence-onset/final.json').exists():return
    b=json.loads((WORK/'cadence-onset/final.json').read_text());t=b['clock_s'];a=states('candidates2')[t]
    q0,_=counts(a);old=json.loads((WORK/'candidates2/events.json').read_text());new=json.loads((WORK/'cadence-onset/events.json').read_text())
    baseline=rows('candidates2');alternative=rows('cadence-onset');r=baseline[np.argmin(abs(baseline['time_s']-t))]
    errors={key:float(max(abs(np.array(a['histories'][key][:q0])-np.array(b['histories'][key][:q0])))) for key in a['histories']}
    save(OUT/'onset-cadence.json',dict(initial_s=.410,final_s=t,cadences_s=[.0025,.00125],
        old_admission_clocks_s=[e['time_s'] for e in old if e['time_s']<=t],new_admission_clocks_s=[e['time_s'] for e in new],
        first_admission_bracket_s=[new[0]['selected_from_time_s'],new[0]['time_s']],
        persistent_geometry_identical=a['geometry'][:a['formed_arcs']]==b['geometry'][:b['formed_arcs']],
        physical_history_errors=errors,new_damage_integral_difference_m=float(alternative['new_damage_integral_m'][-1]-r['new_damage_integral_m'])))


def draw(ax,k,s,title,limits=(5.8,8.8,-.25,.75)):
    formed=s['formed_arcs'];q0,_=counts(s)
    for i,a in enumerate(s['geometry']):
        z=np.linspace(0,1,30);y,_=graph(a,z);ax.plot(a['x']+(a['end']-a['x'])*z,y,
            color='#005f9e' if i<formed else '.65',ls='-' if i<formed else '--',lw=2,zorder=1)
    sc=ax.scatter(k.s['x_m'][:q0],k.s['y_m'][:q0],c=s['histories']['damage'][:q0],cmap='viridis',vmin=0,vmax=1,s=24,zorder=3)
    ix=np.arange(0,q0,5);slip=np.array(s['histories']['jump']);ax.quiver(k.s['x_m'][ix],k.s['y_m'][ix]-.1,
        100*slip[ix]*k.s['tx'][ix],100*slip[ix]*k.s['ty'][ix],angles='xy',scale_units='xy',scale=1,color='#aa3377',width=.004)
    f=functionals(k,s)
    for name,marker,color in [('process','^','#e69f00'),('half','D','white')]:
        ax.scatter(f[name+'_front_x_m'],f[name+'_front_y_m'],marker=marker,s=40,c=color,edgecolors='black',linewidths=.5,zorder=4)
    if f['new_full_front_available']:ax.scatter(f['full_front_x_m'],f['full_front_y_m'],s=35,c='red',marker='s',zorder=4)
    tip=s['geometry'][formed-1];y,p=graph(tip,1);ax.scatter(tip['end'],y,marker='|',s=150,c='#005f9e',zorder=5)
    angle,_,_=k.sample(s);ax.quiver(8.25,.45,.3*np.cos(angle),.3*np.sin(angle),angles='xy',scale_units='xy',scale=1,color='#0072b2',width=.005)
    ax.text(8.05,.58,f'Accepted preference {np.rad2deg(angle):.2f}°',fontsize=8)
    ax.set(xlim=limits[:2],ylim=limits[2:],aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title);ax.grid(alpha=.15)
    return sc


def plots():
    storage=new_material_storage()
    fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for label,ax in zip(['candidates2','candidates8'],axs):
        s=json.loads((WORK/label/'final.json').read_text());k=load(s);sc=draw(ax,k,s,f'{LABELS[label]} at t={s["clock_s"]:.5f} s');k.close()
    fig.colorbar(sc,ax=list(axs),shrink=.8,label='Owned surface damage a')
    fig.suptitle('Persistent formed geometry versus temporary reserve\nSolid blue: persistent; dashed grey: temporary; bars: formed endpoint; triangles/diamonds: a=.05/.5 fronts\nPurple: signed slip (1 mm → 0.1 m); no new fully decohered advance in this short window')
    fig.savefig(OUT/'formation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for label,c in zip(LABELS,COLORS):
        h=rows(label);t=h['time_s'];name=LABELS[label]
        axs[0,0].plot(t,h['new_damage_integral_m'],label=name,color=c)
        axs[0,1].plot(t,h['accepted_preferred_deg'],color=c,lw=1)
        new=h['newly_formed_cells']>0;axs[0,1].scatter(t[new],h['selected_preference_deg'][new],color=c,s=20)
        F=np.cumsum(h['fracture_increment_J_m']);err=np.cumsum(abs(h['physical_remainder_J_m']));relative=np.divide(100*err,F,out=np.full(len(F),np.nan),where=F>0)
        axs[1,0].plot(t,relative,color=c)
        material=[r for r in storage if r['label']==label]
        axs[1,1].plot([r['time_s'] for r in material],[r['new_material_fracture_J_m'] for r in material],color=c,label=name)
    for ax,title,y in zip(axs.flat,['Damage in the region beyond original seed','Accepted preference; dots: formation selection',
        'Physical-work remainder / added fracture','Fracture stored on material added after .410 s'],['∫new a ds (m)','Angle (degrees)','Sum absolute remainder (%)','Fracture inventory (J/m)']):
        ax.set(title=title,xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2)
    axs[1,0].axhline(1,color='black',ls=':',lw=1,label='Unchanged 1% screen');axs[0,0].legend(fontsize=8)
    axs[1,1].axhline(0,color='black',ls=':',label='Additional metadata fracture = 0');axs[1,1].legend(fontsize=7)
    fig.suptitle('Native physical evolution and energy-accounted promotion\nCoarse step fails the physical fracture-work screen; half step passes. Suffix budgets exclude the common prefix.')
    fig.savefig(OUT/'response-energy.png',dpi=160);plt.close(fig)
    h=np.genfromtxt(OUT/'surface-profiles.csv',delimiter=',',names=True,dtype=None,encoding=None)
    fig,axs=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    for label,c in zip(LABELS,COLORS):
        d=h[(h['label']==label)&h['persistent']]
        for ax,col in zip(axs,['damage','jump_m','variational_normal_Pa']):ax.plot(d['s_arc_m']-3,d[col],label=LABELS[label],c=c,lw=1.2)
    for ax,title,y in zip(axs,['Actual Gauss damage','Signed physical slip','Current variational normal pressure'],['a','Slip (m)','Pa']):
        ax.set(title=title,xlabel='Arclength from original seed end (m)',ylabel=y,xlim=(0,1.2));ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.savefig(OUT/'profiles.png',dpi=160);plt.close(fig)


def animate():
    labels=['candidates2','candidates8'];ss={label:states(label) for label in labels};times=sorted(set(ss[labels[0]])&set(ss[labels[1]]));cache={}
    fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=list(axs),label='Surface damage a',shrink=.7)
    def frame(t):
        for ax,label in zip(axs,labels):
            ax.clear();s=ss[label][t];name=s['kernel_path']
            if name not in cache:cache[name]=load(s)
            draw(ax,cache[name],s,f'{LABELS[label]}, accepted t={t:.5f} s; imposed axes {s["alpha_deg"]:.2f}°')
        fig.suptitle('Direction commits with physical formation, not with a reserved tail\nSolid blue: persistent; grey dashed: candidate; coloured dots: actual owned damage\nNo interpolated histories; identical physical frame times')
    FuncAnimation(fig,frame,frames=times,interval=180).save(OUT/'formation.gif',writer=PillowWriter(fps=5),dpi=100)
    save(OUT/'animation.json',dict(times_s=times,common_physical_times=True,interpolated_histories=False))
    for k in cache.values():k.close()
    plt.close(fig)


if __name__=='__main__':evidence();onset_control();plots();animate()
