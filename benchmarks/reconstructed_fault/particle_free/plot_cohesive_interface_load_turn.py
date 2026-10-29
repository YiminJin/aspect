#!/usr/bin/env python3
"""Actual accepted history records; no interpolated constitutive trajectories."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from cohesive_interface_load_turn import WORK, OUT, Kernel, graph, axes, functionals, table, save, TURN_START, TURN_END, Admission

LABELS={'turn-small64-reproduced':'Turned; reserve 0.1875 m','turn-large64':'Turned; reserve 0.75 m',
        'constant-small64':'Unchanged loading; reserve 0.1875 m','constant-large64':'Unchanged loading; reserve 0.75 m','turn-small96':'Turned; 96² bulk',
        'turn-small-half64':'Turned; half timestep','turn-small-quantum64':'Turned; half append quantum'}
COLOURS=['#0072b2','#d55e00','#009e73','#56b4e9','#cc79a7','#e69f00','#777777']


def histories(label):
    path=OUT/(label+'-history.csv')
    if not path.exists():path=OUT/(label+'-partial.csv')
    return np.genfromtxt(path,delimiter=',',names=True)


def states(label):
    ss=json.loads((WORK/label/'snapshots.json').read_text());last=json.loads((WORK/label/'last-accepted.json').read_text())
    ss.append(last)
    # Keep the latest accepted geometry at a shared physical clock. Zero-time
    # insertions are not additional time samples or physical publications.
    unique={s['clock_s']:s for s in ss}
    return [unique[t] for t in sorted(unique)]


def all_events(label):
    folder=WORK/label;events=json.loads((folder/'events.json').read_text());cfg=json.loads((folder/'run-input.json').read_text())
    if cfg['source_checkpoint']:
        source=Path(cfg['source_checkpoint']).parent.name;initial=json.loads((folder/'initial.json').read_text())
        events=[e for e in all_events(source) if e['generation']<=initial['generation']]+events
    return events


def decorate(ax):
    ax.set(xlabel='Time (s)');ax.axvspan(TURN_START,TURN_END,color='.92',zorder=-1);ax.axvline(.455,color='.65',lw=.6)
    ax.grid(alpha=.18)


def comparisons():
    labels=[v for v in LABELS if (WORK/v/'snapshots.json').exists()];summary=[];samples=[];births=[];segments=[]
    for label in labels:
        h=histories(label);s=states(label)[-1];events=all_events(label)
        run=json.loads((WORK/label/'run-input.json').read_text());failure=WORK/label/'failure.json'
        ratio=lambda a,b:float(a/b) if b else 0.
        summary.append(dict(label=label,n=run['n'],reserve_m=run['reserve_m'],extension_m=run['extension_m'],dt_s=run['dt_s'],
            initial_time_s=float(h['time_s'][0]-h['dt_s'][0]),final_time_s=float(h['time_s'][-1]),steps=len(h),physical_publications=s['physical_publications'],
            geometry_publications=s['publications'],damage_integral_m=float(h['new_damage_integral_m'][-1]),
            developed_advance_m=float(h['full_front_arc_m'][-1]-3),process_advance_m=float(h['process_front_arc_m'][-1]-3),
            full_tangent_deg=float(h['full_front_tangent_deg'][-1]),preferred_deg=float(h['preferred_deg'][-1]),
            physical_absolute_remainder_J_m=float(sum(abs(h['physical_remainder_J_m']))),
            physical_vs_sliding=ratio(sum(abs(h['physical_remainder_J_m'])),sum(h['sliding_loss_J_m'])),
            physical_vs_fracture=ratio(sum(abs(h['physical_remainder_J_m'])),sum(h['fracture_increment_J_m'])),
            max_weak_Pa=float(max(h['weak_error_Pa'])),max_ledger_J_m=float(max(abs(h['ledger_error_J_m']))),
            peak_damage_increment=float(max(h['max_damage_increment'])),minimum_normal_Pa=float(min(h['minimum_normal_Pa'])),
            failure='' if not failure.exists() else json.loads(failure.read_text())['error']))
        for t in [.410,.420,.430,.440,.445,.446,.450,.455,.460]:
            if t>h['time_s'][-1]+1e-10:continue
            r=h[np.argmin(abs(h['time_s']-t))]
            samples.append(dict(label=label,**{k:float(r[k]) for k in ['time_s','alpha_deg','preferred_deg','new_damage_integral_m',
                'process_front_arc_m','full_front_arc_m','full_front_lower_arc_m','full_front_upper_arc_m','full_front_x_m','full_front_y_m',
                'full_front_tangent_deg','integrated_slip_m2','bulk_J_m','bond_J_m']}))
        for ev in events:
            births.append(dict(label=label,time_s=ev['time_s'],generation=ev['generation'],terminal_deg=ev['terminal_deg'],
                preferred_deg=ev['decision']['selected_preference_deg'],reserve_before_m=ev['reserve_before_m'],
                required_peak_Pa=ev['transfer']['new_tied_peak_Pa'],stress_transfer_error_Pa=ev['transfer']['tensor_error_Pa'],
                energy_transfer_J_m=ev['transfer']['energy_change_J_m']))
        k=Kernel(WORK/s['kernel_folder'],s['geometry']);a=np.array(s['histories']['damage']);j=np.array(s['histories']['jump'])
        for i,arc in enumerate(s['geometry']):
            mask=(k.s['x_m']>=arc['x'])&(k.s['x_m']<arc['end']);ev=events[i-1] if i else None
            segments.append(dict(label=label,segment=i,birth_time_s=0 if ev is None else ev['time_s'],final_time_s=s['clock_s'],
                x_begin_m=arc['x'],x_end_m=arc['end'],damage_integral_m=float(k.w[mask]@a[mask]),
                fully_decohered_Gauss_measure_m=float(k.w[mask]@(a[mask]==1)),absolute_jump_integral_m2=float(k.w[mask]@abs(j[mask])),
                peak_damage=float(max(a[mask])),terminal_deg=float(np.rad2deg(np.arctan(arc['b'])))))
        k.close()
    table(OUT/'comparison.csv',summary);table(OUT/'common-times.csv',samples);table(OUT/'births.csv',births);table(OUT/'segment-response.csv',segments)
    differences=[]
    pairs=[('reserve','turn-large64','turn-small64-reproduced'),('bulk mesh','turn-small64-reproduced','turn-small96'),
        ('timestep','turn-small64-reproduced','turn-small-half64'),('append quantum','turn-small64-reproduced','turn-small-quantum64'),
        ('loading, large reserve','constant-large64','turn-large64'),('loading, small reserve','constant-small64','turn-small64-reproduced')]
    for test,left,right in pairs:
        if left not in labels or right not in labels:continue
        a=histories(left);b=histories(right);end=min(a['time_s'][-1],b['time_s'][-1])
        for t in sorted(set([.440,.445,float(end)])):
            if t>end:continue
            ar=a[np.argmin(abs(a['time_s']-t))];br=b[np.argmin(abs(b['time_s']-t))]
            at=a[np.argmin(abs(a['time_s']-.410))];bt=b[np.argmin(abs(b['time_s']-.410))]
            def turn_damage(label,record):
                first=json.loads((WORK/label/'initial.json').read_text())
                if abs(first['clock_s']-.410)<1e-12:
                    k=Kernel(WORK/first['kernel_folder'],first['geometry']);v=functionals(k,first)['new_damage_integral_m'];k.close();return v
                return record['new_damage_integral_m']
            growtha=ar['new_damage_integral_m']-turn_damage(left,at);growthb=br['new_damage_integral_m']-turn_damage(right,bt)
            differences.append(dict(test=test,reference=left,comparison=right,time_s=t,
                damage_difference_m=float(br['new_damage_integral_m']-ar['new_damage_integral_m']),
                damage_relative=float(br['new_damage_integral_m']/ar['new_damage_integral_m']-1),
                growth_since_turn_difference_m=float(growthb-growtha),growth_since_turn_relative=float(growthb/growtha-1),
                process_front_difference_m=float(br['process_front_arc_m']-ar['process_front_arc_m']),
                developed_front_difference_m=float(br['full_front_arc_m']-ar['full_front_arc_m']),
                developed_tangent_difference_deg=float(br['full_front_tangent_deg']-ar['full_front_tangent_deg']),
                preferred_difference_deg=float(br['preferred_deg']-ar['preferred_deg']),
                integrated_slip_difference_m2=float(br['integrated_slip_m2']-ar['integrated_slip_m2']),
                integrated_slip_relative=float(br['integrated_slip_m2']/ar['integrated_slip_m2']-1)))
    if differences:table(OUT/'uncertainty.csv',differences)
    return labels


def histories_plot(labels):
    fig,axs=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for label,c in zip(labels,COLOURS):
        h=histories(label);name=LABELS[label]
        for ax,col in zip(axs.flat,['alpha_deg','preferred_deg','new_damage_integral_m','full_front_arc_m','full_front_tangent_deg','integrated_slip_m2']):
            value=h[col]-(3 if col=='full_front_arc_m' else 0)
            ax.plot(h['time_s'],value,color=c,label=name,lw=1.25)
        for ev in all_events(label):
            axs[0,1].scatter(ev['time_s'],ev['decision']['selected_preference_deg'],s=12,color=c,marker='o')
    titles=['Boundary-strain axes: imposed turn','Accepted sampled maximum-shear preference',
        'New-region irreversible damage integral','Fully decohered connected advance',
        'Actual fully decohered front tangent','Integrated signed physical jump']
    units=['Angle (deg)','Angle (deg)','∫new a ds (m)','Advance along Γ (m)','Tangent (deg)','∫Γ jump ds (m²)']
    for ax,title,unit in zip(axs.flat,titles,units):ax.set(title=title,ylabel=unit,xlim=(.395,.462));decorate(ax)
    axs[1,0].legend(fontsize=7);fig.suptitle('Physical loading turn versus permanently committed tied reserve\nDots: actual append decisions; shaded interval: gradual boundary-loading turn')
    fig.savefig(OUT/'response.png',dpi=160);plt.close(fig)


def draw_state(ax,k,s,title):
    x=k.s['x_m'];y=k.s['y_m'];a=np.array(s['histories']['damage']);jump=np.array(s['histories']['jump'])
    for arc in s['geometry']:
        z=np.linspace(0,1,80);yy,_=graph(arc,z);ax.plot(arc['x']+(arc['end']-arc['x'])*z,yy,color='.75',lw=2)
    sc=ax.scatter(x,y,c=a,cmap='viridis',vmin=0,vmax=1,s=18,zorder=3,edgecolors='none')
    full=a>=1;ax.scatter(x[full],y[full],facecolors='none',edgecolors='red',s=32,lw=.65,zorder=4)
    tied=a==0;ax.scatter(x[tied],y[tied],c='.65',s=8,zorder=4)
    ix=np.arange(0,len(x),6);ax.quiver(x[ix],y[ix]-.11,1e3*jump[ix]*k.s['tx'][ix],1e3*jump[ix]*k.s['ty'][ix],
        angles='xy',scale_units='xy',scale=20,width=.004,color='#aa3377',zorder=2)
    tensor=k.probe@k.coefficients(s);angle,_=axes(tensor,np.arctan(s['geometry'][-1]['b']));f=functionals(k,s)
    ax.scatter([f['full_front_x_m']],[f['full_front_y_m']],marker='s',s=32,c='red',zorder=5)
    ax.scatter([f['process_front_x_m']],[f['process_front_y_m']],marker='^',s=42,c='#e69f00',edgecolors='black',linewidths=.4,zorder=5)
    ax.scatter([f['half_front_x_m']],[f['half_front_y_m']],marker='D',s=26,c='white',edgecolors='black',linewidths=.6,zorder=5)
    ax.quiver(8.75,.45,.35*np.cos(angle),.35*np.sin(angle),angles='xy',scale_units='xy',scale=1,color='#0072b2',width=.005)
    ax.text(8.65,.59,f'Preferred {np.rad2deg(angle):.1f}°',fontsize=8,color='#0072b2')
    ax.plot([6,6],[-.18,.65],'k:',lw=.9);ax.text(6.01,.6,'Old seed end',fontsize=7)
    ax.set(title=title,xlim=(5.85,9.45),ylim=(-.2,.72),xlabel='x (m)',ylabel='y (m)',aspect='equal');ax.grid(alpha=.15)
    return sc


def geometry():
    labels=['constant-large64','turn-large64','turn-small64-reproduced'];fig,axs=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    for label,ax in zip(labels,axs):
        ss=states(label);s=min(ss,key=lambda v:abs(v['clock_s']-.440));k=Kernel(WORK/s['kernel_folder'],s['geometry'])
        sc=draw_state(ax,k,s,f'{LABELS[label]}, accepted t={s["clock_s"]:.5f} s');k.close()
    fig.colorbar(sc,ax=list(axs),shrink=.7,label='Surface damage a (owned Gauss histories)')
    fig.suptitle('Actual damaged geometry at a common time\nRed square: a=1 front; white diamond: a=.5; orange triangle: a=.05; grey: tied reserve\nPurple arrows: signed slip (10 mm → 0.5 m); blue arrow: sampled direction key')
    fig.savefig(OUT/'geometry.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained');ax.add_patch(plt.Rectangle((0,-6),12,12,fill=False))
    ax.plot([3,6],[0,0],color='k',lw=3,label='Initial prepared seed')
    s=states('turn-large64')[-1]
    for arc in s['geometry'][1:]:
        z=np.linspace(0,1,40);y,_=graph(arc,z);ax.plot(arc['x']+(arc['end']-arc['x'])*z,y,color='#0072b2',lw=2)
    ax.text(.3,5.35,'uD = (a₀ E₀ + a₁ E₁)x on all outer boundaries\np₀ = 10000 Pa; mean p′ = 0; no stress/history rotation',fontsize=10)
    ax.text(.3,-4.8,'Physical persistence R = 0.75 m\nNumerical vertical lifting half-width w = 0.75 m\nReserve S = 0.1875 or 0.75 m (separate scheduling quantity)',fontsize=10)
    ax.set(xlim=(-.2,12.2),ylim=(-6.2,6.2),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Finite-wall specimen and physical loading');ax.legend(loc='lower right')
    fig.savefig(OUT/'specimen.png',dpi=150);plt.close(fig)


def export_surface():
    rows=[];probes=[];cache={}
    for label in ['constant-large64','turn-large64','turn-small64-reproduced']:
        ss=states(label);events=all_events(label)
        chosen={round(min(ss,key=lambda v:abs(v['clock_s']-t))['clock_s'],12) for t in np.arange(.4,.461,.005)}
        for s in ss:
            if round(s['clock_s'],12) not in chosen:continue
            key=s['kernel_folder']
            if key not in cache:cache[key]=Kernel(WORK/key,s['geometry'])
            k=cache[key];c=k.coefficients(s);normal=10000+k.N@c;tensor=k.probe@c;f=functionals(k,s)
            arc=s['geometry'][-1];y,p=graph(arc,1.);tangent=np.array([1,p])/np.sqrt(1+p*p)
            probes.append(dict(label=label,time_s=s['clock_s'],boundary_alpha_deg=s['alpha_deg'],gamma=s['gamma'],
                disk_x_m=float(arc['end']+.75*tangent[0]),disk_y_m=float(y+.75*tangent[1]),
                tau_xx_Pa=tensor[0],tau_yy_Pa=tensor[1],tau_zz_Pa=tensor[2],tau_xy_Pa=tensor[3],pprime_Pa=tensor[4],
                sigma_xx_Pa=tensor[0]-10000-tensor[4],sigma_yy_Pa=tensor[1]-10000-tensor[4],
                preferred_deg=float(np.rad2deg(axes(tensor,np.arctan(arc['b']))[0])),
                developed_front_tangent_deg=f['full_front_tangent_deg'],process_front_tangent_deg=f['process_front_tangent_deg']))
            for q,x in enumerate(k.s['x_m']):
                segment=next(i for i,a in enumerate(s['geometry']) if x<a['end'])
                birth=0. if segment==0 else events[segment-1]['time_s']
                rows.append(dict(label=label,time_s=s['clock_s'],x_m=float(x),y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),
                    segment=segment,birth_time_s=birth,tangent_deg=float(np.rad2deg(np.arctan2(k.s['ty'][q],k.s['tx'][q]))),
                    damage=s['histories']['damage'][q],jump_m=s['histories']['jump'][q],
                    signed_velocity_m_s=float((k.Q@s['nodal_rate'])[q]),normal_reaction_Pa=float(normal[q])))
    table(OUT/'surface-history.csv',rows);table(OUT/'sampled-stress.csv',probes)
    for k in cache.values():k.close()


def energy(labels):
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label,c in zip(labels,COLOURS):
        h=histories(label);name=LABELS[label]
        axs[0].plot(h['time_s'],np.cumsum(h['trapezoid_work_J_m']),color=c,label=name)
        axs[1].plot(h['time_s'],np.cumsum(h['sliding_loss_J_m']),color=c)
        axs[2].plot(h['time_s'],np.cumsum(abs(h['physical_remainder_J_m'])),color=c)
    for ax,title in zip(axs,['Cumulative physical trapezoidal work','Cumulative friction + radiation loss','Sum of absolute physical-work remainders']):
        decorate(ax);ax.set(title=title,ylabel='Energy per out-of-plane length (J/m)')
    axs[0].legend(fontsize=6);fig.savefig(OUT/'energy.png',dpi=150);plt.close(fig)


def reserve_plot():
    labels=['turn-small64-reproduced','turn-large64','turn-small-quantum64'];target=.44425
    fig,axs=plt.subplots(3,1,figsize=(10,9),layout='constrained');rows=[];metrics=[]
    for label,c in zip(labels,COLOURS):
        ss=states(label);s=max((v for v in ss if v['clock_s']<=target+1e-12),key=lambda v:v['clock_s'])
        assert not any(s['clock_s']<e['time_s']<target for e in all_events(label))
        k=Kernel(WORK/s['kernel_folder'],s['geometry'])
        # Exact native replay to a common physical time, from this run's own
        # accepted checkpoint. No geometry insertion occurs in this window.
        while s['clock_s']<target-1e-12:s=Admission(k).publish(s,k.trial(s,round(s['clock_s']+.00025,12)))
        expected=histories(label);r=expected[np.argmin(abs(expected['time_s']-target))]
        f=functionals(k,s);assert abs(f['new_damage_integral_m']-r['new_damage_integral_m'])<1e-11
        save(WORK/label/'common-044425.json',s)
        a=np.array(s['histories']['damage']);slip=np.array(s['histories']['jump']);new=k.s['x_m']>6
        arc_s=[];birth=[];pref=[];z,w=np.polynomial.legendre.leggauss(16);z=(z+1)/2;w=w/2;events=all_events(label)
        for x in k.s['x_m']:
            distance=0.;b=0.;p=20.
            for i,arc in enumerate(s['geometry']):
                right=min(x,arc['end']);u=(right-arc['x'])/(arc['end']-arc['x'])
                if u<=0:break
                _,slope=graph(arc,u*z);distance+=(right-arc['x'])*(w@np.sqrt(1+slope*slope))
                if x<arc['end']:
                    if i:b=events[i-1]['time_s'];p=events[i-1]['decision']['selected_preference_deg']
                    break
            arc_s.append(distance-3);birth.append(b);pref.append(p)
        arc_s=np.array(arc_s);theta=np.rad2deg(np.arctan2(k.s['ty'],k.s['tx']))
        axs[0].plot(arc_s[new],theta[new],ls='--',color=c,lw=1)
        axs[0].scatter(arc_s[new&(a>=.05)],theta[new&(a>=.05)],s=12,color=c,label=LABELS[label])
        axs[1].plot(arc_s[new],a[new],color=c,lw=1,marker='.',ms=3)
        axs[2].step(arc_s[new],np.array(birth)[new],where='mid',color=c,lw=1)
        for q in np.flatnonzero(new):rows.append(dict(label=label,time_s=target,s_from_seed_end_m=float(arc_s[q]),x_m=float(k.s['x_m'][q]),
            y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),tangent_deg=float(theta[q]),damage=float(a[q]),jump_m=float(slip[q]),
            birth_time_s=birth[q],preferred_at_birth_deg=pref[q]))
        for i,arc in enumerate(s['geometry'][1:],1):
            mask=(k.s['x_m']>=arc['x'])&(k.s['x_m']<arc['end']);ev=events[i-1]
            metrics.append(dict(label=label,time_s=target,segment=i,birth_s=ev['time_s'],terminal_deg=ev['terminal_deg'],
                preferred_birth_deg=ev['decision']['selected_preference_deg'],peak_damage=float(max(a[mask])),
                integrated_damage_m=float(k.w[mask]@a[mask]),full_measure_m=float(k.w[mask]@(a[mask]==1))))
        for ax in axs[:2]:ax.axvline(f['full_front_arc_m']-3,color=c,lw=.7,alpha=.5)
        k.close()
    axs[0].set(ylabel='Actual tangent (deg)',title='Solid points: a≥.05; dashed: complete committed geometry; vertical lines: developed fronts')
    axs[0].axhline(.5,ls=':',c='black',label='Current sampled preference ≈0.5°');axs[0].legend(fontsize=8)
    axs[1].set(ylabel='Owned Gauss damage a',title='New material that actually damaged, separately from virgin reserve')
    axs[2].set(ylabel='Geometry birth time (s)',title='Permanent direction commitment precedes damage');axs[2].axhspan(.410,.440,color='.9')
    for ax in axs:ax.set(xlabel='Arclength from original seed endpoint (m)',xlim=(0,2.5));ax.grid(alpha=.2)
    fig.suptitle('Reserve and scheduling at common accepted t=0.44425 s\nR=0.75 m and numerical support w=0.75 m fixed; no existing geometry/history rewritten')
    fig.savefig(OUT/'reserve.png',dpi=160);plt.close(fig);table(OUT/'common-shape.csv',rows);table(OUT/'common-segments.csv',metrics)


def frozen_plot():
    h=np.genfromtxt(OUT/'frozen-profiles.csv',delimiter=',',names=True,dtype=None,encoding=None);fig,axs=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    for n,mesh in [(64,'cartesian'),(96,'cartesian'),(64,'fitted'),(96,'fitted')]:
        mask=(h['n']==n)&(h['mesh']==mesh);d=h[mask];label=f'{mesh} {n}'
        for ax,col in zip(axs,['displayed_mass_normalized_weak_shear_Pa','weak_normal_perturbation_Pa','raw_shear_Pa']):ax.plot(d['x_m'],d[col],label=label,lw=1)
    for ax,title in zip(axs,['Displayed lumped weak shear (exact comparisons use common virtual-jump work)',
        'Variational normal-reaction perturbation','Unfiltered mean-sided point shear: diagnostic, not qualified']):
        ax.set(title=title,xlabel='x along common physical Γ (m)',ylabel='Pa',xlim=(5.5,8));ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.savefig(OUT/'loads.png',dpi=150);plt.close(fig)


def animate():
    labels=['constant-large64','turn-large64','turn-small64-reproduced'];ss={l:states(l) for l in labels};cache={}
    fig,axs=plt.subplots(3,1,figsize=(10,8),layout='constrained');times=np.round(np.arange(.400,.445001,.0025),12).tolist()+[.44625]
    frames=[];maximum_error=0.;calls=0
    for t in times:
        frame_states={}
        for label in labels:
            s=max((v for v in ss[label] if v['clock_s']<=t+1e-12),key=lambda v:v['clock_s']);key=s['kernel_folder']
            assert not any(s['clock_s']<e['time_s']<t for e in all_events(label))
            if key not in cache:cache[key]=Kernel(WORK/key,s['geometry'])
            k=cache[key]
            while s['clock_s']<t-1e-12:
                end=min(t,round(s['clock_s']+.00025,12));s=Admission(k).publish(s,k.trial(s,end,not label.startswith('constant')));calls+=1
            original=histories(label);r=original[np.argmin(abs(original['time_s']-t))]
            error=abs(functionals(k,s)['new_damage_integral_m']-r['new_damage_integral_m']);maximum_error=max(maximum_error,error)
            assert error<1e-11
            frame_states[label]=s
        frames.append(frame_states)
    def frame(i):
        for label,ax in zip(labels,axs):
            ax.clear();s=frames[i][label];key=s['kernel_folder']
            draw_state(ax,cache[key],s,f'{LABELS[label]} — actual accepted t={s["clock_s"]:.5f} s; boundary axes {s["alpha_deg"]:.1f}°')
        fig.suptitle('Damage, signed slip and computed directions\nRed rings: fully decohered; red square: actual developed front; blue arrow: sampled preference\nCommon physical times; native replays from own accepted states, no interpolated histories')
    FuncAnimation(fig,frame,frames=range(len(times)),interval=180).save(OUT/'evolution.gif',writer=PillowWriter(fps=5),dpi=100)
    save(OUT/'animation-verification.json',dict(times_s=times,maximum_damage_integral_replay_error_m=maximum_error,native_replayed_steps=calls,
        geometry_changes_in_replayed_windows=0,interpolated_histories=False,common_physical_times=True))
    for k in cache.values():k.close()
    plt.close(fig)


if __name__=='__main__':
    labels=comparisons();histories_plot(labels);geometry();energy(labels[:4]);export_surface();reserve_plot()
    if (OUT/'frozen-profiles.csv').exists():frozen_plot()
    animate()
