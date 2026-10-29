#!/usr/bin/env python3
"""Common-time fronts, physical inventories and intuitive surface evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cohesive_interface_formed_turn import WORK,OUT,SOURCE,SPACING,configure
import cohesive_interface_candidate_admission as owner
from cohesive_interface_path import table,save,graph
from cohesive_interface_load_turn import functionals
from check_cohesive_interface_kink import C0,GC

LABELS=['coarse','fine'];COLORS=['#0072b2','#d55e00']


def read(label):return np.genfromtxt(OUT/(label+'-history.csv'),delimiter=',',names=True)


def states(label):
    folder=WORK/label;result=json.loads((folder/'snapshots.json').read_text())
    return {s['clock_s']:s for s in result}


def load(s):
    k=owner.Kernel(Path(s['kernel_path']),s['geometry']);k.anchor(s['formed_arcs']);return k


def summary():
    configure('coarse');ik,initial=owner.open_state(SOURCE);init=functionals(ik,initial);qbase,nbase=owner.counts(initial)
    rows=[];profiles=[];increments=[];events=[]
    for label in LABELS:
        s=json.loads((WORK/label/'final.json').read_text());k=load(s);h=read(label);f=functionals(k,s)
        d=json.loads((OUT/(label+'-summary.json')).read_text());q0,_=owner.counts(s)
        a=np.array(s['histories']['damage']);e=np.array(s['histories']['jump'])-np.array(s['histories']['reference'])
        bond=.5*(1-a)/(C0*np.where(a==0,1,a))*e*e
        rows.append(dict(label=label,start_s=initial['clock_s'],end_s=s['clock_s'],new_cell_spacing_m=SPACING[label],dt_s=.000125,
            new_formed_cells=s['formed_arcs']-initial['formed_arcs'],growth_m=f['new_damage_integral_m']-init['new_damage_integral_m'],
            new_material_damage_m=float(k.w[qbase:q0]@a[qbase:q0]),old_material_damage_increment_m=float(k.w[:qbase]@a[:qbase]-ik.w[:qbase]@np.array(initial['histories']['damage'][:qbase])),
            new_material_slip_m2=float(k.w[qbase:q0]@np.array(s['histories']['jump'][qbase:q0])),maximum_new_Gauss_damage=float(max(a[qbase:q0])),
            new_damage_integral_m=f['new_damage_integral_m'],signed_slip_increment_m2=float(k.w@np.array(s['histories']['jump'])-ik.w@np.array(initial['histories']['jump'])),
            process_front_arc_m=f['process_front_arc_m'],process_front_x_m=f['process_front_x_m'],process_front_y_m=f['process_front_y_m'],
            process_front_tangent_deg=f['process_front_tangent_deg'],process_bracket_lower_arc_m=f['process_front_lower_arc_m'],process_bracket_upper_arc_m=f['process_front_upper_arc_m'],
            half_front_arc_m=f['half_front_arc_m'],full_front_arc_m=f['full_front_arc_m'],full_front_x_m=f['full_front_x_m'],full_front_y_m=f['full_front_y_m'],
            full_front_tangent_deg=f['full_front_tangent_deg'],full_bracket_lower_arc_m=f['full_front_lower_arc_m'],full_bracket_upper_arc_m=f['full_front_upper_arc_m'],
            developed_front_available=f['new_full_front_available'],detached_process_Gauss=f['process_detached_Gauss'],preferred_deg=float(h['accepted_preferred_deg'][-1]),
            fracture_increment_J_m=float(sum(h['fracture_increment_J_m'])),cohesive_storage_J_m=float(k.w@bond),bulk_storage_J_m=float(.5*k.coefficients(s)@k.K@k.coefficients(s)),
            trapezoid_work_J_m=float(sum(h['trapezoid_work_J_m'])),sliding_J_m=float(sum(h['sliding_loss_J_m'])),maxwell_J_m=float(sum(h['maxwell_loss_J_m'])),
            absolute_physical_remainder_J_m=d['physical_error_J_m'],physical_vs_sliding=d['physical_vs_sliding'],physical_vs_fracture=d['physical_vs_fracture'],
            maximum_weak_Pa=float(max(h['weak_error_Pa'])),minimum_contact_Pa=float(min(h['minimum_normal_Pa'])),maximum_ledger_J_m=float(max(abs(h['ledger_error_J_m']))),
            physical_publications=d['physical_publications'],seconds=d['seconds']))
        c=k.coefficients(s);normal=10000+k.N@c;velocity=k.Q@np.array(s['nodal_rate'])
        for j in range(len(k.s)):
            profiles.append(dict(label=label,time_s=s['clock_s'],x_m=float(k.s['x_m'][j]),y_m=float(k.s['y_m'][j]),weight_m=float(k.w[j]),persistent=j<q0,
                damage=float(a[j]),jump_m=s['histories']['jump'][j],velocity_m_s=float(velocity[j]),normal_pressure_Pa=float(normal[j]),
                tangent_deg=float(np.rad2deg(np.arctan2(k.s['ty'][j],k.s['tx'][j])))))
        for event in json.loads((WORK/label/'events.json').read_text()):
            after=json.loads((WORK/label/f'after-formation-{len([e for e in events if e["label"]==label])+1}.json').read_text())
            arc=after['geometry'][after['formed_arcs']-1];y,t=graph(arc,1)
            events.append(dict(label=label,time_s=event['time_s'],selected_from_s=event['selected_from_time_s'],
                selected_preference_deg=event['preferred_deg'],endpoint_x_m=arc['end'],endpoint_y_m=float(y),endpoint_tangent_deg=float(np.rad2deg(np.arctan(t))),
                cells_admitted=event['new_formed']-event['old_formed'],transfer_stress_Pa=event['transfer']['stress_defect_Pa'],transfer_storage_J_m=event['transfer']['storage_defect_J_m']))
        k.close()
    initial_slip=float(ik.w@np.array(initial['histories']['jump']))
    ik.close();table(OUT/'comparison.csv',rows);table(OUT/'surface-profiles.csv',profiles);table(OUT/'formation-events.csv',events)
    a,b=rows
    save(OUT/'spatial-difference.json',dict(incremental_damage_absolute_m=b['growth_m']-a['growth_m'],incremental_damage_relative=b['growth_m']/a['growth_m']-1,
        new_material_damage_difference_m=b['new_material_damage_m']-a['new_material_damage_m'],new_material_damage_relative=b['new_material_damage_m']/a['new_material_damage_m']-1,
        new_material_slip_relative=b['new_material_slip_m2']/a['new_material_slip_m2']-1,
        signed_slip_increment_absolute_m2=b['signed_slip_increment_m2']-a['signed_slip_increment_m2'],signed_slip_increment_relative=b['signed_slip_increment_m2']/a['signed_slip_increment_m2']-1,
        process_front_difference_m=b['process_front_arc_m']-a['process_front_arc_m'],developed_front_difference_m=b['full_front_arc_m']-a['full_front_arc_m'],
        actual_process_front_distance_m=float(np.hypot(b['process_front_x_m']-a['process_front_x_m'],b['process_front_y_m']-a['process_front_y_m'])),
        process_tangent_difference_deg=b['process_front_tangent_deg']-a['process_front_tangent_deg'],developed_tangent_difference_deg=b['full_front_tangent_deg']-a['full_front_tangent_deg'],
        sampled_preference_difference_deg=b['preferred_deg']-a['preferred_deg'],
        damage_gate=.05,front_gate_m=.09375,preference_gate_deg=1.,two_spacings_not_a_convergence_order=True))
    ha,hb=read('coarse'),read('fine');assert np.array_equal(ha['time_s'],hb['time_s'])
    for x,y in zip(ha,hb):
        growth=float(x['new_damage_integral_m']-init['new_damage_integral_m']);slip=float(x['integrated_slip_m2']-initial_slip)
        increments.append(dict(time_s=float(x['time_s']),coarse_damage_m=float(x['new_damage_integral_m']),fine_damage_m=float(y['new_damage_integral_m']),
            damage_difference_m=float(y['new_damage_integral_m']-x['new_damage_integral_m']),
            coarse_growth_m=growth,fine_growth_m=float(y['new_damage_integral_m']-init['new_damage_integral_m']),
            growth_relative=float((y['new_damage_integral_m']-x['new_damage_integral_m'])/growth) if growth!=0 else 0.,
            signed_slip_increment_relative=float((y['integrated_slip_m2']-x['integrated_slip_m2'])/slip) if slip!=0 else 0.,
            process_front_difference_m=float(y['process_front_arc_m']-x['process_front_arc_m']),full_front_difference_m=float(y['full_front_arc_m']-x['full_front_arc_m']),
            preferred_difference_deg=float(y['accepted_preferred_deg']-x['accepted_preferred_deg'])))
    table(OUT/'common-time-differences.csv',increments)
    difference=json.loads((OUT/'spatial-difference.json').read_text())
    for key in ['damage_difference_m','growth_relative','signed_slip_increment_relative','process_front_difference_m','full_front_difference_m','preferred_difference_deg']:
        difference['max_common_time_'+key]=float(max(abs(r[key]) for r in increments))
    save(OUT/'spatial-difference.json',difference)
    sa=json.loads((WORK/'coarse/final.json').read_text());sb=json.loads((WORK/'fine/final.json').read_text());curve=[]
    old=initial['geometry'][initial['formed_arcs']-1]['end'];end=min(sa['geometry'][sa['formed_arcs']-1]['end'],sb['geometry'][sb['formed_arcs']-1]['end'])
    for x in np.linspace(old,end,101):
        values=[]
        for state in [sa,sb]:
            arc=next(v for v in state['geometry'][:state['formed_arcs']] if x<=v['end']+1e-12)
            y,p=graph(arc,(x-arc['x'])/(arc['end']-arc['x']));values.append((float(y),float(np.rad2deg(np.arctan(p)))))
        curve.append(dict(x_m=float(x),coarse_y_m=values[0][0],fine_y_m=values[1][0],y_difference_m=values[1][0]-values[0][0],tangent_difference_deg=values[1][1]-values[0][1]))
    table(OUT/'new-geometry-difference.csv',curve)
    return rows


def draw(ax,k,s,title,limits):
    q0,_=owner.counts(s);initial=json.loads(SOURCE.read_text());old=len(initial['geometry'][:initial['formed_arcs']])
    for i,a in enumerate(s['geometry']):
        z=np.linspace(0,1,22);y,_=graph(a,z)
        ax.plot(a['x']+(a['end']-a['x'])*z,y,color='.25' if i<old else '#005f9e' if i<s['formed_arcs'] else '.65',
            ls='-' if i<s['formed_arcs'] else '--',lw=2)
    ax.scatter(k.s['x_m'][:q0],k.s['y_m'][:q0],c=s['histories']['damage'][:q0],cmap='viridis',vmin=0,vmax=1,s=16,zorder=3)
    ix=np.arange(0,q0,6);slip=np.array(s['histories']['jump']);ax.quiver(k.s['x_m'][ix],k.s['y_m'][ix]-.13,100*slip[ix]*k.s['tx'][ix],
        100*slip[ix]*k.s['ty'][ix],angles='xy',scale_units='xy',scale=1,color='#aa3377',width=.004)
    f=functionals(k,s)
    for name,marker,color in [('process','^','#e69f00'),('half','D','white'),('full','s','red')]:
        if name=='full' and not f['new_full_front_available']:continue
        ax.scatter(f[name+'_front_x_m'],f[name+'_front_y_m'],marker=marker,c=color,s=48,edgecolors='black',linewidths=.5,zorder=5)
    a=initial['geometry'][old-1];y,_=graph(a,1);ax.axvline(a['end'],color='.4',ls=':',lw=.8)
    ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x (m)',ylabel='y (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2)


def plots(summary_rows):
    configure('coarse');data={label:states(label) for label in LABELS};last={label:data[label][.440] for label in LABELS}
    right=max(s['geometry'][-1]['end'] for s in last.values())+.15;top=max(float(graph(a,1)[0]) for s in last.values() for a in s['geometry'])+.3
    limits=(5.8,right,-.3,max(top,.6))
    fig,axs=plt.subplots(1,2,figsize=(12,3.9),layout='constrained')
    for ax,label in zip(axs,LABELS):
        k=load(last[label]);draw(ax,k,last[label],f'Commit spacing {SPACING[label]:.6f} m',limits);k.close()
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=axs,label='Surface damage a',shrink=.8)
    fig.suptitle('Actual formed geometry at t=0.440 s; slip purple (×100), zero-history tail dashed\nFronts: a=.05 triangle, .5 diamond, 1 square; dotted line = incoming formed endpoint',fontsize=12)
    fig.savefig(OUT/'geometry.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for label,color in zip(LABELS,COLORS):
        h=read(label);t=h['time_s'];init=json.loads(SOURCE.read_text());ik=load(init);f0=functionals(ik,init);ik.close()
        name=f'{SPACING[label]:.6f} m'
        axs[0,0].plot(t,h['new_damage_integral_m']-f0['new_damage_integral_m'],c=color,label=name)
        for key,ls in [('process','-'),('full','--')]:axs[0,1].plot(t,h[key+'_front_arc_m']-3,c=color,ls=ls,label=name+f', {key}')
        axs[0,2].plot(t,h['accepted_preferred_deg'],c=color,label=name+' preference');axs[0,2].plot(t,h['process_front_tangent_deg'],c=color,ls=':',label=name+' front tangent')
        axs[1,0].plot(t,np.cumsum(h['trapezoid_work_J_m']),c=color,label=name+' boundary work')
        axs[1,0].plot(t,np.cumsum(h['sliding_loss_J_m']),c=color,ls='--',label=name+' sliding loss')
        error=np.cumsum(abs(h['physical_remainder_J_m']));axs[1,1].plot(t,100*error/np.cumsum(h['fracture_increment_J_m']),c=color,label=name+' / fracture')
        axs[1,1].plot(t,100*error/np.cumsum(h['sliding_loss_J_m']),c=color,ls='--',label=name+' / sliding')
        axs[1,2].plot(t,h['integrated_slip_m2'],c=color,label=name)
    axs[1,1].axhline(1,color='black',ls=':',lw=1)
    for ax,y in zip(axs.flat,['Incremental ∫a ds (m)','Connected front beyond seed (m)','Angle from +x (°)','Cumulative energy/work (J/m)','Absolute physical remainder (%)','∫ signed slip ds (m²)']):
        ax.set(xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same incoming material, physical loading and Δt; only future commit spacing differs')
    fig.savefig(OUT/'response.png',dpi=170);plt.close(fig)
    p=np.genfromtxt(OUT/'surface-profiles.csv',delimiter=',',names=True,dtype=None,encoding=None)
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
    for label,color in zip(LABELS,COLORS):
        x=p[p['label']==label]
        for ax,key,scale in zip(axs,['damage','jump_m','normal_pressure_Pa'],[1,1000,1]):ax.plot(x['x_m'],scale*x[key],'.-',ms=3,c=color,label=f'{SPACING[label]:.6f} m')
    for ax,y in zip(axs,['Surface damage a','Signed jump (mm)','Variational normal pressure (Pa)']):ax.set(ylabel=y,xlim=(5.8,right));ax.grid(alpha=.2);ax.legend()
    axs[-1].set_xlabel('Physical x (m)');fig.savefig(OUT/'profiles.png',dpi=170);plt.close(fig)
    times=sorted(set(data['coarse'])&set(data['fine']));fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=axs,label='Surface damage a',shrink=.8)
    def frame(i):
        t=times[i]
        for ax,label in zip(axs,LABELS):
            ax.clear();s=data[label][t];k=load(s);draw(ax,k,s,f'{SPACING[label]:.6f} m commit spacing',limits);k.close()
        fig.suptitle(f't={t:.4f} s; imposed boundary-axis turn α={data["coarse"][t]["alpha_deg"]:.2f}°\nActual Gauss damage, signed slip ×100, accepted states; no history interpolation')
    FuncAnimation(fig,frame,frames=len(times),interval=180).save(OUT/'turn.gif',writer=PillowWriter(fps=6));plt.close(fig)
    save(OUT/'animation.json',dict(times_s=times,frames=len(times),no_history_interpolation=True,slip_vector_scale=100))


def frozen_plots():
    p=np.genfromtxt(OUT/'frozen-profiles.csv',delimiter=',',names=True,dtype=None,encoding=None)
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for mesh,n,c in [('cartesian',64,'#0072b2'),('fitted',64,'#d55e00'),('fitted',96,'#009e73')]:
        q=p[(p['mesh']==mesh)&(p['n']==n)];name=f'{mesh}{n}'
        axs[0].plot(q['x_m'],q['variational_normal_perturbation_Pa'],c=c,label=name)
        axs[1].plot(q['x_m'],q['raw_shear_Pa'],c=c,label=name)
        axs[2].plot(q['x_m'],q['raw_plus_Pa']-q['raw_minus_Pa'],c=c,label=name)
    for ax,y in zip(axs,['Variational normal perturbation (Pa)','Raw sided shear diagnostic (Pa)','Raw normal plus−minus (Pa)']):ax.set(ylabel=y,xlim=(5.5,max(p['x_m'])));ax.grid(alpha=.2);ax.legend()
    axs[-1].set_xlabel('Physical x (m)');fig.suptitle('Identical physical jump / Maxwell field: independent bulk reconstructions\nSided point tractions remain diagnostic, not accuracy references')
    fig.savefig(OUT/'frozen-contact.png',dpi=170);plt.close(fig)


def stress_maps():
    configure('coarse');fields=[];profiles=[];finals=[]
    for label in LABELS:
        s=json.loads((WORK/label/'final.json').read_text());k=load(s);c=k.coefficients(s)
        f=np.einsum('ijk,j->ik',k.field,c);fields.append((k.centers.copy(),f.copy()));finals.append(s)
        for q,cell in enumerate(k.centers):
            if 5.5<=cell['x_m']<=10 and -.8<=cell['y_m']<=1.2:
                profiles.append(dict(label=label,time_s=s['clock_s'],x_m=float(cell['x_m']),y_m=float(cell['y_m']),area_m2=float(cell['area_m2']),
                    tau_xx_Pa=float(f[q,0]),tau_yy_Pa=float(f[q,1]),tau_zz_Pa=float(f[q,2]),tau_xy_Pa=float(f[q,3]),pprime_Pa=float(f[q,4])))
        k.close()
    table(OUT/'physical-stress-centers.csv',profiles)
    values=[r['tau_xy_Pa'] for r in profiles];vmin,vmax=min(values),max(values)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,label,s,(centres,f) in zip(axs,LABELS,finals,fields):
        # Locate by physical coordinates, not deal.II active-cell ordering.
        xs=np.unique(centres['x_m']);ys=np.unique(centres['y_m']);values=np.empty((len(ys),len(xs)))
        values[np.searchsorted(ys,centres['y_m']),np.searchsorted(xs,centres['x_m'])]=f[:,3]
        image=ax.pcolormesh(xs,ys,values,shading='nearest',cmap='coolwarm',vmin=vmin,vmax=vmax)
        for a in s['geometry'][:s['formed_arcs']]:
            z=np.linspace(0,1,20);curve,_=graph(a,z);ax.plot(a['x']+(a['end']-a['x'])*z,curve,color='black',lw=1.4)
        ax.set(xlim=(5.5,10),ylim=(-.8,1.2),xlabel='x (m)',ylabel='y (m)',title=f'{label}: new commit spacing {SPACING[label]:.6f} m');ax.set_aspect('equal')
    fig.colorbar(image,ax=axs,label='Physical τxy (Pa)',shrink=.8)
    fig.suptitle('Represented physical shear stress at t=0.440 s; formed geometry black\nCell-center diagnostic: does not close the outstanding point-traction failure')
    fig.savefig(OUT/'physical-stress.png',dpi=170);plt.close(fig)


def temporal_control():
    label='temporal-half'
    if not (WORK/label/'final.json').exists():return
    s=json.loads((WORK/label/'initial.json').read_text());old=read('coarse');base=old[old['time_s']>s['clock_s']+1e-12];new=read(label)
    k=load(s);f=functionals(k,s);i0=float(k.w@np.array(s['histories']['jump']));k.close()
    bg=float(base['new_damage_integral_m'][-1]-f['new_damage_integral_m']);ng=float(new['new_damage_integral_m'][-1]-f['new_damage_integral_m'])
    bi=float(base['integrated_slip_m2'][-1]-i0);ni=float(new['integrated_slip_m2'][-1]-i0)
    save(OUT/'temporal-difference.json',dict(start_s=s['clock_s'],end_s=.440,base_dt_s=.000125,half_dt_s=.0000625,
        base_growth_m=bg,half_growth_m=ng,growth_difference_m=ng-bg,growth_relative=ng/bg-1,
        base_slip_increment_m2=bi,half_slip_increment_m2=ni,slip_increment_relative=ni/bi-1,
        process_front_difference_m=float(new['process_front_arc_m'][-1]-base['process_front_arc_m'][-1]),
        developed_front_difference_m=float(new['full_front_arc_m'][-1]-base['full_front_arc_m'][-1]),
        preference_difference_deg=float(new['accepted_preferred_deg'][-1]-base['accepted_preferred_deg'][-1]),
        actual_window_not_full_trajectory_error=True))


def frozen_assessment():
    import csv
    values=list(csv.DictReader((OUT/'frozen-loads.csv').open()));rows=[]
    for cart in [r for r in values if r['mesh']=='cartesian']:
        for n in ['64','96']:
            ref=next(r for r in values if r['region']==cart['region'] and r['mesh']=='fitted' and r['n']==n)
            for quantity in ['weak_shear_Pa','normal_perturbation_Pa']:
                difference=float(cart[quantity])-float(ref[quantity]);denom=abs(float(ref[quantity]));assert denom>0
                rows.append(dict(region=cart['region'],quantity=quantity,fitted_reference_n=int(n),difference_Pa=difference,relative=difference/denom))
    table(OUT/'frozen-reference-pair.csv',rows)
    shear=max(abs(r['relative']) for r in rows if r['quantity']=='weak_shear_Pa')
    normal=max(abs(r['relative']) for r in rows if r['quantity']=='normal_perturbation_Pa')
    save(OUT/'frozen-assessment.json',dict(maximum_shear_relative_to_reference_pair=shear,
        maximum_normal_relative_to_reference_pair=normal,retained_weak_load_target=.02,
        shear_target_supported=shear<.02,normal_target_supported=normal<.02,
        normal_reference_sensitive=True,point_traction_qualification=False,
        interpretation='Normal error is 1.94% against fitted96 and 2.006% against fitted64: the retained 2% target is not robustly established. Historical failures stay open.'))


if __name__=='__main__':
    values=summary();plots(values);stress_maps()
    temporal_control()
    if (OUT/'frozen-profiles.csv').exists():frozen_plots();frozen_assessment()
