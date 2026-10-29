#!/usr/bin/env python3
"""Compact, common-time assessment of the isolated curved native experiment."""
import csv
import json
from pathlib import Path
import numpy as np

from cohesive_interface_turn import WORK, OUT, HISTORY, Kernel, save, table, point, G, C0


def rows(label):
    return np.genfromtxt(OUT/(label+'-history.csv'), delimiter=',', names=True)


def summary(label):
    return json.loads((OUT/(label+'-summary.json')).read_text())


def state(label, clock=.45):
    folder=WORK/'runs'/label
    if clock==0:return json.loads((folder/'initial.json').read_text())
    snapshots=json.loads((folder/'snapshots.json').read_text())
    key=min(snapshots,key=lambda k:abs(float(k)-clock))
    assert abs(float(key)-clock)<1e-10,(label,clock,key)
    return snapshots[key]


def at(label,clock=.45):
    r=rows(label);i=np.argmin(abs(r['time_s']-clock))
    assert abs(r['time_s'][i]-clock)<1e-10
    return {k:float(r[k][i]) for k in r.dtype.names}


def physical_surface(label, clock=.45):
    info=summary(label);s=state(label,clock)
    q=np.genfromtxt(WORK/info['kernel'].removesuffix('-q1')/'surface.csv',delimiter=',',names=True)
    q['x_m']*=info['scale_m'];q['y_m']*=info['scale_m'];q['weight_m']*=info['scale_m']
    return q,s


def extent(label,clock=.45):
    q,s=physical_surface(label,clock);w=q['weight_m'].reshape(-1,3)
    damage=np.array(s['histories']['damage']).reshape(-1,3)
    mean=(w*damage).sum(axis=1)/w.sum(axis=1);start=len(mean)//2
    result={}
    for threshold in [.05,.5,.999]:
        # Whole-cell extent is deliberately a bracket: no interpolated local
        # history or claim of an exact first activation/front position.
        count=0
        for value in mean[start:]:
            if value<threshold:break
            count+=1
        arc=w.sum(axis=1)[start:]
        lo=float(sum(arc[:count]));hi=lo+float(arc[count]) if count<len(arc) else lo
        result['a'+str(threshold)]=dict(connected_cells=count,arc_lower_m=lo,arc_upper_m=hi,
             projected_x_lower_m=count*12/summary(label)['n'],
             detached_cells=int(np.count_nonzero(mean[start+count:]>=threshold)))
    complete=np.min(damage[start:],axis=1)==1
    count=0
    for value in complete:
        if not value:break
        count+=1
    result['fully_decohered']=dict(whole_cells=count,
        arc_length_m=float(w.sum(axis=1)[start:start+count].sum()),
        projected_x_advance_m=count*12/summary(label)['n'],
        definition='All three owned Gauss histories equal a=1 in every counted cell.')
    return result


def profile_difference(coarse,fine,clock=.45):
    qc,sc=physical_surface(coarse,clock);qf,sf=physical_surface(fine,clock)
    # Same physical x-cell bins on the same smooth graph. Integrate actual
    # Gauss histories; never interpolate them into a new constitutive state.
    h=12/summary(coarse)['n'];n=round(6/h)
    def averages(q,s):
        ids=np.clip(np.floor((q['x_m']-3)/h).astype(int),0,n-1)
        w=np.bincount(ids,weights=q['weight_m'],minlength=n)
        return {k:np.bincount(ids,weights=q['weight_m']*np.array(s['histories'][k]),minlength=n)/w
                for k in ['damage','jump']},w
    c,w=averages(qc,sc);f,_=averages(qf,sf)
    return {k:dict(weighted_L2=float(np.sqrt(w@((c[k]-f[k])**2))),
                   relative_L2=float(np.sqrt(w@((c[k]-f[k])**2)/max(w@(f[k]**2),1e-300))),
                   append_relative_L2=float(np.sqrt(w[n//2:]@((c[k][n//2:]-f[k][n//2:])**2)/max(w[n//2:]@(f[k][n//2:]**2),1e-300)))) for k in c}


def geometric_quadrature(nx,angle,width):
    """Independent Gauss12 geometric/weak patch control, with known physical
    uniform stress. No interpolation of evolved nonlinear Gauss history."""
    def integrate(order):
        z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
        length=0.;n=nx//2-1;M=np.zeros((n,n));normal=np.zeros(n);shear=np.zeros(n)
        for cell in range(nx//4,3*nx//4):
            x=(cell+z)/nx;v=np.clip((x-.5)/width,0,1)
            slope=np.tan(np.deg2rad(angle))*(3*v*v-2*v*v*v)
            tx=1/np.sqrt(1+slope*slope);ty=slope*tx;weight=12*w/(nx*tx)
            Q=np.zeros((order,n))
            for j,shape in [(cell-nx//4-1,1-z),(cell-nx//4,z)]:
                if 0<=j<n:Q[:,j]=shape
            axial=.5*np.tan(np.deg2rad(40));tau_xx=-600*axial;tau_yy=600*axial;tau_xy=300.
            pn=10000-(tau_xx*ty**2+tau_yy*tx**2-2*tau_xy*tx*ty)
            tn=tau_xy*(tx**2-ty**2)+(tau_yy-tau_xx)*tx*ty
            length+=sum(weight);M+=Q.T@(weight[:,None]*Q)
            normal+=Q.T@(weight*pn);shear+=Q.T@(weight*tn)
        return length,M,normal,shear
    a=integrate(3);b=integrate(12)
    result=dict(nx=nx,angle_deg=angle,width_m=12*width,length_relative=abs(a[0]-b[0])/b[0],
        mass_relative=float(np.linalg.norm(a[1]-b[1])/np.linalg.norm(b[1])),
        uniform_stress_normal_weak_relative=float(np.linalg.norm(a[2]-b[2])/np.linalg.norm(b[2])),
        uniform_stress_shear_weak_relative=float(np.linalg.norm(a[3]-b[3])/np.linalg.norm(b[3])),
        scope='Independent geometric and uniform-physical-stress patch integrals, not re-quadrature of evolved local histories.')
    assert max(result[k] for k in ['length_relative','mass_relative','uniform_stress_normal_weak_relative','uniform_stress_shear_weak_relative'])<1e-4
    return result


def main():
    decision=json.loads((OUT/'followup-selection.json').read_text())
    base=decision['initial_candidates'];leading=[a for a,_ in decision['leading']]
    candidates=[]
    for a,label in base:
        r=at(label);candidates.append(dict(angle_deg=a,label=label,width_m=1.5,
            mean_new_damage=r['append_mean_damage'],new_damage_integral_m=r['append_damage_integral_m'],
            damage_direction_deg=r['append_damage_direction_deg'],resolved_cells=r['append_cells_a005'],
            time_s=.45,gamma=r['gamma']))
    for a in [-60,-40,-20,0,20,40,60]:
        label='wide-'+('m' if a<0 else 'p')+str(abs(a));r=at(label)
        candidates.append(dict(angle_deg=a,label=label,width_m=3.,mean_new_damage=r['append_mean_damage'],
            new_damage_integral_m=r['append_damage_integral_m'],damage_direction_deg=r['append_damage_direction_deg'],
            resolved_cells=r['append_cells_a005'],time_s=.45,gamma=r['gamma']))
    table(OUT/'candidate-scores.csv',candidates)
    comparisons=[]
    errors={}
    for a,label in decision['leading']:
        fine='fine-p'+str(a);half='half-p'+str(a)
        c,f,h=at(label),at(fine),at(half)
        errors[a]=abs(c['append_mean_damage']-f['append_mean_damage'])+abs(f['append_mean_damage']-h['append_mean_damage'])
        for kind,left,right in [('space',label,fine),('time',fine,half)]:
            l,r=at(left),at(right)
            comparisons.append(dict(angle_deg=a,comparison=kind,coarse=left,reference=right,
                damage_integral_change_m=l['append_damage_integral_m']-r['append_damage_integral_m'],
                damage_integral_relative=abs(l['append_damage_integral_m']-r['append_damage_integral_m'])/r['append_damage_integral_m'],
                score_change=l['append_mean_damage']-r['append_mean_damage'],
                slip_increment_relative=abs(l['integrated_increment_from_initial_m2']-r['integrated_increment_from_initial_m2'])/r['integrated_increment_from_initial_m2'],
                damage_direction_change_deg=l['append_damage_direction_deg']-r['append_damage_direction_deg']))
    table(OUT/'focused-comparisons.csv',comparisons)
    winner,runner=leading
    gap=at('half-p'+str(winner))['append_mean_damage']-at('half-p'+str(runner))['append_mean_damage']
    allowance=2*sum(errors.values())
    assert gap>allowance,('No distinguished primary candidate',gap,allowance)
    primary=dict(angle_deg=winner,runner_deg=runner,half_step_score_gap=gap,
                 twice_observed_combined_uncertainty=allowance,distinguished_in_sampled_primary_family=True,
                 unique_geometry_independent_direction=False,
                 scope='Finite preview score on a declared candidate family, width and loading horizon; not a global potential or universal path law.')
    wide=max([r for r in candidates if r['width_m']==3],key=lambda r:r['mean_new_damage'])
    primary['wide_best_sampled_deg']=wide['angle_deg'];primary['wide_sector_boundary']=abs(wide['angle_deg'])==60
    primary['width_changes_ranking']=wide['angle_deg']!=winner
    # Reflection compares actual native histories and sided pressures at the
    # same physical reflected Gauss locations, not scalar stress magnitudes.
    k=Kernel(WORK/summary('fine-p'+str(winner))['kernel'].removesuffix('-q1'),True,12.,.0021)
    kr=Kernel(WORK/summary('reflection')['kernel'].removesuffix('-q1'),True,12.,.0021)
    a=state('fine-p'+str(winner));b=state('reflection');ca=np.r_[a['wall_memory'],a['memory']];cb=np.r_[b['wall_memory'],b['memory']]
    reflection={}
    for key in HISTORY:
        sign=-1 if key in ['jump','reference','relaxed_jump'] else 1
        reflection[key]=float(max(abs(np.array(a['histories'][key])-sign*np.array(b['histories'][key]))))
    reflection['sided_normal_swap_Pa']=float(max(np.max(abs(k.Nm@ca-kr.Np@cb)),np.max(abs(k.Np@ca-kr.Nm@cb))))
    reflection['coefficient_sign_error']=float(max(abs(ca[1:]+cb[1:])))
    reflection['score_error']=abs(at('fine-p'+str(winner))['append_mean_damage']-at('reflection')['append_mean_damage'])
    def field(kernel,c):
        cells=np.genfromtxt(kernel.path/'field-cells.csv',delimiter=',',names=True)
        data=np.memmap(kernel.path/'field-cell-response.bin',mode='r',dtype=np.float64,
                       shape=(len(cells),len(kernel.full_nodes)+1,5))
        return cells,np.einsum('ijk,j->ik',data,kernel.field_transform@c)
    qa,fa=field(k,ca);qb,fb=field(kr,cb)
    lookup={(round(float(x),10),round(float(y),10)):i for i,(x,y) in enumerate(zip(qb['x_m'],qb['y_m']))}
    order=[lookup[round(float(x),10),round(float(-y),10)] for x,y in zip(qa['x_m'],qa['y_m'])]
    difference=fa-fb[order]*np.array([1,1,1,-1,1])
    reflection['bulk_cell_mean_full_tensor_Pa']=float(max(np.sqrt(np.sum(difference[:,:3]**2,axis=1)+2*difference[:,3]**2)))
    reflection['bulk_cell_mean_pressure_Pa']=float(max(abs(difference[:,4])))
    assert reflection['bulk_cell_mean_full_tensor_Pa']<1e-7 and reflection['bulk_cell_mean_pressure_Pa']<1e-7
    assert max(reflection[key] for key in HISTORY)<1e-9,reflection
    assert reflection['sided_normal_swap_Pa']<1e-7,reflection
    k.close();kr.close()
    mature=[]
    for label in ['mature64','mature128','maturewide128','maturehalf128']:
        r=rows(label);s=summary(label);window=(r['time_s']>=.44)&(r['time_s']<=.55)
        mature.append(dict(label=label,nx=s['n'],ny=s['n']//2,width_m=s['physical_width_m'],dt_s=s['dt_s'],
            integrated_slip_increment_m2=s['final']['integrated_increment_from_initial_m2'],
            min_through_turn_rate_m_s=float(min(r['minimum_abs_turn_velocity_m_s'][window])),
            minimum_contact_Pa=s['minimum_normal_Pa'],max_curvature_slip=float(max(r['maximum_curvature_times_slip'])),
            work_error_relative_sliding=s['energy']['relative_sliding'],seconds=s['seconds']))
    table(OUT/'turning-comparison.csv',mature)
    qualified=['mature64','mature128','maturewide128','maturehalf128','fine-p30','half-p30','fine-p20','half-p20','reflection']
    qualified+=['wide-fine-p'+str(wide['angle_deg'])]
    checks={};cost=[]
    for label in qualified:
        r=rows(label);s=summary(label);energy=s['energy']
        checks[label]=dict(weak_Pa=s['max_weak_Pa'],contact_Pa=s['minimum_normal_Pa'],
            max_normal_gap_m=float(max(r['normal_total_max_m'])),max_normal_work_J_m=float(max(abs(r['normal_work_J_m']))),
            max_ledger_J_m=energy['max_ledger_J_m'],absolute_physical_remainder_J_m=energy['absolute_remainder_J_m'],
            signed_physical_remainder_J_m=float(sum(r['physical_remainder_J_m'])),
            work_relative_sliding=energy['relative_sliding'],work_relative_fracture=energy['relative_fracture'],
            birth=s['birth'],lifecycle=s['checks'],physical_publications=json.loads((WORK/'runs'/label/'final.json').read_text())['physical_publications'],
            steps=s['steps'],extent=None if s['mature'] else extent(label))
        assert checks[label]['max_normal_gap_m']<1e-12
        assert s['max_weak_Pa']<=1e-8 and s['minimum_normal_Pa']>0
        assert energy['relative_sliding']<.01
        assert s['mature'] or energy['relative_fracture']<.01
        cost.append(dict(label=label,evolution_seconds=s['seconds'],accepted_steps=s['steps'],
                         mean_step_ms=1000*s['seconds']/s['steps']))
    table(OUT/'costs.csv',cost)
    accounting=[]
    for label in ['half-p30','half-p20','maturehalf128','wide-fine-p60']:
        r=rows(label);q,s0=physical_surface(label,0);info=summary(label)
        a=np.array(s0['histories']['damage']);e=np.array(s0['histories']['jump'])-np.array(s0['histories']['reference'])
        initial_bond=q['weight_m']@(.5*(1-a)/(C0*np.where(a==0,1,a))*e**2)
        initial_bulk=G*(.5+2*info['axial_ratio']**2)*info['scale_m']**2*s0['wall_memory']**2
        accounting.append(dict(label=label,end_s=float(r['time_s'][-1]),initial_bulk_J_m=float(initial_bulk),
            initial_cohesive_J_m=float(initial_bond),initial_fracture_J_m=float(q['weight_m']@a),
            delta_bulk_J_m=float(r['bulk_J_m'][-1]-initial_bulk),delta_cohesive_J_m=float(r['bond_J_m'][-1]-initial_bond),
            added_fracture_J_m=float(sum(r['fracture_increment_J_m'])),
            friction_J_m=float(sum(r['friction_J_m'])),radiation_J_m=float(sum(r['radiation_J_m'])),
            maxwell_J_m=float(sum(r['maxwell_loss_J_m'])),endpoint_work_J_m=float(sum(r['endpoint_work_J_m'])),
            trapezoid_work_J_m=float(sum(r['trapezoid_work_J_m'])),sum_abs_physical_remainder_J_m=float(sum(abs(r['physical_remainder_J_m']))),
            signed_physical_remainder_J_m=float(sum(r['physical_remainder_J_m']))))
    table(OUT/'energy-accounting.csv',accounting)
    integration=[]
    point_errors={}
    for kernel in sorted({summary(label)['kernel'].removesuffix('-q1') for label in qualified}):
        p=WORK/kernel;m=json.loads((p/'native.json').read_text());K=np.loadtxt(p/'stiffness.csv',delimiter=',');E=np.loadtxt(p/'energy-gauss6.csv',delimiter=',')
        ref=np.genfromtxt(p/'point-reference.csv',delimiter=',',names=True)
        old={key:ref[key] for key in ['damage','jump','reference','theta']}
        independent=point(old,ref['increment'],ref['pressure'],ref['dt'])
        point_errors[kernel]={key:float(np.max(abs(value-ref[key])/np.maximum(abs(ref[key]),1.))) for key,value in independent.items()}
        assert max(point_errors[kernel].values())<1e-12
        integration.append(dict(kernel=kernel,nx=m['nx'],bulk_masters=m['bulk_masters'],seconds=m['seconds'],
             gradient_FD_relative=m['gradient_FD_relative'],independent_energy_relative=float(np.linalg.norm(K-E)/np.linalg.norm(K)),
             raw_unit_strain_stress_error_Pa=m['affine_stress_error_Pa'],normal_jump_operator_error=m['normal_jump_operator_error']))
    save(OUT/'verification.json',dict(primary_selection=primary,reflection=reflection,runs=checks,native_kernels=integration,
         independent_point_errors=point_errors,
         surface_quadrature=[geometric_quadrature(n,a,w) for n,a,w in [(64,30,.125),(128,30,.125),(128,60,.25)]],
         profile_comparison={str(a):profile_difference(label,'fine-p'+str(a)) for a,label in decision['leading']},
         scope='Experimental native FE operators and native C++ signed evolution with independent JSON ownership, not production simulator integration.'))
    print(json.dumps(dict(primary=primary,reflection=reflection,turning=mature,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
