#!/usr/bin/env python3
"""Common-time, physical-front and work assessment; preserve raw pilot data."""
import csv
import json
import numpy as np
from cohesive_interface_path import WORK, OUT, PathKernel, functionals, graph, save, table, G, GC, C0

FINAL=['work64','work96','work-step96','work-q3','work-time96','work-surface96']
V1=['main64','main96','alternative-q3','extension-half64','angular7-64','temporal96-half','surface-half96',
    'consistent64','consistent96','consistent-step96','consistent-q3','consistent-time96']


def summary(label):return json.loads((OUT/(label+'-summary.json')).read_text())
def rows(label):return np.genfromtxt(OUT/(label+'-history.csv'),delimiter=',',names=True,dtype=None,encoding=None)
def snapshots(label):return json.loads((WORK/label/'snapshots.json').read_text())
def state(label,clock=.445):
    if clock==0:return json.loads((WORK/label/'initial.json').read_text())
    states=snapshots(label);s=min(states,key=lambda s:(abs(s['clock_s']-clock),-s['generation']));assert abs(s['clock_s']-clock)<1e-10,(label,clock,s['clock_s'])
    return s
def context(s):
    k=PathKernel(WORK/s['kernel_folder'],s['geometry'],s.get('pressure_mode','point'))
    assert k.key==s['kernel']
    return k


def comparisons(labels):
    output=[]
    for left,right,kind in labels:
        a,b=state(left),state(right);ka,kb=context(a),context(b);fa,fb=functionals(ka,a),functionals(kb,b)
        cpa=np.r_[a['wall_memory'],a['memory']];cpb=np.r_[b['wall_memory'],b['memory']]
        slipa=ka.w@np.array(a['histories']['jump']);slipb=kb.w@np.array(b['histories']['jump'])
        ib=json.loads((WORK/right/'initial.json').read_text());kb0=context(ib)
        initial_slip=float(kb0.w@np.array(ib['histories']['jump']));initial_growth=functionals(kb0,ib)['new_damage_integral_m'];kb0.close()
        growth_denominator=fb['new_damage_integral_m']-(initial_growth if kind=='late-time' else 0.)
        output.append(dict(comparison=kind,left=left,right=right,time_s=.445,
            new_damage_change_m=fa['new_damage_integral_m']-fb['new_damage_integral_m'],
            new_damage_relative=abs(fa['new_damage_integral_m']-fb['new_damage_integral_m'])/growth_denominator,
            slip_relative=abs(slipa-slipb)/abs(slipb-initial_slip),
            process_front_difference_m=float(np.hypot(fa['process_front_x_m']-fb['process_front_x_m'],fa['process_front_y_m']-fb['process_front_y_m'])),
            full_front_difference_m=float(np.hypot(fa['full_front_x_m']-fb['full_front_x_m'],fa['full_front_y_m']-fb['full_front_y_m'])),
            process_tangent_difference_deg=abs(fa['process_front_tangent_deg']-fb['process_front_tangent_deg']),
            full_tangent_difference_deg=abs(fa['full_front_tangent_deg']-fb['full_front_tangent_deg']),
            bulk_storage_difference_J_m=float(.5*cpa@ka.K@cpa-.5*cpb@kb.K@cpb)))
        ka.close();kb.close()
    return output


def main():
    allrows=[];events=[];fronts=[];contact=[];initials=[];gates={};energies=[];kernels={}
    for label in FINAL+V1:
        info=summary(label);s=state(label);k=context(s);f=functionals(k,s);r=rows(label)
        full=np.array(s['histories']['damage']).reshape(-1,3)
        allrows.append(dict(label=label,n=info['n'],degree=info['degree'],extension_m=info['extension_m'],
            angular_sampling_deg=info['sampling_deg'],surface_h_m=info['surface_h_m'],time_s=s['clock_s'],gamma=s['gamma'],
            geometry_events=info['geometry_publications'],new_damage_integral_m=f['new_damage_integral_m'],
            process_front_x_m=f['process_front_x_m'],process_front_y_m=f['process_front_y_m'],process_tangent_deg=f['process_front_tangent_deg'],
            full_front_x_m=f['full_front_x_m'],full_front_y_m=f['full_front_y_m'],full_tangent_deg=f['full_front_tangent_deg'],
            new_fully_decohered_cells=f['new_fully_decohered_cells'],slip_integral_m2=float(k.w@np.array(s['histories']['jump'])),
            seconds=info['seconds'],physical_work_relative_sliding=info['energy']['relative_sliding'],physical_work_relative_fracture=info['energy']['relative_fracture']))
        c=np.r_[s['wall_memory'],s['memory']];jump=(k.raw_Nm-k.raw_Np)@c
        curved=k.s['x_m']>6
        contact.append(dict(label=label,minimum_normal_Pa=float(np.min(r['minimum_normal_Pa'])),
            max_normal_gap_m=float(np.max(r['normal_gap_m'])),paired_normal_jump_RMS_Pa=float(np.sqrt(k.w@(jump*jump)/sum(k.w))),
            curved_paired_normal_jump_RMS_Pa=float(np.sqrt(k.w[curved]@(jump[curved]**2)/sum(k.w[curved]))),
            maximum_true_weak_Pa=float(np.max(r['weak_error_Pa'])),minimum_symmetric_tangent_Pa_m=float(np.min(r['minimum_symmetric_tangent_Pa_m']))))
        for e in info['events']:
            d=e['decision'];m=e['transfer'];events.append(dict(label=label,stage=e['stage'],time_s=e['time_s'],
                old_tangent_deg=d['old_tangent_deg'],compression_deg=d['compression_deg'],max_shear_1_deg=d['max_shear_1_deg'],
                max_shear_2_deg=d['max_shear_2_deg'],preferred_deg=d['selected_preference_deg'],terminal_deg=e['terminal_angle_deg'],
                endpoint_x_m=e['endpoint_x_m'],endpoint_y_m=e['endpoint_y_m'],
                Maxwell_coefficient_max_m=m['nonuniform_incoming_Maxwell_m'],stress_pressure_transfer_Pa=m['physical_tensor_pressure_error_Pa'],
                energy_transfer_J_m=m['energy_change_J_m'],tied_peak_Pa=m['new_tied_peak_Pa'],geometry_birth_fracture_J_m=m['new_fracture_J_m'],
                old_history_bitwise=m['old_histories_bitwise'],response_columns_equal=m['old_native_response_hashes_equal']))
        for ss in snapshots(label):
            kk=context(ss);ff=functionals(kk,ss)
            fronts.append(dict(label=label,time_s=ss['clock_s'],gamma=ss['gamma'],generation=ss['generation'],**ff));kk.close()
        initial=json.loads((WORK/label/'initial.json').read_text());ki=context(initial);ci=np.r_[initial['wall_memory'],initial['memory']]
        a=np.array(initial['histories']['damage']);e=np.array(initial['histories']['jump'])-np.array(initial['histories']['reference'])
        bi=ki.w@(.5*(1-a)/(C0*np.where(a==0,1,a))*e**2)
        initial_error=0.;initial_weak=0.;initial_obstacle_ratio=0.
        if initial['clock_s']==0:
            axial=.5*np.tan(np.deg2rad(40));expected=G*initial['wall_memory']*np.array([-2*axial,2*axial,0.,1.,0.])
            initial_error=float(max(abs(ki.field[:,0,:]*ci[0]-expected).ravel()))
            q=(1-a)*e/(C0*a);initial_weak=float(max(abs((ki.K@ci)[1:]+ki.Q.T@(ki.w*q))/ki.mass))
            initial_obstacle_ratio=float(max(abs(e)/(2*GC/3000*a)))
            assert initial_error<1e-7 and initial_weak<=1e-8 and initial_obstacle_ratio<1
        initials.append(dict(label=label,clock_s=initial['clock_s'],bulk_J_m=float(.5*ci@ki.K@ci),bond_J_m=float(bi),fracture_J_m=float(GC*ki.w@a),
            displacement_offset_max_m=float(max(abs(np.array(initial['jump_offset'])))),physical_tensor_pressure_error_Pa=initial_error,
            weak_equilibrium_Pa=initial_weak,maximum_seed_obstacle_ratio=initial_obstacle_ratio))
        energies.append(dict(label=label,interval_start_s=initial['clock_s'],interval_end_s=s['clock_s'],
            bulk_change_J_m=float(.5*c@k.K@c-.5*ci@ki.K@ci),bond_change_J_m=float(r['bond_J_m'][-1]-bi),
            fracture_J_m=float(sum(r['fracture_increment_J_m'])),friction_J_m=float(sum(r['friction_J_m'])),radiation_J_m=float(sum(r['radiation_J_m'])),
            Maxwell_J_m=float(sum(r['maxwell_loss_J_m'])),endpoint_work_J_m=float(sum(r['endpoint_work_J_m'])),trapezoid_work_J_m=float(sum(r['trapezoid_work_J_m'])),
            physical_sum_absolute_J_m=float(sum(abs(r['physical_remainder_J_m']))),physical_signed_J_m=float(sum(r['physical_remainder_J_m'])),
            maximum_BE_ledger_J_m=float(max(abs(r['ledger_error_J_m']))),hold_absolute_J_m=float(sum(abs(r['physical_remainder_J_m'][r['time_s']>.44])))))
        for name in info['kernels']:
            kernels[name]=json.loads((WORK/name/'native.json').read_text())
        ki.close();k.close()
    table(OUT/'comparison.csv',allrows);table(OUT/'geometry-events.csv',events);table(OUT/'physical-fronts.csv',fronts)
    table(OUT/'contact-and-residuals.csv',contact);table(OUT/'initialization.csv',initials);table(OUT/'energy-accounting.csv',energies)
    focused=comparisons([('work64','work96','bulk-space'),('work96','work-step96','extension-step'),
                        ('work-q3','work96','alternative-Q3'),('work96','work-time96','late-time'),('work96','work-surface96','surface-space')])
    table(OUT/'focused-comparisons.csv',focused)
    historical=comparisons([('main64','main96','v1-bulk'),('main64','extension-half64','v1-extension'),
                           ('main64','angular7-64','v1-angular7'),('main96','surface-half96','v1-surface'),('main96','temporal96-half','v1-time'),
                           ('consistent64','consistent96','v2-bulk'),('consistent96','consistent-step96','v2-extension')])
    table(OUT/'retained-v1-comparisons.csv',historical)
    for r in focused:
        front_threshold=summary(r['right'])['surface_h_m']
        gate=dict(integral_growth=r['new_damage_relative']<.05,integral_slip=r['slip_relative']<.05,
            process_front=r['process_front_difference_m']<front_threshold,full_front=r['full_front_difference_m']<front_threshold,
            process_tangent=r['process_tangent_difference_deg']<3,full_tangent=r['full_tangent_difference_deg']<3)
        gates[r['comparison']]={name:bool(value) for name,value in gate.items()}
    save(OUT/'assessment.json',dict(physical_R_m=.75,focused=focused,gates=gates,
        all_selected_practical_targets_pass=all(all(g.values()) for g in gates.values()),
        numerical_interpolation_correction='Derivative-consistent Hermite slope; physical direction law unchanged.',
        corrected_cases=FINAL,retained_v1_cases=V1,
        kernels=kernels,normal_contact_note='Zero opening and algebraic weak equilibrium are separate from unfiltered paired traction mismatch.',
        scope='Isolated native mixed operators/C++ evolution, not production integration or globally unique free propagation.'))
    from check_cohesive_interface_kink import RAD
    table(OUT/'parameters.csv',[dict(parameter=k,value=v,unit=u,role=role) for k,v,u,role in [
        ('G',1e6,'Pa','unchanged bulk law'),('eta',1e8,'Pa s','unchanged Maxwell law'),('Gc',1.,'J/m2','single surface fracture inventory'),
        ('peak',3000.,'Pa','pressure-independent intact strength'),('delta_f',2/3000,'m','unchanged cohesive calibration'),
        ('p0',10000.,'Pa','physical background confinement'),('mu_reference',.1,'1','RSF reference; not intact Coulomb formation'),
        ('V0',5e-6,'m/s','unchanged RSF'),('Dc',.001,'m','unchanged RSF'),('RSF_direct',.005,'1','unchanged RSF'),('RSF_evolution',.01,'1','unchanged RSF'),
        ('radiation',RAD,'Pa s/m','unchanged radiation damping'),('direction_R',.75,'m','additional uncalibrated physical formation hypothesis'),
        ('domain_width',12.,'m','finite confined specimen'),('domain_height',12.,'m','finite confined specimen'),
        ('loading_rate',.004,'1/s','fixed across comparisons'),('loading_end',.44,'s','followed by .005 s displacement hold')]])
    print(json.dumps(dict(focused=focused,gates=gates),indent=2))


if __name__=='__main__':main()
