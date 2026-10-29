#!/usr/bin/env python3
"""Focused independent controls for zero-candidate ownership and admission."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import numpy as np
from cohesive_interface_candidate_admission import WORK,OUT,SOURCE,Kernel,open_state,kernel,candidate_arcs,prepare,attempt,publish,zero_tail,counts,formed_count,replace_candidate_cache
from cohesive_interface_path import encode,save,table
from cohesive_interface_load_turn import WORK as OLD_WORK
from qualify_cohesive_interface_candidate_admission import reject


def inactive():
    old,s=open_state(SOURCE);angle,_,_=old.sample(s);base=encode(s)
    # The physical incoming tensor is loaded and nonuniform. These are genuine
    # alternative zero histories, not a rotation of its stress or memories.
    rows=[];end_states=[];contexts=[old]
    for count,offset in [(2,0.),(8,0.),(2,np.deg2rad(15.))]:
        view,w,metric=prepare(old,s,candidate_arcs(s,angle+offset,count),64,angle+offset,0);contexts.append(view)
        p=view.trial(w,.41025);q0,n0=counts(s)
        assert np.all(np.array(p['state']['histories']['damage'][q0:])==0)
        end_states.append((view,p['state']))
        bad=copy.deepcopy(w);bad['histories']['reference'][q0]=1e-30
        history_rejected=reject(lambda:zero_tail(view,bad))
        bad=copy.deepcopy(w);bad['memory'][n0]=1e-30
        memory_rejected=reject(lambda:zero_tail(view,bad))
        rows.append(dict(candidate_cells=count,cached_offset_deg=float(np.rad2deg(offset)),
            transfer_stress_Pa=metric['stress_defect_Pa'],transfer_storage_J_m=metric['storage_defect_J_m'],
            incoming_virtual_peak_Pa=metric['intact_virtual_peak_Pa'],nonzero_history_rejected=history_rejected,
            nonzero_memory_rejected=memory_rejected))
    ref,rs=end_states[0];rf=np.einsum('ijk,j->ik',ref.field,ref.coefficients(rs));q0,_=counts(s)
    for row,(k,t) in zip(rows,end_states):
        diff=float(np.max(abs(np.einsum('ijk,j->ik',k.field,k.coefficients(t))-rf)))
        assert diff<1e-7,diff
        row['advanced_stress_difference_Pa']=diff
        for key in s['histories']:
            error=float(np.max(abs(np.array(t['histories'][key][:q0])-np.array(rs['histories'][key][:q0]))))
            assert error<1e-12,(key,error)
    assert encode(s)==base
    table(OUT/'inactive-candidates.csv',rows)
    for k in contexts:k.close()


def sampling():
    old,s=open_state(SOURCE);angle,_,old_tensor=old.sample(s)
    arcs=candidate_arcs(s,angle,2);a,w,metric=prepare(old,s,arcs,64,angle,0);b=kernel(arcs,64,probe_order=12)
    b.anchor(s['formed_arcs']);a.anchor(s['formed_arcs']);ca=a.coefficients(w);ta=a.probe@ca;tb=b.probe@ca
    from cohesive_interface_path import axes
    alpha=float(np.rad2deg(axes(ta)[0]));beta=float(np.rad2deg(axes(tb)[0]));norm=lambda t:float(np.sqrt(t[0]**2+t[1]**2+t[2]**2+2*t[3]**2))
    error=norm(ta-tb)/norm(tb);old_error=norm(ta-old_tensor)/norm(old_tensor)
    assert error<.02 and abs(alpha-beta)<1.
    save(OUT/'sampling.json',dict(fixed_Gauss8_tensor_Pa=ta.tolist(),fixed_Gauss12_tensor_Pa=tb.tolist(),
        fixed_quadrature_tensor_relative=error,angle_difference_deg=alpha-beta,
        fixed8_versus_retained_cut_rule_tensor_relative=old_error,formed_endpoint_anchor=True,
        mechanics_and_source_quadrature_unchanged=True,sampler_independent_of_candidate_cuts=True))
    old.close();a.close();b.close()


def defaults():
    folder=WORK/'default-identity';folder.mkdir(exist_ok=False)
    source=OLD_WORK/'legacy-identity'
    command=[str(WORK/'build/cohesive_interface_path'),'16','2',str(source/'geometry.csv'),str(folder),'6','compress']
    with (WORK/'default-identity.log').open('w') as log:
        subprocess.run(command,check=True,timeout=60,stdout=log,stderr=subprocess.STDOUT)
    names=['stiffness','Q','normal-work','normal-minus','normal-plus','shear','columns','probe']
    bits={name:(folder/(name+'.csv')).read_bytes()==(source/(name+'.csv')).read_bytes() for name in names}
    assert all(bits.values()),bits;save(OUT/'default-identity.json',bits)


def loaded():
    # Revisit the original rejected-birth state, preserving EVERY old piece,
    # including zero records inside an already formed piece. Nothing is retired.
    source=OLD_WORK/'turn-small64-reproduced/last-accepted.json'
    old,s=open_state(source);before=encode(s)
    # Exercise the unchanged legacy guard on a private copy, so it cannot
    # overwrite the protected rejected-birth archive from the earlier task.
    archived=next(p for p in OLD_WORK.glob('kernel-*/rejected-birth.json')
        if json.loads(p.read_text())['incoming']['clock_s']==s['clock_s'])
    private=WORK/'retained-tied-rejection-control'
    if not private.exists():shutil.copytree(archived.parent,private,ignore=shutil.ignore_patterns('rejected*'))
    from cohesive_interface_load_turn import Kernel as LegacyKernel
    trial_geometry=json.loads((private/'input.json').read_text())['arcs'];legacy=LegacyKernel(private,trial_geometry)
    legacy_rejected=reject(lambda:legacy.birth(old,s));legacy.close();assert encode(s)==before
    p=attempt(old,s,s['clock_s']+.00025,cells=4,force_refresh=True)
    t=publish(s,p);assert encode(s)==before
    assert p['transfer']['intact_virtual_peak_Pa']>3000,'Control did not expose overloaded intact admission'
    assert p['diagnostics']['admission_fracture_J_m']>0
    assert p['formed']>s['formed_arcs']
    duplicate=reject(lambda:publish(t,p))
    q0,_=counts(s);v=p['kernel'];r=p['native_payload']['native']['r'];ds=v.Q@p['native_payload']['native']['increment']
    local=dict(cohesive_storage_J_m=float(v.w[q0:]@r[q0:,13]),fracture_J_m=p['diagnostics']['admission_fracture_J_m'],
        friction_J_m=float(v.w[q0:]@r[q0:,7]),radiation_J_m=float(v.w[q0:]@r[q0:,8]),
        cohesive_endpoint_work_J_m=float(v.w[q0:]@(r[q0:,6]*ds[q0:])),
        total_endpoint_work_J_m=float(v.w[q0:]@(r[q0:,10]*ds[q0:])))
    local['cohesive_endpoint_remainder_J_m']=local['cohesive_endpoint_work_J_m']-local['cohesive_storage_J_m']-local['fracture_J_m']
    assert local['fracture_J_m']>0 and local['cohesive_storage_J_m']>0
    save(OUT/'overloaded-joint-admission.json',dict(source=str(source),source_clock_s=s['clock_s'],target_s=t['clock_s'],
        original_pure_tied_birth_still_rejected=legacy_rejected,transfer=p['transfer'],diagnostics=p['diagnostics'],local_admission=local,
        old_accepted_state_unchanged=True,intermediate_tied_publication=False,atomic_physical_publications=1,
        duplicate_rejected=duplicate,new_formed_cells=p['formed']-s['formed_arcs']))
    save(WORK/'overloaded-incoming.json',s);save(WORK/'overloaded-accepted.json',t)
    if p['kernel'] is not old:p['kernel'].close()
    old.close()


def completion():
    old,s=open_state(WORK/'candidates2/before-formation-2.json');direction,_,_=old.sample(s)
    view,w,metric=prepare(old,s,candidate_arcs(s,direction,1),64,direction,0)
    cache=replace_candidate_cache(s,w,metric,encode(s));target=.4175
    padded=attempt(view,cache,target,cells=1,force_refresh=True)
    assert padded['diagnostics']['extension_retries']>=1,'Completion control did not reach the candidate edge'
    accepted=publish(cache,padded)
    direct=attempt(old,s,target,cells=2,force_refresh=True);reference=publish(s,direct)
    q0,_=counts(reference);errors={}
    for key in reference['histories']:
        errors[key]=float(np.max(abs(np.array(accepted['histories'][key][:q0])-np.array(reference['histories'][key][:q0]))))
        assert errors[key]<1e-12
    assert accepted['geometry'][:accepted['formed_arcs']]==reference['geometry'][:reference['formed_arcs']]
    save(OUT/'candidate-completion.json',dict(source_clock_s=s['clock_s'],target_s=target,
        numerical_control_not_a_trajectory_accuracy_test=True,completion_retries=padded['diagnostics']['extension_retries'],
        same_original_history_for_every_retry=True,one_physical_publication=True,history_errors_vs_direct_padding=errors,
        remaining_wholly_zero_cells=padded['diagnostics']['candidate_cells']))
    for k in {id(k):k for k in [old,view,padded['kernel'],direct['kernel']]}.values():k.close()


def lifecycle():
    folder=WORK/'candidates2';old,s=open_state(folder/'before-formation-2.json');base=encode(s);target=.41675
    forecast=old.trial(s,target)
    assert formed_count(old,forecast['state'],s['formed_arcs'])>s['formed_arcs']
    p=attempt(old,s,target,cells=2);assert p['diagnostics']['activation_retries']==1
    from qualify_cohesive_interface_load_turn import finite_difference
    active_fd=finite_difference(p['kernel'],p['input'],p['native_payload']);assert active_fd<1e-6,active_fd
    accepted=publish(s,p);assert encode(s)==base
    # Independent whole-view versus persistent inventory, after the joint solve
    # has ALREADY paid the new cohesive/fracture energy. Promotion is metadata.
    from check_cohesive_interface_kink import C0,GC
    q1,n1=counts(accepted);cfull=p['kernel'].coefficients(accepted);short=cfull[:n1+2];weights=p['kernel'].w
    h=accepted['histories'];damage=np.array(h['damage']);e=np.array(h['jump'])-np.array(h['reference'])
    bond=.5*(1-damage)/(C0*np.where(damage==0,1,damage))*e*e
    inventory=dict(bulk_full_J_m=float(.5*cfull@p['kernel'].K@cfull),bulk_persistent_J_m=float(.5*short@p['kernel'].K[:n1+2,:n1+2]@short),
        cohesive_full_J_m=float(weights@bond),cohesive_persistent_J_m=float(weights[:q1]@bond[:q1]),
        fracture_full_J_m=float(GC*weights@damage),fracture_persistent_J_m=float(GC*weights[:q1]@damage[:q1]),
        joint_physical_fracture_increment_J_m=p['diagnostics']['fracture_increment_J_m'],
        new_cell_physical_fracture_J_m=p['diagnostics']['admission_fracture_J_m'])
    assert abs(inventory['bulk_full_J_m']-inventory['bulk_persistent_J_m'])<1e-10
    eps=np.finfo(float).eps;gamma=(2*len(weights)*eps)/(1-2*len(weights)*eps)
    inventory['cohesive_roundoff_bound_J_m']=float(2*gamma*(abs(weights)@abs(bond)))
    inventory['fracture_roundoff_bound_J_m']=float(2*gamma*(abs(weights)@abs(GC*damage)))
    assert abs(inventory['cohesive_full_J_m']-inventory['cohesive_persistent_J_m'])<=inventory['cohesive_roundoff_bound_J_m']
    assert abs(inventory['fracture_full_J_m']-inventory['fracture_persistent_J_m'])<=inventory['fracture_roundoff_bound_J_m']
    save(OUT/'promotion-inventory.json',inventory)
    stored=json.loads((folder/'after-formation-2.json').read_text());assert encode(accepted)==encode(stored)
    again=attempt(old,s,target,cells=2);retried=publish(s,again);assert encode(retried)==encode(accepted)
    save(WORK/'rejected-stale-direction-predictor.json',dict(incoming=s,rejected_state=forecast['state'],
        diagnostics=forecast['diagnostics'],rejection='Cached direction is stale; native residual convergence is insufficient for formation.'))
    save(WORK/'accepted-fresh-direction.json',accepted)
    restart,restored=open_state(WORK/'accepted-fresh-direction.json')
    k=p['kernel'];a=attempt(k,accepted,.417,cells=2);b=attempt(restart,restored,.417,cells=2)
    uninterrupted=publish(accepted,a);restarted=publish(restored,b);assert encode(uninterrupted)==encode(restarted)
    # An optional candidate cache can be discarded/rebuilt with no physical
    # publication. Force the same fresh selection in both matched controls;
    # restoring a stale predictor is a separate numerical-cadence question.
    direction,_,_=k.sample(accepted)
    alternate,w,metric=prepare(k,accepted,candidate_arcs(accepted,direction,8),64,direction,0)
    changed=replace_candidate_cache(accepted,w,metric,encode(accepted))
    assert changed['physical_publications']==accepted['physical_publications']
    c=attempt(alternate,changed,.417,cells=2,force_refresh=True)
    d=attempt(k,accepted,.417,cells=2,force_refresh=True)
    changed_end=publish(changed,c);control=publish(accepted,d)
    q0,n0=counts(control);errors={}
    assert changed_end['geometry'][:changed_end['formed_arcs']]==control['geometry'][:control['formed_arcs']]
    for key in control['histories']:
        errors[key]=float(np.max(abs(np.array(changed_end['histories'][key][:q0])-np.array(control['histories'][key][:q0]))))
        assert errors[key]<1e-12
    save(OUT/'active-lifecycle.json',dict(source_clock_s=s['clock_s'],target_s=target,
        active_coupled_FD_relative=active_fd,
        stale_predictor_native_weak_Pa=forecast['diagnostics']['weak_error_Pa'],rejected_before_publication=True,
        original_accepted_unchanged=True,successful_retry_bitwise=True,actual_direction_refresh_retry=1,
        single_atomic_physical_publication=True,reference_accepted_endpoint_bitwise=True,
        formed_history_JSON_restart_bitwise=True,changed_zero_cache_physical_errors=errors,
        candidate_cache_replacement_physical_publications=0,candidate_cache_version_changed=True,
        matched_restart_refresh_explicit=True,scope='Cache cadence is numerical state; arbitrary stale-cache removal is not assumed path independent.'))
    for k in {id(k):k for k in [old,p['kernel'],again['kernel'],restart,a['kernel'],b['kernel'],alternate,c['kernel'],d['kernel']]}.values():k.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['inactive','sampling','defaults','loaded','lifecycle','completion']);a=p.parse_args();globals()[a.task]()
