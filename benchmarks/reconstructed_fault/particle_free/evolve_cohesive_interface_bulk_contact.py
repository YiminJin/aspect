#!/usr/bin/env python3
"""Short, separately initialized common-path controls; never checkpoint conversion."""
import copy
import ctypes as ct
import json
import time
import numpy as np

from investigate_cohesive_interface_bulk_contact import WORK, OUT, ARCS, build, save, table, PATH_WORK
from cohesive_interface_path import PathKernel, encode, functionals, HISTORY, sha
from cohesive_interface_admission import Admission
from cohesive_interface_turn import derivative

SOURCE=PATH_WORK/'work-temporal-incoming.json'


def initialize(k):
    incoming=json.loads(SOURCE.read_text());s=copy.deepcopy(incoming)
    old=PathKernel(PATH_WORK/incoming['kernel_folder'],incoming['geometry'],'work')
    n=len(incoming['memory']);nq=len(incoming['histories']['damage'])
    assert k.arcs[:len(incoming['geometry'])]==incoming['geometry']
    assert np.array_equal(k.Q[:nq,:n],old.Q)
    for key in ['memory','delta','nodal_rate']:s[key]+=[0.]*(len(k.nodes)-n)
    for key in HISTORY:s['histories'][key]+=[incoming['virgin_theta'] if key=='theta' else 0.]*(len(k.s)-nq)
    for key in ['jump_offset','last_resistance']:s[key]+=[0.]*(len(k.s)-nq)
    for key in HISTORY:assert s['histories'][key][:nq]==incoming['histories'][key]
    old_memory=old.Q@np.array(incoming['memory']);new_memory=k.Q[:nq]@np.array(s['memory'])
    assert max(abs(old_memory-new_memory))<1e-13
    represented=np.array(s['jump_offset'])+k.Q@np.array(s['delta'])
    assert max(abs(represented[:nq]-np.array(incoming['histories']['jump'])))<1e-13
    old_c=np.r_[incoming['wall_memory'],incoming['memory']];c=np.r_[s['wall_memory'],s['memory']]
    eold=.5*old_c@old.K@old_c;enew=.5*c@k.K@c
    weak=(k.K@c)[1:]+k.Q.T@(k.w*np.array(s['last_resistance']))
    # New tied records have no saved auxiliary multiplier. Their unbalanced
    # nodal force is a reaction, not a surface disequilibrium. Remove all tied
    # reactions and report the part no reaction can balance separately.
    tied=np.array(s['histories']['damage'])==0
    resistance=np.array(s['last_resistance']);resistance[tied]=0
    force=(k.K@c)[1:]+k.Q.T@(k.w*resistance)
    L=(k.Q[tied].T*k.w[tied])/k.mass[:,None]
    multipliers=np.linalg.lstsq(L,-force/k.mass,rcond=None)[0]
    free_defect=force/k.mass+L@multipliers
    initial=dict(source_sha256=sha(SOURCE),clock_s=s['clock_s'],source_physical_publications=incoming['physical_publications'],
        physical_trace_prefix_bitwise=True,old_surface_histories_unchanged=True,
        physical_memory_trace_error_m=float(max(abs(old_memory-new_memory))),
        original_bulk_storage_J_m=float(eold),independently_reconstructed_bulk_storage_J_m=float(enew),
        reconstruction_storage_difference_J_m=float(enew-eold),raw_force_before_new_tied_reactions_Pa=float(max(abs(weak/k.mass))),
        initial_unbalanced_free_residual_Pa=float(max(abs(free_defect))),
        least_squares_tied_peak_Pa=float(max(abs(multipliers),default=0.)),
        least_squares_peak_is_not_an_admissibility_certificate=True,
        scope='New diagnostic initial state on fixed common geometry; bulk independently reconstructed. This storage difference is NOT physical fracture work or a published transfer.')
    s.update(kernel=k.key,kernel_folder=k.path.name,geometry=copy.deepcopy(ARCS),generation=0,publications=0,physical_publications=0,
        diagnostic_source=dict(path=str(SOURCE),sha256=sha(SOURCE),source_publications=incoming['physical_publications']))
    old.close();return s,initial


def failed_contact(k,s):
    n=len(k.nodes);nq=len(k.s);D=ct.POINTER(ct.c_double)
    c0=np.ascontiguousarray(np.r_[s['wall_memory'],s['memory']]);h=np.ascontiguousarray(np.array([s['histories'][key] for key in HISTORY]).T)
    c=np.empty(n+1);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(12000)
    before=encode(s)
    result=k.lib.cohesive_turn_solve(k.context,c0.ctypes.data_as(D),h.ctypes.data_as(D),.00025,.05,None,None,
        *[a.ctypes.data_as(D) for a in [c,inc,r,J]],ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert result!=0 and 'contact' in error.value.decode().lower() and encode(s)==before
    return error.value.decode()


def run(label,kernel_label,n,degree=2,width=.75,mesh='cartesian',dt=.00025,incoming=None):
    tic=time.perf_counter()
    folder=WORK/label;folder.mkdir(exist_ok=False)
    k=build(kernel_label,n,degree,width,mesh)
    if incoming is None:s,initial=initialize(k)
    else:
        s=json.loads(incoming.read_text());assert s['kernel']==k.key and s['geometry']==k.arcs
        initial=dict(source_sha256=sha(incoming),clock_s=s['clock_s'],same_basis_accepted_restart=True,
            reconstruction_storage_difference_J_m=0.,scope='Own accepted state; no bulk reconstruction or history conversion')
    start_publications=s['physical_publications']
    save(folder/'initial.json',s);save(folder/'reconstruction.json',initial)
    rows=[];snapshots=[copy.deepcopy(s)];clock=s['clock_s'];count=round((.445-clock)/dt);checks={}
    try:
        for j in range(1,count+1):
            target=round(clock+j*dt,12);before=encode(s);p=k.trial(s,target,.44);assert encode(s)==before
            if j==1:
                checks['FD_relative']=derivative(k,s,p)
                assert checks['FD_relative']<1e-6
            if j==30:
                checks['native_rejection']=failed_contact(k,s)
                retry=k.trial(s,target,.44);assert encode(retry['state'])==encode(p['state'])
                save(folder/'restart-incoming.json',s);k.close()
                k=PathKernel(WORK/kernel_label,ARCS,'work')
                restored=json.loads((folder/'restart-incoming.json').read_text());retry=k.trial(restored,target,.44)
                assert encode(retry['state'])==encode(p['state']);checks['native_rejection_retry_restart_bitwise']=True
            incoming=s;s=Admission(k).publish(s,p);assert s['physical_publications']==incoming['physical_publications']+1
            if j==30:
                try:Admission(k).publish(s,p)
                except AssertionError:checks['duplicate_rejected']=True
                else:raise AssertionError('Duplicate physical publication')
            rows.append(p['diagnostics']);save(folder/'last-accepted.json',s)
            if j%8==0 or target in [.44,.445]:snapshots.append(copy.deepcopy(s))
    except Exception as e:
        save(folder/'failed-incoming.json',json.loads(before));save(folder/'failure.json',dict(target_s=target,error=str(e)))
        if rows:table(OUT/(label+'-partial.csv'),rows)
        k.close();raise
    save(folder/'final.json',s);save(folder/'snapshots.json',snapshots);table(OUT/(label+'-history.csv'),rows)
    f0=functionals(k,snapshots[0]);f1=functionals(k,s)
    absolute=sum(abs(r['physical_remainder_J_m']) for r in rows);sliding=sum(r['sliding_loss_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
    save(OUT/(label+'-evolution.json'),dict(label=label,kernel=kernel_label,dt_s=dt,steps=count,initial=initial,checks=checks,
        initial_functionals=f0,final_functionals=f1,growth_m=f1['new_damage_integral_m']-f0['new_damage_integral_m'],
        absolute_physical_remainder_J_m=absolute,sliding_J_m=sliding,fracture_J_m=fracture,
        relative_sliding=absolute/sliding,relative_fracture=absolute/fracture,
        maximum_BE_ledger_J_m=max(abs(r['ledger_error_J_m']) for r in rows),publications=s['physical_publications'],
        publication_increment=s['physical_publications']-start_publications,seconds=time.perf_counter()-tic))
    k.close();print(label,f1['new_damage_integral_m'],f1['full_front_x_m'],flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--own-time',action='store_true');args=parser.parse_args()
    if args.own_time:
        for label,kernel,n,mesh in [('fresh-wide96','wide96',96,'cartesian'),('fresh-fitted96','fitted96',96,'fitted')]:
            states=json.loads((WORK/label/'snapshots.json').read_text());state=next(s for s in states if abs(s['clock_s']-.42025)<1e-12)
            incoming=WORK/(label+'-own-time-incoming.json');save(incoming,state)
            run('own-time-'+kernel,kernel,n,2,.75,mesh,.000125,incoming)
        return
    cases=[('short-narrow64','narrow64',64,2,.1875,'cartesian'),('short-narrow96','narrow96',96,2,.125,'cartesian'),
        ('short-narrow-Q3','narrow-Q3',64,3,.1875,'cartesian'),
        ('short-wide64','wide64',64,2,.75,'cartesian'),('short-wide96','wide96',96,2,.75,'cartesian'),
        ('short-wide-Q3','wide-Q3',64,3,.75,'cartesian'),('short-fitted64','fitted64',64,2,.75,'fitted'),
        ('short-fitted96','fitted96',96,2,.75,'fitted'),('short-fitted-Q3','fitted-Q3',64,3,.75,'fitted'),
        ('short-offset96','offset96',96,2,.6875,'cartesian')]
    for args in cases:run(*args)
    run('short-fitted96-half','fitted96',96,2,.75,'fitted',.000125)
    run('short-wide96-half','wide96',96,2,.75,'cartesian',.000125)


if __name__=='__main__':main()
