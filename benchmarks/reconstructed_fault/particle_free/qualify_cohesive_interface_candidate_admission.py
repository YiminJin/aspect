#!/usr/bin/env python3
"""Independent native master-space, representation and atomic lifecycle checks."""
import copy
import ctypes as ct
import json
import numpy as np
from scipy.linalg import null_space
from cohesive_interface_candidate_admission import WORK,OUT,SOURCE,Kernel,open_state,kernel,candidate_arcs,prepare,attempt,publish,zero_tail,counts
from cohesive_interface_path import encode,save,table,HISTORY
from cohesive_interface_admission import Admission
from qualify_cohesive_interface_load_turn import finite_difference


def reject(f):
    try:f()
    except (AssertionError,RuntimeError):return True
    raise AssertionError('Invalid operation was published')


def main():
    old,s=open_state(SOURCE);before=encode(s);direction,_,tensor=old.sample(s)
    view,w,transfer=prepare(old,s,candidate_arcs(s,direction,2),64,direction,0)
    assert encode(s)==before and transfer['stress_defect_Pa']<1e-7
    assert transfer['pure_tied_state_admissible']
    # Reject a REAL native trial that violates closed contact; then retry the
    # identical physical step. Outputs of the failed trial are never published.
    n=len(view.nodes);nq=len(view.s);D=ct.POINTER(ct.c_double)
    c0=np.ascontiguousarray(view.coefficients(w));h=np.ascontiguousarray(np.array([w['histories'][key] for key in HISTORY]).T)
    da=np.ascontiguousarray([.05,0.]);c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(16000)
    status=view.lib.cohesive_turn_solve_multi(view.context,c0.ctypes.data_as(D),h.ctypes.data_as(D),.00025,da.ctypes.data_as(D),None,None,
        *[v.ctypes.data_as(D) for v in [c,inc,r,J]],ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert status and 'contact' in error.value.decode().lower();assert encode(s)==before
    p=view.trial(w,.41025);again=view.trial(json.loads(encode(w)),.41025)
    assert encode(p['state'])==encode(again['state'])
    # A paired native solve changes only the analytical master elimination.
    reference=Kernel(view.path,view.arcs,structural=False);q=reference.trial(w,.41025)
    stress1=np.einsum('ijk,j->ik',view.field,p['native']['c']);stress0=np.einsum('ijk,j->ik',view.field,q['native']['c'])
    difference=float(np.max(abs(stress1-stress0)));assert difference<1e-7,difference
    coefficient=float(np.max(abs(p['native']['c']-q['native']['c'])))
    q0,n0=counts(s)
    assert np.all(p['native']['c'][2+n0:]==0)
    for key in ['jump','damage','reference','relaxed_jump','absolute_slip']:assert np.all(np.array(p['state']['histories'][key][q0:])==0)
    # Independent projector equivalence, on Q1 Gauss values rather than native
    # SVD data. Rank-two Gauss interpolation on a tied cell fixes both nodes.
    tied=np.array(p['tied'],dtype=bool);mask=np.zeros(n,bool)
    for cell in range(nq//3):
        if all(tied[3*cell:3*cell+3]):mask|=np.any(view.Q[3*cell:3*cell+3]!=0,axis=0)
    masters=np.flatnonzero(~mask);C=null_space(view.Q[tied]);reduced=null_space(view.Q[tied][:,masters])
    E=np.zeros((n,reduced.shape[1]));E[masters]=reduced
    projector=float(np.linalg.norm(C@C.T-E@E.T));assert projector<1e-12
    fd=finite_difference(view,w,p);assert fd<1e-6,fd
    reference.close();view.close()
    # Complete atomic owner, including context reconstruction and stale tokens.
    trial=attempt(old,s,.41025,cells=2);t=publish(s,trial)
    wrong=copy.copy(trial);wrong['formed']=s['formed_arcs']+1
    false_commit=reject(lambda:publish(s,wrong));assert encode(s)==before
    duplicate=reject(lambda:publish(t,trial));assert encode(s)==before
    invalid=copy.deepcopy(trial['native_payload']);invalid['state']['histories']['damage'][0]=-1
    rejected=reject(lambda:Admission(trial['kernel']).publish(trial['input'],invalid));assert encode(s)==before
    save(WORK/'qualified-restart.json',t);restart,restored=open_state(WORK/'qualified-restart.json')
    a=attempt(trial['kernel'],t,.4105,cells=2);b=attempt(restart,restored,.4105,cells=2)
    ta=publish(t,a);tb=publish(restored,b);assert encode(ta)==encode(tb)
    assert ta['physical_publications']==s['physical_publications']+2
    zero_tail(a['kernel'],ta)
    save(OUT/'initial-qualification.json',dict(source=str(SOURCE),source_clock_s=s['clock_s'],retained_geometry_all_formed=True,
        candidate_transfer=transfer,actual_native_contact_rejection=True,rejected_input_unchanged=True,
        native_retry_bitwise=True,default_vs_structural_stress_Pa=difference,coefficient_difference_m=coefficient,
        structural_tail_exact_zero=True,independent_Q1_projector_error=projector,coupled_FD_relative=fd,
        invalid_endpoint_rejected=rejected,zero_material_commit_rejected=false_commit,duplicate_rejected=duplicate,native_context_JSON_restart_bitwise=True,
        physical_publication_increment=2,initial_tensor_Pa=tensor.tolist()))
    # Contexts returned unchanged by a trial are shared with their owner.
    for k in {id(k):k for k in [old,trial['kernel'],restart,a['kernel'],b['kernel']]}.values():k.close()


if __name__=='__main__':main()
