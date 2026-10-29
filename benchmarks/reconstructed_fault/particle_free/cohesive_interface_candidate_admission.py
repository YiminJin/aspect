#!/usr/bin/env python3
"""Experimental atomic formation owner; virgin candidate views carry no history."""
import argparse
import copy
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

import cohesive_interface_load_turn as retained
from cohesive_interface_path import ROOT, R, HISTORY, new_arc, axes, graph, save, table, encode, sha
from cohesive_interface_admission import Admission, bounded_multiplier

WORK=ROOT/'build-particle-free/cohesive-interface-candidate-admission'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_candidate_admission'
SOURCE=retained.WORK/'turn-small64-reproduced/before-extension-2.json'
T0=.410
CELL=.09375
PHYSICAL_HISTORY=[key for key in HISTORY if key!='theta']


class Kernel(retained.Kernel):
    def __init__(self,folder,arcs,structural=True):
        super().__init__(folder,arcs,WORK/'build/libcohesive_interface_turn_solver.so')
        if structural:
            self.lib.cohesive_turn_use_structural_q1_ties.argtypes=[ct.c_void_p]
            self.lib.cohesive_turn_use_structural_q1_ties(self.context)
        self.endpoint_probes=None
        if (folder/'formed-probes.csv').exists():
            self.endpoint_probes=np.loadtxt(folder/'formed-probes.csv',delimiter=',').reshape(len(arcs),5,len(self.nodes)+2)

    def anchor(self,count):
        if self.endpoint_probes is None:
            assert count==len(self.arcs),'A retained kernel can sample only its actual endpoint'
        else:self.probe=self.endpoint_probes[count-1]

    def sample(self,s):
        self.anchor(s['formed_arcs']);tensor=self.probe@self.coefficients(s)
        angle,info=axes(tensor,np.arctan(s['geometry'][s['formed_arcs']-1]['b']))
        return angle,info,tensor


def kernel(arcs,n=64,probe_order=8):
    signature=dict(arcs=arcs,n=n,degree=2,order=6,numerical_support_m=.75,affine_modes=2,
        formed_probe_order=probe_order,version=1)
    name='kernel-'+hashlib.sha256(encode(signature).encode()).hexdigest()[:16];folder=WORK/name
    if not (folder/'native.json').exists():
        size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file())
        assert size<1.2*2**30,'Declared raw-output budget reached'
        folder.mkdir(exist_ok=False);save(folder/'input.json',signature);table(folder/'geometry.csv',arcs)
        np.savetxt(folder/'probe-coefficients.txt',np.zeros(sum(a['cells'] for a in arcs)+1))
        command=[str(WORK/'build/cohesive_interface_path'),str(n),'2',str(folder/'geometry.csv'),str(folder),'6','compress',
            '.75','.75','cartesian',str(folder/'probe-coefficients.txt'),'two-affine','candidate-probes',str(probe_order)]
        with (WORK/(name+'.log')).open('w') as log:
            subprocess.run(command,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    return Kernel(folder,arcs)


def open_state(path):
    s=json.loads(Path(path).read_text())
    folder=Path(s.get('kernel_path',retained.WORK/s['kernel_folder']))
    k=Kernel(folder,s['geometry'])
    if 'formed_arcs' not in s:
        # The selected source has no virgin pieces. Do not reinterpret a
        # previously committed virgin reserve as a new temporary candidate.
        offset=0
        for arc in s['geometry']:
            end=offset+3*arc['cells']
            assert any(v!=0 for key in PHYSICAL_HISTORY for v in s['histories'][key][offset:end])
            offset=end
        s.update(formed_arcs=len(s['geometry']),formed_generation=s['generation'],candidate_revision=0,
            candidate_epoch=-1,candidate_anchor=0,candidate_direction_rad=None,kernel_path=str(folder),
            record_ids=['retained:'+str(q) for q in range(len(k.s))],
            arc_metadata=[dict(status='formed',origin='retained',selected_from_time_s=None,committed_time_s=None) for _ in s['geometry']])
    assert s['kernel']==k.key
    zero_tail(k,s);k.anchor(s['formed_arcs'])
    return k,s


def counts(s):
    cells=sum(a['cells'] for a in s['geometry'][:s['formed_arcs']])
    return 3*cells,cells-1


def zero_tail(k,s):
    q0,n0=counts(s)
    for key in PHYSICAL_HISTORY:assert np.all(np.array(s['histories'][key][q0:])==0),(key,'temporary material history')
    for key in ['delta','memory','nodal_rate']:assert np.all(np.array(s[key][n0:])==0),(key,'temporary native mode carries memory')
    assert np.all(np.array(s['jump_offset'][q0:])==0)
    assert np.all(np.array(s['histories']['theta'][q0:])==s['virgin_theta']),'Temporary age differs from virgin clock'
    assert len(s['record_ids'])==q0


def candidate_arcs(s,direction,number):
    arcs=copy.deepcopy(s['geometry'][:s['formed_arcs']])
    for _ in range(number):arcs.append(new_arc(arcs,direction,CELL,CELL))
    return arcs


def prepare(old,s,arcs,n,direction,epoch):
    """A new trial representation, never an accepted tied insertion."""
    before=encode(s);zero_tail(old,s);q0,n0=counts(s)
    new=kernel(arcs,n);assert arcs[:s['formed_arcs']]==s['geometry'][:s['formed_arcs']]
    t=copy.deepcopy(s);extra_q=len(new.s)-q0;extra_n=len(new.nodes)-n0
    for key in HISTORY:t['histories'][key]=s['histories'][key][:q0]+[s['virgin_theta'] if key=='theta' else 0.]*extra_q
    for key in ['delta','memory','nodal_rate']:t[key]=s[key][:n0]+[0.]*extra_n
    for key in ['jump_offset','last_resistance']:t[key]=s[key][:q0]+[0.]*extra_q
    t.update(geometry=copy.deepcopy(arcs),kernel=new.key,kernel_folder=new.name,kernel_path=str(new.path),
        candidate_direction_rad=direction,candidate_anchor=s['formed_arcs'],candidate_epoch=epoch)
    t['arc_metadata']=copy.deepcopy(s['arc_metadata'][:s['formed_arcs']])+[
        dict(status='candidate',origin='new',selected_from_time_s=s['clock_s'],committed_time_s=None,preferred_deg=float(np.rad2deg(direction)))
        for _ in arcs[s['formed_arcs']:]]
    # Same fixed bulk space, identical retained physical basis columns, and
    # exactly zero removed coordinates: no evolved bulk memory is projected.
    columns_old=np.genfromtxt(old.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
    columns_new=np.genfromtxt(new.path/'columns.csv',delimiter=',',names=True,dtype=None,encoding=None)
    assert np.array_equal(columns_old[:n0+2],columns_new[:n0+2])
    assert np.array_equal(old.K[:n0+2,:n0+2],new.K[:n0+2,:n0+2])
    assert np.array_equal(old.Q[:q0,:n0],new.Q[:q0,:n0])
    assert np.array_equal(old.s[:q0],new.s[:q0]),'Persistent physical Gauss ownership changed'
    c0=old.coefficients(s);c1=new.coefficients(t)
    E0=float(.5*c0@old.K@c0);E1=float(.5*c1@new.K@c1);shift=E1-E0
    assert abs(shift)<=1e-10*max(1,abs(E0))
    assert np.array_equal(old.field[:,:n0+2],new.field[:,:n0+2])
    stress=float(np.max(abs(np.einsum('ijk,j->ik',old.field,c0)-np.einsum('ijk,j->ik',new.field,c1))))
    assert stress<1e-7
    tied=np.array(t['histories']['damage'])==0;r=np.array(t['last_resistance']);r[tied]=0
    force=(new.K@c1)[2:]+new.Q.T@(new.w*r)
    lam,error,peak=bounded_multiplier(new,force,tied)
    assert error<1e-8
    assert encode(s)==before;zero_tail(new,t);new.anchor(t['formed_arcs'])
    return new,t,dict(stress_defect_Pa=stress,storage_defect_J_m=shift,storage_allowance_J_m=1e-10*max(1,abs(E0)),physical_columns_bitwise=True,
        old_records_bitwise=True,intact_virtual_peak_Pa=peak,intact_virtual_error_Pa=error,
        pure_tied_state_admissible=peak<=3000,new_fracture_J_m=0.,discarded_material_memory=0.)


def formed_count(k,s,old_count):
    offset=0;last=old_count;seen_zero=False
    for i,arc in enumerate(s['geometry']):
        end=offset+3*arc['cells']
        if i>=old_count:
            active=any(v!=0 for key in PHYSICAL_HISTORY for v in s['histories'][key][offset:end])
            if active:
                if seen_zero:raise RuntimeError('Noncontiguous new formation needs a topology rule; trial not published')
                last=i+1
            else:seen_zero=True
        offset=end
    return last


def replace_candidate_cache(s,prepared,transfer,base):
    """Change optional trial geometry, never the accepted material registry."""
    assert encode(s)==base,'stale candidate-cache replacement'
    q0,n0=counts(s);assert prepared['formed_arcs']==s['formed_arcs']
    assert prepared['geometry'][:s['formed_arcs']]==s['geometry'][:s['formed_arcs']]
    for key in HISTORY:assert prepared['histories'][key][:q0]==s['histories'][key][:q0]
    for key in ['delta','memory','nodal_rate']:assert prepared[key][:n0]==s[key][:n0]
    for key in ['gamma','alpha_deg','imposed','virgin_theta']:assert prepared[key]==s[key]
    assert prepared['jump_offset'][:q0]==s['jump_offset'][:q0]
    assert prepared['arc_metadata'][:s['formed_arcs']]==s['arc_metadata'][:s['formed_arcs']]
    for key in PHYSICAL_HISTORY:assert all(v==0 for v in prepared['histories'][key][q0:])
    for key in ['delta','memory','nodal_rate']:assert all(v==0 for v in prepared[key][n0:])
    assert all(v==0 for v in prepared['jump_offset'][q0:])
    assert all(v==prepared['virgin_theta'] for v in prepared['histories']['theta'][q0:])
    assert prepared['affine_memory']==s['affine_memory'] and prepared['record_ids']==s['record_ids']
    assert transfer['stress_defect_Pa']<1e-7 and abs(transfer['storage_defect_J_m'])<=transfer['storage_allowance_J_m']
    t=copy.deepcopy(prepared);changed=t['geometry']!=s['geometry']
    t['generation']=s['generation']+int(changed);t['candidate_revision']=s['candidate_revision']+int(changed)
    assert t['publications']==s['publications'] and t['physical_publications']==s['physical_publications']
    assert t['formed_generation']==s['formed_generation'] and t['clock_s']==s['clock_s']
    return t


def publish(s,p):
    assert encode(s)==p['base'],'stale/duplicate atomic publication'
    q0,_=counts(s)
    for key in HISTORY:assert p['input']['histories'][key][:q0]==s['histories'][key][:q0]
    assert p['input']['clock_s']==s['clock_s']
    t=Admission(p['kernel']).publish(p['input'],p['native_payload'])
    n=p['formed'];old=s['formed_arcs'];q0,_=counts(s)
    assert n==formed_count(p['kernel'],t,old),'Zero material cannot be committed as formed'
    assert t['geometry'][:old]==s['geometry'][:old]
    assert t['arc_metadata'][:old]==s['arc_metadata'][:old]
    assert np.all(np.array(t['histories']['damage'][:q0])>=np.array(s['histories']['damage'][:q0]))
    t['formed_arcs']=n;t['formed_generation']=s['formed_generation']+int(n>old)
    t['publications']=s['publications']+int(n>old)
    changed=t['geometry']!=s['geometry'];t['generation']=s['generation']+int(changed)
    t['candidate_revision']=s['candidate_revision']+int(changed)
    for i in range(old,n):
        assert t['arc_metadata'][i]['selected_from_time_s']==s['clock_s'],'Stale direction cannot form material'
        t['arc_metadata'][i].update(status='formed',committed_time_s=t['clock_s'])
    q1,_=counts(t);t['record_ids']=s['record_ids']+['formed:'+str(t['physical_publications'])+':'+str(q) for q in range(q0,q1)]
    assert t['physical_publications']==s['physical_publications']+1
    zero_tail(p['kernel'],t)
    return t


def attempt(old,s,target,n=64,cells=2,cadence=.0025,force_refresh=False):
    base=encode(s);direction,decision,tensor=old.sample(s)
    epoch=int(np.floor((s['clock_s']-T0+1e-12)/cadence))
    refresh=force_refresh or s['candidate_epoch']!=epoch or s['candidate_anchor']!=s['formed_arcs']
    view=old;incoming=s;transfer=dict(stress_defect_Pa=0.,storage_defect_J_m=0.,intact_virtual_peak_Pa=0.,pure_tied_state_admissible=True)
    current_number=max(cells,len(s['geometry'])-s['formed_arcs'])
    rebuilt=False;activation_retry=0;extension_retry=0
    try:
        if refresh:
            view,incoming,transfer=prepare(old,s,candidate_arcs(s,direction,current_number),n,direction,epoch);rebuilt=True
        while True:
            view.anchor(s['formed_arcs'])
            try:p=view.trial(incoming,target)
            except Exception as e:
                save(view.path/f'rejected-physical-{target:.8f}.json',dict(incoming=incoming,target_s=target,error=str(e),
                    original_accepted=s,view_geometry=view.arcs,refresh=refresh,extension_retry=extension_retry))
                raise
            count=formed_count(view,p['state'],s['formed_arcs'])
            # An activation predictor cannot commit its previously reserved
            # direction. Rebuild at the actual incoming accepted stress.
            if count>s['formed_arcs'] and not refresh:
                if view is not old:view.close()
                view,incoming,transfer=prepare(old,s,candidate_arcs(s,direction,current_number),n,direction,epoch)
                refresh=True;rebuilt=True;activation_retry+=1;continue
            if count==len(incoming['geometry']):
                if view is not old:view.close()
                extension_retry+=1;assert extension_retry<=4,'Bounded candidate completion retries exhausted'
                current_number*=2;assert current_number*CELL<=1.5,'Bounded trial candidate extent reached'
                view,incoming,transfer=prepare(old,s,candidate_arcs(s,direction,current_number),n,direction,epoch)
                refresh=True;rebuilt=True;continue
            break
        d=p['diagnostics'];d.update(newly_formed_cells=count-s['formed_arcs'],formed_arcs=count,
            persistent_length_m=float(sum(view.w[:3*sum(a['cells'] for a in incoming['geometry'][:count])])),
            candidate_cells=len(incoming['geometry'])-count,selected_preference_deg=float(np.rad2deg(direction)),
            selection_clock_s=s['clock_s'] if refresh else incoming['arc_metadata'][-1]['selected_from_time_s'],
            activation_retries=activation_retry,extension_retries=extension_retry,refreshed=int(rebuilt),
            transfer_stress_Pa=transfer['stress_defect_Pa'],transfer_storage_J_m=transfer['storage_defect_J_m'],
            intact_virtual_peak_Pa=transfer['intact_virtual_peak_Pa'],intact_virtual_admissible=transfer['pure_tied_state_admissible'],
            admission_fracture_J_m=float(retained.GC*view.w[counts(s)[0]:]@np.array(p['state']['histories']['damage'][counts(s)[0]:])),
            metadata_commit_work_J_m=0.,metadata_commit_fracture_J_m=0.)
        assert encode(s)==base
        return dict(base=base,input=incoming,native_payload=p,kernel=view,formed=count,decision=decision,
            sample_tensor_Pa=tensor.tolist(),transfer=transfer,diagnostics=d)
    except Exception:
        if view is not old:view.close()
        assert encode(s)==base
        raise


def run(label,cells=2,dt=.00025,cadence=.0025,checkpoint=SOURCE,end=.425):
    folder=WORK/label;folder.mkdir(exist_ok=False);start=time.perf_counter();k,s=open_state(checkpoint)
    initial=copy.deepcopy(s);save(folder/'initial.json',s)
    save(folder/'run-input.json',dict(source=str(checkpoint),candidate_cells=cells,dt_s=dt,cadence_s=cadence,end_s=end,
        formed_endpoint_sampling=True,virgin_candidate_rewrite=True,R_m=R,numerical_width_m=.75))
    rows=[];snapshots=[copy.deepcopy(s)];events=[]
    grid=np.round(np.arange(s['clock_s']+dt,end+dt/2,dt),12)
    try:
        for target in grid:
            save(folder/'last-accepted.json',s)
            p=attempt(k,s,float(target),cells=cells,cadence=cadence);t=publish(s,p)
            current,_,_=p['kernel'].sample(t)
            p['diagnostics']['accepted_preferred_deg']=float(np.rad2deg(current))
            if p['formed']>s['formed_arcs']:
                event=dict(time_s=float(target),old_formed=s['formed_arcs'],new_formed=p['formed'],
                    selected_from_time_s=s['clock_s'],preferred_deg=p['diagnostics']['selected_preference_deg'],
                    transfer=p['transfer'],diagnostics=p['diagnostics'])
                events.append(event);print(label,'formed',s['formed_arcs'],'->',p['formed'],target,'preference',event['preferred_deg'],flush=True)
                save(folder/f'before-formation-{len(events)}.json',s);save(folder/f'after-formation-{len(events)}.json',t)
            if p['kernel'] is not k:k.close()
            k=p['kernel'];s=t;rows.append(p['diagnostics'])
            if len(rows)%4==0 or target==end:snapshots.append(copy.deepcopy(s));save(folder/'snapshots.json',snapshots)
            save(folder/'events.json',events)
            assert time.perf_counter()-start<1200,'Declared trajectory wall budget reached'
    except Exception as e:
        save(folder/'last-accepted.json',s);save(folder/'failure.json',dict(error=str(e),target_s=float(target),seconds=time.perf_counter()-start))
        save(folder/'snapshots.json',snapshots);save(folder/'events.json',events)
        if rows:table(OUT/(label+'-partial.csv'),rows)
        k.close();raise
    save(folder/'last-accepted.json',s);save(folder/'final.json',s);save(folder/'snapshots.json',snapshots)
    table(OUT/(label+'-history.csv'),rows)
    error=sum(abs(r['physical_remainder_J_m']) for r in rows);friction=sum(r['sliding_loss_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
    summary=dict(label=label,seconds=time.perf_counter()-start,steps=len(rows),new_persistent_commits=s['publications']-initial['publications'],
        new_formed_cells=s['formed_arcs']-initial['formed_arcs'],physical_publications=s['physical_publications']-initial['physical_publications'],
        physical_error_J_m=error,physical_vs_sliding=error/friction,physical_vs_fracture=error/fracture,final=rows[-1])
    save(OUT/(label+'-summary.json'),summary);k.close();print('completed',label,summary['seconds'],summary['new_formed_cells'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--cells',type=int,default=2);p.add_argument('--dt',type=float,default=.00025)
    p.add_argument('--cadence',type=float,default=.0025);p.add_argument('--checkpoint',type=Path,default=SOURCE);p.add_argument('--end',type=float,default=.425)
    a=p.parse_args();run(a.label,a.cells,a.dt,a.cadence,a.checkpoint,a.end)
