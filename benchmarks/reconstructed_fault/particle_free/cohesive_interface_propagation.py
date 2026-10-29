#!/usr/bin/env python3
"""Maintained experimental route: parallel native kernels and atomic propagation.

Historical milestone drivers remain immutable references. This entry point
reuses their validated owner/return instead of duplicating constitutive code.
"""
import argparse
import copy
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import cohesive_interface_candidate_admission as owner
import cohesive_interface_load_turn as retained
from cohesive_interface_path import ROOT,R,HISTORY,encode,save,sha,table,graph
from check_cohesive_interface_kink import C0,GC,DC,V0,RAD

WORK=ROOT/'build-particle-free/cohesive-interface-propagation'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation'
SOURCE=ROOT/'build-particle-free/cohesive-interface-formed-turn/coarse/final.json'
FINE_SOURCE=ROOT/'build-particle-free/cohesive-interface-formed-turn/fine/final.json'
EXEC=WORK/'build/cohesive_interface_path'
MPI='/opt/openmpi/5.0.6/bin/mpiexec'
QUAL=WORK/'qualification-v3'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/aspect-propagation-mpl',OMPI_MCA_mpi_yield_when_idle='1')


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial');save(tmp,value);tmp.replace(path)


def export_native(folder,arcs,n=64,ranks=4,mesh='cartesian',coefficients=None,legacy=False,fail_rank=None,serial_default=False):
    executable=SOURCE.parents[1]/'build/cohesive_interface_path' if legacy else EXEC
    specification=dict(arcs=arcs,n=n,degree=2,order=6,mesh=mesh,ranks=ranks,parallel=not(legacy or serial_default),
        executable_sha256=sha(executable),numerical_support_m=.75,formed_probe_order=8)
    if coefficients is not None:specification['physical_coefficients_sha256']=hashlib.sha256(np.asarray(coefficients,dtype=np.float64).tobytes()).hexdigest()
    marker=folder/'complete.json'
    if marker.exists():
        assert json.loads(marker.read_text())['specification']==specification
        return
    if folder.exists():
        rejected=folder.with_name(folder.name+'.incomplete-'+str(time.time_ns()));folder.rename(rejected)
    folder.mkdir(parents=True);save(folder/'input.json',specification);table(folder/'geometry.csv',arcs)
    nc=sum(a['cells'] for a in arcs)+1
    np.savetxt(folder/'probe-coefficients.txt',np.zeros(nc) if coefficients is None else coefficients)
    command=[str(executable),str(n),'2',str(folder/'geometry.csv'),str(folder),'6','compress','.75','.75',mesh,
        str(folder/'probe-coefficients.txt'),'two-affine','candidate-probes','8']
    if not(legacy or serial_default):command=[MPI,'--bind-to','none','-n',str(ranks)]+command+['column-mpi']
    save(folder/'command.json',command);env=dict(ENV)
    if fail_rank is not None:env['ASPECT_COHESIVE_PATH_FAIL_RANK']=str(fail_rank)
    tic=time.perf_counter()
    try:
        with (folder/'export.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60 if fail_rank is not None else 300)
        meta=json.loads((folder/'native.json').read_text())
        assert meta['energy_relative']<1e-4 and meta['gradient_FD_relative']<1e-6
        atomic(marker,dict(specification=specification,wall_seconds=time.perf_counter()-tic,meta=meta))
    except Exception as e:
        atomic(folder/'failure.json',dict(error=str(e),wall_seconds=time.perf_counter()-tic,specification=specification))
        raise


class NativeCache:
    def __init__(self,ranks=4,spacing=.09375,namespace='kernels',fail_rank=None):
        self.ranks=ranks;self.spacing=spacing;self.base=WORK/namespace;self.fail_rank=fail_rank

    def __call__(self,arcs,n=64,probe_order=8):
        assert probe_order==8
        assert shutil.disk_usage(WORK).free>2**30,'Declared disk reserve reached'
        size=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file() and not p.is_symlink())
        assert size<3*2**30,'Declared raw-output budget reached'
        signature=dict(arcs=arcs,n=n,ranks=self.ranks,mesh='cartesian',binary=sha(EXEC))
        key=hashlib.sha256(encode(signature).encode()).hexdigest()[:16]
        folder=self.base/('kernel-'+key)
        qualified=QUAL/f'short-{self.ranks}'/('kernel-'+key)
        if self.base==WORK/'kernels' and (qualified/'complete.json').exists():folder=qualified
        export_native(folder,arcs,n,self.ranks,fail_rank=self.fail_rank)
        p=folder/'field-mean.bin.gz'
        if p.exists():
            save(folder/'omitted-mean-diagnostic.json',dict(bytes=p.stat().st_size,sha256=sha(p),
                reason='Unused diagnostic only; center bases, native operators and every history retained',
                regenerate='Repeat command.json into a fresh output folder'))
            p.unlink()
        return owner.Kernel(folder,arcs)

    def attach(self):
        owner.WORK=WORK;owner.OUT=OUT;owner.CELL=self.spacing;owner.kernel=self


def physical(state):
    return {key:value for key,value in state.items() if key not in ['kernel_path','kernel_folder']}


def open_checkpoint(path):
    doc=json.loads(Path(path).read_text());s=doc.get('state',doc)
    k=owner.Kernel(Path(s['kernel_path']),s['geometry']);assert k.key==s['kernel']
    owner.zero_tail(k,s);k.anchor(s['formed_arcs'])
    return k,s,doc.get('material_ledger',{})


def cohort(admitted):
    if admitted is None:return 'retained'
    if .425<admitted<=.440:return 'late_turn'
    if .410<admitted<=.425:return 'early_turn'
    return 'post_turn' if admitted>.440 else 'retained'


def update_ledger(k,s,ledger,previous=None):
    """Identity/provenance only: never modifies a constitutive history."""
    result=copy.deepcopy(ledger);q0,_=owner.counts(s);q=0
    for i,arc in enumerate(s['geometry'][:s['formed_arcs']]):
        meta=s['arc_metadata'][i];admitted=meta['committed_time_s']
        for cell in range(arc['cells']):
            cell_id=s['record_ids'][q]
            for local in range(3):
                j=q+local;key=s['record_ids'][j];a=s['histories']['damage'][j]
                if key not in result:
                    result[key]=dict(id=key,cell_id=cell_id,x_m=float(k.s['x_m'][j]),y_m=float(k.s['y_m'][j]),weight_m=float(k.w[j]),
                        admitted_s=admitted,selected_from_s=meta['selected_from_time_s'],cohort=cohort(admitted),
                        selected_preference_deg=meta.get('preferred_deg'),initial_damage=a,
                        first_damage_bracket_s=None if a>0 else [],first_full_bracket_s=[],
                        absolute_slip_at_full_m=None,theta_at_full=None,already_full_at_source=a==1.)
                    if a==1.:
                        result[key].update(absolute_slip_at_full_m=s['histories']['absolute_slip'][j],theta_at_full=s['histories']['theta'][j])
                record=result[key]
                assert record['x_m']==float(k.s['x_m'][j]) and record['y_m']==float(k.s['y_m'][j]) and record['weight_m']==float(k.w[j])
                if previous is not None:
                    olda=previous['histories']['damage'][j] if j<len(previous['record_ids']) else 0.
                    assert a>=olda
                    if olda==0 and a>0 and record['first_damage_bracket_s']==[]:record['first_damage_bracket_s']=[previous['clock_s'],s['clock_s']]
                    if olda<1 and a==1 and record['absolute_slip_at_full_m'] is None:
                        record.update(first_full_bracket_s=[previous['clock_s'],s['clock_s']],
                            absolute_slip_at_full_m=s['histories']['absolute_slip'][j],theta_at_full=s['histories']['theta'][j])
            q+=3
    assert q==q0 and len(result)==q0
    return result


def material_metrics(k,s,ledger):
    q0,_=owner.counts(s);damage=np.array(s['histories']['damage'][:q0]);weights=k.w[:q0]
    connected=np.zeros(q0,bool);continuing=True
    for q in range(q0):
        if k.s['x_m'][q]<=6:continue
        continuing=continuing and damage[q]==1.;connected[q]=continuing
    groups={};late_cells={}
    for q,key in enumerate(s['record_ids']):
        record=ledger[key];groups.setdefault(record['cohort'],[]).append(q)
        if record['cohort']=='late_turn':late_cells.setdefault(record['cell_id'],[]).append(q)
    mature=[qs for qs in late_cells.values() if all(connected[q] for q in qs)]
    target=[];band_length=0.
    for qs in mature:
        target.append(qs);band_length+=sum(weights[q] for q in qs)
        if band_length>=.1875*(1-1e-10):break # geometric quadrature roundoff, not a damage threshold
    selected=[q for qs in target for q in qs]
    def average(values,indices):
        ix=np.array(indices,dtype=int)
        return float(weights[ix]@np.array(values)[ix]/sum(weights[ix])) if len(ix) else 0.
    slip=np.zeros(q0);change=np.zeros(q0)
    for q,key in enumerate(s['record_ids']):
        r=ledger[key]
        if r['absolute_slip_at_full_m'] is not None:
            slip[q]=s['histories']['absolute_slip'][q]-r['absolute_slip_at_full_m']
            change[q]=abs(s['histories']['theta'][q]/r['theta_at_full']-1)
    out=dict(late_mature_cells=len(mature),late_mature_length_m=float(sum(weights[q] for qs in mature for q in qs)),
        first_two_postfull_slip_Dc=average(slip,selected)/DC,first_two_state_change=average(change,selected),mature_measurement_band_m=band_length)
    for label,indices in groups.items():
        ix=np.array(indices);out[label+'_damage_integral_m']=float(weights[ix]@damage[ix])
        out[label+'_slip_integral_m2']=float(weights[ix]@np.array(s['histories']['jump'])[ix])
    out['demonstration_goal']=bool(band_length>=.1875*(1-1e-10) and out['first_two_postfull_slip_Dc']>=.1 and out['first_two_state_change']>=.05)
    return out


def accepted_points(k,s,native=None):
    q0,_=owner.counts(s);a=np.array(s['histories']['damage'][:q0]);pressure=10000+k.N@k.coefficients(s)
    if native is not None:
        velocity=native[:q0,9];q=native[:q0,6];resistance=native[:q0,10]
    else:
        velocity=(k.Q@np.array(s['nodal_rate']))[:q0]
        e=np.array(s['histories']['jump'][:q0])-np.array(s['histories']['reference'][:q0])
        q=(1-a)*e/(C0*np.where(a==0,1,a));resistance=np.array(s['last_resistance'][:q0])
    radiation=a*RAD*velocity;friction=resistance-q-radiation
    return dict(velocity_m_s=velocity.tolist(),pressure_Pa=pressure[:q0].tolist(),cohesive_q_Pa=q.tolist(),
        friction_Pa=friction.tolist(),radiation_Pa=radiation.tolist(),total_resistance_Pa=resistance.tolist(),
        intact_tied=(a==0).tolist())


def step_diagnostics(p,accepted):
    k=p['kernel'];d=copy.deepcopy(p['diagnostics']);native=p['native_payload']['native'];old=p['input']
    c=native['c'];c0=k.coefficients(old);dc=c-c0;r=native['r'];a=np.array(old['histories']['damage'])
    e=np.array(old['histories']['jump'])-np.array(old['histories']['reference'])
    B0=float(k.w@(.5*(1-a)/(C0*np.where(a==0,1,a))*e*e))
    be=float(.5*dc@k.K@dc);surface=float(k.w@(r[:,6]*(k.Q@native['increment']))-(d['bond_J_m']-B0)-d['fracture_increment_J_m'])
    d.update(BE_bulk_numerical_J_m=be,surface_endpoint_remainder_J_m=surface,
        bulk_storage_change_J_m=d['bulk_J_m']-float(.5*c0@k.K@c0),cohesive_storage_change_J_m=d['bond_J_m']-B0,
        accepted_preferred_deg=float(np.rad2deg(k.sample(accepted)[0])))
    return d


def write_transaction(folder,payload):
    """Atomic accepted state+ledger, then an immutable journal hardlink.

    A crash between the two operations is recovered from accepted.json. The
    canonical file is always replaced, never modified through an old hardlink.
    """
    atomic(folder/'accepted.json',payload)
    journal=folder/'steps'/f'step-{payload["state"]["physical_publications"]:08d}.json'
    if journal.exists():assert json.loads(journal.read_text())==payload
    else:os.link(folder/'accepted.json',journal)


def evolve(label,source=SOURCE,end=.460,dt=.000125,ranks=4,spacing=.09375,extend_hold=False,resume=False):
    qualification=json.loads((OUT/'parallel-qualification.json').read_text())
    assert qualification['passed'] and qualification['binary_sha256']==sha(EXEC),'Parallel qualification required before evolution'
    assert ranks in [2,4],'The selected propagation trajectory must use multiple ranks'
    cache=NativeCache(ranks,spacing);cache.attach();folder=WORK/label;start=time.perf_counter()
    if resume:
        assert (folder/'accepted.json').exists();source=folder/'accepted.json'
        prior=json.loads((folder/'run-input.json').read_text())
        assert prior['dt_s']==dt and prior['spacing_m']==spacing and prior['ranks']==ranks,'Resume the declared numerical settings'
    else:
        folder.mkdir(exist_ok=False);(folder/'steps').mkdir();(folder/'events').mkdir()
        atomic(folder/'run-input.json',dict(source=str(source),source_sha256=sha(source),dt_s=dt,ranks=ranks,spacing_m=spacing,
            original_hold_end_s=end,extend_unchanged_hold=extend_hold,max_hold_end_s=.490,physical_R_m=R,numerical_width_m=.75,
            binary_sha256=sha(EXEC),driver_sha256=sha(Path(__file__))))
    k,s,ledger=open_checkpoint(source);ledger=update_ledger(k,s,ledger);initial=copy.deepcopy(s)
    versions=WORK/'driver-versions';versions.mkdir(exist_ok=True);version=versions/(sha(Path(__file__))+'.py')
    if not version.exists():shutil.copy2(Path(__file__),version)
    if not resume:save(folder/'initial.json',s)
    source_payload=json.loads(Path(source).read_text());elapsed=float(source_payload.get('compute_seconds',0)) if resume else 0.
    points=accepted_points(k,s);payload=dict(state=s,material_ledger=ledger,diagnostics={},points=points,compute_seconds=elapsed)
    if not resume:write_transaction(folder,payload)
    else:
        payload=source_payload
        journal=folder/'steps'/f'step-{s["physical_publications"]:08d}.json'
        if not journal.exists():os.link(folder/'accepted.json',journal)
    assert end<=.490 and initial['clock_s']<=end
    stop=end;count=0
    try:
        while s['clock_s']<stop-1e-12:
            target=float(round(min(s['clock_s']+dt,stop),12));before=encode(s)
            atomic(folder/'before-trial.json',s)
            p=owner.attempt(k,s,target,cells=round(.1875/spacing),cadence=.0025)
            t=owner.publish(s,p);assert encode(s)==before
            new_ledger=update_ledger(p['kernel'],t,ledger,s);metrics=material_metrics(p['kernel'],t,new_ledger)
            d=step_diagnostics(p,t);d.update(metrics);points=accepted_points(p['kernel'],t,p['native_payload']['native']['r'])
            payload=dict(state=t,material_ledger=new_ledger,diagnostics=d,points=points,compute_seconds=elapsed+time.perf_counter()-start)
            write_transaction(folder,payload)
            events=[]
            old_metrics=material_metrics(k,s,ledger)
            if metrics['late_mature_cells']>old_metrics['late_mature_cells']:
                events.append('late-mature-cell-'+str(metrics['late_mature_cells']))
            if p['formed']>s['formed_arcs']:events.append('admission-'+str(t['formed_generation']))
            if target in [.455,.460,.465,.470,.475,.480,.485,.490]:events.append(f'clock-{target:.6f}')
            if metrics['demonstration_goal'] and not old_metrics['demonstration_goal']:events.append('mature-sliding-goal')
            for event in events:
                atomic(folder/'events'/(event+'-before.json'),dict(state=s,material_ledger=ledger))
                os.link(folder/'accepted.json',folder/'events'/(event+'-after.json'))
                print(label,event,target,'full-front',d['full_front_arc_m']-3,'mature cells',metrics['late_mature_cells'],
                    'postfull/Dc',metrics['first_two_postfull_slip_Dc'],flush=True)
            if p['kernel'] is not k:k.close()
            k=p['kernel'];s=t;ledger=new_ledger;count+=1
            if count%20==0 or events:atomic(folder/'last-accepted.json',s)
            assert payload['compute_seconds']<5400,'Main computational budget reached'
            tip=s['geometry'][s['formed_arcs']-1];y,_=graph(tip,1)
            assert sum(k.w[:owner.counts(s)[0]])<7.5 and min(12-tip['end'],6-abs(y))>2*R,'Declared geometry/boundary capacity reached'
            if s['clock_s']>=stop-1e-12 and extend_hold and not metrics['demonstration_goal'] and stop<.490-1e-12:
                stop=float(round(stop+.005,12));print(label,'continue unchanged hold to',stop,flush=True)
    except Exception as e:
        committed=json.loads((folder/'accepted.json').read_text());s=committed['state'];payload=committed
        atomic(folder/'failure.json',dict(error=str(e),last_accepted_s=s['clock_s'],target_s=target,accepted_unchanged=True,
            compute_seconds=elapsed+time.perf_counter()-start))
        atomic(folder/'last-accepted.json',s);k.close();raise
    atomic(folder/'last-accepted.json',s);atomic(folder/'final.json',payload)
    atomic(folder/'completion.json',dict(start_s=initial['clock_s'],end_s=s['clock_s'],steps=count,
        runtime_seconds=time.perf_counter()-start,total_compute_seconds=payload['compute_seconds'],
        demonstrated=material_metrics(k,s,ledger)['demonstration_goal'],ranks=ranks,dt_s=dt))
    k.close();print('completed',label,s['clock_s'],payload['compute_seconds'],flush=True)
    return folder/'final.json'


def selected_ranks():
    values=list(csv.DictReader((OUT/'parallel-cost.csv').open()))
    return int(min((v for v in values if int(v['ranks'])>1),key=lambda v:float(v['wall_seconds']))['ranks'])


def route_check(ranks):
    """The same maintained transaction path used by the long experiment."""
    p=WORK/'route-smoke/completion.json'
    if not p.exists():evolve('route-smoke',end=.44025,ranks=ranks)
    completion=json.loads(p.read_text())
    if completion['end_s']<.4405:evolve('route-smoke',end=.4405,ranks=ranks,resume=True)
    if not (WORK/'route-control/completion.json').exists():evolve('route-control',end=.4405,ranks=ranks)
    a=json.loads((WORK/'route-smoke/final.json').read_text());b=json.loads((WORK/'route-control/final.json').read_text())
    assert encode(physical(a['state']))==encode(physical(b['state']))
    assert a['material_ledger']==b['material_ledger']
    save(OUT/'route-check.json',dict(atomic_state_ledger_checkpoint=True,interrupted_resume_matches_uninterrupted=True,
        physical_publications=4,material_identities_exact=True,ranks=ranks))


def journal(label):
    return [json.loads(p.read_text()) for p in sorted((WORK/label/'steps').glob('step-*.json'))]


def owned_functionals(payload):
    s=payload['state'];k=owner.Kernel(Path(s['kernel_path']),s['geometry']);k.anchor(s['formed_arcs'])
    result=retained.functionals(k,s);result['signed_slip_integral_m2']=float(k.w@np.array(s['histories']['jump']))
    k.close();return result


def window_comparison(label,start_s,end_s):
    baseline=journal('main');control=journal(label)
    before=next(p for p in baseline if abs(p['state']['clock_s']-start_s)<1e-10)
    end=next(p for p in baseline if abs(p['state']['clock_s']-end_s)<1e-10)
    initial=owned_functionals(before);a=owned_functionals(end);b=owned_functionals(control[-1])
    g=a['new_damage_integral_m']-initial['new_damage_integral_m'];cg=b['new_damage_integral_m']-initial['new_damage_integral_m']
    slip=a['signed_slip_integral_m2']-initial['signed_slip_integral_m2'];cs=b['signed_slip_integral_m2']-initial['signed_slip_integral_m2']
    result=dict(label=label,start_s=start_s,end_s=end_s,base_growth_m=g,control_growth_m=cg,growth_difference_m=cg-g,
        growth_relative=(cg/g-1) if g!=0 else None,slip_increment_difference_m2=cs-slip,slip_relative=(cs/slip-1) if slip!=0 else None,
        process_front_difference_m=b['process_front_arc_m']-a['process_front_arc_m'],
        full_front_difference_m=b['full_front_arc_m']-a['full_front_arc_m'])
    if label=='time-mature':
        am=material_metrics_from_payload(end);bm=material_metrics_from_payload(control[-1])
        for name in ['first_two_postfull_slip_Dc','first_two_state_change']:
            result[name+'_base']=am[name];result[name+'_control']=bm[name]
            result[name+'_relative']=bm[name]/am[name]-1
    return result


def focused_checks():
    """Verification windows selected by actual maturation and hold, not sweeps."""
    main=json.loads((WORK/'main/final.json').read_text());ranks=json.loads((WORK/'main/run-input.json').read_text())['ranks']
    cache=NativeCache(ranks);cache.attach();values=journal('main');last=values[-1]['state']['clock_s']
    matured=next((p for p in values if p['diagnostics'].get('late_mature_cells',0)>=1),None)
    windows=[('time-hold',.45375,min(last,.45625))]
    if main['diagnostics'].get('demonstration_goal') and last>=.460:
        windows.append(('time-mature',.45875,.460))
    if matured:
        event=matured['state']['clock_s'];start=round(max(.440,event-.00125),12);end=round(min(last,event+.00125),12)
        if start>.45625 or end<.45375:windows.append(('time-entry',start,end))
    results=[];inputs=WORK/'verification-inputs';inputs.mkdir(exist_ok=True)
    for label,start,end in windows:
        source=next(p for p in values if abs(p['state']['clock_s']-start)<1e-10);path=inputs/(label+'.json');atomic(path,source)
        if not (WORK/label/'final.json').exists():evolve(label,path,end,.0000625,ranks,resume=(WORK/label/'accepted.json').exists())
        results.append(window_comparison(label,start,end))
    # Follow the already-qualified finer .440 s branch. It has its own accepted
    # material records; no coarse history is converted onto a finer layout.
    space_end=min(last,round(matured['state']['clock_s']+.00125,12)) if matured else last
    if not (WORK/'spacing/final.json').exists():evolve('spacing',FINE_SOURCE,space_end,.000125,ranks,.046875,resume=(WORK/'spacing/accepted.json').exists())
    initial=json.loads((WORK/'spacing/initial.json').read_text());ik=owner.Kernel(Path(initial['kernel_path']),initial['geometry'])
    initial_f=retained.functionals(ik,initial);initial_slip=float(ik.w@np.array(initial['histories']['jump']));ik.close()
    common=next(p for p in values if abs(p['state']['clock_s']-space_end)<1e-10)
    a=owned_functionals(common);b=owned_functionals(json.loads((WORK/'spacing/final.json').read_text()))
    i0=owned_functionals(values[0]);g=a['new_damage_integral_m']-i0['new_damage_integral_m'];cg=b['new_damage_integral_m']-initial_f['new_damage_integral_m']
    ds=a['signed_slip_integral_m2']-i0['signed_slip_integral_m2'];cs=b['signed_slip_integral_m2']-initial_slip
    results.append(dict(label='spacing',start_s=.440,end_s=space_end,base_growth_m=g,control_growth_m=cg,growth_difference_m=cg-g,
        growth_relative=cg/g-1 if g else None,slip_increment_difference_m2=cs-ds,slip_relative=cs/ds-1 if ds else None,
        process_front_difference_m=b['process_front_arc_m']-a['process_front_arc_m'],full_front_difference_m=b['full_front_arc_m']-a['full_front_arc_m']))
    write_rows(OUT/'focused-uncertainty.csv',results)
    save(OUT/'focused-uncertainty.json',dict(comparisons=results,growth_slip_limit=.05,front_limit_m=.09375,
        passed=all((r['growth_relative'] is None or abs(r['growth_relative'])<.05) and (r['slip_relative'] is None or abs(r['slip_relative'])<.05)
            and all(abs(r.get(key,0))<.05 for key in ['first_two_postfull_slip_Dc_relative','first_two_state_change_relative'])
            and abs(r['process_front_difference_m'])<=.09375 and abs(r['full_front_difference_m'])<=.09375 for r in results),
        fine_branch_not_a_history_conversion=True))
    # Current-pressure impact and lifecycle checks are implemented in the same
    # maintained entry point, using the actual event source selected above.
    cache.attach() # restore the main spacing after the independent fine branch
    pressure_impact(matured or values[-2],ranks)


def pressure_probe(k,old,target,N,base):
    """Frozen numerical-operator sensitivity; never a published trajectory.

    Keep current-pressure derivatives, both mechanical affine components and
    every old history. Only the measured normal-work discrepancy is perturbed.
    """
    D=ct.POINTER(ct.c_double);n=len(k.nodes);nq=len(k.s);dt=target-old['clock_s']
    arrays=[np.ascontiguousarray(v) for v in [k.K,k.Q,N,N,k.w,k.mass]]
    context=k.lib.cohesive_turn_create_multi(n,nq,2,*[a.ctypes.data_as(D) for a in arrays])
    k.lib.cohesive_turn_use_structural_q1_ties(context)
    incoming=np.ascontiguousarray(k.coefficients(old));hist=np.ascontiguousarray(np.array([old['histories'][key] for key in HISTORY]).T)
    da=np.ascontiguousarray(retained.loading(target)[0]-np.array(old['imposed']))
    # A changed normal operator changes the nonlinear active-set problem.
    # Reuse the qualified admission predictor with that same operator; a root
    # or active set for the old operator is not an independent root here.
    search=copy.copy(k);search.N=N
    guess,hint=search.predictor(old,dt,da)
    c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(16000)
    try:
        status=k.lib.cohesive_turn_solve_multi(context,incoming.ctypes.data_as(D),hist.ctypes.data_as(D),dt,da.ctypes.data_as(D),
            guess.ctypes.data_as(D),hint.ctypes.data_as(ct.POINTER(ct.c_int)),*[a.ctypes.data_as(D) for a in [c,inc,r,J]],
            ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
        assert status==0,error.value.decode()
    finally:k.lib.cohesive_turn_destroy(context)
    return dict(c=c,increment=inc,r=r,J=J,tied=ties,meta=meta)


def pressure_impact(payload,ranks):
    from scipy.linalg import null_space
    from qualify_cohesive_interface_load_turn import finite_difference
    assert owner.CELL==.09375,'Restore main numerical settings before the event probe'
    folder=WORK/'pressure-impact-main';folder.mkdir(exist_ok=True)
    source=folder/'incoming.json';atomic(source,payload);k,s,ledger=open_checkpoint(source);before=encode(s);target=round(s['clock_s']+.000125,12)
    p=owner.attempt(k,s,target,cells=2);b=p['kernel'];old=p['input'];base=p['native_payload']
    fd=finite_difference(b,old,base);assert fd<1e-6,fd
    accepted=owner.publish(s,p);again=owner.attempt(k,s,target,cells=2);retried=owner.publish(s,again)
    assert encode(physical(accepted))==encode(physical(retried)) and encode(s)==before
    rk,rs,rl=open_checkpoint(source);restart=owner.attempt(rk,rs,target,cells=2);restarted=owner.publish(rs,restart)
    assert encode(physical(accepted))==encode(physical(restarted))
    try:owner.publish(accepted,p)
    except AssertionError:duplicate=True
    else:raise AssertionError('Duplicate publication')
    # Independent fitted bulk reconstruction of this actual physical incoming
    # surface jump/Maxwell field. Never replace the accepted main bulk state.
    fit_folder=folder/'fitted64';export_native(fit_folder,old['geometry'],n=64,ranks=ranks,mesh='fitted',coefficients=b.coefficients(old))
    fit=owner.Kernel(fit_folder,old['geometry']);fit.anchor(old['formed_arcs']);assert np.array_equal(fit.s,b.s) and np.array_equal(fit.Q,b.Q)
    delta=fit.N-b.N;np.savetxt(folder/'normal-work-difference.csv',delta,delimiter=',')
    c=base['native']['c'];dpressure=delta@c;ties=np.array(base['tied'],bool);C=null_space(b.Q[ties]);rp=base['native']['r'][:,12]
    force=b.Q.T@(b.w*rp*dpressure);J=base['native']['J']
    response=-C@np.linalg.solve(C.T@J@C,C.T@force)
    linear_defect=float(np.linalg.norm(C.T@(J@response+force))/max(1,np.linalg.norm(C.T@force)))
    nonlinear=pressure_probe(b,old,target,fit.N,base)
    late=np.zeros(len(b.s),bool);offset=0
    for arc,meta in zip(old['geometry'],old['arc_metadata']):
        end=offset+3*arc['cells'];late[offset:end]=cohort(meta['committed_time_s'])=='late_turn';offset=end
    weights=b.w*late;mass=sum(weights);assert mass>0
    mean=lambda v:float(weights@v/mass)
    velocity=base['native']['r'][:,9];probe_v=nonlinear['r'][:,9];mean_v=mean(velocity)
    rms=lambda v:float(np.sqrt(weights@(v*v)/mass))
    growth=float(b.w@(base['native']['r'][:,1]-np.array(old['histories']['damage'])))
    probe_growth=float(b.w@(nonlinear['r'][:,1]-np.array(old['histories']['damage'])))
    pressure_data=dict(source_time_s=s['clock_s'],target_s=target,p0_Pa=10000,pressure_lagged=False,
        common_physical_incoming_jump_memory=True,accepted_history_replaced=False,
        late_pressure_discrepancy_RMS_Pa=rms(dpressure),late_friction_force_discrepancy_RMS_Pa=rms(rp*dpressure),
        late_velocity_base_m_s=mean_v,late_velocity_probe_m_s=mean(probe_v),
        late_velocity_relative=mean(probe_v)/mean_v-1 if mean_v else None,
        growth_base_m=growth,growth_probe_m=probe_growth,growth_relative=probe_growth/growth-1 if growth else None,
        complete_current_pressure_tangent=True,coupled_FD_relative=fd,linearized_balance_relative=linear_defect,
        predictor_recomputed_for_measured_normal_operator=True,
        numerical_normal_operator_probe_not_a_physical_trajectory=True,probe_native_weak_Pa=float(nonlinear['meta'][0]),
        material_response_supported=(mean_v!=0 and abs(mean(probe_v)/mean_v-1)<.05 and (growth==0 or abs(probe_growth/growth-1)<.05)))
    # Full fitted one-step response is a separate approximation diagnostic;
    # incoming bulk stresses differ, so its increment is not a pure time error.
    fs=copy.deepcopy(old);fs.update(kernel=fit.key,kernel_path=str(fit.path),kernel_folder=fit.name)
    fp=fit.trial(fs,target)
    pressure_data.update(fitted_bulk_one_step_late_velocity_m_s=mean(fp['native']['r'][:,9]),
        fitted_incoming_bulk_storage_difference_J_m=float(.5*fit.coefficients(fs)@fit.K@fit.coefficients(fs)-.5*b.coefficients(old)@b.K@b.coefficients(old)),
        full_fitted_step_contains_initial_mechanical_representation_difference=True)
    tensor_rows=[];load_rows=[]
    for label,kernel in [('cartesian64',b),('fitted64',fit)]:
        kernel.anchor(accepted['formed_arcs']);tensor=kernel.probe@c
        from cohesive_interface_path import axes
        direction,_=axes(tensor,np.arctan(accepted['geometry'][accepted['formed_arcs']-1]['b']))
        tensor_rows.append(dict(approximation=label,xx_Pa=float(tensor[0]),yy_Pa=float(tensor[1]),zz_Pa=float(tensor[2]),
            xy_Pa=float(tensor[3]),pprime_Pa=float(tensor[4]),preferred_deg=float(np.rad2deg(direction))))
        endpoint=accepted['geometry'][accepted['formed_arcs']-1]['end']
        centre=float(np.average(b.s['x_m'],weights=weights))
        raw=np.loadtxt(kernel.path/'shear.csv',delimiter=',')@c
        for region,x,width in [('old',4.5,.5),('bend',6.4,.25),('late_turn',centre,.2),('zone',endpoint-.25,.2),('ahead',endpoint+.1,.15)]:
            v=np.exp(-.5*((kernel.nodes['x_m']-x)/width)**2);test=kernel.Q@v;mass=float(kernel.w@test)
            load_rows.append(dict(approximation=label,region=region,x_centre_m=x,sigma_m=width,test_mass_m=mass,
                weak_shear_Pa=float(-(kernel.K@c)[2:]@v/mass),normal_perturbation_Pa=float(kernel.w@(test*(kernel.N@c))/mass),
                raw_shear_Pa=float(kernel.w@(test*raw)/mass),raw_normal_jump_RMS_Pa=float(np.sqrt(kernel.w@(test*((kernel.raw_Np-kernel.raw_Nm)@c)**2)/mass))))
    metric=np.array([1,1,1,2]);keys=['xx_Pa','yy_Pa','zz_Pa','xy_Pa']
    reference=np.array([tensor_rows[-1][key] for key in keys]);difference=np.array([tensor_rows[0][key] for key in keys])-reference
    pressure_data.update(selection_tensor_relative=float(np.sqrt(metric@(difference*difference)/(metric@(reference*reference)))),
        selection_angle_difference_deg=tensor_rows[0]['preferred_deg']-tensor_rows[-1]['preferred_deg'],
        geometry_spacing_m=owner.CELL)
    for row in load_rows:
        ref=next(r for r in load_rows if r['region']==row['region'] and r['approximation']=='fitted64')
        for key in ['weak_shear_Pa','normal_perturbation_Pa']:
            row[key+'_difference']=row[key]-ref[key];row[key+'_relative']=(row[key]-ref[key])/abs(ref[key]) if ref[key] else None
    table(OUT/'event-tensors.csv',tensor_rows);table(OUT/'event-weak-loads.csv',load_rows)
    save(OUT/'pressure-impact.json',pressure_data)
    rows=[]
    for q in range(len(b.s)):
        rows.append(dict(x_m=float(b.s['x_m'][q]),y_m=float(b.s['y_m'][q]),weight_m=float(b.w[q]),late_turn=bool(late[q]),
            current_normal_Pa=float(10000+b.N[q]@c),fitted_same_field_normal_Pa=float(10000+fit.N[q]@c),
            raw_minus_normal_Pa=float(10000+b.raw_Nm[q]@c),raw_plus_normal_Pa=float(10000+b.raw_Np[q]@c),
            base_velocity_m_s=float(velocity[q]),normal_probe_velocity_m_s=float(probe_v[q]),
            linear_velocity_change_m_s=float((b.Q@response)[q]/.000125),
            base_damage=float(base['native']['r'][q,1]),normal_probe_damage=float(nonlinear['r'][q,1])))
    table(OUT/'pressure-impact-profiles.csv',rows)
    save(OUT/'mature-lifecycle.json',dict(source_time_s=s['clock_s'],target_s=target,original_accepted_unchanged=encode(s)==before,
        deterministic_retry=True,same_rank_rebuilt_context_restart=True,duplicate_rejected=duplicate,
        single_publication=accepted['physical_publications']==s['physical_publications']+1,coupled_FD_relative=fd))
    for v in {id(v):v for v in [k,b,again['kernel'],rk,restart['kernel'],fit]}.values():v.close()


def write_rows(path,rows):
    keys=set().union(*(r.keys() for r in rows));first=[k for k in ['label','time_s','record_id'] if k in keys]
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=first+sorted(keys-set(first)));writer.writeheader();writer.writerows(rows)


def summarize(label):
    values=journal(label);first,last=values[0],values[-1];history=[];profiles=[];tracked=[]
    ledger=last['material_ledger'];ids=last['state']['record_ids'];late_cells=[]
    for key in ids:
        r=ledger[key]
        if r['cohort']=='late_turn' and r['cell_id'] not in late_cells:late_cells.append(r['cell_id'])
    tracked_ids={key for key,r in ledger.items() if r['cell_id'] in late_cells[:round(.1875/(.046875 if label=='spacing' else .09375))]}
    recurrences=dict(maximum_Maxwell_error_m=0.,maximum_affine_Maxwell_error=0.,maximum_age_error=0.,maximum_trace_error_m=0.,maximum_RSF_force_error_Pa=0.)
    Qcache={}
    selected=[]
    for i,p in enumerate(values):
        s=p['state'];d=p['diagnostics'];h=s['histories'];dt=d.get('dt_s',0.)
        if s['kernel_path'] not in Qcache:Qcache[s['kernel_path']]=np.loadtxt(Path(s['kernel_path'])/'Q.csv',delimiter=',')
        Q=Qcache[s['kernel_path']]
        recurrences['maximum_trace_error_m']=max(recurrences['maximum_trace_error_m'],float(max(abs(Q@np.array(s['memory'])-h['relaxed_jump']))),
            float(max(abs(Q@np.array(s['delta'])+np.array(s['jump_offset'])-h['jump']))))
        a=np.array(h['damage'][:len(s['record_ids'])]);vel=np.array(p['points']['velocity_m_s']);theta=np.array(h['theta'][:len(vel)])
        mu=.005*np.arcsinh(abs(vel)*np.exp((.1+.01*np.log(theta))/.005)/(2*V0))
        expected_force=a*np.sign(vel)*np.array(p['points']['pressure_Pa'])*mu
        recurrences['maximum_RSF_force_error_Pa']=max(recurrences['maximum_RSF_force_error_Pa'],float(max(abs(expected_force-p['points']['friction_Pa']))))
        if d:history.append(dict(label=label,**d))
        if i:
            old=values[i-1]['state'];beta=1/(1+.01*dt)
            n=len(s['memory']);oldm=np.pad(old['memory'],(0,n-len(old['memory'])));oldd=np.pad(old['delta'],(0,n-len(old['delta'])))
            expected=beta*(oldm+np.array(s['delta'])-oldd)
            affine=beta*(np.array(old['affine_memory'])+np.array(s['imposed'])-np.array(old['imposed']))
            recurrences['maximum_Maxwell_error_m']=max(recurrences['maximum_Maxwell_error_m'],float(max(abs(expected-s['memory']))))
            recurrences['maximum_affine_Maxwell_error']=max(recurrences['maximum_affine_Maxwell_error'],float(max(abs(affine-s['affine_memory']))))
            nq=len(h['theta']);oldtheta=np.pad(old['histories']['theta'],(0,nq-len(old['histories']['theta'])),constant_values=old['virgin_theta'])
            oldjump=np.pad(old['histories']['jump'],(0,nq-len(old['histories']['jump'])))
            expected_theta=(oldtheta+V0*dt/DC)/(1+abs(np.array(h['jump'])-oldjump)/DC)
            recurrences['maximum_age_error']=max(recurrences['maximum_age_error'],float(max(abs(expected_theta-h['theta']))))
        important=i%4==0 or i==len(values)-1 or s['clock_s'] in [.455,.460] or d.get('late_mature_cells',0)!=(values[i-1]['diagnostics'].get('late_mature_cells',0) if i else 0)
        if important:selected.append(p)
        for q,key in enumerate(s['record_ids']):
            r=ledger[key];row=dict(label=label,time_s=s['clock_s'],record_id=key,cell_id=r['cell_id'],cohort=r['cohort'],
                x_m=r['x_m'],y_m=r['y_m'],weight_m=r['weight_m'],admitted_s=r['admitted_s'],damage=h['damage'][q],
                jump_m=h['jump'][q],absolute_slip_m=h['absolute_slip'][q],theta=h['theta'][q],age_s=h['theta'][q]*DC/V0,
                relaxed_jump_m=h['relaxed_jump'][q],reference_jump_m=h['reference'][q],
                **{name:data[q] for name,data in p['points'].items()})
            if important:profiles.append(row)
            if key in tracked_ids:tracked.append(row)
    write_rows(OUT/(label+'-history.csv'),history);write_rows(OUT/(label+'-profiles.csv'),profiles);write_rows(OUT/(label+'-tracked-material.csv'),tracked)
    rows=[]
    for key in ids:
        r=copy.deepcopy(ledger[key]);q=ids.index(key)
        for name in ['first_damage_bracket_s','first_full_bracket_s']:
            bracket=r.pop(name);r[name+'_lower']=bracket[0] if bracket else None;r[name+'_upper']=bracket[1] if bracket else None
        r.update(final_damage=last['state']['histories']['damage'][q],final_theta=last['state']['histories']['theta'][q],
            final_velocity_m_s=last['points']['velocity_m_s'][q],final_cohesive_q_Pa=last['points']['cohesive_q_Pa'][q],
            final_friction_Pa=last['points']['friction_Pa'][q],final_normal_Pa=last['points']['pressure_Pa'][q],
            postfull_slip_Dc=(last['state']['histories']['absolute_slip'][q]-r['absolute_slip_at_full_m'])/DC if r['absolute_slip_at_full_m'] is not None else None)
        rows.append(r)
    write_rows(OUT/(label+'-material-ledger.csv'),rows)
    error=sum(abs(d['physical_remainder_J_m']) for d in history);sliding=sum(d['sliding_loss_J_m'] for d in history);fracture=sum(d['fracture_increment_J_m'] for d in history)
    initial_f=owned_functionals(first);final_f=owned_functionals(last)
    result=dict(label=label,initial_s=first['state']['clock_s'],final_s=last['state']['clock_s'],steps=len(history),
        physical_publications=last['state']['physical_publications']-first['state']['physical_publications'],
        formed_transactions=last['state']['publications']-first['state']['publications'],
        damage_growth_m=final_f['new_damage_integral_m']-initial_f['new_damage_integral_m'],
        full_front_beyond_seed_m=final_f['full_front_arc_m']-3,process_front_beyond_seed_m=final_f['process_front_arc_m']-3,
        absolute_physical_work_remainder_J_m=error,physical_vs_sliding=error/sliding if sliding else None,
        physical_vs_fracture=error/fracture if fracture else None,sliding_J_m=sliding,fracture_J_m=fracture,
        maximum_exact_ledger_J_m=max(abs(d['ledger_error_J_m']) for d in history),
        minimum_contact_Pa=min(d['minimum_normal_Pa'] for d in history),maximum_weak_Pa=max(d['weak_error_Pa'] for d in history),
        recurrence_checks=recurrences,final_material_metrics=material_metrics_from_payload(last),
        compute_seconds=last['compute_seconds'],completion=json.loads((WORK/label/'completion.json').read_text()) if (WORK/label/'completion.json').exists() else None)
    save(OUT/(label+'-summary.json'),result)
    return values,selected,result,tracked_ids


def material_metrics_from_payload(p):
    s=p['state'];k=owner.Kernel(Path(s['kernel_path']),s['geometry']);m=material_metrics(k,s,p['material_ledger']);k.close();return m


def plot_results():
    os.environ['MPLCONFIGDIR']=ENV['MPLCONFIGDIR']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    NativeCache().attach();values,frames,summary,tracked_ids=summarize('main')
    others={}
    for label in ['time-hold','time-entry','time-mature','spacing']:
        if (WORK/label/'final.json').exists():others[label]=summarize(label)
    history=[v['diagnostics'] for v in values if v['diagnostics']];t=np.array([r['time_s'] for r in history]);final=values[-1]
    ledger=final['material_ledger'];last=final['state'];limit=max(a['end'] for a in last['geometry'])+.2
    def curve(ax,p,limits=(5.8,None,-.3,.8)):
        s=p['state'];q0=len(s['record_ids'])
        for i,a in enumerate(s['geometry']):
            z=np.linspace(0,1,20);y,_=graph(a,z);formed=i<s['formed_arcs']
            ax.plot(a['x']+(a['end']-a['x'])*z,y,c='.3' if formed else '.7',ls='-' if formed else '--',lw=1.5)
        x=np.array([p['material_ledger'][key]['x_m'] for key in s['record_ids']]);y=np.array([p['material_ledger'][key]['y_m'] for key in s['record_ids']])
        ax.scatter(x,y,c=s['histories']['damage'][:q0],cmap='viridis',vmin=0,vmax=1,s=16,zorder=3)
        ix=np.arange(0,q0,5);k=owner.Kernel(Path(s['kernel_path']),s['geometry']);vel=np.array(p['points']['velocity_m_s'])
        ax.quiver(x[ix],y[ix]-.12,5*vel[ix]*k.s['tx'][ix],5*vel[ix]*k.s['ty'][ix],angles='xy',scale_units='xy',scale=1,color='#aa3377',width=.003)
        f=retained.functionals(k,s)
        for name,marker,c in [('process','^','#e69f00'),('half','D','white'),('full','s','red')]:
            if name=='full' and not f['new_full_front_available']:continue
            ax.scatter(f[name+'_front_x_m'],f[name+'_front_y_m'],marker=marker,c=c,edgecolors='black',s=48,zorder=5)
        k.close();left=min(r['x_m'] for r in ledger.values() if r['cohort']=='late_turn');right=max(r['x_m'] for r in ledger.values() if r['cohort']=='late_turn')
        ax.axvspan(left,right,color='#cc79a7',alpha=.12,label='Late-turn material')
        ax.set(xlim=(limits[0],limits[1] or limit),ylim=limits[2:],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained');curve(ax,final)
    ax.legend(loc='upper right');fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=ax,label='Surface damage a',shrink=.7)
    fig.suptitle(f'Actual formed fault at t={last["clock_s"]:.6f} s; zero-history tail dashed\nRed square a=1; diamond a=.5; triangle a=.05; arrows = slip rate ×5 s')
    fig.savefig(OUT/'propagation.png',dpi=170);plt.close(fig)
    # Physical domain and prescribed loading, independently of the tip zoom.
    fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.2]})
    ax=axs[0];ax.plot([0,12,12,0,0],[-6,-6,6,6,-6],c='black')
    curve(ax,final,(0,12,-6,6));ax.plot([3,6],[0,0],c='black',lw=4,label='Original seed (3–6 m)')
    # Directions of displacement, not stress or crack direction; arrows scaled
    # for visibility. The two native affine strain modes are fixed in the lab.
    coeff=np.array(last['imposed']);axial=.41954981558863996
    E=coeff[0]*np.array([[-axial,.5],[.5,axial]])+coeff[1]*np.array([[-.5,-axial],[-axial,.5]])
    walls=np.array([[0,-4],[0,4],[12,-4],[12,4],[2,-6],[10,-6],[2,6],[10,6]],float);u=(E@walls.T).T
    ax.quiver(walls[:,0],walls[:,1],60*u[:,0],60*u[:,1],angles='xy',scale_units='xy',scale=1,color='#0072b2',width=.006)
    ax.set(xlim=(-1,13),ylim=(-7,7))
    ax.set_title('Finite 12 m × 12 m specimen\nAll-wall affine displacement; arrows ×60')
    ax.legend(loc='lower left',fontsize=8)
    times=np.linspace(.400,.460,241);load=np.array([retained.loading(v)[0] for v in times]);alpha=np.array([retained.loading(v)[1] for v in times])
    table(OUT/'boundary-loading.csv',[dict(time_s=float(t),imposed_0=float(v[0]),imposed_1=float(v[1]),alpha_deg=float(np.rad2deg(a))) for t,v,a in zip(times,load,alpha)])
    axs[1].plot(times,1000*load[:,0],label='Affine mode 0');axs[1].plot(times,1000*load[:,1],label='Affine mode 1')
    axs[1].axvline(.440,c='gray',ls='--',label='This continuation starts');axs[1].axvline(.455,c='black',ls=':',label='Displacement hold')
    axs[1].set(xlabel='Time (s)',ylabel='Imposed dimensionless mode coefficient ×1000');axs[1].legend(loc='upper right',fontsize=8);axs[1].grid(alpha=.2)
    other=axs[1].twinx();other.plot(times,np.rad2deg(alpha),c='#009e73',ls='-.');other.set_ylabel('Physical boundary loading rotation (degrees)',color='#009e73')
    fig.suptitle('Background confinement p₀=10 kPa; solved pressure perturbation has zero mean\nClosed contact, quasi-static bulk, radiation damping; this is a finite specimen')
    fig.savefig(OUT/'specimen.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key,ls,name in [('process','-','a=.05'),('half','-.','a=.5'),('full','--','a=1')]:axs[0,0].plot(t,[r[key+'_front_arc_m']-3 for r in history],ls,label=name)
    axs[0,0].axhspan(1.125,1.6875,color='#cc79a7',alpha=.12,label='Late-turn material')
    axs[0,1].plot(t,[r['late_mature_length_m'] for r in history],label='Connected late-turn mature length')
    entire_band=np.array([r['late_mature_length_m']>=.1875*(1-1e-10) for r in history])
    axs[1,0].plot(t,np.where(entire_band,[r['first_two_postfull_slip_Dc'] for r in history],np.nan),label='Fixed .1875 m: postfull slip / Dc')
    axs[1,1].plot(t,np.where(entire_band,100*np.array([r['first_two_state_change'] for r in history]),np.nan),label='Fixed .1875 m: age change')
    for ax,y in zip(axs.flat,['Front beyond seed (m)','Mature late-turn length (m)','Slip / Dc','Age change (%)']):
        ax.axvline(.455,c='black',ls=':',label='Loading stops' if ax is axs[0,0] else None);ax.set(xlabel='Time (s)',ylabel=y,xlim=(values[0]['state']['clock_s'],last['clock_s']));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Material identities distinguish true mature advance from a newly drawn tail')
    fig.savefig(OUT/'fronts.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(5,1,figsize=(11,12),sharex=True,layout='constrained')
    cells=[]
    for key in last['record_ids']:
        if key in tracked_ids and ledger[key]['cell_id'] not in cells:cells.append(ledger[key]['cell_id'])
    for index,cell in enumerate(cells[:2]):
        selected_ids={key for key in tracked_ids if ledger[key]['cell_id']==cell};rows=[]
        for p in values:
            s=p['state'];qs=[q for q,key in enumerate(s['record_ids']) if key in selected_ids]
            w=np.array([ledger[s['record_ids'][q]]['weight_m'] for q in qs]);avg=lambda x:float(w@np.array(x)[qs]/sum(w))
            rows.append([s['clock_s'],avg(s['histories']['damage']),avg(p['points']['velocity_m_s']),avg(s['histories']['theta']),
                avg(p['points']['pressure_Pa']),avg(p['points']['cohesive_q_Pa']),avg(p['points']['friction_Pa']),avg(p['points']['radiation_Pa'])])
        x=np.array(rows);name=f'Cell admitted {ledger[next(iter(selected_ids))]["admitted_s"]:.6f} s';color=['#0072b2','#d55e00'][index]
        axs[0].plot(x[:,0],x[:,1],c=color,label=name);axs[1].plot(x[:,0],x[:,2],c=color,label=name)
        axs[2].plot(x[:,0],x[:,3],c=color,label=name)
        axs[3].plot(x[:,0],x[:,4]/1000,c=color,label=name)
        for column,style,label in [(5,'-','cohesive'),(6,'--','friction'),(7,':','radiation')]:axs[4].plot(x[:,0],x[:,column],c=color,ls=style,label=f'Cell {index+1} {label}')
    for ax,y in zip(axs,['Damage a','Slip rate (m/s)','Normalized age θ','Normal pressure (kPa)','Resistance (Pa)']):
        ax.axvline(.455,c='black',ls=':');ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    axs[-1].set_xlabel('Time (s)');fig.suptitle('The same newly admitted material: decohesion, slip, age and resistance')
    fig.savefig(OUT/'mature-material.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key,name in [('trapezoid_work_J_m','Trapezoidal work'),('endpoint_work_J_m','Endpoint work'),('bulk_storage_change_J_m','Bulk storage change'),('cohesive_storage_change_J_m','Cohesive storage change'),
                     ('fracture_increment_J_m','Fracture work'),('sliding_loss_J_m','Sliding loss'),('maxwell_loss_J_m','Maxwell loss')]:axs[0,0].plot(t,np.cumsum([r[key] for r in history]),label=name)
    for key,name in [('BE_bulk_numerical_J_m','Bulk BE term'),('surface_endpoint_remainder_J_m','Surface endpoint remainder'),('physical_remainder_J_m','Physical trapezoid remainder')]:axs[0,1].plot(t,np.cumsum([r[key] for r in history]),label=name)
    axs[1,0].plot(t,[r['physical_remainder_J_m'] for r in history],label='Each accepted step')
    err=np.cumsum([abs(r['physical_remainder_J_m']) for r in history]);fr=np.cumsum([r['fracture_increment_J_m'] for r in history]);sl=np.cumsum([r['sliding_loss_J_m'] for r in history])
    axs[1,1].plot(t,100*err/sl,label='/ sliding');axs[1,1].plot(t,100*err/fr,label='/ added fracture');axs[1,1].axhline(1,c='black',ls=':')
    for ax,y in zip(axs.flat,['Cumulative energy/work (J/m)','Numerical remainder (J/m)','Physical remainder (J/m)','Absolute physical remainder (%)']):
        ax.axvline(.455,c='black',ls=':');ax.set(xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'energy.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    axs[0].plot(t,[r['full_front_arc_m']-3 for r in history],label='Main a=1');axs[1].plot(t,[r['new_damage_integral_m'] for r in history],label='Main ∫a ds')
    for label,(data,_,_,_) in others.items():
        rows=[p['diagnostics'] for p in data if p['diagnostics']];ctimes=[r['time_s'] for r in rows]
        axs[0].plot(ctimes,[r['full_front_arc_m']-3 for r in rows],'--',label=label)
        axs[1].plot(ctimes,[r['new_damage_integral_m'] for r in rows],'--',label=label)
    for ax,y in zip(axs,['Developed front beyond seed (m)','Damage integral beyond seed (m)']):ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend();ax.axvline(.455,c='black',ls=':')
    axs[-1].set_xlabel('Time (s)');fig.savefig(OUT/'uncertainty.png',dpi=170);plt.close(fig)
    # Actual accepted frames only; no spatial/temporal history interpolation.
    fig,axs=plt.subplots(2,1,figsize=(12,6),gridspec_kw={'height_ratios':[1,1.4]},layout='constrained')
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=axs[0],label='Surface damage a',shrink=.7)
    if len(frames)>90:
        frames=frames[::2]
        if frames[-1]['state']['clock_s']!=final['state']['clock_s']:frames.append(final)
    def frame(i):
        p=frames[i];now=p['state']['clock_s'];axs[0].clear();axs[1].clear();curve(axs[0],p)
        axs[1].plot(t,[r['full_front_arc_m']-3 for r in history],'r--',label='Developed a=1')
        axs[1].plot(t,[r['process_front_arc_m']-3 for r in history],label='Process a=.05');axs[1].axvline(now,c='black');axs[1].axvline(.455,c='black',ls=':')
        axs[1].axhspan(1.125,1.6875,color='#cc79a7',alpha=.12,label='Late-turn material');axs[1].set(xlabel='Time (s)',ylabel='Front beyond seed (m)');axs[1].grid(alpha=.2);axs[1].legend(fontsize=8)
        fig.suptitle(f't={now:.6f} s; '+('loading' if now<=.455 else 'displacement hold')+'; actual accepted damage/slip rate and fronts')
    FuncAnimation(fig,frame,frames=len(frames),interval=150).save(OUT/'propagation.gif',writer=PillowWriter(fps=6));plt.close(fig)
    save(OUT/'animation.json',dict(frames=len(frames),times_s=[p['state']['clock_s'] for p in frames],histories_interpolated=False))


def evidence_tables():
    """Small material, energy-window and cost inventories; no acceptance tuning."""
    values=journal('main');first,last=values[0],values[-1];history=[p['diagnostics'] for p in values if p['diagnostics']]
    windows=[]
    for label,start,end in [('loading',.440,.455),('hold',.455,.460),('complete',.440,.460)]:
        rows=[d for d in history if start<d['time_s']<=end];total=lambda key:sum(d[key] for d in rows)
        absolute=sum(abs(d['physical_remainder_J_m']) for d in rows);sliding=total('sliding_loss_J_m');fracture=total('fracture_increment_J_m')
        windows.append(dict(label=label,start_s=start,end_s=end,steps=len(rows),
            **{key:total(key) for key in ['endpoint_work_J_m','trapezoid_work_J_m','bulk_storage_change_J_m','cohesive_storage_change_J_m',
                'fracture_increment_J_m','friction_J_m','radiation_J_m','sliding_loss_J_m','maxwell_loss_J_m',
                'BE_bulk_numerical_J_m','surface_endpoint_remainder_J_m','physical_remainder_J_m']},
            absolute_physical_remainder_J_m=absolute,absolute_vs_sliding=absolute/sliding if sliding else None,
            absolute_vs_fracture=absolute/fracture if fracture else None,maximum_absolute_step_remainder_J_m=max(abs(d['physical_remainder_J_m']) for d in rows)))
    table(OUT/'energy-windows.csv',windows)
    events=[]
    for cells in range(1,1+last['diagnostics']['late_mature_cells']):
        i=next(i for i,p in enumerate(values) if p['diagnostics'].get('late_mature_cells',0)>=cells)
        p=values[i];d=p['diagnostics'];events.append(dict(event='late-turn-full-cell-'+str(cells),
            lower_time_s=values[i-1]['state']['clock_s'],upper_time_s=p['state']['clock_s'],
            front_arc_beyond_seed_m=d['full_front_arc_m']-3,front_x_m=d['full_front_x_m'],front_y_m=d['full_front_y_m'],
            actual_front_tangent_deg=d['full_front_tangent_deg'],imposed_0=p['state']['imposed'][0],imposed_1=p['state']['imposed'][1]))
    table(OUT/'mature-events.csv',events)
    write_rows(OUT/'final-geometry.csv',[dict(index=i,formed=i<last['state']['formed_arcs'],**arc,**last['state']['arc_metadata'][i]) for i,arc in enumerate(last['state']['geometry'])])
    # Charge each completed export once, separate from the total coordinator
    # trajectory time. Some cached exports predate that trajectory; list those
    # explicitly instead of subtracting overlapping/previous work silently.
    run=json.loads((WORK/'main/run-input.json').read_text());kernel_rows=[];peak=0
    for p in sorted((WORK/'kernels').glob('*/complete.json')):
        info=json.loads(p.read_text());cost=list(csv.DictReader((p.parent/'parallel-cost.csv').open()))
        rss=max(float(c['peak_rss_KiB']) for c in cost);peak=max(peak,rss)
        kernel_rows.append(dict(kernel=p.parent.name,main_binary=info['specification']['executable_sha256']==run['binary_sha256'],
            ranks=info['specification']['ranks'],wall_seconds=info['wall_seconds'],
            **{key:max(float(c[key]) for c in cost) for key in ['assembly_s','factor_solve_broadcast_s','surface_s','fields_s','probes_output_s']},
            maximum_rank_RSS_KiB=rss))
    table(OUT/'native-export-costs.csv',kernel_rows)
    i=owned_functionals(first);f=owned_functionals(last)
    numerical=json.loads((OUT/'focused-uncertainty.json').read_text());pressure=json.loads((OUT/'pressure-impact.json').read_text())
    results=dict(main=json.loads((OUT/'main-summary.json').read_text()),initial_functionals=i,final_functionals=f,
        energy_windows=windows,maturity_events=events,focused_comparisons=numerical,
        normal_operator_sensitivity=pressure,main_run_input=run,peak_export_rank_RSS_KiB=peak,
        main_binary_completed_exports=sum(r['main_binary'] for r in kernel_rows),
        main_binary_export_wall_seconds=sum(r['wall_seconds'] for r in kernel_rows if r['main_binary']))
    save(OUT/'results.json',results)
    save(OUT/'parameters.json',dict(G_Pa=1e6,eta_Pa_s=1e8,Gc_J_m2=GC,peak_Pa=3000,p0_Pa=10000,
        C0_m_Pa=C0,complete_decohesion_jump_m=2*GC/3000,mu0=.1,RSF_a=.005,RSF_b=.01,
        V0_m_s=V0,Dc_m=DC,radiation_Pa_s_m=RAD,physical_R_m=R,numerical_lifting_half_width_m=.75,
        bulk_cells=[64,64],bulk_degree=2,pressure='discontinuous P1, mean pprime=0',bulk_Gauss_order=6,
        surface_Gauss_order=3,selection_Gauss_order=8,main_future_spacing_m=.09375,reference_future_spacing_m=.046875,
        dt_s=.000125,loading_stops_s=.455,end_s=.460,domain_m=[0,12,-6,6],
        bulk='quasi-static, homogeneous plane-strain Maxwell, fixed background mesh',
        friction='full current variational normal pressure; regularized RSF and existing radiation',
        theta='normalized age; age_s=theta*Dc/V0',phase_phi='provisional geometry only; no duplicate bulk fracture inventory'))
    save(OUT/'initial-accepted.json',dict(state=first['state'],material_ledger=first['material_ledger']))
    save(OUT/'final-accepted.json',last)
    archived=OUT/'source-provenance';archived.mkdir(exist_ok=True)
    for name in ['cohesive_interface_path.cc','cohesive_interface_path_parallel.h']:
        shutil.copyfile(WORK/'parallel-v2-source'/name,archived/name)
    main_driver=WORK/'driver-versions'/(run['driver_sha256']+'.py')
    shutil.copyfile(main_driver,archived/'executed-main-driver.py')
    (archived/'README.md').write_text('Archived executed main-run source, not an additional maintained driver. Use the current cohesive_interface_propagation.py. Main binary/source hashes are in execution-provenance.json; the shape-cached binary preserves the grown operator/basis bytes.\n')
    dependencies=['cohesive_interface_candidate_admission.py','cohesive_interface_load_turn.py','cohesive_interface_path.py',
        'cohesive_interface_turn.py','cohesive_interface_admission.py','check_cohesive_interface_kink.py',
        'cohesive_interface_turn_solver.cc','cohesive_interface_evolution.h','cohesive_interface_front.inc',
        'cohesive_interface_admission_owner.inc','qualify_cohesive_interface_load_turn.py']
    save(OUT/'execution-provenance.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        entry_status_sha256=sha(WORK/'entry-status.txt'),entry_source_sha256=sha(WORK/'entry-source.zip'),
        main_run=run,current_native_binary_sha256=sha(EXEC),current_driver_sha256=sha(Path(__file__)),
        executed_driver_versions={p.name:sha(p) for p in sorted((WORK/'driver-versions').glob('*.py'))},
        retained_dependencies={name:sha(Path(__file__).with_name(name)) for name in dependencies},
        raw_checkpoints_and_operator_caches=str(WORK),physical_histories_not_projected_between_branches=True))
    failure=WORK/'pressure-impact-main/base-predictor-failure.json'
    if failure.exists():shutil.copyfile(failure,OUT/'unpublished-probe-failure.json')
    report=ROOT/'doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_PROPAGATION_REPORT.md'
    if report.exists():
        text=report.read_text().replace('(COHESIVE_INTERFACE_PROPAGATION_SELECTION.md)',
            '('+os.path.relpath(report.with_name('COHESIVE_INTERFACE_PROPAGATION_SELECTION.md'),OUT)+')')
        text=text.replace('(../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation/INDEX.md)','(INDEX.md)')
        (OUT/'REPORT.md').write_text(text)
    index='''# Integrated propagation evidence

See [the concise scientific report](REPORT.md). Reproduce with the maintained
`benchmarks/reconstructed_fault/particle_free/cohesive_interface_propagation.py`.
Native build targets are `cohesive_interface_path` and
`cohesive_interface_turn_solver` in the existing benchmark CMake project.
`demo` reuses verified caches, runs qualification before evolution, resumes
atomic accepted checkpoints, then performs checks/reporting/verification.
`run --resume --ranks 2` resumes only an unfinished main trajectory;
`report` regenerates all figures and tables without advancing any history.
Keep the original formed-turn source checkpoints and their native caches.

| Evidence | Meaning |
|---|---|
| [specimen.png](specimen.png) | Physical walls, loading history and actual fault |
| [propagation.png](propagation.png), [propagation.gif](propagation.gif) | Actual damage and signed slip-rate vectors, connected fronts, temporary tail |
| [fronts.png](fronts.png), [mature-events.csv](mature-events.csv) | Developed front entry and physical activation brackets |
| [mature-material.png](mature-material.png), [main-tracked-material.csv](main-tracked-material.csv) | Same material identities through decohesion, RSF sliding and state evolution |
| [energy.png](energy.png), [energy-windows.csv](energy-windows.csv) | Endpoint/physical work, dissipation, storage and separate numerical remainders |
| [uncertainty.png](uncertainty.png), [focused-uncertainty.csv](focused-uncertainty.csv) | Focused timestep and future-material spacing controls |
| [pressure-impact.json](pressure-impact.json), [pressure-impact-profiles.csv](pressure-impact-profiles.csv) | Fitted common-field normal operator and full-current-pressure response sensitivity |
| [parallel-qualification.json](parallel-qualification.json), [parallel-cost.csv](parallel-cost.csv) | Native 1/2/4-rank correctness and measured local scaling |
| [native-export-costs.csv](native-export-costs.csv) | Setup/application phases and per-rank peak RSS for completed kernels |
| [route-check.json](route-check.json), [mature-lifecycle.json](mature-lifecycle.json), [publication-accounting.csv](publication-accounting.csv) | Actual transaction/retry/restart and once-only advancement |
| [results.json](results.json), [verification.json](verification.json), [manifest.json](manifest.json) | Compact results, separate scientific/engineering gates and provenance |

`main-history.csv` has every accepted positive step; `*-profiles.csv` samples
every fourth accepted step plus events. `*-tracked-material.csv` retains every
accepted step for the first 0.1875 m of late-turn material. `*-material-ledger.csv`
records persistent IDs, admission, growth/full brackets and final histories.
All coordinates are physical metres, time seconds, pressure/stress Pa, signed
jump metres, velocity m/s; energies J/m (unit out-of-plane thickness).
`theta` is normalized age; physical age is theta*Dc/V0. Cohort averages use
actual surface Gauss weights (metres), not point counts. `intact_tied` marks
zero-damage constraints: their stored zero cohesive/friction return is not a
pointwise intact traction. Fronts are connected Gauss-record thresholds
a=.05, .5, exactly 1; lower/upper spatial and temporal brackets are retained.
No plotting interpolation creates a constitutive state. Cell-centred bulk
stress and raw sided tractions are not thereby qualified as point stresses.

Complete solver caches, all accepted/checkpoint journals, failed exports,
executed source versions and logs remain separately in
`build-particle-free/cohesive-interface-propagation`. No production checkpoints
are converted. `parallel-v1/` and `parallel-v2/` retain superseded performance
measurements; the final parallel result must not be represented as MPI speedup.
Compact initial/final accepted states are also retained here. Their kernel
paths refer to the preserved native caches; those caches are required to
restart. `execution-provenance.json` identifies all executed source versions.
'''
    (OUT/'INDEX.md').write_text(index)


def verify_results():
    from PIL import Image
    from zipfile import ZipFile
    protected=json.loads((OUT/'entry-protection.json').read_text())
    changed={'benchmarks/reconstructed_fault/particle_free/cohesive_interface_path.cc',
        'doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md'}
    for name,digest in protected.items():
        if name not in changed:assert sha(ROOT/name)==digest,name
    # Status/audit are append-only evidence: retain the full entry text after
    # the current result, including every historical failed target.
    with ZipFile(WORK/'entry-source.zip') as archive:
        for name in ['doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md']:
            assert (ROOT/name).read_bytes().endswith(archive.read(name)),name
    accounts=[]
    for label in ['main','time-hold','time-entry','time-mature','spacing']:
        if not (WORK/label/'final.json').exists():continue
        vals=journal(label);first,last=vals[0],vals[-1];source=first['state'];count=0;previous=source
        for p in vals[1:]:
            s=p['state'];d=p['diagnostics'];assert s['physical_publications']==previous['physical_publications']+1
            q0=len(previous['record_ids']);assert s['record_ids'][:q0]==previous['record_ids']
            assert s['geometry'][:previous['formed_arcs']]==previous['geometry'][:previous['formed_arcs']]
            assert np.all(np.array(s['histories']['damage'][:q0])>=np.array(previous['histories']['damage'][:q0]))
            assert d['weak_error_Pa']<=1e-8 and d['minimum_normal_Pa']>0 and d['raw_minimum_normal_Pa']>0
            assert abs(d['ledger_error_J_m'])<1e-10*max(1,abs(d['endpoint_work_J_m']))
            assert d['metadata_commit_fracture_J_m']==0 and d['metadata_commit_work_J_m']==0
            assert d['transfer_stress_Pa']<1e-7
            count+=1;previous=s
        assert count==last['state']['physical_publications']-source['physical_publications']
        k,s,ledger=open_checkpoint(WORK/label/'final.json');owner.zero_tail(k,s);k.close()
        summary=json.loads((OUT/(label+'-summary.json')).read_text());r=summary['recurrence_checks']
        assert r['maximum_Maxwell_error_m']<1e-13 and r['maximum_affine_Maxwell_error']<1e-13 and r['maximum_age_error']<1e-12
        assert r['maximum_trace_error_m']<1e-13 and r['maximum_RSF_force_error_Pa']/3000<1e-12
        accounts.append(dict(label=label,steps=count,physical_publications=count,registry_transactions=last['state']['publications']-source['publications'],
            full_front_beyond_seed_m=summary['full_front_beyond_seed_m'],physical_vs_sliding=summary['physical_vs_sliding'],physical_vs_fracture=summary['physical_vs_fracture']))
    table(OUT/'publication-accounting.csv',accounts)
    main=json.loads((WORK/'main/final.json').read_text());s=main['state'];ledger=main['material_ledger']
    mature=[key for key in s['record_ids'] if ledger[key]['cohort']=='late_turn' and s['histories']['damage'][s['record_ids'].index(key)]==1]
    assert len(mature)>=6
    assert all(main['points']['cohesive_q_Pa'][s['record_ids'].index(key)]==0 for key in mature)
    uncertainty=json.loads((OUT/'focused-uncertainty.json').read_text());pressure=json.loads((OUT/'pressure-impact.json').read_text())
    assert pressure['geometry_spacing_m']==.09375
    energy=list(csv.DictReader((OUT/'energy-windows.csv').open()))
    save(OUT/'verification.json',dict(engineering_checks_passed=True,protected_files_unchanged=len(protected)-len(changed),
        late_turn_mature_records=len(mature),cohesive_resistance_exactly_zero_on_mature_records=True,
        parallel_qualification=json.loads((OUT/'parallel-qualification.json').read_text()),
        numerical_response_gates_passed=uncertainty['passed'],normal_uncertainty_material_response_supported=pressure['material_response_supported'],
        physical_energy_screens_passed=all(float(row['absolute_vs_sliding'])<.01 and float(row['absolute_vs_fracture'])<.01 for row in energy),
        selection_angle_screen_passed=abs(pressure['selection_angle_difference_deg'])<1,
        original_point_traction_accuracy_failure_open=True,production_BP3_defaults_unchanged=True,
        original_status_audit_retained=True,
        scope='Homogeneous fixed-mesh finite specimen; parallel native operators, root-owned experimental JSON history; not production Simulator integration'))
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    animation=json.loads((OUT/'animation.json').read_text())
    with Image.open(OUT/'propagation.gif') as im:assert im.n_frames==animation['frames']
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
    files += [Path(__file__),Path(__file__).with_name('cohesive_interface_path.cc'),Path(__file__).with_name('cohesive_interface_path_parallel.h')]
    files += [ROOT/'doc/reconstructed_fault/particle_free'/name for name in ['COHESIVE_INTERFACE_PROPAGATION_REPORT.md','COHESIVE_INTERFACE_PROPAGATION_SELECTION.md','STATUS.md','SOURCE_AUDIT.md']]
    save(OUT/'manifest.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        binary_sha256=sha(EXEC),point_solver_sha256=sha(WORK/'build/libcohesive_interface_turn_solver.so'),
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))


def demo():
    current=json.loads((OUT/'parallel-qualification.json').read_text()) if (OUT/'parallel-qualification.json').exists() else {}
    if not current.get('passed') or current.get('binary_sha256')!=sha(EXEC):qualify()
    ranks=json.loads((WORK/'main/run-input.json').read_text())['ranks'] if (WORK/'main/run-input.json').exists() else selected_ranks()
    route_check(ranks)
    if not (WORK/'main/final.json').exists():evolve('main',ranks=ranks,extend_hold=True,resume=(WORK/'main/accepted.json').exists())
    focused_checks()
    plot_results()
    evidence_tables()
    verify_results()
    print((OUT/'verification.json').read_text(),flush=True)


def qualify():
    """Complete before the long trajectory; never use speed as correctness."""
    cache=NativeCache();cache.attach();s=json.loads(SOURCE.read_text());arcs=s['geometry']
    identity=['stiffness.csv','Q.csv','normal-work.csv','normal-minus.csv','normal-plus.csv','shear.csv','jump-x.csv','jump-y.csv',
        'columns.csv','nodes.csv','surface.csv','cells.csv','normal-measure.csv','field-center.bin.gz','field-mean.bin.gz']
    costs=[];reference=WORK/'qualification/serial-legacy';export_native(reference,arcs,legacy=True,ranks=1)
    default=QUAL/'serial-default';export_native(default,arcs,ranks=1,serial_default=True)
    default_bits={name:sha(default/name)==sha(reference/name) for name in identity+['probe.csv','formed-probes.csv','formed-probe-mass.csv','energy-check.csv']}
    assert all(default_bits.values()),default_bits
    save(OUT/'default-identity.json',default_bits)
    for ranks in [1,2,4]:
        folder=QUAL/f'ranks-{ranks}';export_native(folder,arcs,ranks=ranks)
        bits={name:sha(folder/name)==sha(reference/name) for name in identity}
        assert all(bits.values()),bits
        a=np.genfromtxt(reference/'energy-check.csv',delimiter=',',names=True);b=np.genfromtxt(folder/'energy-check.csv',delimiter=',',names=True)
        energy=float(max(abs(a['physical_J_m']-b['physical_J_m'])/np.maximum(1,abs(a['physical_J_m']))));assert energy<1e-12
        selector={}
        for name in ['probe.csv','formed-probes.csv']:
            aa=np.loadtxt(reference/name,delimiter=',');bb=np.loadtxt(folder/name,delimiter=',')
            selector[name]=float(np.linalg.norm(aa-bb)/max(1,np.linalg.norm(aa)));assert selector[name]<1e-12
        cost=json.loads((folder/'complete.json').read_text());rr=list(csv.DictReader((folder/'parallel-cost.csv').open()))
        costs.append(dict(ranks=ranks,wall_seconds=cost['wall_seconds'],native_seconds=cost['meta']['seconds'],
            max_rank_rss_KiB=max(float(v['peak_rss_KiB']) for v in rr),energy_diagnostic_difference=energy,
            bitwise_operator_basis_tables=len(identity),selector_roundoff_relative=max(selector.values())))
        save(OUT/f'parallel-identity-{ranks}.json',bits)
    endpoints=[];step_cost=[]
    for ranks in [1,2,4]:
        c=NativeCache(ranks,namespace=f'{QUAL.name}/short-{ranks}');c.attach();k,state=owner.open_state(SOURCE);initial=encode(state);tic=time.perf_counter()
        for target in [.440125,.44025]:
            p=owner.attempt(k,state,target,cells=2);accepted=owner.publish(state,p)
            try:owner.publish(accepted,p)
            except AssertionError:pass
            else:raise AssertionError('Duplicate accepted publication')
            if p['kernel'] is not k:k.close()
            k=p['kernel'];state=accepted
        assert state['physical_publications']==s['physical_publications']+2
        path=QUAL/f'short-{ranks}/accepted.json';save(path,state);save(path.parent/'initial.json',json.loads(initial))
        endpoints.append(encode(physical(state)));step_cost.append(dict(ranks=ranks,steps=2,wall_seconds=time.perf_counter()-tic))
        p=owner.attempt(k,state,.440375,cells=2);a=owner.publish(state,p)
        rk,rs=owner.open_state(path);rp=owner.attempt(rk,rs,.440375,cells=2);b=owner.publish(rs,rp)
        assert encode(physical(a))==encode(physical(b)),'Loaded same-rank JSON restart mismatch'
        for v in {id(v):v for v in [k,p['kernel'],rk,rp['kernel']]}.values():v.close()
    assert endpoints[0]==endpoints[1]==endpoints[2],'Rank-dependent physical advancement'
    # Actual nonroot failure while preparing the first trial from immutable s.
    c=NativeCache(4,namespace=f'{QUAL.name}/failure',fail_rank=2);c.attach();k,state=owner.open_state(SOURCE);before=encode(state)
    save(QUAL/'failure-accepted.json',state)
    try:owner.attempt(k,state,.440125,cells=2)
    except subprocess.CalledProcessError:rejected=True
    else:raise AssertionError('Injected nonroot failure did not reject')
    assert encode(state)==before;c.fail_rank=None
    retry=owner.attempt(k,state,.440125,cells=2);accepted=owner.publish(state,retry)
    control=NativeCache(4,namespace=f'{QUAL.name}/short-4');control.attach();rk,rs=owner.open_state(SOURCE)
    p=owner.attempt(rk,rs,.440125,cells=2);expected=owner.publish(rs,p)
    assert encode(physical(accepted))==encode(physical(expected))
    for v in {id(v):v for v in [k,retry['kernel'],rk,p['kernel']]}.values():v.close()
    if (WORK/'main/final.json').exists():
        grown=json.loads((WORK/'main/final.json').read_text())['state'];old=Path(grown['kernel_path']);new=QUAL/'grown-identity'
        export_native(new,grown['geometry'],ranks=2)
        bits={name:sha(old/name)==sha(new/name) for name in identity+['probe.csv','formed-probes.csv'] if (old/name).exists()}
        assert all(bits.values()),bits;save(OUT/'grown-operator-identity.json',bits)
    table(OUT/'parallel-cost.csv',costs);table(OUT/'parallel-short-steps.csv',step_cost)
    save(OUT/'parallel-qualification.json',dict(passed=True,binary_sha256=sha(EXEC),rank_counts=[1,2,4],
        operator_and_basis_bitwise=True,short_advanced_histories_bitwise=True,nonroot_failure_rejected=rejected,
        accepted_state_unchanged=True,retry_matches_control=True,same_rank_restart=True,single_publication=True,
        speedup_1_to_4=costs[0]['wall_seconds']/costs[-1]['wall_seconds'],
        scope='Parallel native response columns; replicated bulk matrix/mesh, coordinated reduced solve and publication'))
    print((OUT/'parallel-qualification.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['qualify','route-check','run','checks','report','verify','demo'])
    p.add_argument('--label',default='main');p.add_argument('--checkpoint',type=Path,default=SOURCE)
    p.add_argument('--ranks',type=int,default=0);p.add_argument('--dt',type=float,default=.000125)
    p.add_argument('--spacing',type=float,default=.09375);p.add_argument('--end',type=float,default=.460)
    p.add_argument('--extend-hold',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    WORK.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if a.command=='qualify':qualify()
    elif a.command=='route-check':route_check(a.ranks or selected_ranks())
    elif a.command=='checks':focused_checks()
    elif a.command=='report':
        plot_results();evidence_tables()
    elif a.command=='verify':verify_results()
    elif a.command=='demo':demo()
    else:evolve(a.label,a.checkpoint,a.end,a.dt,a.ranks or selected_ranks(),a.spacing,a.extend_hold,a.resume)


if __name__=='__main__':main()
