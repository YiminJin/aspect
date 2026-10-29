#!/usr/bin/env python3
"""Selected-branch publication and actual context-rebuild/restart controls."""
import copy
import argparse
import json
import resource
import os
import subprocess
import time
import numpy as np

from cohesive_interface_turn import WORK, OUT, ROOT, main_kernel, run, save, encode, Admission, sha
from analyze_cohesive_interface_turn import state


def fails(action):
    try:action()
    except AssertionError:return True
    raise AssertionError('Invalid ownership operation was accepted')


def main():
    start=time.perf_counter()
    decision=json.loads((OUT/'verification.json').read_text())['primary_selection']
    assert decision['distinguished_in_sampled_primary_family']
    angle=decision['angle_deg'];k=main_kernel(angle,128,gamma_end=.0021)
    old=k.initial();before=encode(old)
    provisional,_=k.birth(old)
    # Geometry and records are transaction-owned until commit: discarding a
    # valid provisional birth changes neither accepted geometry nor history.
    assert encode(old)==before
    retry,_=k.birth(old);assert encode(retry)==encode(provisional)
    assert retry['histories']==old['histories'] and retry['clock_s']==old['clock_s']
    duplicate_geometry_rejected=fails(lambda:k.birth(retry))
    # A newly reconstructed native context must see the same selected physical
    # state, not an accepted trajectory fed back as prescribed evolution.
    incoming=state('fine-p'+str(angle),.44025);saved=encode(incoming)
    payload=k.trial(incoming,.4405)
    wrong=copy.deepcopy(incoming);wrong['geometry']['append']['angle_deg']+=1
    wrong_geometry_rejected=fails(lambda:k.trial(wrong,.4405))
    publisher=Admission(k);bad=copy.deepcopy(payload);bad['state']['histories']['damage'][0]=-1
    rejected=fails(lambda:publisher.publish(incoming,bad));assert encode(incoming)==saved
    checkpoint=WORK/'runs'/'selected-context-checkpoint.json';save(checkpoint,incoming)
    k.close();restored=json.loads(checkpoint.read_text());k=main_kernel(angle,128,gamma_end=.0021)
    restarted=k.trial(restored,.4405)
    assert encode(restarted['state'])==encode(payload['state'])
    accepted=Admission(k).publish(restored,restarted)
    assert accepted['physical_publications']==incoming['physical_publications']+1
    duplicate_physical_rejected=fails(lambda:Admission(k).publish(accepted,restarted))
    k.close()
    # One real native selected evolution from the unchanged incoming archive.
    # No candidate's provisional histories are copied into this computation.
    k=main_kernel(angle,128,gamma_end=.0021)
    result=run(k,'accepted-selection',dt=.00025,hold_s=0.)
    final=json.loads((WORK/'runs/accepted-selection/final.json').read_text())
    reference=json.loads((WORK/('runs/fine-p'+str(angle))/'final.json').read_text())
    assert encode(final)==encode(reference)
    assert final['publications']==1 and final['generation']==1
    assert final['physical_publications']==result['steps'];k.close()
    save(OUT/'selected-lifecycle.json',dict(selected_angle_deg=angle,geometry_reject_retry_bitwise=True,
        accepted_old_archive_unchanged=True,old_histories_preserved=True,duplicate_geometry_rejected=duplicate_geometry_rejected,
        mismatched_geometry_rejected=wrong_geometry_rejected,forced_history_rejection=rejected,
        native_context_rebuilt=True,actual_JSON_restart_retry_bitwise=True,
        duplicate_physical_rejected=duplicate_physical_rejected,selected_native_evolution_bitwise_to_preview=True,
        geometry_publications=final['publications'],physical_publications=final['physical_publications'],
        seconds=time.perf_counter()-start,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='Isolated benchmark owner and actual native mixed/return contexts; not production Simulator/Manager checkpoint integration.'))
    print((OUT/'selected-lifecycle.json').read_text())


def regression():
    results={}
    for label,reference,args in [
        ('legacy-default',ROOT/'build-particle-free/cohesive-interface-kink/native-16',['16','8','3']),
        ('rebuilt-smooth',WORK/'kernel-p40-n32-w250',['32','16','6','smooth','40','.25',format(np.sqrt(3)/2,'.17g'),'1'])]:
        folder=WORK/label
        command=[str(WORK/'build/cohesive_interface_kink'),args[0],args[1],str(folder)]+args[2:]
        if not (folder/'native.json').exists():
            folder.mkdir(exist_ok=False)
            with (WORK/(label+'.log')).open('w') as f:
                subprocess.run(command,check=True,timeout=120,stdout=f,stderr=subprocess.STDOUT,
                               env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        files=[p for p in reference.iterdir() if p.name!='native.json' and (folder/p.name).is_file()]
        matches={p.name:sha(p)==sha(folder/p.name) for p in files}
        metrics={}
        for key in ['stiffness','normal-minus','normal-plus','energy-gauss6','shear','response']:
            a=np.loadtxt(reference/(key+'.csv'),delimiter=',');b=np.loadtxt(folder/(key+'.csv'),delimiter=',')
            metrics[key]=dict(absolute=float(np.max(abs(a-b))),relative_norm=float(np.linalg.norm(a-b)/np.linalg.norm(a)))
            # Dimension-dependent floating arithmetic allowance, separate from
            # the unchanged physical stress and tangent acceptance criteria.
            assert metrics[key]['relative_norm']<64*a.shape[1]*np.finfo(float).eps
        if label=='legacy-default':
            a=np.genfromtxt(reference/'field-response.csv',delimiter=',',names=True)
            b=np.genfromtxt(folder/'field-response.csv',delimiter=',',names=True)
            n=np.loadtxt(folder/'stiffness.csv',delimiter=',').shape[0]
            c=np.r_[.001,1e-4*np.sin(np.arange(n-1)+1)]
            diffs=np.column_stack([(a[key]-b[key]).reshape(-1,n)@c for key in ['tau_xx_Pa','tau_yy_Pa','tau_zz_Pa','tau_xy_Pa','pprime_Pa']])
            tensor_error=np.sqrt(np.sum(diffs[:,:3]**2,axis=1)+2*diffs[:,3]**2)
            physical_error=float(max(np.max(tensor_error),np.max(abs(diffs[:,4]))))
            assert physical_error<1e-7
            assert matches['point-reference.csv'] and matches['surface.csv'] and matches['nodes.csv'] and matches['Q.csv']
        else:
            assert all(matches.values()),matches
            physical_error=0.
        results[label]=dict(command=command,bitwise_files=matches,numerical_operator_differences=metrics,
            full_tensor_and_pressure_error_Pa=physical_error,physical_tolerance_Pa=1e-7,
            initial_bitwise_expectation_failed=label=='legacy-default',
            native=json.loads((folder/'native.json').read_text()))
    save(OUT/'default-regression.json',results);print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--regression',action='store_true');args=parser.parse_args()
    regression() if args.regression else main()
