#!/usr/bin/env python3
"""Endpoint lifecycle and independent common-field fitted diagnostics."""
import argparse
import ctypes as ct
import json
import os
import subprocess
import numpy as np

import cohesive_interface_candidate_admission as owner
from cohesive_interface_formed_turn import WORK,OUT,configure
from cohesive_interface_path import save,table,encode,axes,graph
from qualify_cohesive_interface_load_turn import finite_difference


def lifecycle(mode):
    configure(mode);states=json.loads((WORK/mode/'snapshots.json').read_text())
    s=next(v for v in states if abs(v['clock_s']-.4395)<1e-12)
    path=WORK/(mode+'-lifecycle-input.json');save(path,s)
    k,original=owner.open_state(path);base=encode(original);target=.439625
    p=owner.attempt(k,original,target,cells=2 if mode=='coarse' else 4)
    fd=finite_difference(p['kernel'],p['input'],p['native_payload']);assert fd<1e-6,fd
    accepted=owner.publish(original,p);assert encode(original)==base
    # Actual native contact-loss trial, with no publication or altered history.
    n=len(k.nodes);nq=len(k.s);D=ct.POINTER(ct.c_double)
    old=np.ascontiguousarray(np.array([s['histories'][key] for key in owner.HISTORY]).T)
    c0=np.ascontiguousarray(k.coefficients(s));da=np.ascontiguousarray([.05,0.])
    c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(16000)
    status=k.lib.cohesive_turn_solve_multi(k.context,c0.ctypes.data_as(D),old.ctypes.data_as(D),.000125,da.ctypes.data_as(D),None,None,
        *[a.ctypes.data_as(D) for a in [c,inc,r,J]],ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert status and 'contact' in error.value.decode().lower(),error.value.decode()
    assert encode(original)==base
    retry=owner.attempt(k,original,target,cells=2 if mode=='coarse' else 4);retried=owner.publish(original,retry)
    assert encode(retried)==encode(accepted)
    restored,rs=owner.open_state(path);replay=owner.attempt(restored,rs,target,cells=2 if mode=='coarse' else 4)
    restarted=owner.publish(rs,replay);assert encode(restarted)==encode(accepted)
    try:owner.publish(accepted,p)
    except AssertionError:duplicate=True
    else:raise AssertionError('Duplicate publication accepted')
    save(OUT/(mode+'-lifecycle.json'),dict(input_s=s['clock_s'],target_s=target,coupled_FD_relative=fd,
        native_contact_loss_rejected=True,diagnosis=error.value.decode(),incoming_unchanged=True,
        deterministic_retry_bitwise=True,rebuilt_context_JSON_restart_bitwise=True,
        one_physical_publication=accepted['physical_publications']==s['physical_publications']+1,duplicate_rejected=duplicate,
        scope='Experimental serial owner/native context, not production Simulator restart'))
    for context in {id(v):v for v in [k,p['kernel'],retry['kernel'],restored,replay['kernel']]}.values():context.close()


def export_frozen(s,n,mesh):
    folder=WORK/f'frozen-{mesh}{n}';folder.mkdir(exist_ok=False)
    save(folder/'input.json',dict(arcs=s['geometry'],n=n,degree=2,order=6,mesh=mesh,diagnostic_only=True))
    table(folder/'geometry.csv',s['geometry']);c=np.r_[s['affine_memory'],s['memory']]
    np.savetxt(folder/'physical-coefficients.txt',c)
    command=[str(WORK/'build/cohesive_interface_path'),str(n),'2',str(folder/'geometry.csv'),str(folder),'6','compress',
        '.75','.75',mesh,str(folder/'physical-coefficients.txt'),'two-affine','candidate-probes','8']
    save(folder/'command.json',command)
    with (WORK/f'frozen-{mesh}{n}.log').open('w') as f:
        subprocess.run(command,check=True,timeout=300,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    k=owner.Kernel(folder,s['geometry']);k.anchor(s['formed_arcs']);return k


def frozen():
    configure('coarse');old,s=owner.open_state(WORK/'coarse/final.json');source=encode(s)
    # All surface quadrature and jump/Maxwell memories are identical. Only the
    # independently equilibrated native bulk approximation changes.
    kernels=[export_frozen(s,64,'cartesian'),export_frozen(s,64,'fitted'),export_frozen(s,96,'fitted')]
    assert np.array_equal(old.K,kernels[0].K)
    last=s['geometry'][s['formed_arcs']-1]['end']
    tests=[('old',4.5,.5),('bend',6.4,.25),('zone',last-.25,.2),('ahead',last+.1,.15)]
    rows=[];loads=[];profiles=[]
    for k in kernels:
        assert np.array_equal(k.s,old.s) and np.array_equal(k.Q,old.Q)
        c=k.coefficients(s);tensor=k.probe@c;angle,decision=axes(tensor,np.arctan(s['geometry'][s['formed_arcs']-1]['b']))
        rows.append(dict(mesh='fitted' if k.meta['fitted_mesh'] else 'cartesian',n=k.meta['nx'],
            seconds=k.meta['seconds'],bulk_J_m=float(.5*c@k.K@c),preferred_deg=float(np.rad2deg(angle)),
            affine_J_m=float(.5*c[:2]@k.K[:2,:2]@c[:2]),jump_self_J_m=float(.5*c[2:]@k.K[2:,2:]@c[2:]),
            affine_jump_cross_J_m=float(c[:2]@k.K[:2,2:]@c[2:]),
            **dict(zip(['xx_Pa','yy_Pa','zz_Pa','xy_Pa','pprime_Pa'],tensor))))
        raw=np.loadtxt(k.path/'shear.csv',delimiter=',')@c
        for name,centre,width in tests:
            v=np.exp(-.5*((k.nodes['x_m']-centre)/width)**2);test=k.Q@v;mass=k.w@test
            loads.append(dict(mesh=rows[-1]['mesh'],n=k.meta['nx'],region=name,x_centre_m=centre,sigma_m=width,test_mass_m=float(mass),
                weak_shear_Pa=float(-(k.K@c)[2:]@v/mass),normal_perturbation_Pa=float(k.w@(test*(k.N@c))/mass),
                raw_shear_Pa=float(k.w@(test*raw)/mass),raw_normal_jump_RMS_Pa=float(np.sqrt(k.w@(test*((k.raw_Np-k.raw_Nm)@c)**2)/mass))))
        normal=k.N@c
        for q in range(len(k.s)):
            profiles.append(dict(mesh=rows[-1]['mesh'],n=k.meta['nx'],time_s=s['clock_s'],x_m=float(k.s['x_m'][q]),
                y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),damage=s['histories']['damage'][q],jump_m=s['histories']['jump'][q],
                variational_normal_perturbation_Pa=float(normal[q]),raw_minus_Pa=float(10000+k.raw_Nm[q]@c),
                raw_plus_Pa=float(10000+k.raw_Np[q]@c),raw_shear_Pa=float(raw[q])))
        for d in np.genfromtxt(k.path/'physical-diagnostic.csv',delimiter=',',names=True):
            # Native independent physical quadrature, including pressure/divergence.
            save(OUT/f'physical-{rows[-1]["mesh"]}{k.meta["nx"]}-region{int(d["region"])}.json',
                {name:float(d[name]) for name in d.dtype.names})
    reference=rows[-1];metric=np.array([1,1,1,2]);keys=['xx_Pa','yy_Pa','zz_Pa','xy_Pa'];tref=np.array([reference[z] for z in keys])
    for row in rows:
        t=np.array([row[z] for z in keys]);row['full_tensor_relative']=float(np.sqrt(metric@((t-tref)**2)/(metric@(tref**2))))
        row['preferred_difference_deg']=row['preferred_deg']-reference['preferred_deg']
    errors=[]
    for row in loads:
        ref=next(v for v in loads if v['region']==row['region'] and v['mesh']=='fitted' and v['n']==96)
        for quantity in ['weak_shear_Pa','normal_perturbation_Pa','raw_shear_Pa','raw_normal_jump_RMS_Pa']:
            absolute=row[quantity]-ref[quantity]
            errors.append(dict(mesh=row['mesh'],n=row['n'],region=row['region'],quantity=quantity,reference_Pa=ref[quantity],
                difference_Pa=absolute,relative=absolute/abs(ref[quantity]) if ref[quantity]!=0 else 0.,reference_nonzero=ref[quantity]!=0))
    table(OUT/'frozen-tensors.csv',rows);table(OUT/'frozen-loads.csv',loads);table(OUT/'frozen-load-differences.csv',errors);table(OUT/'frozen-profiles.csv',profiles)
    assert encode(s)==source
    save(OUT/'frozen-input.json',dict(source=str(WORK/'coarse/final.json'),source_time_s=s['clock_s'],
        geometry=s['geometry'],coefficients=kernels[0].coefficients(s).tolist(),formed_arcs=s['formed_arcs'],
        diagnostic_only=True,accepted_history_transferred=False,p0_Pa=10000,pprime_reference='mean zero',
        sampler='radius .75 m at .75 m ahead of formed endpoint, weight (1-r²/R²)², native Gauss8',
        tests='Same physical x-Gaussian Q1 virtual jumps on identical surface nodes; work divided by integral of test'))
    for k in [old]+kernels:k.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['lifecycle','frozen']);p.add_argument('--mode',choices=['coarse','fine'],default='coarse');a=p.parse_args()
    lifecycle(a.mode) if a.task=='lifecycle' else frozen()
