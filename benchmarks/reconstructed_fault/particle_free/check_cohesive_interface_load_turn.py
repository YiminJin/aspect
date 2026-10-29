#!/usr/bin/env python3
"""Loaded two-history lifecycle and common-field Cartesian/fitted diagnostics."""
import argparse
import copy
import ctypes as ct
import json
import numpy as np
from cohesive_interface_load_turn import WORK, OUT, Kernel, kernel, HISTORY, Admission, encode, save, table, axes
from qualify_cohesive_interface_load_turn import finite_difference


def open_state(path):
    s=json.loads(path.read_text())
    return Kernel(WORK/s['kernel_folder'],s['geometry']),s


def lifecycle(label):
    folder=WORK/label
    states=json.loads((folder/'snapshots.json').read_text())
    s=min(states,key=lambda v:abs(v['clock_s']-.440))
    k=Kernel(WORK/s['kernel_folder'],s['geometry']);target=s['clock_s']+.00025
    before=encode(s);p=k.trial(s,target);retry=k.trial(json.loads(before),target)
    assert encode(s)==before and encode(p['state'])==encode(retry['state'])
    fd=finite_difference(k,s,p);assert fd<1e-6,fd
    # Actual native contact-loss trial. No output from it may be published.
    n=len(k.nodes);nq=len(k.s);D=ct.POINTER(ct.c_double)
    c0=np.ascontiguousarray(k.coefficients(s));old=np.ascontiguousarray(np.array([s['histories'][key] for key in HISTORY]).T)
    da=np.ascontiguousarray([.05,0.]);c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));ties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(16000)
    status=k.lib.cohesive_turn_solve_multi(k.context,c0.ctypes.data_as(D),old.ctypes.data_as(D),.00025,da.ctypes.data_as(D),None,None,
        *[v.ctypes.data_as(D) for v in [c,inc,r,J]],ties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert status!=0 and 'contact' in error.value.decode().lower(),error.value.decode()
    assert encode(s)==before
    q=k.trial(s,target);assert encode(q['state'])==encode(p['state'])
    k.close();k=Kernel(WORK/s['kernel_folder'],s['geometry'])
    restarted=k.trial(json.loads(before),target);assert encode(restarted['state'])==encode(p['state'])
    accepted=Admission(k).publish(s,restarted)
    assert accepted['physical_publications']==s['physical_publications']+1
    try:Admission(k).publish(accepted,restarted)
    except AssertionError:duplicate=True
    else:raise AssertionError('duplicate physical publication')
    save(OUT/'loaded-lifecycle.json',dict(label=label,time_s=s['clock_s'],coupled_FD_relative=fd,gate=1e-6,
        actual_contact_rejection=True,diagnosis=error.value.decode(),rejection_preserves_accepted_state=True,
        deterministic_retry=True,rebuilt_context_JSON_restart_bitwise=True,exactly_one_publication=True,duplicate_rejected=duplicate,
        scope='isolated native two-affine solve and experimental JSON owner; not production Simulator restart'))
    k.close()


def metric(k,s):
    c=k.coefficients(s);tensor=k.probe@c;angle,_=axes(tensor,np.arctan(s['geometry'][-1]['b']))
    row=dict(label=k.name,n=k.meta['nx'],fitted=k.meta['fitted_mesh'],bulk_J_m=float(.5*c@k.K@c),
        affine_storage_J_m=float(.5*c[:2]@k.K[:2,:2]@c[:2]),
        jump_self_storage_J_m=float(.5*c[2:]@k.K[2:,2:]@c[2:]),
        affine_jump_cross_J_m=float(c[:2]@k.K[:2,2:]@c[2:]),
        bulk_minus_affine_J_m=float(.5*c@k.K@c-.5*c[:2]@k.K[:2,:2]@c[:2]),preferred_deg=float(np.rad2deg(angle)))
    for name,val in zip(['xx','yy','zz','xy','pprime'],tensor):row['sample_'+name+'_Pa']=float(val)
    raw=np.loadtxt(k.path/'shear.csv',delimiter=',')@c
    # Common physical Q1 virtual jumps; their exact work is F.v.
    for region,centre,sigma in [('old',4.5,.55),('developed',6.4,.25),('zone',7.15,.3),('ahead',7.65,.2)]:
        v=np.exp(-.5*((k.nodes['x_m']-centre)/sigma)**2);test=k.Q@v;mass=k.w@test
        row[region+'_weak_shear_Pa']=float(-(k.K@c)[2:]@v/mass)
        row[region+'_normal_perturbation_Pa']=float(k.w@(test*(k.N@c))/mass)
        row[region+'_raw_shear_Pa']=float(k.w@(test*raw)/mass)
        row[region+'_raw_normal_jump_RMS_Pa']=float(np.sqrt(k.w@(test*((k.raw_Np-k.raw_Nm)@c)**2)/mass))
    return row


def frozen(label):
    states=json.loads((WORK/label/'snapshots.json').read_text());s=min(states,key=lambda v:abs(v['clock_s']-.440))
    baseline=Kernel(WORK/s['kernel_folder'],s['geometry']);rows=[metric(baseline,s)];profiles=[]
    kernels=[baseline]
    for n,mesh in [(96,'cartesian'),(64,'fitted'),(96,'fitted')]:
        k=kernel(s['geometry'],n,mesh=mesh);kernels.append(k)
        assert np.array_equal(k.s,baseline.s) and np.array_equal(k.Q,baseline.Q)
        rows.append(metric(k,s))
    reference=rows[-1]
    for row in rows:
        diff=np.array([row['sample_'+a+'_Pa']-reference['sample_'+a+'_Pa'] for a in ['xx','yy','zz','xy']])
        val=np.array([reference['sample_'+a+'_Pa'] for a in ['xx','yy','zz','xy']]);weights=np.array([1,1,1,2])
        row['tensor_reference_relative']=float(np.sqrt(weights@diff**2/(weights@val**2)))
        row['preferred_reference_error_deg']=float(row['preferred_deg']-reference['preferred_deg'])
    for k in kernels:
        c=k.coefficients(s);force=-(k.K@c)[2:];nodal=force/k.mass
        raw=np.loadtxt(k.path/'shear.csv',delimiter=',')@c;normal=k.N@c;minus=10000+k.raw_Nm@c;plus=10000+k.raw_Np@c;display=k.Q@nodal
        for q in range(len(k.s)):
            profiles.append(dict(n=k.meta['nx'],mesh='fitted' if k.meta['fitted_mesh'] else 'cartesian',time_s=s['clock_s'],
                x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),weight_m=float(k.w[q]),
                weak_normal_perturbation_Pa=float(normal[q]),raw_minus_normal_Pa=float(minus[q]),
                raw_plus_normal_Pa=float(plus[q]),raw_shear_Pa=float(raw[q]),
                displayed_mass_normalized_weak_shear_Pa=float(display[q]),
                damage=s['histories']['damage'][q],jump_m=s['histories']['jump'][q]))
        k.close()
    table(OUT/'frozen-response.csv',rows);table(OUT/'frozen-profiles.csv',profiles)
    save(OUT/'frozen-state.json',dict(label=label,time_s=s['clock_s'],source_kernel=s['kernel_folder'],
        coefficients=baseline.coefficients(s).tolist(),geometry=s['geometry'],diagnostic_only=True,
        note='Common physical jump/Maxwell memories. Independently reconstructed bulk states; none is published as an accepted transfer.',
        tests='Gaussian physical Q1 virtual jumps: centres 4.5,6.4,7.15,7.65 m; sigma .55,.25,.3,.2 m; divide work by integral of test.',
        sample='Native Gauss-weighted (1-r^2/R^2)^2 disk, radius R=.75 m, centre R ahead of geometric endpoint along terminal tangent.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['lifecycle','frozen']);p.add_argument('label');a=p.parse_args()
    globals()[a.task](a.label)
