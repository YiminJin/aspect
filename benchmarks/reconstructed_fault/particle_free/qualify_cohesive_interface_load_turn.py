#!/usr/bin/env python3
"""Independent loading/Maxwell checks and retained native identity controls."""
import ctypes as ct
import json
import os
import subprocess
import numpy as np
from scipy.linalg import null_space
from cohesive_interface_load_turn import WORK, OUT, kernel, Kernel, loading, G, ETA, HISTORY, save, table, encode, point
from cohesive_interface_path import WORK as OLD_WORK, PathKernel, axes


def finite_difference(k,s,p,turned=True):
    dt=p['state']['clock_s']-s['clock_s'];beta=1/(1+G*dt/ETA);bg=np.array(p['state']['affine_memory'])
    old={key:np.array(s['histories'][key]) for key in HISTORY};ties=np.array(p['tied'],bool);C=null_space(k.Q[ties])
    if not C.shape[1]:return 0.
    v=C@np.random.default_rng(124).normal(size=C.shape[1]);v/=np.linalg.norm(v);inc=p['native']['increment']
    def force(value,bg_value=bg):
        c=np.r_[bg_value,beta*(np.array(s['memory'])+value)];dq=k.Q@value;dq[ties]=0
        r=point(old,dq,10000+k.N@c,dt);R=r['resistance'];R[ties]=0
        return C.T@((k.K@c)[2:]+k.Q.T@(k.w*R))
    h=1e-10;fd=(force(inc+h*v)-force(inc-h*v))/(2*h);expected=C.T@p['native']['J']@v
    return float(np.linalg.norm(fd-expected)/max(1,np.linalg.norm(expected)))


def main():
    original=OLD_WORK/'kernel-d6f7f210663962e8';arcs=json.loads((original/'input.json').read_text())['arcs']
    folder=WORK/'legacy-identity';folder.mkdir(exist_ok=False);table(folder/'geometry.csv',arcs)
    with (WORK/'legacy-identity.log').open('w') as f:
        subprocess.run([str(WORK/'build/cohesive_interface_path'),'16','2',str(folder/'geometry.csv'),str(folder),'6','compress'],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    names=['stiffness.csv','Q.csv','normal-work.csv','normal-minus.csv','normal-plus.csv','shear.csv','columns.csv']
    bits={n:(folder/n).read_bytes()==(original/n).read_bytes() for n in names};assert all(bits.values()),bits
    k=kernel(arcs,16);indices=np.r_[0,np.arange(2,len(k.nodes)+2)]
    matrix_error=float(max(abs(k.K[np.ix_(indices,indices)]-np.loadtxt(original/'stiffness.csv',delimiter=',')).ravel()))
    assert matrix_error<1e-7
    # Entirely tied homogeneous block: no fracture/slip, independent full-tensor
    # Maxwell recurrence under the actual non-proportional Dirichlet history.
    s=k.initial();nq=len(k.s)
    for key in HISTORY:s['histories'][key]=[1. if key=='theta' else 0.]*nq
    s['jump_offset']=[0.]*nq;s['last_resistance']=[0.]*nq
    b=np.array([.0003,0.]);data=[]
    axial=.5*np.tan(np.deg2rad(40));E=np.array([[-axial,axial,0.,.5,0.],[-.5,.5,0.,-axial,0.]])
    for target in [.05,.2,.4,.410,.42,.44,.455,.460]:
        imposed,alpha,gamma=loading(target);dt=target-s['clock_s'];b=(b+imposed-np.array(s['imposed']))/(1+.01*dt)
        before=encode(s);p=k.trial(s,target);assert encode(s)==before
        assert max(abs(np.array(p['state']['memory'])))<1e-13
        expected=2*G*(b@E);actual=k.probe@k.coefficients(p['state']);error=float(max(abs(actual-expected)))
        assert error<1e-7 and max(abs(np.array(p['state']['affine_memory'])-b))<1e-17
        direction,_=axes(actual);data.append(dict(time_s=target,imposed_alpha_deg=float(np.rad2deg(alpha)),preferred_deg=float(np.rad2deg(direction)),
            tensor_error_Pa=error,affine_memory_0=b[0],affine_memory_1=b[1],ledger_J_m=p['diagnostics']['ledger_error_J_m']))
        s=p['state']
    # Fixed-direction restriction: identical native guess/active set and
    # identical physical old state, with the independent affine coefficient zero.
    old=PathKernel(original,arcs,'work');one=old.initial();one['geometry']=arcs
    # This old helper's initial prescription assumes all points lie on the old
    # seed. A restricted virgin control avoids constructing damaged append data.
    for key in HISTORY:one['histories'][key]=[1. if key=='theta' else 0.]*len(old.s)
    one['jump_offset']=[0.]*len(old.s);one['last_resistance']=[0.]*len(old.s)
    p=old.trial(one,.2);D=ct.POINTER(ct.c_double);n=len(k.nodes)
    c0=np.ascontiguousarray(np.r_[one['wall_memory'],0.,one['memory']]);h=np.ascontiguousarray(np.array([one['histories'][key] for key in HISTORY]).T)
    da=np.ascontiguousarray([.0008,0.]);guess=np.ascontiguousarray(p['native']['increment']);ties=np.ascontiguousarray(p['tied'],dtype=np.int32)
    c=np.empty(n+2);inc=np.empty(n);r=np.empty((nq,14));J=np.empty((n,n));outties=np.empty(nq,dtype=np.int32);meta=np.empty(6);error=ct.create_string_buffer(12000)
    status=k.lib.cohesive_turn_solve_multi(k.context,c0.ctypes.data_as(D),h.ctypes.data_as(D),.2,da.ctypes.data_as(D),guess.ctypes.data_as(D),ties.ctypes.data_as(ct.POINTER(ct.c_int)),
        *[a.ctypes.data_as(D) for a in [c,inc,r,J]],outties.ctypes.data_as(ct.POINTER(ct.c_int)),meta.ctypes.data_as(D),error,len(error))
    assert status==0,error.value.decode()
    relative=float(np.linalg.norm(c[indices]-p['native']['c'])/max(1,np.linalg.norm(p['native']['c'])))
    assert max(abs(c[indices]-p['native']['c']))<1e-13 and max(abs(r-p['native']['r']).ravel())<1e-7
    save(OUT/'initial-qualification.json',dict(default_tables_bitwise=bits,restricted_matrix_absolute_error=matrix_error,
        native_single_direction_coefficient_relative=relative,homogeneous_max_tensor_error_Pa=max(d['tensor_error_Pa'] for d in data),
        two_fixed_laboratory_bases=True,stress_or_history_rotated=False,pressure_or_material_law_changed=False))
    table(OUT/'homogeneous-loading.csv',data);old.close();k.close()


if __name__=='__main__':main()
