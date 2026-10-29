#!/usr/bin/env python3
"""Read-only alternative hold candidate at the same accepted native history.

No simulator state is changed or advanced. With zero boundary displacement,
the homogeneous Maxwell-scaled stress/pressure is equilibrated if the incoming
state is. Native point returns and actual Q1 weak phase loads test whether
retaining the accepted phase is also admissible. This is not a stability test.
"""
import ctypes,json
import numpy as np
from fracture_analysis import NEW,BUILD,OUT,read

def main():
    n=140;dt=.00125;G=1e6;eta=1e8;Kf=8.;beta=1/(1+dt*G/eta)
    # Original Gauss8 fields and histories: no interpolation of material state.
    a=np.loadtxt(NEW/f'bulk-{n:04d}.csv',delimiter=',',skiprows=1,usecols=range(1,18))
    x,y,w,phi,Txx,Tyy,Txy,txx,tyy,txy,aa,q,k,L,Y,Eb,Ec=a.T
    old=np.ascontiguousarray(np.stack([Txy,aa,k,L,phi,q],axis=-1))
    inc=np.ascontiguousarray(np.stack([np.zeros_like(phi),phi,np.full_like(phi,dt)],axis=-1))
    output=np.zeros((len(phi),19));p=np.array([G,eta,4.,1000.])
    lib=ctypes.CDLL(str((BUILD/'strip/libbulk_strip_oracle.so').resolve()));fn=lib.bulk_strip_points
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_uint,array,array,array,array]
    assert fn(len(phi),p,old,inc,output)==0
    assert np.max(abs(output[:,7]))==0. # exact sticking/no slip rate
    assert np.max(abs(output[:,6]-beta*txy))/1000<1e-12
    assert np.array_equal(output[:,2],k) and np.array_equal(output[:,3],L)
    nodes=np.loadtxt(NEW/'phase-0000.csv',delimiter=',',skiprows=1);nodes=nodes[np.argsort(nodes[:,0])]
    gx,gy=np.unique(nodes[:,1]),np.unique(nodes[:,2]);grid=np.lexsort((nodes[:,1],nodes[:,2])).reshape(len(gy),len(gx))
    dx=gx[1]-gx[0];dy=gy[1]-gy[0];i=np.minimum((x/dx).astype(int),len(gx)-2);j=np.minimum((y/dy).astype(int),len(gy)-2)
    u=(x-gx[i])/dx;v=(y-gy[j])/dy
    newdrive=np.zeros(len(nodes));olddrive=np.zeros(len(nodes))
    for di,dj,N in [(0,0,(1-u)*(1-v)),(1,0,u*(1-v)),(0,1,(1-u)*v),(1,1,u*v)]:
        ids=grid[j+dj,i+di]
        newdrive+=np.bincount(ids,weights=w*N*output[:,8],minlength=len(nodes))
        olddrive+=np.bincount(ids,weights=w*N*Y,minlength=len(nodes))
    f=read(NEW/f'phase-forces-{n:04d}.csv');f.sort(order='dof')
    oldforce_error=float(np.max(abs(olddrive+f['mechanical_gradient'])))
    assert oldforce_error<1e-10
    g=(f['fracture_gradient']-newdrive)/(Kf*f['mass'])
    lower=(f['fracture_gradient']+beta**2*f['mechanical_gradient'])/(Kf*f['mass'])
    assert np.min(g)>0 and np.min(g-lower)>-1e-10
    ledger=read(NEW/'bulk-accepted.csv');row=ledger[ledger['step']==n][0]
    phase=[]
    for step in [140,141]:
        b=np.loadtxt(NEW/f'phase-{step:04d}.csv',delimiter=',',skiprows=1);b=b[np.argsort(b[:,0])];phase.append(b[:,3])
    energy=float(np.sum(w*(Eb+Ec)))
    result=dict(incoming_step=n,incoming_time_s=.175,next_time_s=.17625,
        actual_native_first_hold_phase_increment=float(np.max(phase[1]-phase[0])),
        alternative='Keep physical phase fixed, zero displacement, multiply full T/a/pressure by beta; local q uses actual native return including intact points.',
        native_point_predicted_stress_error_scaled=float(np.max(abs(output[:,6]-beta*txy))/1000),
        native_point_max_slip_rate=float(np.max(abs(output[:,7]))),
        signed_and_absolute_slip_histories_unchanged=True,
        independent_previous_weak_force_error_J_m=oldforce_error,
        alternative_min_scaled_phase_gradient=float(g.min()),
        analytic_lower_bound_min_scaled_phase_gradient=float(lower.min()),
        alternative_projected_phase_KKT=0.,
        mechanical_residual_bound_from_homogeneous_scaling=float(beta*row['mechanical_residual']),
        alternative_external_work_J_m=0.,alternative_friction_J_m=0.,
        alternative_bulk_cohesive_storage_change_J_m=(beta*beta-1)*energy,
        alternative_physical_viscous_J_m=2*(dt*G/eta)*beta*beta*energy,
        alternative_BE_remainder_J_m=(dt*G/eta)**2*beta*beta*energy,
        conclusion='A distinct no-growth KKT hold candidate exists at the same accepted state. Actual native hold growth is branch-dependent; KKT and ledger closure do not select a unique physical continuation. Neither candidate has been qualified here by a native stability test or timestep comparison.')
    (OUT/'hold-branch-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
