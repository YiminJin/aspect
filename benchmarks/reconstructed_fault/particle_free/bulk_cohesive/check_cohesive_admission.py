#!/usr/bin/env python3
"""New-plane admission, accounted history conversion, and interface symbol."""
import json
import numpy as np
from cohesive_interface import *
from structural_rsf import plane,tensor
from reconcile_feng_rsf import A_RSF,B_RSF
from check_cohesive_interface import mapping


def admit_direction(stress,p,preference):
    # Maximize one-sided recoverable-compliance release, not frictional speed.
    values,vectors=np.linalg.eigh(stress[:2,:2]);axis=vectors[:,1]
    angle=np.arctan2(axis[1],axis[0]);candidates=[]
    for theta in [angle-np.pi/4,angle+np.pi/4]:
        n,t,S=plane([np.cos(theta),np.sin(theta),0]);pn=p-n@stress@n
        if pn<=0:continue
        s=t@stress@n;Y=.5*C0*s*s
        candidates.append((Y,n,s,pn))
    if not candidates:return None,[]
    maximum=max(c[0] for c in candidates)
    tied=[c for c in candidates if maximum-c[0]<1e-12*max(1.,maximum)]
    # Unoriented material plane; preference transported by the prescribed basis change.
    chosen=max(tied,key=lambda c:abs(c[1]@preference)) if preference is not None else None
    return chosen,candidates


def main():
    rows=[];stress=tensor(xx=700,xy=3000,mean=100);preference=np.array([0.,1.,0.]);base,_=admit_direction(stress,P0,preference)
    assert base is not None
    for angle in [0.,.37,1.1]:
        Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        for sign in [1,-1]:
            rotated=Q@(sign*stress)@Q.T;choice,candidates=admit_direction(rotated,P0,Q@preference)
            covariance=1-abs(choice[1]@(Q@base[1]));assert abs(covariance)<1e-12
            # Independent dense direction scan checks the maximizing condition.
            maximum=0.
            for theta in np.linspace(0,np.pi,10001):
                n,t,S=plane([np.cos(theta),np.sin(theta),0]);pn=P0-n@rotated@n
                if pn>0:maximum=max(maximum,.5*C0*(t@rotated@n)**2)
            assert choice[0]>=maximum-1e-12
            rows.append(dict(basis_angle_rad=angle,stress_sign=sign,normal_x=choice[1][0],normal_y=choice[1][1],shear_Pa=choice[2],pn_Pa=choice[3],Y_J_m2=choice[0],number_tied=len(candidates),basis_error=abs(covariance)))
    assert admit_direction(stress,P0,None)[0] is None # unresolved symmetry stays explicit
    write_csv('direction_admission.csv',rows)
    # A nonzero conversion-storage difference requires an explicit work port.
    loaded=mapping(.8);required=loaded['switch_work_J_m2']
    def authorize_switch(supplied):
        if abs(supplied-required)>1e-11:raise ValueError('Model switch has unaccounted storage change')
        return loaded
    try:authorize_switch(0.)
    except ValueError:rejected=True
    else:raise AssertionError('unaccounted model switch accepted')
    authorize_switch(required)
    # Retained non-coherent tensor storage relaxes without being discarded.
    E=loaded['residual_tensor_energy_J_m2'];dt=.1;beta=1/(1+RELAX*dt);new=beta*beta*E
    loss=2*RELAX*dt*new;be=(1-beta)**2*E
    assert abs(E-new-loss-be)<1e-14
    # Interface mature RSF+Maxwell/state Fourier symbol. Actual normal-pressure
    # response stays in the complex Schur symbol; no pressure lag or fixed-p shortcut.
    modes=[];V=.001;theta=V0/V;mu,muv,_=state_friction(np.array([V]),np.array([theta]),0.)
    z=V/(2*V0)*np.exp((.1+B_RSF*np.log(theta))/A_RSF);fraction=z/np.hypot(1.,z)
    RV=P0*float(muv[0])+ETA_RAD;RT=P0*B_RSF*fraction/theta
    reflection=np.diag([1.,-1.])
    for k in np.geomspace(np.pi,1e6,61):
        up=layer_DtN(k,.075);down=reflection@layer_DtN(k,.05)@reflection
        full=np.linalg.inv(np.linalg.inv(up)+np.linalg.inv(down));K=full[0,0];N=full[1,0]
        h=(K+mu[0]*N)/RV
        # V'=h m' - RT/RV Theta'; mdot=-r m'-V';
        # Thetadot= -V/Dc Theta' - theta/Dc V'.
        J=np.array([[-RELAX-h,RT/RV],[-theta/DC*h,-V/DC+theta/DC*RT/RV]],complex)
        roots=np.linalg.eigvals(J)
        modes.append(dict(k_inv_m=float(k),wavelength_m=float(2*np.pi/k),Ktt_Pa_m=float(K.real),normal_symbol_imag_Pa_m=float(N.imag),growth_max_s_inv=float(max(roots.real)),growth_min_s_inv=float(min(roots.real))))
    assert max(r['growth_max_s_inv'] for r in modes)<0
    write_csv('mature_mixed_symbol.csv',modes)
    result=dict(direction_covariance_max=max(r['basis_error'] for r in rows),unresolved_tie_without_preference=True,unaccounted_switch_rejected=rejected,required_switch_work_J_m2=required,residual_Maxwell_ledger_error=abs(E-new-loss-be),maximum_symbol_growth_s_inv=max(r['growth_max_s_inv'] for r in modes),scope='Fixed material plane, mature homogeneous steady sliding; not arbitrary direction-changing or phase-active continuum stability')
    (OUT/'admission_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
