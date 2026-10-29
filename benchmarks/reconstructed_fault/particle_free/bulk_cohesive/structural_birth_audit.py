#!/usr/bin/env python3
"""One-sided intact admission audit; no phase evolution or invented state."""
import json
import numpy as np
from structural_rsf import G,P0,ELL,RELAX,ETA_RAD,solve_speed
from structural_sliding import OUT,write_csv


def main():
    rows=[];stress=4000.;I0=.8**1.5;fracture_force=3/(8*ELL) # Gc=1
    # Same zero incoming A/K and same physical trial stress, phase and age.
    # The two normalization topologies differ explicitly, not histories.
    for topology in ['existing_support','new_uniform_ray']:
      for dt in [.05,.025,.0125,.00625,.003125,.0015625,.00078125]:
       for phi in [1e-2,1e-4,1e-6,1e-8,1e-10,1e-12,1e-14,1e-18,1e-22,1e-24]:
        beta=1/(1+RELAX*dt);h=4*phi*(1+phi)/(1-phi)**2;hp=4*(1+3*phi)/(1-phi)**3
        gamma=3*phi/(8*ELL);I=I0+.25*gamma if topology=='existing_support' else .25*gamma
        chi=gamma/I;kappa=G*beta*dt*chi*(1+1/h)
        V,mu,slope,age,error=solve_speed(stress,kappa+ETA_RAD,P0,1.,dt)
        V=float(V);Axy=beta*dt*chi*V/2;Qxy=2*G*Axy/h
        Y=hp*Qxy*Qxy/(2*G)
        rows.append(dict(topology=topology,dt_s=dt,phi=phi,trial_shear_Pa=stress,I=I,chi_inv_m=chi,
            local_V_m_s=V,A_xy=Axy,Q_xy_Pa=Qxy,driving_Pa=Y,fracture_force_Pa=fracture_force,
            weak_uniform_phase_force_Pa=fracture_force-Y,Theta=float(age),rate_error_Pa=error))
    write_csv('intact_admission.csv',rows)
    limits=[r for r in rows if r['phi']==1e-24]
    result=dict(existing_support_small_dt_absorbing=True,new_ray_zero_normalization_undefined_at_exact_intact=True,
        no_native_state_changed=True,limits=limits)
    (OUT/'intact_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(limits,indent=2))

if __name__=='__main__':main()
