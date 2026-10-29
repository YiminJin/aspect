#!/usr/bin/env python3
"""Bounded constrained strip derivative and softening-stop diagnosis."""
import argparse,json
from pathlib import Path
import numpy as np
from fracture_strip import Strip,SETS,G,HISTORY_HEADER

def main(out,library):
    p=dict(SETS['candidate']);p['Kf']=p['Gc']/((8/3)*p['ell']);p['m']=2*G*p['Kf']/p['drop']**2
    strip=Strip(p,library);diagnostics=[];history=[];failure=None
    for n in range(1,309):
        time=n*.00125;dU=.018*.00125 if .05+1e-12<time<=.375+1e-12 else 0.
        if n in [100,140]:
            phase=strip.phase.copy();g,H,*_=strip.evaluate(phase,dU,.00125,True)
            direction=phase*np.sin(np.pi*strip.x)**2
            epsilon=1e-5;values=[]
            for k in range(5):values.append(strip.evaluate(phase+k*epsilon*direction,dU,.00125)[0])
            fd=(-25*values[0]+48*values[1]-36*values[2]+16*values[3]-3*values[4])/(12*epsilon)
            analytic=H@direction;error=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-20))
            assert error<1e-7,error
            diagnostics.append(dict(step=n,time_s=time,complete_reduced_action_relative_error=error,
                Hessian_symmetry_relative_error=float(np.linalg.norm(H-H.T)/np.linalg.norm(H)),
                direction='nonnegative physical phase perturbation; exact global traction elimination'))
        try:history.append([n,time,dU,*strip.step(dU,.00125)])
        except RuntimeError as error:
            phase,g,H,tau=strip.last_trial
            free=np.where(~((phase<=strip.phase+1e-13)&(g>=0)))[0]
            eigen,modes=np.linalg.eigh(H[np.ix_(free,free)])
            mode=np.zeros(len(phase));mode[free]=modes[:,0]
            bound=phase<=strip.phase+1e-13
            feasible=[np.min(sign*mode[bound],initial=0)>=-1e-12 for sign in [1,-1]]
            failure=dict(step=n,time_s=time,reason=str(error),traction_Pa=float(tau),
                free_phase_nodes=len(free),minimum_Newton_free_Hessian_eigenvalue=float(eigen[0]),
                maximum_Newton_free_Hessian_eigenvalue=float(eigen[-1]),
                minimum_mode_feasible_up_to_sign=any(feasible),
                last_accepted_time_s=(n-1)*.00125,last_trial_peak_phi=float(phase.max()),
                last_accepted_peak_phi=float(strip.phase.max()),
                interpretation='Negative Newton-free-space curvature at an unaccepted, non-KKT trial. This diagnoses a nonconvex continuation limit; it does not establish a physical instability or a stable full-softening trajectory.')
            np.savetxt(out/'strip-softening-stop.csv',np.stack([strip.x,strip.phase,phase,g],axis=-1),delimiter=',',comments='',header='y_m,accepted_phi,unaccepted_trial_phi,unaccepted_weak_phase_residual_Pa_m')
            np.savez_compressed(Path('build-particle-free/bulk-fracture/strip-softening-state.npz'),
                y=strip.x,accepted_phase=strip.phase,accepted_state=strip.state,trial_phase=phase,
                trial_residual=g,trial_Hessian=H,free=free,minimum_mode=mode,
                dt=.00125,increment=dU,accepted_traction=strip.tau,parameters=[G,1e8,p['m'],1000.,p['ell'],p['Gc']])
            break
    np.savetxt(out/'candidate-strip-calibration.csv',history,delimiter=',',comments='',header=HISTORY_HEADER)
    result=dict(derivatives=diagnostics,stop=failure,accepted_steps=len(history),
                maximum_native_point_discrepancy=strip.max_oracle,maximum_native_local_ledger_error=strip.max_ledger)
    (out/'strip-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);a=p.parse_args();main(a.output,a.library)
