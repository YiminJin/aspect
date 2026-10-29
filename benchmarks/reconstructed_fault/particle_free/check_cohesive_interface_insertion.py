#!/usr/bin/env python3
"""Independent transfer/restart and bounded-multiplier qualification."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from run_cohesive_interface_insertion import OUT
import analyze_cohesive_interface_native as prior

def table(p):return np.genfromtxt(p,delimiter=',',names=True,ndmin=1)
def main():
    reference=table(OUT/'installed/states.csv');summary={}
    for name in ['insert-early','insert-late','insert-short','insert-loaded','no-extension']:
        data=table(OUT/name/'states.csv');errors={k:float(max(abs(data[k]-reference[k]))) for k in ['jump_m','damage','Theta','reference_m','relaxed_jump_m','absolute_slip_m','shear_Pa','pn_Pa','V_m_s']}
        assert max(errors.values())==0,errors
        before=table(OUT/name/'insertion-before-fields.csv');after=table(OUT/name/'insertion-after-fields.csv');assert np.array_equal(before,after)
        ins=table(OUT/name/'insertion.csv')[0]
        assert all(ins[k]==0 for k in ['history_error','load_error_Pa','pressure_error_Pa','storage_change_J_m','fracture_change_J_m'])
        assert ins['geometry_publications']==1 and ins['rollback_bitwise']==1 and ins['retry_bitwise']==1
        t=ins['time_s'];at=data[abs(data['time_s']-t)<1e-12]
        assert max(at['jump_m'])>0 and max(at['absolute_slip_m'])>0
        summary[name]=dict(errors=errors,insertion_time_s=float(t),incoming_peak_slip_m=float(max(at['absolute_slip_m'])),incoming_peak_shear_Pa=float(max(at['shear_Pa'])),field_bitwise=True)
    # Actual ASPECT restart starts after accepted geometry birth.
    for file in ['states','nodal_history','history']:
        a=table(OUT/'insert-early'/f'{file}.csv');b=table(OUT/'resume'/f'{file}-resume.csv')
        a=a[a['time_s']>=min(b['time_s'])-1e-13];assert len(a)==len(b)
        for k in a.dtype.names:
            if k=='step_seconds':continue
            assert np.array_equal(a[k],b[k]),(file,k,max(abs(a[k]-b[k])))
    summary['restart_bitwise']=True
    for file in ['states','nodal_history','history']:
        a=table(OUT/'insert-loaded'/f'{file}.csv');b=table(OUT/'final-resume'/f'{file}-resume.csv');a=a[a['time_s']>=min(b['time_s'])-1e-13]
        for key in a.dtype.names:
            if key!='step_seconds':assert np.array_equal(a[key],b[key]),(file,key)
    cache=table(OUT/'cache-control/states.csv')
    for key in ['jump_m','damage','Theta','relaxed_jump_m','shear_Pa','pn_Pa','V_m_s']:assert np.array_equal(cache[key],reference[key]),key
    summary['cached_contraction_bitwise']=True;summary['final_restart_bitwise']=True
    short=table(OUT/'buffer-short/states.csv');long=table(OUT/'buffer-long/states.csv')
    for key in ['jump_m','damage','Theta','relaxed_jump_m','shear_Pa','pn_Pa','V_m_s']:assert np.array_equal(short[key],long[key]),key
    summary['tied_buffer_comparison_bitwise']=True
    # Independently recover weak nodal loads, not projected stress magnitudes.
    for case in ['installed','insert-early','insert-late','insert-short']:
        op=table(OUT/case/'operators.csv');n=int(np.sqrt(len(op)));K=op['K_Pa_m'].reshape(n,n);Q,w,x,P=prior.surface_rule(n);R=Q.T*w[None,:]*n
        state=table(OUT/case/'states.csv');node=table(OUT/case/'nodal_history.csv');hist=table(OUT/case/'history.csv');checks=[]
        for k in [0,49,89,119,len(hist)-1]:
            g=state[k*3*n:(k+1)*3*n];z=node[k*n:(k+1)*n]['relaxed_jump_m'];s=1e6/.125*hist[k]['wall_memory_m']-K@z
            cells=np.flatnonzero(g['installed'].reshape(n,3).all(axis=1));free=np.arange(cells[0]+1,cells[-1]+1)
            tied=(g['tied']==1)&(g['installed']==1);fixed=~tied
            mu=.005*np.arcsinh(abs(g['V_m_s'])/(2*5e-6)*np.exp((.1+.01*np.log(g['Theta']))/.005))
            resistance=g['q_Pa']+g['damage']*np.sign(g['V_m_s'])*(g['pn_Pa']*mu+1e6/(2*np.sqrt(1e6/2670))*abs(g['V_m_s']))
            residual=s-R@resistance;error=max(abs(residual[free]));assert error<1e-8+128*np.finfo(float).eps*max(abs(s)),error
            target=s-R[:,fixed]@resistance[fixed]
            result=linprog(np.zeros(sum(tied)),A_eq=R[free][:,tied],b_eq=target[free]/3000,bounds=(-1,1),method='highs')
            assert result.success
            assert max(abs(g['jump_m'][g['tied']==1]),default=0)==0
            checks.append(dict(time_s=float(hist[k]['time_s']),weak_error_Pa=float(error)))
        summary[case]=dict(summary.get(case,{}),independent_checks=checks)
    summary['passed']=True
    (OUT/'qualification_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
