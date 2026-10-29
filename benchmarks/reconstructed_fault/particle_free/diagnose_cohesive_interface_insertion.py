#!/usr/bin/env python3
"""Separate inadmissible tied release from physically admitted softening."""
import json,csv
import numpy as np
from scipy.optimize import linprog
from run_cohesive_interface_insertion import OUT
from analyze_cohesive_interface_insertion import load,read,write
import analyze_cohesive_interface_native as prior

def main():
    case='growth-64';h,q=load(case);k=len(h)-1;n=len(q['x_m'][0])//3
    op=read(OUT/case/'operators.csv');K=op['K_Pa_m'].reshape(n,n);N=op['N_Pa_m'].reshape(n,n)
    Q,w,x,P=prior.surface_rule(n);R=Q.T*w[None,:]*n
    dt=h['dt_s'][k];beta=1/(1+.01*dt);V=q['V_m_s'][k];a=q['damage'][k];old_a=q['damage'][k-1];e=q['jump_m'][k]-q['reference_m'][k]
    mu,muv,theta=prior.state_friction(abs(V),q['Theta'][k-1],dt)
    aa,bond_q,dqde,da,storage=prior.ref.bond(e,old_a,0.,np.ones(3*n))
    tied=(old_a==0)&(V==0);dqde[tied]=-3000/(2/3000);da[tied]=1/(2/3000)
    pn=q['pn_Pa'][k];friction=np.sign(V)*(pn*mu+prior.ref.ETA_RAD*abs(V))
    matrices={'bulk':beta*K,'bond':R@(dqde[:,None]*Q),'rate_and_damage_contact':R@((da*friction+a*(pn*muv+prior.ref.ETA_RAD)/dt)[:,None]*Q),'pressure':R@((beta*a*np.sign(V)*mu)[:,None]*(P@N))}
    M=sum(matrices.values());S=(M+M.T)/2
    ids=np.arange(n//8+1,7*n//8);constrained=np.any(Q[tied]>0,axis=0);free=ids[~constrained[ids]]
    eig,vec=np.linalg.eigh(S[np.ix_(ids,ids)]);v=np.zeros(n);v[ids]=vec[:,0]
    reduced=np.linalg.eigvalsh(S[np.ix_(free,free)])[0]
    assert abs(eig[0]-h['release_curvature_Pa_m'][k])<1e-5
    assert abs(reduced-h['min_curvature_Pa_m'][k])<1e-5
    physical_weights=w*8;direction=Q@v;mask=tied&(q['installed'][k]==1)
    gaps=[float(np.sum(physical_weights[mask]*(3000*abs(direction[mask])-sign*q['q_Pa'][k,mask]*direction[mask]))) for sign in [1.,-1.]]
    RT=R[ids][:,mask];count=sum(mask)
    A=np.r_[np.c_[np.eye(count),np.ones(count)],np.c_[-np.eye(count),np.ones(count)]]
    lp=linprog(np.r_[np.zeros(count),-1.],A_ub=A,b_ub=np.ones(2*count),A_eq=np.c_[RT,np.zeros(len(ids))],b_eq=RT@(q['q_Pa'][k,mask]/3000),bounds=[(-1,1)]*count+[(0,1)],method='highs')
    assert lp.success
    result=dict(negative_mode_first_order_gaps_N_m=gaps,strict_intact_multiplier_margin_Pa=float(3000*lp.x[-1]),time_s=float(h['time_s'][k]),unrestricted_min_Pa_m=float(eig[0]),admissible_min_Pa_m=float(reduced),squared_mode_fraction_on_tied_nodes=float(np.sum(v[constrained]**2)),rayleigh_contributions_Pa_m={name:float(v@A@v) for name,A in matrices.items()},conclusion='Negative unrestricted release is dominated by the prescribed cohesive softening derivative and violates intact ties; the actually admitted tangent stays positive. This is not negative admissible incremental curvature or an insertion defect.')
    (OUT/'curvature_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');write('curvature_mode.csv',[dict(x_m=float(8*i/n),mode=float(v[i]),tied=int(constrained[i])) for i in range(n)]);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
