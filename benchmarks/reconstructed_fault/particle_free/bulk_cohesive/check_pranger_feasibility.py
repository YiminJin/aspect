#!/usr/bin/env python3
"""Mechanical gate, not a rate/state transect implementation."""
import csv
import json
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq
from pranger_feasibility import *


def write(name, rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def force_function(base, angle, y, z):
    mature=base['mature'];r=len(base['active'])
    wave=np.array([np.cos(2*angle),np.sin(2*angle)])
    tau=base['tau']+G*tensor(*(wave*y[0]),mean=y[1])
    Q=base['Q'] if mature else base['Q']+G*tensor(y[2],y[3])
    state=base['theta']+y[-1]
    pressure=base['pressure']+G*(y[0]+y[1])
    W=np.zeros((3,3));planes=[]
    for j,i in enumerate(base['active']):
        plane=geometry(base['psi'][i]+z[r+j],1 if i==0 else -1)
        W+=(base['speeds'][j]+GAMMA_REF*z[j])*plane[0];planes.append(plane)
    q=np.sqrt(2*inner(W,W))/GAMMA_REF
    mu=rsf(q,state)[0]
    forces=[];stationarity=[]
    for m,mp,mpp,N,Np,Npp in planes:
        forces.append((inner(tau-Q,m)-mu*(pressure-inner(N,tau)))/G)
        stationarity.append((inner(tau-Q,mp)+mu*inner(Np,tau))/G)
    return np.r_[forces,stationarity]


def derivative_check(base):
    angle=.37;p=pencil(base,angle,3.);ny=len(p['A']);r=len(base['active'])
    y=np.r_[20/G,5/G,3/G,-4/G,.03] if ny==5 else np.r_[20/G,5/G,.03]
    z=np.r_[np.linspace(.02,-.01,r),np.linspace(.001,-.002,r)]
    expected=p['B']@y+p['C']@z
    epsilon=1e-5
    fd=(force_function(base,angle,epsilon*y,epsilon*z)-force_function(base,angle,-epsilon*y,-epsilon*z))/(2*epsilon)
    error=float(la.norm(fd-expected)/la.norm(expected));assert error<1e-6,error
    return error


def qz_check(base):
    errors=[]
    for angle in [.2,.6]:
        values,p,r=spectrum(base,angle,3.)
        if values is None:continue
        ab=la.eigvals(p['L'],p['M'],homogeneous_eigvals=True)
        finite=abs(ab[1])>1e-9*np.hypot(abs(ab[0]),abs(ab[1]))
        eig=ab[0,finite]/ab[1,finite]
        assert len(eig)==len(values),(eig,values)
        defect=max(min(abs(v-eig))/max(1.,abs(v)) for v in values)
        errors.append(float(defect));assert defect<1e-8,defect
    return max(errors or [0.])


def null_mode_check(base):
    # Oblique 45-degree mode: conjugate rate redistribution changes Wxx,
    # leaves equivalent rate, physical stress, state and phase unchanged.
    rate=1e-7
    m=np.array([geometry(base['psi'][i],1 if i==0 else -1)[0] for i in base['active']])
    speeds=np.array([rate/(m[0,0,0]-m[1,0,0]),-rate/(m[0,0,0]-m[1,0,0])])
    dW=sum((x*y for x,y in zip(speeds,m)),np.zeros((3,3)))
    records=[]
    for kL in [0.,1.,10.,1000.]:
        p=pencil(base,np.pi/4,kL)
        for growth in [1.,1000.,1e6]:
            v=np.r_[np.zeros(len(p['A'])),speeds/GAMMA_REF,np.zeros(2)]
            raw=(p['L']-growth*p['M'])@v
            # The normal-mode stress row vanishes at 45 degrees; the velocity
            # eliminated from this row is E=delta W, an admissible div-free wave.
            residual=float(la.norm(raw)/max(1e-30,la.norm(p['L'])*la.norm(v)))
            assert residual<1e-12,residual
            Aaux=dW/(growth+RELAX);T=2*G*dW/(growth+RELAX);K=dW/growth
            maxwell=float(la.norm((growth+RELAX)*T-2*G*dW)/max(1.,la.norm(2*G*dW)))
            assert maxwell<1e-12 and la.norm(T-2*G*Aaux)==0.
            W=sum((x*y for x,y in zip(base['speeds'],m)),np.zeros((3,3)))
            dq=2*inner(W,dW)/(GAMMA_REF**2)
            records.append(dict(k_times_length=kL,arbitrary_growth_s1=growth,mixed_relative_residual=residual,
                                dq=dq,delta_Wxx_s1=dW[0,0],delta_Wxy_s1=dW[0,1],
                                delta_Axx=Aaux[0,0],delta_Txx_Pa=T[0,0],delta_Kxx=K[0,0],
                                wave_normal_deg=45.,delta_pressure_Pa=0.,
                                interpretation='Nonunique linearized mode; growth is arbitrary, not a predicted physical rate.'))
    return records


def main():
    started=time.monotonic()
    steady=make_double(.99,GAMMA_REF/(2*RELAX))
    steady['origin']='Constructed homogeneous steady Maxwell/RSF shear; K accumulates passively. Spectral base, not a time-integration test.'
    bases=dict(single=archived_single(),double_positive=make_double(.99,.002),double_steady=steady,
               double_nonproportional=make_double(.99,.002,imbalance=.5),
               double_reversed=make_double(.99,-.002),mature=make_double(.99,0.,mature=True))
    data={}
    for name,base in bases.items():
        verified=verify_base(base)
        if name=='double_steady':
            W=np.array(verified['W']);defect=la.norm(2*G/base['h']*W-RELAX*base['Q'])
            assert defect<1e-12,defect
            verified['stationary_Q_residual_Pa_s']=float(defect)
            verified['imposed_steady_strain_rate']= (W+base['tau']/(2*ETA)).tolist()
        data[name]=dict(**verified,phi=base['phi'],h=None if np.isinf(base['h']) else base['h'],
                        tau_Pa=base['tau'].tolist(),Q_Pa=base['Q'].tolist(),angles_rad=base['psi'],
                        complete_direction_pressure_state_FD=derivative_check(base),independent_QZ_error=qz_check(base))
    (OUT/'base-checks.json').write_text(json.dumps(data,indent=2)+'\n')
    records=[]
    for name,base in bases.items():
        for degrees in np.linspace(0,90,361):
            angle=np.deg2rad(degrees)
            for kL in [0.,1.,10.,100.]:
                for law,kw in [('explicit_state',{}),('same_RSF_local',dict(interaction=False)),('direct_effect_only',dict(b=0.))]:
                    values,p,r=spectrum(base,angle,kL,**kw)
                    records.append(dict(state=name,wave_normal_deg=degrees,k_times_length=kL,law=law,
                                        determined=values is not None,
                                        maximum_frozen_growth_s1=float(max(values.real)) if values is not None else '',
                                        C_rank=r['rank'],C_size=p['C'].shape[0],
                                        differentiated_constraint_denominator=r.get('denominator',''),
                                        free_dynamic_norm=r.get('free_dynamic_norm','')))
    write('mixed-spectrum.csv',records)
    # Independent closed-form angular singularity: both rates positive,
    # q-preserving redistribution; no state/operator coefficient enters.
    base=bases['double_reversed'];kappa=base['Q'][0,1]/base['tau'][0,1]
    critical=np.arccos(np.sqrt(-1/(base['h']*kappa)))/2
    roots=[]
    for sign in [-1.,1.]:
        root=critical if sign<0 else np.pi/2-critical
        roots.append(float(np.rad2deg(root)))
    angular=[]
    for offset in [1e-2,1e-3,1e-4,1e-5,1e-6]:
        angle=critical-np.deg2rad(offset)
        for kL in [0.,10.,100.]:
            for law,kw in [('explicit_state',{}),('same_RSF_local',dict(interaction=False)),('direct_effect_only',dict(b=0.))]:
                values,p,r=spectrum(base,angle,kL,**kw)
                angular.append(dict(offset_below_critical_deg=offset,wave_normal_deg=float(np.rad2deg(angle)),
                                    k_times_length=kL,law=law,maximum_frozen_growth_s1=float(max(values.real)),
                                    denominator=r['denominator']))
    write('oblique-pole.csv',angular)
    write('mature-null-mode.csv',null_mode_check(bases['mature']))
    transfer=[]
    for kL in np.logspace(-3,3,121):
        transfer.append(dict(k_times_length=kL,chosen_explicit_multiplier=1-kL*kL,
                             identity_multiplier=1.,unchosen_implicit_multiplier=1/(1+kL*kL),
                             note='Implicit transfer shown for distinction only, not substituted or calibrated.'))
    write('operator-multipliers.csv',transfer)
    (OUT/'decision.json').write_text(json.dumps(dict(proceed=False,prototype_run=False,phase_run=False,
        retired_total_K_candidate=True,critical_reversal_wave_normals_deg=roots,
        structural_failure='Common equivalent-rate/state kinetics leaves a conjugate-redistribution direction uncontrolled. Exact mature mixed pencil is singular; finite-h reversed memory has an oblique constraint pole.',
        operator='M_ex(q)=q+ell_r² Laplacian q, STATE evolution only; no Helmholtz substitution.',
        seconds=time.monotonic()-started),indent=2)+'\n')
    print(json.dumps(dict(critical_normals_deg=roots,FD_max=max(v['complete_direction_pressure_state_FD'] for v in data.values()),
                          QZ_max=max(v['independent_QZ_error'] for v in data.values()),seconds=time.monotonic()-started)),flush=True)


def limits():
    rows=[]
    for a in [.005,.0005,.00005]:
        p=pencil(archived_single(),.37,3.,a=a,b=0.)
        singular=la.svdvals(p['C'])
        rows.append(dict(a=a,sigma_min=float(singular[-1]),sigma_max=float(singular[0]),
                         condition=float(singular[0]/singular[-1])))
    base=archived_single();p=pencil(base,.37,3.)
    # Exact constant-friction endpoint: BOTH mu_q and mu_Theta vanish.
    # At the strict plane maximum the force/angle derivative is also zero;
    # only the orientation curvature remains in this 2x2 algebraic block.
    C=np.zeros_like(p['C']);i=base['active'][0]
    m,mp,mpp,N,Np,Npp=geometry(base['psi'][i],1 if i==0 else -1)
    C[1,1]=(inner(base['tau']-base['Q'],mpp)+MU0*inner(Npp,base['tau']))/G
    endpoint=int(np.linalg.matrix_rank(C))
    assert endpoint==1
    (OUT/'constant-friction-limit.json').write_text(json.dumps(dict(positive_a_direct_only=rows,
        exact_a_b_zero_rank=endpoint,positive_a_rank=2,
        meaning='Structural loss of the direct rate slope; a=b=0 is a different-order inclusion, not a regular evaluation of asinh.'),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


def independent_root():
    # Analytic constraint Qxx-kappa*tauxx=0 and its differentiated action,
    # independently of the generalized eigenvalue calculation.
    import mpmath as mp
    mp.mp.dps=80
    phi=mp.mpf('.99');h=4*phi*(1+phi)/(1-phi)**2
    mu=mp.mpf('.1');pressure=mp.mpf('10000');modulus=mp.mpf('1000000')
    Q=2*modulus*mp.mpf('-.002')/h
    tau=(Q+mp.sqrt((1+mu**2)*(mu*pressure)**2-mu**2*Q**2))/(1+mu**2)
    kappa=Q/tau
    critical=mp.acos(mp.sqrt(-1/(h*kappa)))/2
    base=make_double(.99,-.002);rows=[]
    for degrees in [10.,20.,29.9,30.1,35.,45.,60.,75.]:
        angle=np.deg2rad(degrees);p=pencil(base,angle,3.)
        u,s,vh=la.svd(p['C']);constraint=u[:,-1]@p['B']
        constraint/=constraint[2]  # coefficient of delta Qxx/G is one.
        dW=np.einsum('i,ijk->jk',vh[-1],p['shape'])
        computed=constraint@p['D']@vh[-1]/(2*dW[0,0])
        exact=1/base['h']+base['Q'][0,1]/base['tau'][0,1]*np.cos(2*angle)**2
        rows.append(dict(angle_deg=degrees,reduced_D=float(computed),closed_form_D=float(exact),
                         absolute_error=float(abs(computed-exact))))
    error=max(r['absolute_error'] for r in rows);assert error<1e-12,error
    write('angular-denominator.csv',rows)
    result=dict(precision_digits=80,h=str(h),Qxy_Pa=str(Q),physical_shear_Pa=str(tau),kappa=str(kappa),
                critical_normal_deg=str(critical*180/mp.pi),equation='1/h+kappa*cos(2alpha)^2=0',
                max_matrix_vs_closed_formula_error=error)
    (OUT/'independent-root.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    if '--limits-only' in sys.argv:limits()
    elif '--independent-root-only' in sys.argv:independent_root()
    else:main();limits();independent_root()
