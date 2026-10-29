#!/usr/bin/env python3
"""Bounded direction, dissipation and mixed-mechanical checks."""
import csv,json,time
import numpy as np
from scipy.optimize import minimize_scalar,brentq
from feng_direction import *


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a))+'\n')
def axis_angle(n):return np.rad2deg(np.arctan2(n[1],n[0])%np.pi)
def axis_distance(a,b):return np.rad2deg(np.arccos(np.clip(abs(a@b),0,1)))


def symbol_minimum(C):
    grid=np.linspace(0,np.pi,1441);values=[symbol(C,t) for t in grid]
    j=np.argmin([v['mixed'] for v in values]);center=grid[j]
    result=minimize_scalar(lambda a:symbol(C,a)['mixed'],bounds=(center-.005,center+.005),method='bounded',options={'xatol':1e-14})
    roots=[]
    for left,right,x,y in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
        if x['determinant']*y['determinant']<0:
            roots.append(brentq(lambda a:symbol(C,a)['determinant'],left,right,xtol=1e-14))
    return float(result.fun),float(result.x),min(v['fixed_pressure'] for v in values),roots


def direction_checks():
    tau=tensor(xy=MU*P0/np.sqrt(1+MU**2));zero=np.zeros((3,3));history=np.array([0.,1.,0.])
    base=directions(tau,zero,P0);assert len(base['candidates'])==2
    chosen,reason=select(base,history);assert chosen is not None
    assert select(base)[0] is None
    assert select(base,np.array([1.,1.,0.])/np.sqrt(2))[0] is None
    tests=[];covariance=[]
    cases=[('positive simple shear',1.,np.array([[0.,.02],[0.,0.]]),0.),
           ('same strain plus observer spin',1.,np.array([[0.,.02],[0.,0.]]),.02),
           ('same strain zero total spin',1.,np.array([[0.,.02],[0.,0.]]),.01),
           ('symmetric shear zero spin',1.,np.array([[0.,.01],[.01,0.]]),0.),
           ('velocity reversal with retained stress',1.,np.array([[0.,-.02],[0.,0.]]),0.),
           ('shear reversal',-1.,np.array([[0.,-.02],[0.,0.]]),0.)]
    for label,sign,L,omega_added in cases:
        L=L+omega_added*np.array([[0.,-1.],[1.,0.]])
        stress=sign*tau;selected,_=select(directions(stress,zero,P0),history)
        for literal in [False,True]:
            n,omega=legacy_normal(stress,L,literal=literal)
            beta=np.array([-n[1],n[0],0.]);S=(np.outer(n,beta)+np.outer(beta,n))/2
            slip_spin=.5*np.sign(inner(stress,S))
            tests.append(dict(case=label,selector='literal historical source' if literal else 'intended total-spin selector',
                 total_spin_s1=omega,added_observer_spin_s1=omega_added,normal_deg=axis_angle(n),
                 unit_slip_spin_sign=np.sign(slip_spin),spin_agreement=bool(omega*slip_spin>0) if omega!=0 else 'undefined',
                 continuity_normal_deg=axis_angle(selected['normal']),
                 distance_from_continuity_deg=axis_distance(n,selected['normal'])))
    write('spin-controls.csv',tests)
    for angle in [.37,-1.1,np.pi/2]:
        R=rotation(angle)
        candidate,_=select(directions(R@tau@R.T,zero,P0),R@history)
        error=np.linalg.norm(candidate['N']-R@chosen['N']@R.T)
        covariance.append(dict(rotation_rad=angle,tensor_direction_error=error,
                               normal_projector_error=np.linalg.norm(np.outer(candidate['normal'],candidate['normal'])-R@np.outer(chosen['normal'],chosen['normal'])@R.T)))
        assert error<1e-12
    write('basis-covariance.csv',covariance)
    path=[];n=history
    for degrees in np.linspace(0,40,41):
        R=rotation(np.deg2rad(degrees));r=directions(R@tau@R.T,zero,P0)
        c,reason=select(r,n);assert c is not None
        for v in r['candidates']:
            # Label the analytic rotated family, not the enumerator's ordering.
            i=int(np.argmin([np.linalg.norm(v['N']-R@b['N']@R.T) for b in base['candidates']]))
            path.append(dict(stress_rotation_deg=degrees,candidate=i,normal_deg=axis_angle(v['normal']),
                 plane_tangent_deg=(axis_angle(v['normal'])+90)%180,selected=bool(np.linalg.norm(v['N']-c['N'])<1e-12),
                 maximum_Pa=r['maximum'],selection=reason))
        n=c['normal']
    write('continuity-path.csv',path)
    dominance=[]
    for qxx in [-.01,0.,.01]:
        Q=tensor(xx=qxx);r=directions(tau,Q,P0);c,reason=select(r,history)
        assert c is not None
        dominance.append(dict(Qxx_Pa=qxx,normal_deg=axis_angle(c['normal']),maximizers=len(r['candidates']),reason=reason,
                              maximum_Pa=r['maximum'],selected_objective_Pa=c['maximum']))
    assert dominance[0]['maximizers']==dominance[2]['maximizers']==1
    write('tie-versus-dominance.csv',dominance)
    # Independent dense scan + scalar optimization, using tensor contractions
    # and the actual min-positive-pressure rule, not the analytic enumerator.
    errors=[]
    for Q in [zero,tensor(.01,0),tensor(-20,40),tensor(100,-60)]:
        grid=np.linspace(0,np.pi,4001,endpoint=False)
        values=np.array([objective(tau,Q,P0,t)[0] for t in grid]);j=np.argmax(values)
        opt=minimize_scalar(lambda t:-objective(tau,Q,P0,t)[0],bounds=(grid[j]-.002,grid[j]+.002),method='bounded',options={'xatol':1e-14})
        error=abs(-opt.fun-directions(tau,Q,P0)['maximum']);errors.append(error)
        assert error<1e-9,error
    save('direction-checks.json',dict(maximum_scalar_reference_error_Pa=max(errors),covariance_max=max(r['tensor_direction_error'] for r in covariance),
        no_history_symmetry_unresolved=True,equal_distance_symmetry_unresolved=True,
        history_transport='Identity in retained fixed small-strain material coordinates; rotate director and every tensor together under passive basis changes. No finite-rotation constitutive claim.',
        legacy_literal_issue='2-D rotation_axis sets only NaN-initialized t[2]; is_finite(w[0]) fails, so sign=+1 fallback always executes.'))
    return tau,zero,chosen,base


def redistribution(tau,Q,chosen,base):
    rows=[];speed=1e-4;ms=[c['m'] for c in base['candidates']]
    for i,c in enumerate(base['candidates']):
        W=speed*c['m'];p=power(tau,Q,P0,W)
        rows.append(dict(flow=f'selected single {i}',equivalent_rate_s1=speed,normal_deg=axis_angle(c['normal']),
             Wxx_s1=W[0,0],Wxy_s1=W[0,1],physical_work_W_m3=p['conjugate_work'],Feng_friction_W_m3=p['friction'],
             work_minus_Feng_W_m3=p['defect'],system_sum_friction_W_m3=MU*c['pn']*speed))
        assert abs(p['defect'])<1e-12
    rate=speed/np.sqrt(2*inner(ms[0]+ms[1],ms[0]+ms[1]));W=rate*(ms[0]+ms[1]);p=power(tau,Q,P0,W)
    rows.append(dict(flow='two-system mixture',equivalent_rate_s1=np.sqrt(2)*np.linalg.norm(W),normal_deg='unordered net pair',
        Wxx_s1=W[0,0],Wxy_s1=W[0,1],physical_work_W_m3=p['conjugate_work'],Feng_friction_W_m3=p['friction'],
        work_minus_Feng_W_m3=p['defect'],system_sum_friction_W_m3=sum(MU*c['pn']*rate for c in base['candidates'])))
    write('mature-power.csv',rows)
    target=tensor(xx=1e-7);m=chosen['m'];projection=inner(target,m)/inner(m,m)*m
    residual=np.linalg.norm(target-projection)/np.linalg.norm(target)
    C=rate_columns(chosen['theta'],chosen['sign'],np.inf)
    s=symbol(C,np.pi/4)
    save('redistribution.json',dict(single_direction_relative_mismatch=residual,
        mixed_coefficient_at_old_45deg_over_G=s['mixed'],mixture_Feng_power_defect_W_m3=p['defect'],
        mixture_relative_power_defect=p['defect']/p['friction'],
        meaning='Old zero-stress redistribution is outside the selected single-ray flow tangent. Mechanical localization is tested separately.'))


def local_checks():
    archive=list(csv.DictReader((OUT/'archived-point-inputs.csv').open()))
    results=[];nulls=[];angular=[];comparison=[];ledger=[];fd_errors=[]
    for row in archive:
        h,beta,p,dt=[float(row[k]) for k in ['h','beta','P','dt']]
        T=tensor(float(row['Txx']),float(row['Txy']));Aold=tensor(float(row['oldAxx']),float(row['oldAxy']))
        trial=T-2*G*beta*Aold;Qref=beta*tensor(float(row['oldQxx']),float(row['oldQxy']))
        r=point_return(trial,Qref,p,h,beta);assert r['accepted'] and r['sliding'],row
        C=algorithmic_columns(r);rate=rate_columns(r['theta'],r['sign'],h)
        reference=np.array([[float(row[f'd{j}_{key}']) for j in range(4)] for key in ['xx','yy','zz','xy']])
        error=np.max(abs(C-reference));assert error<1e-12,error
        stress_ref=np.array([float(row[k]) for k in ['tauxx','tauyy','tauzz','tauxy']]);stress_error=np.linalg.norm(components(r['tau'])-stress_ref)
        assert stress_error<1e-8,stress_error
        minimum,angle,fixed,roots=symbol_minimum(C);rmin,_,_,_=symbol_minimum(rate)
        label=f"{row['kind']}-phi{float(row['phi']):g}-sign{row['normal_sign']}-dt{dt:g}"
        W=r['xi']/(beta*dt)*r['m'];power_check=power(r['tau'],r['Q'],p,W)
        assert abs(power_check['defect'])<1e-10
        results.append(dict(case=label,phi=float(row['phi']),h=h,dt_s=dt,normal_sign=row['normal_sign'],
            mixed_min_over_G=minimum,fixed_pressure_min_over_G=fixed,rate_limit_min_over_G=rmin,
            angle_at_min_deg=np.rad2deg(angle),normal_deg=axis_angle(r['normal']),
            global_margin_Pa=r['global_margin'],native_single_tangent_error=error,stress_error_Pa=stress_error,
            phase_obstacle_Pa=8-float(row['Y']),Feng_power_defect_W_m3=power_check['defect']))
        if row['kind']=='path' and float(row['phi'])==.8 and float(row['normal_sign'])==1:
            R=rotation(.37)
            rotated=point_return(R@trial@R.T,R@Qref@R.T,p,h,beta)
            assert rotated['accepted']
            covariance=np.linalg.norm(rotated['tau']-R@r['tau']@R.T)/max(1.,np.linalg.norm(r['tau']))
            assert covariance<1e-12,covariance
            # Independent finite differences include changing directions and p.
            for j,E in enumerate([deviator(xx=1),deviator(yy=1),deviator(gamma=1),np.zeros((3,3))]):
                epsilon=1e-8;dtrial=2*G*beta*E;dp=G if j==3 else 0.
                plus=point_return(trial+epsilon*dtrial,Qref,p+epsilon*dp,h,beta)
                minus=point_return(trial-epsilon*dtrial,Qref,p-epsilon*dp,h,beta)
                assert plus['accepted'] and minus['accepted']
                fd=(components(plus['tau'])-components(minus['tau']))/(2*epsilon*G)
                e=np.linalg.norm(fd-C[:,j])/max(np.linalg.norm(C[:,j]),1e-30);fd_errors.append(e);assert e<1e-6,e
            for kind,operator in [('finite step',C),('continuous rate',rate)]:
                for normal in symbol_minimum(operator)[3]:
                    s=symbol(operator,normal);defect=np.linalg.norm(s['matrix']@s['null_vector'])/np.linalg.norm(s['null_vector'])
                    assert defect<1e-12,defect
                    nulls.append(dict(operator=kind,wave_normal_deg=np.rad2deg(normal),pressure_over_G=s['pressure_over_G'],mixed_null_defect=defect))
            for normal in np.linspace(0,np.pi,721):
                a=symbol(C,normal);b=symbol(rate,normal)
                angular.append(dict(wave_normal_deg=np.rad2deg(normal),algorithmic_mixed_over_G=a['mixed'],
                    algorithmic_fixed_pressure_over_G=a['fixed_pressure'],rate_mixed_over_G=b['mixed']))
            # Complete fixed-phase BE ledger; also compare the paper's elastic limit.
            Told=tensor(float(row['oldTxx']),float(row['oldTxy']));de=(T/beta-Told)/(2*G)
            tauold=Told-2*G*Aold
            for name,b in [('mapped Maxwell',beta),('elastic Feng control',1.)]:
                qref=b*2*G*Aold/h;trial2=b*(Told+2*G*de)-2*G*b*Aold
                s=point_return(trial2,qref,p,h,b);assert s['accepted']
                A=b*Aold+s['xi']*s['m'];W=s['xi']/(b*dt)*s['m']
                oldenergy=inner(tauold,tauold)/(4*G)+G*inner(Aold,Aold)/h
                energy=inner(s['tau'],s['tau'])/(4*G)+G*inner(A,A)/h
                viscous=(inner(s['tau'],s['tau'])+h*inner(s['Q'],s['Q']))/(2*ETA) if b!=1 else 0.
                friction=power(s['tau'],s['Q'],p,W)['friction']
                remainder=inner(s['tau']-tauold,s['tau']-tauold)/(4*G)+G*inner(A-Aold,A-Aold)/h
                work=inner(s['tau'],de);defect=work-(energy-oldenergy+dt*(viscous+friction)+remainder)
                assert abs(defect)<1e-10
                ledger.append(dict(control=name,work_J_m3=work,storage_change_J_m3=energy-oldenergy,
                    viscous_loss_J_m3=dt*viscous,friction_loss_J_m3=dt*friction,BE_remainder_J_m3=remainder,ledger_defect_J_m3=defect))
    write('point-symbols.csv',results);write('mixed-null-modes.csv',nulls);write('angular-symbol.csv',angular);write('point-work.csv',ledger)
    # Rate-limit formula is an independent explicit contraction, including p.
    closed_errors=[];phase_scan=[]
    for phi in [0.,.2,.6,.75,.8,.9,.95,.99,1.]:
        h=h_of(phi)
        for sign in [1.,-1.]:
            theta=.2;C=rate_columns(theta,sign,h);minimum,angle,fixed,roots=symbol_minimum(C)
            for alpha in np.linspace(-.2,.2,81):
                expected=1. if h==0 else 1-np.cos(2*alpha)*(np.cos(2*alpha)-sign*MU*np.sin(2*alpha))/(1+1/h+sign*MU*np.sin(2*alpha))
                closed_errors.append(abs(symbol(C,theta+np.pi/2+alpha)['mixed']-expected))
            phase_scan.append(dict(phi=phi,h=h if np.isfinite(h) else 'mature',slip_sign=sign,
                mixed_min_over_G=minimum,fixed_pressure_min_over_G=fixed,roots_deg=';'.join(str(np.rad2deg(a)) for a in roots)))
    assert max(closed_errors)<1e-12
    t=brentq(lambda t:MU-2*t-3*MU*t*t+2*t**3,0,MU)
    hcrit=1/(4*t*(MU-t)/(1+t*t)**2);phicrit=brentq(lambda p:h_of(p)-hcrit,.1,.99)
    write('phase-range-symbol.csv',phase_scan)
    save('mechanical-checks.json',dict(FD_max=max(fd_errors),archived_single_tangent_max=max(r['native_single_tangent_error'] for r in results),
        stress_error_max_Pa=max(r['stress_error_Pa'] for r in results),analytic_rate_error=max(closed_errors),
        critical_h=hcrit,critical_phi_m4=phicrit,localization_persists=True,
        point_return_basis_covariance=covariance,
        phase_fixed_obstacle_control_Pa=next(r['phase_obstacle_Pa'] for r in results if r['case']=='path-phi0.8-sign1-dt0.0125')))


def global_return_controls():
    # A crucial distinction: selected-plane F=0 and stationarity do not imply
    # the GLOBAL maximizing-direction inequality after a finite increment.
    rows=[];algebra_errors=[]
    for phi,qref in [(.2,0.),(.8,0.),(.99,0.),(1.,0.),(.99,-.05)]:
        r=point_return(tensor(xy=1100.),tensor(xy=qref),P0,h_of(phi),history_normal=np.array([0.,1.,0.]))
        if phi==1.:
            assert r['accepted'] and len(r['candidates'])==2
        elif qref==0.:
            assert not r['accepted'] and r['admissible_count']==0
        else:
            assert not r['accepted'] and r['admissible_count']==2
        if phi<1.:
            # Independent algebraic solution of each closed-contact single
            # branch, to check completeness of angular root enumeration here.
            h=h_of(phi);t=np.array([0.,1100.+h*qref]);ref=np.array([0.,qref])
            for k in [1.,-1.]:
                z=-k*J(t-(1+h)*ref)-MU*(t-h*ref);s=np.linalg.norm(z);R0=MU*P0
                excess=(s-R0)*(s+R0);l=1+h;b=MU*h
                gamma=excess/(R0*l+np.sqrt((R0*l)**2+(l*l+b*b)*excess))
                v=((R0+l*gamma)*z+b*gamma*k*J(z))/(s*s)
                q=ref+gamma*k*J(v)
                err=min(np.linalg.norm(v2(c['Q'])-q) for c in r['candidates'])
                assert err<1e-8,err
                algebra_errors.append(err)
        for c in r['candidates']:
            rows.append(dict(phi=phi,Qref_xy_Pa=qref,normal_deg=axis_angle(c['normal']),xi=c['xi'],
                global_margin_Pa=c['global_margin'],allowance_Pa=c['global_allowance'],
                global_test_pass=c['global_margin']<=c['global_allowance'],
                selection=c['selection'],candidate_continuity_ok=c['continuity_ok'],point_accepted=r['accepted'],
                result='accepted one' if r['accepted'] else r['reason']))
    write('global-return-controls.csv',rows)
    save('global-return-checks.json',dict(independent_single_branch_Q_error_max_Pa=max(algebra_errors),
        finite_h_zero_history_control='Both single-branch roots fail the other global maximum; do not force acceptance.',
        reversed_memory_control='Two distinct globally admissible single roots; end-state direction maximization does not select between them.'))


if __name__=='__main__':
    start=time.monotonic();tau,Q,chosen,base=direction_checks();redistribution(tau,Q,chosen,base)
    local_checks();global_return_controls()
    save('decision.json',dict(propagation_run=False,production_edits=False,BP3_edits=False,
        redistribution_mode_removed=True,mechanical_localization_resolved=False,
        reason='Single maximizing direction removes the old redistribution mode, but pressure-coupled localization remains; finite-step global-max and symmetry issues are explicit.',seconds=time.monotonic()-start))
    print((OUT/'mechanical-checks.json').read_text());print((OUT/'decision.json').read_text())
