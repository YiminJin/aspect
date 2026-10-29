#!/usr/bin/env python3
"""Bounded endpoint, reachability, mapping and full mixed kinetic checks."""
import csv,json,time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import expm
from reconcile_feng_rsf import *
from feng_direction import point_return


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a))+'\n')
def hp(phi):return 4*(1+3*phi)/(1-phi)**3


def endpoints():
    rows=[]
    for phi,q in [(.2,0.),(.8,0.),(.99,0.),(.999999,0.),(1.,0.),(.99,-.05)]:
        trial=tensor(xy=1100.);Q=tensor(xy=q);h=h_of(phi)
        ref=point_return(trial,Q,P0,h,history_normal=np.array([0.,1.,0.]))
        complete=complete_stationary_angles(trial,Q,P0,h)
        root_error=max(min(np.linalg.norm(c['tau']-v['tau'])+np.linalg.norm(c['Q']-v['Q']) for v in ref['candidates']) for c in complete)
        assert len(complete)==len(ref['candidates']) and root_error<1e-8
        r=paper_update(trial,Q,P0,h)
        row=dict(phi=phi,Qref_xy_Pa=q,paper_defined=not r.get('undefined',False),
                 paper_normal_deg=np.nan,paper_stationarity_Pa=np.nan,paper_global_margin_Pa=np.nan,
                 assumed_sign_matches_actual=False,global_allowance_Pa=tolerance(trial,Q,P0),
                 independent_positive_roots=len(ref['candidates']),
                 independent_global_roots=sum(c['global_margin']<=c['global_allowance'] for c in ref['candidates']),
                 independent_unique_admitted=ref['accepted'],paper_stationary_root_distance_Pa=np.nan,
                 complete_E7_root_error_Pa=root_error)
        if not r.get('undefined',False):
            row.update(paper_normal_deg=np.rad2deg(r['theta']),paper_stationarity_Pa=r['stationarity'],
                       paper_global_margin_Pa=r['maximum'],
                       assumed_sign_matches_actual=bool(r['angle']['assumed_sign']==np.sign(inner(trial-Q,geometry(r['theta'])[2]))),
                       paper_stationary_root_distance_Pa=min(np.linalg.norm(r['Q']-c['Q'])+np.linalg.norm(r['tau']-c['tau']) for c in ref['candidates']))
        rows.append(row)
    assert rows[0]['paper_stationary_root_distance_Pa']<1e-8
    assert rows[1]['paper_stationary_root_distance_Pa']<1e-8
    assert rows[2]['paper_global_margin_Pa']>rows[2]['global_allowance_Pa']
    assert not rows[4]['paper_defined']
    assert not rows[5]['assumed_sign_matches_actual']
    write('endpoint-formulas.csv',rows)
    # Equation 73 prints physical trial shear, whereas Eq59 and Appendix E
    # use relative shear. Isolate that distinction in one admissible-obstacle
    # synthetic unloading state; do not call it a reached native history.
    sign_rows=[];phi=.01;trial=tensor(xy=500.);Q=tensor(xy=1600.)
    for relative in [False,True]:
        r=paper_update(trial,Q,P0,h_of(phi),relative)
        sign_rows.append(dict(sign_rule='Eq59 relative' if relative else 'printed Eq73 physical',
                              phi=phi,trial_tau_xy_Pa=500.,Qref_xy_Pa=1600.,
                              xi=r['xi'],sign_conflict=r['sign_conflict'],stationarity_Pa=r['stationarity'],
                              global_margin_Pa=r['maximum'],phase_force_before_Pa=8-hp(phi)*inner(Q,Q)/(4*G)))
    assert sign_rows[0]['xi']==0 and sign_rows[1]['xi']>0 and sign_rows[1]['global_margin_Pa']>0
    write('printed-sign-control.csv',sign_rows)
    # Instantaneous viability at FIRST activation from zero A, not a forced
    # imported reversed state. The uniform physical phase obstacle is active.
    viability=[]
    for phi in [.2,.8,.99,1.]:
        h=h_of(phi);tau=tensor(xy=MU*P0/np.sqrt(1+MU**2));D=tensor(xy=.0012/2)
        cs=directions(tau,tensor(),P0)['candidates']
        for i,c in enumerate(cs):
            m=c['m'];P=np.outer(c['normal'],c['normal']);F=m+MU*P
            w=(inner(F,2*G*D)-RELAX*inner(F,tau))/(2*G*(inner(F,m)+inner(m,m)/h))
            td=2*G*(D-w*m)-RELAX*tau;qd=2*G*w/h*m
            df=[inner(s['m']+MU*np.outer(s['normal'],s['normal']),td)-inner(s['m'],qd) for s in cs]
            viability.append(dict(phi=phi,branch=i,w_s1=w,selected_Fdot_Pa_s=df[i],other_Fdot_Pa_s=df[1-i],
                                  phase_driving_Pa=0.,obstacle_force_Pa=8. if phi<1 else np.nan,
                                  tangent_viable=bool(df[1-i]<=1e-10)))
    assert all(r['other_Fdot_Pa_s']>0 for r in viability if r['phi']<1.)
    assert all(r['tangent_viable'] for r in viability if r['phi']==1.)
    write('continuous-viability.csv',viability)


def loading_histories():
    summaries=[];saved=[]
    for phi in [.2,.8,.99]:
        for n in [32,128,512,2048]:
            T=tensor();A=tensor();K=tensor();last=None;switches=0;bad=0;peak=0.;Ymax=0.;first=None;sign_conflicts=0
            h=h_of(phi);dt=1/n;beta=1/(1+RELAX*dt)
            for i in range(n):
                T=beta*(T+2*G*tensor(xy=.0012*dt/2))
                trial=T-2*G*beta*A;Qref=2*G*beta*A/h
                r=paper_update(trial,Qref,P0,h)
                assert not r.get('undefined',False)
                A=beta*A+r['xi']*r['m'];K+=r['xi']/beta*r['m']
                Y=G*inner(A,A)*hp(phi)/h**2;Ymax=max(Ymax,Y)
                peak=max(peak,r['maximum']);sign_conflicts+=r['sign_conflict']
                if not r['accepted']:
                    bad+=1
                    if first is None:
                        first=(i+1)*dt
                        strict=point_return(trial,Qref,P0,h,history_normal=np.array([0.,1.,0.]))
                        assert not strict['accepted'] and strict['admissible_count']==0
                if r['xi']>0:
                    if last is not None and inner(last,r['m'])<.49:switches+=1
                    last=r['m']
                if n in [128,2048] and (n==128 or i%8==7 or not r['accepted'] and bad==1):
                    saved.append(dict(phi=phi,steps=n,t_s=(i+1)*dt,gamma=.0012*(i+1)*dt,
                        tau_xy_Pa=r['tau'][0,1],Qxy_Pa=r['Q'][0,1],Axx=A[0,0],Axy=A[0,1],
                        normal_deg=np.rad2deg(r['theta']),maxF_Pa=r['maximum'],global_pass=r['accepted'],
                        phase_driving_Pa=Y,physical_phase_force_Pa=8-Y,
                        trajectory_admissible_so_far=bool(bad==0)))
            assert Ymax<8. # uniform obstacle holds; no hidden phase evolution
            summaries.append(dict(phi=phi,steps=n,dt_s=dt,first_failed_endpoint_s=first,failed_endpoints=bad,
                 branch_switches=switches,max_global_violation_Pa=peak,max_phase_driving_Pa=Ymax,
                 final_tau_xy_Pa=r['tau'][0,1],final_Axx=A[0,0],final_Axy=A[0,1],
                 sign_conflicts=sign_conflicts,accepted_trajectory=False))
    write('loading-summary.csv',summaries);write('loading-histories.csv',saved)
    # Zero-history damaged material is admissible under a pre-existing phase
    # obstacle, not virgin damage created for free. At exact first yield A=0.
    save('reachability.json',dict(initialization='Prescribed uniform pre-existing phase obstacle, unloaded T=A=K=0; exact initial phase force +8 Pa.',
         first_activation='Reachable elastic Maxwell ramp; Y=0 and both conjugates tied. Finite-h single-ray tangent violates the other yield inequality.',
         diagnostic_paths='Paper update is continued only to measure defects/chattering; first failure is recorded and no path after it is called accepted.',
         reversed_memory='Qref=-.05 Pa at phi=.99 corresponds to Axy=-.0019701 and Y about .02 Pa. It is obstacle-compatible, but is NOT established reachable from an accepted single-direction symmetric trajectory.'))


def mapping_and_interface():
    rows=[];errors=[]
    for phi in [.2,.8,1.]:
        xi,wi=np.polynomial.legendre.leggauss(24)
        length=2*ELL*np.sqrt(phi);r=.5*length*(xi+1);weights=.5*length*wi
        phase=(np.sqrt(phi)-r/(2*ELL))**2;gradient=-(np.sqrt(phi)-r/(2*ELL))/ELL
        density=3/(8*ELL)*(phase+ELL**2*gradient**2)
        integral=2*weights@density;chi=density/integral
        for signed_V in [V0,4*V0,-2*V0]:
            mu=friction(abs(signed_V)/V0)[0];pn=P0
            bulk=2*weights@(chi*(mu*pn*abs(signed_V)+ETA_RAD*signed_V**2))
            surface=mu*pn*abs(signed_V)+ETA_RAD*signed_V**2
            error=abs(bulk-surface)/surface;errors.append(error)
            rows.append(dict(phi=phi,support_halfwidth_m=length,crack_density_integral=integral,
                             analytic_integral=phi**1.5,chi_center_per_m=chi_center(phi),V_m_s=signed_V,
                             integrated_V_m_s=2*weights@(chi*signed_V),bulk_power_W_m2=bulk,
                             surface_power_W_m2=surface,power_relative_error=error))
        assert abs(integral-phi**1.5)<1e-12
    assert max(errors)<1e-10;write('mapping.csv',rows)
    # Prescribed coherent-plane reference: not a free-direction specimen.
    # State is exact within each constant-speed segment; current and lagged
    # pressure are compared at the same endpoints, including a true stop.
    history=[];summary=[]
    for dt in [1.,.5,.25]:
        for lag in [False,True]:
            Theta=1.;exact=1.;delta=0.;loss=0.;radloss=0.;err=0.
            previous_p=P0
            for i in range(round(40/dt)):
                t=(i+1)*dt;mid=t-dt/2
                q=1. if mid<10 else 4. if mid<20 else -2. if mid<30 else 0.
                qabs=abs(q);current_p=P0+200*np.sin(t/8)
                Theta=(Theta+V0/DC*dt)/(1+V0/DC*dt*qabs)
                exact=(exact-1/qabs)*np.exp(-V0/DC*qabs*dt)+1/qabs if qabs else exact+V0/DC*dt
                mu=friction(qabs,Theta)[0] if qabs else 0.
                exactmu=friction(qabs,exact)[0] if qabs else 0.
                pn=previous_p if lag else current_p
                traction=np.sign(q)*(pn*mu+ETA_RAD*V0*qabs)
                reference=np.sign(q)*(current_p*exactmu+ETA_RAD*V0*qabs)
                delta+=q*V0*dt;loss+=pn*mu*qabs*V0*dt;radloss+=ETA_RAD*(q*V0)**2*dt
                err=max(err,abs(traction-reference));previous_p=current_p
                history.append(dict(dt_s=dt,pressure_update='lagged' if lag else 'coupled',t_s=t,V_m_s=q*V0,
                    Theta=Theta,exact_Theta=exact,pn_current_Pa=current_p,pn_used_Pa=pn,
                    traction_Pa=traction,reference_traction_Pa=reference,slip_m=delta,
                    friction_J_m2=loss,radiation_J_m2=radloss,added_K_storage_J_m2=0.))
            summary.append(dict(dt_s=dt,pressure_update='lagged' if lag else 'coupled',
                max_traction_error_Pa=err,final_state_error=abs(Theta-exact),final_slip_m=delta,
                friction_J_m2=loss,radiation_J_m2=radloss))
    write('interface-histories.csv',history);write('interface-timesteps.csv',summary)


def bases(rad=ETA_RAD):
    archived=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/feng_direction/archived-point-inputs.csv'
    row=next(r for r in csv.DictReader(archived.open()) if r['kind']=='path' and float(r['phi'])==.8 and r['normal_sign']=='1')
    Q=tensor(float(row['Qxx']),float(row['Qxy']));tau=tensor(float(row['tauxx']),float(row['tauxy']))
    return [('mature branch 0',base_state(1.,tensor(),branch=0,rad=rad)),
            ('mature branch 1',base_state(1.,tensor(),branch=1,rad=rad)),
            ('archived shape phi .8',base_state(.8,Q,tau,rad=rad)),
            ('reversed memory branch 0',base_state(.99,tensor(xy=-.05),branch=0,rad=rad)),
            ('reversed memory branch 1',base_state(.99,tensor(xy=-.05),branch=1,rad=rad)),
            ('strict reversed memory',base_state(.99,tensor(xx=.01,xy=-.05),rad=rad))]


def kinetic_symbols():
    angular=[];summary=[];timesteps=[];wavelength=[];verification=[];states=[]
    variants=[('RSF+radiation',A_RSF,B_RSF,ETA_RAD),('RSF only',A_RSF,B_RSF,0.),
              ('state off',A_RSF,0.,ETA_RAD),('radiation only',0.,0.,ETA_RAD)]
    angles=np.unique(np.r_[np.linspace(0,np.pi,721),np.deg2rad([30.04149248,59.95850752,165.60683257,168.27995082])])
    fdmax=0.;qzmax=0.;blockmax=0.
    for variant,a,b,rad in variants:
        for label,base in bases(rad):
            states.append(dict(case=label,variant=variant,phi=base['phi'],tau_xx_Pa=base['tau'][0,0],tau_xy_Pa=base['tau'][0,1],
                               Qxx_Pa=base['Q'][0,0],Qxy_Pa=base['Q'][0,1],normal_deg=np.rad2deg(base['theta']),
                               chi_per_m=base['chi'],w_s1=base['w0'],global_error_Pa=base['global_error'],ties=base['ties']))
            values=[]
            for angle in angles:
                op=evolution(base,angle,a,b,rad);vals=np.linalg.eigvals(op);growth=max(vals.real);values.append(growth)
                angular.append(dict(case=label,variant=variant,wave_normal_deg=np.rad2deg(angle),max_growth_s1=growth))
            idx=int(np.argmax(values));theta=angles[idx]
            opt=minimize_scalar(lambda th:-max(np.linalg.eigvals(evolution(base,th,a,b,rad)).real),
                                bounds=(theta-.006,theta+.006),method='bounded',options={'xatol':1e-13})
            summary.append(dict(case=label,variant=variant,max_growth_s1=-opt.fun,wave_normal_deg=np.rad2deg(opt.x)%180,
                                min_growth_s1=min(values),direction_ties=base['ties'],rate_slope_Pa=-(pencil(base,theta,a,b,rad)[-1][0,0])*G))
            if variant not in ['RSF+radiation','radiation only'] or label!='reversed memory branch 0':continue
            for dt in [.1,.0125,.00625,.003125,.0001]:
                for lag in [False,True]:
                    amp=amplification(base,theta,dt,lag,a,b,rad)
                    growth=np.log(max(abs(np.linalg.eigvals(amp))))/dt
                    error=np.linalg.norm((amp-np.eye(len(amp)))/dt-evolution(base,theta,a,b,rad))/max(1.,np.linalg.norm(evolution(base,theta,a,b,rad)))
                    timesteps.append(dict(variant=variant,dt_s=dt,pressure_update='lagged' if lag else 'coupled',
                        wave_normal_deg=np.rad2deg(theta),effective_growth_s1=growth,
                        continuous_growth_s1=max(np.linalg.eigvals(evolution(base,theta,a,b,rad)).real),generator_relative_error=error))
            for wave in [theta,np.deg2rad(30.04149248),np.deg2rad(165.60683257),np.pi/4]:
                for k in [.01,.1,1.,10.,100.,1000.]:
                    F,E=uneliminated_pencil(base,wave,k,a,b,rad)
                    z=eigvals(F,E);finite=z[np.isfinite(z)];reference=np.linalg.eigvals(evolution(base,wave,a,b,rad))
                    err=max(min(abs(v-reference)) for v in finite)/max(1.,max(abs(reference)))
                    assert len(finite)==len(reference) and err<1e-10,err
                    qzmax=max(qzmax,err)
                    wavelength.append(dict(variant=variant,wave_normal_deg=np.rad2deg(wave),k_ell=k,wavelength_m=2*np.pi*ELL/k,
                        mixed_QZ_max_growth_s1=max(finite.real),reduced_growth_s1=max(reference.real),spectral_error=err))
    # Derivatives are checked with current and previous stress/pressure varied
    # independently; then the full mixed block time update is checked directly.
    for label,base in bases():
        for angle in [.31,.73,np.deg2rad(165.60683257)]:
            A,D,B,L,C=pencil(base,angle);ny=len(A);eps=1e-7
            for lag in [False,True]:
                expected_matrix=np.column_stack([B,C,L]) if lag else np.column_stack([B+L,C])
                differences=[]
                for j in range(ny+2+(ny if lag else 0)):
                    y=np.zeros(ny);z=np.zeros(2);old=np.zeros(ny)
                    if j<ny:y[j]=eps;expected=(B if lag else B+L)[:,j]
                    elif j<ny+2:z[j-ny]=eps;expected=C[:,j-ny]
                    else:old[j-ny-2]=eps;expected=L[:,j-ny-2]
                    plus=nonlinear_rows(base,angle,y,z,old if lag else None)
                    minus=nonlinear_rows(base,angle,-y,-z,-old if lag else None)
                    fd=(plus-minus)/(2*eps)
                    differences.append(fd-expected)
                # Complete-Jacobian relative norm includes null columns, without
                # dividing zero-column roundoff by an arbitrary tiny number.
                err=np.linalg.norm(np.column_stack(differences))/np.linalg.norm(expected_matrix)
                assert err<1e-6,(label,lag,err)
                fdmax=max(fdmax,err)
                dt=.0125;cur=B if lag else B+L;prev=L if lag else np.zeros_like(L)
                full=np.block([[np.eye(ny)-dt*A,-dt*D],[cur,C]])
                rhs=np.vstack([np.eye(ny),-prev]);direct=np.linalg.solve(full,rhs)[:ny]
                reduced=amplification(base,angle,dt,lag)
                err=np.linalg.norm(direct-reduced)/np.linalg.norm(direct);blockmax=max(blockmax,err)
                assert err<1e-10
    zero=base_state(.99,tensor(xx=.01,xy=-.05),rad=0.)
    C=pencil(zero,.73,0.,0.,0.)[-1]
    assert np.linalg.matrix_rank(C)==1
    write('kinetic-states.csv',states);write('kinetic-angles.csv',angular);write('kinetic-summary.csv',summary)
    write('pressure-lag-timesteps.csv',timesteps);write('wavelengths.csv',wavelength)
    save('kinetic-checks.json',dict(coupled_current_previous_FD_max=fdmax,full_mixed_QZ_error=qzmax,
         full_block_timestep_error=blockmax,zero_direct_effect_and_radiation_rank=int(np.linalg.matrix_rank(C)),
         meaning='Finite kinetic slope regularizes each smooth selected branch; no spatial cutoff or global switching uniqueness is established.'))


def kinetic_point_histories():
    rows=[];summary=[];failures=[]
    for dt in [.0125,.00625,.003125]:
        for lag in [False,True]:
            old=dict(tau=tensor(),Q=tensor(),p=P0,Theta=1.,normal=np.array([0.,1.,0.]))
            A=tensor();T=tensor();K=tensor();ledgermax=0.;work=fric=radloss=visc=be=0.
            passed=0
            for i in range(round(.3/dt)):
                t=(i+1)*dt;p=P0+500*np.sin(2*np.pi*t/.3)
                de=tensor(xy=.5*dt*(.02 if t<=.15+1e-12 else -.02))
                r=kinetic_step(old,de,dt,p,lag)
                if not r['accepted']:
                    for c in r['candidates']:
                        precise=precise_lagged_root(old,de,dt,p,c) if lag else {}
                        failures.append(dict(dt_s=dt,t_s=t,pressure_update='lagged' if lag else 'coupled',
                            global_margin_Pa=c['global_error'],allowance_Pa=c['global_allowance'],reason=c['reason'],
                            last_accepted_time_s=i*dt,**precise))
                    break
                passed+=1
                deltaK=dt*r['chi']*r['V']*r['m'];K+=deltaK
                A=r['beta']*(A+deltaK);T=r['beta']*(T+2*G*de)
                assert np.linalg.norm(r['tau']-(T-2*G*A))<1e-8
                n=r['normal'];pn_used=(old['p']-inner(old['tau'],np.outer(n,n))) if lag else p-inner(r['tau'],np.outer(n,n))
                W=deltaK/dt
                dwork=inner(r['tau'],de);denergy=(inner(r['tau'],r['tau'])-inner(old['tau'],old['tau']))/(4*G)
                df=dt*r['chi']*r['mu']*pn_used*r['V'];dr=dt*r['chi']*ETA_RAD*r['V']**2
                dv=dt*inner(r['tau'],r['tau'])/(2*ETA)
                db=inner(r['tau']-old['tau'],r['tau']-old['tau'])/(4*G)
                defect=dwork-denergy-df-dr-dv-db;ledgermax=max(ledgermax,abs(defect))
                assert abs(defect)<1e-10*max(1.,abs(dwork)),defect
                work+=dwork;fric+=df;radloss+=dr;visc+=dv;be+=db
                rows.append(dict(dt_s=dt,pressure_update='lagged' if lag else 'coupled',t_s=t,
                    gamma=.02*t if t<=.15 else .003-.02*(t-.15),tau_xy_Pa=r['tau'][0,1],tau_xx_Pa=r['tau'][0,0],
                    p_Pa=p,V_m_s=r['V'],signed_Wxy_s1=W[0,1],Theta=r['Theta'],normal_deg=np.rad2deg(np.arctan2(n[1],n[0])%np.pi),
                    Kxx=K[0,0],Kxy=K[0,1],Axx=A[0,0],Axy=A[0,1],work_J_m3=work,
                    storage_J_m3=inner(r['tau'],r['tau'])/(4*G),friction_J_m3=fric,radiation_J_m3=radloss,
                    Maxwell_J_m3=visc,BE_J_m3=be,global_margin_Pa=r['global_error'],ledger_defect_J_m3=defect))
                old=r
            summary.append(dict(dt_s=dt,pressure_update='lagged' if lag else 'coupled',accepted_steps=passed,
                 reached_time_s=passed*dt,final_tau_xy_Pa=old['tau'][0,1],final_Theta=old['Theta'],
                 max_ledger_defect_J_m3=ledgermax,work_J_m3=work,friction_J_m3=fric,radiation_J_m3=radloss,BE_J_m3=be))
    assert all(r['reached_time_s']>=.3-1e-12 for r in summary if r['pressure_update']=='coupled')
    write('point-histories.csv',rows);write('point-timesteps.csv',summary);write('lagged-endpoint-failures.csv',failures)


def precise_lagged_root(old,de,dt,p,c):
    """80-digit independent closed-contact root and GLOBAL force envelope."""
    import mpmath as mp
    with mp.workdps(80):
        d=mp.mpf(str(dt));g=mp.mpf(str(G));v0=mp.mpf(str(V0));chi=mp.mpf(16)
        a=mp.mpf(str(A_RSF));b=mp.mpf(str(B_RSF));mu0=mp.mpf(str(MU));eta=mp.mpf(str(ETA_RAD))
        beta=1/(1+mp.mpf(str(RELAX))*d);dc=mp.mpf(str(DC))
        xx=mp.mpf(str(old['tau'][0,0]));xy=mp.mpf(str(old['tau'][0,1]));pn=mp.mpf(str(old['p']))
        txx=beta*(xx+2*g*mp.mpf(str(de[0,0])));txy=beta*(xy+2*g*mp.mpf(str(de[0,1])))
        def quantities(logq,theta):
            q=mp.exp(logq);V=v0*q;state=(mp.mpf(str(old['Theta']))+v0*d/dc)/(1+V*d/dc)
            mu=a*mp.asinh(q/2*mp.exp((mu0+b*mp.log(state))/a))
            sn,cs=mp.sin(2*theta),mp.cos(2*theta);mx,my=-sn/2,cs/2
            tx=txx-2*g*beta*d*chi*V*mx;ty=txy-2*g*beta*d*chi*V*my
            normal=xx*cs+xy*sn;pnormal=pn-abs(normal)
            F=2*(tx*mx+ty*my)-mu*pnormal-eta*V
            angular=2*(-tx*cs-ty*sn)+2*mu*mp.sign(normal)*(-xx*sn+xy*cs)
            maximum=max(mp.sqrt((sgn*ty+mu*xx)**2+(-sgn*tx+mu*xy)**2) for sgn in [-1,1])-mu*pn-eta*V
            return F,angular,maximum
        sol=mp.findroot(lambda x,y:quantities(x,y)[:2],(mp.log(mp.mpf(str(c['q']))),mp.mpf(str(c['theta']))),tol=mp.mpf('1e-65'))
        f,angular,maximum=quantities(*sol)
        assert max(abs(f),abs(angular))<mp.mpf('1e-55')
        return dict(high_precision_global_margin_Pa=float(maximum),high_precision_residual_Pa=float(max(abs(f),abs(angular))))


if __name__=='__main__':
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    endpoints();loading_histories();mapping_and_interface();kinetic_symbols();kinetic_point_histories()
    save('decision.json',dict(endpoint_ambiguity_resolved=False,strict_branch_kinetics_regular=True,
          globally_unique_flow_established=False,propagation=False,production_edits=False,
          seconds=time.monotonic()-start))
    print((OUT/'kinetic-checks.json').read_text());print((OUT/'decision.json').read_text())
