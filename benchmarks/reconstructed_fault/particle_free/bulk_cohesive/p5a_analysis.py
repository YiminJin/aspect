#!/usr/bin/env python3
"""Independent convex point-potential checks and P5a physical plots.

No native solver orchestration. Histories are actual C++ point exports. The
high-precision reference solves stationarity of the energy plus friction norm,
without using the implementation's h-scaled radial formula or derivatives.
"""
from pathlib import Path
import argparse,csv,json,time,hashlib
import numpy as np
import mpmath as mp

ROOT=Path(__file__).resolve().parents[4]
G,ETA,M,R,KF=1e6,1e8,4.,1000.,8.

def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def norm3(t):return np.sqrt(np.sum(np.asarray(t)[...,:3]**2,axis=-1)+2*np.asarray(t)[...,3]**2)
def h(phi):return M*phi*(1+phi)/(1-phi)**2
def hp(phi):return M*(1+3*phi)/(1-phi)**3
def write(path,header,rows):
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def independent(Tn,An,increment,phi,dt,restricted):
    mp.mp.dps=60
    v=lambda a:[mp.mpf(str(float(x))) for x in a]
    Tn,An,increment=v(Tn),v(An),v(increment)
    phi,dt=mp.mpf(str(float(phi))),mp.mpf(str(float(dt)))
    g,eta,m,resistance=map(mp.mpf,['1000000','100000000','4','1000'])
    beta=1/(1+dt*g/eta);Tp=[beta*(t+2*g*e) for t,e in zip(Tn,increment)]
    abar=[beta*x for x in An]
    compliance=m*phi*(1+phi)/(1-phi)**2
    def potential(a):
        tau=[Tp[0]-2*g*a[0],Tp[1]+2*g*a[0],Tp[2],Tp[3]-2*g*a[1]]
        bulk=(sum(t*t for t in tau[:3])+2*tau[3]**2)/(4*g)
        cohesive=2*g*sum(x*x for x in a)/compliance if compliance else mp.mpf(0)
        delta=[a[i]-abar[i] for i in range(2)]
        return bulk+cohesive+2*resistance*mp.sqrt(sum(x*x for x in delta))+8*phi
    trial=[(Tp[0]-Tp[1])/2,Tp[3]]
    if not compliance:
        a=[mp.mpf(0),mp.mpf(0)]
        size=abs(trial[1]) if restricted else mp.sqrt(sum(x*x for x in trial))
        q=[mp.mpf(0),mp.mpf(0)] if size<=resistance else [x*(1-resistance/size) for x in trial]
        if restricted:q[0]=mp.mpf(0)
    else:
        # Independent subgradient test at the kink; elsewhere solve derivatives
        # of the complete scalar energy, including the norm, at 60 digits.
        c=2*g*(1+1/compliance)
        gradient=[trial[i]-c*abar[i] for i in range(2)]
        size=abs(gradient[1]) if restricted else mp.sqrt(sum(x*x for x in gradient))
        if size<=resistance:a=abar
        elif restricted:
            f=lambda y:potential([mp.mpf(0),y])
            y=mp.findroot(lambda y:mp.diff(f,y),trial[1]/c,tol=mp.mpf('1e-50'),maxsteps=80)
            a=[mp.mpf(0),y]
        else:
            f=lambda x,y:potential([x,y])
            fx=lambda x,y:mp.diff(lambda z:f(z,y),x)
            fy=lambda x,y:mp.diff(lambda z:f(x,z),y)
            x,y=mp.findroot((fx,fy),(trial[0]/c,trial[1]/c),tol=mp.mpf('1e-50'),maxsteps=80)
            a=[x,y]
        q=[2*g*x/compliance for x in a]
    tau=[Tp[0]-2*g*a[0],Tp[1]+2*g*a[0],Tp[2],Tp[3]-2*g*a[1]]
    driving=m*(1+3*phi)/(1-phi)**3*2*sum(x*x for x in q)/(4*g)
    return dict(A=np.array([float(x) for x in a]),Q=np.array([float(x) for x in q]),
        tau=np.array([float(x) for x in tau]),Y=float(driving),potential=potential(a),beta=float(beta))

def references(run):
    started=time.monotonic();maximum=dict(A=0.,Q=0.,stress=0.,driving=0.,potential_gradient_original=0.,potential_gradient=0.,potential_gradient_extrapolated=0.)
    rows=[];frozen_direction_error=None
    for c in read(run/'derivative-inputs.csv'):
        old=np.array([c['old_Txx'],c['old_Tyy'],-c['old_Txx']-c['old_Tyy'],c['old_Txy']])
        An=np.array([c['old_Axx'],c['old_Axy']])
        inc=np.array([c['Exx'],c['Eyy'],-c['Exx']-c['Eyy'],c['Exy']])
        restricted=bool(c['restricted']);ref=independent(old,An,inc,c['phi'],c['dt'],restricted)
        actual_tau=np.array([c[k] for k in ['tauxx','tauyy','tauzz','tauxy']])
        errors=dict(A=np.sqrt(2)*np.linalg.norm(ref['A']-np.array([c['Axx'],c['Axy']]))/max(.001,np.sqrt(2)*np.linalg.norm(ref['A'])),
            Q=np.sqrt(2)*np.linalg.norm(ref['Q']-np.array([c['Qxx'],c['Qxy']]))/max(1000.,np.sqrt(2)*np.linalg.norm(ref['Q'])),
            stress=float(norm3(actual_tau-ref['tau']))/max(1000.,float(norm3(ref['tau']))),
            driving=abs(ref['Y']-c['Y'])/max(8.,abs(ref['Y'])))
        de=np.array([.0007,-.0007,0.,-.0004]);dphi=.02;differences=[]
        for eps in [mp.mpf('.001'),mp.mpf('.0005')]:
            plus=independent(old,An,inc+float(eps)*de,c['phi']+float(eps)*dphi,c['dt'],restricted)
            if c['phi']==0:
                twice=independent(old,An,inc+2*float(eps)*de,c['phi']+2*float(eps)*dphi,c['dt'],restricted)
                fd=(-3*ref['potential']+4*plus['potential']-twice['potential'])/(2*eps)
            else:
                minus=independent(old,An,inc-float(eps)*de,c['phi']-float(eps)*dphi,c['dt'],restricted)
                fd=(plus['potential']-minus['potential'])/(2*eps)
            differences.append(float(fd))
        expected=ref['beta']*(np.dot(actual_tau[:3],de[:3])+2*actual_tau[3]*de[3])+(KF-c['Y'])*dphi
        errors['potential_gradient_original']=abs(differences[0]-expected)/max(1.,abs(expected))
        errors['potential_gradient']=abs(differences[1]-expected)/max(1.,abs(expected))
        errors['potential_gradient_extrapolated']=abs((4*differences[1]-differences[0])/3-expected)/max(1.,abs(expected))
        for k,e in errors.items():maximum[k]=max(maximum[k],e)
        rows.append([int(c['case']),int(restricted),*errors.values()])
        if int(c['case'])==1 and not restricted:
            epsilon=1e-4
            qa=independent(old,An,inc+epsilon*de,c['phi'],c['dt'],False)['Q']
            qb=independent(old,An,inc-epsilon*de,c['phi'],c['dt'],False)['Q']
            actual=(qa-qb)/(2*epsilon)
            frozen=2*G*ref['beta']*np.array([(de[0]-de[1])/2,de[3]])/(1+h(c['phi']))
            frozen_direction_error=float(np.sqrt(2)*np.linalg.norm(actual-frozen)/max(1000.,np.sqrt(2)*np.linalg.norm(actual)))
    assert all(v<=1e-12 for k,v in maximum.items() if not k.startswith('potential_gradient')),maximum
    assert maximum['potential_gradient']<=1e-7,maximum
    assert frozen_direction_error>1e-6,frozen_direction_error
    return dict(maximum_defects=maximum,precision_digits=60,solved_cases=len(rows),
        potential_gradient_bound=1e-7,original_probe_passed=bool(maximum['potential_gradient_original']<=1e-7),
        perturbation_sizes=[.001,.0005],coarse_fine_error_ratio=maximum['potential_gradient_original']/maximum['potential_gradient'],
        frozen_direction_negative_control_defect=frozen_direction_error,
        seconds=time.monotonic()-started),rows

def histories(run,out):
    summaries={};datasets={};directions=[];reference_samples=[]
    sample_errors=dict(A=0.,Q=0.,stress=0.,driving=0.)
    for name in ['radial','fixed']:
        a=read(run/(name+'-history.csv'));datasets[name]=a
        previous=np.zeros(4);oldA=np.zeros(2);oldK=np.zeros(2);oldL=0.;oldgamma=oldnormal=0.
        maxerr=dict(T=0.,memory=0.,K=0.,L=0.,stress=0.,storage=0.,dissipation=0.,kkt=0.,energy_ledger=0.)
        for row in a:
            dt=row['dt_s'];beta=1/(1+dt*G/ETA)
            inc=np.array([.5*(row['engineering_normal']-oldnormal),-.5*(row['engineering_normal']-oldnormal),0.,.5*(row['engineering_xy']-oldgamma)])
            # Recover imposed increments from the declared rate interval,
            # avoiding subtraction of rounded cumulative engineering strain.
            inc=np.array([.5*dt*(1e-4 if 90<row['time_s']<=150 else 0),-.5*dt*(1e-4 if 90<row['time_s']<=150 else 0),0.,.5*dt*(1e-4 if row['time_s']<=50 else 0)])
            T=np.array([row[k] for k in ['Txx_Pa','Tyy_Pa','Tzz_Pa','Txy_Pa']]);A=np.array([row['Axx'],row['Axy']])
            K=np.array([row['Kxx'],row['Kxy']]);Q=np.array([row['Qxx_Pa'],row['Qxy_Pa']]);W=np.array([row['Wxx_s_inv'],row['Wxy_s_inv']])
            tau=np.array([row[k] for k in ['tauxx_Pa','tauyy_Pa','tauzz_Pa','tauxy_Pa']])
            metric=lambda x:np.sqrt(2)*np.linalg.norm(x)
            maxerr['T']=max(maxerr['T'],float(norm3(T-beta*(previous+2*G*inc)))/max(1000.,float(norm3(T))))
            maxerr['memory']=max(maxerr['memory'],metric(A-beta*oldA-beta*dt*W)/max(.001,metric(A)))
            maxerr['K']=max(maxerr['K'],metric(K-oldK-dt*W)/max(.001,metric(K)))
            # metric is the FULL tensor norm sqrt(2)*hypot(xx,xy).
            # Engineering flow magnitude is another sqrt(2) times that norm.
            speed=np.sqrt(2)*metric(W)
            maxerr['L']=max(maxerr['L'],abs(row['L']-oldL-dt*speed)/max(.001,row['L']))
            reconstructed=T-2*G*np.array([A[0],-A[0],0.,A[1]])
            maxerr['stress']=max(maxerr['stress'],float(norm3(tau-reconstructed))/max(1000.,float(norm3(tau))))
            Eb=float(norm3(tau))**2/(4*G);Ec=G*metric(A)**2/h(row['phi'])
            maxerr['storage']=max(maxerr['storage'],abs(Eb-row['Eb_J_m3'])/max(1.,Eb),abs(Ec-row['Ec_J_m3'])/max(1.,Ec))
            D=[float(norm3(tau))**2/(2*ETA),2*G*G*metric(A)**2/(ETA*h(row['phi'])),R*speed]
            maxerr['dissipation']=max(maxerr['dissipation'],*[abs(x-row[k])/max(.01,x) for x,k in zip(D,['Db_W_m3','Dc_W_m3','Df_W_m3'])])
            assert min(D)>=0 and row['L']+1e-12>=np.sqrt(2)*metric(K)
            c=2*G*(1+1/h(row['phi']));effective=np.array([(T[0]-T[1])/2,T[3]])-c*A
            if row['direction_defined']:
                delta=A-beta*oldA;gradient=-effective+R*delta/np.linalg.norm(delta)
                kkt=abs(gradient[1]) if name=='fixed' else np.linalg.norm(gradient)
                alpha=np.degrees(np.arctan2(W[0],W[1]))
                directions.append([name,row['time_s'],alpha,row['speed_s_inv']])
            else:
                trial=np.array([(T[0]-T[1])/2,T[3]])-c*beta*oldA
                kkt=max(0.,(abs(trial[1]) if name=='fixed' else np.linalg.norm(trial))-R)
            maxerr['kkt']=max(maxerr['kkt'],kkt/1000.)
            closure=row['work_J_m3']-Eb-Ec-row['cumulative_Db_J_m3']-row['cumulative_Dc_J_m3']-row['cumulative_Df_J_m3']-row['numerical_J_m3']
            maxerr['energy_ledger']=max(maxerr['energy_ledger'],abs(closure)/max(1.,abs(row['work_J_m3'])))
            if row['time_s'] in [20.,50.,90.,100.,120.,150.,175.,200.]:
                ref=independent(previous,oldA,inc,row['phi'],dt,name=='fixed')
                errors=dict(A=metric(ref['A']-A)/max(.001,metric(ref['A'])),Q=metric(ref['Q']-Q)/max(1000.,metric(ref['Q'])),stress=float(norm3(ref['tau']-tau))/max(1000.,float(norm3(ref['tau']))),driving=abs(ref['Y']-row['Y_Pa'])/max(8.,ref['Y']))
                for k,v in errors.items():sample_errors[k]=max(sample_errors[k],v)
                reference_samples.append([name,row['time_s'],*errors.values()])
            previous=T;oldA=A;oldK=K;oldL=row['L'];oldgamma=row['engineering_xy'];oldnormal=row['engineering_normal']
        assert max(maxerr.values())<=1e-12,(name,maxerr)
        active=a[a['direction_defined']>0];angles=np.degrees(np.arctan2(active['Wxx_s_inv'],active['Wxy_s_inv']))
        assert np.all(a['direction_defined']==(a['speed_s_inv']>0))
        final=a[-1];summaries[name]=dict(maximum_independent_history_defects=maxerr,active_steps=len(active),flow_angle_range_deg=[float(min(angles)),float(max(angles))],
            final={k:float(final[k]) for k in ['time_s','Axx','Axy','Kxx','Kxy','L','tauxx_Pa','tauxy_Pa','Y_Pa','Eb_J_m3','Ec_J_m3','work_J_m3','cumulative_Db_J_m3','cumulative_Dc_J_m3','cumulative_Df_J_m3','numerical_J_m3']})
    assert max(sample_errors.values())<=1e-12,sample_errors
    prefix=datasets['radial']['time_s']<=90.
    assert np.array_equal(datasets['radial'][prefix],datasets['fixed'][prefix])
    first_new=datasets['radial'][(datasets['radial']['time_s']>90)&(datasets['radial']['direction_defined']>0)][0]
    assert first_new['old_Axy']>0. and first_new['new_Axx']>0.
    write(out/'active-directions.csv',['mode','time_s','tensor_flow_angle_deg','speed_s_inv'],directions)
    write(out/'independent-path-reference.csv',['mode','time_s','A_defect','Q_defect','stress_defect','Y_defect'],reference_samples)
    # Exact recurrence decomposition of the loaded old tensor, not a rotated
    # scalar. Each new tensor contribution relaxes only on later intervals.
    a=datasets['radial'];start=a[a['time_s']==90.][0];end=a[a['time_s']==150.][0]
    beta=1/(1+.25*G/ETA);steps=240
    oldpart=beta**steps*np.array([start['Axx'],start['Axy']]);newpart=np.zeros(2)
    for row in a[(a['time_s']>90)&(a['time_s']<=150)]:
        newpart=beta*newpart+np.array([row['new_Axx'],row['new_Axy']])
    actual=np.array([end['Axx'],end['Axy']]);error=np.sqrt(2)*np.linalg.norm(actual-oldpart-newpart)/max(.001,np.sqrt(2)*np.linalg.norm(actual))
    assert error<=1e-12,error
    decomposition=dict(start_time_s=90.,end_time_s=150.,initial=[float(start['Axx']),float(start['Axy'])],relaxed_old=oldpart.tolist(),accumulated_new=newpart.tolist(),actual=actual.tolist(),scaled_defect=float(error))
    return datasets,dict(histories=summaries,high_precision_sample_defects=sample_errors,memory_decomposition=decomposition,
        identical_loaded_prefix_through_90_s=True,
        first_new_direction_flow_bracket_s=[float(first_new['time_s']-first_new['dt_s']),float(first_new['time_s'])],
        incoming_Axy_at_first_new_flow=float(first_new['old_Axy']))

def plots(data,summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'radial':'#136f9b','fixed':'#d27817'}
    def finish(fig,name):fig.savefig(out/name,dpi=155);plt.close(fig)
    def style(ax):
        for t in [50,90,150]:ax.axvline(t,c='.7',lw=.7,ls=':')
        ax.set_xlim(0,200);ax.set_xlabel('Time [s]');ax.grid(alpha=.2)
    fig,axs=plt.subplots(3,2,figsize=(11,10),layout='constrained');a=data['radial'];t=a['time_s']
    axs[0,0].plot(t,a['engineering_xy'],label='xy engineering shear');axs[0,0].plot(t,a['engineering_normal'],label='normal difference')
    axs[0,0].set_ylabel('Imposed strain [1]');axs[0,0].set_title('Same strain increments; phase prescribed at 0.2');axs[0,0].legend(fontsize=8)
    for name,a in data.items():
        t=a['time_s'];c=colors[name];label='Changing flow' if name=='radial' else 'Fixed direction'
        axs[0,1].plot(t,a['tauxy_Pa']/1000,c=c,label=label)
        axs[1,0].plot(t,(a['tauxx_Pa']-a['tauyy_Pa'])/2000,c=c,label=label)
        axs[1,1].plot(t,a['speed_s_inv'],c=c,label=label)
        angle=np.ma.masked_where(a['direction_defined']==0,np.degrees(np.arctan2(a['Wxx_s_inv'],a['Wxy_s_inv'])))
        axs[2,0].plot(t,angle,c=c,label=label)
        axs[2,1].plot(t,a['Y_Pa'],c=c,label=label)
    axs[0,1].set_ylabel('Physical shear stress τxy [kPa]');axs[0,1].legend(fontsize=8)
    axs[1,0].set_ylabel('Physical (τxx−τyy)/2 [kPa]');axs[1,1].set_ylabel('Flow magnitude √2‖W‖ [1/s]')
    axs[2,0].set_ylabel('Active tensor-flow angle α [degrees]');axs[2,0].set_title('α=atan2(Wxx,Wxy); absent when flow is zero',fontsize=10)
    axs[2,1].set_ylabel('Mechanical phase driving Y [Pa]');axs[2,1].axhline(KF,c='.4',ls='--',label='Uniform AT1 resistance Kf');axs[2,1].legend(fontsize=8)
    for ax in axs.flat:style(ax)
    fig.suptitle('Point response: changing new flow, preserving accepted tensor memory',fontsize=14);finish(fig,'response.png')
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for name,a in data.items():
        t=a['time_s'];c=colors[name];label='Changing flow' if name=='radial' else 'Fixed direction'
        axs[0,0].plot(t,a['Axx'],c=c,label=label+' Axx');axs[0,0].plot(t,a['Axy'],c=c,ls='--',label=label+' Axy')
        axs[0,1].plot(t,a['Kxx'],c=c,label=label+' Kxx');axs[0,1].plot(t,a['Kxy'],c=c,ls='--',label=label+' Kxy')
        axs[1,0].plot(t,a['L'],c=c,label=label+' L');axs[1,0].plot(t,2*np.hypot(a['Kxx'],a['Kxy']),c=c,ls=':',label=label+' √2‖K‖')
    for ax,label in zip([axs[0,0],axs[0,1],axs[1,0]],['Recoverable memory A [1]','Accumulated tensor K [1]','Accumulated magnitudes [1]']):style(ax);ax.set_ylabel(label);ax.legend(fontsize=8)
    d=summary['memory_decomposition'];ax=axs[1,1]
    # Display (xy,xx), matching the plotted flow-angle chart.
    initial=np.array(d['initial'])[::-1]*1000;old=np.array(d['relaxed_old'])[::-1]*1000;new=np.array(d['accumulated_new'])[::-1]*1000;total=np.array(d['actual'])[::-1]*1000
    for origin,v,c,label in [(np.zeros(2),initial,'.6','A at 90 s'),(np.zeros(2),old,'#a23d30','Old memory after relaxation'),(old,new,'#268447','New increments after relaxation'),(np.zeros(2),total,'#136f9b','A at 150 s')]:
        ax.quiver(*origin,*v,angles='xy',scale_units='xy',scale=1,color=c,width=.009,label=label)
    ax.set_xlim(-.05,max(initial[0],total[0])*1.2);ax.set_ylim(-.05,total[1]*1.2);ax.set_aspect('equal');ax.grid(alpha=.2)
    ax.set_xlabel('Axy × 1000 [1]');ax.set_ylabel('Axx × 1000 [1]')
    ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2)
    ax.set_title('A150 = relaxed old A90 + relaxed new increments',fontsize=10)
    fig.suptitle('Old components relax; they are not rotated into a new scalar',fontsize=14);finish(fig,'tensor-memory.png')
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for name,a in data.items():
        t=a['time_s'];c=colors[name];label='Changing flow' if name=='radial' else 'Fixed direction'
        axs[0,0].plot(t,a['Eb_J_m3'],c=c,label=label+' bulk');axs[0,0].plot(t,a['Ec_J_m3'],c=c,ls='--',label=label+' cohesive')
        storage=a['Eb_J_m3']+a['Ec_J_m3'];physical=sum(a[k] for k in ['cumulative_Db_J_m3','cumulative_Dc_J_m3','cumulative_Df_J_m3'])
        axs[0,1].plot(t,a['work_J_m3'],c=c,label=label+' endpoint work');axs[0,1].plot(t,storage+physical+a['numerical_J_m3'],c=c,ls=':',label=label+' ΔE+D+N')
        for key,ls,component in [('cumulative_Db_J_m3','-','bulk viscosity'),('cumulative_Dc_J_m3','--','cohesive viscosity'),('cumulative_Df_J_m3',':','friction')]:
            axs[1,0].plot(t,a[key],c=c,ls=ls,label=label+' '+component)
        axs[1,1].plot(t,a['numerical_J_m3'],c=c,label=label+' BE numerical remainder')
    axs[0,0].set_ylabel('Recoverable storage [J/m³]');axs[0,1].set_ylabel('Work accounting [J/m³]')
    axs[1,0].set_ylabel('Cumulative physical dissipation [J/m³]');axs[1,1].set_ylabel('Numerical remainder [J/m³]')
    for ax in axs.flat:style(ax);ax.legend(fontsize=8)
    fig.suptitle('Point work: recoverable storage, physical dissipation and BE remainder',fontsize=13);finish(fig,'energy.png')

def main(run,out):
    out.mkdir(exist_ok=True,parents=True)
    reference,rows=references(run)
    write(out/'independent-point-reference.csv',['case','restricted','A_defect','Q_defect','stress_defect','Y_defect','original_potential_gradient_defect','halved_potential_gradient_defect','extrapolated_potential_gradient_defect'],rows)
    data,summary=histories(run,out)
    entry=json.load(open(ROOT/'build-particle-free/p5a-point/entry.json'))
    unchanged={p:digest for p,digest in entry['files'].items() if not p.endswith('P5_DIRECTION_SPECIFICATION.md')}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in unchanged.items())
    native=json.load(open(run/'native-checks.json'));assert native['passed']
    result=dict(scope='Standalone stateless C++ point law/fixture only; no ASPECT native solver run.',
        cpp_point_checks=native,independent_point_potential=reference,**summary,
        protected_native_P4_BP3_and_scalar_files_unchanged=unchanged,
        native_solver_integrated=False,production_checkpoint_changed=False,
        finite_rotation_or_objective_rate_qualified=False,phase_evolution_qualified=False,
        strict_reference_match_passed=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for f in ['radial-history.csv','fixed-history.csv','intact-threshold.csv','derivative-inputs.csv','native-checks.json']:
        import shutil;shutil.copy2(run/f,out/f)
    plots(data,summary,out)
    print(json.dumps(dict(reference=reference,path_summary=summary['histories'],memory_decomposition=summary['memory_decomposition']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.run,a.out)
