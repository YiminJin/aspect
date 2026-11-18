#!/usr/bin/env python3
"""One nonlinear Ritz diagnostic pair with each law's unperturbed trajectory.

This is NOT a native growth trajectory. Measured native mechanical modes,
all responsive ages, signed RSF and midpoint/exact-stage aging are retained.
The scalar control reference coefficient matches the same uniform initial age,
traction and rate. Backgrounds are transient under identical constant driving.
"""
from stability import *

def main():
    data=np.load(OUT/'reduced-operator.npz');record=json.loads((OUT/'stability.json').read_text())
    K=data['K'];Psi=data['Psi'];w=data['weights'];vbar=data['modal_rate'];Vbar=data['V'];theta0=data['theta']
    dc=float(data['Dc']);r=float(data['r']);dt=float(data['dt']);mu_control=data['mu_neutral'];n=len(K)
    assert np.ptp(theta0)==0 and np.ptp(mu_control)==0
    mu_control=float(mu_control[0])
    Fbar=Psi.T@(w*friction(Vbar,theta0,dc)[0]);load=K@vbar+r*Fbar
    spec,A,M=matrix_spectrum(K,Psi,w,Vbar,theta0,dc,r)
    growth=float(max(spec.real))
    perturb=1e-5*(M@vbar)
    # Initial state/mode resolution and the native root guard select dt.
    # Measure state/rate errors after halving; a small work defect is not enough.
    steps=int(np.ceil(2/growth/dt));assert steps<=10000,'Bounded reduced diagnostic';end=steps*dt
    invK=np.linalg.inv(K)
    def solve(Fold,old_age,h,guess,b,mu):
        beta=1/(1+h*r);v=guess.copy()
        for iteration in range(30):
            V=Psi@v;stage_age,age_derivative=aging(V,old_age,h,dc)
            force,fv,ft=friction(V,stage_age,dc,b,mu)
            fstage=beta*(Fold+h*load-h*K@v)
            R=Psi.T@(w*force)-fstage;norm=np.linalg.norm(R)
            J=Psi.T@((w*(fv+ft*age_derivative))[:,None]*Psi)+beta*h*K
            if norm<1e-11:return v,V,stage_age,fstage,force,norm,J,ft
            delta=np.linalg.solve(J,-R);alpha=1.
            for search in range(24):
                trial=v+alpha*delta;TV=Psi@trial;ta,_=aging(TV,old_age,h,dc)
                tf=friction(TV,ta,dc,b,mu)[0]
                rr=Psi.T@(w*tf)-beta*(Fold+h*load-h*K@trial)
                if np.linalg.norm(rr)<norm:v=trial;break
                alpha*=.5
            else:raise RuntimeError(('root line search',iteration,norm))
        raise RuntimeError('root cap')
    def run(name,b,mu,step):
        count=round(end/step);Fb=Fbar.copy();Fp=Fbar+perturb;ab=theta0.copy();ap=theta0.copy();vb=vbar.copy();vp=vbar.copy()
        sb=np.zeros_like(Vbar);sp=sb.copy();dF=perturb.copy();da=np.zeros_like(theta0)
        histories=[];profiles=[];max_root=0.;max_state_step=0.;signed=np.zeros(2);absolute=np.zeros(2)
        def sample(t):
            nonlocal vb,vp
            eb=solve(Fb,ab,0.,vb,b,mu);ep=solve(Fp,ap,0.,vp,b,mu);vb,vp=eb[0],ep[0]
            dv=ep[1]-eb[1];_,fv,ft=friction(eb[1],ab,dc,b,mu)
            tangent=Psi@np.linalg.solve(Psi.T@((w*fv)[:,None]*Psi),dF-Psi.T@(w*ft*da))
            norm=lambda v:float(np.sqrt(sum(w*v*v)))
            histories.append(dict(time=t,background_peak_rate=float(max(abs(eb[1]))),perturbed_peak_rate=float(max(abs(ep[1]))),
                rate_perturb_L2=norm(dv),rate_perturb_max=float(max(abs(dv))),tangent_rate_L2=norm(tangent),
                finite_vs_tangent_relative=norm(dv-tangent)/max(1e-30,norm(tangent)),
                relative_age_perturb_max=float(max(abs((ap-ab)/ab))),max_slip_over_Dc=float(max(abs(sp))/dc),
                max_perturbed_slip_over_Dc=float(max(abs(sp-sb))/dc)))
            return eb[1],ep[1]
        initial=sample(0.)
        profiles.append((0.,*initial,ab.copy(),ap.copy(),sb.copy(),sp.copy()))
        selected={count//4,count//2,count}
        for k in range(1,count+1):
            h=step/2;beta=1/(1+h*r)
            base=solve(Fb,ab,h,vb,b,mu);pert=solve(Fp,ap,h,vp,b,mu)
            vb,Vb,sab,Fmb,fb,rb,J,ft=base;vp,Vp,sap,Fmp,fp,rp,_,_=pert
            # Differentiate the entire midpoint step around its own evolving
            # unperturbed background. This does not freeze transient ages.
            dv=np.linalg.solve(J,beta*dF-Psi.T@(w*ft*np.exp(-abs(Vb)*h/dc)*da))
            _,full_derivative=aging(Vb,ab,step,dc)
            da=np.exp(-abs(Vb)*step/dc)*da+full_derivative*(Psi@dv)
            dF=(2*beta-1)*dF-2*beta*h*K@dv
            for j,(old,Fm,V,force) in enumerate([(Fb,Fmb,Vb,fb),(Fp,Fmp,Vp,fp)]):
                new=2*Fm-old;external=step*Fm@invK@load;maxwell=step*r*Fm@invK@Fm
                defect=external-.5*(new@invK@new-old@invK@old)-maxwell-step*np.dot(w,force*V)
                signed[j]+=defect;absolute[j]+=abs(defect)
            Fb=2*Fmb-Fb;Fp=2*Fmp-Fp;ab,_=aging(Vb,ab,step,dc);ap,_=aging(Vp,ap,step,dc)
            sb+=step*Vb;sp+=step*Vp
            max_root=max(max_root,rb,rp);max_state_step=max(max_state_step,float(step*max(max(abs(Vb)),max(abs(Vp)))/dc))
            endpoint=sample(k*step)
            if k in selected:profiles.append((k*step,*endpoint,ab.copy(),ap.copy(),sb.copy(),sp.copy()))
        with (OUT/(name+'-history.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(histories[0]));writer.writeheader();writer.writerows(histories)
        with (OUT/(name+'-profiles.csv')).open('w') as f:
            writer=csv.writer(f);writer.writerow(['time','section','x','ds','V_initial','V_background','V','theta_background','theta','slip_background','slip'])
            for t,Vb,Vp,abp,app,sbp,spp in profiles:
                for i in range(len(w)):writer.writerow([t,int(data['section'][i]),data['x'][i],w[i],Vbar[i],Vb[i],Vp[i],abp[i],app[i],sbp[i],spp[i]])
        times=np.array([h['time'] for h in histories]);amp=np.array([h['rate_perturb_L2'] for h in histories]);mask=times>end/2
        slope=float(np.polyfit(times[mask],np.log(amp[mask]),1)[0])
        return dict(name=name,dt=step,intervals=count,end=end,initial_coefficient_frozen_growth=(record['pilot']['predicted_neutral_max_real'] if name.startswith('neutral') else growth),
            measured_late_log_amplitude_slope=slope,note='Transient background: this fitted slope is descriptive, not a constant-coefficient eigenvalue.',
            initial_rate_perturb_max=histories[0]['rate_perturb_max'],final=histories[-1],
            maximum_rate_amplification=float(max(amp)/amp[0]),final_rate_amplification=float(amp[-1]/amp[0]),
            maximum_finite_vs_tangent_relative=float(max(h['finite_vs_tangent_relative'] for h in histories)),
            signed_reduced_energy_error=signed.tolist(),absolute_reduced_energy_error=absolute.tolist(),root_residual=max_root,max_dt_speed_Dc=max_state_step),[Fb,Fp,ab,ap,*endpoint]
    result=[];states={}
    for name,b,mu,step in [('weakening',EVOLUTION,MU,dt),('neutral',DIRECT,mu_control,dt),('weakening-half',EVOLUTION,MU,dt/2),('neutral-half',DIRECT,mu_control,dt/2)]:
        summary,state=run(name,b,mu,step);result.append(summary);states[name]=state;print(name,summary,flush=True)
    comparisons={}
    for name in ['weakening','neutral']:
        c=states[name];f=states[name+'-half'];cv=c[5]-c[4];fv=f[5]-f[4];ca=c[3]-c[2];fa=f[3]-f[2]
        comparisons[name]=dict(age_perturb_relative=float(np.linalg.norm((ca-fa)/theta0)/np.linalg.norm(fa/theta0)),
            rate_perturb_relative=float(np.sqrt(sum(w*(cv-fv)**2)/sum(w*fv**2))))
    final=dict(scope='Reduced fixed-phase finite-body/cohesive Ritz pair, not a native coupled run or completed event.',
        baseline_matching=dict(max_initial_force_difference=float(max(abs(friction(Vbar,theta0,dc)[0]-friction(Vbar,theta0,dc,DIRECT,mu_control)[0]))),
            uniform_age=float(theta0[0]),neutral_mu0=mu_control,theta_equal=True,V_equal=True,generalized_drive_equal=True,
            perturbation='Identical generalized force kick at unchanged age; entire initial instantaneous rate law matches.'),
        changing_parameter=dict(Dc=dc,baseline_Dc=DC),driving=load.tolist(),initial_force=Fbar.tolist(),force_perturbation=perturb.tolist(),
        native_boundary_note='Uniform age and scalar reference coefficients are native-compatible. This constant generalized drive/preload has not been realized as the native boundary/history state.',
        background_note='Each law has its own unperturbed trajectory. The initial uniform age is steady only where |V| equals the selected peak; perturbations are compared to these evolving backgrounds.',
        results=result,temporal_comparisons=comparisons)
    (OUT/'reduced-pair.json').write_text(json.dumps(final,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for name,color in [('weakening','C3'),('neutral','C0')]:
        h=np.genfromtxt(OUT/(name+'-history.csv'),delimiter=',',names=True)
        axs[0,0].semilogy(h['time'],h['rate_perturb_L2']/h['rate_perturb_L2'][0],color=color,label=name)
        axs[0,0].semilogy(h['time'],h['tangent_rate_L2']/h['rate_perturb_L2'][0],color=color,ls='--',lw=.7)
        axs[0,1].plot(h['time'],h['background_peak_rate'],color=color,label=name+' background')
        axs[1,0].plot(h['time'],h['relative_age_perturb_max'],color=color,label=name)
        rows=np.genfromtxt(OUT/(name+'-profiles.csv'),delimiter=',',names=True);last=rows[abs(rows['time']-end)<1e-12]
        axs[1,1].plot(last['x'],(last['V']-last['V_background'])/max(abs(Vbar)),color=color,label=name)
    axs[0,0].set(ylabel='Rate-perturbation norm / initial',xlabel='Time (s)',title='Solid: pair; dashed: evolving tangent')
    axs[0,1].set(ylabel='Background peak slip rate (m/s)',xlabel='Time (s)')
    axs[1,0].set(ylabel='Maximum relative age perturbation',xlabel='Time (s)')
    axs[1,1].set(ylabel='Final signed rate perturbation / initial peak',xlabel='Section x (m)')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Fixed-phase reduced diagnostic — native coupled pilot not yet qualified')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/('reduced-pair.'+ext),dpi=180)
if __name__=='__main__':main()
