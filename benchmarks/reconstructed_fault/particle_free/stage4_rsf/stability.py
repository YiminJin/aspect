#!/usr/bin/env python3
"""Native restoring actions + continuous RSF/aging stability; diagnostic only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-stage4-mpl')
from pathlib import Path
import json,csv
import numpy as np
from scipy.linalg import eigh,eigvals
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf'
RUN=ROOT/'build-particle-free/bulk-tip/runs/stage4-fixed-phase-stiffness-homogeneous/output'
PRES=200.;DIRECT=.015;EVOLUTION=.020;MU=.6;VREF=.001;RAD=1000.;DC=.0005

def friction(v,theta,dc=DC,b=EVOLUTION,mu=MU):
    """Actual signed regularized law and fixed-state partials; no rate floor."""
    v=np.asarray(v);theta=np.asarray(theta)
    assert np.all(theta>0)
    logs=(mu+b*np.log(theta*VREF/dc))/DIRECT-np.log(2*VREF)
    with np.errstate(divide='ignore'):
        logx=np.log(abs(v))+logs
    root=.5*np.logaddexp(0,2*logx)
    rho=np.exp(logx-root)
    value=PRES*DIRECT*np.sign(v)*np.logaddexp(logx,root)+RAD*v
    fv=PRES*DIRECT*np.exp(logs-root)+RAD
    ft=PRES*b*np.sign(v)*rho/theta
    return value,fv,ft

def aging(v,theta,dt,dc):
    speed=np.abs(v);z=speed*dt/dc
    exponential=np.exp(-z)
    small=abs(z)<1e-4
    q=np.empty_like(z);dq=np.empty_like(z)
    q[small]=1-z[small]/2+z[small]**2/6-z[small]**3/24+z[small]**4/120
    dq[small]=-.5+z[small]/3-z[small]**2/8+z[small]**3/30-z[small]**4/144
    q[~small]=-np.expm1(-z[~small])/z[~small]
    dq[~small]=((z[~small]+1)*exponential[~small]-1)/z[~small]**2
    age=theta*exponential+dt*q
    derivative=(-theta*exponential+dt*dq)*dt/dc*np.sign(v)
    return age,derivative

def matrix_spectrum(K,Psi,weights,v,theta,dc,r,b=EVOLUTION,mu=MU):
    # Relative age coordinates have fixed scale theta of the operating point;
    # they are a similarity transform, not a changing-age approximation.
    _,fv,ft=friction(v,theta,dc,b,mu)
    M=Psi.T@((weights*fv)[:,None]*Psi)
    N=Psi.T*(weights*ft*theta)[None,:]
    L=-np.linalg.solve(M,K);Q=-np.linalg.solve(M,N)
    T=np.diag(np.sign(v)/dc)@Psi
    A=np.block([[L-r*np.eye(len(K)),Q],[-T@L,-T@Q-np.diag(abs(v)/dc)]])
    return eigvals(A),A,M

def main():
    sections=np.genfromtxt(RUN/'restoring-sections.csv',delimiter=',',names=True)
    raw=np.genfromtxt(RUN/'restoring-modes.csv',delimiter=',',names=True)
    parameters=np.genfromtxt(RUN/'restoring-parameters.csv',delimiter=',',names=True)
    modes=6;n=len(sections);w=sections['ds']
    allP=raw['force_basis_per_sqrt_m'].reshape(7,n).T
    allC=raw['slip_response_per_force_coefficient'].reshape(7,n).T
    allCp=allP.T@(w[:,None]*allC)
    assert np.linalg.norm(allCp-allCp.T)/np.linalg.norm(allCp)<2e-8
    P=allP[:,:modes];C=allC[:,:modes]
    grid_probe=float(allCp[-1,-1]/sum(w*allC[:,-1]**2))
    allK=np.linalg.inv(.5*(allCp+allCp.T));allPsi=allC@allK
    augmented=eigh(allK,allPsi.T@(w[:,None]*allPsi),eigvals_only=True)
    gram=P.T@(w[:,None]*P)
    Cp=P.T@(w[:,None]*C)
    symmetry=np.linalg.norm(Cp-Cp.T)/np.linalg.norm(Cp)
    assert np.linalg.norm(gram-np.eye(modes))<1e-9
    assert symmetry<2e-8,symmetry
    Cp=.5*(Cp+Cp.T);K=np.linalg.inv(Cp);Psi=C@K
    assert np.linalg.norm(P.T@(w[:,None]*Psi)-np.eye(modes))<1e-10
    B=Psi.T@(w[:,None]*Psi)
    stiffness,eigenvectors=eigh(K,B)
    active=sections['cohesive_compliance_bound_m_Pa']>0
    assert np.max(abs(Psi[~active]))<1e-14
    lower=1/max(sections['cohesive_compliance_bound_m_Pa'][active])
    assert stiffness[0]>=lower*(1-1e-8)
    c0=sections['clamped_compliance_m_Pa']
    clamped=P.T@((w*c0)[:,None]*P)
    finite=Cp-clamped
    assert np.linalg.eigvalsh(finite).min()>-1e-12*np.linalg.norm(Cp)
    r=float(parameters['G']/parameters['eta'])
    native=np.genfromtxt(RUN/'restoring-friction.csv',delimiter=',',names=True)
    ff,fv,ft=friction(native['V'],native['theta'])
    friction_errors={key:float(np.max(abs(x-native[key])/np.maximum(1,abs(native[key])))) for key,x in [('resistance',ff),('fV',fv),('ftheta',ft)]}
    assert max(friction_errors.values())<2e-12,friction_errors
    eps=1e-5
    finite_v=(friction(native['V']*(1+eps),native['theta'])[0]-friction(native['V']*(1-eps),native['theta'])[0])/(2*eps*native['V'])
    finite_t=(friction(native['V'],native['theta']*(1+eps))[0]-friction(native['V'],native['theta']*(1-eps))[0])/(2*eps*native['theta'])
    derivative_error=max(np.max(abs(finite_v-fv)/fv),np.max(abs(finite_t-ft)/ft));assert derivative_error<1e-7
    Pa=Psi[active];wa=w[active]
    observed_v=sections['V'][active];observed_age=sections['stage_age'][active]
    normal=np.genfromtxt(RUN/'tip-sections-0016.csv',delimiter=',',names=True)
    magnitude=normal['positive_w']-normal['negative_w']
    cancellation=1-np.divide(abs(normal['V']),magnitude,out=np.ones_like(magnitude),where=magnitude>0)
    normal_signs=dict(mixed_sections=int(sum((normal['positive_w']>0)&(normal['negative_w']<0))),
        maximum_cancellation=float(max(cancellation)),positive_local_power=float(sum(normal['ds']*normal['force']*normal['positive_w'])),
        negative_local_power=float(sum(normal['ds']*normal['force']*normal['negative_w'])))
    observed_force=friction(observed_v,observed_age)[0]
    force_defect=observed_force-sections['force'][active]
    power_weights=sections['ds'][active]*abs(observed_v)
    force_check=dict(maximum_absolute=float(max(abs(force_defect))),rate_weighted_RMS=float(np.sqrt(sum(power_weights*force_defect**2)/sum(power_weights))))
    actual,A,M=matrix_spectrum(K,Pa,wa,observed_v,observed_age,DC,r)
    # Keep ALL eigenvalues including quiescent age modes; no activity filter.
    np.savetxt(OUT/'observed-spectrum.csv',np.c_[actual.real,actual.imag],delimiter=',',header='real_s_inv,imag_s_inv',comments='')
    rng=np.random.default_rng(421);direction=rng.normal(size=A.shape[0]);direction[:modes]*=1e-7
    # Independent instantaneous algebraic solve around the transient point.
    # Hold its coefficients fixed only for this local derivative test.
    def local_rhs(y):
        z=y[:modes];age=observed_age*(1+y[modes:]);vmodal=np.zeros(modes)
        base=friction(observed_v,observed_age)[0]
        for _ in range(8):
            vel=observed_v+Pa@vmodal;force,dv,_=friction(vel,age)
            residual=K@z+Pa.T@(wa*(force-base))
            step=np.linalg.solve(Pa.T@((wa*dv)[:,None]*Pa),-residual);vmodal+=step
            if np.max(abs(step))<1e-18:break
        return np.r_[vmodal-r*z,(-abs(observed_v+Pa@vmodal)*age+abs(observed_v)*observed_age)/DC/observed_age]
    fd=(local_rhs(1e-5*direction)-local_rhs(-1e-5*direction))/(2e-5)
    action_error=float(np.linalg.norm(fd-A@direction)/np.linalg.norm(A@direction));assert action_error<2e-5,action_error
    modal_cases=[]
    peak=int(np.argmax(abs(observed_v)));actual_point=(float(abs(observed_v[peak])),float(observed_age[peak]))
    for name,V,theta,steady in [('slow_steady',1e-6,DC/1e-6,True),('nucleation_steady',1e-4,DC/1e-4,True),('fast_steady',actual_point[0],DC/actual_point[0],True),('fast_observed_transient',*actual_point,False)]:
        F,q,t=friction(V,theta);d=V/DC;coupling=float(t*theta/DC)
        roots=np.concatenate([np.roots([q,q*(r+d)+k-coupling,q*r*d+k*d-r*coupling]) for k in stiffness])
        modal_cases.append(dict(name=name,minimum_restoring_Pa_m=float(stiffness[0]),signed_driving_slope_Pa_m=-float(stiffness[0]),cohesive_bound_Pa_m=float(lower),V=V,theta=theta,steady_age=steady,age_dot=1-V*theta/DC,resistance=float(F),fV=float(q),ftheta=float(t),kc_nominal=2000.,effective_elastic_kc=float(coupling-q*d),max_real=float(max(roots.real)),max_imag=float(max(abs(roots.imag))),characteristic_seconds=float(1/abs(max(roots.real)))))
    # Select ONE constitutive change, from the measured lowest patch stiffness.
    Vpilot=2e-4
    dc_new=(PRES*(EVOLUTION-DIRECT)-RAD*Vpilot)/(2*float(stiffness[0]))
    dt_admission=2*RAD*dc_new/(PRES*EVOLUTION)
    mode=eigenvectors[:,0]
    shape=Psi@mode
    if shape[np.argmax(abs(shape))]<0:mode=-mode;shape=-shape
    modal_rate=mode*(Vpilot/max(abs(shape)))
    pilot_v=Pa@modal_rate
    assert np.all(abs(pilot_v)>0),'Exactly zero-response section needs explicit passive-state handling'
    pilot_theta=np.full_like(pilot_v,dc_new/Vpilot)
    compensated_mu=MU+(EVOLUTION-DIRECT)*np.log(pilot_theta*VREF/dc_new)
    weak,Aw,Mw=matrix_spectrum(K,Pa,wa,pilot_v,pilot_theta,dc_new,r)
    neutral,An,Mn=matrix_spectrum(K,Pa,wa,pilot_v,pilot_theta,dc_new,r,b=DIRECT,mu=compensated_mu)
    fweak=friction(pilot_v,pilot_theta,dc_new)[0]
    fneutral=friction(pilot_v,pilot_theta,dc_new,DIRECT,compensated_mu)[0]
    assert np.max(abs(fweak-fneutral))<1e-10
    assert max(weak.real)>0 and max(neutral.real)<=1e-9
    # Retain weak/neutral age modes separately, even near zero.
    for name,spec in [('pilot-weakening',weak),('pilot-neutral',neutral)]:
        np.savetxt(OUT/(name+'-spectrum.csv'),np.c_[spec.real,spec.imag],delimiter=',',header='real_s_inv,imag_s_inv',comments='')
    dt=min(.5*dt_admission,.05*dc_new/max(abs(pilot_v)))
    timescale=1/max(weak.real)
    outside=(sections['x']<2.5)|(sections['x']>5.5)
    slip_shape=Psi@eigenvectors[:,0]
    norm=w*slip_shape**2
    center=np.sum(norm*sections['x'])/sum(norm)
    width=np.sqrt(np.sum(norm*(sections['x']-center)**2)/sum(norm))
    record=dict(phase_parent_time=.075,observed_endpoint_time=float(parameters['time']),observed_stage_time=float(parameters['time']-parameters['physical_dt']/2),
        G=float(parameters['G']),eta=float(parameters['eta']),m=float(parameters['m']),maxwell_rate=r,maxwell_time=1/r,
        patch=[2.5,5.5],modes=modes,grid_probe=dict(sine_number=48,half_wavelength_m=.0625,effective_restoring_Pa_m=grid_probe,augmented_minimum_Pa_m=float(augmented[0]),minimum_relative_change=float(augmented[0]/stiffness[0]-1)),spatial_note='Six sine traction patterns on the established 3 m patch. All section responses retained; zero outside traction is not tied outside slip. Reduced Ritz spectrum is not a full native spectrum.',
        gram_error=float(np.linalg.norm(gram-np.eye(modes))),compliance_symmetry=symmetry,stiffness_Pa_m=stiffness.tolist(),signed_driving_slopes_Pa_m=(-stiffness).tolist(),cohesive_lower_bound_Pa_m=float(lower),
        clamped_compliance=clamped.tolist(),finite_body_compliance=finite.tolist(),relaxed_compliance=Cp.tolist(),
        stiffness_3_modes=eigh(K[:3,:3],B[:3,:3],eigvals_only=True).tolist(),
        weakest_mode_rms_width=float(width),weakest_mode_rms_width_cells=float(width/.0625),smallest_traction_half_wave_cells=3/6/.0625,
        outside_patch_fraction_of_mode_squared_slip=float(sum(norm[outside])/sum(norm)),friction_native_relative_errors=friction_errors,friction_fd_error=float(derivative_error),local_jacobian_action_error=action_error,
        observed_signed_sections=normal_signs,observed_force_reconstruction=force_check,
        observed_spectrum=dict(max_real=float(max(actual.real)),max_imag=float(max(abs(actual.imag))),all_eigenvalues_retained=True,steady=False,note='Local coefficient-frozen variational diagnostic at actual stage rates/ages. Not an autonomous steady trajectory.'),
        representative_cases=modal_cases,
        pilot=dict(change='Dc only in the weakening specimen; uniform initial age and a single neutral reference coefficient match initial point resistance.',negative_rate_sections=int(sum(pilot_v<0)),negative_rate_fraction=float(sum(wa*np.maximum(-pilot_v,0))/sum(wa*abs(pilot_v))),Dc=dc_new,Vpeak=Vpilot,initial_age=dc_new/Vpilot,neutral_mu0=float(compensated_mu[0]),steady_age=False,initial_age_dot_range=[float(min(1-abs(pilot_v)*pilot_theta/dc_new)),float(max(1-abs(pilot_v)*pilot_theta/dc_new))],dt=dt,midpoint_root_dt_max=dt_admission,
            initial_frozen_coefficient_growth=float(max(weak.real)),predicted_max_growth=float(max(weak.real)),growth_efold_s=float(timescale),predicted_neutral_max_real=float(max(neutral.real)),intervals_per_efold=float(timescale/dt),
            peak_state_time_s=dc_new/Vpilot,selected_growth_window_s=float(4*timescale),expected_intervals=int(np.ceil(4*timescale/dt)),
            native_status='Not a native trajectory: six-dimensional mechanically relaxed diagnostic with prescribed generalized constant driving. Uniform initial age and scalar reference coefficient are native-compatible, but a native preload/boundary realization is required before a free-phase pilot.'))
    np.savez(OUT/'reduced-operator.npz',K=K,Psi=Pa,weights=wa,section=sections['section'][active],x=sections['x'][active],V=pilot_v,theta=pilot_theta,modal_rate=modal_rate,mu_neutral=compensated_mu,dt=dt,Dc=dc_new,r=r)
    with (OUT/'stability-responses.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(modal_cases[0]));writer.writeheader();writer.writerows(modal_cases)
    (OUT/'stability.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['stiffness_Pa_m','cohesive_lower_bound_Pa_m','observed_spectrum','pilot']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    axs[0].semilogy(np.arange(1,7),stiffness,'o-',label='Relaxed native patch modes');axs[0].axhline(lower,color='C1',label='Full-profile cohesive lower bound');axs[0].axhline(2000,color='0.4',ls='--',label='Nominal RSF kc');axs[0].set(xlabel='Mode',ylabel='Stiffness (Pa/m)');axs[0].legend(fontsize=7)
    axs[1].plot(sections['x'],slip_shape/max(abs(slip_shape)));axs[1].axvspan(2.5,5.5,alpha=.1,color='C0');axs[1].set(xlabel='x (m)',ylabel='Normalized relaxed slip mode',title='Finite-body response retained')
    axs[2].scatter(weak.real,weak.imag,s=10,label='Selected Dc, weakening');axs[2].scatter(neutral.real,neutral.imag,s=10,label='Matched b=a control');axs[2].axvline(0,color='k',lw=.6);axs[2].set(xlabel='Growth rate (1/s)',ylabel='Frequency (rad/s)');axs[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/('stability.'+ext),dpi=180)
if __name__=='__main__':main()
