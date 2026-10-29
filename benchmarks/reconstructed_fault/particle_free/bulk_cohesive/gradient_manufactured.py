#!/usr/bin/env python3
"""Inverse-manufactured strip: exact microfree gradient energy and BE work.

K is prescribed in the plastic segment; compatible shear loading is calculated.
This verifies one admissible solution, not a forward solver or its uniqueness.
"""
import csv,json
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_point import OUT
G=1e6;eta=1e8;phi=.8;h=4*phi*(1+phi)/(1-phi)**2
dt=.00625;beta=1/(1+dt*G/eta);length=.25;p=10000.;mu=.1;R=mu*p
nodes,weights=leggauss(64);x=(nodes+1)*length/2;weights*=length/2
f=1+.3*np.cos(2*np.pi*x/length);fp=-.3*2*np.pi/length*np.sin(2*np.pi*x/length)
fpp=-.3*(2*np.pi/length)**2*np.cos(2*np.pi*x/length)
tau_r=R/np.sqrt(1+mu*mu);history=[];profiles=[];maximum_error=0.;maximum_force=0.;maxwell=0.;minimum_pressure=p
for ls in [0.,.003,.006]:
    A=np.zeros(64);K=A.copy();T=A.copy();Q=A.copy();tau=A.copy();strain=A.copy();amp=0.;cumulative_diss=0.;cumulative_work=0.;previous_grad=0.
    for step in range(1,25):
        time=step*dt;new_amp=.01*max(0.,min(time,.125)-.05);new_K=new_amp*f;delta_K=new_K-K;W=delta_K/dt
        new_A=beta*(A+delta_K);new_Q=2*G*new_A/h;X=-2*G*ls*ls*new_amp*fpp
        if time<=.05+1e-12:new_tau=np.full(64,.99*tau_r*time/.05)
        elif time<=.125+1e-12:
            B=new_Q+X;disc=(1+mu*mu)*R*R-mu*mu*B*B;assert min(disc)>0
            new_tau=(B+np.sqrt(disc))/(1+mu*mu)
        else:new_tau=beta*tau
        de=new_tau/(G*beta)-tau/G+2*delta_K if time<=.125+1e-12 else np.zeros(64)
        new_T=beta*(T+G*de);computed_tau=new_T-2*G*new_A
        maxwell=max(maxwell,float(max(abs(computed_tau-new_tau)))/max(1.,float(max(abs(new_T)))))
        F=computed_tau-new_Q-X;norm=np.sqrt(F*F+mu*mu*computed_tau*computed_tau);margin=norm-R
        sliding=W!=0
        maximum_force=max(maximum_force,float(max(abs(margin[sliding])))/R if any(sliding) else float(max(np.maximum(margin,0)))/R)
        assert max(margin)<=1e-12*max(R,float(max(abs(new_T))))
        assert min(F[sliding])>0 if any(sliding) else True
        minimum_pressure=min(minimum_pressure,float(min(p-abs(computed_tau))))
        bulk=computed_tau*computed_tau/(2*G);cohesive=h*new_Q*new_Q/(2*G)
        db=computed_tau*computed_tau/eta;dc=h*new_Q*new_Q/eta;df=2*F*W
        assert min(df)>=0
        old_energy=(tau*tau+h*Q*Q)/(2*G);change=weights@(bulk+cohesive-old_energy)
        grad=2*G*ls*ls*new_amp*new_amp*(weights@(fp*fp));grad_loss=2*G*ls*ls*(new_amp-amp)**2*(weights@(fp*fp))
        stress_loss=weights@((computed_tau-tau)**2/(2*G));traction_loss=weights@(h*(new_Q-Q)**2/(2*G))
        work=weights@(computed_tau*de);diss=dt*(weights@(db+dc+df));micro_bulk=dt*(weights@(2*X*W))
        error=work-change-(grad-previous_grad)-diss-stress_loss-traction_loss-grad_loss
        weak_error=micro_bulk-(grad-previous_grad)-grad_loss
        maximum_error=max(maximum_error,abs(float(error)),abs(float(weak_error)));assert abs(error)<=1e-10 and abs(weak_error)<=1e-10
        cumulative_diss+=diss;cumulative_work+=work;strain+=de
        history.append({'ell_s_m':ls,'step':step,'time_s':time,'K_amplitude':new_amp,'mean_engineering_shear_strain':float(weights@strain/length),
            'mean_physical_tau_xy_Pa':float(weights@computed_tau/length),'max_abs_Xxy_Pa':float(max(abs(X))),
            'bulk_storage_J_m-2':float(weights@bulk),'cohesive_storage_J_m-2':float(weights@cohesive),
            'gradient_storage_J_m-2':float(grad),'gradient_BE_loss_J_m-2':float(grad_loss),'physical_dissipation_step_J_m-2':float(diss),
            'work_step_J_m-2':float(work),'ledger_error_J_m-2':float(error),'gradient_microforce_work_error_J_m-2':float(weak_error),
            'max_yield_margin_Pa':float(max(margin)),'max_bulk_flow_rate_s-1':float(2*max(abs(W))),
            'cumulative_physical_dissipation_J_m-2':float(cumulative_diss),'cumulative_work_J_m-2':float(cumulative_work)})
        if step in [8,12,20,24]:
            for i in range(64):profiles.append({'ell_s_m':ls,'time_s':time,'x_m':x[i],'weight_m':weights[i],
                'Kxy':new_K[i],'Axy':new_A[i],'Qxy_Pa':new_Q[i],'Txy_auxiliary_Pa':new_T[i],'tau_xy_physical_Pa':computed_tau[i],
                'Xxy_Pa':X[i],'flow_rate_xy_s-1':W[i],'engineering_shear_strain':strain[i],'yield_margin_Pa':margin[i]})
        A,K,T,Q,tau,amp,previous_grad=new_A,new_K,new_T,new_Q,computed_tau,new_amp,grad
for name,rows in [('manufactured-history.csv',history),('manufactured-profiles.csv',profiles)]:
    with (OUT/name).open('w') as stream:
        w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
# Independent affine-K nonzero-microtraction identity. Full tensor metric.
ls=.006;a0=.0005;a1=.0007
E=lambda a:2*G*ls*ls*a*a/length
boundary=4*G*ls*ls*a1*(a1-a0)/length
boundary_error=boundary-(E(a1)-E(a0))-E(a1-a0)
assert abs(boundary_error)<=1e-12
(OUT/'manufactured-checks.json').write_text(json.dumps({'inverse_manufactured_only':True,'forward_field_solve':False,'native_run':False,
 'max_storage_microforce_work_ledger_defect_J_m-2':maximum_error,'force_scaled_defect':maximum_force,'Maxwell_scaled_defect':maxwell,
 'minimum_all_plane_normal_pressure_Pa':minimum_pressure,'affine_nonzero_microtraction_BE_identity_defect_J_m-2':boundary_error,
 'hold_slip_rate_exactly_zero_and_yield_admissible':all(r['max_bulk_flow_rate_s-1']==0 and r['max_yield_margin_Pa']<=0 for r in history if r['time_s']>.125+1e-12),
 'initial_history':'Fresh zero T/A/K/Q; phase=.8 fixed. The known K target determines a different loading for each length.',
 'constant_initial_fracture_storage_J_m-2':8*phi*length,
 'whole_candidate_qualified':False},indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,ax=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
for ls in [0.,.003,.006]:
    data=[r for r in history if r['ell_s_m']==ls];label=f'{ls*1000:g} mm'
    ax[0].plot([r['time_s'] for r in data],[r['mean_engineering_shear_strain'] for r in data],label=label)
    cut=[r for r in profiles if r['ell_s_m']==ls and abs(r['time_s']-.125)<1e-12]
    ax[1].plot([r['x_m'] for r in cut],[r['Xxy_Pa'] for r in cut],label=label)
    ax[2].plot([r['time_s'] for r in data],[r['gradient_storage_J_m-2'] for r in data],label=label)
ax[0].set(xlabel='Time (s)',ylabel='Computed mean engineering shear strain',title='Common K target; computed strain loads')
ax[1].set(xlabel='Cross-strip coordinate x (m)',ylabel=r'Gradient backstress $X_{xy}$ (Pa)',title='Nonuniform signed memory adds a backstress')
ax[2].set(xlabel='Time (s)',ylabel=r'Gradient storage (J m$^{-2}$)',title='Admissible hold retains gradient energy')
for a in [ax[0],ax[2]]:a.axvspan(.125,.15,color='grey',alpha=.12);a.legend(fontsize=8)
fig.suptitle('Manufactured microfree strip: energy check, not a forward solution',fontsize=12)
fig.savefig(OUT/'manufactured-energy.png',dpi=170);plt.close(fig)
print((OUT/'manufactured-checks.json').read_text())
