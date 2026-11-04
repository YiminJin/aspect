#!/usr/bin/env python3
"""Bounded Stage B fixed-profile particle-free history comparison.

See STAGE_B_FIXED_PROTOCOL.md. Cartesian orthonormal Q2, analytically integrated
semidiscrete reaction; independent direct component solution. No ASPECT, mesh
transport, nonlinear mechanics, particle state, or checkpoint conversion.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed'
G, ETA, ELL, HALF, END = 3e6, 1.5e7, .06, .36, 1.7
RATE, V0, AMP, NU = G/ETA, .03, .25, 1.3
THETA = np.deg2rad(37.)
TANGENT = np.array([np.cos(THETA), np.sin(THETA)])
NORMAL = np.array([-TANGENT[1], TANGENT[0]])
S = np.array([TANGENT[0]*NORMAL[0], TANGENT[1]*NORMAL[1],
              .5*(TANGENT[0]*NORMAL[1]+TANGENT[1]*NORMAL[0])])
EXT = np.array([TANGENT[0]**2-NORMAL[0]**2, TANGENT[1]**2-NORMAL[1]**2,
                TANGENT[0]*TANGENT[1]-NORMAL[0]*NORMAL[1]])
INITIAL = G*np.array([.07,.03,.04])


def fields(z):
    chi = np.exp(-(z/ELL)**2)/(ELL*np.sqrt(np.pi))
    shear = .55+.35*np.tanh(z/ELL)
    # v=t U(z)+.04 xi t-.04 z n; total W10=-k/2 for this orientation.
    omega = -.5*shear
    strain = shear[...,None]*S+.04*EXT
    return chi, omega, strain


def slip(time):
    return V0*(1+AMP*np.sin(NU*time))


def memory(time):
    return V0*((1-np.exp(-RATE*time))/RATE
               +AMP*(RATE*np.sin(NU*time)-NU*np.cos(NU*time)+NU*np.exp(-RATE*time))/(RATE**2+NU**2))


def convolution(lam, mu, time):
    """Integral exp(lam*(t-u))*exp(mu*u) du, stable at coincident exponents."""
    delta = np.asarray(mu-lam, dtype=complex)
    small = np.abs(delta*time)<1e-8
    safe = np.where(small, 1., delta)
    quotient = np.expm1(delta*time)/safe
    quotient = np.where(small, time*(1+delta*time/2+(delta*time)**2/6), quotient)
    return np.exp(lam*time)*quotient


def forcing_integrals(lam, time):
    constant = convolution(lam, 0., time)
    sinusoid = (convolution(lam, 1j*NU, time)-convolution(lam, -1j*NU, time))/(2j)
    return constant, sinusoid


def memory_convolution(lam, time):
    den = RATE**2+NU**2
    return (V0/RATE*convolution(lam,0.,time)
            +V0*(-1/RATE+AMP*NU/den)*convolution(lam,-RATE,time)
            +V0*AMP*(-NU-1j*RATE)/(2*den)*convolution(lam,1j*NU,time)
            +V0*AMP*(-NU+1j*RATE)/(2*den)*convolution(lam,-1j*NU,time))


def from_complex(u, mean):
    return np.stack([mean+u.real, mean-u.real, u.imag],axis=-1)


def direct_reference(z, time):
    """Closed physical component equation, independent of T, B, R and DG."""
    chi, omega, strain = fields(np.asarray(z))
    lam = -RATE+2j*omega
    init = (INITIAL[0]-INITIAL[1])/2+1j*INITIAL[2]
    drive = (strain[...,0]-strain[...,1])/2+1j*strain[...,2]
    slip_tensor = (S[0]-S[1])/2+1j*S[2]
    c, sn = forcing_integrals(lam,time)
    u = np.exp(lam*time)*init+2*G*(drive*c-chi*slip_tensor*V0*(c+AMP*sn))
    mean = .5*(INITIAL[0]+INITIAL[1])*np.exp(-RATE*time)+np.zeros_like(z)
    return from_complex(u,mean)


def residual_reference(z, time):
    """Independent integral of R_dot=[W,R]-rR+[W,B], not a difference of answers."""
    chi, omega, _ = fields(np.asarray(z))
    su = (S[0]-S[1])/2+1j*S[2]
    u = 4j*G*omega*chi*su*memory_convolution(-RATE+2j*omega,time)
    return from_complex(u,np.zeros_like(z))


def norm3_squared(a):
    return a[...,0]**2+a[...,1]**2+(a[...,0]+a[...,1])**2+2*a[...,2]**2


def rotate(a, omega, dt):
    mean = .5*(a[...,0]+a[...,1])
    diff = .5*(a[...,0]-a[...,1])
    c,sn = np.cos(2*omega*dt),np.sin(2*omega*dt)
    return np.stack([mean+c*diff-sn*a[...,2],mean-c*diff+sn*a[...,2],sn*diff+c*a[...,2]],axis=-1)


def spin_source(omega,a):
    return np.stack([-2*omega*a[...,2],2*omega*a[...,2],omega*(a[...,0]-a[...,1])],axis=-1)


def temporal(steps):
    z = np.linspace(-4*ELL,4*ELL,257)
    chi,omega,strain = fields(z)
    tau = np.broadcast_to(INITIAL,(len(z),3)).copy()
    trial=tau.copy();residual=np.zeros_like(tau);s=0.
    dt=END/steps;beta=np.exp(-RATE*dt);gain=-np.expm1(-RATE*dt)/RATE
    start=time.perf_counter()
    for i in range(steps):
        v = slip((i+1)*dt)
        # Direct and separated histories are independently evolved. The
        # residual is never assigned from tau_direct or reconstruction closure.
        tau=beta*rotate(tau,omega,dt)+2*G*gain*(strain-chi[:,None]*v*S)
        trial=beta*rotate(trial,omega,dt)+2*G*gain*strain
        s=beta*s+gain*v
        B=2*G*chi[:,None]*s*S
        residual=beta*rotate(residual,omega,dt)+gain*spin_source(omega,B)
    reconstructed=trial-B-residual
    exact=direct_reference(z,END)
    rms=lambda a:float(np.sqrt(np.mean(norm3_squared(a)))/G)
    return dict(steps=steps,dt_s=dt,direct_tensor_scaled=rms(tau-exact),
                separated_tensor_scaled=rms(reconstructed-exact),closure_scaled=rms(reconstructed-tau),
                elapsed_s=time.perf_counter()-start)


def reference_audit():
    z=np.linspace(-3*ELL,3*ELL,17)
    chi,w,D=fields(z)
    # Explicit real component RHS; it does not call the rotation or convolution code.
    def rhs(t,y):
        xx,yy,xy=np.moveaxis(y,-1,0)
        out=np.empty_like(y)
        forcing=D-chi[:,None]*slip(t)*S
        out[:,0]=-2*w*xy-RATE*xx+2*G*forcing[:,0]
        out[:,1]=2*w*xy-RATE*yy+2*G*forcing[:,1]
        out[:,2]=w*(xx-yy)-RATE*xy+2*G*forcing[:,2]
        return out
    solutions=[]
    for steps in (2048,4096):
        y=np.broadcast_to(INITIAL,(len(z),3)).copy();dt=END/steps
        for i in range(steps):
            t=i*dt;k1=rhs(t,y);k2=rhs(t+dt/2,y+dt*k1/2)
            k3=rhs(t+dt/2,y+dt*k2/2);k4=rhs(t+dt,y+dt*k3)
            y+=dt*(k1+2*k2+2*k3+k4)/6
        solutions.append(y)
    error=float(np.max(np.sqrt(norm3_squared(solutions[-1]-direct_reference(z,END))))/G)
    difference=float(np.max(np.sqrt(norm3_squared(solutions[-1]-solutions[0])))/G)
    z_identity=np.linspace(-4*ELL,4*ELL,257)
    chi,omega,strain=fields(z_identity)
    lam=-RATE+2j*omega
    c,_=forcing_integrals(lam,END)
    initial=(INITIAL[0]-INITIAL[1])/2+1j*INITIAL[2]
    drive=(strain[...,0]-strain[...,1])/2+1j*strain[...,2]
    trial=from_complex(np.exp(lam*END)*initial+2*G*drive*c,
                       np.full_like(z_identity,.5*(INITIAL[0]+INITIAL[1])*np.exp(-RATE*END)))
    # R here comes from its own source integral, never from a direct-stress difference.
    reconstructed=trial-2*G*chi[:,None]*memory(END)*S-residual_reference(z_identity,END)
    closure=float(np.max(np.sqrt(norm3_squared(reconstructed-direct_reference(z_identity,END))))/G)
    return dict(component_RK4_steps=[2048,4096],max_reference_error_scaled=error,
                refinement_difference_scaled=difference,independent_integral_closure_scaled=closure)


def basis(points):
    a=legvander(points[:,0],2)*np.sqrt([1,3,5])
    b=legvander(points[:,1],2)*np.sqrt([1,3,5])
    return np.einsum('qi,qj->qij',a,b).reshape(len(points),9)


def rule(order):
    x,w=leggauss(order)
    xx,yy=np.meshgrid(x,x,indexing='ij');wx,wy=np.meshgrid(w,w,indexing='ij')
    points=np.column_stack([xx.ravel(),yy.ravel()])
    return points,(wx*wy).ravel()/4,basis(points)


def mesh(n,order):
    h=2*HALF/n
    centres=-HALF+h*(np.arange(n)+.5)
    xx,yy=np.meshgrid(centres,centres,indexing='ij')
    centres=np.column_stack([xx.ravel(),yy.ravel()])
    p,w,phi=rule(order)
    points=centres[:,None,:]+h/2*p[None,:,:]
    return points,w,phi,h


def project(values,phi,weights):
    return np.einsum('cq,qi,q->ci',values,phi,weights,optimize=True)


def semidiscrete(n,order):
    start=time.perf_counter();points,weights,phi,h=mesh(n,order)
    chi,omega,D=fields(points@NORMAL)
    spin=np.einsum('cq,qi,qj,q->cij',omega,phi,phi,weights,optimize=True)
    frequencies,vectors=np.linalg.eigh(spin)
    lam=-RATE+2j*frequencies
    d=project((D[...,0]-D[...,1])/2+1j*D[...,2],phi,weights)
    su=(S[0]-S[1])/2+1j*S[2]
    hchi=project(chi*su,phi,weights)
    # This is projection of omega*analytic chi, NOT omega times projected chi.
    source=4j*G*project(omega*chi*su,phi,weights)
    initial=np.zeros((n*n,9),complex)
    initial[:,0]=(INITIAL[0]-INITIAL[1])/2+1j*INITIAL[2]
    to_eigen=lambda c:np.einsum('cij,ci->cj',vectors,c,optimize=True)
    from_eigen=lambda c:np.einsum('cij,cj->ci',vectors,c,optimize=True)
    c,sn=forcing_integrals(lam,END)
    T=from_eigen(np.exp(lam*END)*to_eigen(initial)+2*G*to_eigen(d)*c)
    direct=from_eigen(np.exp(lam*END)*to_eigen(initial)
                      +2*G*(to_eigen(d)*c-to_eigen(hchi)*V0*(c+AMP*sn)))
    R=from_eigen(to_eigen(source)*memory_convolution(lam,END))
    mean=np.zeros((n*n,9));mean[:,0]=(INITIAL[0]+INITIAL[1])/2*np.exp(-RATE*END)
    residual_mean=np.zeros_like(mean)
    mass=phi.T@(weights[:,None]*phi)
    eigen_defect=np.max(np.abs(np.einsum('cij,cjk->cik',spin,vectors)-vectors*frequencies[:,None,:]))
    return dict(n=n,order=order,h=h,T=from_complex(T,mean),R=from_complex(R,residual_mean),
                direct=from_complex(direct,mean),D=from_complex(d,np.zeros_like(mean)),mass_defect=float(np.max(np.abs(mass-np.eye(9)))),
                eigen_defect=float(eigen_defect),elapsed_s=time.perf_counter()-start,
                tensor_scalar_dofs_direct=27*n*n,tensor_scalar_dofs_separated=54*n*n,
                coefficient_bytes_direct=int(from_complex(direct,mean).nbytes),
                coefficient_bytes_separated=int(from_complex(T,mean).nbytes+from_complex(R,residual_mean).nbytes))


def evaluate(solution,points,kind):
    n=solution['n'];h=solution['h']
    ij=np.floor((points+HALF)/h).astype(int)
    ij=np.clip(ij,0,n-1);cells=ij[:,0]*n+ij[:,1]
    centres=-HALF+h*(ij+.5)
    phi=basis(2*(points-centres)/h)
    eval_coeff=lambda key:np.einsum('qi,qik->qk',phi,solution[key][cells],optimize=True)
    if kind in ('direct','R','D'):return eval_coeff(kind)
    chi=fields(points@NORMAL)[0]
    B=2*G*chi[:,None]*memory(END)*S
    if kind=='omitted':return eval_coeff('T')-B
    if kind=='wrong_sign':return eval_coeff('T')-B+eval_coeff('R')
    return eval_coeff('T')-B-eval_coeff('R')


def traction(stress):
    t,n=TANGENT,NORMAL
    shear=t[0]*n[0]*stress[...,0]+t[1]*n[1]*stress[...,1]+(t[0]*n[1]+t[1]*n[0])*stress[...,2]
    normal=n[0]**2*stress[...,0]+n[1]**2*stress[...,1]+2*n[0]*n[1]*stress[...,2]
    return shear,normal


def line_points(n,xi,order=12):
    lower=max(min((-HALF-xi*TANGENT[i])/NORMAL[i],(HALF-xi*TANGENT[i])/NORMAL[i]) for i in range(2))
    upper=min(max((-HALF-xi*TANGENT[i])/NORMAL[i],(HALF-xi*TANGENT[i])/NORMAL[i]) for i in range(2))
    knots=[lower,upper]
    for i in range(2):
        knots.extend(z for z in (np.linspace(-HALF,HALF,n+1)-xi*TANGENT[i])/NORMAL[i] if lower<z<upper)
    knots=np.unique(knots);left,right=knots[:-1],knots[1:]
    g,w=leggauss(order)
    z=(left[:,None]+right[:,None])/2+(right-left)[:,None]*g/2
    weights=(right-left)[:,None]*w/2
    points=xi*TANGENT+z.ravel()[:,None]*NORMAL
    return points,z.ravel(),weights.ravel()


def metrics(solution,kind='separated',eval_order=12):
    start=time.perf_counter();n=solution['n']
    points,w,phi,h=mesh(n,eval_order)
    flat=points.reshape(-1,2)
    actual=evaluate(solution,flat,kind).reshape(n*n,-1,3)
    exact=direct_reference(points@NORMAL,END)
    delta=actual-exact
    avg=lambda f:float(np.mean(np.sum(f*w[None,:],axis=1)))
    shear,normal=traction(delta)
    area=(2*HALF)**2
    energy=area*avg(norm3_squared(actual))/(4*G)
    exact_energy=area*avg(norm3_squared(exact))/(4*G)
    weak_shear=[];weak_normal=[]
    for xi in (-.15,0.,.15):
        lp,z,lw=line_points(n,xi)
        da=evaluate(solution,lp,kind)-direct_reference(z,END)
        qs,qn=traction(da);chi=fields(z)[0]
        weak_shear.append(float(np.dot(lw,chi*qs)))
        weak_normal.append(float(np.dot(lw,chi*qn)))
    # Best L2 residual projection under a finer independent integration rule.
    rexact=residual_reference(points@NORMAL,END)
    rcoeff=np.stack([project(rexact[...,k],phi,w) for k in range(3)],axis=-1)
    rbest=np.einsum('qi,cik->cqk',phi,rcoeff,optimize=True)
    ractual=evaluate(solution,flat,'R').reshape(n*n,-1,3)
    projection=float(np.sqrt(avg(norm3_squared(rbest-rexact)))/G)
    dprojected=evaluate(solution,flat,'D').reshape(n*n,-1,3)
    dsource_error=float(np.sqrt(avg(norm3_squared(dprojected-fields(points@NORMAL)[2]))))
    result=dict(n=n,source_order=solution['order'],kind=kind,h_over_ell=h/ELL,
                tensor_scaled=float(np.sqrt(avg(norm3_squared(delta)))/G),
                tensor_relative=float(np.sqrt(avg(norm3_squared(delta))/avg(norm3_squared(exact)))),
                normal_scaled=float(np.sqrt(avg(normal**2))/G),shear_scaled=float(np.sqrt(avg(shear**2))/G),
                zz_scaled=float(np.sqrt(avg((delta[...,0]+delta[...,1])**2))/G),
                tensor_max_scaled=float(np.max(np.sqrt(norm3_squared(delta)))/G),
                R_max_error_scaled=float(np.max(np.sqrt(norm3_squared(ractual-rexact)))/G),
                energy_J_m=energy,reference_energy_J_m=exact_energy,energy_relative_error=abs(energy-exact_energy)/exact_energy,
                weak_shear_error_Pa=max(abs(x) for x in weak_shear),weak_normal_error_Pa=max(abs(x) for x in weak_normal),
                D_source_projection_L2_per_s=dsource_error,
                best_R_projection_scaled=projection,R_tensor_scaled=float(np.sqrt(avg(norm3_squared(ractual-rexact)))/G),
                integration_order=eval_order,metric_elapsed_s=time.perf_counter()-start,
                max_stress_over_G=float(np.max(np.sqrt(norm3_squared(actual)))/G))
    return result


def solution_difference(a,b,order=12):
    points,w,_,_=mesh(a['n'],order);flat=points.reshape(-1,2)
    difference=evaluate(a,flat,'separated')-evaluate(b,flat,'separated')
    sq=norm3_squared(difference).reshape(a['n']**2,-1)
    return float(np.sqrt(np.mean(np.sum(sq*w[None,:],axis=1)))/G)


def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot(output,space,temporal_rows,profile):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    z=np.array([r['z_m'] for r in profile])/ELL
    axs[0,0].plot(z,[r['exact_shear_Pa']/1e6 for r in profile],color='black',label='Exact direct reference')
    for n in (8,64):axs[0,0].plot(z,[r[f'separated_{n}_shear_Pa']/1e6 for r in profile],label=f'Separated Q2, N={n}',lw=1)
    axs[0,0].set(title='A  Shear stress through the oblique fault',xlabel='Normal distance / ell',ylabel='Shear stress (MPa)')
    axs[0,0].legend(fontsize=8)
    for kind in ('direct','separated'):
        rows=[r for r in space if r['kind']==kind]
        axs[0,1].loglog([1/r['n'] for r in rows],[r['tensor_scaled'] for r in rows],'o-',label=f'{kind}: full tensor')
    rows=[r for r in space if r['kind']=='separated']
    axs[0,1].loglog([1/r['n'] for r in rows],[r['best_R_projection_scaled'] for r in rows],'s--',label='Best residual projection')
    axs[0,1].set(title='B  Spatial error · exact semidiscrete time',xlabel='1 / cells per direction',ylabel='Full3D L2 error / (G sqrt(area))')
    axs[0,1].legend(fontsize=8)
    for key,label in [('direct_tensor_scaled','Direct Lie update'),('separated_tensor_scaled','Separated Lie update'),('closure_scaled','Direct / separated difference')]:
        axs[1,0].loglog([r['dt_s'] for r in temporal_rows],[r[key] for r in temporal_rows],'o-',label=label)
    axs[1,0].set(title='C  Temporal error · exact spatial samples',xlabel='Timestep (s)',ylabel='Full3D RMS error / G')
    axs[1,0].legend(fontsize=8)
    axs[1,1].plot(z,[r['exact_R_norm_Pa']/1e6 for r in profile],color='black',label='Exact residual integral')
    for n in (8,64):axs[1,1].plot(z,[r[f'R_{n}_norm_Pa']/1e6 for r in profile],label=f'Evolved residual Q2, N={n}',lw=1)
    axs[1,1].set(title='D  Missing history has a localized tensor profile',xlabel='Normal distance / ell',ylabel='Full3D residual norm (MPa)')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle('Stage B first part · fixed profile, varying total spin, nonzero slip',fontsize=14)
    fig.text(.5,.015,'Cartesian DGQ2, 37° fault · no advection / coupled mechanics · synthetic hypoelastic control',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();reference=reference_audit()
    times=[temporal(n) for n in (80,160,320)]
    space=[];cost=[];solutions={}
    for n in (8,16,32,64):
        sol=semidiscrete(n,8);solutions[n]=sol
        cost.append({k:v for k,v in sol.items() if not isinstance(v,np.ndarray)})
        for kind in ('direct','separated'):space.append(metrics(sol,kind))
        print(f'Space N={n}: direct={space[-2]["tensor_scaled"]:.6g}, separated={space[-1]["tensor_scaled"]:.6g}',flush=True)
    negative=[metrics(solutions[64],kind) for kind in ('omitted','wrong_sign')]
    qref=semidiscrete(8,12)
    quadrature=[]
    for q in (3,4,6):
        sol=semidiscrete(8,q)
        quadrature.append(dict(n=8,source_order=q,reference_order=12,solution_difference_scaled=solution_difference(sol,qref)))
    # Independent evaluation-rule check; only the metric quadrature changes.
    fine=solutions[64]
    check16=metrics(fine,'separated',eval_order=16)
    energy_metric_defect=abs(check16['energy_J_m']-[r for r in space if r['n']==64 and r['kind']=='separated'][0]['energy_J_m'])/check16['reference_energy_J_m']
    z=np.linspace(-4*ELL,4*ELL,513);points=z[:,None]*NORMAL
    exact=direct_reference(z,END);rr=residual_reference(z,END)
    prof=[]
    vals={n:(evaluate(solutions[n],points,'separated'),evaluate(solutions[n],points,'R')) for n in (8,64)}
    for i,zi in enumerate(z):
        row=dict(z_m=float(zi),exact_shear_Pa=float(traction(exact[i])[0]),exact_R_norm_Pa=float(np.sqrt(norm3_squared(rr[i]))))
        for n,(tau,R) in vals.items():
            row[f'separated_{n}_shear_Pa']=float(traction(tau[i])[0]);row[f'R_{n}_norm_Pa']=float(np.sqrt(norm3_squared(R[i])))
        prof.append(row)
    checks=[]
    def gate(name,ok,value,criterion,required=True):checks.append(dict(name=name,passed=bool(ok),value=value,criterion=criterion,required_for_selected_q8=required))
    gate('independent_direct_reference',reference['max_reference_error_scaled']<1e-10 and reference['refinement_difference_scaled']<1e-10,reference,'RK4/analytic and refined-reference differences <1e-10')
    gate('continuous_product_rule_closure',reference['independent_integral_closure_scaled']<1e-12,reference['independent_integral_closure_scaled'],'Independent evolved R integral and direct tensor differ <1e-12')
    gate('Q2_constant_mass_and_eigen',max(r['mass_defect'] for r in cost)<1e-12 and max(r['eigen_defect'] for r in cost)<1e-12,
         {k:max(r[k] for r in cost) for k in ('mass_defect','eigen_defect')},'Both defects <1e-12')
    for key in ('direct_tensor_scaled','separated_tensor_scaled','closure_scaled'):
        ratios=[times[i][key]/times[i+1][key] for i in (0,1)]
        gate('first_order_'+key,all(1.8<x<2.2 for x in ratios) and times[-1][key]<.01,dict(ratios=ratios,finest=times[-1][key]),'Ratios in (1.8,2.2), finest <.01')
    for kind in ('direct','separated'):
        rows=[r for r in space if r['kind']==kind]
        orders=[float(np.log2(rows[i]['tensor_scaled']/rows[i+1]['tensor_scaled'])) for i in (1,2)]
        gate('spatial_order_'+kind,all(o>2 for o in orders),orders,'Last two L2 orders >2')
        last=rows[-1]
        gate('finest_accuracy_'+kind,all(last[k]<1e-3 for k in ('tensor_scaled','normal_scaled','shear_scaled','energy_relative_error')) and max(last['weak_shear_error_Pa'],last['weak_normal_error_Pa'])/G<1e-3,
             {k:last[k] for k in ('tensor_scaled','normal_scaled','shear_scaled','zz_scaled','energy_relative_error','weak_shear_error_Pa','weak_normal_error_Pa')},'Tensor/normal/shear/relative storage <1e-3; weak traction error <1e-3 G')
    qerrors=[r['solution_difference_scaled'] for r in quadrature]
    gate('quadrature_separate_from_space',qerrors[0]>qerrors[1]>qerrors[2] and qerrors[-1]<1e-8,qerrors,'q3>q4>q6 errors; q6 difference from q12 <1e-8',required=False)
    q8_error=solution_difference(solutions[8],qref)
    gate('selected_q8_quadrature',q8_error<1e-8,q8_error,'Selected q8 difference from q12 <1e-8; original q6 failure retained')
    gate('independent_energy_integration',energy_metric_defect<1e-10,energy_metric_defect,'Metric q12 versus q16 relative energy difference <1e-10')
    for row in negative:gate('detect_'+row['kind'],row['tensor_scaled']>1e-3,row['tensor_scaled'],'Incomplete source gives scaled tensor error >1e-3')
    zz_ref=abs(-INITIAL[0]-INITIAL[1])*np.exp(-RATE*END)
    scales=dict(G_Pa=G,eta_Pa_s=ETA,ell_m=ELL,duration_s=END,fault_angle_deg=37,
                square_area_m2=(2*HALF)**2,slip_m_per_s_range=[V0*(1-AMP),V0*(1+AMP)],
                integrated_spin_rad_range=[.1*END,.45*END],initial_zz_Pa=float(-INITIAL[0]-INITIAL[1]),final_zz_Pa=float(-zz_ref),
                maximum_reference_stress_over_G=float(np.max(np.sqrt(norm3_squared(direct_reference(np.linspace(-.5,.5,4097),END))))/G),
                maximum_total_symmetric_strain_norm=float(END*np.sqrt(.9**2/2+2*.04**2)))
    result=dict(scope='Stage B first part only: fixed geometry/profile, no advection, constant matched coefficients; not native mechanics or transport qualification',
                equations='Rdot=[W,R]-rR+[W,B], tau=T-B-R; no residual inferred from direct stress',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,
                parameters=scales,reference_audit=reference,temporal=times,spatial=space,quadrature=quadrature,negative_controls=negative,cost=cost,
                acceptance=checks,initial_q6_candidate_rejected=not next(c['passed'] for c in checks if c['name']=='quadrature_separate_from_space'),
                status='passed_selected_q8' if all(c['passed'] for c in checks if c['required_for_selected_q8']) else 'failed',elapsed_s=time.perf_counter()-began)
    save_csv(output/'time.csv',times);save_csv(output/'space.csv',space);save_csv(output/'quadrature.csv',quadrature);save_csv(output/'cost.csv',cost);save_csv(output/'profiles.csv',prof)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');plot(output,space,times,prof)
    print(json.dumps(dict(status=result['status'],elapsed_s=result['elapsed_s'],checks=checks),indent=2),flush=True)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
