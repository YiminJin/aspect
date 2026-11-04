#!/usr/bin/env python3
"""Remaining Stage B: Eulerian DGQ2 versus independent moving references.

Run the predeclared STAGE_B_MOVING_PROTOCOL.md; no native solver or checkpoints.
The high-order DG reference integrator isolates history-space error. The native
lagged Lie proposal is tested independently at exact material characteristics.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from stage_b_fixed import basis, rule, norm3_squared, save_csv
import moving_model as m

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving'


def mesh(n,order):
    h=2*m.HALF/n;c=-m.HALF+h*(np.arange(n)+.5)
    x,y=np.meshgrid(c,c,indexing='ij');centres=np.column_stack([x.ravel(),y.ravel()])
    p,w,phi=rule(order)
    return centres[:,None,:]+h/2*p[None,:,:],w,phi,h


def derivative_basis(points):
    x,y=points.T
    p=lambda a:np.column_stack([np.ones_like(a),np.sqrt(3)*a,np.sqrt(5)*(.5*(3*a*a-1))])
    dp=lambda a:np.column_stack([np.zeros_like(a),np.sqrt(3)*np.ones_like(a),3*np.sqrt(5)*a])
    dx=np.einsum('qi,qj->qij',dp(x),p(y)).reshape(-1,9)
    dy=np.einsum('qi,qj->qij',p(x),dp(y)).reshape(-1,9)
    return np.stack([dx,dy],axis=-1)


def project(values,phi,weights):
    return np.einsum('qi,cqk,q->cik',phi,values,weights,optimize=True)


class Operator:
    """Fixed Cartesian material-advection weak form and local relaxation.

    Same volume/upwind-face structure as PF4 dg_transport.cc; independent Python
    coefficient fixture. No density/div(v) approximation: div(v)=0 here exactly.
    """
    def __init__(self,n,order,cx):
        self.n,self.order,self.cx=n,order,cx
        points,w,phi,h=mesh(n,order);self.points,self.weights,self.phi,self.h=points,w,phi,h
        unit,_,_=rule(order)
        grad=derivative_basis(unit)*2/h
        velocity=np.empty_like(points);velocity[...,0]=cx+m.SHEAR*points[...,1];velocity[...,1]=m.CY
        self.A=np.einsum('cqd,qid,qj,q->cij',velocity,grad,phi,w,optimize=True)
        self.relax=np.einsum('cq,qi,qj,q->cij',m.rate(points[...,1]),phi,phi,w,optimize=True)
        self.neighbours=[];self.inflow=[]
        line,weights=m.gauss(order)
        centres=points.mean(axis=1)
        all_i,all_j=np.unravel_index(np.arange(n*n),(n,n))
        for axis in (0,1):
            for side in (-1,1):
                unit=np.zeros((order,2));unit[:,axis]=side;unit[:,1-axis]=line
                own=basis(unit);other=unit.copy();other[:,axis]=-side;other=basis(other)
                x=centres[:,None,:]+h/2*unit
                vn=side*(cx+m.SHEAR*x[...,1] if axis==0 else np.full(x.shape[:2],m.CY))
                if np.all(vn>0):
                    self.A-=np.einsum('cq,qi,qj,q->cij',vn,own,own,weights,optimize=True)/(2*h)
                elif np.all(vn<0):
                    idx=all_i if axis==0 else all_j
                    boundary=(idx==0) if side==-1 else (idx==n-1)
                    cells=np.flatnonzero(~boundary)
                    neighbour=cells+side*(n if axis==0 else 1)
                    block=-np.einsum('cq,qi,qj,q->cij',vn[cells],own,other,weights,optimize=True)/(2*h)
                    self.neighbours.append((cells,neighbour,block))
                    cells=np.flatnonzero(boundary)
                    lift=-np.einsum('cq,qi,q->ciq',vn[cells],own,weights,optimize=True)/(2*h)
                    self.inflow.append((cells,x[cells],lift))
                else:
                    raise ValueError('Protocol selected single-sign fluxes; sign-changing face requires a split')
        self.local=self.A-self.relax
        mass=phi.T@(w[:,None]*phi)
        self.mass_defect=float(np.max(np.abs(mass-np.eye(9))))

    def advection(self,coefficients,boundary_values):
        result=self.A@coefficients
        for cells,neighbour,block in self.neighbours:result[cells]+=block@coefficients[neighbour]
        for (cells,points,lift),values in zip(self.inflow,boundary_values):result[cells]+=lift@values
        return result

    def audit(self):
        # Three independent analytic polynomials:1,x,y. No physical-reference
        # code supplies their derivatives or boundary data.
        values=np.stack([np.ones(self.points.shape[:2]),self.points[...,0],self.points[...,1]],axis=-1)
        coeff=project(values,self.phi,self.weights)
        boundary=[np.stack([np.ones(p.shape[:2]),p[...,0],p[...,1]],axis=-1) for _,p,_ in self.inflow]
        actual=self.advection(coeff,boundary)
        expected=np.zeros_like(values);expected[...,1]=-(self.cx+m.SHEAR*self.points[...,1]);expected[...,2]=-m.CY
        difference=actual-project(expected,self.phi,self.weights)
        return dict(polynomial_advection_defect=float(np.max(np.abs(difference))),mass_defect=self.mass_defect)

    @lru_cache(maxsize=4)
    def forcing(self,t):
        values=m.sources(self.points.reshape(-1,2),float(t),self.cx)[0].reshape(self.n*self.n,-1,9)
        return project(values,self.phi,self.weights)

    @lru_cache(maxsize=4)
    def boundary(self,t):
        result=np.zeros((self.n*self.n,9,9))
        for cells,p,lift in self.inflow:
            values=m.reference(p.reshape(-1,2),float(t),self.cx,32).reshape(len(cells),self.order,9)
            result[cells]+=lift@values
        return result

    def rhs(self,t,state):
        result=self.local@state
        for cells,neighbour,block in self.neighbours:result[cells]+=block@state[neighbour]
        tensor=state.reshape(self.n*self.n,9,3,3)
        spin=m.spin_source(m.OMEGA,tensor).reshape(state.shape)
        return result+spin+self.forcing(float(t))+self.boundary(float(t))


def solve(n,order=8,cx=.24,steps=None):
    start=time.perf_counter();op=Operator(n,order,cx);audit=op.audit()
    count=int(np.ceil(m.END*5*(abs(cx)+m.SHEAR*m.HALF+m.CY)/op.h/.25)) if steps is None else steps
    dt=m.END/count
    initial=m.initial(op.points)
    state=project(np.concatenate([initial,initial,np.zeros_like(initial)],axis=-1),op.phi,op.weights)
    for i in range(count):
        t=i*dt;k1=op.rhs(t,state);k2=op.rhs(t+dt/2,state+dt*k1/2)
        k3=op.rhs(t+dt/2,state+dt*k2/2);k4=op.rhs((i+1)*dt,state+dt*k3)
        state+=dt*(k1+2*k2+2*k3+k4)/6
    return dict(n=n,order=order,cx=cx,steps=count,dt_s=dt,coefficients=state,
                solve_elapsed_s=time.perf_counter()-start,operator_audit=audit,
                direct_coefficients=27*n*n,separated_generic_coefficients=54*n*n)


def evaluate(solution,points):
    n=solution['n'];h=2*m.HALF/n
    ij=np.floor((points+m.HALF)/h).astype(int);ij=np.clip(ij,0,n-1)
    centres=-m.HALF+h*(ij+.5)
    return np.einsum('qi,qik->qk',basis(2*(points-centres)/h),solution['coefficients'][ij[:,0]*n+ij[:,1]],optimize=True)


def metrics(solution,order=12):
    start=time.perf_counter();n=solution['n'];p,w,phi,h=mesh(n,order);flat=p.reshape(-1,2)
    values=evaluate(solution,flat)
    ref=m.reference(flat,m.END,solution['cx'],32)
    actual=[values[:,:3],m.reconstruct(values,flat,m.END,solution['cx'])]
    exact=ref[:,:3].reshape(n*n,-1,3)
    average=lambda f:float(np.mean(np.sum(f*w[None,:],axis=1)))
    rref=ref[:,6:9].reshape(n*n,-1,3)
    bestR=project(rref,phi,w);bestR=np.einsum('qi,cik->cqk',phi,bestR,optimize=True)
    bestTau=project(exact,phi,w);bestTau=np.einsum('qi,cik->cqk',phi,bestTau,optimize=True)
    rows=[]
    for kind,a in zip(('direct','separated'),actual):
        a=a.reshape(n*n,-1,3);delta=a-exact;shear,normal=m.tractions(delta,m.END)
        E=(2*m.HALF)**2*average(norm3_squared(a))/(4*m.G)
        Er=(2*m.HALF)**2*average(norm3_squared(exact))/(4*m.G)
        rows.append(dict(n=n,order=solution['order'],cx=solution['cx'],kind=kind,steps=solution['steps'],
            tensor_scaled=float(np.sqrt(average(norm3_squared(delta)))/m.G),
            tensor_max_scaled=float(np.max(np.sqrt(norm3_squared(delta)))/m.G),
            normal_scaled=float(np.sqrt(average(normal**2))/m.G),shear_scaled=float(np.sqrt(average(shear**2))/m.G),
            zz_scaled=float(np.sqrt(average((delta[...,0]+delta[...,1])**2))/m.G),
            energy_J_m=E,reference_energy_J_m=Er,energy_relative_error=abs(E-Er)/Er,
            best_R_projection_scaled=float(np.sqrt(average(norm3_squared(bestR-rref)))/m.G),
            best_tau_projection_scaled=float(np.sqrt(average(norm3_squared(bestTau-exact)))/m.G),
            metric_elapsed_s=time.perf_counter()-start))
    return rows


def difference(a,b):
    p,w,_,_=mesh(a['n'],12);flat=p.reshape(-1,2)
    av,bv=evaluate(a,flat),evaluate(b,flat)
    direct=av[:,:3]-bv[:,:3]
    separated=(av[:,3:6]-av[:,6:9])-(bv[:,3:6]-bv[:,6:9])
    rms=lambda x:float(np.sqrt(np.mean(np.sum(norm3_squared(x).reshape(a['n']**2,-1)*w,axis=1)))/m.G)
    return dict(direct=rms(direct),separated=rms(separated))


def characteristic_lie(steps):
    z=np.linspace(-4*m.ELL,4*m.ELL,257)
    material=z[:,None]*m.N0+(.12*np.sin(4*z))[:,None]*m.T0
    tau=m.initial(material);state=np.concatenate([tau,tau,np.zeros_like(tau)],axis=1)
    dt=m.END/steps
    for j in range(steps):
        t=(j+1)*dt;points=m.position(material,t,.24)
        rate=m.rate(points[:,1]);beta=np.exp(-rate*dt);gain=-np.expm1(-rate*dt)/rate
        state=m.propagate(state,dt,rate*dt)+gain[:,None]*m.sources(points,t,.24)[0]
    points=m.position(material,m.END,.24);exact=m.reference(points,m.END,.24,64)[:,:3]
    separated=m.reconstruct(state,points,m.END,.24)
    rms=lambda x:float(np.sqrt(np.mean(norm3_squared(x)))/m.G)
    return dict(steps=steps,dt_s=dt,direct=rms(state[:,:3]-exact),separated=rms(separated-exact),closure=rms(separated-state[:,:3]))


def reference_audits():
    p=np.array([(x,y) for x in (-.25,0.,.25) for y in (-.12,-.04,.02,.1)])
    refs=[m.reference(p,m.END,.24,q) for q in (32,48,64)]
    err=lambda a:float(np.max(np.sqrt(norm3_squared(a.reshape(-1,3))))/m.G)
    closure=err(refs[-1][:,:3]-m.reconstruct(refs[-1],p,m.END,.24))
    # Real component RK4 along exact material paths is separate from propagator.
    material=m.material_coordinates(p,m.END,.24);dt=m.END/2048
    y=m.initial(material)
    def rhs(t,y):
        at=m.position(material,t,.24);f=m.sources(at,t,.24)[0][:,:3];r=m.rate(at[:,1])
        xx,yy,xy=y.T
        return np.column_stack([-2*m.OMEGA*xy-r*xx+f[:,0],2*m.OMEGA*xy-r*yy+f[:,1],m.OMEGA*(xx-yy)-r*xy+f[:,2]])
    for j in range(2048):
        t=j*dt;k1=rhs(t,y);k2=rhs(t+dt/2,y+dt*k1/2);k3=rhs(t+dt/2,y+dt*k2/2);k4=rhs(t+dt,y+dt*k3)
        y+=dt*(k1+2*k2+2*k3+k4)/6
    # Independent finite coordinate differences for the geometry/profile jets.
    t=.43;x=np.array([[.13,-.03],[-.1,.08],[.2,.02]])
    velocity=np.column_stack([.24+m.SHEAR*x[:,1],np.full(len(x),m.CY)])
    eps=1e-6;before=m.kinematics(x-eps*velocity,t-eps,.24);after=m.kinematics(x+eps*velocity,t+eps,.24);f=m.kinematics(x,t,.24)
    alpha_error=float(np.max(np.abs((after['alpha']-before['alpha'])/(2*eps)-f['alpha_dot'])))
    chi_error=float(np.max(np.abs((after['chi']-before['chi'])/(2*eps)-f['stretch']*f['chi']))*m.ELL)
    tangent,normal,_,_,turn,_,_=m.geometry(t)
    normal_error=float(np.linalg.norm((m.geometry(t+eps)[1]-m.geometry(t-eps)[1])/(2*eps)+turn*tangent))
    # Profile normalization/material width, and actual bad Eulerian-max trail.
    profile=[]
    for t in (0.,m.END/2,m.END):
        tangent,normal,g,*_=m.geometry(t);q,w=m.gauss(96)
        z=q*8*m.ELL/g;points=m.centre(t,.24)+z[:,None]*normal
        chi=m.kinematics(points,t,.24)['chi']
        profile.append(dict(time_s=t,g=g,width_m=m.ELL/g,normalization=float(np.dot(w*8*m.ELL/g,chi))))
    c0,c1=m.surface_coefficients(.4);d0,d1=m.surface_coefficients(.4,128)
    return dict(gauss_differences_scaled=[err(refs[0]-refs[1]),err(refs[1]-refs[2])],
                direct_RK4_error_scaled=err(y-refs[-1][:,:3]),continuous_closure_scaled=closure,
                alpha_derivative_error=alpha_error,profile_derivative_scaled_error=chi_error,
                normal_derivative_error=normal_error,surface_projection_96_128_defect=max(abs(c0-d0),abs(c1-d1)),profiles=profile)


def constitutive_and_negative_audits():
    # Sources integrated independently, never reconstructed differences.
    p,w,_,_=mesh(16,12);flat=p.reshape(-1,2)
    refs,terms=m.reference(flat,m.END,.24,48,True)
    mean=lambda x:float(np.sqrt(np.mean(np.sum(norm3_squared(x).reshape(16*16,-1)*w,axis=1)))/m.G)
    negatives={name:mean(terms[:,3*i:3*i+3]) for i,name in enumerate(('omit_geometry_profile','omit_extension_gradient','omit_rate_mismatch'))}
    material=np.array([[0.,-.06],[.1,-.04],[-.1,-.02]])
    old=m.initial(material);initial_energy=norm3_squared(old)/(4*m.G)
    attenuation=m.accumulated_rate(material[:,1],0.,m.END)
    end_energy=initial_energy*np.exp(-2*attenuation)
    q,wq=m.gauss(64);loss=np.zeros(3)
    for t,weight in zip(.5*m.END*(q+1),.5*m.END*wq):
        y=material[:,1]+m.CY*t
        psi=initial_energy*np.exp(-2*m.accumulated_rate(material[:,1],0.,t))
        loss+=weight*2*m.rate(y)*psi
    homogeneous=m.propagate(old,m.END,m.rate(material[:,1])*m.END)
    changing=m.propagate(old,m.END,attenuation)
    return dict(negative_source_errors_scaled=negatives,
                relaxation_energy_relative_defect=float(np.max(np.abs(end_energy+loss-initial_energy)/initial_energy)),
                homogeneous_eta_error_scaled=float(np.max(np.sqrt(norm3_squared(homogeneous-changing)))/m.G),
                path_y_start_m=material[:,1].tolist(),path_y_end_m=(material[:,1]+m.CY*m.END).tolist())


def plots(output,space,times,profiles):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for kind in ('direct','separated'):
        r=[x for x in space if x['kind']==kind]
        axs[0,0].loglog([1/x['n'] for x in r],[x['tensor_scaled'] for x in r],'o-',label=kind)
        axs[0,1].loglog([x['dt_s'] for x in times],[x[kind] for x in times],'o-',label=kind)
    axs[0,0].set(title='A  Eulerian DGQ2 spatial convergence',xlabel='1 / cells per direction',ylabel='Full3D L2 error / (G sqrt(area))')
    axs[0,1].set(title='B  Independent characteristic Lie time test',xlabel='Timestep (s)',ylabel='Full3D RMS error / G')
    for key,label in [('chi_initial','Initial profile'),('chi_final','Transported / deformed profile'),('chi_bad_max','Incorrect Eulerian maximum')]:
        axs[1,0].plot([x['line_m'] for x in profiles],[x[key] for x in profiles],label=label)
    axs[1,0].set(title='C  Material profile through a fixed observation line',xlabel='Distance on initial normal (m)',ylabel='Localization (1/m)')
    axs[1,1].plot([x['line_m'] for x in profiles],[x['exact_shear_Pa']/1e6 for x in profiles],color='black',label='Independent direct reference')
    for kind in ('direct','separated'):
        axs[1,1].plot([x['line_m'] for x in profiles],[x[kind+'_shear_Pa']/1e6 for x in profiles],label=kind)
    axs[1,1].set(title='D  Transported normal-section shear stress',xlabel='Distance on initial normal (m)',ylabel='Shear stress (MPa)')
    for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Remaining Stage B · material shear/translation, evolving profile and layered viscosity',fontsize=13)
    fig.text(.5,.015,'Independent characteristic reference · prescribed surface memory / motion · native mechanics and lifecycle remain Stage C',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();audits=reference_audits();negatives=constitutive_and_negative_audits()
    print('Reference/geometry/coefficient audits complete',flush=True)
    temporal=[characteristic_lie(n) for n in (80,160,320)]
    solutions={};space=[];cost=[]
    for n in (8,16,32):
        sol=solve(n);solutions[n]=sol;rows=metrics(sol);space.extend(rows)
        cost.append({k:v for k,v in sol.items() if k!='coefficients'})
        print(f'N{n}, steps{sol["steps"]}, cost{sol["solve_elapsed_s"]:.2f}s: direct {rows[0]["tensor_scaled"]:.5g}, separated {rows[1]["tensor_scaled"]:.5g}',flush=True)
    half=solve(16,steps=2*solutions[16]['steps']);time_difference=difference(solutions[16],half)
    print('Half-step check complete',flush=True)
    qref=solve(8,12,steps=solutions[8]['steps']);quadrature=[]
    for q in (4,6,8):
        sol=solutions[8] if q==8 else solve(8,q,steps=solutions[8]['steps'])
        quadrature.append(dict(order=q,**difference(sol,qref)))
    reverse=solve(16,cx=-.24);reverse_metrics=metrics(reverse)
    print('Quadrature and reverse-motion checks complete',flush=True)
    grid=np.linspace(-.32,.32,513);points=grid[:,None]*m.N0
    f0=m.kinematics(points,0.,.24);f1=m.kinematics(points,m.END,.24)
    sampled=evaluate(solutions[32],points);ref=m.reference(points,m.END,.24,64)
    reconstructed=m.reconstruct(sampled,points,m.END,.24)
    profiles=[dict(line_m=float(grid[i]),chi_initial=float(f0['chi'][i]),chi_final=float(f1['chi'][i]),
                   chi_bad_max=float(max(f0['chi'][i],f1['chi'][i])),exact_shear_Pa=float(m.tractions(ref[i,:3],m.END)[0]),
                   direct_shear_Pa=float(m.tractions(sampled[i,:3],m.END)[0]),separated_shear_Pa=float(m.tractions(reconstructed[i],m.END)[0])) for i in range(len(grid))]
    checks=[]
    def gate(name,ok,value,bound):checks.append(dict(name=name,passed=bool(ok),value=value,criterion=bound))
    gate('independent_reference',max(audits['gauss_differences_scaled']+[audits['direct_RK4_error_scaled'],audits['continuous_closure_scaled']])<1e-10,audits,'All independent tensor/reference differences <1e-10G')
    gate('geometry_and_profile',max(audits[k] for k in ('alpha_derivative_error','profile_derivative_scaled_error','normal_derivative_error'))<2e-9 and max(abs(x['normalization']-1) for x in audits['profiles'])<1e-10,audits['profiles'],'Geometric derivative <2e-9; normalized profile <1e-10')
    gate('surface_projection',audits['surface_projection_96_128_defect']<1e-12,audits['surface_projection_96_128_defect'],'Rate-projection q96/q128 difference <1e-12/s')
    for sol in cost:
        gate('operator_'+str(sol['n']),sol['operator_audit']['polynomial_advection_defect']<1e-11 and sol['operator_audit']['mass_defect']<1e-12,sol['operator_audit'],'Constant/linear advection <1e-11; mass <1e-12')
    for kind in ('direct','separated'):
        rows=[r for r in space if r['kind']==kind]
        orders=[float(np.log2(rows[i]['tensor_scaled']/rows[i+1]['tensor_scaled'])) for i in (0,1)]
        gate('spatial_'+kind,all(v>2 for v in orders) and all(rows[-1][key]<1e-3 for key in ('tensor_scaled','normal_scaled','shear_scaled','energy_relative_error')),dict(orders=orders,finest=rows[-1]),'Both orders >2, finest tensor/normal/shear/storage <1e-3')
        ratios=[temporal[i][kind]/temporal[i+1][kind] for i in (0,1)]
        gate('Lie_'+kind,all(1.8<r<2.2 for r in ratios) and temporal[-1][kind]<.01,dict(ratios=ratios,finest=temporal[-1][kind]),'Ratios1.8–2.2, finest <.01G')
        scale=[r['tensor_scaled'] for r in space if r['n']==16 and r['kind']==kind][0]
        gate('spatial_time_isolation_'+kind,time_difference[kind]<.1*scale,time_difference[kind],'Half dt solution change <10% of N16 spatial error')
        errors=[q[kind] for q in quadrature]
        gate('quadrature_'+kind,errors[0]>errors[1]>errors[2] and errors[-1]<1e-8,errors,'q4>q6>q8 errors vsq12, finest <1e-8G')
    gate('reverse_translation',max(x['tensor_scaled'] for x in reverse_metrics)<.01,reverse_metrics,'N16 reversed horizontal motion tensor RMS <.01G')
    for key,value in negatives['negative_source_errors_scaled'].items():gate(key,value>1e-5,value,'Omitted source leaves continuous tensor RMS >1e-5G')
    gate('changing_eta_relaxation',negatives['relaxation_energy_relative_defect']<1e-10 and negatives['homogeneous_eta_error_scaled']>1e-4,negatives,'Relaxation storage+loss closes <1e-10; wrong homogeneous eta defect >1e-4G')
    # Include actual history-domain and coefficient ranges in the selected test.
    p,_,_,h=mesh(32,8);flat=p.reshape(-1,2)
    min_rate=1e9;max_label=0.;max_spin=abs(m.OMEGA)*m.END
    for t in (0.,m.END/2,m.END):
        f=m.kinematics(flat,t,.24);min_rate=min(min_rate,float(np.min(f['rg'])));max_label=max(max_label,float(np.max(np.abs(f['alpha']))))
    gate('positive_surface_rate_and_label_domain',min_rate>0 and max_label<m.ALPHA_LIMIT,dict(min_rate=min_rate,max_abs_label=max_label),'All sampled r_Gamma>0 and |alpha|<1.2m')
    scales=dict(G_Pa=m.G,ell_initial_m=m.ELL,ell_final_m=m.ELL/m.geometry(m.END)[2],duration_s=m.END,
                velocity_cx_m_s=[.24,-.24],cy_m_s=m.CY,rate_range_per_s=[.2,.8],eta_range_Pa_s=[m.G/.8,m.G/.2],spin_rad=max_spin,
                symmetric_strain_norm=float(m.END*np.sqrt(m.SHEAR**2/2)),centre_displacement_m=m.centre(m.END,.24).tolist(),
                displacement_fine_cells=(m.centre(m.END,.24)/h).tolist(),displacement_over_initial_ell=(m.centre(m.END,.24)/m.ELL).tolist(),
                max_reference_stress_over_G=float(np.max(np.sqrt(norm3_squared(ref[:,:3])))/m.G))
    result=dict(scope='Rest of Stage B only: prescribed moving-interface complete particle-free history comparison; no native integration',
                command=sys.argv,parameters=scales,reference=audits,negative_controls=negatives,temporal=temporal,spatial=space,
                time_isolation=time_difference,quadrature=quadrature,reverse=reverse_metrics,cost=cost,acceptance=checks,
                status='passed' if all(c['passed'] for c in checks) else 'failed',elapsed_s=time.perf_counter()-start,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('moving_model.py'))})
    for name,rows in [('space',space),('time',temporal),('quadrature',quadrature),('profiles',profiles)]:save_csv(out/(name+'.csv'),rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');plots(out,space,temporal,profiles)
    print(json.dumps(dict(status=result['status'],elapsed_s=result['elapsed_s'],failed=[c for c in checks if not c['passed']]),indent=2),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
