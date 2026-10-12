#!/usr/bin/env python3
"""One continuum seeded-H initialization and unchanged exact-bulk P2 controls."""
import argparse
import hashlib
import resource
import time

import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy import sparse

import asymmetric_branch as old
from total_energy_step import ELL, M, h, dh

HERE=old.HERE
E=HERE/'matched_initialization_evidence'
A,G,ETA=old.A,old.G,old.ETA
TARGET=old.TARGET
CORE=.6
ACTIVATION=1e-6  # Existing fixture's seeded-H branch, not an evolution mask.


class Seed:
    def __init__(self,points):
        theta=np.linspace(0,np.pi/2,points)
        phi=CORE*np.cos(theta)**2;hc=h(CORE)
        denominator=hc*phi-CORE*h(phi)
        integrand=np.zeros_like(theta)
        integrand[1:-1]=2*ELL*CORE*np.sin(theta[1:-1])*np.cos(theta[1:-1])*np.sqrt(hc/denominator[1:-1])
        integrand[0]=2*ELL*np.sqrt(CORE*hc/(CORE*dh(CORE)-hc))
        integrand[-1]=2*ELL*np.sqrt(CORE*hc/(hc-CORE*dh(0.)))
        self.distance=cumulative_trapezoid(integrand,theta,initial=0.)
        self.phi=phi
        self.cut=float(np.interp(np.arccos(np.sqrt(ACTIVATION/CORE)),theta,self.distance))
    def value(self,distance):
        return np.interp(abs(distance),self.distance,self.phi,right=0.)
    def history(self,distance):
        phi=self.value(distance)
        return np.where(phi>ACTIVATION,A*CORE/h(CORE)*(1+h(phi))**2,.5)


def reference():
    E.mkdir(exist_ok=False)
    # Preserve the complete previous selected evidence and source baseline.
    manifest=old.read(old.E/'preservation.json')
    paths=set(old.ROOT/name for name in manifest)
    paths.update(p for p in old.E.rglob('*') if p.is_file())
    paths.update(HERE/name for name in ['asymmetric_branch.py','assess_asymmetric_branch.py','probe_asymmetric_release.py'])
    old.write(E/'preservation.json',{str(p.relative_to(old.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    seed=Seed(400001);fine=Seed(800001)
    x=np.unique(np.r_[np.linspace(0,1,401),np.linspace(.98,1,101)])
    cut=seed.cut;outside=.5-cut
    def ode(t,z):
        inner=cut*t;outer=cut+outside*t
        Hin=A*CORE/h(CORE)*(1+h(seed.value(inner)))**2
        def second(phi,H):return (1-H/A*dh(phi)/(1+h(phi))**2)/(2*ELL**2)
        return np.array([cut*z[1],cut*second(z[0],Hin),outside*z[3],outside*second(z[2],.5)])
    def bc(left,right):return np.array([left[1],right[3],right[0]-left[2],right[1]-left[3]])
    def guess(t):
        pi=seed.value(cut*t);po=seed.value(cut+outside*t)
        def slope(phi):return -np.sqrt(np.maximum(phi-CORE*h(phi)/h(CORE),0.))/ELL
        return np.array([pi,slope(pi),po,slope(po)])
    solutions=[]
    for tolerance in [1e-9,1e-10]:
        result=solve_bvp(ode,bc,x,guess(x),tol=tolerance,max_nodes=40000)
        assert result.success,(result.message,max(result.rms_residuals))
        solutions.append(result)
    # Save piecewise cubic coefficients of the one shared reference, not
    # separate mesh-specific phase equilibria or particle-sampled H fields.
    b=solutions[-1].sol
    np.savez_compressed(E/'reference.npz',knots=b.x,coefficients=b.c,cut=cut)
    y=np.linspace(0,1,65537);coarse=evaluate_reference(y,solutions[0].sol,cut);finephi=evaluate_reference(y,b,cut)
    raw=evaluate_reference(y,b,cut,clip=False)
    old.write(E/'reference.json',dict(cut=cut,seed_points=400001,
         seed_refinement_error=float(max(abs(seed.value(y-.5)-fine.value(y-.5)))),
         BVP_refinement_error=float(max(abs(coarse-finephi))),
         maximum_continuum_residual=float(max(solutions[-1].rms_residuals)),
         nonnegative_initialization_correction=float(max(abs(finephi-raw))),
         core=float(finephi[len(y)//2]),minimum_raw=float(min(raw)),
         boundary_residual=float(max(abs(bc(solutions[-1].y[:,0],solutions[-1].y[:,-1])))),
         meaning='Common continuum seeded-H stationary definition; tiny nonnegativity correction if needed, protected unloaded seed thereafter'))
    print(old.json.dumps(old.read(E/'reference.json')),flush=True)


def evaluate_reference(y,polynomial=None,cut=None,clip=True):
    if polynomial is None:
        from scipy.interpolate import PPoly
        f=np.load(E/'reference.npz');polynomial=PPoly.construct_fast(f['coefficients'],f['knots'],axis=1);cut=float(f['cut'])
    distance=abs(np.asarray(y)-.5)
    inner=distance<=cut;out=np.empty_like(distance)
    out[inner]=polynomial(distance[inner]/cut)[0]
    out[~inner]=polynomial((distance[~inner]-cut)/(.5-cut))[2]
    return np.maximum(out,0.) if clip else out


def check_reference():
    from scipy.interpolate import PPoly
    f=np.load(E/'reference.npz');p=PPoly.construct_fast(f['coefficients'],f['knots'],axis=1)
    seed=Seed(400001);fine=Seed(800001);cut=fine.cut;outside=.5-cut
    x=np.unique(np.r_[np.linspace(0,1,401),np.linspace(.98,1,101)])
    def ode(t,z):
        H=A*CORE/h(CORE)*(1+h(fine.value(cut*t)))**2
        second=lambda phi,H:(1-H/A*dh(phi)/(1+h(phi))**2)/(2*ELL**2)
        return np.array([cut*z[1],cut*second(z[0],H),outside*z[3],outside*second(z[2],.5)])
    def bc(left,right):return np.array([left[1],right[3],right[0]-left[2],right[1]-left[3]])
    result=solve_bvp(ode,bc,x,p(x),tol=1e-10,max_nodes=40000);assert result.success
    y=np.linspace(0,1,65537);phi=evaluate_reference(y);guess=seed.value(y-.5)
    error=float(max(abs(phi-evaluate_reference(y,result.sol,cut))))
    assert error<1e-10 and min(phi)>=0 and max(phi)<1
    j=int(np.argmax(abs(phi-guess)));record=old.read(E/'reference.json')
    record.update(maximum_relaxation_change_from_physical_seed=float(max(abs(phi-guess))),maximum_change_y=float(y[j]),
        phi_at_H_transition=float(evaluate_reference(np.array([.5+seed.cut]))[0]),outside_H_Pa=.5,
        doubled_seed_stationary_field_error=error)
    old.write(E/'reference.json',record)
    initial_checks=[]
    for n in [511,512,1024]:
        model=Exact(n);fields=np.load(E/f'n{n}/fields.npz')
        assert np.array_equal(model.r.initial,fields['phase'][0]) and np.all(model.r.rq==0) and model.r.Z==0
        np.savez_compressed(E/f'common-initial-H-n{n}.npz',particle_y=model.r.initial_particle_y,H=model.r.initial_particle_H)
        initial_checks.append(dict(n=n,same_initial_phase=True,same_unloaded_state=True,
            shared_H_definition=True,unloaded_KKT=model.r.startup_defect))
    old.write(E/'direct-initialization-equivalence.json',initial_checks)
    print(old.json.dumps(record),flush=True)


def integral_derivatives(r,phi):
    """Actual analytic I/J plus the existing 24-point local I Hessian."""
    I,J=r.direct(phi);z,w=np.polynomial.legendre.leggauss(24);s=(z+1)/2
    ph=(1-s)*phi[:-1,None]+s*phi[1:,None]
    hp=M*(12/(1-ph)**4-6/(1-ph)**3)
    left=r.dy/2*np.sum(w*hp*(1-s)**2,axis=1)
    right=r.dy/2*np.sum(w*hp*s*s,axis=1)
    cross=r.dy/2*np.sum(w*hp*s*(1-s),axis=1)
    H=sparse.diags([cross,np.r_[left,0.]+np.r_[0.,right],cross],[-1,0,1]).toarray()
    return I,J,H


class SharedReference(old.core.Reference):
    def initialize(self):
        # Initialize directly from the shared definition. No particle-sampled
        # relaxation is performed and then discarded in these controls.
        self.initial_particle_y=(np.arange(3*self.n)+.5)*self.dy/3+self.y[0]
        self.initial_particle_H=Seed(400001).history(self.initial_particle_y-.5)
        x=evaluate_reference(self.y)
        self.startup_defect=float(max(np.maximum(-(self.flinear+self.F@x)/A,0.)))
        assert self.startup_defect<=TARGET
        return x


class Exact(old.ExactCoupled):
    def __init__(self,n):
        self.r=SharedReference(n,1.,6,exact_bulk=True)
        self.evaluations=0;self.last=None
    def hessian(self,phi):
        """Algebraic reduction of the old exact-bulk Schur operator, same rule.

        J depends on C=W chi and I, plus quadratic fracture energy. C is
        retained at its sampled value; never silently set it to one.
        """
        self.evaluate(phi);r=self.r;I,J,HI=integral_derivatives(r,phi)
        ph=r.B@phi;K=float(r.w@h(ph));Kp=np.asarray(r.B.T@(r.w*dh(ph)))
        hpp=M*(12/(1-ph)**4-6/(1-ph)**3)
        HK=(r.B.T@sparse.diags(r.w*hpp)@r.B).toarray()
        C=K/I;Cp=Kp/I-K*J/I**2
        HC=HK/I-(np.outer(Kp,J)+np.outer(J,Kp))/I**2-K*HI/I**2+2*K*np.outer(J,J)/I**3
        Z=r.Z;e=self.beta*G*(r.w@self.Ra)+self.kappa*.256*(self.time-self.dt/2)
        T=e-Z*C;gc=-Z*T/G;gi=-Z*Z/(2*G*I*I)
        cc=Z*Z/G;ci=0.;ii=Z*Z/(G*I**3)
        if self.last[3]['V']>0:
            denominator=C*C+1/I
            cc-=(e-2*Z*C)**2/(G*denominator)
            ci=-(e-2*Z*C)*Z/(G*I**2*denominator)
            ii-=Z*Z/(G*I**4*denominator)
        return (r.F+gc*HC+gi*HI+cc*np.outer(Cp,Cp)+ci*(np.outer(Cp,J)+np.outer(J,Cp))+ii*np.outer(J,J))/A


def equivalence():
    p=Exact(512);q=old.ExactCoupled(512)
    f=np.load(old.E/'n512-d1024-exact/search/incoming.npz')
    for model in [p,q]:model.incoming(f['phi'],f['R'],float(f['Z']),float(f['slip']),float(f['time']),float(f['dt']))
    x=f['symmetric'];H=p.hessian(x);HH=q.hessian(x)
    error=float(max(abs(H-HH).ravel()));assert error<1e-10
    v=np.sin(np.arange(len(x))*1.234)*(x>.001);v/=np.linalg.norm(v)
    differences=[]
    for eps in [1e-5,2e-6]:
        fd=(p.evaluate(x+eps*v)[1]-p.evaluate(x-eps*v)[1])/(2*eps)
        differences.append(float(max(abs(fd-H@v))))
    assert min(differences)<1e-7
    old.write(E/'operator-equivalence.json',dict(maximum_Hessian_difference=error,finite_difference_errors=differences,
              meaning='Algebraic exact-bulk Schur reduction; unchanged sampled C and normalization adjoint'))
    print(error,differences,flush=True)


def run(n):
    out=E/f'n{n}';out.mkdir(exist_ok=False);p=Exact(n);r=p.r
    phi=r.initial.copy();R=np.zeros_like(r.rq);Z=0.;slip=0.;dt=1/1024
    phases=[phi.copy()];strains=[R.copy()];zs=[Z];velocities=[];rows=[];energies=[];failures=[]
    previous_increment=np.zeros_like(phi);started=time.perf_counter();first_saddle=None
    old.write(out/'initialization.json',dict(phase_max=float(max(phi)),minimum=float(min(phi)),
          unloaded_KKT=float(r.defect((r.flinear+r.F@phi)/A,phi,phi)),
          definition='Nodal interpolation of one continuum seeded-H reference; unloaded R=Z=slip=0'))
    np.savez_compressed(out/'initialized-H.npz',particle_y=r.initial_particle_y,H=r.initial_particle_H)
    for k in range(640):
        t=(k+1)*dt;p.incoming(phi,R,Z,slip,t,dt)
        guess=np.maximum(phi,phi+previous_increment)
        x=r.refine(guess,p.lower,p.evaluate,p.hessian)
        if r.defect(p.evaluate(x)[1],x,p.lower)>TARGET:x,_=old.minimize_phase(p,x,1)
        status,v=old.spectrum(p,x)
        if status['admissible_negative']:
            if first_saddle is None:first_saddle=t
            x,_=old.minimize_phase(p,x+.5*status['amplitude_two_sided']*v,1)
            status,_=old.spectrum(p,x)
        p.evaluate(x);data=p.last[3].copy();Rb=p.last[4].copy();Zb=p.last[5];u=r.velocity.copy()
        kkt=float(r.defect(p.evaluate(x)[1],x,p.lower))
        if kkt>TARGET or data['force']>1e-7 or data['contact']>1e-7 or not status['stable']:
            failures.append(dict(time=t,KKT=kkt,stability=status,mechanical=data))
            np.savez_compressed(out/'failed-trial.npz',phase=x,lower=phi,Ra=R,Za=Z,R=Rb,Z=Zb)
            break
        assert np.array_equal(phi,p.lower) and np.array_equal(R,p.Ra) and Z==p.Za
        assert min(x-phi)>=-4.922e-12 and max(x)<1
        ledger=old.exact_ledger(p,phi,x,R,Z,Rb,Zb,u,data['V'],dt)
        ledger.update(time=t,phase_increment=float(max(x-phi)))
        status.pop('lower_y',None)
        rows.append(dict(time=t,KKT=kkt,phase_max=float(max(x)),phase_increment=ledger['phase_increment'],
                         centroid=old.centroid(r,x),stability=status,**data))
        energies.append(ledger);previous_increment=x-phi
        phi=x.copy();R=Rb.copy();Z=Zb;slip=data['slip']
        phases.append(phi.copy());strains.append(R.copy());zs.append(Z);velocities.append(u)
        if k%32==0 or t>.59:
            old.write(out/'history.json',rows);old.write(out/'energy.json',energies)
            print(old.json.dumps(dict(n=n,time=t,phase=max(phi),centroid=rows[-1]['centroid'],minimum=status['minimum'])),flush=True)
    np.savez_compressed(out/'fields.npz',phase=np.array(phases),R=np.array(strains),Z=np.array(zs),velocity=np.array(velocities),
                        y=r.y,yq=r.yq,initial=r.initial)
    old.write(out/'history.json',rows);old.write(out/'energy.json',energies)
    old.write(out/'summary.json',dict(n=n,dt=dt,exact_bulk=True,first_saddle=first_saddle,completed=not failures,
         last_accepted=rows[-1]['time'] if rows else 0.,accepted=len(rows),failures=failures,
         runtime_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['reference','check-reference','equivalence','run'])
    parser.add_argument('--cells',type=int,choices=[511,512,1024],default=512);args=parser.parse_args()
    if args.action=='reference':reference()
    elif args.action=='check-reference':check_reference()
    elif args.action=='equivalence':equivalence()
    else:run(args.cells)
