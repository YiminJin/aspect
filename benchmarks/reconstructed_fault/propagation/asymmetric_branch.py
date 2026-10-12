#!/usr/bin/env python3
"""Selected P2 translation/branch diagnostics; production and old evidence frozen."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, eigh, solve

import coupled_branch as core
import extend_coupled_branch as previous

HERE = core.HERE
ROOT = HERE.parents[2]
E = HERE/'asymmetric_branch_evidence'
TARGET = core.TARGET
A,G,ETA = core.A,core.G,core.ETA
read = previous.read
write = core.write


class MeshCoupled(core.Coupled):
    """Reuse all existing FE equations on another uniform normal mesh."""
    def __init__(self,n):
        self.r = core.Reference(n,1.,6)
        self.D = self.r.D[:,1:-1]
        self.gram = self.D.T@(self.r.w[:,None]*self.D)
        self.factor = cho_factor(self.gram)
        # Same Q2 derivative entries and cached Gram factor; avoid repeatedly
        # multiplying its overwhelmingly zero dense storage in this harness.
        # Fixed-trial residual/Hessian equivalence is checked independently.
        self.r.D=sparse.csr_matrix(self.r.D);self.D=self.r.D[:,1:-1]
        self.evaluations = 0; self.last = None


class ExactCoupled(core.Coupled):
    """Existing exact-bulk control, with its consistent reduced phase tangent.

    R is the quadrature strain state in this diagnostic control, not an FE
    conversion. Fresh histories only. Retain C=quadrature(chi) rather than
    asserting sampled normalization equals one.
    """
    def __init__(self,n):
        self.r=core.Reference(n,1.,6,exact_bulk=True)
        self.evaluations=0;self.last=None

    def incoming(self,phi,R,Z,slip,time_end,dt):
        super().incoming(phi,np.zeros_like(self.r.r),Z,slip,time_end,dt)
        self.Ra=R.copy();self.Ra.flags.writeable=False
        self.anchor_shear=self.Ra.copy()

    def evaluate(self,phi):
        if self.last is not None and np.array_equal(phi,self.last[0]):
            self.r.rq=self.last[4].copy();self.r.Z=self.last[5];self.r.phi=phi.copy()
            return self.last[1:3]
        if min(phi)<0 or max(phi)>=1:return np.inf,np.zeros_like(phi)
        r=self.r;I,_=r.direct(phi);chi=core.h(r.B@phi)/I;C=r.w@chi
        tau0=G*self.Ra-self.Za*chi;mean=r.w@tau0
        rate=.256*(self.time-self.dt/2)
        V=max(0.,(C*(self.beta*mean+self.kappa*rate)-self.beta*self.Za/I-core.MU*core.PRESSURE)
                     /(self.kappa*(C*C+1/I)))
        T=self.beta*mean+self.kappa*(rate-C*V)
        du=(T-self.beta*tau0)/self.kappa+chi*V
        Rb=self.beta*self.Ra+self.kappa/G*du;Zb=self.beta*self.Za+self.kappa*V
        r.rq=Rb.copy();r.Z=Zb;r.phi=phi.copy();r.velocity=du.copy()
        value,g,terms=r.objective(phi);rf=core.MU*core.PRESSURE+Zb/I-C*T
        data=dict(q=float(Zb/I),V=float(V),slip=float(self.slip_a+self.dt*V),
             bulk=float(terms[0]),cohesive=float(terms[1]),fracture=float(terms[2]),
             I=float(I),force=float(max(abs(terms[-1]-T))),contact_signed=float(rf),
             contact=float(abs(rf) if V>0 else max(-rf,0.)),stress_min=float(min(terms[-1])),
             stress_max=float(max(terms[-1])))
        potential=value+self.kappa/G*core.MU*core.PRESSURE*V/A
        self.last=(phi.copy(),potential,g.copy(),data,Rb.copy(),Zb)
        self.evaluations+=1
        return potential,g

    def hessian(self,phi):
        self.evaluate(phi);r=self.r;I,J=r.direct(phi);hp=core.h(r.B@phi)
        Cphi=(sparse.diags(core.dh(r.B@phi)/I)@r.B).toarray()-np.outer(hp/I**2,J)
        chi=hp/I;C=r.w@chi;Z=r.Z;tau=G*r.rq-Z*chi
        integral=r.w@Cphi
        correction=Z*Z*(Cphi.T@(r.w[:,None]*Cphi)-np.outer(integral,integral))
        if self.last[3]['V']>0:
            row=-Z*J/I**2-Cphi.T@(r.w*tau)+Z*C*integral
            correction+=np.outer(row,row)/(C*C+1/I)
        return r.phase_hessian(phi)-correction/(G*A)


def model(n,exact=False):
    return ExactCoupled(n) if exact else MeshCoupled(n)


def mass(r):
    return sparse.diags([np.full(r.n,r.dy/6),
        np.r_[r.dy/3,np.full(r.n-1,2*r.dy/3),r.dy/3],
        np.full(r.n,r.dy/6)],[-1,0,1]).toarray()


def centroid(r,phi):
    values = r.B@phi; hp = core.h(values)
    return dict(phase=float(r.w@(r.yq*values)/(r.w@values)),
                localization=float(r.w@(r.yq*hp)/(r.w@hp)))


def spectrum(p,x):
    _,g = p.evaluate(x); r = p.r; M = mass(r)
    free = x>p.lower+1e-13; weak = (~free)&(abs(g)<=10*TARGET)
    ids = np.flatnonzero(free|weak)
    out = dict(KKT=float(r.defect(g,x,p.lower)),free=int(sum(free)),weak=int(sum(weak)),
               lower_y=r.y[~free].tolist(),centroid=centroid(r,x))
    v = np.zeros(len(x))
    if not len(ids):
        out.update(minimum=None,admissible_negative=False,stable=True)
        return out,v
    H = p.hessian(x)
    ev,vec = eigh(A*H[np.ix_(ids,ids)],M[np.ix_(ids,ids)],subset_by_index=[0,min(1,len(ids)-1)])
    v[ids] = vec[:,0]
    if r.w@(r.yq*(r.B@v))<0: v=-v
    w=v[weak]; admissible=not len(w) or min(w)>=-1e-12 or max(w)<=1e-12
    pd=np.diff(x)/r.dy; vq=r.B@v
    translation=np.repeat(-pd,len(r.w)//r.n)
    cosine=float(r.w@(vq*translation)/np.sqrt((r.w@(vq*vq))*(r.w@(translation*translation))))
    nz=abs(v)>1e-12
    amplitude=float(min((x-p.lower)[nz]/abs(v[nz])))
    out.update(minimum=float(ev[0]),second=float(ev[-1]),
        admissible_negative=bool(ev[0]<0 and admissible),stable=bool(ev[0]>0),
        translation_cosine=cosine,antisymmetry=float(max(abs(v+v[::-1]))),
        amplitude_two_sided=amplitude,mass_norm=float(v@M@v))
    return out,v


def minimize_phase(p,start,direction_sign=1,maxiter=400):
    """Projected Newton descent with a positive search metric, exact KKT/energy.

    No indefinite root polish. Negative curvature changes the search direction,
    never the physical Hessian. The caller retains the immutable incoming state.
    """
    x=start.copy(); anchor=p.Ra.copy(); lower=p.lower.copy(); Za=p.Za
    iterations=[]
    for k in range(maxiter):
        value,g=p.evaluate(x); defect=float(p.r.defect(g,x,p.lower))
        free=np.flatnonzero((x>p.lower+1e-13)|(g<0.))
        if not len(free): break
        H=p.hessian(x); ev,Q=eigh(H[np.ix_(free,free)])
        if defect<=TARGET:
            status,v=spectrum(p,x)
            if status['stable'] or status['minimum'] is None: break
            if not status['admissible_negative']: break
            nz=abs(v)>1e-12
            feasible_plus=min((x-p.lower)[v< -1e-12]/(-v[v< -1e-12])) if np.any(v< -1e-12) else np.inf
            feasible_minus=min((x-p.lower)[v>1e-12]/v[v>1e-12]) if np.any(v>1e-12) else np.inf
            amplitude=feasible_plus if direction_sign>0 else feasible_minus
            step=direction_sign*.8*amplitude*v
        else:
            metric=np.maximum(ev,max(abs(min(ev)),1e-10))
            step=np.zeros(len(x));step[free]=-Q@((Q.T@g[free])/metric)
        accepted=False
        for ls in range(60):
            trial=np.maximum(p.lower,np.minimum(1.,x+2.**(-ls)*step))
            if max(trial)>=1: continue
            vt,gt=p.evaluate(trial)
            pairing=float(g@(trial-x))
            allowance=32*np.finfo(float).eps*max(1.,abs(value))
            if vt<=value+1e-4*min(pairing,0.)+allowance:
                if np.array_equal(trial,x): continue
                x=trial;accepted=True;break
        iterations.append(dict(iteration=k,KKT=defect,curvature=float(ev[0]),
                               line_search=ls,potential=value))
        if not accepted: break
    _,g=p.evaluate(x);status,v=spectrum(p,x)
    assert np.array_equal(anchor,p.Ra) and np.array_equal(lower,p.lower) and Za==p.Za
    assert not p.lower.flags.writeable and not p.Ra.flags.writeable
    return x,dict(KKT=float(p.r.defect(g,x,p.lower)),iterations=len(iterations),
                  stable=status['stable'],stability=status,trace=iterations)


def prepare():
    E.mkdir(exist_ok=False)
    manifest=read(previous.E/'preservation.json')
    paths=set(ROOT/name for name in manifest)
    paths.update(p for p in previous.E.rglob('*') if p.is_file())
    paths.update(HERE/name for name in ['coupled_branch.py','extend_coupled_branch.py',
                                      'localization_1d.py','total_energy_step.py'])
    write(E/'preservation.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def search(d):
    out=E/f'n256-d{d}-search';out.mkdir(exist_ok=False)
    old=previous.E/f'dt-1-{d}';trial=np.load(old/'failed-trial.npz');hist=read(old/'history.json')
    p=MeshCoupled(256);t=read(old/'summary.json')['failures'][0]['time']
    p.incoming(trial['lower'],trial['Ra'],trial['Za'],hist[-1]['slip'],t,1/d)
    search_entry(p,trial['phi'],out)


def search_entry(p,symmetric,out):
    base,_=p.evaluate(symmetric);base_data=p.last[3].copy();spec,v=spectrum(p,symmetric)
    spec.update(potential=float(A*base),endpoint_energy=float(sum(base_data[k] for k in ['bulk','cohesive','fracture'])))
    np.savez_compressed(out/'incoming.npz',phi=p.lower,R=p.Ra,Z=p.Za,slip=p.slip_a,time=p.time,dt=p.dt,
                        symmetric=symmetric,mode=v,y=p.r.y)
    write(out/'mode.json',spec)
    results=[]
    for fraction in [.25,.5,.8]:
        for sign in [-1,1]:
            amplitude=fraction*spec['amplitude_two_sided']
            guess=symmetric+sign*amplitude*v
            assert min(guess-p.lower)>=0
            x,answer=minimize_phase(p,guess,sign)
            value,g=p.evaluate(x);data=p.last[3].copy()
            tag=f'{fraction:g}-{sign:+d}'
            np.savez_compressed(out/f'candidate-{tag}.npz',phase=x,R=p.last[4],Z=p.last[5],velocity=p.r.velocity,y=p.r.y)
            row=dict(fraction=fraction,sign=sign,amplitude=amplitude,
                physical_L2_guess=float(np.sqrt((guess-symmetric)@mass(p.r)@(guess-symmetric))),
                potential_reduction=float(A*(base-value)),endpoint_energy=float(sum(data[k] for k in ['bulk','cohesive','fracture'])),
                phase_increment=float(max(x-p.lower)),centroid=centroid(p.r,x),mechanical=data,**answer)
            results.append(row);write(out/'candidates.json',results)
            print(json.dumps({k:row[k] for k in ['fraction','sign','KKT','stable','iterations','potential_reduction','centroid']}),flush=True)


def first_instability(n,d,exact=False):
    out=E/(f'n{n}-d{d}'+('-exact' if exact else ''))
    out.mkdir(exist_ok=False);p=model(n,exact);r=p.r
    phi=r.initial.copy();R=np.zeros_like(r.rq if exact else r.r);Z=0.;slip=0.
    phases=[phi.copy()];strains=[R.copy()];zs=[Z];rows=[];statuses=[];started=time.perf_counter()
    for k in range(round(.625*d)):
        t=(k+1)/d;p.incoming(phi,R,Z,slip,t,1/d)
        x,kkt,_=p.solve();p.evaluate(x);data=p.last[3].copy();Rb=p.last[4].copy();Zb=p.last[5]
        assert kkt<=TARGET and data['force']<1e-7 and data['contact']<1e-7
        status,v=spectrum(p,x);status['time']=t
        statuses.append(status)
        if status['admissible_negative']:
            np.savez_compressed(out/'failed-trial.npz',phi=x,lower=phi,R=Rb,Z=Zb,Ra=R,Za=Z)
            search_out=out/'search';search_out.mkdir()
            search_entry(p,x,search_out)
            break
        if status['minimum'] is not None and not status['stable']:
            raise RuntimeError(('critical cone unresolved',t,status))
        phi=x.copy();R=Rb.copy();Z=Zb;slip=data['slip']
        phases.append(phi.copy());strains.append(R.copy());zs.append(Z)
        rows.append(dict(time=t,phase_max=float(max(phi)),phase_increment=float(max(phi-p.lower)),KKT=kkt,
                         centroid=centroid(r,phi),**data))
        if k%32==0 or t>.59:print(json.dumps(dict(n=n,d=d,exact=exact,time=t,phase=max(phi),minimum=status['minimum'])),flush=True)
    np.savez_compressed(out/'fields.npz',phase=np.array(phases),R=np.array(strains),Z=np.array(zs),y=r.y,yq=r.yq,
                        initial=r.initial,particle_y=r.initial_particle_y,H=r.initial_particle_H)
    write(out/'history.json',rows);write(out/'stability.json',statuses)
    write(out/'summary.json',dict(n=n,d=d,exact_bulk=exact,initial_max=float(max(r.initial)),
          last_accepted=rows[-1]['time'],first_instability=status['time'] if status['admissible_negative'] else None,
          runtime_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def search_path(n,d,exact=False):
    return E/f'n256-d{d}-search' if n==256 and not exact else E/(f'n{n}-d{d}'+('-exact' if exact else ''))/'search'


def exact_ledger(p,pa,pb,Ra,Za,Rb,Zb,du,V,dt,order=16):
    """Continuous-profile control; reaction work uses analytical boundary traces.

    Bulk-rate minus reaction work is the integrated strong mechanical residual
    pairing. No FE force or FE projection is invented for the exact-bulk control.
    """
    r=p.r
    def state(phi,R,Z):
        r.rq=R;r.Z=Z
        return r.objective(phi)[2]
    a=state(pa,Ra,Za);b=state(pb,Rb,Zb)
    cost=sum(state(pb,Ra,Za)[:3])-sum(a[:3])
    Ia=r.direct(pa)[0];Ib=r.direct(pb)[0]
    Ta=float(r.w@(G*Ra-Za*core.h(r.B@pa)/Ia))
    Tb=float(r.w@(G*Rb-Zb*core.h(r.B@pb)/Ib))
    chi_a_boundary=core.h(pa[[0,-1]])/Ia;chi_b_boundary=core.h(pb[[0,-1]])/Ib
    Ra_boundary=(Ta+Za*chi_a_boundary)/G
    du_boundary=(Tb-p.beta*(Ta+Za*(chi_a_boundary-chi_b_boundary)))/p.kappa+chi_b_boundary*V
    rate=.256*(p.time-p.dt/2);phidot=(pb-pa)/dt
    keys=['external_work','bulk_work','lifting_work','bulk_viscous','cohesive_viscous','friction',
          'mechanical_power','friction_residual_power','phase_power','bulk_profile_power','normalization_power','fracture_power']
    ints={k:0. for k in keys};absints={k:0. for k in ['mechanical_power','friction_residual_power','phase_power']}
    maximum=dict(contact=0.,phase_KKT=0.,chi_rate=0.,identity_power=0.)
    z,w=np.polynomial.legendre.leggauss(order)
    for s,weight in zip((z+1)*dt/2,w*dt/2):
        beta=np.exp(-G*s/ETA);kappa=-ETA*np.expm1(-G*s/ETA)
        phi=pa+s*phidot;R=beta*Ra+kappa/G*du;Z=beta*Za+kappa*V
        terms=state(phi,R,Z);I,J=r.direct(phi);hp=core.h(r.B@phi);chi=hp/I
        chid=core.dh(r.B@phi)*(r.B@phidot)/I-hp*(J@phidot)/I**2
        tau=terms[-1];gb=sum(terms[3:6]);rf=core.MU*core.PRESSURE+Z/I-r.w@(chi*tau)
        trace=G*(beta*Ra_boundary+kappa/G*du_boundary)-Z*core.h(phi[[0,-1]])/I
        external=rate/2*sum(trace);bulkwork=r.w@(du*tau)
        db=r.w@(tau*tau)/ETA;dc=Z*Z/(ETA*I);friction=core.MU*core.PRESSURE*V
        npower=-Z*Z/(2*G*I**2)*(J@phidot);fp=(r.flinear+r.F@phi)@phidot
        values=dict(external_work=external,bulk_work=bulkwork,lifting_work=rate*(r.w@tau),
            bulk_viscous=db,cohesive_viscous=dc,friction=friction,mechanical_power=bulkwork-external,
            friction_residual_power=rf*V,phase_power=gb@phidot,
            bulk_profile_power=-Z/G*r.w@(tau*chid),normalization_power=npower,fracture_power=fp)
        taud=G*du-G*chi*V-G/ETA*tau-Z*chid
        edot=r.w@(tau*taud)/G+Z/(G*I)*(G*V-G/ETA*Z)+npower+fp
        rhs=external-db-dc-friction+sum(values[k] for k in absints)
        maximum['identity_power']=max(maximum['identity_power'],abs(edot-rhs))
        maximum['contact']=max(maximum['contact'],abs(rf) if V>0 else max(-rf,0.))
        maximum['phase_KKT']=max(maximum['phase_KKT'],r.defect(gb/A,phi,pa))
        maximum['chi_rate']=max(maximum['chi_rate'],max(abs(chid)))
        for k in ints:ints[k]+=weight*values[k]
        for k in absints:absints[k]+=weight*abs(values[k])
    change=np.array(b[:3])-a[:3]
    remainder=ints['external_work']-sum(change)-sum(ints[k] for k in ['bulk_viscous','cohesive_viscous','friction'])
    return dict(**{k:float(v) for k,v in ints.items()},**{k+'_absolute':float(v) for k,v in absints.items()},
        bulk_change=float(change[0]),cohesive_change=float(change[1]),fracture_change=float(change[2]),
        signed_profile_cost=float(cost),external_remainder=float(remainder),
        identity_error=float(abs(remainder+sum(ints[k] for k in absints))),
        intermediate_max={k:float(v) for k,v in maximum.items()},time_quadrature=order,
        mechanical_measure='bulk-rate minus analytical boundary-reaction work')


def continuation(n,d,sign,exact=False):
    base=search_path(n,d,exact);out=base.parent/(base.name+f'-continue-{sign:+d}')
    out.mkdir(exist_ok=False);start=np.load(base/'incoming.npz')
    candidate=np.load(base/f'candidate-0.5-{sign:+d}.npz')
    p=model(n,exact);phi=start['phi'].copy();R=start['R'].copy();Z=float(start['Z']);slip=float(start['slip'])
    initial=phi.copy();dt=1/d;starttime=float(start['time']);started=time.perf_counter()
    rows=[];energies=[];phases=[phi.copy()];strains=[R.copy()];zs=[Z];velocities=[];failures=[]
    previous_increment=None
    for k in range(round((.625-starttime)*d)+1):
        t=starttime+k/d;p.incoming(phi,R,Z,slip,t,dt)
        if k==0:
            x=candidate['phase'].copy();answer={'iterations':0};status,_=spectrum(p,x)
        else:
            if n>=511:
                # Numerical trial guess only: the anchor and obstacle remain
                # the last accepted state. No extrapolation becomes history.
                guess=np.maximum(phi,phi+previous_increment)
                x=p.r.refine(guess,p.lower,p.evaluate,p.hessian)
                if p.r.defect(p.evaluate(x)[1],x,p.lower)>TARGET:
                    x,answer=minimize_phase(p,x,sign)
            else:x,_,_=p.solve()
            status,v=spectrum(p,x)
            if status['admissible_negative']:
                amplitude=.5*status['amplitude_two_sided']
                x,answer=minimize_phase(p,x+sign*amplitude*v,sign);status,_=spectrum(p,x)
            else:answer={'iterations':0}
        p.evaluate(x);data=p.last[3].copy();Rb=p.last[4].copy();Zb=p.last[5];u=p.r.velocity.copy()
        _,g=p.evaluate(x);kkt=float(p.r.defect(g,x,p.lower))
        if kkt>TARGET or data['force']>1e-7 or data['contact']>1e-7 or not status['stable']:
            failures.append(dict(time=t,KKT=kkt,stability=status,mechanical=data))
            np.savez_compressed(out/'failed-trial.npz',phi=x,lower=phi,Ra=R,Za=Z,R=Rb,Z=Zb)
            break
        assert np.array_equal(phi,p.lower) and np.array_equal(R,p.Ra) and Z==p.Za
        assert min(x-phi)>=-4.922e-12 and max(x)<1
        ledger=(exact_ledger(p,phi,x,R,Z,Rb,Zb,u,data['V'],dt) if exact else
                core.path_ledger(p.r,phi,x,R,Z,Rb,Zb,u,data['V'],dt))
        ledger.update(time=t,phase_increment=float(max(x-phi)))
        row=dict(time=t,KKT=kkt,phase_max=float(max(x)),phase_increment=ledger['phase_increment'],
                 centroid=centroid(p.r,x),stability=status,**data)
        previous_increment=x-phi
        phi=x.copy();R=Rb.copy();Z=Zb;slip=data['slip']
        rows.append(row);energies.append(ledger);phases.append(phi.copy());strains.append(R.copy());zs.append(Z);velocities.append(u)
        write(out/'history.json',rows);write(out/'energy.json',energies)
        print(json.dumps(dict(n=n,d=d,sign=sign,exact=exact,time=t,centroid=row['centroid'],minimum=status['minimum'])),flush=True)
    np.savez_compressed(out/'fields.npz',phase=np.array(phases),R=np.array(strains),Z=np.array(zs),velocity=np.array(velocities),
                        y=p.r.y,yq=p.r.yq,incoming_phase=initial)
    write(out/'summary.json',dict(n=n,d=d,sign=sign,exact_bulk=exact,starttime=starttime-dt,
          completed=not failures,last_accepted=rows[-1]['time'] if rows else starttime-dt,
          accepted=len(rows),failures=failures,runtime_seconds=time.perf_counter()-started,
          peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def bracket(n,d,exact=False):
    base=search_path(n,d,exact);start=np.load(base/'incoming.npz');p=model(n,exact)
    end=float(start['time']);ta=end-float(start['dt']);rows=[]
    prior=(np.load(previous.E/f'dt-1-{d}/fields.npz')['phase'][-2] if n==256 and not exact
           else np.load(base.parent/'fields.npz')['phase'][-2])
    def probe(t):
        p.incoming(start['phi'],start['R'],float(start['Z']),float(start['slip']),t,t-ta)
        x,kkt,_=p.solve();s,v=spectrum(p,x)
        p.evaluate(x);g=p.evaluate(x)[1];ids=np.flatnonzero(x>p.lower+1e-13)
        correction=solve(p.hessian(x)[np.ix_(ids,ids)],-g[ids],assume_a='sym')
        released=(x>p.lower+1e-13)&(start['phi']<=prior+1e-13)
        row=dict(time=t,dt=t-ta,KKT=kkt,minimum=s['minimum'],unstable=s['admissible_negative'],
                 weak=s['weak'],free=s['free'],correction_uncertainty=float(max(abs(correction))),
                 newly_released_y=p.r.y[released].tolist(),
                 release_margin=float(min(x[released]-p.lower[released])) if np.any(released) else None,
                 centroid=s['centroid'])
        rows.append(row);return s['admissible_negative']
    lo=ta+float(start['dt'])/8;hi=end
    assert not probe(lo) and probe(hi)
    for _ in range(10):
        mid=(lo+hi)/2
        if probe(mid):hi=mid
        else:lo=mid
    write(base/'release-bracket.json',dict(incoming_time=ta,stable_through=lo,
           first_unstable_by=hi,width=hi-lo,probes=rows,
           restriction='Same immutable incoming history and obstacle; fractional final step, not a new committed trajectory'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','search','prefix','continue','bracket'])
    ap.add_argument('--denominator',type=int,default=1024,choices=[256,512,1024])
    ap.add_argument('--cells',type=int,default=512,choices=[256,511,512])
    ap.add_argument('--exact-bulk',action='store_true');ap.add_argument('--sign',type=int,choices=[-1,1],default=1);args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='search':search(args.denominator)
    elif args.action=='prefix':first_instability(args.cells,args.denominator,args.exact_bulk)
    elif args.action=='continue':continuation(args.cells,args.denominator,args.sign,args.exact_bulk)
    else:bracket(args.cells,args.denominator,args.exact_bulk)
