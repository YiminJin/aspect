#!/usr/bin/env python3
"""Focused derivative/normalization arithmetic evidence and read-only checks."""
import argparse
import hashlib
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from coupled_branch import E, HERE, START, FINAL, TARGET, Coupled, G, ETA, A, h, dh, write


def read(p):return json.loads(p.read_text())


def arithmetic():
    p=Coupled(); output=[]
    for d in [256,512,1024]:
        f=np.load(E/f'dt-1-{d}'/'fields.npz');hist=read(E/f'dt-1-{d}'/'history.json')
        checks=[]
        for t in [.53125,.546875,FINAL]:
            j=round((t-START)*d)-1;pa=f['phase'][j];pb=f['phase'][j+1];Ra=f['R'][j];Za=f['Z'][j]
            p.incoming(pa,Ra,Za,hist[j-1]['slip'],t,1/d)
            _,g=p.evaluate(pb);H=p.hessian(pb);free=np.flatnonzero(pb>pa+1e-13)
            correction=solve(H[np.ix_(free,free)],-g[free],assume_a='sym')
            checks.append(dict(time=t,Newton_correction_uncertainty=float(max(abs(correction))),
                               KKT=float(p.r.defect(g,pb,pa))))
        # Normalization: independent point integration of the fixed Q1 field.
        norm=[]
        for j in [0,round((.53125-START)*d),len(f['phase'])-1]:
            phi=f['phase'][j];I,J=p.r.direct(phi);estimates=[]
            for order in [12,24]:
                z,w=leggauss(order);s=(z+1)/2
                values=phi[:-1,None]*(1-s)+phi[1:,None]*s
                direct=np.sum(h(values)*(w*p.r.dy/2))
                left=np.sum(dh(values)*(1-s)*(w*p.r.dy/2),axis=1)
                right=np.sum(dh(values)*s*(w*p.r.dy/2),axis=1)
                jac=np.r_[left,0.]+np.r_[0.,right]
                estimates.append(dict(order=order,I_error_relative=float(abs(direct-I)/I),
                                      jacobian_error_relative=float(max(abs(jac-J))/max(abs(J)))))
            norm.append(dict(time=START+j/d,I=I,positive=bool(I>0),
                             bulk_rule_integral_chi=float(p.r.w@(h(p.r.B@phi)/I)),checks=estimates))
        output.append(dict(denominator=d,arithmetic=checks,normalization=norm))
    # Same final entry replay after deliberately changing unrelated scratch state.
    d=1024;f=np.load(E/f'dt-1-{d}'/'fields.npz');hist=read(E/f'dt-1-{d}'/'history.json')
    p.incoming(f['phase'][-2],f['R'][-2],f['Z'][-2],hist[-2]['slip'],FINAL,1/d)
    p.r.r=np.cos(np.arange(len(p.Ra)))*.017;p.r.Z=-17.;p.r.phi=np.zeros_like(p.lower)
    repeated,kkt,message=p.solve()
    repeat_error=float(max(abs(repeated-f['phase'][-1])))
    assert kkt<=TARGET and repeat_error<1e-8
    # Finite-difference energy along the continuous path, independent of g.
    pa=f['phase'][-2];pb=f['phase'][-1];Ra=f['R'][-2];Za=f['Z'][-2]
    u=f['velocity'][-1];V=hist[-1]['V'];dt=1/d
    def point(s):
        beta=np.exp(-G*s/ETA);k=-ETA*np.expm1(-G*s/ETA)
        phi=pa+s/dt*(pb-pa);R=beta*Ra+k/G*u;Z=beta*Za+k*V
        p.r.r=R;p.r.Z=Z
        return p.r.objective(phi)[0]*A,phi,R,Z
    t=dt/2;energy,phi,R,Z=point(t);I,J=p.r.direct(phi);phidot=(pb-pa)/dt
    chi=h(p.r.B@phi)/I;chid=dh(p.r.B@phi)*(p.r.B@phidot)/I-h(p.r.B@phi)*(J@phidot)/I**2
    tau=G*(p.r.D@R)-Z*chi
    taudot=G*(p.r.D@u)-G*chi*V-G/ETA*tau-Z*chid
    exact=p.r.w@(tau*taudot)/G+Z/(G*I)*(G*V-G/ETA*Z)-Z*Z/(2*G*I**2)*(J@phidot)+(p.r.flinear+p.r.F@phi)@phidot
    derivative=[]
    for epsilon in [dt*1e-2,dt*1e-3,dt*1e-4]:
        fd=(point(t+epsilon)[0]-point(t-epsilon)[0])/(2*epsilon)
        derivative.append(dict(epsilon=epsilon,analytic_power=float(exact),FD_power=float(fd),error=float(abs(fd-exact))))
    assert min(x['error'] for x in derivative)<1e-5
    write(E/'arithmetic.json',dict(cases=output,identical_entry_repeat_phase_error=repeat_error,
                                 repeat_KKT=kkt,continuous_energy_FD=derivative))


def check():
    manifest=read(E/'preservation.json');root=HERE.parents[2]
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    provenance=E/'provenance.json'
    if provenance.exists():
        for name,digest in read(provenance)['new_sources'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    v=read(E/'assessment/verification.json');a=read(E/'arithmetic.json')
    assert a['repeat_KKT']<=TARGET
    for x in a['cases']:
        for y in x['normalization']:
            assert y['positive'] and y['checks'][-1]['I_error_relative']<1e-11
            assert y['checks'][-1]['jacobian_error_relative']<1e-11
        for y in x['arithmetic']:assert y['KKT']<=TARGET
    for x in v:
        for y in x['checks']:assert y['identity_error']<1e-10 and y['time_quadrature_uncertainty']<1e-10
    for d in [256,512,1024]:
        s=read(E/f'dt-1-{d}'/'summary.json');assert s['completed']
        f=np.load(E/f'dt-1-{d}'/'fields.npz');assert len(f['R'])==s['accepted']+1
        hist=read(E/f'dt-1-{d}'/'history.json');start=np.load(E/'start.npz')
        beta=np.exp(-G/(d*ETA));kappa=-ETA*np.expm1(-G/(d*ETA))
        assert max(abs((f['R'][1:]-beta*f['R'][:-1]-kappa/G*f['velocity']).ravel()))<1e-16
        assert max(abs(f['Z'][1:]-beta*f['Z'][:-1]-kappa*np.array([x['V'] for x in hist])))<1e-10
        assert abs(hist[-1]['slip']-float(start['slip'])-sum(x['V']/d for x in hist))<1e-15
        displacement=sum((u[-1]-u[0])/d for u in f['velocity'])
        assert abs(displacement-.128*(FINAL**2-START**2))<1e-15
        assert min(np.diff(f['phase'],axis=0).ravel())>=-4.922e-12 and min(f['phase'].ravel())>=0 and max(f['phase'].ravel())<1
        for x in read(E/f'dt-1-{d}'/'stability.json'):assert x['equilibrium'] and not x['admissible_negative']
    print(json.dumps(dict(short_branch_checks='pass',preserved_files=len(manifest),
                          native_adoption=False,full_time_interval_qualified=False,P3=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record-arithmetic',action='store_true');args=parser.parse_args()
    if args.record_arithmetic:
        assert not (E/'arithmetic.json').exists();arithmetic()
    check()
