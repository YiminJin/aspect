#!/usr/bin/env python3
"""Independent 1D variational explanation of unresolved numerical lifting."""
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial, Legendre
from investigate_cohesive_interface_bulk_contact import OUT, COEFFICIENTS, build, save, table, WORK

G=1e6;HEIGHT=12.;JUMP=.001;GAMMA=.002


def projection_remainder(n,width,degree):
    z,w=np.polynomial.legendre.leggauss(12);residual=0.;h=HEIGHT/n
    for j in range(n):
        lo=-6+j*h;hi=lo+h;a=max(lo,-width);b=min(hi,width)
        if b<=a:continue
        zz=(a+b)/2+(b-a)/2*z;ww=(b-a)/2*w
        norm=b-a
        for k in range(degree):
            moment=ww@Legendre.basis(k)((zz-lo)*2/h-1)
            norm-=moment**2/(h/(2*k+1))
        residual+=norm
    return max(0.,residual)


def finite_elements(n,width,degree):
    x=np.linspace(-6,6,n+1);N=n*degree+1
    K=np.zeros((N,N));f=np.zeros(N)
    grid=np.linspace(0,1,degree+1);basis=[]
    for j in range(degree+1):
        p=Polynomial([1.])
        for i in range(degree+1):
            if i!=j:p*=Polynomial([-grid[i],1])/(grid[j]-grid[i])
        basis.append(p.deriv())
    z,w=np.polynomial.legendre.leggauss(6);samples=[]
    for cell in range(n):
        lo,hi=x[cell:cell+2];h=hi-lo;ids=np.arange(cell*degree,cell*degree+degree+1)
        cuts=sorted(set([lo,hi]+[v for v in [-width,width] if lo<v<hi]))
        for a,b in zip(cuts,cuts[1:]):
            xx=(a+b)/2+(b-a)/2*z;weight=(b-a)/2*w
            grad=np.array([p((xx-lo)/h)/h for p in basis]);lift=np.where(abs(xx)<width,-JUMP/(2*width),0.)
            K[np.ix_(ids,ids)]+=G*(grad*weight)@grad.T
            f[ids]-=G*grad@(weight*(GAMMA+lift))
            samples.append((xx,weight,ids,grad,lift))
    u=np.zeros(N);u[1:-1]=np.linalg.solve(K[1:-1,1:-1],f[1:-1]);energy=0.;profiles=[]
    for xx,w,ids,grad,lift in samples:
        strain=GAMMA+u[ids]@grad+lift;energy+=G/2*(w@(strain*strain))
        for y,e in zip(xx,strain):profiles.append(dict(n=n,width_m=width,degree=degree,y_m=float(y),shear_Pa=float(G*e)))
    return float(energy),profiles


def main():
    rows=[];profiles=[]
    baseline=G/2*HEIGHT*(GAMMA-JUMP/HEIGHT)**2
    for n,width,degree in [(64,.1875,2),(96,.125,2),(96,.1875,2),(64,.75,2),(96,.75,2),(96,.1875,3)]:
        residual=projection_remainder(n,width,degree)
        extra=G*JUMP**2/(8*width**2)*residual
        value,p=finite_elements(n,width,degree);profiles+=p
        assert abs(value-baseline-extra)<1e-10
        rows.append(dict(n=n,width_m=width,degree=degree,exact_incompressible_energy_J_m2=baseline,
            computed_energy_J_m2=value,projection_remainder_m=residual,predicted_extra_J_m2=extra,
            extra_force_Pa=2*extra/JUMP,error_J_m2=abs(value-baseline-extra)))
    # Same physical old-seed test as the actual 2D comparison. No history is
    # changed; only its saved scalar m(x) is evaluated in the analytic formula.
    k=build('wide96',96,2,.75);x=k.nodes['x_m'];test=np.where((x>3.75)&(x<5.5),np.sin(np.pi*(x-3.75)/1.75)**2,0.);tq=k.Q@test
    residual=projection_remainder(96,.1875,2);coefficient=G/(4*.1875**2)*residual
    predicted=-coefficient*(k.w*tq)@(k.Q@COEFFICIENTS[1:])/(k.w@tq)
    import csv
    def load(label):return next(r for r in csv.DictReader((OUT/(label+'-loads.csv')).open()) if r['region']=='old')
    actual=float(load('fixed96')['tangential_work_conjugate_Pa'])-float(load('wide96')['tangential_work_conjugate_Pa'])
    save(OUT/'lifting-mechanism.json',dict(formula='Delta E=G*m^2/(8*w^2)*||(Id-Pi_{k-1,h})1_{|y|<w}||^2',
        meaning='Continuous counter-lifting has gradient kinks at arbitrary support edges. An unresolved kink adds numerical resistance, even in divergence-free pure shear.',
        pressure_free_control=True,old_region=[3.75,5.5],predicted_load_change_Pa=float(predicted),actual_2D_load_change_Pa=actual,
        prediction_error_Pa=float(actual-predicted),relative_error=float(abs(actual-predicted)/abs(actual))))
    table(OUT/'lifting-reference.csv',rows);table(OUT/'lifting-reference-profiles.csv',profiles);k.close()


if __name__=='__main__':main()
