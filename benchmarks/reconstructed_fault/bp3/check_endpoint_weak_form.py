"""Manufactured algebra audit, NOT an ASPECT/free-endpoint coupled solve.

Check the work-conjugate weights and frozen-state derivatives proposed in
stage_K5_endpoint_weak_form_review.md. No production state or data are changed.
"""
import json
import math


def contract(a, b):
    # Symmetric tensors stored as xx, yy, xy.
    return a[0]*b[0]+a[1]*b[1]+2*a[2]*b[2]


s=(.5, math.sqrt(3)/2)
n=(-s[1], s[0])
S=(s[0]*n[0], s[1]*n[1], .5*(s[0]*n[1]+s[1]*n[0]))
N=(n[0]*n[0], n[1]*n[1], n[0]*n[1])
assert abs(contract(S,N))<1e-15

# A last-segment sample, an endpoint-clamped wedge sample, and intact material.
# All quantities are dimensionless manufactured data; weights are independent
# of V and bulk state. The friction law is the existing stateless weakening
# law, used only to check generic mu/mu' algebra with a nonzero derivative.
points=[(1.7,.8,2.,(.4,.6)), (.9,1.3,3.,(0.,1.)), (2.1,0.,4.,(0.,1.))]
v=[.7,.9]
dv=[.13,-.17]
strain=(.11,-.06,.09)
dstrain=(.03,-.02,-.04)
pressure=.2
dp=.07
old=(.12,-.08,.03)
damping=.15


def mu(rate):
    return .4+.2/(1+rate/.5)


def dmu(rate):
    return -.2*.5/(rate+.5)**2


def evaluate(rates, eps, p):
    residual=[0.,0.]
    for volume,chi,kappa,basis in points:
        rate=sum(a*b for a,b in zip(basis,rates))
        tau=tuple(2*kappa*(e-chi*rate*t)+o for e,t,o in zip(eps,S,old))
        driving=2.+contract(tau,S)
        sigma=5.+p-contract(tau,N)
        F=driving-mu(rate)*sigma-damping*rate
        for i in range(2):residual[i]+=volume*chi*basis[i]*F
    return residual


def main():
    K=[[0.,0.],[0.,0.]]
    G=[0.,0.]
    B_work=0.
    shear_G_work=0.
    for volume,chi,kappa,basis in points:
        rate=sum(a*b for a,b in zip(basis,v))
        tau=tuple(2*kappa*(e-chi*rate*t)+o for e,t,o in zip(strain,S,old))
        sigma=5.+pressure-contract(tau,N)
        tangent=2*kappa*chi*contract(S,S)+sigma*dmu(rate)+damping
        bulk_direction=2*kappa*(contract(S,dstrain)+mu(rate)*contract(N,dstrain))-mu(rate)*dp
        for i in range(2):
            G[i]+=volume*chi*basis[i]*bulk_direction
            for j in range(2):K[i][j]+=volume*chi*basis[i]*basis[j]*tangent
        rate_direction=sum(a*b for a,b in zip(basis,dv))
        B_work+=volume*2*kappa*chi*rate_direction*contract(S,dstrain)
        shear_G_work+=sum(dv[i]*volume*chi*basis[i]*2*kappa*contract(S,dstrain) for i in range(2))
    step=1e-6
    plus=evaluate([a+step*b for a,b in zip(v,dv)],strain,pressure)
    minus=evaluate([a-step*b for a,b in zip(v,dv)],strain,pressure)
    fd_V=[(a-b)/(2*step) for a,b in zip(plus,minus)]
    exact_V=[-sum(K[i][j]*dv[j] for j in range(2)) for i in range(2)]
    plus=evaluate(v,tuple(a+step*b for a,b in zip(strain,dstrain)),pressure+step*dp)
    minus=evaluate(v,tuple(a-step*b for a,b in zip(strain,dstrain)),pressure-step*dp)
    fd_bulk=[(a-b)/(2*step) for a,b in zip(plus,minus)]
    err_V=max(abs(a-b) for a,b in zip(fd_V,exact_V))
    err_G=max(abs(a-b) for a,b in zip(fd_bulk,G))
    assert err_V<1e-8 and err_G<1e-8
    assert abs(B_work-shear_G_work)<1e-15

    # Same tangential coordinate, two transverse samples: no row rescaling can
    # turn a zero volume residual into a nonzero work-conjugate residual.
    F=[1.,-1.];chi=[1.,3.];basis=.5
    volume_residual=sum(basis*f for f in F)
    work_residual=sum(basis*c*f for c,f in zip(chi,F))
    assert volume_residual==0. and work_residual==-1.

    # A perturbation confined to intact material cannot enter the proposed
    # work residual. Its friction resistance need not itself be zero.
    intact_F=-mu(v[-1])*5.-damping*v[-1]
    assert intact_F!=0. and points[-1][0]*points[-1][1]*intact_F==0.
    print(json.dumps(dict(status='passed: manufactured algebra only',
        normal_shear_contraction=contract(S,N),
        V_derivative_max_absolute_error=err_V,
        bulk_derivative_max_absolute_error=err_G,
        shear_virtual_work_error=abs(B_work-shear_G_work),
        interior_counterexample=dict(volume_residual=volume_residual,work_residual=work_residual),
        intact_friction_residual=intact_F,intact_weighted_contribution=0.),indent=2))


if __name__=='__main__':main()
