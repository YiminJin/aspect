#!/usr/bin/env python3
"""Independent fixed-chi mature transformation and complete Q2 quadrature checks."""
from decimal import Decimal, localcontext
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from pathlib import Path
import argparse


def basis(x):
    return np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])


def mass(order):
    x,w=leggauss(order);x=(x+1)/2;w=w/2
    M=np.zeros((9,9))
    for i,a in enumerate(x):
        for j,b in enumerate(x):
            v=np.outer(basis(a),basis(b)).reshape(9)
            M+=w[i]*w[j]*np.outer(v,v)
    return M


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    M3,M8=mass(3),mass(8)
    x,w=leggauss(3);x=(x+1)/2;w=w/2
    # Each of nine tensor-product monomials is projected independently, then
    # evaluated away from support and quadrature nodes. Includes x^2 y^2.
    polynomial_error=0.
    for px in range(3):
        for py in range(3):
            rhs=sum(w[i]*w[j]*np.outer(basis(u),basis(v)).reshape(9)*u**px*v**py
                    for i,u in enumerate(x) for j,v in enumerate(x))
            coefficients=np.linalg.solve(M3,rhs)
            for u,v in ((.17,.39),(.72,.91),(.03,.87),(1.,0.)):
                polynomial_error=max(polynomial_error,abs(np.outer(basis(u),basis(v)).reshape(9)@coefficients-u**px*v**py))
    max_error=Decimal(0)
    with localcontext() as ctx:
        ctx.prec=70
        D=Decimal
        # Normal/tangent for a straight 60-degree thrust, full embedded 3-D.
        root3=D(3).sqrt();S=[-root3/4,root3/4,D(0),D('.25')]
        T=[D('1.1'),D('-.3'),D('-.8'),D('.7')]
        E=[D('.2'),D('-.5'),D('.3'),D('-.4')]
        Z=D('.9');chi=D('.37');V=D('1.4');G=D('2.1');eta=D('3.2')
        for ratio in ('1e-12','1e-6','.1','1','4'):
            beta=(-D(ratio)).exp();kappa=eta*(1-beta)
            old=[T[i]-2*chi*Z*S[i] for i in range(4)]
            updated_T=[beta*T[i]+2*kappa*E[i] for i in range(4)]
            updated_Z=beta*Z+kappa*V
            reconstruction=[updated_T[i]-2*chi*updated_Z*S[i] for i in range(4)]
            direct=[beta*old[i]+2*kappa*(E[i]-chi*V*S[i]) for i in range(4)]
            max_error=max(max_error,max(abs(reconstruction[i]-direct[i]) for i in range(4)))
            assert sum(reconstruction[:3]).copy_abs()<D('1e-65')
        # Oblique tensor traction/work contraction needs both symmetric xy
        # contributions and zz; D3 and S remain trace-free.
        assert abs(sum(S[:3]))<D('1e-65') and sum(T[:3])==0 and sum(E[:3])==0
    result=dict(mass_Gauss3_vs_Gauss8_max=float(abs(M3-M8).max()),
                complete_Q2_reproduction_max=float(polynomial_error),number_of_Q2_modes=9,
                mature_recurrence_70_digit_error=str(max_error),
                convention='xx,yy,zz=-(xx+yy),xy; full tensor norm counts xy twice',
                cohesive_resistance=0,cohesive_storage=0,auxiliary_Z_is_nonzero=True)
    assert result['mass_Gauss3_vs_Gauss8_max']<1e-14 and polynomial_error<1e-14 and max_error<Decimal('1e-65')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
