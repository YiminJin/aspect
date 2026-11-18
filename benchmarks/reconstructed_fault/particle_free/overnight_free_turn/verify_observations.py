#!/usr/bin/env python3
"""Independent analytic checks of rectangular Q2 cuts, DG traces and L2 measures."""
from observations import *
import copy

results=[]
for hy in [.03125,.015625]:
    hx=.0625
    poly=RectBernstein.__new__(RectBernstein)
    poly.hx,poly.hy,poly.nx,poly.ny=hx,hy,128,round(4/hy)
    x=np.arange(poly.nx)*hx-4;y=np.arange(poly.ny)*hy-2
    xx=np.array([x*x,x*(x+hx),(x+hx)**2]).T/2.2**2
    yy=np.array([y*y,y*(y+hy),(y+hy)**2]).T/.2**2
    poly.coefficients=1.05-yy[:,None,:,None]-xx[None,:,None,:]
    result=front(poly,poly)
    assert abs(result['point'][0]-6.2)<5e-11 and abs(result['point'][1]-2)<1e-8
    jumped=copy.copy(poly);jumped.coefficients=poly.coefficients.copy()
    jumped.coefficients[poly.ny//2:]+=.02
    jump=front(jumped,jumped);expected=4+2.2*np.sqrt(1.02)
    assert abs(jump['point'][0]-expected)<5e-11 and abs(jump['point'][1]-2)<1e-8
    assert jump['cell'][1]==poly.ny//2,'Must select the upper native trace, not an average'
    results.append(dict(hy=hy,continuous=result,discontinuous=jump,expected_discontinuous_x=float(expected)))
a=copy.copy(poly);a.coefficients=np.ones_like(poly.coefficients)
b=copy.copy(poly);b.hy=.03125;b.ny=128;b.coefficients=np.zeros((128,128,3,3))
norm=l2(a,b,parent=b)
assert abs(norm['absolute']-np.sqrt(32))<2e-13 and abs(norm['relative_increment']-1)<2e-14
(OUT/'observation-verification.json').write_text(json.dumps(dict(fronts=results,constant_field_L2=norm),indent=2)+'\n')
print('Analytic ellipse fronts, separate DG face traces and common-area L2 measures pass.')
