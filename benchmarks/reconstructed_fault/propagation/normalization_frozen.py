#!/usr/bin/env python3
"""Frozen Q1 Lorentz normalization: analytic transverse integrals and surface projection.
No simulator state is advanced. The captured native g values check the independent law.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

parser = argparse.ArgumentParser()
parser.add_argument('phase', type=Path)
parser.add_argument('output', type=Path)
a = parser.parse_args()
a.output.mkdir(exist_ok=False, parents=True)
data = np.genfromtxt(a.phase, delimiter=',', names=True)
phi = np.full((33,33), np.nan)
for r in data:
    phi[round(r['x']*32), round(r['y']*32)] = r['phi']
assert np.isfinite(phi).all() and np.min(phi)>=0 and np.max(phi)<=1
native_g = data['g']
g = (1-data['phi'])**2 / ((1-data['phi'])**2+128*data['phi']*(1+data['phi']))
assert np.max(np.abs(g-native_g)) < 4e-16

# Each transverse Q1 restriction is linear on each of the 32 bulk cells.
# u=1-phi, h=128(2/u^2-3/u+1). The exact cell average uses
# <1/u^2>=1/(u0*u1), <1/u>=log(u1/u0)/(u1-u0).
def integral(x):
    xs = np.atleast_1d(x)
    j = np.minimum((xs*32).astype(int),31)
    t = (xs*32-j)[:,None]
    col = ((1-t)*phi[j,:]+t*phi[j+1,:]).astype(np.longdouble)
    u = 1-col
    u0,u1 = u[:,:-1],u[:,1:]
    with np.errstate(divide='ignore',invalid='ignore'):
        z = (u1-u0)/u0
        logarithmic = np.divide(np.log1p(z), z, out=np.ones_like(z), where=z!=0)/u0
        cell = 4*(2/(u0*u1)-3*logarithmic+1)
    cell[(u0==0)|(u1==0)] = np.inf
    result=np.sum(cell,axis=1).astype(float)
    return result[0] if np.isscalar(x) else result

saturated = np.argwhere(phi==1)
report = dict(phase_file=str(a.phase), maximum=float(phi.max()),
              maximum_locations=(np.argwhere(phi==phi.max())/32).tolist(),
              exact_saturation_locations=(saturated/32).tolist(),
              finite_transverse_and_surface_integral=len(saturated)==0,
              native_g_max_difference=float(np.max(abs(g-native_g))), projections=[])
# Uniform dense points plus endpoint-focused points resolve the side regions.
validation=np.unique(np.r_[np.linspace(0,1,10001),np.geomspace(1e-9,.03125,500),
                          1-np.geomspace(1e-9,.03125,500)])
truth=integral(validation)
export=np.unique(np.r_[np.arange(0,len(validation),25),len(validation)-1])
np.savetxt(a.output/'direct.csv',np.c_[validation,truth][export],delimiter=',',header='x,I_direct',comments='')
if not len(saturated):
    total=sum(quad(integral,j/32,(j+1)/32,epsabs=1e-7,epsrel=2e-11)[0] for j in range(32))
    report['direct_surface_integral']=total
else:
    report['mathematical_limit']='At each side endpoint u=A*x+B*abs(y-.5)+O(x*abs(y-.5)), A,B>0. Transverse I(0)=infinity; I(x)~C/x and surface integral diverges logarithmically.'

for spacing in [.1,.05]:
    n=round(1/spacing)
    vertices=np.linspace(0,1,n+1)
    for subdivisions in [1,8,32,128]:
        nodes,weights=leggauss(3);nodes=(nodes+1)/2;weights=weights/2
        xi=np.concatenate([(k+nodes)/subdivisions for k in range(subdivisions)])
        w=np.tile(weights/subdivisions,subdivisions)*spacing
        shape=np.c_[1-xi,xi]
        assert np.min(shape)>=0 and np.min(w)>0 and np.all(shape.sum(axis=1)==1)
        M=np.zeros((n+1,n+1));rhs=np.zeros(n+1);constant=np.zeros(n+1)
        for j in range(n):
            values=integral(vertices[j]+spacing*xi)
            assert np.isfinite(values).all() and min(values)>0
            M[j:j+2,j:j+2]+=shape.T@(w[:,None]*shape)
            rhs[j:j+2]+=shape.T@(w*values)
            constant[j:j+2]+=shape.T@w
        rows=M.sum(axis=1)
        assert min(rows)>0
        lumped=rhs/rows;consistent=np.linalg.solve(M,rhs)
        item=dict(spacing=spacing,surface_subdivisions=subdivisions,
                  minimum_basis=float(shape.min()), minimum_weight=float(w.min()),
                  minimum_row_sum=float(rows.min()),
                  constant_error=float(max(abs(constant/rows-1))),
                  quadrature_surface_integral=float(rhs.sum()),
                  lumped_conservation_relative=float(abs(rows@lumped-rhs.sum())/rhs.sum()),
                  consistent_conservation_relative=float(abs(rows@consistent-rhs.sum())/rhs.sum()),
                  lumped_min=float(min(lumped)),consistent_min=float(min(consistent)))
        interpolated=np.interp(validation,vertices,lumped)
        cons=np.interp(validation,vertices,consistent)
        if not len(saturated):
            ratio=truth/interpolated
            # This is the exact transverse integral of h/I_projected. Its target is one.
            item.update(surface_quadrature_relative_error=abs(rhs.sum()/total-1),
                        lumped_relative_L1=float(np.trapezoid(abs(interpolated-truth),validation)/total),
                        lumped_max_relative_error=float(max(abs(interpolated/truth-1))),
                        lumped_normalized_source_min=float(min(ratio)),
                        lumped_normalized_source_max=float(max(ratio)),
                        lumped_mean_absolute_source_error=float(np.trapezoid(abs(ratio-1),validation)),
                        consistent_relative_L1=float(np.trapezoid(abs(cons-truth),validation)/total),
                        consistent_nonpositive_fraction=float(np.mean(cons<=0)))
        report['projections'].append(item)
        np.savetxt(a.output/f'nodes-{spacing}-{subdivisions}.csv',np.c_[vertices,lumped,consistent,rows],delimiter=',',header='x,lumped,consistent,row_sum',comments='')
        if subdivisions in (1,128):
            np.savetxt(a.output/f'comparison-{spacing}-{subdivisions}.csv',np.c_[validation,truth,cons,interpolated,truth/interpolated][export],delimiter=',',header='x,direct,consistent,lumped,normalized_source_integral',comments='')
# Independent high-order transverse quadrature check on non-singular columns.
# Near saturated endpoints only analytic asymptotics qualify integrability.
nodes,weights=leggauss(64);ys=(np.arange(32)[:,None]+(nodes+1)/2)/32
xs=np.linspace(.01,.99,31);errors=[]
for x in xs:
    j=min(int(x*32),31);t=x*32-j;c=(1-t)*phi[j,:]+t*phi[j+1,:]
    samples=c[:-1,None]+(c[1:]-c[:-1])[:,None]*(nodes+1)/2
    numerical=np.sum(weights/64*128*samples*(1+samples)/(1-samples)**2)
    errors.append(abs(numerical/integral(x)-1))
report['analytic_vs_Q64_transverse_max_relative']=max(errors)
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
