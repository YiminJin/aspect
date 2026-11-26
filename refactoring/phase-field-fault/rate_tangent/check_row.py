"""Compare the reduced production Newton bands to a fixed-parent Q8 directional difference."""
from pathlib import Path
import sys

import numpy as np
extended = np.longdouble

def check(path):
    out=Path(path)/'output'
    row=19
    files=sorted(out.glob('review-points-*'))
    if not files:
        raise ValueError(f"No captured bulk samples in {out}")
    bands=[np.loadtxt(out/f'review-bands-0-{r}') for r in range(len(files))]
    for b in bands[1:]:
        np.testing.assert_array_equal(b,bands[0])
    bands0=bands[0]
    n=len(bands0)
    np.testing.assert_array_equal(bands0[:,0],np.arange(38))
    previous=np.loadtxt(out/'labels-1-0.txt',skiprows=1)
    np.testing.assert_array_equal(bands0[:,4:6],previous[:,[4,6]])
    # Band columns 0--5 give label ID, x, y, normal integral (m), parent
    # age (s) and test speed (m/s). Columns 6--10 give diagonal, upper
    # diagonal, residual, residual scale and Maxwell viscosity (Pa s).
    rates=bands0[:,5].astype(extended)
    direction=rates*(extended('.25')+extended('.45')*np.cos(extended('.6')*(np.arange(n)-row)))
    tangent=np.diag(bands0[:,6])+np.diag(bands0[:-1,7],1)+np.diag(bands0[:-1,7],-1)
    action=tangent.astype(extended)@direction
    action_norm=np.max(np.abs(action))
    kappa=-extended(10)*np.expm1(-extended('.0125'))
    np.testing.assert_allclose(bands0[:,10].astype(float), float(kappa),rtol=1e-15)
    dtype=[('cell','U32'),('q','i4'),('x',extended),('y',extended),('phase',extended),('j','i4'),('xi',extended),('chi',extended),('w',extended),('tau',extended),('sigma',extended)]
    pieces=[np.loadtxt(f,dtype=dtype) for f in files]
    print(path,'active Q8 per rank',[len(p) for p in pieces],'row19 by rank',[np.count_nonzero(np.isin(p['j'],[row-1,row])) for p in pieces])
    p=np.sort(np.concatenate(pieces),order=['cell','q'])
    ids=p[['cell','q']]
    if len(set(zip(p['cell'],p['q']))) != len(p):
        raise ValueError("Duplicate cell/quadrature identities in the reduced section")
    j=p['j']
    vertices=bands0[:,1:3].astype(extended)
    edge=vertices[j+1]-vertices[j]
    x=np.column_stack([p['x'],p['y']])
    xi=np.clip(np.sum((x-vertices[j])*edge,axis=1)/np.sum(edge*edge,axis=1), 0,1)
    print('max |xi_ref - xi|', np.max(abs(xi-p['xi'])))
    left_shape=1-xi
    def interpolate(v):
        return left_shape*v[j]+xi*v[j+1]
    theta=interpolate(bands0[:,4].astype(extended))
    phase=p['phase']
    chi=128*phase*(1+phase)/((1-phase)**2 * interpolate(bands0[:,3].astype(extended)))
    # Q8 points on the 32 × 32 Cartesian grid have tensor-product area weights.
    g=np.polynomial.legendre.leggauss(8)[1].astype(extended)/2
    w=g[p['q']%8]*g[p['q']//8]/(32*32)
    print('phase range active',phase.min(),phase.max(),'chi range',chi.min(),chi.max(),'V range',rates.min(),rates.max(),'parent age range',bands0[:,4].min(),bands0[:,4].max())
    print('largest weight,chi differences',np.max(abs(w-p['w'])),np.max(abs(chi-p['chi'])))
    np.testing.assert_allclose(w,p['w'],rtol=2e-14)
    np.testing.assert_allclose(chi,p['chi'],rtol=2e-13)
    # Independent direct asinh law, with fixed incoming age and normal compression.
    a,b,dc,v0,f0,damp=map(extended,('.025','.015','.008','.001','.6','1'))
    psi=np.exp((f0+b*np.log(theta*v0/dc))/a)/(2*v0)
    # Dropping the inner weight removes the slip-gradient localization.
    # Dropping the outer weight removes it from the Q1 test function.
    def residual(v,outer=True,inner=True):
        speed=interpolate(v)
        resistance=p['sigma']*a*np.arcsinh(psi*speed)+damp*speed
        traction=p['tau']-kappa*(chi if inner else 1)*speed-resistance
        weighted=w*(chi if outer else 1)*traction
        rows=np.zeros(n,dtype=extended)
        np.add.at(rows,j,left_shape*weighted)
        np.add.at(rows,j+1,xi*weighted)
        return rows
    def minus_residual_difference(h,**kw):
        h=extended(h)
        return (residual(rates-h*direction,**kw)-residual(rates+h*direction,**kw))/(2*h)
    rows=residual(rates)
    residual_scale=np.max(bands0[:,9])
    print('base residual absolute and scale',np.max(abs(rows-bands0[:,8])),residual_scale,'Newton residual',np.max(abs(bands0[:,8])/bands0[:,9]))
    print('row19 bands',tangent[row,row-1:row+2], 'rates',rates[row-1:row+2], 'direction',direction[row-1:row+2], 'R',rows[row], 'kappa',kappa)
    print('action scale',action_norm,'row action',action[row])
    for h in ('.1','.01','.001','.0001','.00005','.00001','.000005','.000001','.0000001','.00000001'):
        y=minus_residual_difference(h)
        err=np.max(abs(y-action))
        print(h, 'row minus-FD %.15g row defect %.4g max abs %.4g relative %.4g' %(y[row],abs(y[row]-action[row]),err,err/action_norm))
    y=minus_residual_difference('1e-4')
    half=minus_residual_difference('5e-5')
    est=np.max(abs(y-half))/3
    print('Richardson 5e-5 estimate',est,'extrapolated defect',np.max(abs((4*half-y)/3-action)))
    print('mutations relative',*(np.max(abs(ybad-action))/action_norm for ybad in (-half,minus_residual_difference('5e-5',outer=False),minus_residual_difference('5e-5',inner=False),np.diag(tangent)*direction)))
    later=np.loadtxt(out/'review-bands-1-0')
    correction=np.linalg.solve(tangent, bands0[:,8])
    initial_rates=bands0[:,5]
    below_minimum=initial_rates+correction < 1e-20
    fraction=min(1,min(.9*(initial_rates[below_minimum]-1e-20)/-correction[below_minimum],default=1.))
    print('fraction',fraction,'dense vs next production rates',np.max(abs(initial_rates+fraction*correction-later[:,5])))
    np.testing.assert_allclose(half, action, rtol=0, atol=2e-8*action_norm)
    np.testing.assert_allclose(rows, bands0[:,8], rtol=0, atol=1e-11*residual_scale)
    np.testing.assert_allclose(initial_rates+fraction*correction, later[:,5], rtol=0, atol=1e-13)
    return ids,bands0,p,action
if len(sys.argv) not in (2,3):
    raise SystemExit("usage: check_row.py SERIAL_RUN [MPI_RUN]")
results=[check(x) for x in sys.argv[1:]]
if len(results)==2:
    (ids,b,p,a),(ids2,b2,p2,a2)=results
    np.testing.assert_array_equal(ids,ids2)
    np.testing.assert_array_equal(p['j'],p2['j'])
    np.testing.assert_array_equal(b[:,0],b2[:,0])
    for key in ('x','y','phase','xi','chi','w','tau','sigma'):
        np.testing.assert_allclose(p[key],p2[key],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(b,b2,rtol=1e-8,atol=1e-10)
    print('serial / MPI max abs bands (D,U)',np.max(abs(b[:,6:8]-b2[:,6:8])),'max abs action',np.max(abs(a-a2)))
