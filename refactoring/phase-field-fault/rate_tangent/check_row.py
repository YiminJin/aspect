"""Compare the reduced production Newton bands to a fixed-parent Q8 directional difference."""
from pathlib import Path
import sys, numpy as np
F=np.longdouble
def check(path):
    out=Path(path)/'output'; row=19
    files=sorted(out.glob('review-points-*'))
    if not files:
        raise ValueError(f"No captured bulk samples in {out}")
    bands=[np.loadtxt(out/f'review-bands-0-{r}') for r in range(len(files))]
    for b in bands[1:]: np.testing.assert_array_equal(b,bands[0])
    B=bands[0]; n=len(B)
    np.testing.assert_array_equal(B[:,0],np.arange(38))
    previous=np.loadtxt(out/'labels-1-0.txt',skiprows=1)
    np.testing.assert_array_equal(B[:,4:6],previous[:,[4,6]])
    V=B[:,5].astype(F)
    direction=V*(F('.25')+F('.45')*np.cos(F('.6')*(np.arange(n)-row)))
    A=np.diag(B[:,6])+np.diag(B[:-1,7],1)+np.diag(B[:-1,7],-1)
    action=A.astype(F)@direction
    anorm=np.max(np.abs(action))
    kappa=-F(10)*np.expm1(-F('.0125'))
    np.testing.assert_allclose(B[:,10].astype(float), float(kappa),rtol=1e-15)
    dtype=[('cell','U32'),('q','i4'),('x',F),('y',F),('phase',F),('j','i4'),('xi',F),('chi',F),('w',F),('tau',F),('sigma',F)]
    pieces=[np.loadtxt(f,dtype=dtype) for f in files]
    print(path,'active Q8 per rank',[len(p) for p in pieces],'row19 by rank',[np.count_nonzero(np.isin(p['j'],[row-1,row])) for p in pieces])
    p=np.sort(np.concatenate(pieces),order=['cell','q'])
    ids=p[['cell','q']]
    if len(set(zip(p['cell'],p['q']))) != len(p):
        raise ValueError("Duplicate cell/quadrature identities in the reduced section")
    j=p['j']
    X=B[:,1:3].astype(F); e=X[j+1]-X[j]
    x=np.column_stack([p['x'],p['y']])
    t=np.clip(np.sum((x-X[j])*e,axis=1)/np.sum(e*e,axis=1), 0,1)
    print('max |xi_ref - xi|', np.max(abs(t-p['xi'])))
    L=1-t
    def interp(v): return L*v[j]+t*v[j+1]
    theta=interp(B[:,4].astype(F))
    phase=p['phase']; chi=128*phase*(1+phase)/((1-phase)**2 * interp(B[:,3].astype(F)))
    g=np.polynomial.legendre.leggauss(8)[1].astype(F)/2
    w=g[p['q']%8]*g[p['q']//8]/(32*32)
    print('phase range active',phase.min(),phase.max(),'chi range',chi.min(),chi.max(),'V range',V.min(),V.max(),'parent age range',B[:,4].min(),B[:,4].max())
    print('largest weight,chi differences',np.max(abs(w-p['w'])),np.max(abs(chi-p['chi'])))
    np.testing.assert_allclose(w,p['w'],rtol=2e-14)
    np.testing.assert_allclose(chi,p['chi'],rtol=2e-13)
    # Independent direct asinh law, with fixed incoming age and normal compression.
    a,b,dc,v0,f0,damp=map(F,('.025','.015','.008','.001','.6','1'))
    psi=np.exp((f0+b*np.log(theta*v0/dc))/a)/(2*v0)
    def residual(v,outer=True,inner=True):
        speed=interp(v)
        resistance=p['sigma']*a*np.arcsinh(psi*speed)+damp*speed
        traction=p['tau']-kappa*(chi if inner else 1)*speed-resistance
        weighted=w*(chi if outer else 1)*traction
        R=np.zeros(n,dtype=F)
        np.add.at(R,j,L*weighted)
        np.add.at(R,j+1,t*weighted)
        return R
    def fd(h,**kw):
        h=F(h)
        return (residual(V-h*direction,**kw)-residual(V+h*direction,**kw))/(2*h)
    R=residual(V)
    denom=np.max(B[:,9])
    print('base residual absolute and scale',np.max(abs(R-B[:,8])),denom,'Newton residual',np.max(abs(B[:,8])/B[:,9]))
    print('row19 bands',A[row,row-1:row+2], 'rates',V[row-1:row+2], 'direction',direction[row-1:row+2], 'R',R[row], 'kappa',kappa)
    print('action scale',anorm,'row action',action[row])
    for h in ('.1','.01','.001','.0001','.00005','.00001','.000005','.000001','.0000001','.00000001'):
        y=fd(h); err=np.max(abs(y-action))
        print(h, 'row minus-FD %.15g row defect %.4g max abs %.4g relative %.4g' %(y[row],abs(y[row]-action[row]),err,err/anorm))
    y=fd('1e-4'); half=fd('5e-5'); est=np.max(abs(y-half))/3
    print('Richardson 5e-5 estimate',est,'extrapolated defect',np.max(abs((4*half-y)/3-action)))
    print('mutations relative',*(np.max(abs(ybad-action))/anorm for ybad in (-half,fd('5e-5',outer=False),fd('5e-5',inner=False),np.diag(A)*direction)))
    later=np.loadtxt(out/'review-bands-1-0')
    dv=np.linalg.solve(A, B[:,8]); v0=B[:,5]; neg=v0+dv < 1e-20
    fraction=min(1,min(.9*(v0[neg]-1e-20)/-dv[neg],default=1.))
    print('fraction',fraction,'dense vs next production rates',np.max(abs(v0+fraction*dv-later[:,5])))
    np.testing.assert_allclose(half, action, rtol=0, atol=2e-8*anorm)
    np.testing.assert_allclose(R, B[:,8], rtol=0, atol=1e-11*denom)
    np.testing.assert_allclose(v0+fraction*dv, later[:,5], rtol=0, atol=1e-13)
    return ids,B,p,action
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
