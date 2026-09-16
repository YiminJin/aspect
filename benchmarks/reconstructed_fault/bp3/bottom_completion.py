"""Bottom-only virtual Q1 profile completion; no outside mechanical domain."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_banded
from analyze_uniform_sliding import read,cat,write

HERE=Path(__file__).resolve().parent
BASE=HERE/'uniform-sliding-50-local4'
OUT=HERE/'bottom-completion-50-local4'
sn=np.sqrt(3)/2
xt=50000*(1+.5/sn)
normal=np.array([-sn,.5])
profile=read(BASE/'uniform_reference_profile.csv')
mesh=cat(BASE.glob('initial_mesh_*.csv'),('cell',))
near=(mesh['y']<2000)&(mesh['distance']<700)
spacing=np.unique(mesh['h'][near])
assert len(spacing)==1
h=float(spacing[0])
radius=float(profile['r'].max())
# m is recovered from the actual exported g, then checked at every profile
# sample. Both material phases have the same degradation in this fixture.
phi=profile['phi'];valid=phi>1e-6
m=float(profile['h'][0]*(1-phi[0])**2/(phi[0]*(1+phi[0])))
np.testing.assert_allclose(profile['h'][valid],m*phi[valid]*(1+phi[valid])/(1-phi[valid])**2,rtol=1e-9)

def q1_phi(x,y):
    x,y=np.broadcast_arrays(x,y);i=np.floor(x/h);j=np.floor(y/h)
    a=x/h-i;b=y/h-j;value=np.zeros(x.shape)
    for u in (0,1):
        for v in (0,1):
            r=(xt-(i+u)*h)*sn-(100000-(j+v)*h)*.5
            nodal=np.interp(abs(r),profile['r'],phi,right=0)
            value+=((a if u else 1-a)*(b if v else 1-b))*nodal
    return value

def integrand(origin,r):
    p=origin[:,None]+normal[:,None]*r
    f=q1_phi(p[0],p[1]);return m*f*(1+f)/(1-f)**2

def integrate(origin,lo,hi,order=8):
    if lo>=hi:return 0.
    cuts=[lo,hi]
    for d in (0,1):
        ends=origin[d]+normal[d]*np.array([lo,hi])
        for j in range(int(np.floor(min(ends)/h)),int(np.ceil(max(ends)/h))+1):
            r=(j*h-origin[d])/normal[d]
            if lo<r<hi:cuts.append(r)
    cuts=np.unique(cuts);nodes,weights=leggauss(order)
    def panel(a,b):
        r=.5*(a+b)+.5*(b-a)*nodes
        return .5*(b-a)*np.dot(weights,integrand(origin,r))
    def adaptive(a,b,depth=0):
        low=panel(a,b);mid=.5*(a+b);high=panel(a,mid)+panel(mid,b)
        if abs(high-low)<=1e-11*max(1.,abs(high)):return high
        assert depth<20
        return adaptive(a,mid,depth+1)+adaptive(mid,b,depth+1)
    return sum(adaptive(a,b) for a,b in zip(cuts[:-1],cuts[1:]))

def prepare():
    OUT.mkdir()
    saved=cat(BASE.glob('uniform_bulk_0_rank*.csv'),('cell',))
    mask=saved['y']<2000
    error=float(np.max(abs(q1_phi(saved['x'][mask],saved['y'][mask])-saved['phi'][mask])))
    assert error<2e-12,('Virtual Q1 does not match saved bottom FE',error)
    f=read(BASE/'fault_0.csv');vertices=np.array([f['x'],f['y']]).T
    nodes,weights=leggauss(3);nodes=(nodes+1)/2;weights=weights/2
    # Q1 can remain nonzero up to a cell's projected diameter beyond the
    # stationary nodal support. This enclosure is geometric, not a cutoff tune.
    extent=radius+h*(abs(normal[0])+abs(normal[1]))
    rows=[];max_error=0.
    for j in range(len(vertices)-1):
        length=np.linalg.norm(vertices[j+1]-vertices[j])
        for q,z in enumerate(nodes):
            p=(1-z)*vertices[j]+z*vertices[j+1]
            limit=-p[1]/normal[1]
            missing=integrate(p,-extent,min(limit,extent)) if limit>-extent else 0.
            check=integrate(p,-extent,min(limit,extent),16) if missing else 0.
            max_error=max(max_error,abs(missing-check))
            rows.append([3*j+q,*p,missing,j,z,length*weights[q]])
    rows=np.array(rows)
    with (OUT/'completion.txt').open('w') as out:
        out.write(str(len(rows))+'\n')
        np.savetxt(out,rows[:,:4],fmt=['%d','%.17g','%.17g','%.17g'])
    write(OUT/'completion_profiles.csv',dict(zip(['id','x','y','outside','segment','xi','surface_weight'],rows.T)))
    # Complementary in/out integration must reproduce each full Q1 column.
    checks=[]
    for p in [np.array([xt-(100000-y)/sn*.5,y]) for y in (0.,100.,300.,500.,1000.,1500.)]:
        limit=-p[1]/normal[1]
        inside=integrate(p,max(-extent,limit),extent)
        outside=integrate(p,-extent,min(limit,extent)) if limit>-extent else 0.
        full=integrate(p,-extent,extent)
        assert abs(inside+outside-full)<1e-6
        checks.append(dict(y=float(p[1]),inside=inside,outside=outside,full=full,error=inside+outside-full))
    # Constant full-profile reference: exactly the actual Q1 mass weights,
    # including nonuniform fault elements. M*constant = projected RHS.
    n=len(vertices);band=np.zeros((3,n));rhs=np.zeros(n);constant=checks[-1]['full']
    for row in rows:
        j=int(row[4]);z=row[5];w=row[6];N=np.array([1-z,z])
        band[1,j:j+2]+=w*N*N;band[0,j+1]+=w*N[0]*N[1];band[2,j]+=w*N[0]*N[1]
        rhs[j:j+2]+=w*N*constant
    constant_error=float(np.max(abs(solve_banded((1,1),band,rhs)-constant)))
    assert constant_error<1e-9 and max_error<1e-6
    assert np.all(rows[rows[:,2]>=extent*normal[1],3]==0)
    (OUT/'preflight.json').write_text(json.dumps(dict(spacing=h,radius=radius,enclosure=extent,
        q1_match_max_phi_error=error,order_check_max_integral_error=max_error,
        constant_projection_error=constant_error,nonzero_profiles=int(np.count_nonzero(rows[:,3])),
        max_nonzero_height=float(np.max(rows[rows[:,3]>0,2])),columns=checks),indent=2)+'\n')
    print((OUT/'preflight.json').read_text())

if __name__=='__main__':prepare()
