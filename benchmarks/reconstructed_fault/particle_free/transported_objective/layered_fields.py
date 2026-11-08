"""Rectangular Cartesian Q2 diagnostic reconstruction, without square-box assumptions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_material_native import State as LegacyState,basis,read as plain_read,NAMES
import csv
import gzip

def read(path):
    """Read retained CSV observations, including losslessly compacted new runs."""
    if not path.exists() and path.with_suffix(path.suffix+".gz").exists():
        path=path.with_suffix(path.suffix+".gz")
    if path.suffix==".gz":
        with gzip.open(path,"rt") as stream:return list(csv.DictReader(stream))
    return plain_read(path)

from compare_bounded_profile import dot3
class State:
    memory=LegacyState.memory
    def __init__(self,path,step,h,m=50.):
        self.h=h;self.m=m;self.direct=True
        meta=read(path/"material-model.csv")[0]
        self.background_shear=float(meta.get("background_shear",2.7));self.sense=float(meta['sense'])
        records=[r for p in sorted(path.glob(f'material-{step}-rank*.csv*')) for r in read(p)]
        records.sort(key=lambda r:(r['cell'],int(r['q'])))
        self.x=np.array([[float(r['x']),float(r['y'])] for r in records])
        self.values=np.array([[float(r[k]) for k in NAMES+['pressure']] for r in records])
        centers=self.x.reshape(-1,9,2).mean(axis=1)
        self.lo=centers.min(axis=0)-h/2;self.hi=centers.max(axis=0)+h/2
        self.lo=np.round(self.lo,10);self.hi=np.round(self.hi,10)
        self.shape=np.rint((self.hi-self.lo)/h).astype(int)
        self.coeff=np.zeros((*self.shape,9,18));ij=np.rint((centers-self.lo)/h-.5).astype(int)
        inv=np.linalg.inv(basis(2*(self.x[:9]-centers[0])/h))
        self.coeff[ij[:,0],ij[:,1]]=np.einsum('ij,cjk->cik',inv,self.values.reshape(-1,9,18))
        self.surface=np.array([[float(v) for v in r.values()] for r in read(path/f'material-{step}-surface.csv')])
    def at(self,x):
        x=np.asarray(x);u=(x-self.lo)/self.h;ij=np.floor(u).astype(int)
        ij=np.maximum(0,np.minimum(self.shape-1,ij));valid=np.all((x>=self.lo-1e-9)&(x<=self.hi+1e-9),axis=1)
        out=np.zeros((len(x),18));a=ij[valid]
        out[valid]=np.einsum('ni,nij->nj',basis(2*(u[valid]-a)-1),self.coeff[a[:,0],a[:,1]])
        return out
    def stress(self,x):return self.at(x)[:,3:6]
    def quadrature(self,order=8):
        g,w=np.polynomial.legendre.leggauss(order)
        axes=[(self.lo[j]+self.h*(np.arange(self.shape[j])[:,None]+.5*(g+1))).ravel() for j in (0,1)]
        ww=[np.tile(.5*self.h*w,self.shape[j]) for j in (0,1)]
        return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2),np.outer(*ww).ravel()
    def faces(self,order=8):
        g,w=np.polynomial.legendre.leggauss(order)
        for axis in (0,1):
            other=1-axis;line=(self.lo[other]+self.h*(np.arange(self.shape[other])[:,None]+.5*(g+1))).ravel()
            weights=np.tile(.5*self.h*w,self.shape[other])
            for side in (-1,1):
                x=np.zeros((len(line),2));x[:,axis]=(self.lo if side<0 else self.hi)[axis];x[:,other]=line
                yield axis,side,x,weights

def localization(s,x,v):
    X=s.surface[:,1:3];I=s.surface[:,5];width=s.surface[:,7];best=np.full(len(x),np.inf)
    owner=np.full(len(x),-1);qvalues=np.zeros(len(x));chi=np.zeros(len(x));shear=np.zeros(len(x));normal=np.zeros(len(x))
    for j,d in enumerate(np.diff(X,axis=0)):
        t=d/np.linalg.norm(d);n=np.array([-t[1],t[0]]);q=(x-X[j])@d/(d@d);dist=abs((x-X[j])@n)
        mask=(q>=0)&(q<=1)&(dist<best)&(dist<=(1-q)*width[j]+q*width[j+1]);u=q[mask];p=v[mask,0];tau=v[mask,3:6]
        owner[mask]=j;qvalues[mask]=u;chi[mask]=s.m*p*(1+p)/(1-p)**2/((1-u)*I[j]+u*I[j+1])
        shear[mask]=t[0]*(tau[:,0]*n[0]+tau[:,2]*n[1])+t[1]*(tau[:,2]*n[0]+tau[:,1]*n[1])
        normal[mask]=tau[:,0]*n[0]**2+tau[:,1]*n[1]**2+2*tau[:,2]*n[0]*n[1];best[mask]=dist[mask]
    return owner,qvalues,chi,shear,normal

def work_traction(s,order=8):
    x,w=s.quadrature(order);v=s.at(x);owner,q,chi,shear,_=localization(s,x,v)
    mask=owner>=0;j=owner[mask];u=q[mask];ww=w[mask]*chi[mask];size=len(s.surface)
    add=lambda a:np.bincount(j,weights=a,minlength=size-1)
    diagonal=np.zeros(size);diagonal[:-1]+=add(ww*(1-u)**2);diagonal[1:]+=add(ww*u*u)
    off=add(ww*u*(1-u));M=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    rhs=np.zeros(size);rhs[:-1]+=add(ww*(1-u)*shear[mask]);rhs[1:]+=add(ww*u*shear[mask])
    return np.linalg.solve(M,rhs),float(w@chi)

def surface(s,labels,cumulative=None,work=None):
    X=s.surface[:,1:3];count=len(X)-1;j=np.minimum((labels*count).astype(int),count-1);q=labels*count-j
    d=X[j+1]-X[j];length=np.linalg.norm(d,axis=1);t=d/length[:,None];n=np.stack([-t[:,1],t[:,0]],axis=1)
    points=X[j]+q[:,None]*d;v=s.at(points);tau=v[:,3:6]
    normal=tau[:,0]*n[:,0]**2+tau[:,1]*n[:,1]**2+2*tau[:,2]*n[:,0]*n[:,1]
    interp=lambda a:(1-q)*a[j]+q*a[j+1]
    if work is None:work=work_traction(s)[0]
    result=dict(stress=tau,geometry=points,jacobian=count*length,normal_traction=normal-v[:,17],
      shear_traction=t[:,0]*(tau[:,0]*n[:,0]+tau[:,2]*n[:,1])+t[:,1]*(tau[:,2]*n[:,0]+tau[:,1]*n[:,1]),
      work_shear_traction=interp(work),slip_rate=interp(s.surface[:,6]),age=interp(s.surface[:,3]),I=interp(s.surface[:,5]))
    # The common fixture's original straight segment has uniform material labels.
    # Loading remains attached to those labels, not to the moved x coordinate.
    background=s.background_shear+.15*np.cos(3*np.linspace(-.8,.8,len(X)))
    result['effective_compression']=5-result['normal_traction']
    result['driving_shear_traction']=result['shear_traction']+interp(background)
    result['driving_work_shear_traction']=interp(work+background)
    if cumulative is not None:result['cumulative_slip']=interp(np.asarray(cumulative))
    return result

def powers(s,old,initial,order=8):
    x,w=s.quadrature(order);v=s.at(x);tau=v[:,3:6];tr=v[:,12]+v[:,15]
    D=np.stack([v[:,12]-tr/3,v[:,15]-tr/3,.5*(v[:,13]+v[:,14])],axis=1)
    eta=5.5+4.5*np.tanh(x[:,1]/.35);psi=dot3(tau,tau)/40
    owner,q,chi,shear,normal=localization(s,x,v);mask=owner>=0;j=owner[mask];u=q[mask]
    interp=lambda a:(1-u)*a[j]+u*a[j+1]
    V=interp(s.surface[:,6]);theta=interp(old.surface[:,3]);background=interp(initial.background_shear+.15*np.cos(3*initial.surface[:,1]))
    mu=.025*np.arcsinh(V/.002*np.exp((.6+.015*np.log(theta*.001/.008))/.025));compression=5+v[mask,17]-normal[mask]
    ww=w[mask]*chi[mask]
    e=dict(storage=float(w@psi),mechanical_power=float(w@dot3(tau,D)),maxwell_loss=float(w@(dot3(tau,tau)/(2*eta))),
      dilatation_power=float(w@(psi*tr)),pressure_dilatation=float(w@(v[:,17]*tr)),
      slip_power=float(ww@(V*shear[mask])),background_power=float(ww@(V*background)),
      friction_power=float(ww@(V*mu*compression)),radiation_power=float(ww@(V*V)),localization_measure=float(w@chi),
      profile_mass=float(w@v[:,0]),peak_phase=float(v[:,0].max()),stress_L2=float(np.sqrt(w@dot3(tau,tau))))
    omega=.5*(v[mask,14]-v[mask,13]);omega_f=-.5*s.sense*chi[mask]*V
    band=chi[mask]>0  # Positive localization support; no artificial amplitude cutoff.
    tendency=np.sqrt(2.)*abs(omega_f)*np.hypot(tau[mask,0]-tau[mask,1],2*tau[mask,2])
    e.update(sampled_max_abs_omega_f=float(np.max(abs(omega_f))),
             sampled_max_abs_omega=float(np.max(abs(omega[band]))),sampled_max_abs_omega_b=float(np.max(abs((omega-omega_f)[band]))),
             sampled_max_chi=float(chi.max()),weighted_rotation_tendency_L2=float(np.sqrt(w[mask]@(tendency*tendency))))
    e['phase_dilatation']=float(w@(v[:,0]*tr))
    center=np.sum((w*v[:,0])[:,None]*x,axis=0)/e['profile_mass'];d=s.surface[-1,1:3]-s.surface[0,1:3];n=np.array([-d[1],d[0]])/np.linalg.norm(d)
    e['profile_centroid']=center.tolist();e['profile_normal_variance']=float((w*v[:,0])@((x-center)@n)**2/e['profile_mass'])
    flux=face=phase_flux=0.;force=np.zeros(2)
    for axis,side,points,weights in s.faces(order):
        a=s.at(points);t=a[:,3:6];vn=side*a[:,axis+1];flux+=weights@(vn*dot3(t,t)/40);phase_flux+=weights@(vn*a[:,0])
        traction=np.stack([t[:,0]-a[:,17],t[:,2]],axis=1) if axis==0 else np.stack([t[:,2],t[:,1]-a[:,17]],axis=1)
        face+=weights@(side*np.sum(a[:,1:3]*traction,axis=1));force+=np.sum((side*weights)[:,None]*traction,axis=0)
    e.update(flux=float(flux),phase_flux=float(phase_flux),face_mechanical_power=float(face),weak_mechanical_power=e['mechanical_power']-e['pressure_dilatation'],face_force=force.tolist())
    e['face_weak_defect']=e['face_mechanical_power']-e['weak_mechanical_power']
    e['surface_work_residual']=e['slip_power']+e['background_power']-e['friction_power']-e['radiation_power']
    e['normalization_relative']=e['localization_measure']/np.linalg.norm(np.diff(s.surface[:,1:3],axis=0),axis=1).sum()-1
    return e

def surface_rule(states,order=6):
    breaks=[0.,.2,.8,1.]
    for s in states:
        X=s.surface[:,1:3];label=np.linspace(0,1,len(X));breaks.extend(label)
        for j,d in enumerate(np.diff(X,axis=0)):
            for axis in (0,1):
                if abs(d[axis])<1e-15:continue
                q=(np.linspace(s.lo[axis],s.hi[axis],s.shape[axis]+1)-X[j,axis])/d[axis]
                q=q[(q>0)&(q<1)];breaks.extend(label[j]+q*(label[j+1]-label[j]))
    b=np.unique(breaks);g,w=np.polynomial.legendre.leggauss(order)
    return (b[:-1,None]+.5*(g+1)*np.diff(b)[:,None]).ravel(),(.5*np.diff(b)[:,None]*w).ravel()
