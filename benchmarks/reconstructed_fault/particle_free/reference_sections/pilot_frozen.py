"""Read-only localized score on the certified native horizontal midpoint archives.

Restricted to the recorded 256 x 128, six-Gauss, 16-double point schema. Check
identity and validate the native geometry, section rates and published planes
before interpreting the result. Never import this archive into the new law.
"""
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def checkpoint(path,manifest):
    packed=path.read_bytes()
    assert hashlib.sha256(packed).hexdigest()==manifest[path.resolve()]
    chunks,size,_,compressed=struct.unpack_from('<4I',packed)
    assert chunks==1 and len(packed)==16+compressed
    data=zlib.decompress(packed[16:])
    assert len(data)==size and data.count(b'bulk tip physical tau A material strips midpoint v1')==1
    def vector(n):
        marker=struct.pack('<Q',n)
        off=data.index(marker)
        assert data.find(marker,off+1)<0
        return np.frombuffer(data,dtype='<f8',offset=off+8,count=n)
    return vector(32768*593).reshape(-1,593),vector(1536*8).reshape(-1,8)


def scan(folder,parent_path,end_path,step,dt,manifest):
    old,stations=checkpoint(parent_path,manifest)
    end,endstations=checkpoint(end_path,manifest)
    planes=np.genfromtxt(folder/f'tip-planes-{step:04}.csv',delimiter=',',names=True)
    assert np.array_equal(planes['x'],stations[:,0]) and np.array_equal(planes['angle'],endstations[:,6])
    assert np.array_equal(stations[:,7],planes['parent_formed']) and np.array_equal(planes['age'],endstations[:,2])
    assert np.array_equal(old[:,:8],end[:,:8])
    h=1/32
    assert np.all(old[:,2]-old[:,0]==h) and np.all(old[:,5]-old[:,1]==h)
    y_order=np.lexsort((old[:,1],old[:,0]))
    assert np.all(old[y_order,1].reshape(256,128)==np.arange(128)*h)
    def cells(a): return a[y_order].reshape(256,128,593)
    old,end=cells(old),cells(end)
    gauss,weight=np.polynomial.legendre.leggauss(6)
    u=(gauss+1)/2
    B=np.stack(((1-u)**2,2*u*(1-u),u**2),axis=1)
    G=1e6
    beta=1/(1+dt*G/1e8)
    ell=.125
    m=(2/3)/((8/3)*ell)/(1000**2/(2*G))
    radii=ell*np.array([1.,2.,4.])
    candidates=[[] for _ in radii]
    max_rate_error=0.
    yy=(np.arange(128)[:,None]+u)*h
    for k in np.flatnonzero((stations[:,0]>6)&(stations[:,0]<6.5)&(stations[:,7]==0)):
        col,g=divmod(k,6)
        x,ds,age,_,_,_,incoming,_=stations[k]
        assert abs(x-(col+u[g])*h)<1e-14
        dx=ds*np.cos(incoming)
        assert abs(dx-h*weight[g]/2)<1e-14
        area=np.broadcast_to(dx*h*weight[None,:]/2,yy.shape)
        def history(a): return a[col,:,17:].reshape(128,6,6,16)[:,:,g,:]
        H0,H1=history(old),history(end)
        assert np.max(abs(H0[:,:,6:12]))==0.
        rate=np.sum(area*H1[:,:,14])/endstations[k,1]
        max_rate_error=max(max_rate_error,abs(rate-endstations[k,4]))
        def pred(i):return .5*(H0[:,:,i]+H1[:,:,i])+G*(H1[:,:,6+i]+(1-2*beta)*H0[:,:,6+i])
        p,q=.5*(pred(3)-pred(0)),pred(1)
        coeff=.5*(old[col,:,8:17]+end[col,:,8:17]).reshape(128,3,3)
        phi=np.einsum('hji,qj,i->hq',coeff,B,B[g])
        F=m*phi*(1+phi)/(m*phi*(1+phi)+(1-phi)**2)
        r2=(x-6)**2+(yy-2)**2
        kernel=np.maximum(0,1-r2[None,:,:]/radii[:,None,None]**2)**2
        angles=np.radians(np.linspace(-45,45,181))
        scores=[]
        for a in angles:
            tau=p*np.sin(2*a)+q*np.cos(2*a)
            dn=area*np.cos(a)/dx
            M=np.sum(dn*F)/(beta*G*dt)
            free=np.sum(dn*F*tau)/(beta*G*dt)
            def resistance(v):
                Dc=3.2211126126299625e-7
                y=dt*abs(v)/Dc
                theta=age*np.exp(-y)+dt*(-np.expm1(-y)/y if y else 1.)
                return 3*np.arcsinh(v/.002*np.exp((.6+.02*np.log(.001*theta/Dc))/.015))+1000*v
            V=brentq(lambda V:V+M*resistance(V)-free,min(0.,free),max(0.,free),xtol=1e-35) if free else 0.
            scores.append(np.sum(kernel*(area*((1-F)*(tau-resistance(V)))**2)[None,:,:],axis=(1,2))*m/(2*G))
        scores=np.array(scores)
        for j in range(3):
            if scores[:,j].max()>0:
                i=np.argmax(scores[:,j])
                candidates[j].append((x,float(scores[i,j]),float(np.degrees(angles[i]))))
    assert max_rate_error<1e-12
    return dict(max_rate_reconstruction_error=max_rate_error,eligible=[len(c) for c in candidates],
                strongest=[max(c,key=lambda v:v[1]) if c else None for c in candidates])


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('manifest',type=Path)
    args=a.parse_args()
    files={Path(k).resolve():v for k,v in json.loads(args.manifest.read_text()).items()}
    root=Path('build-particle-free/overnight-recovery')
    coarse=root/'rate-true-residual/output'
    fine=root/'rate-final40/output'
    def snap(folder,i):return folder/f'restart/{i:02}/resume.z'
    # The accepted step 79 is snapshot 02 of the coarse run; it is also
    # the common parent of the two finer intervals. Verify each endpoint.
    # Intermediate fine endpoint, additionally pinned and checked against its
    # native station table. The original manifest covers the common parent
    # and both terminal snapshots.
    files[snap(fine,3).resolve()]="c3627a4812c768beb3c518fa49a51e0aca70f71fc59c9c70c0c8852e085e7c0b"
    print(json.dumps(dict(coarse=scan(coarse,snap(coarse,2),snap(coarse,3),80,4e-5,files),
                          fine=scan(fine,snap(fine,3),snap(fine,1),81,2e-5,files)),indent=2))
