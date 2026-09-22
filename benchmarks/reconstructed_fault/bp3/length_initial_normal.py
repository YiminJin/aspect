"""Locate initial normal-stress extrema offline from high-order VTU velocity.

At t=0 the working Maxwell history is zero and S:N=0, so sigma_n depends
only on p and the normal strain. Published tau fields are NOT current stress.
Float32 visualization makes this a location check, not a convergence norm.
"""
import argparse
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from length_coupled import OUT
from run_mechanical_width import G,SN,XT,stationary


def audit(label):
    run=OUT/label; kappa=-1e26*np.expm1(-4e6*G/1e26)
    z=(leggauss(3)[0]+1)/2
    basis=np.array([2*(z-.5)*(z-1),-4*z*(z-1),2*z*(z-.5)]).T
    deriv=np.array([4*z-3,4-8*z,4*z-1]).T
    phi_radius=stationary(50)[0][-1]
    rows=[]
    for path in sorted((run/'solution').glob('solution-00000.*.vtu')):
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
        grid=reader.GetOutput();data=grid.GetPointData()
        assert np.all(vtk_to_numpy(grid.GetCellTypesArray())==70)
        ids=vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1,9)
        xyz=vtk_to_numpy(grid.GetPoints().GetData()).astype(float)[ids,:2]
        # Cartesian origins/h are exactly known binary subdivisions. Restore
        # those coordinates, not noisy Float32 differences of large positions.
        h=50000./2**np.rint(np.log2(50000./np.ptp(xyz[:,:,0],axis=1)))
        x=-100000.+np.rint((xyz[:,:,0].min(axis=1)+100000.)/h)*h
        y=np.rint(xyz[:,:,1].min(axis=1)/h)*h
        r=(XT-x-.5*h)*SN-(100000-y-.5*h)*.5
        keep=abs(r)<=phi_radius+.5*(SN+.5)*h
        ids=ids[keep];xyz=xyz[keep];x=x[keep];y=y[keep];h=h[keep]
        local=np.rint(2*(xyz-np.stack([x,y],axis=1)[:,None,:])/h[:,None,None]).astype(int)
        index=local[:,:,0]+3*local[:,:,1]
        order=np.argsort(index,axis=1);ids=np.take_along_axis(ids,order,axis=1)
        assert np.all(np.take_along_axis(index,order,axis=1)==np.arange(9))
        velocity=vtk_to_numpy(data.GetArray('velocity')).astype(float)[ids,:2].reshape(-1,3,3,2)
        p=vtk_to_numpy(data.GetArray('p')).astype(float)[ids].reshape(-1,3,3)
        phi=vtk_to_numpy(data.GetArray('phase_field')).astype(float)[ids].reshape(-1,3,3)
        for name in ['tau_xx','tau_yy','tau_xy']:
            assert np.max(abs(vtk_to_numpy(data.GetArray(name))))==0
        # Axes are cell, normal tensor-grid y, x, vector component.
        dx=np.einsum('aj,bk,cjkv->cabv',basis,deriv,velocity)/h[:,None,None,None]
        dy=np.einsum('aj,bk,cjkv->cabv',deriv,basis,velocity)/h[:,None,None,None]
        pressure=np.einsum('aj,bk,cjk->cab',basis,basis,p)
        phase=np.einsum('aj,bk,cjk->cab',basis,basis,phi)
        normal_rate=.75*dx[:,:,:,0]+.25*dy[:,:,:,1]-.5*SN*(dx[:,:,:,1]+dy[:,:,:,0])
        tauN=2*kappa*normal_rate; sigma=50e6+pressure-tauN
        xx=np.broadcast_to(x[:,None,None]+h[:,None,None]*z[None,None,:],sigma.shape)
        yy=np.broadcast_to(y[:,None,None]+h[:,None,None]*z[None,:,None],sigma.shape)
        distance=(XT-xx)*SN-(100000-yy)*.5
        active=(abs(distance)<=phi_radius)&(phase>0)
        flat=np.flatnonzero(active)
        selected=sigma.ravel()[flat]
        for kind,locations in [('min',np.argsort(selected)[:20]),('max',np.argsort(selected)[-20:])]:
            for location in flat[locations]:
                c,a,b=np.unravel_index(location,sigma.shape)
                rows.append(dict(kind=kind,x=float(xx[c,a,b]),y=float(yy[c,a,b]),
                    xd=float((XT-xx[c,a,b])*.5+(100000-yy[c,a,b])*SN),r=float(distance[c,a,b]),
                    h=float(h[c]),p=float(pressure[c,a,b]),tauN=float(tauN[c,a,b]),sigma=float(sigma[c,a,b])))
    result={kind:sorted([r for r in rows if r['kind']==kind],key=lambda r:r['sigma'],reverse=kind=='max')[:20]
            for kind in ['min','max']}
    result['caveat']='Float32 visualization reconstruction at t=0 only; current stress from velocity/p, not published tau.'
    (run/'initial_normal_locations.json').write_text(json.dumps(result,indent=2)+'\n')
    print(label,result['min'][0],result['max'][0])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');audit(p.parse_args().label)
