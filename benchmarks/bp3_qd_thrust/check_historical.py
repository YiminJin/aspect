"""Independent historical AT1, Q1 figure, traction operator and accepted-history checks."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
from native_frames import frames, read

SINE = math.sqrt(.75)
PLATE = 1e-9
TAU = 26546122.365133367


def weight(phi):
    return phi*(1+phi)/(1-phi)**2


# Independently cancel the removable singularities of D in the first integral.
# D=phi*(.6-phi)*(9-6*phi)/(1-phi)^2).
theta = np.linspace(0, np.pi/2, 65537)
phi = .6*np.cos(theta)**2
radius = cumulative_trapezoid(40*np.sqrt(6)*(1-phi)/np.sqrt(9-6*phi), theta, initial=0.)
primitive = cumulative_trapezoid(weight(phi),radius,initial=0.)


def phase(x,y,top):
    return np.interp(abs(SINE*x+.5*(y-top)),radius,phi,right=0.)


def q1(x,y,top,h,origin):
    ix,fx=np.divmod((x-origin)/h,1.)
    iy,fy=np.divmod(y/h,1.)
    return sum((fx if i else 1-fx)*(fy if j else 1-fy)*phase(origin+(ix+i)*h,(iy+j)*h,top)
               for i in (0,1) for j in (0,1))


def profile(path):
    with path.open() as stream:
        time=float(stream.readline().split()[-1])
        stream.readline()
        data=np.genfromtxt(stream,names=True)
    a=np.column_stack([data[name] for name in data.dtype.names])
    assert np.isfinite(a).all() and np.array_equal(data['id'],np.arange(len(data)))
    return time,data


def matrix(rows,diag,off):
    return diag*rows+np.r_[off*rows[1:],0]+np.r_[0,off*rows[:-1]]


def solve(diag,off,rhs):
    return solve_banded((1,1),np.vstack((np.r_[0,off],diag,np.r_[off,0])),rhs)


def check(directory,figure=None):
    mesh,fields,size=read(frames(directory)[0])
    x0,y0,x1,y1=mesh.astype(float).T
    top=y1.max()
    origin=x0.min()
    h=2*np.min(x1-x0)
    x=np.column_stack((x0,x1,x1,x0))
    y=np.column_stack((y0,y0,y1,y1))
    expected=q1(x,y,top,h,origin)
    phase_error=np.max(abs(expected-fields['phase_field']))
    assert phase_error<3e-7,phase_error
    r=SINE*x+.5*(y-top)
    ramp=np.sign(r)*np.interp(abs(r),radius,primitive)/primitive[-1]*.5
    prescribed=(y==0)|(x==origin)|(x==x1.max())
    vel=fields['velocity']
    velocity_error=[np.max(abs(vel[:,:,i][prescribed]-value[prescribed])) for i,value in
                    enumerate((-.5*PLATE*ramp,SINE*PLATE*ramp))]
    assert max(velocity_error)<1e-16,velocity_error
    print(f"{directory}: h={h} m, {len(mesh)//4} cells, {size} bytes/frame, radius={radius[-1]:.9f} m, full integral={2*primitive[-1]:.9f} m")
    print("FE phase L_inf",phase_error,"wall ux,uy errors (m/s)",velocity_error)
    print("time-zero native max abs velocity, p, tau_xx/yy/xy", [np.max(abs(fields[k])) for k in ('velocity','p','fault_tau_xx','fault_tau_yy','fault_tau_xy')])
    files=sorted(directory.glob("fault-profile-*.txt"))
    clock,initial=profile(files[0])
    assert clock==0 and np.all(initial['slip_m']==0)
    s=initial['s_m']
    np.testing.assert_allclose(initial['x_m'],.5*s,atol=1e-7,rtol=0)
    np.testing.assert_allclose(initial['y_m'],top-SINE*s,atol=1e-7,rtol=0)
    probes=(s[:-1]+s[1:])/2
    a=.01+.015*np.clip((probes-15000)/3000,0,1)
    ages=np.sqrt(initial['age_s'][:-1]*initial['age_s'][1:])
    R=5e7*a*np.arcsinh(PLATE/2e-6*np.exp((.6+.015*np.log(ages*1e-6/.008))/a))+4624440*PLATE
    assert max(abs(R-TAU))<1e-6
    print("initial midpoint forward resistance max error (Pa)", max(abs(R-TAU)))
    for pos in (0.,15000.,16500.,18000.,40000.,s[-1]):
        if pos<=s[-1]:
            print(f"s={pos:.6f} m: V={np.interp(pos,s,initial['rate_m_s']):.12g} m/s, shear/raw/filtered Pa",
                  *(np.interp(pos,s,initial[name]) for name in ('shear_Pa','raw_compression_Pa','friction_compression_Pa')))
    # Q8 section mass and tangential stiffness from cell geometry, Q1 phase
    # and accepted normal integrals. Endpoint continuation has zero derivative.
    import json
    params=json.loads((directory/'parameters.json').read_text())
    settings=params['Material model']['Phase field fault']
    magnitude=float(settings['Degradation magnitude']['value'])
    integral=initial['normal_integral_m']/magnitude
    interval=float(settings['Initialization interval']['value'])
    kappa=-1e26*np.expm1(-interval*32038120320/1e26)
    sel=np.max(fields['phase_field'],axis=1)>0
    cellw=2*(x1-x0)
    all_cells,group=np.unique(np.column_stack((origin+cellw*np.floor((x0-origin)/cellw),
                                              cellw*np.floor(y0/cellw),cellw)),axis=0,return_inverse=True)
    selected=np.unique(group[sel])
    rows=np.isin(group,selected)
    n=np.searchsorted(selected,group[rows])
    X,Y,H=all_cells[selected].T
    # Recover all nine Q2 nodal velocities, including distinct DG traces, from
    # the four VTK patches of each active cell. Differentiate in physical metres.
    nodal=np.empty((len(X),3,3,3))
    cell_values=np.concatenate((vel[:,:,:2],fields['p'][:,:,None]),axis=2)
    for q in range(4):
        ix=np.rint(2*(x[rows,q]-X[n])/H[n]).astype(int)
        iy=np.rint(2*(y[rows,q]-Y[n])/H[n]).astype(int)
        nodal[n,iy,ix]=cell_values[rows,q]
    def basis(u):
        return np.array([(1-u)*(1-2*u),4*u*(1-u),u*(2*u-1)])
    def derivative(u):
        return np.array([4*u-3,4-8*u,4*u-1])
    diag=np.zeros_like(s); off=np.zeros(len(s)-1)
    kd=diag.copy(); ko=off.copy()
    normal_rhs=diag.copy(); shear_rhs=diag.copy()
    nodes,weights=np.polynomial.legendre.leggauss(8)
    for i,u in enumerate((nodes+1)/2):
        for j,v in enumerate((nodes+1)/2):
            xx,yy=X+u*H,Y+v*H
            pos=.5*xx-SINE*(yy-top)
            k=np.clip(np.searchsorted(s,pos)-1,0,len(s)-2)
            ds=s[k+1]-s[k]
            xi=np.clip((pos-s[k])/ds,0,1)
            w=H**2*weights[i]*weights[j]/4*weight(q1(xx,yy,top,h,origin))/np.interp(pos,s,integral)
            dx=np.einsum('i,nijc,j->nc',basis(v),nodal,derivative(u))/H[:,None]
            dy=np.einsum('i,nijc,j->nc',derivative(v),nodal,basis(u))/H[:,None]
            pressure=np.einsum('i,nij,j->n',basis(v),nodal[:,:,:,2],basis(u))
            div=dx[:,0]+dy[:,1]
            xx_stress=2*kappa*(dx[:,0]-div/3)
            yy_stress=2*kappa*(dy[:,1]-div/3)
            xy_stress=kappa*(dy[:,0]+dx[:,1])
            normal=pressure-(.75*xx_stress+SINE*xy_stress+.25*yy_stress)
            chi=w/(H**2*weights[i]*weights[j]/4)
            shear=-.5*SINE*(xx_stress-yy_stress)+.5*xy_stress-kappa*chi*np.interp(pos,s,initial['rate_m_s'])
            for rhs,traction in ((normal_rhs,normal),(shear_rhs,shear)):
                np.add.at(rhs,k,w*(1-xi)*traction);np.add.at(rhs,k+1,w*xi*traction)
            np.add.at(diag,k,w*(1-xi)**2)
            np.add.at(diag,k+1,w*xi**2)
            np.add.at(off,k,w*xi*(1-xi))
            grad=w*((pos>0)&(pos<s[-1]))/ds**2
            np.add.at(kd,k,grad);np.add.at(kd,k+1,grad);np.add.at(ko,k,-grad)
    raw_error=solve(diag,off,normal_rhs)-(initial['raw_compression_Pa']-5e7)
    shear_error=solve(diag,off,shear_rhs)-(initial['shear_Pa']-TAU)
    # Native VTU stores Float32 velocity. Bound the differentiated Q2 nodal
    # rounding amplified by the Maxwell coefficient, separately for each mesh.
    traction_budget=2e-5+48*np.finfo(np.float32).eps*kappa*np.max(abs(vel))/h
    print("independent initial stress projection errors (Pa), raw normal and shear, both contacts",
          raw_error[[0,-1]],shear_error[[0,-1]],"all-ID maxima",max(abs(raw_error)),max(abs(shear_error)))
    assert max(max(abs(raw_error)),max(abs(shear_error)))<traction_budget,(raw_error,shear_error,traction_budget)
    length=float(params['Material model']['Phase field fault']['Normal traction filter length']['value'])
    def filtered(p):
        return solve(diag+length**2*kd,off+length**2*ko,matrix(p,diag,off))
    assert np.max(abs(filtered(np.ones_like(s))-1))<1e-12
    perturbation=np.cos(2*np.pi*s/100.)
    print("20 m Helmholtz test: constant and nonconstant max-amplitude ratio",np.max(abs(filtered(np.ones_like(s))-1)),np.linalg.norm(filtered(perturbation))/np.linalg.norm(perturbation))
    if length and max(np.diff(s))<20:
        assert np.linalg.norm(filtered(perturbation))<.8*np.linalg.norm(perturbation)
    first=None
    for path in files:
        clock,data=profile(path)
        sigma=filtered(data['raw_compression_Pa']-5e7)+5e7
        error=max(abs(sigma-data['friction_compression_Pa']))
        assert error<3e-5,(clock,error)
        if first is not None:
            # Every step in qualification inputs. Production profiles may be sparse.
            if int(path.stem.split('-')[-1])==last_step+1:
                rate=data['rate_m_s'];dt=clock-last_clock
                z=-dt*rate/.008
                age=first['age_s']*np.exp(z)-.008/rate*np.expm1(z)
                np.testing.assert_allclose(data['age_s'],age,rtol=5e-15,atol=0)
                np.testing.assert_allclose(data['slip_m'],first['slip_m']+dt*rate,rtol=5e-15,atol=0)
        print(f"{clock:.12g} s filter residual {error:.3g} Pa, peak {max(data['rate_m_s']):.12g} m/s")
        first=data
        last_clock=clock
        last_step=int(path.stem.split('-')[-1])
    r=np.linspace(-radius[-1]-2*h,radius[-1]+2*h,4001)
    cuts=[]
    for yy in (top,top/2,0.):
        xx=(r-.5*(yy-top))/SINE
        pos=.5*xx-SINE*(yy-top)
        chi=weight(q1(xx,yy,top,h,origin))/np.interp(pos,s,integral)
        cuts.append(chi)
        cut_mass=np.trapezoid(chi,r)
        print(f"y={yy:g} m horizontal cut; integral chi d(normal)={cut_mass:.8g}, integrated slip={np.trapezoid(chi*np.interp(pos,s,initial['rate_m_s']),r):.10g} m/s")
    observers=params['Postprocess']['List of postprocessors']['value'].split(',')
    if 'BP3 startup cuts' in (name.strip() for name in observers):
        observed=np.genfromtxt(directory/'startup-cuts.txt',names=True)
        for yy in (top,top/2,0.):
            line=observed[observed['y_m']==yy]
            rr=line['r_m']
            assert len(rr)==513 and np.all(np.diff(rr)>0) and rr[0]<-radius[-1] and rr[-1]>radius[-1],yy
            xx=(rr-.5*(yy-top))/SINE
            pos=.5*xx-SINE*(yy-top)
            predicted=q1(xx,yy,top,h,origin)
            np.testing.assert_allclose(line['phase'],predicted,atol=6e-8,rtol=0)
            density=weight(predicted)/np.interp(pos,s,integral)
            np.testing.assert_allclose(line['chi_per_m'],density,atol=3e-8,rtol=0)
            np.testing.assert_allclose(line['returned_rate_per_s'],density*np.interp(pos,s,initial['rate_m_s']),
                                       atol=3e-17,rtol=0)
            if h<5:
                assert abs(np.trapezoid(line['chi_per_m'],rr)-1.)<.01
    if figure:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
        for v,n in zip(cuts,('top contact','mid-depth','base contact')):
            axs[0].plot(r,v,label=n)
        axs[0].plot(r,np.interp(abs(r),radius,weight(phi))/(2*primitive[-1]),'k:',label='continuum AT1')
        axs[0].set(xlabel='Signed normal distance (m)',ylabel='Normalized slip density (1/m)')
        axs[0].legend(fontsize=8)
        axs[1].plot(s/1000,initial['raw_compression_Pa']-5e7,label='raw')
        axs[1].plot(s/1000,initial['friction_compression_Pa']-5e7,label='used in friction')
        axs[1].set(xlabel='Distance down dip (km)',ylabel='Normal traction perturbation (Pa)')
        axs[1].legend()
        fig.savefig(figure,dpi=160)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--figure",type=Path)
    args=parser.parse_args()
    check(args.directory,args.figure)
