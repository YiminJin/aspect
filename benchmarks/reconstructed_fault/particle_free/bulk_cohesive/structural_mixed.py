#!/usr/bin/env python3
"""Full mixed Fourier pencil for the recorded physical plane; no pressure lag."""
import json,time
import numpy as np
from scipy.linalg import eigvals,expm
from scipy.optimize import linear_sum_assignment
from structural_rsf import G,P0,RELAX,ETA_RAD,V0,DC,ELL,plane,inner,tensor,h_of,chi_center,rotation
from reconcile_feng_rsf import friction,A_RSF,B_RSF
from structural_sliding import OUT,write_csv
BASIS=[tensor(xx=1),tensor(xy=1),tensor(mean=1)]


def components(t):return np.array([.5*(t[0,0]-t[1,1]),t[0,1],.5*(t[0,0]+t[1,1])])


def base(phi,V,sign=1,Theta=None,b=B_RSF):
    age=V0/V if Theta is None else Theta
    mu,mq,mt=friction(V/V0,age,b=b)
    return dict(phi=phi,h=h_of(phi),chi=chi_center(phi),V=V,sign=sign,Theta=age,b=b,mu=mu,
                Z=ETA_RAD+P0*mq/V0,mt=mt,w=chi_center(phi)*V)


def reduced(b,angle):
    n,t,S=plane([0,1,0]);P=np.outer(n,n);wave=np.array([np.cos(2*angle),np.sin(2*angle)])
    stress=[tensor(xx=wave[0],xy=wave[1]),tensor(mean=1)]
    # y=(two equilibrated tau/G modes, three A modes, relative age).
    A=-RELAX*np.eye(6);D=np.zeros(6);B=np.zeros(6)
    D[0]=-2*b['w']*(components(S)[:2]@wave);D[2:5]=b['w']*components(S)
    for j,shape in enumerate(stress):B[j]=inner(S+b['sign']*b['mu']*P,shape)-b['sign']*b['mu']
    for j,shape in enumerate(BASIS):B[2+j]=-2/b['h']*inner(S,shape)
    B[5]=-b['sign']*P0*b['mt']*b['Theta']/G
    C=-b['Z']*b['V']/G
    A[5,5]=-b['V']/DC;D[5]=-b['sign']*b['V']/DC
    return A-np.outer(D,B)/C,B,C


def full(b,angle,kell=1.):
    n,t,S=plane([0,1,0]);P=np.outer(n,n);k=np.array([np.cos(angle),np.sin(angle),0.])
    # tau/G(3), A(3), relative age, i*u/(chi*V*ell)(2), p/G, v/V.
    F=np.zeros((11,11));E=np.zeros_like(F);E[:7,:7]=np.eye(7);F[:7,:7]=-RELAX*np.eye(7)
    for j in range(2):
        u=np.eye(3)[j];D=(np.outer(k,u)+np.outer(u,k))/2-np.eye(3)*(k@u)/3
        F[:3,7+j]=2*b['w']*kell*components(D)
    F[:3,10]=-2*b['w']*components(S);F[3:6,10]=b['w']*components(S)
    F[6,6]=-b['V']/DC;F[6,10]=-b['sign']*b['V']/DC
    for j,shape in enumerate(BASIS):
        F[7:9,j]=(shape@k)[:2]
        F[10,j]=inner(S+b['sign']*b['mu']*P,shape)
        F[10,3+j]=-2/b['h']*inner(S,shape)
    F[7:9,9]=-k[:2];F[9,7:9]=kell*k[:2]
    F[10,9]=-b['sign']*b['mu'];F[10,6]=-b['sign']*P0*b['mt']*b['Theta']/G;F[10,10]=-b['Z']*b['V']/G
    return F,E


def finite_spectrum(F,E):
    # Invertible row/column balancing on both sides of the same mixed pencil.
    F=F.copy();E=E.copy()
    for _ in range(12):
        columns=np.sqrt(np.maximum(abs(F).max(axis=0),abs(E).max(axis=0)));F/=columns;E/=columns
        rows=np.sqrt(np.maximum(abs(F).max(axis=1),abs(E).max(axis=1)));F/=rows[:,None];E/=rows[:,None]
    roots=eigvals(F,E);return roots[np.isfinite(roots)]


def main():
    start=time.perf_counter();rows=[];maximum_error=0.;max_fd=0.;kerrors=[]
    angles=np.linspace(0,np.pi,181)
    for phi in [.2,.8,.99,1.]:
      for V in [5e-6,5e-5,.0005,.002,.02]:
       for state in ['steady','fresh','no_state']:
        b=base(phi,V,Theta=1. if state=='fresh' else None,b=0. if state=='no_state' else B_RSF)
        for angle in angles:
            M,B,C=reduced(b,angle);roots=np.linalg.eigvals(M);maxreal=max(roots.real)
            # Analytic active quadratic, including normal-pressure feedback.
            sn,cs=np.sin(2*angle),np.cos(2*angle)
            D=sn*sn-b['sign']*b['mu']*sn*(1+cs)
            stiffness=G*b['chi']*(D+1/b['h']);nu=V/DC;stateforce=P0*b['mt']*b['Theta']/DC
            quadratic=np.roots([b['Z'],b['Z']*(RELAX+nu)+stiffness-stateforce,b['Z']*RELAX*nu+stiffness*nu-stateforce*RELAX])
            expected=np.r_[quadratic,[-RELAX]*4];ri,ci=linear_sum_assignment(abs(roots[:,None]-expected[None,:]));err=max(abs(roots[ri]-expected[ci]))/max(1.,max(abs(roots)))
            assert err<1e-7,err
            rows.append(dict(phi=phi,V_m_s=V,state=state,angle_deg=np.degrees(angle),mu=b['mu'],chi_inv_m=b['chi'],mixed_D=D,
                max_growth_s_inv=maxreal,active_root1_real_s_inv=quadratic[0].real,active_root2_real_s_inv=quadratic[1].real,
                state_removed=int(state=='no_state')))
        for angle in [.013,.06,.4,1.1,1.57,2.8]:
            M,_,_=reduced(b,angle);expected=np.linalg.eigvals(M)
            for kell in [.01,1.,100.,10000.]:
                roots=finite_spectrum(*full(b,angle,kell));assert len(roots)==6,(roots,b)
                ri,ci=linear_sum_assignment(abs(roots[:,None]-expected[None,:]));error=max(abs(roots[ri]-expected[ci]))/max(1.,max(abs(expected)))
                maximum_error=max(maximum_error,error);assert error<1e-7,error
                kerrors.append(dict(phi=phi,V_m_s=V,state=state,angle_rad=angle,kell=kell,max_growth_s_inv=max(roots.real),mixed_reduction_error=error))
    # Independent finite differences of actual normal-traction/state flow row.
    for sign in [-1,1]:
      for phi in [.2,.8,1.]:
       b=base(phi,.0005,sign=sign,Theta=.03);F,E=full(b,.2);n,t,S=plane([0,1,0])
       tau0=2*sign*(P0*b['mu']+ETA_RAD*b['V'])*S
       def residual(x):
        tau=tau0+G*sum((x[j]*BASIS[j] for j in range(3)),np.zeros((3,3)))
        A=sum((x[3+j]*BASIS[j] for j in range(3)),np.zeros((3,3)))
        p=P0+G*x[9];v=sign*b['V']+b['V']*x[10];age=b['Theta']*(1+x[6]);pn=p-inner(tau,np.outer(n,n))
        mu=friction(abs(v)/V0,age)[0]
        return (inner(tau-2*G*A/b['h'],S)-np.sign(v)*(pn*mu+ETA_RAD*abs(v)))/G
       fd=[];epsilon=1e-7
       for j in range(11):
        dx=np.eye(11)[j]*epsilon;fd.append((residual(dx)-residual(-dx))/(2*epsilon))
       error=np.linalg.norm(np.array(fd)-F[10])/np.linalg.norm(F[10]);max_fd=max(max_fd,error);assert error<1e-7,error
    # A known physical oblique perturbation; all pressure and state equations remain.
    b=base(1.,.0005);angle=.04;M,B,C=reduced(b,angle);values,vectors=np.linalg.eig(M)
    index=np.argmax(values.real);growth=float(values[index].real);y0=np.real(vectors[:,index]);y0*=1e-7/np.linalg.norm(y0)
    perturb=[]
    for dt in [.002,.001,.0005]:
        y=y0.copy();inverse=np.linalg.inv(np.eye(6)-dt*M)
        for step in range(round(.5/dt)):
            y=inverse@y;t=(step+1)*dt
            if (step+1)%round(.01/dt):continue
            exact=expm(t*M)@y0;relative=np.linalg.norm(y-exact)/np.linalg.norm(exact)
            perturb.append(dict(dt_s=dt,time_s=t,angle_rad=angle,growth_s_inv=growth,relative_error=relative,
                amplitude=np.linalg.norm(y),exact_amplitude=np.linalg.norm(exact),pressure_Pa=G*(y[0]+y[1]),flow_perturbation_m_s=float(-B@y/C*b['V'])))
    # Spatial symbol consistency: prescribed physical oblique wave, central-difference k.
    spatial=[];wave=np.array([1.,2.]);L=1.;physical=2*np.pi*wave/L;angle=np.arctan2(physical[1],physical[0]);reference=reduced(b,angle)[0]
    y0=np.array([1,0,0,0,0,0.],dtype=float)*1e-7;end=expm(.5*reference)@y0
    for cells in [32,64,128]:
        dx=L/cells;k=np.sin(physical*dx)/dx;discrete=reduced(b,np.arctan2(k[1],k[0]))[0];sample=expm(.5*discrete)@y0
        spatial.append(dict(cells=cells,physical_kx_inv_m=physical[0],physical_ky_inv_m=physical[1],relative_error=np.linalg.norm(sample-end)/np.linalg.norm(end)))
    write_csv('mixed_spectrum.csv',rows);write_csv('mixed_wavenumber.csv',kerrors);write_csv('mixed_perturbation.csv',perturb);write_csv('mixed_spatial.csv',spatial)
    result=dict(seconds=time.perf_counter()-start,mixed_QZ_relative_error=maximum_error,pressure_flow_FD_error=max_fd,
        tested_oblique_growth_s_inv=growth,temporal_endpoint_errors=[r['relative_error'] for r in perturb if abs(r['time_s']-.5)<1e-12],spatial=spatial,
        maximum_growth_s_inv=max(r['max_growth_s_inv'] for r in rows),positive_no_state_growth_s_inv=max(r['max_growth_s_inv'] for r in rows if r['state']=='no_state'))
    (OUT/'mixed_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
