#!/usr/bin/env python3
"""Independent continuous-time RSF linearization of exported native K/N/Q.
Freeze mature-patch damage. Do not modify/evolve native histories here.
"""
from pathlib import Path
import argparse,json,numpy as np
from scipy.linalg import eigvals,eigh,solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('stem',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 stem=str(a.stem);K=np.loadtxt(stem+'-K.csv',delimiter=',')[1:,1:];Q=np.loadtxt(stem+'-Q.csv',delimiter=',');N=np.loadtxt(stem+'-N.csv',delimiter=',')[:,1:];pts=np.genfromtxt(stem+'-points.csv',delimiter=',',names=True)
 x,w=pts['x_m'],pts['weight_m'];mature=(x>5)&(x<8)
 # Exact admissible nodal perturbations whose complete support lies in a=1.
 columns=np.flatnonzero(np.all(abs(Q[~mature])<1e-14,axis=0)&(abs(Q).sum(axis=0)>0))
 rows=np.flatnonzero(np.any(abs(Q[:,columns])>0,axis=1));C=Q[np.ix_(rows,columns)];W=w[rows];kk=K[np.ix_(columns,columns)];nn=N[np.ix_(rows,columns)];mass=C.T@(W[:,None]*C)
 ks,modes=eigh((kk+kk.T)/2,mass)
 V=1e-7;V0=1e-6;Dc=6e-5;theta=10.;pa=10000.;direct=.005;rad=25836.021365527624;rM=1e-6
 rowsout=[];spectra={}
 for name,b,mu0 in [('weakening',.02,.1),('neutral',.005,.1+.015*np.log(10))]:
  z=V/(2*V0)*np.exp((mu0+b*np.log(theta))/direct);frac=z/np.hypot(1,z);mu=direct*np.arcsinh(z)
  dv=pa*direct*frac/V+rad;ds=pa*b*frac/theta
  M=dv*mass;P=kk+C.T@(W[:,None]*mu*nn);B=C.T*(W*ds)[None,:]
  Vm=-solve(M,P);Vs=-solve(M,B)
  A=np.block([[Vm-rM*np.eye(len(columns)),Vs],[-theta/Dc*C@Vm,-theta/Dc*C@Vs-V/Dc*np.eye(len(rows))]])
  spec=eigvals(A);spectra[name]=spec
  # Independent nonlinear algebraic solve: force residual zero at perturbed
  # m/theta; pressure moves with N*m. Central differences test ALL Jacobian blocks.
  def rhs(y):
   mm=y[:len(columns)];age=theta+y[len(columns):];pn=pa+nn@mm;v=np.full(len(columns),0.)
   # Increment velocity represented by C; baseline V is prescribed locally.
   for _ in range(8):
    vv=V+C@v;zz=vv/(2*V0)*np.exp((mu0+b*np.log(age))/direct)
    friction=direct*np.arcsinh(zz);f=kk@mm+C.T@(W*(pn*friction+rad*vv-(pa*mu+rad*V)))
    J=C.T@((W*(pn*direct*(zz/np.hypot(1,zz))/vv+rad))[:,None]*C)
    dvv=solve(J,-f);v+=dvv
    if np.linalg.norm(dvv)<1e-20:break
   return np.r_[v-rM*mm,V0/Dc-(V+C@v)*age/Dc]
  rng=np.random.default_rng(381);direction=np.r_[rng.normal(size=len(columns))*1e-6,rng.normal(size=len(rows))]
  eps=1e-4;fd=(rhs(eps*direction)-rhs(-eps*direction))/(2*eps);error=np.linalg.norm(fd-A@direction)/np.linalg.norm(A@direction)
  assert error<2e-6,error
  # Independently compare scalar spring-slider roots for the zero-N,elastic limit.
  roots=np.concatenate([np.roots([dv,k+(pa*direct*frac+rad*V-pa*b*frac)/Dc,k*V/Dc]) for k in ks])
  Vm0=-solve(M,kk);A0=np.block([[Vm0,Vs],[-theta/Dc*C@Vm0,-theta/Dc*C@Vs-V/Dc*np.eye(len(rows))]])
  rr=eigvals(A0);rooterror=abs(rr.real.max()-roots.real.max())
  assert rooterror<1e-10
  row=dict(case=name,free_jump_modes=len(columns),state_records=len(rows),minimum_stiffness_Pa_m=float(ks[0]),maximum_stiffness_Pa_m=float(ks[-1]),critical_stiffness_Pa_m=float((pa*(b-direct)*frac-rad*V)/Dc),max_growth_rate_s_inv=float(spec.real.max()),efold_s=float(1/spec.real.max()),max_imag_s_inv=float(abs(spec.imag).max()),jacobian_action_relative_error=float(error),scalar_root_error_s_inv=float(rooterror),pressure_coupling_relative=float(np.linalg.norm(P-kk)/np.linalg.norm(kk)),mu=mu)
  rowsout.append(row)
 (a.output/'stability.json').write_text(json.dumps(rowsout,indent=2)+'\n');print(json.dumps(rowsout,indent=2))
 np.savetxt(a.output/'mature-mode.csv',np.c_[x[rows],C@modes[:,0]],delimiter=',',header='x_m,mode_shape',comments='')
 for name,spec in spectra.items():np.savetxt(a.output/(name+'-spectrum.csv'),np.c_[spec.real,spec.imag],delimiter=',',header='growth_s_inv,frequency_rad_s',comments='')
 fig,axs=plt.subplots(1,2,figsize=(9,3.4));axs[0].plot(x[rows],C@modes[:,0]);axs[0].set(xlabel='x (m)',ylabel='Lowest-stiffness slip mode (normalized)',title='Mature patch; all other slip tied')
 for name,spec in spectra.items():axs[1].scatter(spec.real,spec.imag,s=15,label=name)
 axs[1].axvline(0,color='k',lw=.6);axs[1].set(xlabel='Growth rate (1/s)',ylabel='Frequency (rad/s)',title='Native K + current pressure + RSF state');axs[1].legend();fig.tight_layout();fig.savefig(a.output/'stability.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
