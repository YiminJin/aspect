#!/usr/bin/env python3
"""50-digit independent check at operands of the saved FD roundoff failure."""
import json,csv
import numpy as np
import mpmath as mp
from run_cohesive_interface_domain import OUT,WORK
from analyze_cohesive_interface_domain import case,read,table

def main():
 mp.mp.dps=50
 p=WORK/'runs/period-16/output';target=1.9425
 _,z=read(p/'derivative-context-1942500.csv');dt=z[0,1];ns=len(z);dx=16/ns
 _,op=read(p/'operators.csv');K=op[:,2].reshape(ns,ns);N=op[:,3].reshape(ns,ns);mass=op[:,4].reshape(ns,ns)[:,0]
 xi=np.array([(1-np.sqrt(.6))/2,.5,(1+np.sqrt(.6))/2]);w=np.tile(np.array([5/18,4/9,5/18])*dx,ns)
 Q=np.zeros((3*ns,ns))
 for i in range(ns):Q[3*i:3*i+3,i]=1-xi;Q[3*i:3*i+3,(i+1)%ns]=xi
 M=Q.T@(w[:,None]*Q);QP=Q@np.linalg.inv(M)*dx
 c=case('period-16',16,256,4);time=z[0,0];incoming=time-dt;i=np.argmin(abs(c['h']['time_s']-incoming));assert abs(c['h']['time_s'][i]-incoming)<1e-10
 beta=1/(1+.01*dt);memory=z[:,4];inc=z[:,5];dq=Q@inc;pn=QP@(10000+N@(beta*(memory+inc)));dn=beta*QP@N
 _,ld=read(p/'derivative-extended.csv');ld=ld[abs(ld[:,0]-target)<1e-9]
 rows=[]
 def number(v):return mp.mpf(float(v))
 def resistance(q,increment,pressure):
  e=number(c['s']['jump_m'][i,q])+increment-number(c['s']['reference_m'][i,q]);a=max(number(c['s']['damage'][i,q]),min(abs(e)/(mp.mpf(2)/3000),mp.mpf(1)))
  cohesive=mp.mpf(3000) if a==0 else (mp.mpf(0) if a==1 else (1-a)/(mp.mpf(2)/3000**2*a)*e)
  v=increment/number(dt);speed=abs(v);theta=(number(c['s']['Theta'][i,q])+number(5e-6)*number(dt)/number(.001))/(1+speed*number(dt)/number(.001))
  friction=number(.005)*mp.asinh(speed/(2*number(5e-6))*mp.exp((number(.1)+number(.01)*mp.log(theta))/number(.005)))
  return cohesive+a*mp.sign(v)*(pressure*friction+number(1e6/(2*np.sqrt(1e6/2670)))*speed)
 active=z[:,6]>0;responsive=np.any(Q[:,active]>0,axis=1)
 theta=(c['s']['Theta'][i]+5e-6*dt/.001)/(1+abs(dq)/.001)
 scale=dt*np.hypot(dq/dt,2*5e-6*np.exp(-(.1+.01*np.log(theta))/.005));qmin=int(np.argmin(np.where(responsive,scale,np.inf)))
 responsive_nodes=np.flatnonzero((Q[qmin]>0)&active);assert len(responsive_nodes)
 local=int(responsive_nodes[np.argmax(Q[qmin,responsive_nodes])])
 requests=[(r,int(r[-2]),int(r[-1]),'largest_extended_roundoff_entry') for r in ld]+[(r,local,local,'minimum_rate_scale_active_diagonal') for r in ld]
 for r,row,col,kind in requests:
  eps=number(r[1]);ids=np.flatnonzero(Q[:,row]);weights=[number(w[q]*Q[q,row]/mass[row]) for q in ids]
  def delta_force(h):
   return -number(beta*K[row,col])*h-sum(weight*(resistance(q,number(dq[q])+h*number(Q[q,col]),number(pn[q])+h*number(dn[q,col]))-resistance(q,number(dq[q]),number(pn[q]))) for q,weight in zip(ids,weights))
  analytic=mp.diff(delta_force,mp.mpf(0));fd=(delta_force(eps)-delta_force(-eps))/(2*eps)
  rows.append(dict(time_s=target,entry_kind=kind,epsilon_m=float(eps),row=row,column=col,entry_tangent_Pa_m=float(analytic),MP50_entry_FD_relative=float(abs(fd-analytic)/max(abs(analytic),mp.mpf('1e-40'))),native_extended_global_relative=float(r[2]),biased_global_relative=float(r[3])))
 table(OUT/'derivative-MP50.csv',rows)
 active=z[:,6]>0;responsive=np.any(Q[:,active]>0,axis=1)
 theta=(c['s']['Theta'][i]+5e-6*dt/.001)/(1+abs(dq)/.001)
 scale=dt*np.hypot(dq/dt,2*5e-6*np.exp(-(.1+.01*np.log(theta))/.005));q=int(np.argmin(np.where(responsive,scale,np.inf)))
 result=dict(precision_decimal_digits=50,minimum_scale_q=q,minimum_scale_physical_x_m=float((q//3+xi[q%3])*dx-4),increment_m=float(dq[q]),normal_pressure_Pa=float(pn[q]),dt_s=float(dt),rate_scale_m=float(scale[q]),maximum_MP50_entry_relative=max(r['MP50_entry_FD_relative'] for r in rows),reconstructed_QP_from_same_Gauss3_mass=True,rows=rows)
 (OUT/'derivative-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 assert result['maximum_MP50_entry_relative']<1e-6
 assert max(r['native_extended_global_relative'] for r in rows)<1e-6
 assert min(r['biased_global_relative'] for r in rows)>1e-6
if __name__=='__main__':main()
