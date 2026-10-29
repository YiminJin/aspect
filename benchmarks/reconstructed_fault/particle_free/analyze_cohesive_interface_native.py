#!/usr/bin/env python3
"""Independent comparison of native Gamma operators/owners with prior reference."""
import sys,os,json,csv
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-interface-native')
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'))
import cohesive_interface as ref
from structural_rsf import state_friction
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_native'

def read(path):
 with path.open() as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
def write(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def operators(name):
 rows=read(OUT/name/'operators.csv');n=1+max(int(r['i']) for r in rows)
 K=np.zeros((n,n));N=K.copy();weights=np.zeros(n)
 for r in rows:i,j=int(r['i']),int(r['j']);K[i,j]=r['K_Pa_m'];N[i,j]=r['N_Pa_m'];weights[i]=r['weight_m']
 return K,N,weights

def surface_rule(n):
 xi,wg=np.polynomial.legendre.leggauss(3);xi=(xi+1)/2;wg=wg/2
 Q=np.zeros((3*n,n));w=np.tile(wg/n,n);x=np.zeros(3*n)
 for i in range(n):
  for k in range(3):Q[3*i+k,i]=1-xi[k];Q[3*i+k,(i+1)%n]=xi[k];x[3*i+k]=(i+xi[k])/n
 M=Q.T@(w[:,None]*Q);projection=Q@np.linalg.inv(M)/n
 return Q,w,x,projection

def response(name,mode):
 K,N,w=operators(name);n=len(w);Q,wq,x,projection=surface_rule(n);R=Q.T*wq[None,:]/w[:,None]
 histories=read(OUT/name/'history.csv');states=read(OUT/name/'states.csv');ng=len(wq)
 mem=np.zeros(n);jump=np.zeros(ng);age=np.ones(ng);absolute=np.zeros(ng);a=np.ones(ng) if mode=='mature' else .6+.1*np.cos(2*np.pi*x) if mode=='nonuniform' else np.zeros(ng)
 errors={key:0. for key in ['jump_m','damage','Theta','relaxed_jump_m','absolute_slip_m','shear_Pa','pn_Pa','q_Pa','V_m_s']}
 compare=[];absolute_work=0.;added_fracture=0.;sliding=0.;min_curvature=float('inf');initial_fracture=float(wq@a);previous_energy=0.
 # Independent scalar interface control, separate from the supplied FE operator.
 uniform=ref.UniformBlock();uniform_old=ref.initial(damage=1. if mode=='mature' else 0.);uniform_error=0.
 for k,h in enumerate(histories):
  t,dt=h['time_s'],h['dt_s'];mid=t-dt/2
  speed=(.001 if mid<2 else .002 if mid<4 else 0 if mid<4.5 else -.001) if mode=='mature' else .001
  beta=1/(1+ref.RELAX*dt);trialmem=beta*(mem+speed*dt);inc=np.zeros(n)
  active=(R@a>0)|(abs(K@trialmem)>ref.TAUP);ids=np.flatnonzero(active);hint=np.sign(projection@(K@trialmem))
  def evaluate(increment):
   m=trialmem-beta*increment;stress=K@m;pressure=ref.P0-N@m;pn=projection@pressure;dq=Q@increment;V=dq/dt
   mu,muv,theta=state_friction(abs(V),age,dt);aa,q,derivative,da,storage=ref.bond(jump+dq,a,0.,hint)
   friction=np.sign(V)*(pn*mu+ref.ETA_RAD*abs(V));f=stress-R@(q+aa*friction)
   J=-beta*K-(R*(derivative+da*friction+aa*(pn*muv+ref.ETA_RAD)/dt)[None,:])@Q-beta*(R*(aa*np.sign(V)*mu)[None,:])@projection@N
   return f,J,(m,stress,pn,dq,V,theta,aa,q,storage)
  for iteration in range(60):
   f,J,data=evaluate(inc);error=max(abs(f[ids])) if len(ids) else 0.
   if error<=1e-8:break
   d=np.zeros(n);d[ids]=np.linalg.solve(J[np.ix_(ids,ids)],-f[ids]);scale=1.
   for ls in range(30):
    candidate=inc+scale*d;ff,_,_=evaluate(candidate)
    if max(abs(ff[ids]))<error or max(abs(ff[ids]))<1e-8:inc=candidate;break
    scale*=.5
   else:raise AssertionError('Independent Gauss return line search')
  else:raise AssertionError('Independent Gauss return iteration limit')
  mem,stress,pn,dq,V,theta,aa,q,storage=data;q=np.where(aa==0,projection@stress,q)
  if len(ids):
   matrix=-J[np.ix_(ids,ids)];min_curvature=min(min_curvature,float(np.linalg.eigvalsh((matrix+matrix.T)/2).min()))
  jump+=dq;absolute+=abs(dq);age=theta;a=aa
  for i,row in enumerate(states[k*ng:(k+1)*ng]):
   expected=dict(jump_m=jump[i],damage=a[i],Theta=age[i],relaxed_jump_m=(Q@(h['wall_memory_m']-mem))[i],absolute_slip_m=absolute[i],shear_Pa=(projection@stress)[i],pn_Pa=pn[i],q_Pa=q[i],V_m_s=V[i])
   for key in errors:errors[key]=max(errors[key],abs(row[key]-expected[key]))
  if mode!='nonuniform':
   un=ref.solve(uniform,uniform_old,dt,np.array([speed*dt]));uniform_error=max(uniform_error,float(max(abs(stress-un['stress'][0]))));uniform_old=ref.commit(uniform_old,un)
  state_energy=h['bulk_J_m']+h['bond_J_m'];physical=h['Maxwell_J_m']+h['friction_J_m']+h['radiation_J_m'];added_fracture+=h['fracture_increment_J_m'];sliding+=h['friction_J_m']+h['radiation_J_m']
  absolute_work+=abs(h['trapezoid_work_J_m']-(state_energy-previous_energy+h['fracture_increment_J_m']+physical));previous_energy=state_energy
  compare.append(dict(case=name,time_s=t,native_shear_Pa=np.mean([r['shear_Pa'] for r in states[k*ng:(k+1)*ng]]),reference_shear_Pa=float(np.mean(projection@stress)),mean_damage=float(wq@a),mean_jump_m=float(wq@jump),pn_range_Pa=float(np.ptp(pn)),absolute_physical_work_error_J_m=absolute_work))
 assert errors['shear_Pa']<1e-7 and uniform_error<1e-7,errors
 assert errors['jump_m']<1e-10 and errors['damage']<1e-9 and errors['Theta']<1e-9,errors
 assert min_curvature>0
 summary=dict(case=name,steps=len(histories),errors=errors,scalar_identity_Pa=uniform_error,initial_fracture_J_m=initial_fracture,added_fracture_J_m=added_fracture,sliding_J_m=sliding,absolute_work_error_J_m=absolute_work,absolute_work_error_over_sliding=absolute_work/sliding if sliding else None,min_incremental_curvature_Pa_m=min_curvature,max_native_force_Pa=max(h['force_error_Pa'] for h in histories),max_native_ledger_J_m=max(abs(h['ledger_error_J_m']) for h in histories),max_pressure_range_Pa=max(r['pn_range_Pa'] for r in compare),final_abs_slip_Dc=float(wq@absolute/ref.DC))
 return compare,summary

def mode_symbol(K,m=1):
 n=len(K);v=np.exp(2j*np.pi*m*np.arange(n)/n);return np.vdot(v,K@v)/n

def sharp_q1(n,m=1,aliases=400):
 # Galerkin projection of a periodic Q1 displacement jump, using the SAME
 # nodal quadrature force normalization as the native kernel. No collocation
 # vs Galerkin discrepancy is mislabeled as a physical width effect.
 result=np.zeros((2,),complex);reflection=np.diag([1.,-1.])
 for alias in range(-aliases,aliases+1):
  mode=m+alias*n;k=2*np.pi*mode
  up=ref.layer_DtN(abs(k),.075);down=reflection@ref.layer_DtN(abs(k),.05)@reflection
  full=np.linalg.inv(np.linalg.inv(up)+np.linalg.inv(down));normal=full[1,0] if k>0 else full[1,0].conjugate()
  amplitude=np.sinc(mode/n)**4
  result+=amplitude*np.array([full[0,0],normal])
 return result

def main():
 rows=[];summaries=[]
 for name,mode in [('formation','formation'),('formation-half','formation'),('mature','mature'),('nonuniform','nonuniform')]:
  r,s=response(name,mode);rows+=r;summaries.append(s)
 write('native_reference_histories.csv',rows)
 # Actual simulator restart data, including all node histories and physical tractions.
 control=read(OUT/'nonuniform/states.csv');loaded=read(OUT/'resume/restart_loaded.csv')[0];restart=[r for r in read(OUT/'split/states.csv') if r['time_s']<=loaded['accepted_time_s']]+read(OUT/'resume/states-resume.csv')
 assert len(control)==len(restart)
 restart_errors={key:max(abs(a[key]-b[key]) for a,b in zip(control,restart)) for key in control[0]}
 assert all(v==0 for v in restart_errors.values()),restart_errors
 h0=read(OUT/'nonuniform/history.csv');hr=[r for r in read(OUT/'split/history.csv') if r['time_s']<=loaded['accepted_time_s']]+read(OUT/'resume/history-resume.csv')
 assert [r['publications'] for r in h0]==[r['publications'] for r in hr]
 node_control=read(OUT/'nonuniform/nodal_history.csv');node_restart=[r for r in read(OUT/'split/nodal_history.csv') if r['time_s']<=loaded['accepted_time_s']]+read(OUT/'resume/nodal_history-resume.csv')
 assert node_control==node_restart
 mature_nodes=read(OUT/'mature/nodal_history.csv');assert min(r['signed_rate_m_s'] for r in mature_nodes)<0
 tail_error=float(np.linalg.norm(sharp_q1(32,aliases=200)-sharp_q1(32,aliases=400))/np.linalg.norm(sharp_q1(32)))
 assert tail_error<1e-7
 mode_rows=[]
 for name in ['formation','nonuniform','operator-16','operator-32','width-half','width-half-32']:
  if not (OUT/name/'operators.csv').exists():continue
  K,N,w=operators(name);info=read(OUT/name/'operator_checks.csv')[0];sharp=sharp_q1(len(K));k=mode_symbol(K);normal=mode_symbol(N)
  mode_rows.append(dict(case=name,n=len(K),width_m=info['width_m'],K_mode_Pa_m=k.real,N_mode_imag_Pa_m=normal.imag,sharp_Q1_K_Pa_m=sharp[0].real,sharp_Q1_N_imag_Pa_m=sharp[1].imag,K_width_difference=(k.real-sharp[0].real)/sharp[0].real,N_width_difference=(normal.imag-sharp[1].imag)/abs(sharp[1].imag)))
 write('operator_comparison.csv',mode_rows)
 K0,N0,_=operators('formation');K6,N6,_=operators('quadrature6')
 quad=dict(K_relative=float(np.linalg.norm(K6-K0)/np.linalg.norm(K0)),N_relative=float(np.linalg.norm(N6-N0)/np.linalg.norm(N0)))
 assert quad['K_relative']<1e-10 and quad['N_relative']<1e-10
 result=dict(runs=summaries,signed_rate_restart_bitwise=True,sharp_alias_tail_relative=tail_error,restart_bitwise=True,restart_errors=restart_errors,quadrature=quad,operator_modes=mode_rows)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
