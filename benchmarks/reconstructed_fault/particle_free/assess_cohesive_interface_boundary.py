#!/usr/bin/env python3
"""Native reflection, continuation, and focused temporal comparisons."""
import sys,json
import numpy as np
from run_cohesive_interface_boundary import ROOT,WORK,OUT
import analyze_cohesive_interface_domain as prior
from analyze_cohesive_interface_domain import read,table,at

def load(name,previous=None,L=16,nx=256):
 p=WORK/'runs'/name/'output';resume=(p/'history-resume.csv').exists();suffix='-resume' if resume else ''
 hn,h=read(p/f'history{suffix}.csv');sn,s=read(p/f'states{suffix}.csv');nn,n=read(p/f'nodal_history{suffix}.csv');shift=L/2-4
 assert len(s)==len(h)*3*nx and len(n)==len(h)*nx
 hd={k:h[:,i] for i,k in enumerate(hn)};sd={k:s[:,i].reshape(-1,3*nx) for i,k in enumerate(sn)};nd={k:n[:,i].reshape(-1,nx) for i,k in enumerate(nn)};sd['x_m']-=shift
 segments=[]
 if previous is not None:
  _,saved=read(p/'restart_loaded.csv');clock=saved[-1,0]
  for current,old in [(hd,previous['h']),(sd,previous['s']),(nd,previous['n'])]:
   for key in current:current[key]=np.concatenate([old[key][previous['h']['time_s']<=clock+1e-10],current[key]],axis=0)
  for segment in previous.get('segments',[dict(path=previous['p'],start=-1.,end=previous['h']['time_s'][-1])]):
   if segment['start']<clock:segments.append(dict(path=segment['path'],start=segment['start'],end=min(segment['end'],clock)))
 else:clock=-1.
 segments.append(dict(path=p,start=clock,end=float(hd['time_s'][-1])))
 return dict(name=name,p=p,segments=segments,length=L,nx=nx,shift=shift,h=hd,s=sd,n=nd,node_x=np.arange(nx)*L/nx-shift,weights=np.tile(np.array([5/18,4/9,5/18])*L/nx,nx))

def original(n=16):return prior.case('period-'+str(n),n,128 if n==8 else 256,0 if n==8 else 4)

def baseline():
 c=load('longer',original())
 if (WORK/'runs/longer-completion/output/history-resume.csv').exists():
  c=load('longer-completion',c);c['name']='longer'
 return c

def field(c,t):
 for segment in reversed(c.get('segments',[dict(path=c['p'],start=-1,end=1e99)])):
  p=segment['path']/f'physical-fields-{round(t*1e6)}.csv'
  if segment['start']<t+1e-10 and t<=segment['end']+1e-10 and p.exists():return p
 raise FileNotFoundError((c['name'],t))

def reflection():
 a=original(8);b=load('reflected',L=8,nx=128);errors={}
 assert np.array_equal(a['h']['time_s'],b['h']['time_s'])
 for key in ['jump_m','damage','Theta','reference_m','relaxed_jump_m','absolute_slip_m','shear_Pa','pn_Pa','q_Pa','V_m_s']:
  errors[key]=float(np.max(abs(b['s'][key]-a['s'][key][:,::-1])))
 permutation=(-np.arange(128))%128
 for key in ['jump_m','relaxed_jump_m','signed_rate_m_s']:errors['nodal_'+key]=float(np.max(abs(b['n'][key]-a['n'][key][:,permutation])))
 _,aa=read(a['p']/'operators.csv');_,bb=read(b['p']/'operators.csv');operators={}
 for col,key in [(2,'K'),(3,'N')]:
  A=aa[:,col].reshape(128,128)[np.ix_(permutation,permutation)];B=bb[:,col].reshape(128,128);operators[key]=float(np.linalg.norm(A-B)/np.linalg.norm(A))
 fields=[]
 for t in [2.3,2.45,2.47]:
  _,aa=read(a['p']/f'physical-fields-{round(t*1e6)}.csv');_,bb=read(b['p']/f'physical-fields-{round(t*1e6)}.csv');aa[:,0]=8-aa[:,0];aa[:,1]=1-aa[:,1]
  aa=aa[np.lexsort((np.round(aa[:,1],10),np.round(aa[:,0],10)))];bb=bb[np.lexsort((np.round(bb[:,1],10),np.round(bb[:,0],10)))];assert np.max(abs(aa[:,:3]-bb[:,:3]))<1e-10
  fields.append(dict(time_s=t,stress_max_Pa=float(np.max(abs(aa[:,4:8]-bb[:,4:8]))),pressure_max_Pa=float(np.max(abs(aa[:,8:10]-bb[:,8:10])))))
 limits={key:(1e-7 if key.endswith('_Pa') else 1e-9 if key in ['damage','Theta'] else 1e-10) for key in errors}
 result=dict(passed=all(errors[k]<limits[k] for k in errors) and max(operators.values())<1e-10 and all(max(r['stress_max_Pa'],r['pressure_max_Pa'])<1e-7 for r in fields),errors=errors,limits=limits,operator_rotation_relative=operators,fields=fields)
 (OUT/'reflection-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']

def summary(c):
 saved=prior.OUT;scratch=WORK/'analysis-scratch';scratch.mkdir(exist_ok=True);prior.OUT=scratch
 diagnostic=dict(c)
 if not (c['p']/'derivative_checks.csv').exists():diagnostic['p']=next(s['path'] for s in reversed(c['segments']) if (s['path']/'derivative_checks.csv').exists())
 try:rows,result=prior.metrics(diagnostic)
 finally:prior.OUT=saved
 gate_values={};rawmax=0.
 for segment in c.get('segments',[dict(path=c['p'],start=-1,end=1e99)]):
  path=segment['path']/'derivative_checks.csv'
  if not path.exists():continue
  _,d=read(path);d=d[(d[:,0]>segment['start']+1e-10)&(d[:,0]<=segment['end']+1e-10)]
  if len(d):rawmax=max(rawmax,float(d[:,-1].max()))
  for r in d:gate_values[round(float(r[0]),10)]=float(r[-1])
  path=segment['path']/'derivative-extended.csv'
  if path.exists():
   _,d=read(path);d=d[(d[:,0]>segment['start']+1e-10)&(d[:,0]<=segment['end']+1e-10)]
   for r in d:gate_values[round(float(r[0]),10)]=float(r[2])
 result['maximum_raw_double_derivative_relative']=rawmax;result['maximum_accepted_derivative_gate_relative']=max(gate_values.values(),default=0.)
 # Retain the old summary's physical ledger, but recompute its auxiliary
 # background diagnostics for THIS forward loading plan, not the old 2.45 cap.
 memory=0.;previous=0.;reaction_old=0.;work_bg=0.;loss_bg=0.;memory_error=0.
 friction=np.cumsum(c['h']['friction_J_m']);radiation=np.cumsum(c['h']['radiation_J_m'])
 for i,row in enumerate(rows):
  t=row['time_s'];dt=c['h']['dt_s'][i];du=.001*(min(t,2.55)-min(previous,2.55));memory=(1/(1+.01*dt))*(memory+du)
  reaction=1e6*c['length']*memory;energy=.5*1e6*c['length']*memory**2;work_bg+=.5*(reaction+reaction_old)*du;loss_bg+=.02*dt*energy
  row.update(imposed_displacement_m=.001*min(t,2.55),reaction_excess_N_m=row['reaction_N_m']-reaction,bulk_excess_J_m=row['bulk_J_m']-energy,work_excess_J_m=row['work_J_m']-work_bg,Maxwell_excess_J_m=row['Maxwell_J_m']-loss_bg)
  row['friction_cumulative_J_m']=float(friction[i]);row['radiation_cumulative_J_m']=float(radiation[i])
  memory_error=max(memory_error,abs(memory-c['h']['wall_memory_m'][i]));reaction_old=reaction;previous=t
 result['background_memory_error_m']=memory_error
 result['older_245_to_end']=result.pop('hold')
 result['outside_older_8m_candidate']=result.pop('added_records')
 result['outside_older_8m_candidate']['age_difference_from_zero_rate_control']=result['outside_older_8m_candidate'].pop('theta_recurrence_error')
 result['maximum_weak_outside_older_8m_candidate_Pa']=result.pop('maximum_common_remote_weak_shear_Pa')
 uninstalled=c['s']['installed'][0]==0;result['maximum_uninstalled_weak_shear_Pa']=float(np.max(abs(c['s']['shear_Pa'][:,uninstalled])))
 x=c['node_x'];first=.5-c['shift'];last=c['length']-.5-c['shift'];left=(x>=first)&(x<=2.5)
 inc=c['n']['signed_rate_m_s']*c['h']['dt_s'][:,None]
 result['new_left_monotone']=bool(np.min(inc[:,left])>=0)
 for i,row in enumerate(rows):
  v=c['n']['jump_m'][i];cells=(x[:-1]>=first)&(x[1:]<=2.5)&(v[:-1]>=2/3000)&(v[1:]>=2/3000)
  row['full_complete_cells']=int(sum(cells))
  olda=c['s']['damage'][i-1] if i else ((c['s']['x_m'][0]>2.5)&(c['s']['x_m'][0]<5.5)).astype(float)
  row['maximum_phase_increment']=float(max(c['s']['damage'][i]-olda));row['peak_abs_velocity_m_s']=float(max(abs(c['s']['V_m_s'][i])))
  if not result['new_left_monotone']:raise RuntimeError('Continuous full-front reconstruction needs a history-aware treatment after reversal')
 result['final']=rows[-1]
 h=c['h'];E=h['bulk_J_m'];B=h['bond_J_m'];F=h['fracture_increment_J_m'];D=h['friction_J_m']+h['radiation_J_m'];rem=h['trapezoid_work_J_m']-(E-np.r_[0,E[:-1]]+B-np.r_[0,B[:-1]]+F+D+h['Maxwell_J_m'])
 for label,start in [('new_window',2.45),('new_hold',2.55)]:
  mask=h['time_s']>start+1e-10;fracture=float(sum(F[mask]));sliding=float(sum(D[mask]));absolute=float(sum(abs(rem[mask])))
  result[label]=dict(start_s=start,absolute_remainder_J_m=absolute,signed_remainder_J_m=float(sum(rem[mask])),fracture_J_m=fracture,sliding_J_m=sliding,relative_fracture=absolute/fracture if fracture else None,relative_sliding=absolute/sliding if sliding else None)
  result[label]['maximum_phase_increment']=max(r['maximum_phase_increment'] for r in rows if r['time_s']>start+1e-10)
  result[label]['peak_abs_velocity_m_s']=max(r['peak_abs_velocity_m_s'] for r in rows if r['time_s']>start+1e-10)
  i0=at(c,start);result[label].update(friction_J_m=float(sum(h['friction_J_m'][mask])),radiation_J_m=float(sum(h['radiation_J_m'][mask])),Maxwell_J_m=float(sum(h['Maxwell_J_m'][mask])),external_trapezoid_work_J_m=float(sum(h['trapezoid_work_J_m'][mask])),bulk_storage_change_J_m=float(E[-1]-E[i0]),bond_storage_change_J_m=float(B[-1]-B[i0]))
 result['fracture_inventory_identity_J_m']=abs(rows[-1]['integrated_new_damage_m']-rows[-1]['fracture_J_m'])
 table(OUT/(c['name']+'-history.csv'),rows)
 return rows,result

def longer():
 a=original();b=baseline();rows,result=summary(b);prefix={}
 for group in ['h','s','n']:
  # The old run snaps its actual load endpoint to 2.45; the extended run has
  # no loading break there. Preserve and measure that roundoff-sized clock
  # difference separately instead of silently rewriting either trajectory.
  ia=a['h']['time_s']<2.45-1e-10;ib=b['h']['time_s']<2.45-1e-10;errors={}
  for key in a[group]:
   if key=='step_seconds':continue
   errors[key]=float(np.max(abs(a[group][key][ia]-b[group][key][ib])))
  prefix[group]=dict(bitwise=all(v==0 for v in errors.values()),errors=errors)
  assert prefix[group]['bitwise'],prefix[group]
 i,j=at(a,2.45),at(b,2.45);endpoint={k:float(max(abs(a['s'][k][i]-b['s'][k][j]))) for k in ['jump_m','damage','Theta','reference_m','relaxed_jump_m','absolute_slip_m','shear_Pa','pn_Pa']}
 clock=abs(a['h']['time_s'][i]-b['h']['time_s'][j]);assert clock<=32*np.finfo(float).eps*2.45
 assert all(value<(1e-7 if key.endswith('_Pa') else 1e-9 if key in ['damage','Theta'] else 1e-10) for key,value in endpoint.items())
 result['old_endpoint_comparison']=dict(clock_difference_s=float(clock),field_differences=endpoint)
 partial=load('longer',a)
 if b['p']!=partial['p']:
  overlap={}
  for group in ['h','s','n']:
   end=partial['h']['time_s'][-1];mask=b['h']['time_s']<=end+1e-10
   errors={key:float(np.max(abs(partial[group][key]-b[group][key][mask]))) for key in partial[group] if key!='step_seconds'}
   overlap[group]=dict(bitwise=all(v==0 for v in errors.values()),maximum_error=max(errors.values()));assert overlap[group]['bitwise']
  result['completion_replayed_overlap']=overlap
 result['unchanged_past_prefix']=prefix;result['physical_target_passed']=result['final']['full_advance_m']>=1.5 and result['final']['cohesive_cells_left']>=8
 (OUT/'longer-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def compare():
 a=baseline();b=load('event-half',a);ra,sa=summary(a);rb,sb=summary(b);comparisons=[];profiles=[]
 for t in [2.45,2.50,2.55,2.58]:
  i,j=at(a,t),at(b,t);aa,bb=ra[i],rb[j];growth=a['s']['damage'][i]-((a['s']['x_m'][i]>2.5)&(a['s']['x_m'][i]<5.5));d=b['s']['damage'][j]-a['s']['damage'][i]
  comparisons.append(dict(time_s=t,growth_absolute_m=abs(bb['integrated_new_damage_m']-aa['integrated_new_damage_m']),growth_relative=abs(bb['integrated_new_damage_m']-aa['integrated_new_damage_m'])/bb['integrated_new_damage_m'],growth_L2_relative=float(np.sqrt(a['weights']@(d*d)/(a['weights']@(growth*growth)))),slip_absolute_m2=abs(bb['integrated_abs_jump_m2']-aa['integrated_abs_jump_m2']),slip_relative=abs(bb['integrated_abs_jump_m2']-aa['integrated_abs_jump_m2'])/bb['integrated_abs_jump_m2'],full_front_difference_m=abs(bb['continuous_left_full_x_m']-aa['continuous_left_full_x_m']),reaction_difference_N_m=bb['reaction_N_m']-aa['reaction_N_m']))
  for c,k in [(a,i),(b,j)]:
   for q,x in enumerate(c['s']['x_m'][k]):profiles.append(dict(case=c['name'],time_s=t,x_m=x,weight_m=c['weights'][q],damage=c['s']['damage'][k,q],jump_m=c['s']['jump_m'][k,q],theta=c['s']['Theta'][k,q],velocity_m_s=c['s']['V_m_s'][k,q],normal_pressure_Pa=c['s']['pn_Pa'][k,q],work_adjoint_shear_load_Pa=c['s']['shear_Pa'][k,q]))
 fields=[]
 for t in [2.55,2.58]:
  _,aa=read(field(a,t));_,bb=read(field(b,t));assert np.max(abs(aa[:,:3]-bb[:,:3]))<1e-10
  w=aa[:,2];metric=np.array([1,1,1,2]);diff=bb[:,4:8]-aa[:,4:8]
  fields.append(dict(time_s=t,full_tensor_relative_L2=float(np.sqrt(w@np.sum(diff*diff*metric,axis=1)/(w@np.sum(bb[:,4:8]**2*metric,axis=1)))),normal_RMS_Pa=float(np.sqrt(w@((bb[:,9]-aa[:,9])**2)/sum(w)))))
 gates=dict(growth_slip_under_1_percent=all(max(r['growth_relative'],r['slip_relative'])<=.01 for r in comparisons),full_front_under_quarter_cell=all(r['full_front_difference_m']<=.015625 for r in comparisons),new_window_energy_under_1_percent=all(max(s['new_window']['relative_fracture'],s['new_window']['relative_sliding'])<=.01 for s in [sa,sb]),hold_energy_under_1_percent=all(max(s['new_hold']['relative_fracture'],s['new_hold']['relative_sliding'])<=.01 for s in [sa,sb]))
 table(OUT/'temporal-comparison.csv',comparisons);table(OUT/'selected-profiles.csv',profiles);table(OUT/'temporal-fields.csv',fields)
 mechanical=frozen_mechanical_accuracy(b,rb)
 secants=[];fronts=[];cs=float(np.sqrt(1e6/2670))
 for c,rr in [(a,ra),(b,rb)]:
  for t in [2.45,2.50,2.55,2.58]:
   f=rr[at(c,t)];fronts.append(dict(case=c['name'],time_s=t,imposed_displacement_m=f['imposed_displacement_m'],left_full_advance_m=f['full_advance_m'],right_full_advance_m=f['right_full_advance_m'],complete_left_cells=f['full_complete_cells']))
  for t0,t1 in [(2.45,2.50),(2.50,2.55),(2.55,2.58)]:
   first,last=rr[at(c,t0)],rr[at(c,t1)];speed=(last['full_advance_m']-first['full_advance_m'])/(t1-t0)
   secants.append(dict(case=c['name'],start_s=t0,end_s=t1,full_advance_difference_m=last['full_advance_m']-first['full_advance_m'],secant_model_front_speed_m_s=speed,nominal_cs_for_radiation_m_s=cs,ratio_to_nominal_cs=speed/cs,interpretation='quasi-static model kinematics; not a physical rupture-speed qualification'))
 table(OUT/'fronts-versus-loading.csv',fronts);table(OUT/'front-secants.csv',secants)
 (OUT/'continuation-verification.json').write_text(json.dumps(dict(longer=sa,half=sb,comparisons=comparisons,fields=fields,frozen_mechanical_accuracy=mechanical,gates=gates),indent=2)+'\n');print(json.dumps(gates,indent=2))

def frozen_mechanical_accuracy(c,history):
 from analyze_cohesive_interface_native import surface_rule
 _,symbols=read(OUT/'analytical-L16-symbols.csv');K=symbols[:,1];N=1j*symbols[:,2];Q,_,_,P=surface_rule(256);rows=[];profiles=[]
 for t in [2.55,2.58]:
  i=at(c,t);m=c['n']['relaxed_jump_m'][i];shear=1e6*c['h']['wall_memory_m'][i]-P@np.fft.ifft(K*np.fft.fft(m)).real;normal=10000+P@np.fft.ifft(N*np.fft.fft(m)).real;x=c['s']['x_m'][i];native_s=c['s']['shear_Pa'][i];native_p=c['s']['pn_Pa'][i]
  front=history[i]['continuous_left_full_x_m']
  for label,mask in [('installed_Gamma',c['s']['installed'][i]>0),('left_tip',(x>=front-.5)&(x<=front+.5))]:
   w=c['weights'][mask];rows.append(dict(time_s=t,region=label,shear_RMS_Pa=float(np.sqrt(w@((native_s[mask]-shear[mask])**2)/sum(w))),normal_RMS_Pa=float(np.sqrt(w@((native_p[mask]-normal[mask])**2)/sum(w))),shear_max_Pa=float(max(abs(native_s[mask]-shear[mask]))),normal_max_Pa=float(max(abs(native_p[mask]-normal[mask])))))
  for q in range(768):profiles.append(dict(time_s=t,x_m=x[q],weight_m=c['weights'][q],native_shear_Pa=native_s[q],exact_Q1_Galerkin_shear_Pa=shear[q],native_normal_Pa=native_p[q],exact_Q1_Galerkin_normal_Pa=normal[q]))
 table(OUT/'frozen-mechanical-errors.csv',rows);table(OUT/'frozen-mechanical-profiles.csv',profiles);return rows

if __name__=='__main__':{'reflection':reflection,'longer':longer,'compare':compare}[sys.argv[1]]()
