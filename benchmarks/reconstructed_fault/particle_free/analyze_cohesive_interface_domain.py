#!/usr/bin/env python3
"""Physical-domain differences, with actual quadrature and no history remapping."""
import json,csv,sys
import numpy as np
from run_cohesive_interface_domain import ROOT,WORK,OUT
from analyze_cohesive_interface_evolution import read,table,case as archived_case,fronts
TIMES=[1.8,2.,2.3,2.38,2.42,2.45,2.47]
GAUSS=np.array([5/18,4/9,5/18])

def case(name,length,nx,shift):
 p=WORK/'runs'/name/'output';hn,h=read(p/'history.csv');sn,s=read(p/'states.csv');nn,n=read(p/'nodal_history.csv')
 if (p/'history-resume.csv').exists():
  _,loaded=read(p/'restart_loaded.csv');clock=loaded[-1,0]
  hn,hh=read(p/'history-resume.csv');sn,ss=read(p/'states-resume.csv');nn,nnn=read(p/'nodal_history-resume.csv')
  h=np.vstack([h[h[:,0]<=clock+1e-10],hh]);s=np.vstack([s[s[:,0]<=clock+1e-10],ss]);n=np.vstack([n[n[:,0]<=clock+1e-10],nnn])
 assert len(s)==len(h)*3*nx and len(n)==len(h)*nx
 c=dict(name=name,length=length,nx=nx,shift=shift,p=p,h={k:h[:,i] for i,k in enumerate(hn)},s={k:s[:,i].reshape(-1,3*nx) for i,k in enumerate(sn)},n={k:n[:,i].reshape(-1,nx) for i,k in enumerate(nn)},weights=np.tile(GAUSS*length/nx,nx))
 c['s']['x_m']=c['s']['x_m']-shift;c['node_x']=np.arange(nx)*length/nx-shift
 return c

def at(c,t):
 i=int(np.argmin(abs(c['h']['time_s']-t)));assert abs(c['h']['time_s'][i]-t)<1e-9,(c['name'],t);return i

def reference_check():
 a=case('period-8',8,128,0);b=archived_case('straight-half-window',128);errors={}
 indices=np.array([int(np.argmin(abs(b['h']['time_s']-t))) for t in a['h']['time_s']])
 assert max(abs(a['h']['time_s']-b['h']['time_s'][indices]))<1e-10
 for key in ['damage','jump_m','Theta','reference_m','relaxed_jump_m','absolute_slip_m','shear_Pa','pn_Pa']:
  errors[key]=float(np.max(abs(a['s'][key]-b['s'][key][indices])))
 for key in ['jump_m','relaxed_jump_m']:errors['nodal_'+key]=float(np.max(abs(a['n'][key]-b['n'][key][indices])))
 errors['wall_memory_m']=float(max(abs(a['h']['wall_memory_m']-b['h']['wall_memory_m'][indices])))
 for key,value in errors.items():assert value<(1e-7 if key in ['shear_Pa','pn_Pa'] else 1e-10),(key,value)
 aa=a['h']['dt_s'];bb=b['h']['dt_s'];assert np.array_equal(np.round(aa[aa>1e-10],10),np.round(bb[bb>1e-10],10))
 result=dict(passed=True,principal_timestep_sequence_identical=True,maximum_component_differences=errors,fresh_steps=len(aa),archived_steps=len(bb),snapshot_diagnostics_nonpublication_verified=True)
 (OUT/'reference-identity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def metrics(c):
 h,s,w=c['h'],c['s'],c['weights'];x=s['x_m'][0];old=((x>2.5)&(x<5.5)).astype(float);t=h['time_s'];E=h['bulk_J_m'];B=h['bond_J_m'];F=h['fracture_increment_J_m'];D=h['friction_J_m']+h['radiation_J_m'];M=h['Maxwell_J_m'];W=h['trapezoid_work_J_m'];rem=W-(E-np.r_[0.,E[:-1]]+B-np.r_[0.,B[:-1]]+F+D+M)
 # Independently integrate a homogeneous Maxwell strip on this exact time grid.
 memory=0.;Rold=0.;prev=0.;bgE=[];bgW=[];bgM=[];memory_error=0.
 for i,time in enumerate(t):
  du=.001*(min(time,2.45)-min(prev,2.45));memory=(1/(1+.01*h['dt_s'][i]))*(memory+du);reaction=1e6*c['length']*memory;energy=.5*1e6*c['length']*memory**2
  bgE.append(energy);bgW.append(.5*(reaction+Rold)*du);bgM.append(.02*h['dt_s'][i]*energy);memory_error=max(memory_error,abs(memory-h['wall_memory_m'][i]));Rold=reaction;prev=time
 bgE=np.array(bgE);bgW=np.cumsum(bgW);bgM=np.cumsum(bgM)
 new=(s['damage']-old)@w;slip=abs(s['jump_m'])@w;reaction=s['shear_Pa']@w
 left=(c['node_x']>=.5)&(c['node_x']<=2.5);increments=c['n']['signed_rate_m_s']*h['dt_s'][:,None];assert np.min(increments[:,left])>=0.
 roots=[];rightroots=[];cells=[]
 for v in c['n']['jump_m']:
  ids=np.flatnonzero((c['node_x']<=2.5)&(v>=2/3000));root=2.5
  if len(ids):
   j=ids[0];root=c['node_x'][j-1]+(2/3000-v[j-1])/(v[j]-v[j-1])*(c['length']/c['nx'])
  roots.append(root);cell=(c['node_x'][:-1]>=.5)&(c['node_x'][1:]<=2.5)&(v[:-1]>=2/3000)&(v[1:]>=2/3000);cells.append(int(sum(cell)))
  ids=np.flatnonzero((c['node_x']>=5.5)&(v>=2/3000));root=5.5
  if len(ids):
   j=ids[-1];assert j+1<len(v);root=c['node_x'][j]+(2/3000-v[j])/(v[j+1]-v[j])*(c['length']/c['nx'])
  rightroots.append(root)
 values={k:np.cumsum(v) for k,v in [('fracture_J_m',F),('sliding_J_m',D),('Maxwell_J_m',M),('work_J_m',W),('absolute_remainder_J_m',abs(rem)),('signed_remainder_J_m',rem)]}
 rows=[]
 for i,time in enumerate(t):
  q=s['damage'][i];a=fronts(x,q,.01);full=fronts(x,q,1.);rows.append(dict(case=c['name'],time_s=float(time),imposed_displacement_m=.001*min(time,2.45),integrated_new_damage_m=float(new[i]),integrated_abs_jump_m2=float(slip[i]),peak_jump_m=float(max(abs(s['jump_m'][i]))),continuous_left_full_x_m=float(roots[i]),full_advance_m=float(2.5-roots[i]),full_complete_cells=cells[i],cohesive_left_001_m=float(a['left']),cohesive_right_001_m=float(a['right']),cohesive_cells_left=float((full['left']-a['left'])/(c['length']/c['nx'])),detached_001_points=a['detached_points'],reaction_N_m=float(reaction[i]),reaction_excess_N_m=float(reaction[i]-1e6*c['length']*h['wall_memory_m'][i]),bulk_J_m=float(E[i]),bond_J_m=float(B[i]),bulk_excess_J_m=float(E[i]-bgE[i]),work_excess_J_m=float(values['work_J_m'][i]-bgW[i]),Maxwell_excess_J_m=float(values['Maxwell_J_m'][i]-bgM[i]),minimum_normal_Pa=float(h['min_side_contact_pressure_Pa'][i]),**{key:float(v[i]) for key,v in values.items()}))
  for level,label in [(.01,'001'),(.05,'005'),(.5,'050'),(1.,'100')]:
   for key,value in fronts(x,q,level).items():rows[-1]['a'+label+'_'+key]=float(value)
  rows[-1]['continuous_right_full_x_m']=float(rightroots[i]);rows[-1]['right_full_advance_m']=float(rightroots[i]-5.5)
  for side,mask in [('left',x<2.5),('right',x>5.5)]:rows[-1][side+'_integrated_new_damage_m']=float((q[mask]-old[mask])@w[mask])
 table(OUT/(c['name']+'-history.csv'),rows)
 # Added records must remain a zero-work representation. Use all accepted steps.
 added=((x<.5)|(x>7.5))&(s['installed'][0]>0)
 activity={key:float(np.max(abs(s[key][:,added]),initial=0)) for key in ['damage','jump_m','reference_m','relaxed_jump_m','absolute_slip_m']}
 theta=1.;error=0.
 for i,dt in enumerate(h['dt_s']):
  theta=(theta+5e-6*dt/.001);error=max(error,float(np.max(abs(s['Theta'][i,added]-theta),initial=0)))
 nodes=((c['node_x']<.5)|(c['node_x']>7.5))&(c['node_x']>=.5-c['shift'])&(c['node_x']<=c['length']-.5-c['shift'])
 activity['native_nodal_jump_m']=float(np.max(abs(c['n']['jump_m'][:,nodes]),initial=0));activity['theta_recurrence_error']=error
 results=dict(final=rows[-1],maximum_weak_Pa=float(max(h['force_error_Pa'])),max_ledger_J_m=float(max(abs(h['ledger_error_J_m']))),minimum_curvature_Pa_m=float(min(h['min_curvature_Pa_m'])),background_memory_error_m=memory_error,added_records=activity)
 results['raw_energy_over_fracture']=rows[-1]['absolute_remainder_J_m']/rows[-1]['fracture_J_m'];results['raw_energy_over_sliding']=rows[-1]['absolute_remainder_J_m']/rows[-1]['sliding_J_m']
 _,derivatives=read(c['p']/'derivative_checks.csv')
 results['maximum_raw_double_derivative_relative']=float(derivatives[:,-1].max())
 checked={round(float(row[0]),10):float(row[-1]) for row in derivatives}
 if (c['p']/'derivative-extended.csv').exists():
  _,extended=read(c['p']/'derivative-extended.csv')
  for row in extended:checked[round(float(row[0]),10)]=float(row[2])
 results['maximum_accepted_derivative_gate_relative']=max(checked.values())
 results['publications_once']=bool(np.array_equal(h['publications'],np.arange(1,len(t)+1)))
 assert results['publications_once']
 remote=(x<.5)|(x>7.5)
 results['maximum_common_remote_weak_shear_Pa']=float(np.max(abs(s['shear_Pa'][:,remote])))
 for name,mask in [('event_window',t>2.2975+1e-10),('hold',t>2.45+1e-10)]:
  results[name]=dict(absolute_remainder_J_m=float(sum(abs(rem[mask]))),signed_remainder_J_m=float(sum(rem[mask])),fracture_J_m=float(sum(F[mask])),sliding_J_m=float(sum(D[mask])))
  results[name]['relative_fracture']=results[name]['absolute_remainder_J_m']/results[name]['fracture_J_m'];results[name]['relative_sliding']=results[name]['absolute_remainder_J_m']/results[name]['sliding_J_m']
 return rows,results

def restart_overlap(c):
 original=WORK/'failed-first-attempt';results={}
 # Compare only accepted, physical columns. Solver timing is not a state.
 for filename,key in [('states.csv','s'),('nodal_history.csv','n')]:
  names,old=read(original/filename)
  if 'x_m' in names:old[:,names.index('x_m')]-=c['shift']
  new=np.column_stack([c[key][name].reshape(-1) for name in names])
  same=new[new[:,0]<=old[-1,0]+1e-10];assert same.shape==old.shape
  results[filename]=dict(bitwise=bool(np.array_equal(same,old)),maximum_error=float(np.max(abs(same-old))))
  assert results[filename]['bitwise'],results[filename]
 names,old=read(original/'history.csv');new=np.column_stack([c['h'][name] for name in names]);same=new[new[:,0]<=old[-1,0]+1e-10]
 cols=[i for i,name in enumerate(names) if name!='step_seconds']
 results['history_physical_columns']=dict(bitwise=bool(np.array_equal(old[:,cols],same[:,cols])),maximum_error=float(np.max(abs(old[:,cols]-same[:,cols]))))
 assert results['history_physical_columns']['bitwise']
 (OUT/'restart-overlap.json').write_text(json.dumps(results,indent=2)+'\n')


def field_comparison(a,b):
 rows=[];cells=[];temporal=[];global_rows=[]
 for t in [2.3,2.45,2.47]:
  arrays=[]
  for c in [a,b]:
   _,v=read(c['p']/f'physical-fields-{round(t*1e6)}.csv');v[:,0]-=c['shift'];common=v[(v[:,0]>=0)&(v[:,0]<=8)]
   order=np.lexsort((np.round(common[:,1],11),np.round(common[:,0],11)));common=common[order];arrays.append((v,common))
  v,x=arrays[0];vv,y=arrays[1];assert x.shape==y.shape and np.max(abs(x[:,:3]-y[:,:3]))<1e-10
  for region,mask in [('common_8m',np.ones(len(x),bool)),('tip_box',(x[:,0]>=1)&(x[:,0]<=3)&(x[:,1]>=.2)&(x[:,1]<=.6))]:
   weights=x[mask,2];ta=x[mask,4:8];tb=y[mask,4:8];d=tb-ta;metric=np.array([1,1,1,2]);norm=lambda z:np.sum(z*z*metric,axis=1)
   rows.append(dict(time_s=t,region=region,area_m2=float(sum(weights)),full_tensor_relative_L2=float(np.sqrt(weights@norm(d)/(weights@norm(ta)))),normal_pressure_RMS_difference_Pa=float(np.sqrt(weights@((y[mask,9]-x[mask,9])**2)/sum(weights))),pprime_RMS_difference_Pa=float(np.sqrt(weights@((y[mask,8]-x[mask,8])**2)/sum(weights))),max_shear_difference_Pa=float(max(abs(y[mask,7]-x[mask,7]))),bulk_storage_8_J_m=float(weights@norm(ta)/(4e6)),bulk_storage_16_common_J_m=float(weights@norm(tb)/(4e6))))
   rows[-1].update(mean_pprime_8_Pa=float(weights@x[mask,8]/sum(weights)),mean_pprime_16_Pa=float(weights@y[mask,8]/sum(weights)),mean_normal_8_Pa=float(weights@x[mask,9]/sum(weights)),mean_normal_16_Pa=float(weights@y[mask,9]/sum(weights)))
  oldpath=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_evolution/physical-fields'/f'physical-fields-{round(t*1e6)}.csv'
  _,coarse_time=read(oldpath);order=np.lexsort((np.round(coarse_time[:,1],11),np.round(coarse_time[:,0],11)));coarse_time=coarse_time[order]
  assert coarse_time.shape==x.shape and np.max(abs(coarse_time[:,:3]-x[:,:3]))<1e-10
  for region,mask in [('common_8m',np.ones(len(x),bool)),('tip_box',(x[:,0]>=1)&(x[:,0]<=3)&(x[:,1]>=.2)&(x[:,1]<=.6))]:
   weights=x[mask,2];difference=x[mask,4:8]-coarse_time[mask,4:8];reference=x[mask,4:8];metric=np.array([1,1,1,2])
   temporal.append(dict(time_s=t,region=region,full_tensor_relative_L2=float(np.sqrt(weights@np.sum(difference*difference*metric,axis=1)/(weights@np.sum(reference*reference*metric,axis=1)))),normal_pressure_RMS_difference_Pa=float(np.sqrt(weights@((x[mask,9]-coarse_time[mask,9])**2)/sum(weights)))))
  for c,(v,_) in zip([a,b],arrays):
   global_rows.append(dict(case=c['name'],time_s=t,global_mean_pprime_Pa=float(v[:,2]@v[:,8]/sum(v[:,2])),global_min_normal_Pa=float(min(v[:,9])),old_Gamma_exterior_max_abs_shear_Pa=float(max(abs(v[(v[:,0]<.5)|(v[:,0]>7.5),7])))))
   q=v.reshape(-1,36,10);w=q[:,:,2];vol=w.sum(axis=1);mean=(q*w[:,:,None]).sum(axis=1)/vol[:,None]
   for r,area in zip(mean,vol):cells.append(dict(case=c['name'],time_s=t,x_m=float(r[0]),y_m=float(r[1]),area_m2=float(area),tau_xy_Pa=float(r[7]),pprime_Pa=float(r[8]),normal_pressure_Pa=float(r[9])))
 table(OUT/'physical-comparison.csv',rows);table(OUT/'physical-cells.csv',cells);table(OUT/'temporal-field-scale.csv',temporal);table(OUT/'bulk-global.csv',global_rows);return rows

def operator_comparison(a,b):
 _,aa=read(a['p']/'operators.csv');_,bb=read(b['p']/'operators.csv')
 ia=np.arange(9,120);ib=ia+64;rows=[]
 for col,label in [(2,'K'),(3,'normal_map_N')]:
  A=aa[:,col].reshape(128,128)[np.ix_(ia,ia)];B=bb[:,col].reshape(256,256)[np.ix_(ib,ib)]
  rows.append(dict(operator=label,quantity='common_movable_block_relative_Frobenius',time_s=0.,value=float(np.linalg.norm(B-A)/np.linalg.norm(A))))
  for t in [2.3,2.45,2.47]:
   memory=a['n']['relaxed_jump_m'][at(a,t),ia];d=(B-A)@memory
   rows.append(dict(operator=label,quantity='same_reference_memory_RMS_response_difference_Pa',time_s=t,value=float(np.sqrt(np.mean(d*d)))))
 table(OUT/'operator-comparison.csv',rows)


def main():
 if len(sys.argv)>1 and sys.argv[1]=='reference-check':reference_check();return
 a=case('period-8',8,128,0);b=case('period-16',16,256,4);restart_overlap(b);ra,sa=metrics(a);rb,sb=metrics(b);comparisons=[];profiles=[];surface=[]
 for t in TIMES:
  i,j=at(a,t),at(b,t);aa=ra[i];bb=rb[j]
  for key in ['integrated_new_damage_m','integrated_abs_jump_m2','reaction_excess_N_m','bulk_excess_J_m','work_excess_J_m']:
   diff=bb[key]-aa[key];comparisons.append(dict(time_s=t,quantity=key,value_8=aa[key],value_16=bb[key],signed_difference=diff,relative_change=abs(diff)/max(abs(aa[key]),1e-30)))
  comparisons.append(dict(time_s=t,quantity='continuous_left_full_x_m',value_8=aa['continuous_left_full_x_m'],value_16=bb['continuous_left_full_x_m'],signed_difference=bb['continuous_left_full_x_m']-aa['continuous_left_full_x_m'],relative_change=abs(bb['continuous_left_full_x_m']-aa['continuous_left_full_x_m'])/.0625))
  ma=(a['s']['x_m'][i]>=.5)&(a['s']['x_m'][i]<=7.5);mb=(b['s']['x_m'][j]>=.5)&(b['s']['x_m'][j]<=7.5)
  assert np.max(abs(a['s']['x_m'][i,ma]-b['s']['x_m'][j,mb]))<1e-10
  weights=a['weights'][ma];old=((a['s']['x_m'][i,ma]>2.5)&(a['s']['x_m'][i,ma]<5.5));growth=a['s']['damage'][i,ma]-old
  change=b['s']['damage'][j,mb]-a['s']['damage'][i,ma]
  surface.append(dict(time_s=t,growth_relative_L2=float(np.sqrt(weights@(change*change)/max(weights@(growth*growth),1e-30))),jump_RMS_difference_m=float(np.sqrt(weights@((b['s']['jump_m'][j,mb]-a['s']['jump_m'][i,ma])**2)/sum(weights))),theta_RMS_difference=float(np.sqrt(weights@((b['s']['Theta'][j,mb]-a['s']['Theta'][i,ma])**2)/sum(weights))),current_normal_RMS_difference_Pa=float(np.sqrt(weights@((b['s']['pn_Pa'][j,mb]-a['s']['pn_Pa'][i,ma])**2)/sum(weights)))))
  for c,k in [(a,i),(b,j)]:
   for q in range(3*c['nx']):profiles.append(dict(case=c['name'],time_s=t,x_m=c['s']['x_m'][k,q],weight_m=c['weights'][q],damage=c['s']['damage'][k,q],jump_m=c['s']['jump_m'][k,q],theta=c['s']['Theta'][k,q],velocity_m_s=c['s']['V_m_s'][k,q],weak_shear_Pa=c['s']['shear_Pa'][k,q],normal_pressure_Pa=c['s']['pn_Pa'][k,q]))
 table(OUT/'comparison.csv',comparisons);table(OUT/'profiles.csv',profiles);table(OUT/'surface-comparison.csv',surface);fields=field_comparison(a,b);operator_comparison(a,b)
 late=[r for r in comparisons if r['time_s']>=2.3]
 gates=dict(late_growth_slip_below_2_percent=all(r['relative_change']<.02 for r in late if r['quantity'] in ['integrated_new_damage_m','integrated_abs_jump_m2']),left_full_front_below_005_m=all(abs(r['signed_difference'])<.05 for r in late if r['quantity']=='continuous_left_full_x_m'),tip_tensor_below_1_percent=all(r['full_tensor_relative_L2']<.01 for r in fields if r['region']=='tip_box'),tip_normal_RMS_below_10_Pa=all(r['normal_pressure_RMS_difference_Pa']<10 for r in fields if r['region']=='tip_box'),common_surface_normal_RMS_below_10_Pa=all(r['current_normal_RMS_difference_Pa']<10 for r in surface if r['time_s']>=2.3),raw_whole_energy_below_1_percent=all(c['raw_energy_over_fracture']<.01 and c['raw_energy_over_sliding']<.01 for c in [sa,sb]),added_records_exactly_tied=all(v==0 for key,v in sb['added_records'].items() if key!='theta_recurrence_error'))
 schedule_identical=bool(np.array_equal(a['h']['time_s'],b['h']['time_s']) and np.array_equal(a['h']['dt_s'],b['h']['dt_s']))
 result=dict(period8=sa,period16=sb,accepted_time_sequence_bitwise=schedule_identical,physical_fields=fields,comparisons=comparisons,predeclared_gates=gates)
 assert schedule_identical,'Domain controls acquired different accepted timestep sequences'
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
