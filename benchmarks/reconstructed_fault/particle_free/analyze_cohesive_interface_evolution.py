#!/usr/bin/env python3
"""Accepted native histories: no reconstructed ages or constitutive remapping."""
import json,csv
from pathlib import Path
import numpy as np
from run_cohesive_interface_evolution import ROOT,OUT
DX_GAUSS=np.array([5/18,4/9,5/18])
TIMES=[1.8,2.,2.3,2.38,2.42,2.45,2.47]

def read(path):
 with path.open() as f: names=f.readline().strip().split(',')
 a=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
 return names,a

def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def case(name,nx):
 p=OUT/name;hn,h=read(p/'history.csv');sn,s=read(p/'states.csv');nn,n=read(p/'nodal_history.csv')
 if (p/'history-resume.csv').exists():
  rn,r=read(p/'restart_loaded.csv');clock=r[-1,0];hn,v=read(p/'history-resume.csv');sn,z=read(p/'states-resume.csv');nn,y=read(p/'nodal_history-resume.csv')
  h=np.vstack([h[h[:,0]<=clock+1e-10],v]);s=np.vstack([s[s[:,0]<=clock+1e-10],z]);n=np.vstack([n[n[:,0]<=clock+1e-10],y])
 assert len(s)==len(h)*3*nx and len(n)==len(h)*nx,(name,len(s),len(h))
 assert np.all(np.diff(h[:,0])>0),name
 return dict(name=name,nx=nx,h={key:h[:,i] for i,key in enumerate(hn)},s={key:s[:,i].reshape(-1,3*nx) for i,key in enumerate(sn)},n={key:n[:,i].reshape(-1,nx) for i,key in enumerate(nn)},w=np.tile(DX_GAUSS*8/nx,nx))

def at(c,t):
 i=np.argmin(abs(c['h']['time_s']-t));assert abs(c['h']['time_s'][i]-t)<1e-8,(c['name'],t,c['h']['time_s'][i]);return i

def fronts(x,a,level):
 mask=a>=level;center=np.argmin(abs(x-4.));assert mask[center]
 left=right=center
 while left>0 and mask[left-1]:left-=1
 while right+1<len(mask) and mask[right+1]:right+=1
 detached=mask.copy();detached[left:right+1]=False
 return dict(left=x[left],right=x[right],left_lower=x[left-1] if left else x[left],right_upper=x[right+1] if right+1<len(x) else x[right],detached_points=int(sum(detached)))

def summary(c):
 rows=[];h=c['h'];s=c['s'];w=c['w'];x=s['x_m'][0];initial=((x>2.5)&(x<5.5)).astype(float)
 E0=0.;B0=0.;prevE=np.r_[E0,h['bulk_J_m'][:-1]];prevB=np.r_[B0,h['bond_J_m'][:-1]]
 rem=h['trapezoid_work_J_m']-(h['bulk_J_m']-prevE+h['bond_J_m']-prevB+h['fracture_increment_J_m']+h['friction_J_m']+h['radiation_J_m']+h['Maxwell_J_m'])
 sums={key:np.cumsum(value) for key,value in h.items() if key in ['fracture_increment_J_m','trapezoid_work_J_m','Maxwell_J_m']}
 sliding=np.cumsum(h['friction_J_m']+h['radiation_J_m']);absolute=np.cumsum(abs(rem))
 for i,t in enumerate(h['time_s']):
  a=s['damage'][i];phase=(a-initial)@w;slip=abs(s['jump_m'][i])@w
  r=dict(case=c['name'],time_s=t,imposed_displacement_m=.001*min(t,2.45),integrated_new_damage_m=phase,peak_new_damage=float(max(a-initial)),integrated_abs_jump_m2=slip,peak_abs_jump_m=float(max(abs(s['jump_m'][i]))),reaction_N_m=float(w@s['shear_Pa'][i]),min_normal_Pa=float(min(s['pn_Pa'][i])),remote_weak_shear_Pa=float(max(abs(s['shear_Pa'][i][(x<.5)|(x>7.5)]))),physical_remainder_J_m=float(rem[i]),absolute_remainder_to_date_J_m=float(absolute[i]),sliding_to_date_J_m=float(sliding[i]),fracture_to_date_J_m=float(sums['fracture_increment_J_m'][i]),bulk_J_m=h['bulk_J_m'][i],bond_J_m=h['bond_J_m'][i],trapezoid_work_to_date_J_m=float(sums['trapezoid_work_J_m'][i]),Maxwell_to_date_J_m=float(sums['Maxwell_J_m'][i]),curvature_Pa_m=h['min_curvature_Pa_m'][i],weak_error_Pa=h['force_error_Pa'][i],step_seconds=h['step_seconds'][i])
  for lev,label in [(.01,'001'),(.05,'005'),(.5,'050'),(1.,'100')]:
   f=fronts(x,a,lev)
   for key,value in f.items():r['a'+label+'_'+key]=value
  r['left_process_zone_width_m']=r['a100_left']-r['a001_left'];r['left_process_zone_cells']=r['left_process_zone_width_m']/(8/c['nx'])
  rows.append(r)
 return rows

def main():
 cases=[case('straight-hold-coarse',64),case('straight-long',128),case('straight-half-window',128)]
 results={};allrows=[];profiles=[];comparisons=[];cached={}
 for c in cases:
  rows=summary(c);allrows+=rows;cached[c['name']]=rows
  table(OUT/(c['name']+'-compact.csv'),rows)
  h=c['h'];s=c['s'];r=rows[-1];energy=dict(absolute_remainder_J_m=r['absolute_remainder_to_date_J_m'],signed_remainder_J_m=float(sum(h['trapezoid_work_J_m'])-(h['bulk_J_m'][-1]+h['bond_J_m'][-1]+sum(h['fracture_increment_J_m']+h['friction_J_m']+h['radiation_J_m']+h['Maxwell_J_m']))),relative_fracture=r['absolute_remainder_to_date_J_m']/r['fracture_to_date_J_m'],relative_sliding=r['absolute_remainder_to_date_J_m']/r['sliding_to_date_J_m'])
  results[c['name']]=dict(final=r,energy=energy,max_ledger_J_m=float(max(abs(h['ledger_error_J_m']))),max_weak_Pa=float(max(h['force_error_Pa'])),minimum_curvature_Pa_m=float(min(h['min_curvature_Pa_m'])),max_derivative_relative=float(np.loadtxt(OUT/c['name']/'derivative_checks.csv',delimiter=',',skiprows=1,ndmin=2)[:,-1].max()))
  for t in TIMES:
   if t>h['time_s'][-1]+1e-8:continue
   i=at(c,t)
   for q in range(3*c['nx']):
    profiles.append(dict(case=c['name'],time_s=t,x_m=s['x_m'][i,q],weight_m=c['w'][q],damage=s['damage'][i,q],jump_m=s['jump_m'][i,q],theta=s['Theta'][i,q],V_m_s=s['V_m_s'][i,q],cohesive_q_Pa=s['q_Pa'][i,q],weak_shear_Pa=s['shear_Pa'][i,q],normal_pressure_Pa=s['pn_Pa'][i,q],installed=s['installed'][i,q]))
 coarse,fine,half=cases
 for c,label in [(coarse,'spatial equal dt'),(half,'temporal fixed mesh')]:
  for t in TIMES:
   if t>min(c['h']['time_s'][-1],fine['h']['time_s'][-1])+1e-8:continue
   i,j=at(c,t),at(fine,t);x=c['s']['x_m'][i];y=fine['s']['x_m'][j];a=c['s']['damage'][i]-((x>2.5)&(x<5.5));b=fine['s']['damage'][j]-((y>2.5)&(y<5.5))
   # Same .125 m physical box average of actual Gauss histories.
   aa=(a.reshape(-1,3)@DX_GAUSS).reshape(64,-1).mean(axis=1);bb=(b.reshape(-1,3)@DX_GAUSS).reshape(64,-1).mean(axis=1)
   ra=cached[c['name']][i];rb=cached[fine['name']][j]
   da=ra['integrated_new_damage_m'];db=rb['integrated_new_damage_m'];sa=ra['integrated_abs_jump_m2'];sb=rb['integrated_abs_jump_m2']
   comparisons.append(dict(kind=label,time_s=t,growth_absolute_difference_m=abs(da-db),growth_relative_difference=abs(da-db)/max(abs(db),1e-30),growth_box_L2_relative=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30)),slip_relative_difference=abs(sa-sb)/max(sb,1e-30),left_full_Gauss_front_difference_m=abs(ra['a100_left']-rb['a100_left']),left_005_front_difference_m=abs(ra['a005_left']-rb['a005_left'])))
 table(OUT/'selected-profiles.csv',profiles);table(OUT/'comparison.csv',comparisons)
 # Focused replay errors exclude the shared prefix; no credit for its old error.
 for c in [fine,half]:
  h=c['h'];_,loaded=read(OUT/c['name']/'restart_loaded.csv');start=loaded[-1,0];mask=h['time_s']>start+1e-10
  srows=cached[c['name']];raw=np.array([r['physical_remainder_J_m'] for r in srows])[mask]
  results[c['name']]['window']=dict(incoming_s=float(start),end_s=float(h['time_s'][-1]),absolute_remainder_J_m=float(sum(abs(raw))),sliding_J_m=float(sum((h['friction_J_m']+h['radiation_J_m'])[mask])),fracture_J_m=float(sum(h['fracture_increment_J_m'][mask])))
 (OUT/'growth-verification.json').write_text(json.dumps(dict(cases=results,comparisons=comparisons),indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
