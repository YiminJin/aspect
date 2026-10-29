#!/usr/bin/env python3
"""Continuous Q1 jump threshold: uses no interpolated frictional histories."""
import numpy as np
from analyze_cohesive_interface_evolution import case,at,OUT,table,TIMES

def main():
 rows=[];milestones=[];df=2/3000
 for name,nx in [('straight-hold-coarse',64),('straight-long',128),('straight-half-window',128)]:
  c=case(name,nx);x=np.arange(nx)*8/nx;left=(x>=.5)&(x<=2.5);step=c['n']['signed_rate_m_s']*c['h']['dt_s'][:,None]
  assert step[:,left].min()>=0.,'Continuous damage inference requires monotonic new-material slip'
  roots=[]
  for v in c['n']['jump_m']:
   ids=np.flatnonzero((x<=2.5)&(v>=df));root=2.5
   if len(ids):
    j=ids[0];assert j>0 and v[j]>v[j-1];root=x[j-1]+(df-v[j-1])/(v[j]-v[j-1])*(8/nx)
   roots.append(root)
  for t in TIMES:
   if t>c['h']['time_s'][-1]+1e-9:continue
   i=at(c,t);v=c['n']['jump_m'][i];complete=0
   for j in range(nx-1):
    if x[j]>=.5 and x[j+1]<=2.5 and v[j]>=df and v[j+1]>=df:complete+=1
   rows.append(dict(case=name,time_s=t,left_full_jump_threshold_x_m=roots[i],advance_m=2.5-roots[i],complete_new_full_cells=complete,complete_new_full_length_m=complete*8/nx,min_new_left_nodal_increment_m=float(step[:,left].min()),interpretation='Q1 physical jump threshold; monotonicity verified; no age remapping'))
  for label,mask in [('Gauss new damage >1e-8',np.max(c['s']['damage']-((c['s']['x_m']>2.5)&(c['s']['x_m']<5.5)),axis=1)>1e-8),('continuous new full region',np.array(roots)<2.5),('continuous full advance >=.5 m',np.array(roots)<=2.)]:
   idx=np.flatnonzero(mask)
   if len(idx):
    j=idx[0];milestones.append(dict(case=name,event=label,lower_s=float(c['h']['time_s'][j-1]) if j else 0.,upper_s=float(c['h']['time_s'][j])))
 table(OUT/'front-reference.csv',rows);table(OUT/'milestones.csv',milestones)
 print(rows[-1]);print(milestones)
if __name__=='__main__':main()
