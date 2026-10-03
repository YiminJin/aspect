#!/usr/bin/env python3
"""Signed, physically matched timestep-zero comparisons; unchanged qualification gate."""
from pathlib import Path
import csv, json
import numpy as np
r=Path(__file__).resolve().parents[1]; dest=Path(__file__).resolve().parent/'results';dest.mkdir(exist_ok=True)
def out(c):return r/f'output-endpoint-t0-{c}'
def read(p):return list(csv.DictReader(p.open()))
def nodes(c,obs=None):
 if obs is None:obs=max(int(p.stem.split('_')[1]) for p in out(c).glob('nodes_*.csv'))
 data=read(out(c)/f'nodes_{obs}.csv')
 return {k:np.array([float(x[k]) for x in data]) for k in data[0]},obs
def match(a,rev):return a[::-1] if rev else a
def mat(d,diag,edge):return np.diag(d[diag])+np.diag(d[edge][:-1],1)+np.diag(d[edge][:-1],-1)
def mapped_mat(d,diag,edge,rev):
 m=mat(d,diag,edge);return m[::-1,::-1] if rev else m
def qp(c,obs):return {(x['cell'],x['qp']):{k:(v if k=='cell' else float(v)) for k,v in x.items()} for x in read(out(c)/f'samples_{obs}.csv')}
def basis(q,rev):
 j=int(q['segment']);xi=q['xi'];n=np.zeros(37);n[j]=1-xi;n[j+1]=xi
 return n[::-1] if rev else n
summary={};tables=[]
for suffix,fc,rc in [('production','forward','reverse'),('tight','tight','reverse-tight')]:
 a,ao=nodes(fc);b,bo=nodes(rc);A={k:v for k,v in a.items()};B={k:match(v,True) for k,v in b.items()}
 assert np.max(abs(A['x']-B['x']))<1e-10 and np.max(abs(A['y']-B['y']))<1e-10
 qa,qb=qp(fc,ao),qp(rc,bo);assert qa.keys()==qb.keys()
 Ma=mat(a,'M_diag','M_right');Mb=mapped_mat(b,'M_diag','M_right',True)
 Ka=mat(a,'K_diag','K_right');Kb=mapped_mat(b,'K_diag','K_right',True)
 Ha=Ma+400*Ka;Hb=Mb+400*Kb
 Mmu=mat(a,'Mmu_diag','Mmu_right')
 # Derivative of discrete friction+damping at the observed forward state.
 D=np.zeros((37,37));load=np.zeros(37);moment=np.zeros(37);diff_geom=np.zeros(37)
 for key, q in qa.items():
  z=qb[key];na,nb=basis(q,False),basis(z,True)
  D+=q['weight']*(q['filtered']*q['dmu']+4624440.)*np.outer(na,na)
  # Explicit geometry/quadrature contribution at common forward nodal V,z.
  vf=nb@A['V'];tf=nb@A['Theta'];sf=nb@A['z']
  mu=.025*np.arcsinh(vf/(2e-6)*np.exp((.6+.015*np.log(tf*1e-6/.008))/.025))
  fa=q['mu']*q['filtered']+4624440.*q['V']
  fb=mu*sf+4624440.*vf
  diff_geom+=z['weight']*nb*fb-q['weight']*na*fa
  moment+=q['weight']*na*(q['mu']*q['filtered']+4624440.*q['V'])
 dv=B['V']-A['V'];dz=B['z']-A['z'];dq=B['shear']-A['shear'];dR=B['R']-A['R']
 pred=np.linalg.solve(D,dq-Mmu@dz-diff_geom-dR)
 pred_no_residual=np.linalg.solve(D,dq-Mmu@dz-diff_geom)
 # Attribute filtered input differences using the same absolute pressure datum.
 pieces={name:np.linalg.solve(Hb,B[key]-A[key]) for name,key in [('pressure','p'),('deviatoric','dev'),('background','bg')]}
 pieces['mass']=np.linalg.solve(Hb,-(Mb-Ma)@A['z'])
 # Apply K to variations, avoiding catastrophic subtraction of its constant null mode.
 zdev=A['z']-5e7
 pieces['stiffness']=np.linalg.solve(Hb,-400*(Kb-Ka)@zdev)
 i=int(np.argmax(abs(dv)));idx=[i,0,18]
 shearA=np.linalg.solve(Ma,A['shear']);shearB=np.linalg.solve(Mb,B['shear'])
 rawA=np.linalg.solve(Ma,A['raw_normal']);rawB=np.linalg.solve(Mb,B['raw_normal'])
 for j in idx:
  tables.append(dict(pair=suffix,node=j,x=A['x'][j],y=A['y'][j],V_forward=A['V'][j],V_reverse=B['V'][j],delta_V=dv[j],predicted_delta_V=pred[j],Theta_forward=A['Theta'][j],delta_Theta=B['Theta'][j]-A['Theta'][j],delta_projected_shear=shearB[j]-shearA[j],delta_raw_projected_normal=rawB[j]-rawA[j],filter_forward=A['z'][j],delta_filter=dz[j],**{f'delta_filter_{k}':v[j] for k,v in pieces.items()}))
 kdiff=[]
 for key,q in qa.items():
  z=qb[key]
  if q['extended']!=z['extended']:
   kdiff.append(dict(cell=key[0],qp=key[1],x=q['x'],y=q['y'],forward_extended=q['extended'],reverse_extended=z['extended'],forward_along=q['along'],reverse_along=z['along'],forward_length=q['length'],reverse_length=z['length'],weight=q['weight']))
 # Independently reassemble K with exactly the production classification.
 Kq=[]
 for qs,rev in [(qa,False),(qb,True)]:
  m=np.zeros((37,37))
  for q in qs.values():
   j=int(q['segment']);d=np.zeros(37);d[j]=-1/q['length'];d[j+1]=1/q['length']
   if rev:d=d[::-1]
   if not q['extended']:m+=q['weight']*np.outer(d,d)
  Kq.append(m)
  groups={name:np.zeros((37,37)) for name in ['endpoint_branch','interior_segment','roundoff']}
 segment_ties=[]
 for key,q in qa.items():
  z=qb[key];contributions=[]
  for value,rev in [(q,False),(z,True)]:
   j=int(value['segment']);grad=np.zeros(37);grad[j]=-1/value['length'];grad[j+1]=1/value['length']
   if rev:grad=grad[::-1]
   contributions.append(np.zeros((37,37)) if value['extended'] else value['weight']*np.outer(grad,grad))
  kind='endpoint_branch' if q['extended']!=z['extended'] else 'interior_segment' if int(q['segment'])!=35-int(z['segment']) else 'roundoff'
  groups[kind]+=contributions[1]-contributions[0]
  if kind=='interior_segment':segment_ties.append(dict(cell=key[0],qp=key[1],x=q['x'],y=q['y'],forward_segment=q['segment'],forward_xi=q['xi'],reverse_segment=z['segment'],reverse_xi=z['xi'],weight=q['weight']))
 group_effects={name:np.linalg.solve(Hb,-400*matrix@zdev) for name,matrix in groups.items()}
 group_summary={name:dict(max_delta_K=float(np.max(abs(matrix))),delta_filter_at_max=float(group_effects[name][i]),predicted_velocity_at_max=float(np.linalg.solve(D,-Mmu@group_effects[name])[i])) for name,matrix in groups.items()}
 summary[suffix]=dict(stiffness_groups=group_summary,interior_segment_ties=segment_ties,prediction_without_residual_correction_at_max=float(pred_no_residual[i]),observations=[ao,bo],maximum_node=i,max_delta_V=float(abs(dv[i])),qualification_bound=float(5e-10*max(np.max(abs(A['V'])),np.max(abs(B['V'])))+1e-22),predicted_delta_V_at_max=float(pred[i]),signed_delta_V_at_max=float(dv[i]),prediction_max_absolute_error=float(np.max(abs(pred-dv))),discrete_friction_reassembly_error=float(np.max(abs(moment-A['friction']-A['damping']))),filter_decomposition_error=float(np.max(abs(sum(pieces.values())-dz))),K_difference_max=float(np.max(abs(Kb-Ka))),filter_stiffness_fraction_at_max=float(pieces['stiffness'][i]/dz[i]),delta_V_components_at_max={key:float(np.linalg.solve(D,-Mmu@v)[i]) for key,v in pieces.items()},delta_V_shear_contribution_at_max=float(np.linalg.solve(D,dq)[i]),K_reassembly_errors=[float(np.max(abs(Kq[0]-Ka))),float(np.max(abs(Kq[1]-Kb)))],classification_differences=kdiff,common_rate_mu_difference=max(abs(qa[k]['mu_common']-qb[k]['mu_common']) for k in qa),common_rate_derivative_difference=max(abs(qa[k]['dmu_common']-qb[k]['dmu_common']) for k in qa),minimum_filtered_input=min(q['filtered'] for q in list(qa.values())+list(qb.values())),fresh_linear=[read(p)[0] for c in [fc,rc] for p in sorted(out(c).glob('linear_*.csv'))])
 # Common initial iterate and physically ordered blocks.
 x0,ignore=nodes(fc,0);y0,ignore=nodes(rc,0)
 init={}
 for key in ['V','Theta','R','shear','p','dev','bg','z']:
  init[key]=float(np.max(abs(y0[key][::-1]-x0[key])))
 for key in ['M','K','Mmu']:
  f=mat(x0,key+'_diag',key+'_right');g=mapped_mat(y0,key+'_diag',key+'_right',True)
  init[key]=dict(max_abs=float(np.max(abs(g-f))),relative=float(np.max(abs(g-f))/np.max(abs(f))))
 for file in ['constraints.txt','dofs.csv','iterate_0.csv']:
  init[file+'_identical']=(out(fc)/file).read_bytes()==(out(rc)/file).read_bytes()
 for file in ['rhs_0.csv']:
  f=np.loadtxt(out(fc)/file,delimiter=',',skiprows=1);g=np.loadtxt(out(rc)/file,delimiter=',',skiprows=1)
  init[file]=dict(max_abs=float(np.max(abs(g[:,1]-f[:,1]))),relative=float(np.max(abs(g[:,1]-f[:,1]))/np.max(abs(f[:,1]))))
 # Row/column indices coincide after the separately checked physical bulk map.
 for file in ['A.csv','B.csv','KV.csv','G_probes.csv']:
  f=read(out(fc)/file);g=read(out(rc)/file)
  def keyed(rows,rev):
   v={}
   for x in rows:
    key=[]
    for k,val in x.items():
     if k=='value':continue
     n=int(val)
     if rev and ((file=='B.csv' and k=='col') or (file=='KV.csv') or (file=='G_probes.csv' and k=='row')):n=36-n
     key.append(n)
    v[tuple(key)]=float(x['value'])
   return v
  ff,gg=keyed(f,False),keyed(g,True);keys=ff.keys()|gg.keys();error=max(abs(ff.get(k,0)-gg.get(k,0)) for k in keys);scale=max(abs(v) for v in ff.values())
  init[file]=dict(max_abs=error,relative=error/scale,nonzero_counts=[len(ff),len(gg)])
  if file=='A.csv':
   init[file]['blocks']={str((br,bc)):dict(max_abs=max((abs(ff.get(k,0)-gg.get(k,0)) for k in keys if k[:2]==(br,bc)),default=0.)) for br in [0,1] for bc in [0,1]}
  if file=='G_probes.csv':
   init[file]['probes']={str(probe):dict(max_abs=max(abs(ff.get(k,0)-gg.get(k,0)) for k in keys if k[0]==probe),scale=max(abs(ff[k]) for k in ff if k[0]==probe)) for probe in range(9)}
 summary[suffix]['common_initial']=init
 # Verify observer leaves the accepted reference fields unchanged.
 if suffix=='production':
  refs=['dip-45-serial','dip-45-reverse-serial']
  summary[suffix]['observer_profiles_identical']=[(out(c)/'profiles/fault_0.csv').read_bytes()==(r/f'output-endpoint-{ref}/profiles/fault_0.csv').read_bytes() for c,ref in zip([fc,rc],refs)]
(dest/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
with (dest/'matched_locations.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
print(json.dumps({k:{x:v[x] for x in ['maximum_node','max_delta_V','predicted_delta_V_at_max','prediction_max_absolute_error','K_difference_max','common_initial']} for k,v in summary.items()},indent=2))
