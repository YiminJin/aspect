#!/usr/bin/env python3
"""Independent 80-digit replay of saved prepublication cell operands."""
import argparse,csv,json
from pathlib import Path
import mpmath as m
import numpy as np
m.mp.dps=80
p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d={}
for row in csv.DictReader(a.input.open()):d.setdefault(row['key'],{})[tuple(int(row[k])for k in ['i','j','k'])]=m.mpf(float(row['value']))
def v(key,i=0,j=0,k=0):return d[key][i,j,k]
def mat(key,ni,nj,k=0):return m.matrix([[v(key,i,j,k)for j in range(nj)]for i in range(ni)])
def lag(x):return [2*(x-m.mpf('.5'))*(x-1),4*x*(1-x),2*x*(x-m.mpf('.5'))]
def der(x):return [4*x-3,4-8*x,4*x-1]
nd=len(d['native_component']);nc=len(d['old_T'])//3
coeffs=[(int(v('native_component',i)),v('native_coefficient',i),int(2*v('native_support',i,0)),int(2*v('native_support',i,1)))for i in range(nd)]
hx=v('vertex',1,0)-v('vertex',0,0);hy=v('vertex',2,1)-v('vertex',0,1)
def gradient(r,s):return m.matrix([[sum(c*(der(r)[i]*lag(s)[j]/hx if b==0 else lag(r)[i]*der(s)[j]/hy)for cpt,c,i,j in coeffs if cpt==a)for b in range(2)]for a in range(2)])
def strain(g):return m.matrix([g[0,0]-(g[0,0]+g[1,1])/3,g[1,1]-(g[0,0]+g[1,1])/3,(g[0,1]+g[1,0])/2])
def norm(t):return m.sqrt(t[0]**2+t[1]**2+2*t[2]**2)
r,s=v('unit',0),v('unit',1);g=gradient(r,s);remote=mat('remote_gradient',2,2)
point_shape=m.matrix([v('dg_value',0,i)for i in range(nc)])
B=m.matrix([[v('dg_value',q,i)for i in range(nc)]for q in range(1,10)])
W=m.diag([v('JxW',q)for q in range(1,10)])
# Reassemble the exact real linear algebra from the exported rounded FEValues
# basis/gradient/JxW operands; compare with exact polynomial differentiation.
Gs=[];Es=[];exactEs=[]
for q in range(1,10):
 G=m.matrix([[sum(v('native_coefficient',i)*v('native_gradient',q,i,b)for i in range(nd)if int(v('native_component',i))==a)for b in range(2)]for a in range(2)])
 Gs.append(G);Es.append(strain(G));exactEs.append(strain(gradient(v('quadrature',q,0),v('quadrature',q,1))))
M=B.T*W*B
C=m.matrix(nc,3);Ce=m.matrix(nc,3)
for c in range(3):
 C[:,c]=m.lu_solve(M,B.T*W*m.matrix([e[c]for e in Es]));Ce[:,c]=m.lu_solve(M,B.T*W*m.matrix([e[c]for e in exactEs]))
beta,k=v('beta'),v('kappa');new=mat('new_T',nc,3);old=mat('old_T',nc,3)
projected=(new-beta*old)/(2*k)
sg=(point_shape.T*projected).T;sc=(point_shape.T*C).T;sce=(point_shape.T*Ce).T
native_point=m.matrix([[sum(v('native_coefficient',i)*v('native_gradient',0,i,b)for i in range(nd)if int(v('native_component',i))==a)for b in range(2)]for a in range(2)])
# Same I/Z, phase and physical localization as the saved production operands.
x=v('point',0);ns=len(d['old_I']);j=min(ns-2,int(x*(ns-1)));xi=x*(ns-1)-j
interp=lambda key:(1-xi)*v(key,j)+xi*v(key,j+1)
phase_components={int(v('native_component',i))for i in range(nd)if v('old_anchor',i)!=0}
assert len(phase_components)==1
phase_component=phase_components.pop()
local_phase=sum(v('new_anchor',i)*v('native_value',0,i)for i in range(nd)if int(v('native_component',i))==phase_component)
phi=v('phase');h=v('m')*phi*(1+phi)/(1-phi)**2;corr=interp('new_Z')*h/interp('new_I')
gold=beta*(point_shape.T*old).T+2*k*strain(g)-m.matrix([0,0,corr])
reported=m.matrix([v('dg_stress',c)for c in range(3)]);particle=m.matrix([v('maxwell_stress',c)for c in range(3)])
# Independent evaluation of the derived source allowance. Use the exact
# analytic inverse here; production also bounds its computed inverse defect.
u=m.mpf(2)**-53
gamma=lambda n:n*u/(1-n*u)
velocity_sum=[sum(abs(c)for cpt,c,i,j in coeffs if cpt==a)for a in range(2)]
gs=m.matrix([[4*velocity_sum[a]/[hx,hy][b]for b in range(2)]for a in range(2)])
ds=m.matrix([(2*gs[0,0]+gs[1,1])/3,(2*gs[1,1]+gs[0,0])/3,(gs[0,1]+gs[1,0])/2])
M1=m.matrix([[4,2,-1],[2,16,2],[-1,2,4]])/30
Me=m.matrix(nc,nc)
for i in range(nc):
 for j in range(nc):Me[i,j]=hx*hy*M1[i%3,j%3]*M1[i//3,j//3]
Ai=Me**-1
ve=8*gamma(6)+4*gamma(4);de=3*gamma(6)+2*gamma(4)
gre=(1+gamma(2*nc+10))*(1+ve)*(1+de)-1
se=(1+ve)**2-1;rbe=(1+gre)*(1+gamma(28))-1
wtotal=sum(v('JxW',q)for q in range(1,10))
ws=m.matrix([sum(abs(v('JxW',q)*v('dg_value',q,j))for q in range(1,10))for j in range(nc)])
pe=m.matrix(3,1)
for c in range(3):
 errors=[]
 rhs=B.T*W*m.matrix([e[c]for e in Es])
 for i in range(nc):
  errors.append(sum(abs(Ai[i,j])*(rbe*ws[j]+wtotal*se*(1+rbe))*ds[c]for j in range(nc))+gamma(2*nc)*sum(abs(Ai[i,j]*rhs[j])for j in range(nc)))
 pe[c]=max(errors)
pr=(1+gamma(2*nc+10))*(1+8*gamma(6))*(1+3*gamma(6))-1
ge=m.matrix(2,2)
for a0 in range(2):
 for b0 in range(2):
  ce=sum((2 if d0==b0 else 4)*gamma(4)*(abs(v('point',d0))+abs(v('vertex',0,d0)))/[hx,hy][d0]for d0 in range(2))
  ge[a0,b0]=(pr+2*ce)*gs[a0,b0]
ne=m.matrix([(2*ge[0,0]+ge[1,1])/3,(2*ge[1,1]+ge[0,0])/3,(ge[0,1]+ge[1,0])/2])
lebesgue=sum(abs(t)for t in point_shape)
be=2*k*(ne+lebesgue*pe)
source_allowance=m.sqrt(be[0]**2+be[1]**2+(be[0]+be[1])**2+2*be[2]**2)
converted=lambda x:[float(t)for t in x]
result=dict(precision_digits=m.mp.dps,error=float(v('error')),old_bound=float(v('bound')),ratio=float(v('ratio')),point=converted([r,s]),
 independent_source_allowance_exact_inverse=float(source_allowance),gradient_operand_upper=converted(gs),projection_coefficient_bound=converted(pe),native_point_strain_bound=converted(ne),
 beta_vs_exact=float(beta-m.exp(-v('G')*v('dt')/v('eta'))),kappa_vs_exact=float(k+v('eta')*m.expm1(-v('G')*v('dt')/v('eta'))),
 exact_gradient=converted(g),remote_gradient=converted(remote),local_shape_gradient=converted(native_point),
 remote_minus_exact_gradient=converted(remote-g),local_minus_exact_gradient=converted(native_point-g),
 DG_source_minus_exact_point=converted(sg-strain(g)),
 exported_operand_projection_minus_exact_point=converted(sc-strain(g)),
 exact_polynomial_projection_minus_exact_point=converted(sce-strain(g)),
 DG_coefficients_minus_exported_operand_projection_max=float(max(abs(t)for t in projected-C)),
 DG_coefficients_minus_exact_polynomial_projection_max=float(max(abs(t)for t in projected-Ce)),
 DG_stress=converted(reported),native_stress=converted(particle),exact_polynomial_stress=converted(gold),
 DG_stress_error=converted(reported-gold),native_stress_error=converted(particle-gold),
 physical_correction_Pa=float(corr),same_timestep=True,incoming_zero=all(t==0 for t in old),
 phase_change=float(v('phase')-v('old_phase')),local_minus_remote_phase=float(local_phase-v('phase')),
 max_I_change=float(max(abs(v('old_I',j)-v('new_I',j))for j in range(ns))),
 surface_recurrence_error=float(max(abs(v('new_Z',j)-beta*v('old_Z',j)-k*v('V',j))for j in range(ns))),
 gradient_operand_abs_sum=converted(m.matrix([[sum(abs(v('native_coefficient',i)*v('native_gradient',0,i,b))for i in range(nd)if int(v('native_component',i))==a)for b in range(2)]for a in range(2)])))
checks=dict(original_guard_rejects=v('error')>v('bound'),
 same_physical_points=all(v('point',i)==v('sample_point',i)for i in range(2)),
 same_reference_points=all(v('unit',i)==v('particle_unit',i)for i in range(2)),
 same_phase=v('phase')==v('old_phase'),same_I=all(v('old_I',j)==v('new_I',j)for j in range(ns)),
 actual_Maxwell_recurrence=norm(particle-(beta*m.matrix([v('incoming_stress',c)for c in range(3)])+2*k*m.matrix([v('effective_rate',c)for c in range(3)])))<gamma(8)*max(1,norm(particle)),
 native_point_differentiation_bounded=all(abs((strain(remote)-strain(g))[c])<=ne[c]for c in range(3)),
 projection_coefficients_bounded=all(max(abs((projected-Ce)[i,c])for i in range(nc))<=pe[c]for c in range(3)),
 discrepancy_bounded=v('error')<source_allowance,
 actual_point_differentiation_dominates=abs((particle-gold)[2])>10*abs((reported-gold)[2]))
result['checks']={k:bool(v)for k,v in checks.items()};result['passed']=all(checks.values())
assert result['passed'],result['checks']
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
