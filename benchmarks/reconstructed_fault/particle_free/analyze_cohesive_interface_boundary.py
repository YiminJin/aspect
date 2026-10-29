#!/usr/bin/env python3
"""Fixed-memory mixed-strip response; no physical evolution or pressure lag."""
import sys,json,csv
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).parent/'bulk_cohesive'))
from cohesive_interface import layer_DtN,G
from analyze_cohesive_interface_domain import case,at,read,table
from analyze_cohesive_interface_native import surface_rule
WORK=ROOT/'build-particle-free/cohesive-interface-boundary'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary'
R=np.diag([1.,-1.])

def symbol(k,top,bottom):
 if k==0:return np.array([G/(top+bottom),0.],complex)
 if abs(k)*min(top,bottom)>24:return np.array([G*abs(k),0.],complex)
 up=layer_DtN(abs(k),top);down=R@layer_DtN(abs(k),bottom)@R
 full=np.linalg.inv(np.linalg.inv(up)+np.linalg.inv(down))
 return np.array([full[0,0],full[1,0] if k>0 else full[1,0].conjugate()])

def q1_symbols(length,n,top,bottom,aliases=300):
 result=np.zeros((2,n),complex)
 for j,m in enumerate(np.fft.fftfreq(n)*n):
  for a in range(-aliases,aliases+1):
   mode=m+a*n
   if mode==0:result[:,j]+=symbol(0,top,bottom);continue
   weight=np.sinc(mode/n)**4
   result[:,j]+=weight*symbol(2*np.pi*mode/length,top,bottom)
 result[1,0]=0.
 if n%2==0:result[1,n//2]=0.
 return result

def long_wave_precision():
 mp.mp.dps=50;k=mp.mpf('0.00001');g=mp.mpf(1000000);reflection=mp.matrix([[1,0],[0,-1]])
 def layer(h):
  z=k*h;e=mp.exp(-z);A=mp.matrix([[1,0,e,z*e],[-1,1,e,(z-1)*e],[e,z*e,1,0],[-e,(1-z)*e,1,-1]]);out=mp.matrix(2,2)
  for j in range(2):
   a,b,c,d=mp.lu_solve(A,mp.matrix([1j/k if j==1 else 0,1/k if j==0 else 0,0,0]));psi=a+(c+d*z)*e;pz=-a+b+(c+d*(z-1))*e;pzz=a-2*b+(c+d*(z-2))*e;pzzz=-a+3*b+(c+d*(z-3))*e
   out[0,j]=-g*k*k*(pzz+psi);out[1,j]=-1j*g*k*k*(pzzz-3*pz)
  return out
 rows=[]
 for top,bottom in [('0.6','0.4'),('0.4','0.6'),('0.5','0.5'),('1.1','0.9')]:
  t,b=mp.mpf(top),mp.mpf(bottom);H=t+b;d=t-b;F=(layer(t)**-1+(reflection*layer(b)*reflection)**-1)**-1
  limit=g/H+3*g*d*d/H**3;C=6*g*d/H**3;ke=float(abs(F[0,0]-limit)/limit);ne=float(abs(k*mp.im(F[1,0])+C)/abs(C)) if C else float(abs(F[1,0]))
  rows.append(dict(k_inv_m=float(k),top_m=float(t),bottom_m=float(b),K_relative_to_long_limit=ke,kN_relative_to_limit=ne,precision_digits=50));assert ke<1e-8 and ne<1e-8
 return rows

def main():
 OUT.mkdir(exist_ok=True,parents=True)
 a=case('period-8',8,128,0);b=case('period-16',16,256,4)
 _,aa=read(a['p']/'operators.csv');_,bb=read(b['p']/'operators.csv')
 ids=np.arange(9,120);ib=ids+64;x=ids*.0625;responses=[];checks=[];profiles=[];low=[]
 for top,bottom in [(.6,.4),(.5,.5),(.4,.6),(1.1,.9),(2.1,1.9)]:
  H=top+bottom;d=top-bottom;C=6*G*d/H**3;DK=3*G*d*d/H**3
  for k in [.01,.05,.1,.2,.4,.8,1.6,3.2,6.4,12.8]:
   s=symbol(k,top,bottom);low.append(dict(top_m=top,bottom_m=bottom,k_inv_m=k,wavelength_m=2*np.pi/k,K_Pa_m=float(s[0].real),normal_imag_Pa_m=float(s[1].imag),k_normal_imag_Pa_m2=float(k*s[1].imag),limiting_k_normal_Pa_m2=-C,K_zero_Pa_m=G/H,K_long_nonzero_Pa_m=G/H+DK))
   if top==bottom:assert abs(s[1])<1e-8
   reflected=symbol(k,bottom,top);assert abs(reflected[0]-s[0])<1e-7*max(1,abs(s[0])) and abs(reflected[1]+s[1])<1e-7*max(1,abs(s[1]))
 for t in [2.3,2.45,2.47]:
  m=a['n']['relaxed_jump_m'][at(a,t)];A=float(.0625*sum(m));J=float(.0625*((np.arange(128)*.0625-4)*m).sum())
  dtau=-(bb[:,2].reshape(256,256)[np.ix_(ib,ib)]-aa[:,2].reshape(128,128)[np.ix_(ids,ids)])@m[ids]
  dp=(bb[:,3].reshape(256,256)[np.ix_(ib,ib)]-aa[:,3].reshape(128,128)[np.ix_(ids,ids)])@m[ids]
  predicted_t=-120000*A/16;predicted_p=1200000/16*(A*(x-4)-J)
  responses.append(dict(time_s=t,integrated_memory_m2=A,first_memory_moment_m3=J,predicted_shear_change_Pa=predicted_t,native_shear_RMS_change_Pa=float(np.sqrt(np.mean(dtau**2))),shear_prediction_RMS_error_Pa=float(np.sqrt(np.mean((dtau-predicted_t)**2))),native_normal_RMS_change_Pa=float(np.sqrt(np.mean(dp**2))),normal_prediction_RMS_error_Pa=float(np.sqrt(np.mean((dp-predicted_p)**2)))))
  for j in range(len(x)):profiles.append(dict(time_s=t,x_m=x[j],native_shear_difference_Pa=dtau[j],asymptotic_shear_difference_Pa=predicted_t,native_normal_difference_Pa=dp[j],asymptotic_normal_difference_Pa=predicted_p[j]))
 # Exact linear decomposition at common Gauss points. Constitutive histories
 # remain untouched; fixed-state pressure resistance is an explicit diagnostic.
 Q8,_,_,P8=surface_rule(128);Q16,_,_,P16=surface_rule(256)
 K8=aa[:,2].reshape(128,128);K16=bb[:,2].reshape(256,256);N8=aa[:,3].reshape(128,128);N16=bb[:,3].reshape(256,256)
 decomposition=[];regional=[];closure=0.
 for t in [2.3,2.45,2.47]:
  i,j=at(a,t),at(b,t);m=a['n']['relaxed_jump_m'][i];extended=np.zeros(256);extended[64:192]=m;changed=b['n']['relaxed_jump_m'][j]-extended
  pop=(P16@(N16@extended))[192:576]-P8@(N8@m);pevo=(P16@(N16@changed))[192:576]
  sop=-(P16@(K16@extended))[192:576]+P8@(K8@m);sevo=-(P16@(K16@changed))[192:576]
  actualp=b['s']['pn_Pa'][j,192:576]-a['s']['pn_Pa'][i];actuals=b['s']['shear_Pa'][j,192:576]-a['s']['shear_Pa'][i]
  closure=max(closure,float(np.max(abs(pop+pevo-actualp))),float(np.max(abs(sop+sevo-actuals))))
  V=a['s']['V_m_s'][i];theta=a['s']['Theta'][i];damage=a['s']['damage'][i]
  mu=.005*np.arcsinh(abs(V)/(2*5e-6)*np.exp((.1+.01*np.log(theta))/.005));resistance=damage*np.sign(V)*mu*pop;xx=a['s']['x_m'][i]
  for q in range(384):decomposition.append(dict(time_s=t,x_m=xx[q],weight_m=a['weights'][q],damage_8=damage[q],normal_operator_Pa=pop[q],normal_evolution_Pa=pevo[q],normal_total_Pa=actualp[q],shear_operator_Pa=sop[q],shear_evolution_Pa=sevo[q],shear_total_Pa=actuals[q],fixed_state_friction_pressure_change_Pa=resistance[q],fixed_state_net_force_change_Pa=sop[q]-resistance[q]))
  for label,lo,hi in [('left_flank',.5,2.5),('right_flank',5.5,7.5),('seed',2.5,5.5)]:
   # Same represented Q1 weak-load test on both representations. Full weak
   # equations include tied reactions; this is not a free reduced direction.
   psi=Q8@((a['node_x']>lo)&(a['node_x']<hi)).astype(float);w=a['weights']*psi;rr=dict(time_s=t,region=label,Q1_test_integral_m=float(sum(w)))
   for key,z in [('normal_operator',pop),('normal_evolution',pevo),('normal_total',actualp),('shear_operator',sop),('shear_evolution',sevo),('shear_total',actuals),('fixed_state_net_force',sop-resistance)]:rr[key+'_mean_Pa']=float(w@z/sum(w));rr[key+'_RMS_Pa']=float(np.sqrt(w@(z**2)/sum(w)))
   regional.append(rr)
 table(OUT/'operator-evolution-decomposition.csv',decomposition);table(OUT/'operator-evolution-regions.csv',regional)
 assert closure<1e-7,closure
 models=[('L8-H1-offset',8,128,.6,.4),('L16-H1-offset',16,256,.6,.4),('L32-H1-offset',32,512,.6,.4),('L16-H1-centred',16,256,.5,.5),('L16-H1-reflected',16,256,.4,.6),('L16-H2-offset',16,256,1.1,.9),('L32-H4-offset',32,512,2.1,1.9)]
 maps={};frozen=[]
 for name,L,n,top,bottom in models:
  s=q1_symbols(L,n,top,bottom);maps[name]=s
  coord=(np.arange(n)*L/n)+(4-L/2);mass_symbol=2/3+np.cos(2*np.pi*np.arange(n)/n)/3
  m=np.zeros(n);offset=round((L/2-4)/.0625);m[offset:offset+128]=a['n']['relaxed_jump_m'][at(a,2.47)]
  shear=np.fft.ifft(s[0]*np.fft.fft(m)).real;normal=np.fft.ifft(s[1]/mass_symbol*np.fft.fft(m)).real
  # Same background Maxwell stress on all heights: wall displacement scales H.
  bg=G*a['h']['wall_memory_m'][at(a,2.47)]
  for i in range(n):frozen.append(dict(case=name,x_m=coord[i],tau_Pa=bg-shear[i],normal_Pa=10000+normal[i],memory_m=m[i],height_m=top+bottom,wall_displacement_scale=top+bottom))
  if L in [8,16] and name.endswith('H1-offset'):
   data=aa if L==8 else bb
   for mode in [1,2,4,8,16]:
    v=np.exp(2j*np.pi*mode*np.arange(n)/n)
    for col,index,label in [(2,0,'K'),(3,1,'N')]:
     native=np.vdot(v,data[:,col].reshape(n,n)@v)/n;expected=s[index,mode]
     checks.append(dict(case=name,mode=mode,operator=label,native_real=float(native.real),native_imag=float(native.imag),reference_real=float(expected.real),reference_imag=float(expected.imag),relative_difference=float(abs(native-expected)/max(abs(expected),1.))))
 # Alias tail on the most demanding 8 m representation.
 s=maps['L16-H1-offset'];table(OUT/'analytical-L16-symbols.csv',[dict(mode_index=j,K_real_Pa_m=float(s[0,j].real),N_imag_Pa_m=float(s[1,j].imag)) for j in range(256)])
 fine=q1_symbols(8,128,.6,.4,600);tail=float(np.linalg.norm(fine-maps['L8-H1-offset'])/np.linalg.norm(fine))
 table(OUT/'boundary-low-modes.csv',low);table(OUT/'same-memory-period.csv',responses);table(OUT/'same-memory-period-profiles.csv',profiles);table(OUT/'frozen-boundaries.csv',frozen);table(OUT/'native-analytical-modes.csv',checks)
 zero=[]
 for data,n,L in [(aa,128,8),(bb,256,16)]:
  K=data[:,2].reshape(n,n);N=data[:,3].reshape(n,n);zero.append(dict(period_m=L,K_constant_max_error_Pa_m=float(max(abs(K@np.ones(n)-G))),N_constant_max_error_Pa_m=float(max(abs(N@np.ones(n)))),N_full_period_mean_load_max_Pa_m=float(max(abs(np.sum(N,axis=0)/n)))))
 result=dict(period_prediction=responses,decomposition_closure_Pa=closure,alias_300_to_600_relative=tail,native_zero_modes=zero,MP50_long_limits=long_wave_precision(),native_analytical_checks=checks,pressure_reference='one pinned algebraic DoF followed by global mean removal; physical p0 remains 10000 Pa',long_wave=dict(K_zero='G/H',K_nonzero_limit='G/H+3G(top-bottom)^2/H^3',normal_imag_leading='-6G(top-bottom)/(H^3 k)'),continuation_choice='Retain explicit finite-height L16/H1 periodic specimen; native reflected L8/H1 diagnostic before longer original-offset run')
 (OUT/'boundary-reference.json').write_text(json.dumps(result,indent=2)+'\n')
 assert max(r['normal_prediction_RMS_error_Pa'] for r in responses)<.1
 assert tail<1e-6
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
