"""Independent reduction of supplied server data; never launches ASPECT."""
from pathlib import Path
import hashlib,json,math,re
import numpy as np
DATA=Path(__file__).resolve().parent.parent/'server_3610007';OUT=Path(__file__).resolve().parent

def read(name):return np.genfromtxt(DATA/name,delimiter=',',names=True)
def prm(path):
 stack=[];values={}
 for raw in path.read_text().splitlines():
  line=raw.split('#',1)[0].strip()
  if line.startswith('subsection '):stack.append(line[11:].strip())
  elif line=='end':stack.pop()
  elif line.startswith('set '):
   key,val=line[4:].split('=',1);values['/'.join(stack+[key.strip()])]=val.strip()
 assert not stack;return values

a=read('accepted_steps.csv');c=read('rank-0-coupled-checks.csv');p=read('coupled-profile-815.csv');w=read('rank-0-surface-weak-815.csv');params=prm(DATA/'parameters.prm')
assert len(a)==816 and a['step'][-1]==815 and np.array_equal(a['step'],c['step'])
assert np.array_equal(p['node'],w['node']) and len(p)==5776
assert all(np.all(np.isfinite(x[f])) for x in (a,c,p,w) for f in x.dtype.names)
assert np.all(c['publications']==1) and np.all(a['fresh_linear_checks_passed']==1)
assert np.all(c['error_fraction_of_allowance']<=1)

def mass(x):return w['mass_diagonal']*x+np.r_[0.,w['mass_edge'][:-1]*x[:-1]]+np.r_[w['mass_edge'][:-1]*x[1:],0.]
m1=mass(np.ones(len(p)));assert m1.min()>0
checks={}
for col,key in [('shear_Pa','shear'),('raw_normal_Pa','raw_normal'),('filtered_normal_Pa','filtered_normal'),('pressure_Pa','pressure'),('deviatoric_Pa','deviatoric'),('background_Pa','background')]:
 checks[col]=dict(max_weak_reprojection_difference_Pa_m=float(max(abs(mass(p[col])-w[key]))))
checks['normal_load_split_max_Pa_m']=float(max(abs(w['raw_normal']-w['pressure']-w['deviatoric']-w['background'])))
damping=float(params['Material model/Phase field fault/Radiation damping coefficients'])
checks['residual_assembly_max_Pa_m']=float(max(abs(w['shear']-w['friction']-damping*mass(p['V_m_s'])-w['residual'])))
# Scalar independent native-geometry reconstruction, using the same documented
# segment-by-segment resampling arithmetic. No assumption of missing QP stress.
anchors=[(28867.513459481292,0.),(20000.,15358.983848622454),(9000.,34411.542731880108),(7500.,37009.618943233421),(0.,50000.)];points=[]
for left,right in zip(anchors,anchors[1:]):
 d=(right[0]-left[0],right[1]-left[1]);L=math.sqrt(d[0]*d[0]+d[1]*d[1]);ns=math.ceil(L/10.);h=L/ns
 for i in range(ns+1):
  if points and i==0:continue
  s=L if i==ns else i*h;u=s/L;points.append((left[0]+u*d[0],left[1]+u*d[1]))
points=np.array(points);assert len(points)==len(p)
delta=np.diff(points,axis=0);t=delta/np.sqrt(np.sum(delta*delta,axis=1))[:,None];n=np.stack([-t[:,1],t[:,0]],axis=1)
S=-.5*(t[:,:,None]*n[:,None,:]+n[:,:,None]*t[:,None,:]);N=n[:,:,None]*n[:,None,:]
dS=S[0]-S;dN=N[0]-N;norm=lambda x:np.sqrt(np.sum(x*x,axis=(1,2)))
idx=int(np.argmax(norm(dN)))
# A physical, constant, trace-free 3D tensor counterexample at a native segment.
# Its zz is zero and xy contributes twice in the Frobenius contraction.
tau=2*15e6*S[idx]
old=float(np.sum(tau*N[0]));matched=float(np.sum(tau*N[idx]));assert abs(old-matched)>1e-5
bg0=float(params['Material model/Phase field fault/Reference friction coefficients'])
Dc=float(params['Material model/Phase field fault/Characteristic slip distance']);vr=float(params['Material model/Phase field fault/Reference slip rate']);b=float(params['Material model/Phase field fault/Evolution effect parameters']);amax=.025;vp=1e-9;sig=50e6
background_shear=sig*amax*math.asinh(vp/(2*vr)*math.exp((bg0+b*math.log(vr/vp))/amax))+damping*vp
angle=t[:,0]*t[0,1]-t[:,1]*t[0,0];nodeangle=np.r_[angle[0],.5*(angle[:-1]+angle[1:]),angle[-1]]
surrogate=2*nodeangle*(p['shear_Pa']-background_shear);k=int(np.argmax(abs(surrogate)))
G=float(params['Material model/Phase field fault/Elastic shear moduli']);Gc=float(params['Material model/Phase field fault/Critical energy release rates']);coh=float(params['Material model/Phase field fault/Cohesions']);ell=float(params['Phase field model/Length scale']);Hc=coh*coh/(2*G);m=Gc/((8/3)*ell*Hc);h_peak=m*.6*(1+.6)/(1-.6)**2
active=a[a['max_V']>=1e-3]
result=dict(strict_reference_match_passed=False,input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.iterdir() if f.is_file()},
 checkpoint=dict(slot=(DATA/'last_good_checkpoint.txt').read_text().strip(),accepted_label=(DATA/'bp3_accepted_state.txt').read_text().strip()),
 last_accepted={k:float(a[k][-1]) for k in a.dtype.names},last_checks={k:float(c[k][-1]) for k in c.dtype.names},
 trajectory=dict(all_finite=True,all_publications_one=True,all_fresh_linear_checks_passed=True,max_publication_fraction=float(max(c['error_fraction_of_allowance'])),
  max_normalized_nonlinear_residual=float(max(a['normalized_nonlinear_residual'])),min_theta=float(min(c['min_theta_s'])),
  min_normal=float(min(a['min_sigma_n'])),first_event_threshold_step=int(active['step'][0]) if len(active) else None),
 profile_ranges={f:dict(min=float(min(p[f])),max=float(max(p[f]))) for f in p.dtype.names if f not in ('node','time_s')},weak_internal_checks=checks,
 frame_audit=dict(nodes=len(p),max_slip_tensor_difference=float(max(norm(dS))),max_normal_tensor_difference=float(max(norm(dN))),worst_segment=idx,
  manufactured_tensor_Pa=tau.tolist(),manufactured_global_normal_Pa=old,manufactured_matching_normal_Pa=matched,manufactured_guard_violation_Pa=abs(old-matched),
  surrogate_only=True,surrogate_max_Pa=float(abs(surrogate[k])),surrogate_node=int(p['node'][k]),surrogate_s_m=float(p['s_m'][k]),
  caveat='Q1/equal-adjacent-weight surrogate, not actual failing QP/element stress integrals; does not prove which node or component failed at step 816.'),
 cancellation_scale=dict(Z_max_Pa_m=float(max(p['Z_Pa_m'])),continuum_peak_h=h_peak,peak_correction_norm_scale_Pa=float(math.sqrt(.5)*2*h_peak/min(p['I_h_m'])*max(p['Z_Pa_m'])),
  caveat='Scale estimate from peak phase and nodal Z/I, not measured cell coefficients or a derived floating-point allowance.'),
 failing_step_available=False,underlying_failure_confirmed=False)
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
replay=OUT/'replay-3610007.prm'
if replay.exists():
 replay_params=prm(replay)
 expected={'Resume computation':'true','Output directory':'./work_replay_3610007/',
  'Checkpointing/Steps between checkpoint':'1','Postprocess/BP3/Last accepted step':'816',
  'Postprocess/BP3/Graceful wall seconds':'7200'}
 changed={k:dict(original=params.get(k),replay=v) for k,v in replay_params.items() if params.get(k)!=v}
 assert set(replay_params)==set(params) and set(changed)==set(expected)
 assert all(replay_params[k]==v for k,v in expected.items())
 (OUT/'replay-parameter-delta.json').write_text(json.dumps(changed,indent=2)+'\n')
print(json.dumps(dict(checks=checks,frame=result['frame_audit'],cancellation=result['cancellation_scale'],trajectory=result['trajectory']),indent=2))
