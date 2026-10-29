#!/usr/bin/env python3
"""Archived physical stress/pressure diagnostics and independent point equations."""
from pathlib import Path
import csv,json,hashlib,shutil,time
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/coulomb-direction'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction'
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def rows(path):return list(csv.DictReader(path.open()))
def write(name,data):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle():
 mp.mp.dps=70;started=time.monotonic();maximum=0.;results=[];selected={}
 for c in rows(WORK/'points-release/cases.csv'):
  key=(c['mu'],c['active']);selected.setdefault(key,[])
  if len(selected[key])>=2:continue
  selected[key].append(c)
 for group in selected.values():
  for c in group:
   n=lambda k:mp.mpf(c[k]);mu=n('mu');h=lambda a:4*a*(1+a)/(1-a)**2
   beta=1/(1+n('dt')/100);H=h(n('phi'));hn=h(n('old_phi'))
   t=mp.matrix([beta*((n('old_Txx')-n('old_Tyy'))/2+2e6*n('Exx')),beta*(n('old_Txy')+2e6*n('Exy'))])
   ref=beta*hn/H*mp.matrix([n('old_Qxx'),n('old_Qxy')]);pb=n('pressure')-beta*(n('old_Txx')+n('old_Tyy'))/2
   active=[i for i in range(2) if int(c['active'])&(1<<i)];m=len(active)
   if not m:q=ref
   else:
    initial=[n('Qxx')+3,n('Qxy')-2]+[n(['gamma_plus','gamma_minus'][i])*mp.mpf('1.02') for i in active]+[n(['theta_plus','theta_minus'][i])+mp.mpf('.002') for i in active]
    def residual(*x):
     q=mp.matrix(x[:2]);e=t-(1+H)*q;d=t-H*q;flow=mp.matrix([0,0]);f=[];orientation=[]
     for j,i in enumerate(active):
      k=1 if i==0 else -1;theta=x[2+m+j];v=mp.matrix([mp.cos(2*theta),mp.sin(2*theta)]);jv=mp.matrix([-v[1],v[0]])
      flow+=x[2+j]*k*jv
      f.append((k*mp.fdot(e,jv)-mu*(pb+mp.fdot(d,v)))/1000)
      orientation.append((-k*mp.fdot(e,v)-mu*mp.fdot(d,jv))/1000)
     return tuple([(q[0]-ref[0]-flow[0])/1000,(q[1]-ref[1]-flow[1])/1000]+f+orientation)
    solution=mp.findroot(residual,tuple(initial),tol=mp.mpf('1e-55'),maxsteps=80);q=mp.matrix(solution[:2])
   error=float(mp.sqrt((q[0]-n('Qxx'))**2+(q[1]-n('Qxy'))**2)/max(mp.mpf(1000),mp.sqrt(mp.fdot(q,q))));maximum=max(maximum,error)
   assert error<=1e-12,(c['case'],error)
   results.append({'case':int(c['case']),'mu':float(mu),'active':int(c['active']),'Q_relative_scaled_error':error})
 save('independent-reference.json',{'precision_digits':70,'cases':results,'maximum_scaled_Q_defect':maximum,'seconds':time.monotonic()-started,'method':'Independent trigonometric plane/flow/yield/stationarity equations; perturbed starts; no quadratic candidate formula.'})

def archived():
 records=json.loads((WORK/'archives.json').read_text());allcells=[];glyphs=[];datasets={};restore=[]
 mapping={'radial-133':('radial-first_intact',133),'radial-176':('radial-recovered-target',176),'radial-177':('radial-recovered-target',177),'fixed-176':('fixed-to170000',176)}
 for record in records:
  name=record['name'];run=WORK/'archives'/name
  log=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('coulomb-archive-'+name+'.log')
  text=log.read_text();assert 'no physical advancement' in text and '*** Timestep' not in text
  assert not (run/'bulk-accepted.csv').exists()
  pointer=(run/'restart/last_good_checkpoint.txt').read_text().strip();cp=run/'restart'/pointer.zfill(2)
  source=ROOT/record['checkpoint'];p0=(source/'restart/last_good_checkpoint.txt').read_text().strip();source=source/'restart'/p0.zfill(2)
  for filename,digest in record['metadata']['files'].items():assert sha(cp/filename)==sha(source/filename)==digest
  pressure={r['cell']:r for p in run.glob('pressure-rank-*.csv') for r in rows(p)}
  case,step=mapping[name];old=ROOT/'build-particle-free/p5c-campaign/runs'/case
  if name=='radial-133':old=ROOT/'build-particle-free/p5c-campaign/runs/radial-to155'
  data=[]
  for r in rows(old/f'means-{step:04}.csv'):
   if r['cell'] not in pressure:continue
   p=pressure[r['cell']];a={k:float(r[k]) for k in ['x','y','tauxx','tauyy','tauxy','phi','Wxx','Wxy','abs_rate','Y']};a['p_mech_Pa']=float(p['p_mech_mean_Pa']);a['p_mech_min_Pa']=float(p['p_mech_min_Pa']);a['p_mech_max_Pa']=float(p['p_mech_max_Pa'])
   a['cell']=r['cell'];a['state']=name;a['time_s']=record['metadata']['time_s'];a['p_N_reference_Pa']=10000.
   d=.5*(a['tauxx']-a['tauyy']);b=a['tauxy'];mean=.5*(a['tauxx']+a['tauyy']);radius=np.hypot(d,b);pb=a['p_mech_Pa']-mean
   a['compression_axis_rad']=.5*np.arctan2(-2*b,-2*d);a['C_major_Pa']=pb+radius;a['C_minor_Pa']=pb-radius
   a['minimum_added_confinement_for_all_closed_Pa']=max(0.,radius-pb)
   a['flow_defined']=np.hypot(a['Wxx'],a['Wxy'])>1e-12;a['flow_plane_label_rad']=.5*np.arctan2(-a['Wxx'],a['Wxy']) if a['flow_defined'] else None
   for mu in [.1,.6]:
    for k in [1,-1]:
     v=np.array([k*b-mu*d,-k*d-mu*b]);v/=np.linalg.norm(v);theta=.5*np.arctan2(v[1],v[0]);pn=pb+d*v[0]+b*v[1]
     a[f'mu{mu}_plane{k}_rad']=float(theta);a[f'mu{mu}_normal{k}_Pa']=float(pn)
   data.append(a)
  assert len(data)==len(pressure)
  for x,y in [(1.01,.5),(1.08,.46),(1.08,.54),(1.16,.5)]:glyphs.append(min(data,key=lambda a:(a['x']-x)**2+(a['y']-y)**2))
  datasets[name]=data;allcells+=data
  restore.append({'state':name,'accepted_time_s':record['metadata']['time_s'],'publications':record['metadata']['publications'],'no_physical_advance':True,'checkpoint_hashes_unchanged':True,'cells':len(data),'p_mech_mean_range_Pa':[min(a['p_mech_Pa'] for a in data),max(a['p_mech_Pa'] for a in data)],'all_plane_closed_cells':int(sum(a['C_minor_Pa']>0 for a in data)),'candidate_mu0.1_positive_normal_cells':int(sum(a['mu0.1_normal1_Pa']>0 for a in data))})
 write('archived-cells.csv',allcells);write('archived-directions.csv',glyphs);save('restore-checks.json',restore)
 fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
 scale=max(abs(a['tauxy']) for name in ['radial-133','radial-177'] for a in datasets[name])
 for ax,name in zip(axes[0],['radial-133','radial-177']):
  data=datasets[name];x=np.unique([a['x'] for a in data]);y=np.unique([a['y'] for a in data]);v=np.zeros((len(y),len(x)))
  for a in data:v[np.searchsorted(y,a['y']),np.searchsorted(x,a['x'])]=a['tauxy']
  im=ax.pcolormesh(x,y,v,cmap='RdBu_r',vmin=-scale,vmax=scale,shading='nearest');fig.colorbar(im,ax=ax,shrink=.8,label='physical deviatoric τxy [Pa]')
  step=133 if name.endswith('133') else 177;case='radial-to155' if step==133 else 'radial-recovered-target';phase=np.loadtxt(ROOT/'build-particle-free/p5c-campaign/runs'/case/f'phase-{step:04}.csv',delimiter=',',skiprows=1);xx=np.unique(phase[:,1]);yy=np.unique(phase[:,2]);phi=np.zeros((len(yy),len(xx)));phi[np.searchsorted(yy,phase[:,2]),np.searchsorted(xx,phase[:,1])]=phase[:,3]
  ax.contour(xx,yy,phi,levels=[.01,.05,.1,.2,.4,.6],colors='white',linewidths=.7)
  for a in [g for g in glyphs if g['state']==name]:
   def line(theta,color,lw,ls='-'):
    half=.014;dx=half*np.cos(theta);dy=half*np.sin(theta);ax.plot([a['x']-dx,a['x']+dx],[a['y']-dy,a['y']+dy],color=color,lw=lw,ls=ls)
   line(a['compression_axis_rad'],'black',1.6)
   for k in [1,-1]:line(a[f'mu0.1_plane{k}_rad'],'darkorange',1.5,'-' if a[f'mu0.1_normal{k}_Pa']>0 else '--')
   if a['flow_defined']:
    for theta in [a['flow_plane_label_rad'],a['flow_plane_label_rad']+np.pi/2]:line(theta,'cyan',1.)
  ax.plot(1,.5,'w+',ms=8);ax.axvline(1.09375,color='.6',ls=':',lw=.8);ax.set_aspect('equal');ax.set_xlim(.94,1.24);ax.set_ylim(.35,.65);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(f'{name}: accepted t={data[0]["time_s"]:.6f} s\nCell-mean stress, pressure and net flow; white=phase contours',fontsize=10)
 ax=axes[1,0];a=min(datasets['radial-177'],key=lambda a:(a['x']-1.09)**2+(a['y']-.5)**2);theta=np.linspace(-np.pi/2,np.pi/2,721);d=.5*(a['tauxx']-a['tauyy']);b=a['tauxy'];pb=a['p_mech_Pa']-.5*(a['tauxx']+a['tauyy']);pn=pb+d*np.cos(2*theta)+b*np.sin(2*theta);shear=abs(b*np.cos(2*theta)-d*np.sin(2*theta));angles=theta*180/np.pi
 ax.plot(angles,shear,'k',label='|physical shear traction|');ax.plot(angles,np.where(pn>0,.1*pn,np.nan),color='darkorange',label='μ pn from solved pressure (closed planes only)');ax.axhline(1000,color='purple',ls=':',label='Current constant R=μ p_N=1000 Pa');ax.plot(angles,.1*(pn+10000),'--',color='green',label='Hypothetical +10000 Pa mechanical confinement');ax.fill_between(angles,0,shear.max()*1.1,where=pn<=0,color='.8',alpha=.4,label='pn≤0: outside closed-contact option');ax.set_xlabel('plane tangent angle to x [degrees]');ax.set_ylabel('Traction / resistance [Pa]');ax.set_title(f'Traction-only diagnostic at ({a["x"]:.4f},{a["y"]:.4f}) m\nSolved p={a["p_mech_Pa"]:.1f} Pa; prescribed p_N=10000 Pa',fontsize=10);ax.legend(fontsize=7);ax.grid(alpha=.2)
 write('traction-angle.csv',[{'plane_rad':float(th),'physical_abs_shear_Pa':float(sh),'physical_normal_pressure_Pa':float(p),'mu_pn_closed_Pa':float(.1*p) if p>0 else None,'constant_R_control_Pa':1000.,'hypothetical_plus10k_resistance_Pa':float(.1*(p+10000))} for th,sh,p in zip(theta,shear,pn)])
 ax=axes[1,1];tests=rows(WORK/'points-release/angles.csv');mu=np.linspace(0,.65,200);ax.plot(mu,45-.5*np.degrees(np.arctan(mu)),'k',label='Analytic 45°−atan(μ)/2');ax.axhline(45,color='cyan',ls=':',label='Constant-resistance shear-plane pair')
 for row in tests:
  angle=np.degrees(np.arccos(np.clip(abs(np.cos(float(row['plane_rad'])-float(row['compression_axis_rad']))),0,1)));ax.plot(float(row['mu']),angle,'o',color='darkorange',ms=4)
 ax.set_xlabel('friction coefficient μ');ax.set_ylabel('plane angle to major compression axis [degrees]');ax.set_title('Controlled point tests: physical p=10000 Pa, Q=0 endpoint\nBoth conjugates, rotated bases and reversed shear',fontsize=10);ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle('Black: major in-plane compression axis. Cyan pair: equivalent signed shear representations of net W.\nOrange: physical-stress Coulomb candidates at μ=.1 (dashed if pn≤0). None of these lines is a crack path.\nCandidate diagnostics omit cohesive Q; the current material return uses τ−Q.',fontsize=10)
 fig.savefig(OUT/'directions.png',dpi=160);plt.close(fig)
 save('plot-definitions.json',{'fields':'Actual Gauss8 cell-mean physical deviatoric stress and net W from archived exports; mechanical pressure restored and averaged with the same rule. Principal axes belong to the cell-mean Cauchy tensor. No new constitutive state is evaluated from averaged histories.','candidate_planes':'Bare physical-stress Q=0 Mohr-Coulomb diagnostics; actual new return with Q uses the generalized criterion, not this bare-stress angle.','pressure':'p_mech is solved pressure; prescribed p_N=10000 Pa only sets old R. Hypothetical +10000 Pa confinement is not native input and would require different boundary tractions, not a gauge shift.','flow':'Cyan orthogonal plane labels represent the same signed shear tensor W. Undefined when norm(mean W)<=1e-12/s. Not crack/ridge angle or finite-rotation qualification.','phase_contours':[.01,.05,.1,.2,.4,.6],'all_plane_closed':'p_mech-(tau_xx+tau_yy)/2 > hypot((tau_xx-tau_yy)/2,tau_xy)','compression_convention':'sigma=tau-p_mech I is tensile positive; C=-sigma, major in-plane compression axis is largest eigenvalue.'})

if __name__=='__main__':
 oracle();archived()
 for name in ['checks.json','angles.csv','cases.csv','path.csv']:shutil.copy2(WORK/'points-release'/name,OUT/name)
 shutil.copy2(WORK/'radial-control/native-checks.json',OUT/'constant-radial-control.json')
 calibration=[]
 for mu in [.1,.6]:
  R=mu*10000;q=2000.;peak=(q+np.sqrt((1+mu*mu)*R*R-mu*mu*q*q))/(1+mu*mu)
  calibration.append({'mu':mu,'mechanical_p_Pa':10000.,'c_res_Pa':0.,'m_retained':4.,'residual_shear_Pa':R/np.sqrt(1+mu*mu),'intact_peak_shear_Pa':peak,'angle_to_major_compression_deg':45-.5*np.degrees(np.arctan(mu))})
 write('calibration.csv',calibration)
 print(json.dumps({'calibration':calibration,'point_checks':json.loads((OUT/'checks.json').read_text()),'restores':json.loads((OUT/'restore-checks.json').read_text())},indent=2))
