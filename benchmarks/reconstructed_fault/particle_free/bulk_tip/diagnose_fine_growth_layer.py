#!/usr/bin/env python3
"""One retained-state diagnostic of the central mechanical layer, not a new run."""
import os,json
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from analyze_material_overnight import fast_frame
from analyze_time_update import read
from growth_spatial_analysis import CellPolynomials
import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];RUNS=ROOT/'build-particle-free/bulk-tip/runs';OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation'
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained');records={}
for index,(name,old_label,new_label,h) in enumerate([('coarse','material-midpoint-M3','material-midpoint-M3',.125),('fine','material-midpoint-S1','material-obstacle-window02',.0625)]):
 old=fast_frame(RUNS/old_label/'output',73);p=fast_frame(RUNS/new_label/'output',74);assert np.array_equal(old.cell,p.cell) and np.array_equal(old.q,p.q)
 cfg=json.loads((RUNS/new_label/'output/parameters.json').read_text());material=cfg['Material_20model']['Phase_20field_20fault'] if 'Material_20model' in cfg else cfg['Material model']['Phase field fault']
 def value(key):return float(material[key]['value'])
 G=value('Elastic shear moduli');eta=value('Reference viscosities');gc=value('Critical energy release rates');ell=float(cfg['Phase field model']['Length scale']['value'])
 K=gc/((8/3)*ell);m=2*G*K/value('Cohesions')**2
 dt=.000625
 assert max(abs(p.stage_phi-.5*(p.phi+old.phi)))<1e-12
 assert max(abs(p.stage_Axy-.5*(p.Axy+old.Axy)))<1e-12
 assert max(abs(p.stage_tauxy-.5*(p.tauxy+old.tauxy)))<1e-9
 # Add the two matched-Maxwell equations. This is total shear strain rate,
 # distinct from fault deformation w, which alone enters section slip.
 Dxy=((p.tauxy-old.tauxy)/(2*G)+p.Axy-old.Axy)/dt+p.stage_tauxy/(2*eta)+G/eta*p.stage_Axy
 fields=append_fields(p,'Dxy',Dxy,usemask=False).view(np.recarray)
 phi=CellPolynomials(fields,h,['phi','Dxy'],2);w=CellPolynomials(fields,h,['w'],5)
 central=abs(p.x-4)<.5
 c=m*p.stage_phi*(1+p.stage_phi);d=(1-p.stage_phi)**2;ratio=d/(c+d)
 records[name]={'G':G,'eta':eta,'ell':ell,'m':m,'phase_Dxy_Q2_reconstruction_error':phi.reconstruction_error,
                'minimum_fixed_phase_shear_stiffness_ratio':float(min(ratio)),
                'central_Dxy_max':float(max(abs(Dxy[central]))),'central_fault_rate_max':float(max(abs(p.w[central]))),
                'central_phase_max':float(max(p.phi[central])),
                'central_fault_to_total_integrated_shear':float(sum(p.weight[central]*p.w[central])/(2*sum(p.weight[central]*Dxy[central])))}
 trace_x=np.array([4-1e-12,4-1e-12,4+1e-12,4+1e-12]);trace_y=np.array([2-1e-12,2+1e-12,2-1e-12,2+1e-12])
 records[name]['common_center_phase_Dxy_traces']=phi.evaluate(trace_x,trace_y).tolist()
 initial=CellPolynomials(fast_frame(RUNS/old_label/'output',0),h,['phi'],2)
 records[name]['initial_center_phase_traces']=initial.evaluate(trace_x,trace_y)[:,0].tolist()
 # Width of the high-compliance core at fixed phase and section resistance.
 # Use the retained midpoint phase, not an assumed analytic seed profile.
 stage_phase=CellPolynomials(fields,h,['stage_phi'],2)
 def stiffness(x,y):
  v=stage_phase.evaluate(np.array([x]),np.array([y]))[0,0]
  return (1-v)**2/(m*v*(1+v)+(1-v)**2)
 widths=[];center_ratios=[]
 for side_x in (-1,1):
  x=4+side_x*1e-12;offsets=[]
  for side_y in (-1,1):
   center=stiffness(x,2+side_y*1e-12);lo=1e-12;hi=.0625
   assert stiffness(x,2+side_y*hi)>2*center
   for _ in range(45):
    mid=(lo+hi)/2
    if stiffness(x,2+side_y*mid)>2*center:hi=mid
    else:lo=mid
   offsets.append((lo+hi)/2);center_ratios.append(center)
  widths.append(sum(offsets))
 records[name]['center_fixed_force_stiffness_ratios']=center_ratios
 records[name]['half_compliance_full_width_m']=widths
 records[name]['cell_width_over_compliance_width']=[h/w for w in widths]
 y=np.concatenate([(cell+np.r_[np.linspace(.000001,.999999,25),np.nan])*h for cell in range(round(4/h))]);valid=np.isfinite(y)
 for side in (-1,1):
  x=np.full(sum(valid),4+side*1e-12);v=phi.evaluate(x,y[valid]);rate=w.evaluate(x,y[valid])[:,0]
  for ax,data in zip(axes,[v[:,0],v[:,1],rate]):
   full=np.full(y.size,np.nan);full[valid]=data;ax.plot(y,full,'-' if side==-1 else '--',color=f'C{index}',label=name if side==-1 else None)
for ax,title,unit in zip(axes,['Endpoint phase','Total shear strain rate Dxy','Fault deformation rate w'],['phase','1/s','1/s']):ax.set(xlabel='y (m), x=4 left/right traces',ylabel=unit,title=title,xlim=(1.7,2.3));ax.grid(alpha=.2);ax.legend()
fig.suptitle('Accepted interval 73→74: midpoint kinematics; diagnostic observation, no new simulation')
for ext in ('png','pdf'):fig.savefig(OUT/f'central-layer-parent74.{ext}',dpi=170)
records['meaning']='Dxy recovered from matched-Maxwell stress+A equations at midpoint. Phase/Dxy use Q2 cell-interior polynomials; w uses a Q5 observation interpolant. Stiffness ratio holds phase and section resistance fixed, not a condensed Hessian eigenvalue.'
(OUT/'central-layer-parent74.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
