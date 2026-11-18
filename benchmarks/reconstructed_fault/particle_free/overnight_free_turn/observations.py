"""Native DG observations for the overnight comparisons; no numerical evolution."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-overnight-free-turn-mpl')
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'first_free_turn'))
from analyze import native,read,gather,RectBernstein,contour_segments,intervals,pulse,RUNS,masks
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn'

def field(label,step,hy=.03125):return RectBernstein(native(RUNS/label/'output',step),.0625,hy)
def seed(label,step,hy=.03125):
 p=RUNS/label/'output';return RectBernstein(native(p,step),.0625,hy,gather(p,'tip-seed-fracture-r',['cell','i']).B)

def horizontal(poly,y):
 iy=min(poly.ny-1,max(0,int(y/poly.hy)));v=y/poly.hy-iy
 by=np.array([(1-v)**2,2*v*(1-v),v*v])
 return intervals(np.einsum('j,xji->xi',by,poly.coefficients[iy]),poly.hx,.05,strict=True)

def sections_of_contour(poly,initial,stations=None):
 """Exact vertical roots. Keep only pieces with an exact seed-connected horizontal cut.

 This is a sufficient connection test. An unverified/multiple intersection is
 reported rather than selecting a convenient lobe. All stations are x-cell
 interiors; separate DG face traces must pass to join native pieces in y or x.
 """
 if stations is None:
  # Fixed physical cuts on both grids. Extra interior cuts resolve observations
  # of a subcell contour advance; they do not make that advance mesh resolved.
  stations=np.r_[6.001,6.002,6.0078125,6.015625,np.arange(6.03125,6.8,.0625)]
 records=[]
 for x in stations:
  ix=int(x/poly.hx);u=x/poly.hx-ix;bx=np.array([(1-u)**2,2*u*(1-u),u*u])
  pieces=intervals(np.einsum('yji,i->yj',poly.coefficients[:,ix],bx),poly.hy,.05,strict=True)
  connected=[]
  for lo,hi in pieces:
   y=(lo+hi)/2
   if float(initial.evaluate(5.375,y))<=.05:continue
   if any(left<5.375 and right>x for left,right in horizontal(poly,y)):connected.append([lo,hi])
  r=dict(x=float(x),vertical_intervals=pieces,verified_seed_connected_intervals=connected)
  if len(connected)==1:
   lo,hi=connected[0];r.update(lower=lo,upper=hi,center=(lo+hi)/2,width=hi-lo,lateral=(lo+hi)/2-2)
  records.append(r)
 return records

def front(poly,initial):
 """Refine native-Q2 candidates on both sides of a sampled DG face, without averaging."""
 from scipy.optimize import minimize_scalar
 seg,bounds=contour_segments(poly,64);vertices=np.concatenate(seg);p=vertices[np.argmax(vertices[:,0])]
 def owners(value,h,n):
  i=int(value/h);out={min(n-1,max(0,i))}
  if abs(value/h-round(value/h))<1e-9:out.add(min(n-1,max(0,i-1)))
  return sorted(out)
 candidates=[]
 for iy in owners(p[1],poly.hy,poly.ny):
  for ix in owners(p[0],poly.hx,poly.nx):
   coeff=poly.coefficients[iy,ix]
   def root(v):
    b=np.array([(1-v)**2,2*v*(1-v),v*v])@coeff
    roots=np.roots([b[0]-2*b[1]+b[2],2*(b[1]-b[0]),b[0]-.05])
    valid=[r.real for r in roots if abs(r.imag)<1e-12 and 0<=r.real<=1]
    return max(valid) if valid else -1.
   v0=float(np.clip(p[1]/poly.hy-iy,0,1));lo=max(0,v0-2/64);hi=min(1,v0+2/64)
   opt=minimize_scalar(lambda v:-root(v),bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
   for v in [lo,hi,v0,float(opt.x)]:
    u=root(v)
    if u<0:continue
    point=[float((ix+u)*poly.hx),float((iy+v)*poly.hy)]
    bx=np.array([(1-u)**2,2*u*(1-u),u*u]);by=np.array([(1-v)**2,2*v*(1-v),v*v])
    assert abs(float(by@coeff@bx)-.05)<1e-11
    pieces=intervals(np.einsum('j,xji->xi',by,poly.coefficients[iy]),poly.hx,.05,strict=True)
    connected=bool(float(initial.evaluate(5.375,point[1]))>.05 and any(left<5.375 and right>=point[0]-1e-10 for left,right in pieces))
    candidates.append(dict(point=point,verified_seed_connection=connected,cell=[ix,iy],local_coordinates=[u,v]))
 assert candidates
 connected=[c for c in candidates if c['verified_seed_connection']]
 selected=max(connected or candidates,key=lambda c:c['point'][0]);point=selected['point']
 selected.update(dx=point[0]-6,dy=point[1]-2,dx_cells=(point[0]-6)/poly.hx,dy_cells=(point[1]-2)/poly.hy,bounds=bounds,
    sampled_global_front=p.tolist(),convention='Selected native cell trace; both adjacent traces are examined at a DG face, never averaged. Horizontal connection stays on that trace.')
 return selected

def l2(a,b,parent=None):
 """Integrate on common cell subdivisions, preserving each mesh's DG polynomial."""
 hx=min(a.hx,b.hx);hy=min(a.hy,b.hy)
 z,w=np.polynomial.legendre.leggauss(6)
 x=(np.arange(round(8/hx))[:,None]+(z+1)/2)*hx
 y=(np.arange(round(4/hy))[:,None]+(z+1)/2)*hy
 # Array (normal cell,normal q,tangential cell,tangential q).
 va=a.evaluate(x[None,None,:,:],y[:,:,None,None]);vb=b.evaluate(x[None,None,:,:],y[:,:,None,None])
 norm=lambda v:float(np.sum(v*v*w[None,:,None,None]*w[None,None,None,:])*hx*hy/4)
 numerator,denominator=norm(va-vb),norm(va)
 result=dict(absolute=float(np.sqrt(numerator)),relative_field=float(np.sqrt(numerator/denominator)) if denominator else None)
 if parent is not None:
  vp=parent.evaluate(x[None,None,:,:],y[:,:,None,None]);n=norm(va-vp)
  result['relative_increment']=float(np.sqrt(norm(va-vb)/n)) if n else None
 return result

def endpoint(label,step,hy,initial):
 p=RUNS/label/'output';accepted=read(p/'bulk-accepted.csv');a=accepted[accepted.step==step][0]
 rows=native(p,step);phi=RectBernstein(rows,.0625,hy)
 # Selected AT1 fixture: Kf=Gc/((8/3)*ell)=2 Pa. The exported
 # derivative already contains Kf*mass; one more copy restores the linear energy.
 E=.5*rows.B*(rows.fracture_gradient+2*rows.mass)
 assert abs(sum(E)-a.fracture)<2e-8
 # Seed has the same native coordinates and operator on this grid.
 seedrows=gather(RUNS/initial['label']/'output','tip-seed-fracture-r',['cell','i'])
 E0=.5*seedrows.B*(seedrows.fracture_gradient+2*rows.mass)
 recorded_seed_energy=float(read(RUNS/initial['label']/'output/tip-initial.csv')[0].fracture)
 assert abs(sum(E0)-recorded_seed_energy)<2e-8
 d=phi.coefficients-initial['field'].coefficients
 s=read(p/f'tip-sections-{step:04d}.csv')
 plane=read(p/f'tip-planes-{step:04d}.csv') if (p/f'tip-planes-{step:04d}.csv').exists() else None
 pos=float(sum(s.ds*s.positive_w));neg=float(sum(s.ds*s.negative_w))
 absolute=s.positive_w-s.negative_w
 ratio=np.divide(abs(s.V),absolute,out=np.ones_like(absolute),where=absolute>0)
 cancellation=1-ratio
 positive_power=s.ds*s.force*s.positive_w;negative_power=s.ds*s.force*s.negative_w
 r=dict(label=label,step=step,time=float(a.time),dt=float(a.dt),stage_time=float(a.time-a.dt/2),hy=hy,front=front(phi,initial['field']),centerlines=sections_of_contour(phi,initial['field']),width=pulse(phi),
  seed_fracture_energy=float(sum(E0)),added_fracture=float(sum(E-E0)),regional_fracture={name:float(sum((E-E0)[mask])) for name,mask in masks(rows.x,rows.y).items()},
  maximum_phase_increment=float(np.max(d)),minimum_phase_increment=float(np.min(d)),
  physical_work_error=float(a.Wext-a.energy_change-a.Db-a.Dc-a.Df),phase_KKT=float(a.phase_KKT),mechanical=float(a.mechanical_residual),
  integrated_slip=float(sum(s.ds*s.slip)),local_rate_positive=pos,local_rate_negative=neg,
  absolute_integrated_rate=pos-neg,net_integrated_rate=pos+neg,
  cancellation=dict(mixed_sections=int(sum((s.positive_w>0)&(s.negative_w<0))),maximum=float(max(cancellation)),
   positive_local_power=float(sum(np.maximum(positive_power,0)+np.maximum(negative_power,0))),
   negative_local_power=float(sum(np.minimum(positive_power,0)+np.minimum(negative_power,0))),
   minimum_section_power=float(min(s.ds*s.force*s.V)),
   largest=[dict(section=int(k),x=float(s.s[k]),cancellation=float(cancellation[k]),V=float(s.V[k]),absolute_rate=float(absolute[k]),slip=float(s.slip[k])) for k in np.argsort(cancellation)[-5:][::-1]]),
  original_or_staged_planes=None if plane is None else dict(formed=int(sum(plane.formed)),
   maximum_formed_angle=float(max(abs(plane.angle[plane.formed>0]))) if any(plane.formed>0) else None,
   maximum_prospective_angle=float(max(abs(plane.angle[plane.formed<=0]))) if any(plane.formed<=0) else None))
 return r,phi

def section_difference(labela,stepa,labelb,stepb,previousa,previousb,window):
 pa,pb=RUNS/labela/'output',RUNS/labelb/'output'
 a=read(pa/f'tip-sections-{stepa:04d}.csv');b=read(pb/f'tip-sections-{stepb:04d}.csv')
 # Endpoint ages/slips align. Terminal stage rates generally do not.
 p=read(RUNS/previousa[0]/'output'/f'tip-sections-{previousa[1]:04d}.csv')
 q=read(RUNS/previousb[0]/'output'/f'tip-sections-{previousb[1]:04d}.csv')
 result={n:dict(maximum=float(max(abs(a[n]-b[n]))),weighted_L2=float(np.sqrt(sum(a.ds*(a[n]-b[n])**2)))) for n in ['age','slip']}
 ma=(a.slip-p.slip)/window;mb=(b.slip-q.slip)/window
 result['matched_interval_mean_rate']=dict(start_offset=-window,maximum=float(max(abs(ma-mb))),weighted_L2=float(np.sqrt(sum(a.ds*(ma-mb)**2))),definition='Accepted signed-slip increment divided by the same physical window. Terminal stage rates are not compared.')
 return result

def work(labels):
 rows=[]
 for label in labels:
  for r in read(RUNS/label/'output/bulk-accepted.csv'):
   rows.append(dict(time=float(r.time),dt=float(r.dt),error=float(r.Wext-r.energy_change-r.Db-r.Dc-r.Df)))
 errors=np.array([r['error'] for r in rows])
 return dict(signed=float(sum(errors)),absolute=float(sum(abs(errors))),steps=rows)
