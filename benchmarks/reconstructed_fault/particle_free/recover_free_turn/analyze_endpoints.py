#!/usr/bin/env python3
"""Common-time native endpoint reductions for recovered positive/reflected runs."""
import argparse,json,sys,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-recover-free-turn-matplotlib')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'first_free_turn'))
from analyze import reduce,pair,polynomial,native,gather,read,RectBernstein,contour_segments,intervals,OUT as ORIGINAL,RUNS
import numpy as np

def exact_front(field):
 """Refine the sampled global extremum in its native Q2 cell; preserve DG traces."""
 from scipy.optimize import minimize_scalar
 samples={}
 for n in (32,64,128):
  seg,_=contour_segments(field,n);vertices=np.concatenate(seg);front=vertices[np.argmax(vertices[:,0])]
  samples[str(n)]=front.tolist()
 ix,iy=int(front[0]/field.hx),int(front[1]/field.hy);coeff=field.coefficients[iy,ix]
 def right_root(v):
  b=np.array([(1-v)**2,2*v*(1-v),v*v])@coeff
  roots=np.roots([b[0]-2*b[1]+b[2],2*(b[1]-b[0]),b[0]-.05])
  valid=[r.real for r in roots if abs(r.imag)<1e-12 and 0<=r.real<=1]
  return max(valid) if valid else -1.
 v=front[1]/field.hy-iy
 opt=minimize_scalar(lambda v:-right_root(v),bounds=(max(0,v-2/128),min(1,v+2/128)),method='bounded',options={'xatol':1e-12})
 point=[float((ix+right_root(opt.x))*field.hx),float((iy+opt.x)*field.hy)]
 assert abs(float(field.evaluate(*point))-.05)<1e-12
 b=np.array([(1-opt.x)**2,2*opt.x*(1-opt.x),opt.x**2])
 pieces=intervals(np.einsum('j,xji->xi',b,field.coefficients[iy]),field.hx,.05,strict=True)
 connected=[p for p in pieces if p[0]<5.5 and p[1]>=point[0]-1e-11]
 return dict(sampled_fronts=samples,point=point,extension_over_hx=(point[0]-6)/field.hx,
   lateral_over_hy=(point[1]-2)/field.hy,horizontal_superlevel_intervals=pieces,
   horizontal_seed_connection=bool(connected),
   convention='Local Q2 extremum; horizontal connection requires both independent DG traces above .05. No physical mesh refinement.')

def figure(positive,reflected,step,out):
 """Plot only equal-time accepted fields; step numbers change after dt halving."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 pa,pb=(RUNS/label/'output' for label in (positive,reflected))
 seedpath=RUNS/'first-turn-positive-prefix/output'
 seed=gather(seedpath,'tip-seed-fracture-r',['cell','i'])
 initial=RectBernstein(native(seedpath,1),.0625,.03125,seed.B)
 time=read(pa/'bulk-accepted.csv')[-1].time
 fields=[(initial,'black','Initial seed'),(polynomial(pa,step,.0625,.03125),'#c44e52','Positive load'),
         (polynomial(pb,step,.0625,.03125),'#4c72b0','Reflected load')]
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax in axes:
  for field,color,label in fields:
   seg,_=contour_segments(field,32)
   ax.add_collection(LineCollection(seg,colors=color,linewidths=1.2,label=label))
  ax.axvline(6,color='.35',ls='--',lw=.8)
  ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
 axes[0].set(xlim=(1.6,6.5),ylim=(1.45,2.55),title=r'Absolute $\phi=0.05$ contour')
 axes[0].legend(loc='upper center',ncol=2,fontsize=8)
 axes[1].set(xlim=(5.75,6.3),ylim=(1.78,2.22),title=f'Accepted t={time:g} s; local mesh')
 # Explicit cell edges avoid confusing plotting ticks with the actual mesh.
 for x in np.arange(5.75,6.31,.0625):axes[1].axvline(x,color='.87',lw=.5,zorder=-1)
 for y in np.arange(1.75,2.26,.03125):axes[1].axhline(y,color='.87',lw=.5,zorder=-1)
 for ext in ('png','pdf'):fig.savefig(out/f'free-turn-contours-{step:04d}.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for path,color,label in [(pa,'#c44e52','Positive'),(pb,'#4c72b0','Reflected')]:
  s=read(path/f'tip-sections-{step:04d}.csv');p=read(path/f'tip-planes-{step:04d}.csv');formed=p.formed>0
  axes[0,0].plot(p.x[formed],np.degrees(p.angle[formed]),'.',ms=2,color=color,label=label+' formed')
  # Break prospective curves across formed strips; do not bridge the two tails.
  axes[0,0].plot(p.x,np.where(formed,np.nan,np.degrees(p.angle)),':',lw=.7,color=color)
  axes[0,1].plot(p.x,s.age,color=color,label=label)
  axes[1,0].plot(p.x,s.slip,color=color,label=label)
  axes[1,1].plot(p.x,s.V,color=color,label=label)
 for ax,title in zip(axes.flat,['Structural plane (degrees); dots formed','Age (s)','Signed cumulative slip (m)','Stage slip rate (m/s)']):
  ax.axvline(6,color='.6',ls='--',lw=.7);ax.set(xlabel='Material x (m)',title=title,xlim=(5.5,6.5));ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8)
 for ext in ('png','pdf'):fig.savefig(out/f'free-turn-histories-{step:04d}.{ext}',dpi=180)
 plt.close(fig)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('positive');p.add_argument('--reflected');p.add_argument('--step',type=int,required=True)
p.add_argument('--reflected-seed',default='recover-free-turn-reflected-prefix')
a=p.parse_args();out=ORIGINAL.parent/'recover_free_turn_step16'
positive=reduce(a.positive,.0625,.03125,'first-turn-positive-prefix')
result={'positive':positive}
assert positive['steps'][-1]['step']==a.step
result['positive_exact_front']=exact_front(polynomial(RUNS/a.positive/'output',a.step,.0625,.03125))
if a.reflected:
 result['reflected']=reduce(a.reflected,.0625,.03125,a.reflected_seed)
 assert abs(result['reflected']['steps'][-1]['time']-positive['steps'][-1]['time'])<1e-12
 result['reflection']=pair(a.positive,a.reflected,a.step,.0625,.03125)
 x=polynomial(RUNS/a.positive/'output',a.step,.0625,.03125)
 y=polynomial(RUNS/a.reflected/'output',a.step,.0625,.03125)
 difference=x.coefficients-y.coefficients[::-1,:,::-1,:]
 z,w=np.polynomial.legendre.leggauss(6);u=(z+1)/2
 basis=np.array([(1-u)**2,2*u*(1-u),u*u]).T
 error=np.einsum('ai,yxij,bj->yxab',basis,difference,basis)
 norm=np.einsum('ai,yxij,bj->yxab',basis,x.coefficients,basis)
 squared=lambda f:float(np.sum(f*f*w[None,None,:,None]*w[None,None,None,:])*.0625*.03125/4)
 result['reflection']['phase_absolute_L2']=float(np.sqrt(squared(error)))
 result['reflection']['phase_relative_L2']=float(np.sqrt(squared(error)/squared(norm)))
 seedpath=RUNS/'first-turn-positive-prefix/output'
 seed=gather(seedpath,'tip-seed-fracture-r',['cell','i'])
 initial=RectBernstein(native(seedpath,1),.0625,.03125,seed.B)
 growth=np.einsum('ai,yxij,bj->yxab',basis,x.coefficients-initial.coefficients,basis)
 result['reflection']['phase_error_relative_to_growth_L2']=float(np.sqrt(squared(error)/squared(growth))) if squared(growth) else None
 result['reflected_exact_front']=exact_front(y)
 figure(a.positive,a.reflected,a.step,out)
(out/('endpoints-'+str(a.step)+'.json')).write_text(json.dumps(result,indent=2)+'\n')
for name,info in result.items():
 if name=='reflection' or name.endswith('_exact_front'):print(name,json.dumps(info))
 else:print(name,info['steps'][-1]['time'],info['work'])
