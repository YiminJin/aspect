#!/usr/bin/env python3
"""Cellwise native Bernstein contours and exact axial roots on both y=1 traces."""
import argparse,json,os
import numpy as np
from analyze_time_update import RUNS,OUT,read,balance
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-tip-time-mpl')
p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');a=p.parse_args();d=RUNS/a.label/'output'
rows=read(d/'bulk-accepted.csv');last=int(rows.step[-1]);final=balance(d,f'tip-balance-{last:04d}-r*.csv');seed=balance(d,'tip-seed-fracture-r*.csv');levels=[.001,.01,.1,.5];h=.25
assert np.array_equal(seed.cell,final.cell) and np.array_equal(seed.i,final.i)
def basis(t):return np.array([(1-t)**2,2*t*(1-t),t*t])
def extent(field,side,level):
 xs=[]
 for cell in np.unique(final.cell):
  m=final.cell==cell;c=final[m];cy=c.y[0];target=1.+side*h/2
  if abs(cy-target)>1e-12:continue
  B=field[m][np.argsort(c.i)].reshape(3,3);b=basis(0. if side>0 else 1.)@B
  coeff=[b[0]-2*b[1]+b[2],2*(b[1]-b[0]),b[0]-level];roots=np.roots(np.trim_zeros(coeff,'f')) if np.any(np.array(coeff)!=0) else []
  grid=sorted([0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-9 and 0<=z.real<=1])
  for l,r in zip(grid[:-1],grid[1:]):
   t=.5*(l+r)
   if basis(t)@b>=level:xs += [c.x[0]-h/2+h*l,c.x[0]-h/2+h*r]
 return [min(xs),max(xs)] if xs else None
result=dict(label=a.label,mesh_h=h,trace='Exact quadratic roots on both DG traces y=1; no contour smoothing or averaged interface values',levels={})
for level in levels:
 result['levels'][str(level)]={}
 for side in [-1,1]:
  e0=extent(seed.B,side,level);e1=extent(final.B,side,level)
  result['levels'][str(level)][str(side)]=dict(initial=e0,final=e1,left_advance=e0[0]-e1[0],right_advance=e1[1]-e0[1])
(OUT/(a.label+'-native-contours.json')).write_text(json.dumps(result,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained');steps=[0,int(rows.step[len(rows)//2]),last];colors=['#ffffff','#efb84a','#e86850','#54278f'];t=np.linspace(0,1,25);Bt=basis(t);X,Y=np.meshgrid(t,t)
for ax,step in zip(axes,steps):
 b=seed if step==0 else balance(d,f'tip-balance-{step:04d}-r*.csv')
 for cell in np.unique(final.cell):
  m=final.cell==cell;c=final[m];values=b.B[m][np.argsort(c.i)].reshape(3,3);z=Bt.T@values@Bt;x=c.x[0]+h*(X-.5);y=c.y[0]+h*(Y-.5)
  plot=ax.pcolormesh(x,y,z,shading='nearest',vmin=0,vmax=.8,cmap='viridis',rasterized=True)
  for level,color in zip(levels,colors):
   if z.min()<level<z.max():ax.contour(x,y,z,levels=[level],colors=[color],linewidths=.7)
 for x in np.arange(0,4.01,h):ax.axvline(x,color='gray',lw=.2,alpha=.4)
 for y in np.arange(0,2.01,h):ax.axhline(y,color='gray',lw=.2,alpha=.4)
 ax.set(xlim=(0,4),ylim=(0,2),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f'Step {step}; cellwise DGQ2')
fig.colorbar(plot,ax=list(axes),label='Physical phase',shrink=.7)
axes[0].legend(handles=[Line2D([],[],color=c,label=f'{v:g}') for v,c in zip(levels,colors)],title='Contour',fontsize=7,ncol=2,loc='upper left')
fig.savefig(OUT/(a.label+'-native-contours.png'),dpi=180);fig.savefig(OUT/(a.label+'-native-contours.pdf'));plt.close(fig)
print(json.dumps(result['levels'],indent=2))
