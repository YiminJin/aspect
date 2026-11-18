"""Accepted slip-rate history, right-flank evolution, and physical work in the repaired pilot."""
import os
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.ndimage import label
from scipy.optimize import brentq
from phase import H, NX, NY, coefficients, fronts, normal_maximum, read, sample, value

root=Path(__file__).resolve().parents[4]
build=root/'build-particle-free'
original=build/'stage4-propagation-pilot/analysis_views/through70'
evidence=root/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('segments',nargs='+',type=Path,help='Ordered repaired run directories')
parser.add_argument('--through-step',type=int,help='Last completed native snapshot to include')
parser.add_argument('--destination',type=Path,default=evidence/'guard80')
args=parser.parse_args()
dest=args.destination;dest.mkdir(parents=True,exist_ok=True)

# Identify the connected component first, then maximize x in each half independently.
# The unrestricted front may already be in a different column from the lower flank.
def component(b,n,level):
    labels,_=label(sample(b,n)>=level)
    seed=labels[NY*n//2,NX*n//2]
    assert seed
    return labels==seed

def extent(b,mask,upper):
    candidates=np.argwhere(mask & ((np.arange(NY)[:,None]>=NY//2)==upper))
    right=-np.inf
    for iy,ix in candidates[np.argsort(candidates[:,1])[::-1]]:
        if (ix+1)*H<right:break
        cell=b[iy,ix]
        u=np.linspace(0.,1.,65)
        def residual(t):return normal_maximum(cell,t)[0]-.05
        hit=np.flatnonzero([residual(t)>=0 for t in u])
        if len(hit):
            j=hit[-1]
            x=1. if j+1==len(u) else brentq(residual,u[j],u[j+1],xtol=1e-13)
            right=max(right,(ix+x)*H)
    assert np.isfinite(right)
    return right

def flanks(b,n):
    cells=component(b,n,.05).reshape(NY,n,NX,n).any(axis=(1,3))
    return extent(b,cells,True),extent(b,cells,False)

def read_sections(folder,step):
    return read(folder/f'pilot-sections-{step:04d}.csv')

initial=coefficients(original,0)
parent=coefficients(original,68)
zero=abs(initial).max(axis=(2,3))==0.
iy,ix=np.mgrid[:NY,:NX]
xx=(ix+.5)*H;yy=(iy+.5)*H
upper=(xx>=6)&(yy>=2)
remote=(abs(yy-2)>.5)|(xx<1.5)|(xx>6.5)
initial_front=fronts(initial,16)
u0,l0=flanks(initial,16)
planes0=read(original/'tip-planes-0068.csv')
original_history=read(evidence/'through70-history.csv')
prior_table=read(original/'bulk-accepted.csv')
initial_fracture=float(prior_table[prior_table['step']==68]['fracture'][0]-original_history[original_history['step']==68]['added_fracture'][0])
series=[(original,68)]
for run in args.segments:
    out=run/'output'
    if (out/'pilot-summary.csv').exists():
        series.extend((out,int(row['step'])) for row in read(out/'pilot-summary.csv') if row['step']>68 and (args.through_step is None or row['step']<=args.through_step))
assert [s for _,s in series]==list(range(68,series[-1][1]+1))
history=[]; fresh_cells=[];previous=parent
for folder,step in series:
    b=coefficients(folder,step)
    summary=read(folder/'pilot-summary.csv')
    r=summary[summary['step']==step][0]
    e=read(folder/'bulk-accepted.csv');e=e[e['step']==step][0]
    assert e['publications']==step and e['phase_KKT']<=1e-7 and e['mechanical_residual']<=1e-9
    f=fronts(b,16);u,l=flanks(b,16);u8,l8=flanks(b,8)
    assert abs(max(u,l)-f['right_x'])<1e-10
    diff=b-initial; delta=b-previous
    assert delta.min()>=-1e-14
    means=diff.mean(axis=(2,3))
    active=zero & (means>1e-12)
    for y,x in np.argwhere(active):
        fresh_cells.append(dict(step=step,x=xx[y,x],y=yy[y,x],mean=means[y,x],max_coefficient=diff[y,x].max()))
    section=read_sections(folder,step);x=section['x'];rate=section['rate_stage'];weighted=section['ds']*np.maximum(rate,0.)
    planes=read(folder/f'tip-planes-{step:04d}.csv')
    assert np.array_equal(planes['x'],planes0['x'])
    old=planes0['formed']>0
    assert np.all(planes['formed'][old]) and abs(planes['angle'][old]-planes0['angle'][old]).max()==0.
    newly=(planes['formed']>0)&~old
    obs=dict(step=step,time=r['time'],stage_time=r['stage_time'],upper_right_m=u,lower_right_m=l,
             upper_advance_mm=1000*(u-u0),lower_advance_mm=1000*(l-l0),right_x=f['right_x'],
             left_advance_mm=1000*(initial_front['left_x']-f['left_x']),
             max_sampling_difference_m=max(abs(u-u8),abs(l-l8)),tip_y=f['right_y'],
             components=f['sampled_components'],detached_area=f['detached_sample_area'],
             fresh_cells=int(active.sum()),fresh_upper_right_cells=int((active&upper).sum()),
             fresh_integral_m2=means[zero].sum()*H**2,fresh_upper_right_integral_m2=means[zero&upper].sum()*H**2,
             fresh_peak_coefficient=diff[zero].max(),remote_max_abs=float(abs(diff[remote]).max()),
             new_sections=int(newly.sum()),first_new_section_x=x[newly].min() if newly.any() else None,
             peak_rate_m_s=abs(rate).max(),peak_x=x[np.argmax(abs(rate))],
             positive_rate_centroid_m=weighted@x/weighted.sum(),
             integrated_slip_m2=section['slip_endpoint']@section['ds'],
             tip_rate_m_s=np.interp(f['right_x'],x,rate),
             stage_slip_over_Dc=abs(rate).max()*r['dt']/3.2211126126299625e-7,
             added_fracture_J_m=e['fracture']-initial_fracture,phase_kkt=e['phase_KKT'],
             work_error_J_m=e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df'])
    for threshold in (1,5,10):
        above=x[rate>=threshold/1000]
        obs[f'rightmost_{threshold}mm_s_m']=above.max() if len(above) else None
    history.append(obs);previous=b

def save_csv(name,rows):
    with (dest/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
save_csv('window-history.csv',history)
if fresh_cells:save_csv('initially-intact-cells.csv',fresh_cells)

# Use the accepted prehistory once, ending exactly at the common parent.
frames=[(original,i) for i in range(1,68)]+series
rates=[];times=[]
for folder,step in frames:
    s=read_sections(folder,step);assert np.array_equal(s['x'],x)
    summary=read(folder/'pilot-summary.csv')
    stamp=summary[summary['step']==step]
    assert len(stamp)==1 and abs(stamp['dt'][0]-8e-5)<1e-14
    rates.append(s['rate_stage']);times.append(1e3*stamp['stage_time'][0])
rates=np.asarray(rates)*1e3
t=np.array([r['time'] for r in history])*1e3
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
im=ax[0,0].pcolormesh(times,x,rates.T,rasterized=True,shading='nearest',cmap='RdBu_r',norm=SymLogNorm(linthresh=.3,vmin=-max(abs(rates).max(),1),vmax=max(abs(rates).max(),1)))
ax[0,0].axvline(5.44,color='k',ls=':',lw=1)
ax[0,0].axhline(initial_front['right_x'],color='k',ls='--',lw=.7)
ax[0,0].set(ylabel='Material station x (m)',xlabel='Midpoint time (ms)',title='Signed section slip rate')
fig.colorbar(im,ax=ax[0,0],label='mm/s')
for k in ['upper','lower']:
    ax[0,1].plot(t,[r[k+'_advance_mm'] for r in history],'-o',ms=3,label=k.capitalize()+' flank')
ax[0,1].set(title='Connected right φ = 0.05 contour',ylabel='Advance from prepared seed (mm)',xlabel='Endpoint time (ms)');ax[0,1].legend()
stations=(3.01,4.5,5.5,5.9,6.)
for pos in stations:
    ax[1,0].plot(times,[np.interp(pos,x,v) for v in rates],label=f'{pos:g} m')
ax[1,0].axvline(5.44,color='k',ls=':',lw=1)
ax[1,0].set(title='Slip at fixed material stations',xlabel='Midpoint time (ms)',ylabel='Signed rate (mm/s)');ax[1,0].legend(fontsize=8)
gx,gy=np.meshgrid(np.linspace(5.89,6.22,385),np.linspace(1.94,2.23,317))
at68,final=[value(a,gx,gy) for a in (parent,b)]
im=ax[1,1].pcolormesh(gx,gy,final-at68,rasterized=True,cmap='magma',shading='auto',vmin=0)
fig.colorbar(im,ax=ax[1,1],label='Phase increment since 5.44 ms')
for vals,color,style in [(at68,'cyan','dashed'),(final,'white','solid')]:
    ax[1,1].contour(gx,gy,vals,levels=[.05],colors=color,linestyles=style,linewidths=1.4)
from matplotlib.patches import Rectangle
for y,i in np.argwhere(active & upper):
    ax[1,1].add_patch(Rectangle((i*H,y*H),H,H,fill=False,edgecolor='lime',lw=1.0))
ax[1,1].set(title='Right flank and initially intact cells (green)',xlabel='x (m)',ylabel='y (m)',aspect='equal')
from matplotlib.lines import Line2D
ax[1,1].legend(handles=[Line2D([],[],color=c,ls=s,label=l) for c,s,l in [('cyan','--','5.44 ms'),('white','-',f'{t[-1]:.2f} ms')]],facecolor='.22',labelcolor='white',fontsize=8)
for ext in ('png','pdf'):fig.savefig(dest/('slip-and-upper-flank.'+ext),dpi=180)
rows=history[1:]
report=dict(first=history[0],last=history[-1],steps=len(rows),
            signed_work_68_to_last=sum(r['work_error_J_m'] for r in rows),
            absolute_work_68_to_last=sum(abs(r['work_error_J_m']) for r in rows),
            fresh_cell_rule='Initially all nine coefficients exactly zero. Count cells with mean increment >1e-12. Integrals include all initially zero cells without threshold.',
            max_sampling_difference_m=max(r['max_sampling_difference_m'] for r in history),
            final_support={})
for level in (.0001,.001,.01,.05):
    c=component(b,8,level)
    intact_points=np.repeat(np.repeat(zero,8,0),8,1)
    above=sample(b,8)>=level
    report['final_support'][str(level)]=dict(connected_sample_points_in_initially_intact_cells=int((c&intact_points).sum()),detached_sample_points=int((above&~c).sum()))
(dest/'window-summary.json').write_text(json.dumps(report,indent=2)+'\n')
