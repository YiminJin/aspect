#!/usr/bin/env python3
"""Accepted positive-load result and failed solve; never plot an unaccepted field.

Run analyze.py for the three positive windows, then summarize.py first.
Writes figures, their compact numerical tables, and a log-only failure diagnosis.
No simulation, checkpoint modification, or reflected-field substitution.
"""
import csv
import json
import re
from analyze import (OUT, RUNS, native, read, gather, RectBernstein,
                     polynomial, connectivity, contour_segments, intervals)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PREFIX='first-turn-positive-prefix'
FINAL='first-turn-positive-fourcore-end'
HX,HY=.0625,.03125
data=json.loads((OUT/'accepted-history-summary.json').read_text())['positive']
step=data['steps'][-1]['step'];time=data['last_accepted_time']
assert step==15 and time==.075 and not data['target_reached']
path=RUNS/FINAL/'output';seedpath=RUNS/PREFIX/'output'
first=native(seedpath,1);seed=gather(seedpath,'tip-seed-fracture-r',['cell','i'])
initial=RectBernstein(first,HX,HY,seed.B)
field=polynomial(path,step,HX,HY)
straight=polynomial(RUNS/'resolution60-coarse-growth/output',step,HX,HY)
result=dict(last_accepted_step=step,last_accepted_time=time,reflected_case='not run',
            connectivity=connectivity(field),contour_observation={})
for n in (16,32,64,128):
    segments,bounds=contour_segments(field,n)
    points=np.concatenate(segments);front=points[np.argmax(points[:,0])]
    result['contour_observation'][str(n)]=dict(bounds=bounds,rightmost=front.tolist(),
        extension_over_hx=float((front[0]-6)/HX),lateral_over_hy=float((front[1]-2)/HY))
# Refine the sampled extremum directly in the same cell's Q2 polynomial.
from scipy.optimize import minimize_scalar
ix,iy=int(front[0]/HX),int(front[1]/HY)
coeff=field.coefficients[iy,ix]
def right_root(v):
    b=np.array([(1-v)**2,2*v*(1-v),v*v])@coeff
    roots=np.roots([b[0]-2*b[1]+b[2],2*(b[1]-b[0]),b[0]-.05])
    valid=[r.real for r in roots if abs(r.imag)<1e-12 and 0<=r.real<=1]
    return max(valid) if valid else -1.
v=front[1]/HY-iy
optimum=minimize_scalar(lambda v:-right_root(v),bounds=(max(0,v-2/128),min(1,v+2/128)),
                        method='bounded',options={'xatol':1e-12})
native_front=[float((ix+right_root(optimum.x))*HX),float((iy+optimum.x)*HY)]
assert abs(float(field.evaluate(*native_front))-.05)<1e-12
result['native_root_front']=dict(point=native_front,extension_over_hx=(native_front[0]-6)/HX,
    lateral_over_hy=(native_front[1]-2)/HY,
    observation_note='Local extremum refinement of the sampled global front; no physical mesh refinement.')
v=optimum.x
line=np.einsum('j,xji->xi',np.array([(1-v)**2,2*v*(1-v),v*v]),field.coefficients[iy])
pieces=intervals(line,HX,.05,strict=True)
connected=[p for p in pieces if p[0]<5.5 and p[1]>6]
assert len(connected)==1 and abs(connected[0][1]-native_front[0])<1e-11
result['exact_native_line_connection']=dict(y=native_front[1],superlevel_intervals=pieces,
    connected_seed_to_ahead=connected[0],
    convention='Exact Q2 roots; both separate DG face traces must exceed .05 to join adjacent pieces.')
sections=read(path/f'tip-sections-{step:04d}.csv')
planes=read(path/f'tip-planes-{step:04d}.csv')
initial_planes=read(seedpath/'tip-planes-0001.csv')
old=initial_planes.formed>0;new=(planes.formed>0)&~old
assert max(abs(planes.angle[old]))==0
result['sections']={}
for name,mask in [('original',old),('formed_after_seed',new),('still_virgin',planes.formed==0)]:
    result['sections'][name]=dict(count=int(sum(mask)),
        angle_degrees_range=[float(min(planes.angle[mask])*180/np.pi),float(max(planes.angle[mask])*180/np.pi)],
        age_range=[float(min(sections.age[mask])),float(max(sections.age[mask]))],
        slip_range=[float(min(sections.slip[mask])),float(max(sections.slip[mask]))])
positive=sections.ds*sections.force*sections.positive_w
negative=sections.ds*sections.force*sections.negative_w
result['local_work']=dict(positive_power=float(sum(np.maximum(positive,0)+np.maximum(negative,0))),
    negative_power=float(sum(np.minimum(positive,0)+np.minimum(negative,0))),
    section_power=float(sum(sections.ds*sections.force*sections.V)),
    signed_rate_positive=float(sum(sections.ds*sections.positive_w)),
    signed_rate_negative=float(sum(sections.ds*sections.negative_w)))
result['local_work']['power_identity_error']=result['local_work']['positive_power']+result['local_work']['negative_power']-result['local_work']['section_power']
assert abs(result['local_work']['power_identity_error'])<1e-10
result['baseline_contour']=dict(zip(('segments','bounds'),contour_segments(straight,64)))['bounds']

# Q2 centerline traces remain separate across the DG face y=2.
x=np.concatenate([(ix+np.linspace(1e-11,1-1e-11,33))*HX for ix in range(84,105)])
profiles=[]
for side in (-1,1):
    y=2+side*1e-12
    phi0=initial.evaluate(x,y);phi=field.evaluate(x,y)
    profiles.extend(zip([side]*len(x),x,phi0,phi,phi-phi0))
with (OUT/'endpoint-profiles-0015.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['y_trace_sign','x','phi0','phi','delta_phi']);writer.writerows(profiles)

def save(fig,name):
    for suffix in ('png','pdf'):fig.savefig(OUT/(name+'.'+suffix),dpi=190)
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax in axes[0]:
    for poly,color,label,style in [(initial,'black','Initial seed','-'),
            (straight,'.5','Original straight load, same time','--'),
            (field,'#b2182b','Off-axis load, accepted .075 s','-')]:
        seg,_=contour_segments(poly,64)
        ax.add_collection(LineCollection(seg,colors=color,linewidths=1.2,linestyles=style,label=label))
    ax.axvline(6,color='.4',ls=':',lw=.8)
    ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')
axes[0,0].set(xlim=(1.7,6.35),ylim=(1.5,2.5),title=r'Accepted absolute $\phi=0.05$ contours')
axes[0,0].legend(fontsize=7,loc='upper center')
axes[0,1].set(xlim=(5.85,6.14),ylim=(1.91,2.09),title='Right endpoint; cell boundaries shown')
for xx in np.arange(5.8125,6.2,HX):axes[0,1].axvline(xx,color='.85',lw=.6,zorder=0)
for yy in np.arange(1.875,2.126,HY):axes[0,1].axhline(yy,color='.85',lw=.6,zorder=0)
xx=np.linspace(5.5+HX/256,6.4-HX/256,900)
yy=np.linspace(1.8+HY/256,2.2-HY/256,400)
delta=field.evaluate(xx[None,:],yy[:,None])-initial.evaluate(xx[None,:],yy[:,None])
im=axes[1,0].pcolormesh(xx,yy,delta,cmap='magma',vmin=0,shading='auto',rasterized=True)
axes[1,0].axvline(6,color='cyan',ls=':',lw=.8)
axes[1,0].set(xlabel='x (m)',ylabel='y (m)',title=r'Accepted $\Delta\phi$ near the old endpoint',aspect='equal')
fig.colorbar(im,ax=axes[1,0],fraction=.05)
for side,style in [(-1,'--'),(1,'-')]:
    p=np.array([row for row in profiles if row[0]==side])
    for col,color,label in [(2,'black',r'$\phi_0$'),(3,'#b2182b',r'$\phi$'),(4,'#2166ac',r'$\Delta\phi$')]:
        axes[1,1].plot(p[:,1],p[:,col],color=color,ls=style,lw=1.1,label=label if side==1 else None)
axes[1,1].axvline(6,color='.4',ls=':',lw=.8);axes[1,1].axhline(.05,color='.5',ls=':',lw=.8)
axes[1,1].set(xlim=(5.5,6.35),ylim=(-.002,.24),xlabel='x (m)',ylabel='Phase',
              title='Along-fault traces: solid above / dashed below y=2')
axes[1,1].legend(fontsize=9);axes[1,1].grid(alpha=.15)
save(fig,'free-turn-accepted-phase')

fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for name,mask,color in [('Original formed',old,'.25'),('Newly formed',new,'#b2182b'),
                       ('Prospective',planes.formed==0,'#2166ac')]:
    axes[0,0].plot(planes.x[mask],planes.angle[mask]*180/np.pi,'.',ms=3,color=color,label=name)
for ax,quantity,title in [(axes[0,1],sections.age,'Section age (s)'),
                        (axes[1,0],sections.slip*1e6,'Signed cumulative slip (µm)'),
                        (axes[1,1],sections.V*1e3,'Stage slip rate (mm/s)')]:
    ax.plot(planes.x,quantity,color='#b2182b',lw=1.2);ax.set_title(title)
axes[0,0].set_title('Structural plane (degrees); not the phase path');axes[0,0].legend(fontsize=8)
axes[0,1].axhline(.5+time,color='.5',ls=':',label='Quiescent age');axes[0,1].legend(fontsize=8)
for ax in axes.flat:
    ax.axvline(6,color='.5',ls=':',lw=.8);ax.set(xlim=(5.5,6.4),xlabel='Material x (m)');ax.grid(alpha=.2)
save(fig,'free-turn-accepted-histories')

# Log values are trial diagnostics, not accepted physical fields.
log=(path/'log.txt').read_text();rows=[];current_step=None
for line in log.splitlines():
    m=re.search(r'Bulk tip accepted parent: step=(\d+)',line)
    if m:current_step=int(m[1])+1
    m=re.search(r'Bulk tip coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+), section rate defect=(\S+)',line)
    if m:rows.append(dict(step=current_step,iteration=int(m[1]),mechanics=float(m[2]),phase_KKT=float(m[3]),rate_defect=float(m[4]),alpha=None))
    m=re.search(r'Bulk tip accepted search: shift=([^,]+), alpha=(\S+)',line)
    if m and rows:rows[-1].update(shift=float(m[1]),alpha=float(m[2]))
failed=[r for r in rows if r['step']==16]
assert failed[-1]['iteration']==80 and failed[-1]['phase_KKT']>1e-7
assert failed[-1]['mechanics']<1e-9
command=json.loads((RUNS/FINAL/'command.json').read_text())
assert command['exit_code']==1
result['failure']=dict(exit_code=1,total_process_seconds=command['seconds'],
    last_trial=failed[-1],accepted_step=15,checkpoint_slot=(path/'restart/last_good_checkpoint.txt').read_text().strip(),
    first_failed_gate='phase KKT; mechanics satisfies tolerance',
    limitation='No failing-row field export; row localization requires a separate diagnostic replay. No step16 physical state is published.')
with (OUT/'four-rank-nonlinear-history.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=['step','iteration','mechanics','phase_KKT','rate_defect','alpha','shift'])
    writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for s,color,label in [(15,'#2166ac','Step15 accepted'),(16,'#b2182b','Step16 failed')]:
    rr=[r for r in rows if r['step']==s];it=[r['iteration'] for r in rr]
    axes[0,0].semilogy(it,[r['phase_KKT'] for r in rr],color=color,label=label)
    axes[0,1].semilogy(it,[r['mechanics'] for r in rr],color=color,label=label)
    axes[1,0].semilogy(it,[r['alpha'] if r['alpha'] is not None else np.nan for r in rr],color=color)
axes[0,0].axhline(1e-7,color='.4',ls=':');axes[0,1].axhline(1e-9,color='.4',ls=':')
for ax,title in zip([axes[0,0],axes[0,1],axes[1,0]],['Phase KKT residual','Mechanical residual','Accepted line-search fraction']):
    ax.set(xlabel='Newton iteration',title=title);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
solver=data['cost'][-1]['solver_seconds']
axes[1,1].bar(['Accepted step15\nsolver','Other process time\n(includes failed step16)'],[solver/60,(command['seconds']-solver)/60],color=['#2166ac','#b2182b'])
axes[1,1].set(ylabel='Wall minutes',title='Four-rank continuation: measured cost')
save(fig,'free-turn-four-rank-convergence')
(OUT/'completed-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
