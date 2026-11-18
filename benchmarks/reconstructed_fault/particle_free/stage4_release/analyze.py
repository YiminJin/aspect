#!/usr/bin/env python3
"""Reduce the selected pair after numerical ranks exit; no new simulation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-release-mpl')
from pathlib import Path
import json,csv,shutil
import numpy as np
from scipy.ndimage import label,binary_dilation
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
RUNS=ROOT/'build-particle-free/bulk-tip/runs';OUT=HERE.parent/'evidence/stage4_release';OUT.mkdir(exist_ok=True)
H=1/32
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def coefficients(out,step):
    files=sorted(out.glob(f'release-phase-{step:04d}-r*.csv'));assert len(files)==4
    a=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1,usecols=range(1,12)) for p in files])
    ix=np.rint(a[:,0]/H-.5).astype(int);iy=np.rint(a[:,1]/H-.5).astype(int)
    assert len(a)==32768 and len(np.unique(256*iy+ix))==32768
    b=np.empty((128,256,3,3));b[iy,ix]=a[:,2:].reshape(-1,3,3);return b
def basis(t):return np.stack([(1-t)**2,2*t*(1-t),t*t],axis=-1)
def sample(b,n=8):
    q=basis((np.arange(n)+.5)/n)
    return np.einsum('ai,yxij,bj->yaxb',q,b,q).reshape(128*n,256*n)
def value(b,x,y):
    x=np.asarray(x);y=np.broadcast_to(y,x.shape)
    ix=np.minimum((x/H).astype(int),255);iy=np.minimum((y/H).astype(int),127)
    return np.einsum('...i,...ij,...j->...',basis(y/H-iy),b[iy,ix],basis(x/H-ix))
def max_normal(b,u):
    d=b@basis(np.asarray(u));a=d[0]-2*d[1]+d[2]
    v=(d[0]-d[1])/a if a<0 else -1.
    return max(d[0],d[2],float(basis(np.asarray(v))@d) if 0<v<1 else -1.)
def connected_front(b,n=8,level=.05):
    f=sample(b,n);labels,count=label(f>=level);seed=labels[64*n,128*n];assert seed>0
    connected=labels==seed
    cells=connected.reshape(128,n,256,n).any(axis=(1,3))
    nearby=binary_dilation(cells)
    # Root of max_v phi(u,v)=level locates the rightmost cell-local contour,
    # with both DG traces retained. Dilate once to include sub-sample tip caps.
    candidates=np.argwhere(nearby & (np.max(b,axis=(2,3))>=level))
    right=-np.inf
    for iy,ix in candidates[np.argsort(candidates[:,1])[::-1]]:
        if (ix+1)*H<right:break
        cell=b[iy,ix];u=np.linspace(0,1,65);g=np.array([max_normal(cell,v)-level for v in u]);positive=np.flatnonzero(g>=0)
        if not len(positive):continue
        j=positive[-1];r=1. if j==64 else brentq(lambda v:max_normal(cell,v)-level,u[j],u[j+1],xtol=1e-13)
        right=max(right,(ix+r)*H)
    assert np.isfinite(right)
    return dict(right_x=float(right),sampled_components=int(count),connected_sample_area=float(connected.sum()*(H/n)**2))

histories={};records={};initial=None;final_fields={}
for law in ('weakening','neutral'):
    name=f'stage4-release-256-{law}';run=RUNS/name;out=run/'output'
    if not (out/'release-driving.csv').exists():continue
    summary=read(out/'pilot-summary.csv');drive=read(out/'release-driving.csv');regions=read(out/'release-regions.csv')
    accepted=read(out/'bulk-accepted.csv') if (out/'bulk-accepted.csv').exists() else np.array([])
    complete_steps=sorted(set(int(x) for x in drive['step']))
    rows=[];b0=coefficients(out,0)
    if initial is None:initial=b0
    else:assert np.array_equal(initial,b0),'Pair must share exact initial phase.'
    first_front=connected_front(b0);front4=connected_front(b0,4)
    initial_left=8-connected_front(b0[:,::-1,:,::-1])['right_x']
    for step in complete_steps:
        b=coefficients(out,step);front=connected_front(b)
        left=8-connected_front(b[:,::-1,:,::-1])['right_x']
        state=summary[summary['step']==step][0]
        intact=np.max(b0,axis=(2,3))==0.
        intact_change=np.max(b-b0,axis=(2,3))[intact]
        changed=np.argwhere(np.max(b-b0,axis=(2,3))>0.)
        front_column=int(first_front['right_x']/H)
        row=dict(step=step,time=float(state['time']),stage_time=float(state['stage_time']),
                 peak_rate=float(state['peak_rate']),integrated_rate=float(state['integrated_rate']),
                 integrated_slip=float(state['integrated_slip']),mean_resistance=float(state['mean_patch_resistance']),
                 right_phi05_x=front['right_x'],right_advance=front['right_x']-first_front['right_x'],
                 right_advance_cells=(front['right_x']-first_front['right_x'])/H,
                 left_phi05_x=float(left),left_advance=float(initial_left-left),
                 left_advance_cells=float((initial_left-left)/H),
                 sampled_components=front['sampled_components'],max_coefficient_increment=float(np.max(b-b0)),
                 minimum_coefficient_increment=float(np.min(b-b0)),
                 max_increment_in_exactly_intact_cells=float(np.max(intact_change)),
                 new_nonzero_intact_cells=int(np.count_nonzero(intact_change>0.)),
                 rightmost_changed_cell_edge=None if not len(changed) else float((changed[:,1].max()+1)*H),
                 max_increment_at_or_ahead_of_initial_front_cell=float(np.max((b-b0)[:,front_column:])))
        for probe in (0,1):
            r=drive[(drive['step']==step)&(drive['probe']==probe)][0]
            for key in ('mechanical_driving','fracture_derivative','net_driving','mean_net_driving','admissible_stage_epsilon'):
                row[f'probe{probe}_{key}']=float(r[key])
        for region in range(4):
            r=regions[(regions['step']==step)&(regions['region']==region)][0]
            for key in ('max_phase_increment','integrated_phase_increment','added_fracture_row_partition'):
                row[f'region{region}_{key}']=float(r[key])
            # Selected fixture: Gc=2/3 J/m², ell=.125 m, c_w=8/3 -> Kf=2 Pa.
            row[f'region{region}_potential_increment']=2.*float(r['integrated_phase_increment'])
            row[f'region{region}_gradient_face_increment']=float(r['added_fracture_row_partition'])-row[f'region{region}_potential_increment']
        if step:
            e=accepted[accepted['step']==step][0]
            row.update(added_fracture=sum(row[f'region{r}_added_fracture_row_partition'] for r in range(4)),
                       physical_work_error=float(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']),
                       cohesive_remainder=float(e['phase_remainder']),phase_residual_work=float(e['phase_work']),
                       mechanical_work_error=float(e['mechanical_work']),phase_KKT=float(e['phase_KKT']),
                       mechanical_residual=float(e['mechanical_residual']))
        else:row.update(added_fracture=0.,physical_work_error=0.,cohesive_remainder=0.,phase_residual_work=0.,mechanical_work_error=0.,phase_KKT=None,mechanical_residual=None)
        rows.append(row)
    last=rows[-1];last_field=coefficients(out,last['step']);final_fields[law]=last_field
    final4=connected_front(last_field,4)
    errors=np.array([r['physical_work_error'] for r in rows])
    command=json.loads((run/'command.json').read_text())
    reference=read(RUNS/f'stage4-native-256-{law}-perturbed'/'output/pilot-summary.csv')
    n=min(len(rows),len(reference));reference_errors={}
    for key in ('peak_rate','integrated_rate','integrated_slip'):
        v=np.array([r[key] for r in rows[:n]])
        reference_errors[key]=float(np.max(abs(v-reference[key][:n]))/np.max(abs(reference[key][:n]))) if n>1 else None
    end_sections=read(out/f'tip-sections-{last["step"]:04d}.csv') if last['step'] else None
    sign={} if end_sections is None else dict(positive=float(sum(end_sections['ds']*end_sections['positive_w'])),negative=float(sum(end_sections['ds']*end_sections['negative_w'])))
    if sign:sign['cancellation_fraction']=1-abs(sign['positive']+sign['negative'])/(sign['positive']-sign['negative'])
    records[law]=dict(exit_code=command.get('exit_code'),seconds=command.get('seconds'),accepted_steps=last['step'],
       initial_front=first_front,front_sampling_4_to_8_max_difference=max(abs(first_front['right_x']-front4['right_x']),abs(last['right_phi05_x']-final4['right_x'])),
       endpoint=last,common_fixed_phase_relative_differences=reference_errors,common_fixed_phase_steps=n-1,
       signed_cumulative_physical_work_error=float(sum(errors)),accumulated_absolute_physical_work_error=float(sum(abs(errors))),
       accumulated_absolute_external_work=float(sum(abs(accepted['Wext']))) if len(accepted) else 0.,
       final_signed_deformation=sign,initial_phase_hash_note='Exact equality of all initial coefficients verified across laws.')
    records[law]['potential_fracture_increment']=sum(last[f'region{r}_potential_increment'] for r in range(4))
    records[law]['gradient_face_fracture_increment']=sum(last[f'region{r}_gradient_face_increment'] for r in range(4))
    histories[law]=rows
    with (OUT/f'{law}-history.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    target=OUT/name;target.mkdir(exist_ok=True)
    for p in [run/'command.json',run/'input.prm',run/'run.log',*out.glob('*.csv')]:
        if p.name.startswith('release-phase-'):continue
        shutil.copy2(p,target/p.name)
    np.savez_compressed(OUT/f'{law}-phase.npz',initial=b0,final=last_field)

assert histories,'No complete diagnostic observations.'
colors={'weakening':'tab:red','neutral':'tab:blue'}
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for law,rows in histories.items():
    c=colors[law];label_name='Weakening' if law=='weakening' else 'Matched b=a'
    t=np.array([r['time'] for r in rows])*1e3;stage=np.array([r['stage_time'] for r in rows])*1e3
    axs[0,0].plot(stage,[r['peak_rate']*1e3 for r in rows],'-o',color=c,ms=3,label=label_name)
    for probe,style in ((0,'-'),(1,'--')):
        axs[0,1].plot(stage,[r[f'probe{probe}_mean_net_driving'] for r in rows],style,color=c,label=f'{label_name}, '+('ahead of .05' if probe==0 else 'intact'))
    axs[1,0].plot(t,[r['right_advance_cells'] for r in rows],'-o',color=c,ms=3,label=label_name)
    axs[1,1].plot(t,[r['added_fracture'] for r in rows],'-o',color=c,ms=3,label=label_name+' added fracture')
    axs[1,1].plot(t,np.cumsum([abs(r['physical_work_error']) for r in rows]),':',color=c,label=label_name+' absolute work error')
for ax,y,title in zip(axs.flat,['Peak slip rate (mm/s)','Mean net driving (Pa)','Right tip advance / cell width','Energy per thickness (J/m)'],['Accepted midpoint slip rate','Fixed admissible ahead-tip probes','Connected absolute φ=.05 contour','Fracture work and physical-work error']):
    ax.set(xlabel='Time (ms)',ylabel=y,title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[0,1].axhline(0,color='.4',lw=.7)
if max(abs(r['right_advance_cells']) for rows in histories.values() for r in rows)<1e-8:axs[1,0].set_ylim(-.01,.01)
for ext in ('png','pdf'):fig.savefig(OUT/f'release-history.{ext}',dpi=170)
plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,6),layout='constrained')
n=8;x=(np.arange(256*n)+.5)*H/n;y=(np.arange(128*n)+.5)*H/n
base=sample(initial,n);vmax=max(float(np.max(sample(b,n)-base)) for b in final_fields.values());vmax=max(vmax,1e-15)
for column,(law,b) in enumerate(final_fields.items()):
    f=sample(b,n);delta=f-base
    for row,field in enumerate((f,delta)):
        ax=axs[row,column];xm=(x>=1.5)&(x<=6.5);ym=(y>=1.7)&(y<=2.3)
        crop=np.ix_(ym,xm)
        im=ax.pcolormesh(x[xm],y[ym],field[crop],shading='nearest',vmin=0,vmax=.6 if row==0 else vmax,cmap='viridis' if row==0 else 'magma',rasterized=True)
        ax.contour(x[xm],y[ym],base[crop],levels=[.05],colors='white',linestyles='--',linewidths=1.)
        ax.contour(x[xm],y[ym],f[crop],levels=[.05],colors='cyan',linewidths=.7)
        ax.set(xlim=(1.5,6.5),ylim=(1.7,2.3),xlabel='x (m)',ylabel='y (m)',title=law+(': φ' if row==0 else ': φ−φ₀'))
        fig.colorbar(im,ax=ax,shrink=.85)
fig.suptitle('Initial φ=.05 dashed white; final φ=.05 cyan (overlap where unchanged)')
for ext in ('png','pdf'):fig.savefig(OUT/f'release-phase.{ext}',dpi=170)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');xline=np.linspace(5.5,6.5,2001)
seed=value(initial,xline,2.)
axs[0].plot(xline,seed,color='.3',label='φ₀');axs[0].axhline(.05,color='.5',ls=':')
for ax in axs:ax.grid(alpha=.2)
for law,b in final_fields.items():
    f=value(b,xline,2.);axs[0].plot(xline,f,color=colors[law],label=law+' φ')
    axs[1].plot(xline,f-seed,color=colors[law],label=law+' Δφ')
axs[0].set(title='Axial trace y=2 m: φ₀ and φ',xlabel='x (m)',ylabel='Phase')
axs[1].set(title='Same trace: φ−φ₀',xlabel='x (m)',ylabel='Phase increment')
for ax in axs:ax.legend(fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'release-tip-profile.{ext}',dpi=170)
plt.close(fig)
if len(final_fields)==2:
    difference=final_fields['weakening']-final_fields['neutral']
    z,w=np.polynomial.legendre.leggauss(3);q=basis((z+1)/2)
    sampled=np.einsum('ai,yxij,bj->yxab',q,difference,q)
    records['pair']=dict(maximum_final_coefficient_difference=float(np.max(abs(difference))),
                        final_phase_L2_difference=float(np.sqrt(np.einsum('yxab,a,b->',sampled*sampled,w,w)*H*H/4)),
                        difference_in_added_fracture=records['weakening']['endpoint']['added_fracture']-records['neutral']['endpoint']['added_fracture'])
(OUT/'comparison.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
