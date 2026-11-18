#!/usr/bin/env python3
"""Connected fronts and native quadratic fracture partitions after free runs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-preparation-mpl')
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import H,coefficients,fronts,value,read,sample
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-preparation'
RUNS=ROOT/'build-particle-free/bulk-tip/runs';OUT=HERE.parent/'evidence/stage4_preparation'
p=argparse.ArgumentParser();p.add_argument('family',default='free',nargs='?');p.add_argument('--reference',default='free');p.add_argument('--view',type=Path);p.add_argument('--reference-root',type=Path);a=p.parse_args()
assert a.family.isidentifier() and a.reference.isidentifier()
if a.view:RUNS=a.view.resolve()
REFERENCE_RUNS=a.reference_root.resolve() if a.reference_root else RUNS
b0=np.loadtxt(BUILD/'input-prepared/phase.txt',skiprows=1).reshape(128,256,3,3)
initial=fronts(b0,8);initial4=fronts(b0,4);initial_uncertainty=max(abs(initial[k]-initial4[k]) for k in ['right_x','left_x'])
histories={};records={};final_fields={}
for law in ['weakening','neutral']:
    run=RUNS/f'stage4-preparation-{a.family}-{law}';out=run/'output'
    if not (out/'pilot-summary.csv').exists():continue
    summary=read(out/'pilot-summary.csv');energy=read(out/'bulk-accepted.csv') if (out/'bulk-accepted.csv').exists() else []
    reference=REFERENCE_RUNS/f'stage4-preparation-{a.reference}-{law}'/'output'
    initial_fracture=float(read(reference/'preparation-iterations.csv')[-1]['fracture'])
    seed_files=sorted(reference.glob('tip-seed-fracture-r*.csv*'));assert len(seed_files)==4
    seed=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1,usecols=[1,2,3]) for p in seed_files])
    rows=[];uncertainty=initial_uncertainty
    first_step=int(summary[0]['step'])
    previous_b=b0 if first_step==0 else coefficients(reference,first_step-1)
    previous_fracture=initial_fracture
    if first_step:
        reference_energy=read(reference/'bulk-accepted.csv')
        previous_fracture=float(reference_energy[reference_energy['step']==first_step-1][0]['fracture'])
    for state in summary:
        step=int(state['step']);b=coefficients(out,step);f=fronts(b,8);f4=fronts(b,4)
        uncertainty=max(uncertainty,*(abs(f[k]-f4[k]) for k in ['right_x','left_x']))
        delta=b-b0;intact=np.max(b0,axis=(2,3))==0.;changed=np.max(delta,axis=(2,3))
        row={key:float(state[key]) for key in summary.dtype.names};row.update(step=step,right_phi05_x=f['right_x'],left_phi05_x=f['left_x'],right_advance=f['right_x']-initial['right_x'],left_advance=initial['left_x']-f['left_x'],right_advance_cells=(f['right_x']-initial['right_x'])/H,left_advance_cells=(initial['left_x']-f['left_x'])/H,components=f['sampled_components'],detached_phi05_area=f['detached_sample_area'],maximum_coefficient_increment=float(delta.max()),minimum_coefficient_increment=float(delta.min()),integrated_phase_increment=float(delta.sum()*H*H/9),initially_intact_max_increment=float(changed[intact].max()),initially_intact_changed_cells=int(sum(changed[intact]>1e-12)),work_error=0.,added_fracture=0.,cohesive_remainder=0.)
        for side,sign in [('right',1),('left',-1)]:
            row[side+'_phi05_y']=f[side+'_y']
            row[side+'_phi05_y_min']=f[side+'_y_range'][0]
            row[side+'_phi05_y_max']=f[side+'_y_range'][1]
            for branch in ['upper','lower']:
                key=side+'_'+branch+'_x';row[side+'_'+branch+'_advance']=sign*(f[key]-initial[key])
        step_delta=b-previous_b
        row.update(maximum_step_coefficient_increment=float(step_delta.max()),minimum_step_coefficient_increment=float(step_delta.min()),step_fracture_change=0.)
        tip_iy,tip_ix=int(initial['right_y']/H),int(initial['right_x']/H)
        row['right_outer_cell_max_total_increment']=float(delta[tip_iy,tip_ix].max())
        row['right_outer_cell_max_step_increment']=float(step_delta[tip_iy,tip_ix].max())
        if step:
            e=energy[energy['step']==step][0]
            row.update(work_error=float(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']),added_fracture=float(e['fracture']-initial_fracture),cohesive_remainder=float(e['phase_remainder']),phase_residual_work=float(e['phase_work']),mechanics=float(e['mechanical_residual']),phase_KKT=float(e['phase_KKT']))
            row['step_fracture_change']=float(e['fracture']-previous_fracture)
            previous_fracture=float(e['fracture'])
            files=sorted(out.glob(f'tip-balance-{step:04d}-r*.csv*'));assert len(files)==4
            balance=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1,usecols=range(1,12)) for p in files])
            assert np.array_equal(seed[:,0],balance[:,0])
            ix=np.rint(balance[:,1]/H-.5).astype(int);iy=np.rint(balance[:,2]/H-.5).astype(int);j=balance[:,0].astype(int)
            assert np.max(abs(seed[:,1]-b0[iy,ix,j//3,j%3]))<1e-15
            partition=.5*(balance[:,3]-seed[:,1])*(balance[:,6]+seed[:,2])
            # Row partitions assign every face term through its native gradient.
            # Separate cells crossed by the prepared front from wholly ahead cells.
            xx,yy=balance[:,1:3].T
            masks=dict(right_ahead=xx-H/2>=initial['right_x'],left_ahead=xx+H/2<=initial['left_x'],right_straddling=abs(xx-initial['right_x'])<H/2,left_straddling=abs(xx-initial['left_x'])<H/2,remote=(abs(xx-4)>3)|(abs(yy-2)>1),exactly_intact=intact[iy,ix])
            for name,mask in masks.items():
                row[name+'_fracture_partition']=float(sum(partition[mask]))
                row[name+'_max_coefficient_increment']=float(max(delta[iy[mask],ix[mask],j[mask]//3,j[mask]%3]))
            row['fracture_partition_identity_error']=float(sum(partition)-row['added_fracture'])
        else:
            for name in ['right_ahead','left_ahead','right_straddling','left_straddling','remote','exactly_intact']:
                row[name+'_fracture_partition']=0.;row[name+'_max_coefficient_increment']=0.
            row.update(phase_residual_work=0.,mechanics=0.,phase_KKT=0.,fracture_partition_identity_error=0.)
        rows.append(row);previous_b=b
    if not rows:continue
    assert np.min([r['minimum_coefficient_increment'] for r in rows])>=-1e-10
    assert np.min([r['minimum_step_coefficient_increment'] for r in rows])>=-1e-10
    cmd=json.loads((run/'command.json').read_text());records[law]=dict(initial_front=initial,final=rows[-1],front_sampling_uncertainty_m=uncertainty,seconds=cmd.get('seconds'),exit_code=cmd.get('exit_code'),signed_work_error=sum(r['work_error'] for r in rows),absolute_work_error=sum(abs(r['work_error']) for r in rows),maximum_fracture_partition_identity_error=max(abs(r['fracture_partition_identity_error']) for r in rows),onsets={})
    planes=read(out/f'tip-planes-{rows[-1]["step"]:04d}.csv')
    prepared_planes=np.loadtxt(BUILD/'input-prepared/sections.txt',skiprows=1)
    assert np.max(abs(planes['x']-prepared_planes[:,0]))<1e-12
    original=prepared_planes[:,2]>0;born=(planes['formed']>0)&~original
    records[law]['structural_planes']=dict(newly_formed_sections=int(sum(born)),maximum_old_formed_angle_change_rad=float(max(abs(planes['angle'][original]-prepared_planes[original,1]))),newly_formed_angle_range_deg=None if not np.any(born) else [float(min(planes['angle'][born])*180/np.pi),float(max(planes['angle'][born])*180/np.pi)],note='native structural planes are distinct from phase-contour tangents and from merely selected virgin planes')
    for level in [.001,.01,.05]:
        passed=[r for r in rows if r['right_advance_cells']>=level]
        records[law]['onsets'][str(level)+'cell']=None if not passed else dict(first_endpoint_time=passed[0]['time'],previous_endpoint_time=max(0.,passed[0]['time']-passed[0]['dt']),criterion_m=level*H)
    # The existing contour reducer groups tied extrema within1nm. Use that
    # numerical extraction floor separately from cell-scale growth thresholds.
    # It is not a mesh-accuracy claim or a physical crack-onset definition.
    detection=max(1e-9,2*uncertainty)
    passed=[r for r in rows if r['right_advance']>=detection]
    records[law]['contour_detection_floor_m']=detection
    records[law]['onsets']['numerical_contour_detection']=None if not passed else dict(first_endpoint_time=passed[0]['time'],previous_endpoint_time=max(0.,passed[0]['time']-passed[0]['dt']),criterion_m=detection,note='numerical contour motion only; compare local timestep sensitivity separately')
    histories[law]=rows;final_fields[law]=b
    with (OUT/f'{a.family}-{law}-growth.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/f'{a.family}-{law}-phase.npz',initial=b0,final=b)
assert histories
result=dict(laws=records,contour='absolute phi=.05, connected to seed; native cell roots and both DG traces',fracture_partition='native quadratic row partition; wholly ahead and front-straddling cells reported separately',detection_levels_cells=[.001,.01,.05])
if len(histories)==2:
    by_time={law:{round(r['time'],13):r for r in rows} for law,rows in histories.items()}
    common=sorted(set(by_time['weakening'])&set(by_time['neutral']));assert common
    last=common[-1];w,n=(by_time[law][last] for law in ['weakening','neutral'])
    result.update(common_last_endpoint_time=last,common_endpoints=dict(weakening=w,neutral=n),right_front_law_difference_m=w['right_phi05_x']-n['right_phi05_x'],left_advance_law_difference_m=w['left_advance']-n['left_advance'])
    result['common_work_errors']={law:dict(signed=sum(r['work_error'] for r in rows if round(r['time'],13)<=last),absolute=sum(abs(r['work_error']) for r in rows if round(r['time'],13)<=last)) for law,rows in histories.items()}
(OUT/f'{a.family}-growth.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for law,rows in histories.items():
    t=np.array([r['time'] for r in rows])*1e3;label='Weakening' if law=='weakening' else 'Matched b=a';color='tab:red' if law=='weakening' else 'tab:blue'
    for key,style in [('right','-'),('left','--')]:axs[0,0].plot(t,[r[key+'_advance_cells'] for r in rows],style,color=color,label=label+', '+key)
    axs[0,1].plot(t,[r['peak_rate']*1e3 for r in rows],color=color,label=label)
    for key,style in [('right_ahead','-'),('right_straddling','--')]:axs[1,0].plot(t,[r[key+'_fracture_partition'] for r in rows],style,color=color,label=label+', '+key)
    axs[1,1].plot(t,[r['added_fracture'] for r in rows],color=color,label=label+', fracture')
    axs[1,1].plot(t,np.cumsum([abs(r['work_error']) for r in rows]),':',color=color,label=label+', absolute work error')
for ax,title,ylabel in zip(axs.flat,['Connected absolute φ=.05 advance','Accepted stage acceleration','Native fracture partition beyond prepared tip','Physical work and fracture inventory'],['Advance / cell width','Peak V (mm/s)','Fracture change (J/m)','Energy (J/m)']):
    ax.set(xlabel='Endpoint time (ms)',title=title,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
# Stage rates are associated with a midpoint; the common CSV preserves that
# precise time. This panel's rate abscissa is relabelled and plotted separately.
axs[0,1].clear()
for law,rows in histories.items():axs[0,1].plot([r['stage_time']*1e3 for r in rows],[r['peak_rate']*1e3 for r in rows],color='tab:red' if law=='weakening' else 'tab:blue',label='Weakening' if law=='weakening' else 'Matched b=a')
axs[0,1].set(xlabel='Stage time (ms)',ylabel='Peak V (mm/s)',title='Accepted stage acceleration');axs[0,1].legend(fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'{a.family}-growth.{ext}',dpi=170)
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
reference_planes=np.loadtxt(BUILD/'input-prepared/sections.txt',skiprows=1)
for side,ax in zip(['left','right'],axs):
    xx=np.linspace(initial[side+'_x']-.25,initial[side+'_x']+.35,1801)
    angle=np.interp(initial[side+'_x'],reference_planes[:,0],reference_planes[:,1])
    yy=initial[side+'_y']+np.tan(angle)*(xx-initial[side+'_x'])
    base=value(b0,xx,yy)
    ax.plot(xx,base,'k',label='Prepared φ₀');ax.axhline(.05,color='.4',ls=':',label='Absolute .05')
    twin=ax.twinx()
    for law,b in final_fields.items():
        color='tab:red' if law=='weakening' else 'tab:blue'
        field=value(b,xx,yy);ax.plot(xx,field,color=color,label=law+' φ');twin.plot(xx,field-base,'--',color=color,label=law+' Δφ, right axis')
    ax.set(xlabel='x (m)',ylabel='Phase',title=side+f' prepared plane through tip, θ={angle*180/np.pi:.2f}°')
    twin.set_ylabel('φ−φ₀');twin.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    lines,labels=ax.get_legend_handles_labels();extra,names=twin.get_legend_handles_labels();ax.legend(lines+extra,labels+names,fontsize=7)
for ext in ('png','pdf'):fig.savefig(OUT/f'{a.family}-profiles.{ext}',dpi=170)
fig,axs=plt.subplots(1,len(final_fields),figsize=(6*len(final_fields),4),squeeze=False,constrained_layout=True)
n=8;xx=(np.arange(256*n)+.5)*H/n;yy=(np.arange(128*n)+.5)*H/n;base=sample(b0,n)
maximum=max(float((sample(b,n)-base).max()) for b in final_fields.values());maximum=max(maximum,1e-15)
crop=np.ix_((yy>=1.7)&(yy<=2.3),(xx>=initial['right_x']-.5)&(xx<=initial['right_x']+.5))
for ax,(law,b) in zip(axs.flat,final_fields.items()):
    f=sample(b,n);delta=f-base
    im=ax.pcolormesh(xx[crop[1].ravel()],yy[crop[0].ravel()],delta[crop],vmin=0,vmax=maximum,cmap='magma',shading='nearest',rasterized=True)
    ax.contour(xx[crop[1].ravel()],yy[crop[0].ravel()],base[crop],levels=[.05],colors='white',linestyles='--')
    ax.contour(xx[crop[1].ravel()],yy[crop[0].ravel()],f[crop],levels=[.05],colors='cyan')
    ax.set(xlabel='x (m)',ylabel='y (m)',title=law+': Δφ and absolute φ=.05',aspect='equal');fig.colorbar(im,ax=ax,label='φ−φ₀')
fig.suptitle('Prepared contour dashed white; endpoint cyan. Native strip planes are distinct.')
for ext in ('png','pdf'):fig.savefig(OUT/f'{a.family}-tip-map.{ext}',dpi=170)
print(json.dumps(result,indent=2))
