#!/usr/bin/env python3
"""Compact bulk-tip validation reduction. Never alters simulation data."""
import argparse,json
from pathlib import Path
import numpy as np
import csv
def read(file):
    return np.atleast_1d(np.genfromtxt(file,delimiter=",",names=True,dtype=None,encoding="utf8")).view(np.recarray)
root=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('labels',nargs='+');p.add_argument('--figures',action='store_true');p.add_argument('--time',type=float);a=p.parse_args()
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip';out.mkdir(exist_ok=True)
def initial(x,y):return .2*np.maximum(0,1-((x-1)/.5)**2)*np.maximum(0,1-((y-.5)/.25)**2)
def points(directory,step):
    files=sorted(directory.glob(f'tip-points-{step:04d}-r*.csv'))
    result=np.concatenate([read(f) for f in files]).view(np.recarray)
    return result[np.lexsort((result.y,result.x))]
def norm(frame,columns):
    value=sum((2 if key.endswith('xy') else 1)*frame[key]**2 for key in columns)
    return float(np.sqrt(np.sum(frame.weight*value)))
results={};data={}
for label in a.labels:
    path=root/'build-particle-free/bulk-tip/runs'/label;command=json.loads((path/'command.json').read_text())
    result=dict(exit_code=command.get('exit_code'),seconds=command.get('seconds'),binary_sha256=command['binary_sha256'])
    directory=path/'output';file=directory/'bulk-accepted.csv'
    if not file.exists():results[label]=result;continue
    steps=read(file)
    if a.time is not None:steps=steps[steps.time<=a.time+1e-12]
    if len(steps)==0:results[label]=result;continue
    last=steps[-1];frame=points(directory,int(last.step));sections=read(directory/f'tip-sections-{int(last.step):04d}.csv')
    power=read(directory/'tip-power.csv');power=power[power.step<=last.step];phase0=initial(frame.x,frame.y)
    if command['arguments']['intact']:phase0=np.zeros(len(frame))
    virgin=phase0==0.;growth=frame.phi-phase0
    result.update(accepted_steps=len(steps),last_time=float(last.time),max_phase=float(frame.phi.max()),
        minimum_phase=float(frame.phi.min()),minimum_damage_increment=float(growth.min()),
        virgin_damage_integral=float(np.sum(frame.weight[virgin]*frame.phi[virgin])),
        virgin_volume_above_1e_6=float(np.sum(frame.weight[virgin & (frame.phi>1e-6)])),
        volume_above_0_99=float(np.sum(frame.weight[frame.phi>.99])),
        stress_L2=norm(frame,['tauxx','tauyy','tauzz','tauxy']),
        max_slip_rate=float(np.abs(sections.V).max()),max_slip=float(np.abs(sections.slip).max()),
        minimum_age=float(sections.age.min()),maximum_age=float(sections.age.max()),
        max_mechanical_residual=float(steps.mechanical_residual.max()),max_phase_KKT=float(steps.phase_KKT.max()),
        max_ledger_error=float(np.abs(steps.ledger_error).max()),max_power_adjoint_error=float(np.abs(power.power_adjoint_error).max()),
        max_section_power_defect=float(np.abs(power.section_residual_work).max()),
        max_integral_rate_defect=float(np.abs(sections.integral_rate_defect).max()))
    result['threshold_extent']={str(t):([float(frame.x[frame.phi>t].min()),float(frame.x[frame.phi>t].max())] if np.any(frame.phi>t) else None) for t in [1e-6,.001,.01,.1,.99]}
    storage=steps.bulk+steps.cohesive
    added_fracture=steps.energy_change-np.diff(storage,prepend=0.)
    remainder=steps.Wext-steps.energy_change-steps.Db-steps.Dc-power.friction-power.radiation
    terms=['stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error']
    result['energy']={k:dict(signed=float(steps[k].sum()),absolute=float(np.abs(steps[k]).sum())) for k in terms}
    result['energy'].update(physical_remainder=dict(signed=float(remainder.sum()),absolute=float(np.abs(remainder).sum())),
        fracture_added=float(added_fracture.sum()),active_external_work=float(np.abs(steps.Wext).sum()),
        friction=float(power.friction.sum()),radiation=float(power.radiation.sum()),bulk_viscosity=float(steps.Db.sum()),cohesive_viscosity=float(steps.Dc.sum()))
    # The active window excludes the initial unloaded fracture-profile adjustment.
    active=steps.time>.04+1e-12
    result['active_window']=dict(start=.04,remainder_signed=float(remainder[active].sum()),
       remainder_absolute=float(np.abs(remainder[active]).sum()),fracture_added=float(added_fracture[active].sum()),
       work_absolute=float(np.abs(steps.Wext[active]).sum()),
       components={k:dict(signed=float(steps[k][active].sum()),absolute=float(np.abs(steps[k][active]).sum())) for k in terms})
    result['regions']={}
    for name,mask in [('central',(frame.x>=.75)&(frame.x<=1.25)),('endpoints',((frame.x>=.25)&(frame.x<.75))|((frame.x>1.25)&(frame.x<=1.75)))]:
        q=frame[mask];measure=q.weight.sum()
        result['regions'][name]={k:float(np.sum(q.weight*q[k])/measure) for k in ['phi','tauxy','tauyy','qcoh','signed_shear','absolute_shear']}
    # Diagnostic only, not a constitutive localization/admission rule.
    bands=[];negative=0.;positive=0.;variance=0.;cohesive=0.
    for k in np.unique(frame.section):
        q=frame[frame.section==k];q=q[np.argsort(q.y)];ds=sections[int(k)].ds;dn=q.weight/ds
        negative+=float(np.sum(q.weight*np.minimum(q.w,0.)));positive+=float(np.sum(q.weight*np.maximum(q.w,0.)))
        threshold=max(1e-10,1e-6*np.abs(q.w).max());active_rate=np.abs(q.w)>threshold
        bands.append(int(np.sum(active_rate & ~np.r_[False,active_rate[:-1]])))
        if q.phi.max()<1:
            h=(26.6666666667/((8./3.)*.25)/.5)*q.phi*(1+q.phi)/(1-q.phi)**2  # fixture K/(c^2/(2G)) = 80
            I=float(np.sum(dn*h));elastic=float(np.sum(dn*2*q.Axy));E=float(np.sum(dn*q.Axy*q.qcoh))
            cohesive+=ds*E
            if I>0:variance+=ds*(E-1e6*elastic**2/(2*I))
    result['section_diagnostics']=dict(raw_positive_integral=positive,raw_negative_integral=negative,
       cancellation_fraction=(1-abs(positive+negative)/(positive-negative) if positive>negative else 0.),
       maximum_sampled_active_bands=max(bands,default=0),multiple_band_sections=sum(b>1 for b in bands),
       band_threshold='max(1e-10 /s, 1e-6 section max |w|), diagnostic only',cohesive=cohesive,transverse_variance=variance)
    history=dict(negative_local_work=0.,positive_local_work=0.,maximum_bands=0,steps_with_multiple_bands=0,
                 maximum_resolved_negative_rate=0.,maximum_raw_negative_rate=0.)
    for step in steps:
        q=points(directory,int(step.step));ss=read(directory/f'tip-sections-{int(step.step):04d}.csv')
        local=step.dt*q.weight*q.w*ss.force[q.section.astype(int)]
        history['negative_local_work']+=float(np.sum(np.minimum(local,0.)))
        history['positive_local_work']+=float(np.sum(np.maximum(local,0.)))
        history['maximum_raw_negative_rate']=max(history['maximum_raw_negative_rate'],float(np.max(np.maximum(-q.w,0.))))
        multiple=False
        for k in np.unique(q.section):
            samples=q[q.section==k];samples=samples[np.argsort(samples.y)]
            threshold=max(1e-10,1e-6*np.max(np.abs(samples.w)))
            active_samples=np.abs(samples.w)>threshold
            count=int(np.sum(active_samples & ~np.r_[False,active_samples[:-1]]))
            history['maximum_bands']=max(history['maximum_bands'],count);multiple=multiple or count>1
            negative=samples.w[samples.w < -threshold]
            if len(negative):history['maximum_resolved_negative_rate']=max(history['maximum_resolved_negative_rate'],float(np.max(-negative)))
        history['steps_with_multiple_bands']+=int(multiple)
    result['signed_history']=history
    energy_columns=dict(time=steps.time,physical_remainder=remainder,
        added_fracture=added_fracture,BE_bulk=steps.stress_loss,BE_cohesive=steps.traction_loss,
        compliance=steps.phase_remainder,fracture_numerical=steps.fracture_loss,closure=power.section_residual_work)
    np.savetxt(out/(f'{label}-energy.csv' if a.time is None else f'{label}-energy-at-{a.time:g}.csv'),np.column_stack(list(energy_columns.values())),delimiter=',',header=','.join(energy_columns),comments='')
    results[label]=result;data[label]=(steps,frame,sections,power,remainder)
comparisons={}
for i,name in enumerate(a.labels):
    if name not in data:continue
    for other in a.labels[i+1:]:
        if other not in data or abs(results[name]['last_time']-results[other]['last_time'])>1e-12:continue
        f=data[name][1];g=data[other][1]
        if len(f)!=len(g) or max(np.max(np.abs(f.x-g.x)),np.max(np.abs(f.y-g.y)))>1e-12:continue
        diff=np.recarray(len(g),dtype=[(c,float) for c in ['weight','tauxx','tauyy','tauzz','tauxy','phi','Axy','signed_shear']]);diff.weight=g.weight
        for col in ['tauxx','tauyy','tauzz','tauxy','phi','Axy','signed_shear']:diff[col]=g[col].astype(float)-f[col].astype(float)
        comparisons[f'{name}:{other}']=dict(stress_relative_L2=norm(diff,['tauxx','tauyy','tauzz','tauxy'])/max(norm(g,['tauxx','tauyy','tauzz','tauxy']),1e-30),
            phase_L2=norm(diff,['phi']),cohesive_strain_L2=norm(diff,['Axy']),signed_strain_L2=norm(diff,['signed_shear']))
cell_comparisons={}
for i,name in enumerate(a.labels):
    if name not in data:continue
    for other in a.labels[i+1:]:
        if other not in data or abs(results[name]['last_time']-results[other]['last_time'])>1e-12:continue
        f=data[name][1];g=data[other][1]
        if len(f)==len(g):continue
        cols=['phi','tauxx','tauyy','tauzz','tauxy','Axy','signed_shear']
        def averages(q):
            ids=np.floor(q.x/.25).astype(int)+8*np.floor(q.y/.25).astype(int)
            weights=np.bincount(ids,weights=q.weight,minlength=32)
            return np.array([np.bincount(ids,weights=q.weight*q[c],minlength=32)/weights for c in cols])
        af,ag=averages(f),averages(g);err=ag-af
        cell_comparisons[f'{name}:{other}']={c:float(np.sqrt(.0625*np.sum(err[j]**2))) for j,c in enumerate(cols)}
        cell_comparisons[f'{name}:{other}']['stress_relative_L2']=float(np.sqrt(np.sum(err[1:4]**2)+2*np.sum(err[4]**2))/max(np.sqrt(np.sum(ag[1:4]**2)+2*np.sum(ag[4]**2)),1e-300))
for comparison in cell_comparisons.values():
    if comparison['stress_relative_L2']>1e100:comparison['stress_relative_L2']=None
for comparison in comparisons.values():
    if comparison['stress_relative_L2']>1e20:comparison['stress_relative_L2']=None
summary=dict(cases=results,same_mesh_comparisons=comparisons,coarse_cell_mean_comparisons=cell_comparisons)
(out/('summary.json' if a.time is None else f'summary-at-{a.time:g}.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if a.figures and data:
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    selected=list(data)[:3]
    fig,axes=plt.subplots(1,min(3,len(selected)),figsize=(5*min(3,len(selected)),3.5),squeeze=False,constrained_layout=True)
    for ax,name in zip(axes[0],selected[:3]):
        f=data[name][1];sc=ax.scatter(f.x,f.y,c=f.phi,s=6,norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=0,vmax=max(results[k]['max_phase'] for k in selected)),cmap='magma')
        ax.plot([.5,1.5,1.5,.5,.5],[.25,.25,.75,.75,.25],'c--',lw=.8,label='initial support')
        ax.set(xlabel='x (m)',ylabel='y (m)',title=f'{name}, t={results[name]["last_time"]:g} s');ax.set_aspect('equal');ax.legend(fontsize=7)
    fig.colorbar(sc,ax=axes[0].tolist(),label='bulk damage');fig.savefig(out/'tip-specimen.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,8),constrained_layout=True)
    for name in selected:
        s=data[name][2];axes[0,0].plot(s.s,s.force,label=name);axes[0,1].plot(s.s,s.V,label=name)
        axes[1,0].plot(s.s,s.age,label=name);axes[1,1].plot(s.s,s.slip,label=name)
        f=data[name][1];driving=[];traction=[]
        for k in range(len(s)):
            q=f[f.section==k];phi=q.phi;c=80*phi*(1+phi);d=(1-phi)**2
            Y=80*(1+3*phi)*q.qcoh**2/(2e6*np.maximum((1-phi)**3,1e-300))
            driving.append(np.sum(q.weight*Y)/np.sum(q.weight));traction.append(np.sum(q.weight*q.tauxy)/np.sum(q.weight))
        axes[2,0].plot(s.s,driving,label=name);axes[2,1].plot(s.s,traction,label=name)
    for ax,title,units in zip(axes.flat,['Section resistance','Signed integrated fault rate','Physical section age','Cumulative section slip','Normal-averaged cohesive driving Y','Normal-averaged shear stress'],['Pa','m/s','s','m','Pa','Pa']):
        ax.set(xlabel='section x (m)',ylabel=units,title=title);ax.axvline(.5,color='.7',ls=':');ax.axvline(1.5,color='.7',ls=':');ax.legend(fontsize=7)
    fig.savefig(out/'tip-sections.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for name in selected:
        steps,f,s,power,r=data[name];axes[0].plot(steps.time,r.cumsum(),label=name);axes[1].plot(steps.time,np.abs(r).cumsum(),label=name)
    axes[0].set(title='Signed cumulative physical remainder',xlabel='t (s)',ylabel='J/m');axes[1].set(title='Accumulated absolute remainder',xlabel='t (s)',ylabel='J/m')
    axes[2].bar(selected,[results[k]['seconds'] for k in selected]);axes[2].tick_params(axis='x',rotation=35);axes[2].set(title='Bounded run wall time',ylabel='seconds')
    for ax in axes[:2]:ax.legend(fontsize=7)
    fig.savefig(out/'tip-energy-cost.png',dpi=170);plt.close(fig)
