#!/usr/bin/env python3
"""Native near-tip state/margin reduction; execute after numerical ranks exit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-near-tip-mpl')
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BUILD=ROOT/'build-particle-free/stage4-near-tip';RUNS=ROOT/'build-particle-free/bulk-tip/runs'
OUT=HERE.parent/'evidence/stage4_near_tip';OUT.mkdir(exist_ok=True);H=1/32
POSITIONS=[1.95,2.,2.05,5.5,5.75,5.9,6.,6.05]
def table(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def output(name):return RUNS/('stage4-near-tip-'+name)/'output'
def scan(out,step):
    files=sorted(out.glob(f'preload-scan-{step:04d}-r*.csv'));assert len(files)==4
    a=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1) for p in files]);a=a[np.lexsort((a[:,2],a[:,0],a[:,1]))]
    assert len(a)==128*256*9;return a
def tent(a,center):
    # Nonnegative Bernstein coefficients sampled from a tent. It is exactly
    # piecewise Q1 when the tent's breaks coincide with the cell faces.
    x=a[:,0]+(a[:,2]%3-1)*H/2;y=a[:,1]+(a[:,2]//3-1)*H/2
    return np.maximum(0,1-abs(x-center)/.0625)*np.maximum(0,1-abs(y-2)/.125)
def masks(a):
    x,y=a[:,:2].T;intact=np.repeat(np.max(abs(a[:,3].reshape(-1,9)),axis=1)==0,9)
    return dict(right_tip=(x==6+H/2)&(abs(y-2)<H),left_tip=(x==2-H/2)&(abs(y-2)<H),
                right_cap=(x>=6)&(x<6.15625)&(abs(y-2)<.125),
                left_cap=(x<=2)&(x>1.84375)&(abs(y-2)<.125),
                right_intact=intact&(x>=6.3125)&(x<6.5)&(abs(y-2)<.125),
                left_intact=intact&(x<=1.6875)&(x>1.5)&(abs(y-2)<.125),
                all_intact=intact,remote=(abs(x-4)>3)|(abs(y-2)>1))
def measure(a,regions):
    net=(a[:,7]-a[:,5])/a[:,4];row={}
    for key,mask in regions.items():
        ids=np.flatnonzero(mask);i=ids[np.argmax(net[mask])]
        row[key+'_max_net']=float(net[i]);row[key+'_positive_rows']=int(np.sum(net[mask]>1e-8))
    for side,centers in [('right',[6.03125,6.09375,6.375]),('left',[1.96875,1.90625,1.625])]:
        for name,center in zip(['tip_probe','ahead_probe','intact_probe'],centers):
            chi=tent(a,center);row[side+'_'+name]=float(chi@(a[:,7]-a[:,5])/(chi@a[:,4]))
    for x in POSITIONS:
        chi=tent(a,x);mass=chi@a[:,4]
        row[f'probe_driving_x{x:g}']=float(chi@a[:,7]/mass)
        row[f'probe_fracture_x{x:g}']=float(chi@a[:,5]/mass)
    return row
def initial():
    status=json.loads((BUILD/'preflight-finished.json').read_text());assert status['complete']
    a=scan(output('initial'),0);oldout=RUNS/'stage4-preload-near-weakening/output';old=scan(oldout,0)
    assert np.array_equal(a[:,:6],old[:,:6]);regions=masks(a)
    record=dict(new=measure(a,regions),old=measure(old,regions),gamma=.000365)
    for key in ('right_tip','left_tip'):
        ids=np.flatnonzero(regions[key]);i=ids[np.argmax((a[ids,7]-a[ids,5])/a[ids,4])]
        record[key+'_critical_row']=a[i,:4].tolist()
    rates=table(output('initial')/'pilot-sections-0000.csv');before=table(oldout/'pilot-sections-0000.csv')
    record['rate_samples']={str(x):dict(old=float(np.interp(x,before['x'],before['rate_stage'])),new=float(np.interp(x,rates['x'],rates['rate_stage'])),old_resistance=float(np.interp(x,before['x'],before['resistance_stage'])),new_resistance=float(np.interp(x,rates['x'],rates['resistance_stage']))) for x in POSITIONS}
    record['initial_integrated_rates']=dict(old=float(np.sum(before['ds']*before['rate_stage'])),new=float(np.sum(rates['ds']*rates['rate_stage'])))
    (OUT/'initial.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savetxt(OUT/'actual-initial-rates.csv',np.c_[rates['x'],before['rate_stage'],rates['rate_stage']],delimiter=',',header='x,old_initial_V,new_initial_V',comments='',fmt='%.17g')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax in axs:
        ax.plot(before['x'],1e3*before['rate_stage'],label='Old, actual t=0');ax.plot(rates['x'],1e3*rates['rate_stage'],label='New, actual t=0')
        ax.axvspan(5.5,6,color='gray',alpha=.12);ax.axvline(6,color='k',ls=':',label='Age perturbation ends')
        ax.set(xlabel='Section x (m)',ylabel='Initial slip rate (mm/s)');ax.legend(fontsize=8)
    axs[0].set(xlim=(1.5,6.5),title='Actual t=0, weakening law, same elastic preload')
    axs[1].set(xlim=(5.4,6.2),title='Actual sliding persists through the formed taper')
    for ext in ('png','pdf'):fig.savefig(OUT/f'initial-rates.{ext}',dpi=170)
    print(json.dumps(record,indent=2))
def pair(family='fixed'):
    status=json.loads((BUILD/f'{family}-finished.json').read_text());assert status['complete']
    a0=scan(output(f'{family}-weakening'),0);regions=masks(a0);histories={};fields={};records={}
    initial_net=(a0[:,7]-a0[:,5])/a0[:,4];fixed_rows={}
    for side in ('right','left'):
        indices=np.flatnonzero(regions[side+'_tip']);fixed_rows[side]=indices[np.argmax(initial_net[indices])]
    for law in ('weakening','neutral'):
        out=output(f'{family}-{law}');summary=table(out/'pilot-summary.csv');energy=table(out/'bulk-accepted.csv');rows=[];nets=[]
        for state in summary:
            step=int(state['step']);a=scan(out,step);net=(a[:,7]-a[:,5])/a[:,4];nets.append(net)
            assert np.array_equal(a[:,:3],a0[:,:3])
            if family=='fixed':assert np.array_equal(a[:,:4],a0[:,:4])
            if step==0:assert np.array_equal(a[:,3],a0[:,3])
            r={key:float(state[key]) for key in summary.dtype.names};r['step']=step;r.update(measure(a,regions))
            for side,index in fixed_rows.items():r[side+'_initial_row_net']=float(net[index])
            sections=table(out/f'pilot-sections-{step:04d}.csv')
            r['mean_legacy_window_resistance']=r.pop('mean_patch_resistance')
            for key,lo,hi in [('support',3.,6.),('right_taper',5.5,6.),('left_taper',2.,2.5)]:
                inside=(sections['x']>lo)&(sections['x']<hi);weights=sections['ds'][inside]
                r[key+'_integrated_rate']=float(np.sum(weights*sections['rate_stage'][inside]))
                r[key+'_mean_resistance']=float(np.sum(weights*sections['resistance_stage'][inside])/np.sum(weights))
            for x in POSITIONS:
                for key in ['rate_stage','slip_endpoint','resistance_stage']:
                    r[f'{key}_x{x:g}']=float(np.interp(x,sections['x'],sections[key]))
            r.update(work_error=0.,added_fracture=0.,mechanics=0.)
            if step:
                e=energy[energy['step']==step][0];assert int(e['publications'])==step
                r.update(work_error=float(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']),mechanics=float(e['mechanical_residual']))
                reg=table(out/'release-regions.csv');r['added_fracture']=float(sum(reg[reg['step']==step]['added_fracture_row_partition']))
            rows.append(r)
        histories[law]=rows;fields[law]=np.array(nets)
        command=json.loads((out.parent/'command.json').read_text())
        signed=table(out/f'tip-sections-{rows[-1]["step"]:04d}.csv')
        pos=float(sum(signed['ds']*signed['positive_w']));neg=float(sum(signed['ds']*signed['negative_w']))
        records[law]=dict(initial=rows[0],final=rows[-1],seconds=command['seconds'],exit_code=command['exit_code'],signed_work_error=sum(r['work_error'] for r in rows),absolute_work_error=sum(abs(r['work_error']) for r in rows),absolute_external_work=float(sum(abs(energy['Wext']))),maximum_mechanics=max(r['mechanics'] for r in rows),positive_rate_integral=pos,negative_rate_integral=neg,cancellation_fraction=1-abs(pos+neg)/(pos-neg))
        power=table(out/'tip-power.csv')
        records[law].update(maximum_power_adjoint_error=float(max(abs(power['power_adjoint_error']))),friction_work=float(sum(power['friction'])),radiation_work=float(sum(power['radiation'])),total_section_work=float(sum(energy['Df'])))
        assert np.max(abs(signed['s']-sections['x']))<1e-12
        records[law]['regional_signed_deformation']={}
        for key,lo,hi in [('right_taper',5.5,6.),('left_taper',2.,2.5)]:
            inside=(signed['s']>lo)&(signed['s']<hi)
            positive=float(sum(signed['ds'][inside]*signed['positive_w'][inside]))
            negative=float(sum(signed['ds'][inside]*signed['negative_w'][inside]))
            positive_power=float(sum(signed['ds'][inside]*signed['force'][inside]*signed['positive_w'][inside]))
            negative_power=float(sum(signed['ds'][inside]*signed['force'][inside]*signed['negative_w'][inside]))
            records[law]['regional_signed_deformation'][key]=dict(positive=positive,negative=negative,cancellation_fraction=1-abs(positive+negative)/(positive-negative) if positive>negative else 0.,positive_power=positive_power,negative_power=negative_power)
        with (OUT/f'{family}-{law}-history.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    assert len(histories['weakening'])==len(histories['neutral'])==9
    delta=fields['weakening']-fields['neutral'];difference={}
    assert np.max(abs(delta[0]))<1e-9,'The two laws must start from matching mechanical margins.'
    for key,mask in regions.items():
        difference[key]=dict(min=float(delta[1:,mask].min()),max=float(delta[1:,mask].max()),final_min=float(delta[-1,mask].min()),final_max=float(delta[-1,mask].max()),weakening_positive_control_negative=int(np.sum((fields['weakening'][:,mask]>0)&(fields['neutral'][:,mask]<0))))
    result=dict(laws=records,row_differences=difference,initial_law_margin_mismatch=float(np.max(abs(delta[0]))),fixed_row_locations={side:a0[i,:4].tolist() for side,i in fixed_rows.items()})
    result['right_tip_decision_metrics']={}
    for side in ('right','left'):
        key=side+'_initial_row_net';w=histories['weakening'];n=histories['neutral']
        change=n[-1][key]-n[0][key];remaining=-n[-1][key];excess=w[-1][key]-n[-1][key]
        result['right_tip_decision_metrics'][side]=dict(signed_law_difference=excess,control_history_change=change,control_remaining_deficit=remaining,
            difference_fraction_of_remaining_deficit=abs(excess/remaining) if remaining>0 else None,
            difference_fraction_of_control_change=abs(excess/change) if change else None)
    (OUT/f'{family}-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/f'{family}-margins.npz',locations=a0[:,:3],initial_phase=a0[:,3],weakening=fields['weakening'],neutral=fields['neutral'])
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        rows=histories[law];t=np.array([r['stage_time'] for r in rows])*1e3;label='Weakening' if law=='weakening' else 'Matched b=a'
        for x,style in [(5.75,'-'),(6.,'--')]:axs[0,0].plot(t,[1e3*r[f'rate_stage_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
        for key,style in [('right','-'),('left','--')]:axs[0,1].plot(t,[r[key+'_initial_row_net'] for r in rows],style,color=color,label=label+', '+key)
        axs[1,0].plot(t,[r['right_ahead_probe'] for r in rows],color=color,label=label)
    t=np.array([r['stage_time'] for r in histories['weakening']])*1e3
    for key,style in [('right','-'),('left','--')]:
        d=np.array([w[key+'_initial_row_net']-n[key+'_initial_row_net'] for w,n in zip(histories['weakening'],histories['neutral'])]);axs[1,1].plot(t,d,style,label=key+' tip')
    axs[0,0].set(title='Local sliding, fixed physical positions',ylabel='Stage V (mm/s)')
    axs[0,1].axhline(0,color='k',lw=.6);axs[0,1].set(title='Initially limiting rows, fixed positions',ylabel='Net damage driving (Pa)')
    axs[1,0].set(title='Same immediately-ahead right probe',ylabel='Mean net driving (Pa)')
    axs[1,1].axhline(0,color='k',lw=.6);axs[1,1].set(title='Weakening − b=a, signed tip difference',ylabel='Difference (Pa)')
    for ax in axs.flat:ax.set_xlabel('Stage time (ms)');ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{family}-histories.{ext}',dpi=170)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        rows=histories[law];stage=np.array([r['stage_time'] for r in rows])*1e3;end=np.array([r['time'] for r in rows])*1e3
        label='Weakening' if law=='weakening' else 'Matched b=a'
        for x,style in [(5.75,'-'),(6.,'--')]:
            axs[0].plot(end,[1e6*r[f'slip_endpoint_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
            axs[1].plot(stage,[r[f'resistance_stage_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
    axs[0].set(xlabel='Endpoint time (ms)',ylabel='Cumulative local slip (µm)',title='Right-taper slip at fixed positions')
    axs[1].set(xlabel='Stage time (ms)',ylabel='Section resistance (Pa)',title='Same locations and common stage times')
    for ax in axs:ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{family}-local-slip.{ext}',dpi=170)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initial','fixed','free']);a=p.parse_args()
    initial() if a.mode=='initial' else pair(a.mode)
