#!/usr/bin/env python3
"""Native near-tip state/margin reduction; execute after numerical ranks exit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-near-tip-mpl')
from pathlib import Path
import argparse,csv,json,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BUILD=ROOT/'build-particle-free/stage4-preparation';RUNS=ROOT/'build-particle-free/bulk-tip/runs'
STATUS_ROOT=BUILD
CACHE_MARGINS=False
OUT=HERE.parent/'evidence/stage4_preparation';OUT.mkdir(exist_ok=True);H=1/32
PREPARATION=json.loads((OUT/'preparation.json').read_text());FRONT=PREPARATION['prepared_front'];DC=3.2211126126299625e-7
SIGN_TOL=2e-7 # Kf=2 Pa times the unchanged native phase KKT tolerance 1e-7.
POSITIONS=[FRONT['left_x']-.05,FRONT['left_x'],FRONT['left_x']+.05,5.5,5.75,5.9,6.,FRONT['right_x'],FRONT['right_x']+.05]
def table(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def output(name):return RUNS/('stage4-preparation-'+name)/'output'
def scan(out,step):
    files=sorted(out.glob(f'preload-scan-{step:04d}-r*.csv*'));assert len(files)==4
    a=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1) for p in files]);a=a[np.lexsort((a[:,2],a[:,0],a[:,1]))]
    assert len(a)==128*256*9;return a
def tent(a,center):
    # Nonnegative Bernstein coefficients sampled from a tent. It is exactly
    # piecewise Q1 when the tent's breaks coincide with the cell faces.
    x=a[:,0]+(a[:,2]%3-1)*H/2;y=a[:,1]+(a[:,2]//3-1)*H/2
    return np.maximum(0,1-abs(x-center)/.0625)*np.maximum(0,1-abs(y-2)/.125)
def masks(a):
    x,y=a[:,:2].T;intact=np.repeat(np.max(abs(a[:,3].reshape(-1,9)),axis=1)==0,9)
    return dict(right_tip=(x==(int(FRONT['right_x']/H)+.5)*H)&(abs(y-FRONT['right_y'])<H),left_tip=(x==(int(FRONT['left_x']/H)+.5)*H)&(abs(y-FRONT['left_y'])<H),
                right_cap=(x>=6)&(x<6.15625)&(abs(y-2)<.125),
                left_cap=(x<=2)&(x>1.84375)&(abs(y-2)<.125),
                right_intact=intact&(x>=6.3125)&(x<6.5)&(abs(y-2)<.125),
                left_intact=intact&(x<=1.6875)&(x>1.5)&(abs(y-2)<.125),
                all_intact=intact,remote=(abs(x-4)>3)|(abs(y-2)>1))
def profile_direction(a,side):
    b=a[:,3].reshape(128,256,9);shift=np.zeros_like(b)
    if side=='right':shift[:,1:]=b[:,:-1]
    else:shift[:,:-1]=b[:,1:]
    chi=np.maximum(0,shift-b).ravel()
    window=np.maximum(0,1-((a[:,0]-FRONT[side+'_x'])/.5)**2)**2
    return chi*window
def measure(a,regions,profiles):
    net=(a[:,7]-a[:,5])/a[:,4];row={}
    for key,mask in regions.items():
        ids=np.flatnonzero(mask);i=ids[np.argmax(net[mask])]
        row[key+'_max_net']=float(net[i]);row[key+'_positive_rows']=int(np.sum(net[mask]>SIGN_TOL))
    for side,centers in [('right',[6.03125,6.09375,6.375]),('left',[1.96875,1.90625,1.625])]:
        for name,center in zip(['tip_probe','ahead_probe','intact_probe'],centers):
            chi=tent(a,center);row[side+'_'+name]=float(chi@(a[:,7]-a[:,5])/(chi@a[:,4]))
    for x in POSITIONS:
        chi=tent(a,x);mass=chi@a[:,4]
        row[f'probe_driving_x{x:g}']=float(chi@a[:,7]/mass)
        row[f'probe_fracture_x{x:g}']=float(chi@a[:,5]/mass)
    for side,chi in profiles.items():
        mass=chi@a[:,4];work=chi@(a[:,7]-a[:,5])
        row[side+'_profile_net_Pa']=float(work/mass)
        row[side+'_profile_net_directional_work']=float(work)
        row[side+'_profile_fracture_cost']=float(chi@a[:,5])
        row[side+'_profile_mechanical_driving']=float(chi@a[:,7])
    return row
def initial_match(family):
    arrays=[];forces=[];phase=[];sections=[]
    for law in ('weakening','neutral'):
        out=output(f'{family}-{law}');parts=[]
        for rank in range(4):
            p=out/f'prepared-history-r{rank}.bin';p=p if p.exists() else p.with_suffix('.bin.gz')
            raw=p.read_bytes();raw=gzip.decompress(raw) if p.suffix=='.gz' else raw
            parts.append(np.frombuffer(raw,dtype=np.float64).reshape(-1,12))
            if law=='neutral':
                assert (out/f'prepared-history-r{rank}.csv').read_bytes()==(output(f'{family}-weakening')/f'prepared-history-r{rank}.csv').read_bytes()
        section=table(out/'pilot-sections-0000.csv');sections.append(section)
        arrays.append(np.concatenate(parts));forces.append(section['resistance_stage'])
        phase.append(scan(out,0)[:,3])
    assert np.array_equal(*phase)
    assert np.array_equal(sections[0]['age_endpoint'],sections[1]['age_endpoint'])
    assert all(np.max(abs(section['slip_endpoint']))==0. for section in sections)
    difference=arrays[0]-arrays[1]
    stress_error=float(abs(difference[:,:6]).max());strain_error=float(abs(difference[:,6:]).max());force_error=float(abs(forces[0]-forces[1]).max())
    relative_stress=stress_error/float(abs(arrays[0][:,:6]).max());relative_strain=strain_error/float(abs(arrays[0][:,6:]).max())
    assert max(relative_stress,relative_strain)<1e-9 and force_error<1e-9,'Initial physical states differ beyond the declared comparison tolerance.'
    return dict(exact_phase_equality=True,exact_age_equality=True,zero_initial_slip=True,maximum_initial_rate_difference_after_common_age_perturbation=float(np.max(abs(sections[0]['rate_stage']-sections[1]['rate_stage']))),maximum_stress_component_difference_Pa=stress_error,maximum_cohesive_strain_component_difference=strain_error,maximum_section_resistance_difference_Pa=force_error,relative_maximum_stress_difference=relative_stress,relative_maximum_cohesive_strain_difference=relative_strain,comparison_tolerance=1e-9,tensor_order='tau then A; xx,xy,xz,yy,yz,zz; all native quadrature points; matching rank/CellId maps verified')
def pair(family='fixed'):
    status=json.loads((STATUS_ROOT/f'{family}-finished.json').read_text());assert status['complete']
    a0=scan(output(f'{family}-weakening'),0);regions=masks(a0);profiles={side:profile_direction(a0,side) for side in ['right','left']};histories={};fields={};records={}
    initial_net=(a0[:,7]-a0[:,5])/a0[:,4];fixed_rows={};outer_rows={}
    for side in ('right','left'):
        indices=np.flatnonzero(regions[side+'_tip']);fixed_rows[side]=indices[np.argmax(initial_net[indices])]
        cell=(a0[:,0]==(int(FRONT[side+'_x']/H)+.5)*H)&(a0[:,1]==(int(FRONT[side+'_y']/H)+.5)*H)
        indices=np.flatnonzero(cell);assert len(indices)==9
        outer_rows[side]=indices[np.argmax(initial_net[indices])]
    for law in ('weakening','neutral'):
        out=output(f'{family}-{law}');summary=table(out/'pilot-summary.csv');energy=table(out/'bulk-accepted.csv');rows=[];nets=[]
        for state in summary:
            step=int(state['step']);a=scan(out,step);net=(a[:,7]-a[:,5])/a[:,4];nets.append(net)
            assert np.array_equal(a[:,:3],a0[:,:3])
            if family.startswith('fixed'):assert np.array_equal(a[:,:4],a0[:,:4])
            if step==0:assert np.array_equal(a[:,3],a0[:,3])
            r={key:float(state[key]) for key in summary.dtype.names};r['step']=step;r.update(measure(a,regions,profiles))
            for side,index in fixed_rows.items():r[side+'_initial_row_net']=float(net[index])
            for side,index in outer_rows.items():r[side+'_outer_initial_row_net']=float(net[index])
            sections=table(out/f'pilot-sections-{step:04d}.csv')
            r['peak_rate_location_x']=float(sections['x'][np.argmax(abs(sections['rate_stage']))])
            r['mean_legacy_window_resistance']=r.pop('mean_patch_resistance')
            for key,lo,hi in [('support',3.,6.),('right_taper',5.5,6.),('left_taper',2.,2.5)]:
                inside=(sections['x']>lo)&(sections['x']<hi);weights=sections['ds'][inside]
                r[key+'_integrated_rate']=float(np.sum(weights*sections['rate_stage'][inside]))
                r[key+'_mean_resistance']=float(np.sum(weights*sections['resistance_stage'][inside])/np.sum(weights))
            for x in POSITIONS:
                for key in ['rate_stage','slip_endpoint','resistance_stage']:
                    r[f'{key}_x{x:g}']=float(np.interp(x,sections['x'],sections[key]))
            for x in POSITIONS:r[f'slip_over_Dc_x{x:g}']=r[f'slip_endpoint_x{x:g}']/DC
            for side,lo,hi in [('left',2.,2.5),('right',5.5,6.)]:
                positive=negative=None # no physical strain-rate field is assigned by static preparation
                if step:
                    local=table(out/f'tip-sections-{step:04d}.csv');inside=(local['s']>lo)&(local['s']<hi)
                    positive=float(sum(local['ds'][inside]*local['positive_w'][inside]));negative=float(sum(local['ds'][inside]*local['negative_w'][inside]))
                r[side+'_taper_positive_deformation_integral']=positive;r[side+'_taper_negative_deformation_integral']=negative
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
        for key,lo,hi in [('right_taper',5.5,6.),('left_taper',2.,2.5),('right_tip',FRONT['right_x']-.0625,FRONT['right_x']+.0625),('left_tip',FRONT['left_x']-.0625,FRONT['left_x']+.0625)]:
            inside=(signed['s']>lo)&(signed['s']<hi)
            positive=float(sum(signed['ds'][inside]*signed['positive_w'][inside]))
            negative=float(sum(signed['ds'][inside]*signed['negative_w'][inside]))
            positive_power=float(sum(signed['ds'][inside]*signed['force'][inside]*signed['positive_w'][inside]))
            negative_power=float(sum(signed['ds'][inside]*signed['force'][inside]*signed['negative_w'][inside]))
            records[law]['regional_signed_deformation'][key]=dict(positive=positive,negative=negative,cancellation_fraction=1-abs(positive+negative)/(positive-negative) if positive>negative else 0.,positive_power=positive_power,negative_power=negative_power)
        with (OUT/f'{family}-{law}-history.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    common_count=min(len(histories['weakening']),len(histories['neutral']));assert common_count>1
    for law in histories:
        # A graceful wall-time stop can give unequal complete observation
        # windows. Compare the common accepted prefix, retaining each full CSV.
        records[law]['last_accepted']=histories[law][-1]
        histories[law]=histories[law][:common_count];fields[law]=fields[law][:common_count]
        records[law]['comparison_final']=histories[law][-1]
        records[law]['comparison_signed_work_error']=sum(r['work_error'] for r in histories[law])
        records[law]['comparison_absolute_work_error']=sum(abs(r['work_error']) for r in histories[law])
    assert all(w['time']==n['time'] and w['stage_time']==n['stage_time'] for w,n in zip(histories['weakening'],histories['neutral']))
    delta=fields['weakening']-fields['neutral'];difference={}
    assert np.max(abs(delta[0]))<1e-9,'The two laws must start from matching mechanical margins.'
    for key,mask in regions.items():
        difference[key]=dict(min=float(delta[1:,mask].min()),max=float(delta[1:,mask].max()),final_min=float(delta[-1,mask].min()),final_max=float(delta[-1,mask].max()),weakening_positive_control_negative=int(np.sum((fields['weakening'][:,mask]>SIGN_TOL)&(fields['neutral'][:,mask]<-SIGN_TOL))))
    result=dict(prepared_front=FRONT,physical_positions=POSITIONS,profile_directions='positive part of one-cell axial shift of prepared Bernstein field, multiplied by a quartic axial window of half-width .5 m at each prepared tip; fixed throughout the pair',law_parameter_change='b=.020 versus .015 with matched mu, common a=.015 and Dc',initial_physical_state_match=initial_match(family),laws=records,row_differences=difference,initial_law_margin_mismatch=float(np.max(abs(delta[0]))),fixed_row_locations={side:a0[i,:4].tolist() for side,i in fixed_rows.items()})
    result['outer_contour_cell_row_locations']={side:a0[i,:4].tolist() for side,i in outer_rows.items()}
    result['common_windows']={}
    result['common_last_endpoint_time']=histories['weakening'][-1]['time']
    result['common_accepted_intervals']=common_count-1
    result['favorability_threshold_Pa']=SIGN_TOL
    result['first_favorable_direction_brackets']={}
    for law,rows in histories.items():
        result['first_favorable_direction_brackets'][law]={}
        for key in ['right_initial_row_net','right_outer_initial_row_net','right_tip_max_net','right_profile_net_Pa','right_ahead_probe']:
            positive=[i for i,r in enumerate(rows) if r[key]>SIGN_TOL]
            i=positive[0] if positive else None
            result['first_favorable_direction_brackets'][law][key]=None if i is None else dict(previous_stage_time=rows[max(0,i-1)]['stage_time'],first_stage_time=rows[i]['stage_time'],previous_net_Pa=rows[max(0,i-1)][key],first_net_Pa=rows[i][key],note='infinitesimal direction, not connected extension')
    for step in [8,16,24,32,48,56,60,64,65,66,68,69,70,72,76]:
        w=[r for r in histories['weakening'] if r['step']==step];n=[r for r in histories['neutral'] if r['step']==step]
        if w and n:
            w,n=w[0],n[0];key='right_initial_row_net';excess=w[key]-n[key];deficit=-n[key]
            result['common_windows'][str(step)]=dict(endpoint_time=w['time'],stage_time=w['stage_time'],right_row_law_difference_Pa=excess,control_right_deficit_Pa=deficit,excess_fraction_of_remaining_deficit=abs(excess/deficit) if deficit>0 else None,weakening=w,neutral=n)
            key='right_outer_initial_row_net';outer_excess=w[key]-n[key];outer_deficit=-n[key]
            shared_change=n[key]-histories['neutral'][0][key]
            result['common_windows'][str(step)]['outer_tip_cell']=dict(signed_law_difference_Pa=outer_excess,control_remaining_deficit_Pa=outer_deficit,control_change_from_initial_Pa=shared_change,law_difference_fraction_of_remaining_deficit=abs(outer_excess/outer_deficit) if outer_deficit>0 else None,law_difference_fraction_of_control_change=abs(outer_excess/shared_change) if shared_change else None,note='native coupled stage gradient at the fixed initially least-resistant row in the outer-contour cell; active released rows equilibrate toward zero')
    result['right_tip_decision_metrics']={}
    for side in ('right','left'):
        key=side+'_initial_row_net';w=histories['weakening'];n=histories['neutral']
        change=n[-1][key]-n[0][key];remaining=-n[-1][key];excess=w[-1][key]-n[-1][key]
        result['right_tip_decision_metrics'][side]=dict(signed_law_difference=excess,control_history_change=change,control_remaining_deficit=remaining,
            difference_fraction_of_remaining_deficit=abs(excess/remaining) if remaining>0 else None,
            difference_fraction_of_control_change=abs(excess/change) if change else None)
    (OUT/f'{family}-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    if CACHE_MARGINS:
        cache=BUILD/'analysis-cache';cache.mkdir(exist_ok=True)
        np.savez_compressed(cache/f'{family}-margins.npz',locations=a0[:,:3],initial_phase=a0[:,3],weakening=fields['weakening'],neutral=fields['neutral'])
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        rows=histories[law];t=np.array([r['stage_time'] for r in rows])*1e3;label='Weakening' if law=='weakening' else 'Matched b=a'
        for x,style in [(5.75,'-'),(FRONT['right_x'],'--')]:axs[0,0].plot(t,[1e3*r[f'rate_stage_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
        for key,style in [('right','-'),('left','--')]:axs[0,1].plot(t,[r[key+'_initial_row_net'] for r in rows],style,color=color,label=label+', '+key)
        axs[1,0].plot(t,[r['right_profile_net_Pa'] for r in rows],color=color,label=label)
    t=np.array([r['stage_time'] for r in histories['weakening']])*1e3
    for key,style in [('right','-'),('left','--')]:
        d=np.array([w[key+'_initial_row_net']-n[key+'_initial_row_net'] for w,n in zip(histories['weakening'],histories['neutral'])]);axs[1,1].plot(t,d,style,label=key+' tip')
    axs[0,0].set(title='Local sliding, fixed physical positions',ylabel='Stage V (mm/s)')
    axs[0,1].axhline(0,color='k',lw=.6);axs[0,1].set(title='Initially limiting rows, fixed positions',ylabel='Net damage driving (Pa)')
    axs[1,0].set(title='Prepared-profile forward damage direction',ylabel='Mean net driving (Pa)')
    axs[1,1].axhline(0,color='k',lw=.6);axs[1,1].set(title='Weakening − b=a, signed tip difference',ylabel='Difference (Pa)')
    for ax in axs.flat:ax.set_xlabel('Stage time (ms)');ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{family}-histories.{ext}',dpi=170)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for law,color in [('weakening','tab:red'),('neutral','tab:blue')]:
        rows=histories[law];stage=np.array([r['stage_time'] for r in rows])*1e3;end=np.array([r['time'] for r in rows])*1e3
        label='Weakening' if law=='weakening' else 'Matched b=a'
        for x,style in [(5.75,'-'),(FRONT['right_x'],'--')]:
            axs[0].plot(end,[1e6*r[f'slip_endpoint_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
            axs[1].plot(stage,[r[f'resistance_stage_x{x:g}'] for r in rows],style,color=color,label=f'{label}, x={x:g}')
    axs[0].set(xlabel='Endpoint time (ms)',ylabel='Cumulative local slip (µm)',title='Right-taper slip at fixed positions')
    axs[1].set(xlabel='Stage time (ms)',ylabel='Section resistance (Pa)',title='Same locations and common stage times')
    for ax in axs:ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{family}-local-slip.{ext}',dpi=170)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--view',type=Path);p.add_argument('--cache-margins',action='store_true');a=p.parse_args()
    assert a.mode.isidentifier()
    STATUS_ROOT=BUILD
    CACHE_MARGINS=a.cache_margins
    if a.view:RUNS=a.view.resolve();STATUS_ROOT=RUNS
    pair(a.mode)
