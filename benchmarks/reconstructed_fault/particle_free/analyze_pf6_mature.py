#!/usr/bin/env python3
"""Read pinned mature runs; compare physical profiles and the full 3-D tensor metric."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty evidence: {path}')
    return {k:np.array([r[k] for r in rows],dtype=(str if k=='cell' else float)) for k in rows[0]}


def norm3(data):
    return data['tau_xx']**2+data['tau_yy']**2+data['tau_zz']**2+2*data['tau_xy']**2


def profile_compare(first, second):
    assert np.array_equal(first['s_m'],second['s_m'])
    return {k:float(np.max(np.abs(first[k]-second[k]))) for k in first if k not in ('node','s_m','time_s')}


def stress_compare(first, second):
    assert np.array_equal(first['cell'],second['cell'])
    for k in ('x','y','JxW','phi','chi'):
        if k!='chi':
            assert np.array_equal(first[k],second[k]),k
    delta={k:first[k]-second[k] for k in ('tau_xx','tau_yy','tau_zz','tau_xy')}
    weights=second['JxW']
    result={}
    for name,w in [('whole',weights),('localized',weights*second['chi'])]:
        numerator=float(np.sum(w*norm3(delta)))
        denominator=float(np.sum(w*norm3(second)))
        result[name+'_RMS_Pa']=float(np.sqrt(numerator/np.sum(w)))
        result[name+'_relative_L2']=float(np.sqrt(numerator/denominator)) if denominator else 0.
    result['max_full_tensor_Pa']=float(np.sqrt(np.max(norm3(delta))))
    result['p_max_Pa']=float(np.max(np.abs(first['p']-second['p'])))
    return result


def cut_jumps(data):
    groups={}
    for i,(station,r) in enumerate(zip(data['station'],data['r_m'])):
        groups.setdefault((int(station),round(float(r),8)),[]).append(i)
    maximum=0.;normal_maximum=0.;location=None
    for key,indices in groups.items():
        for i in indices:
            for j in indices:
                d={k:np.array([data[k][i]-data[k][j]]) for k in ('tau_xx','tau_yy','tau_zz','tau_xy')}
                jump=float(np.sqrt(norm3(d)[0]))
                normal_maximum=max(normal_maximum,abs(float(data['normal_Pa'][i]-data['normal_Pa'][j])))
                if jump>maximum:
                    maximum=jump;location={'station':key[0],'r_m':key[1],'cells':[data['cell'][i],data['cell'][j]]}
    return dict(max_full_tensor_jump_Pa=maximum,max_raw_normal_jump_Pa=normal_maximum,worst_tensor_location=location)


def common_cut_compare(left, right, physical_profile):
    def interior(data):
        groups={}
        for i,(station,r) in enumerate(zip(data['station'],data['r_m'])):
            if abs(r/2-round(r/2))<1e-10:
                groups.setdefault((int(station),round(float(r),8)),[]).append(i)
        # Retain the individual face traces in the source CSVs; compare only
        # common interior points, so no averaged trace is silently introduced.
        return {key:ids[0] for key,ids in groups.items() if len(ids)==1}
    a,b=interior(left),interior(right)
    keys=sorted(a.keys() & b.keys())
    ia,ib=[a[k] for k in keys],[b[k] for k in keys]
    difference={k:left[k][ia]-right[k][ib] for k in ('tau_xx','tau_yy','tau_zz','tau_xy')}
    reference={k:right[k][ib] for k in difference}
    r=np.array([key[1] for key in keys])
    h=np.interp(abs(r),physical_profile['r_m'],physical_profile['h'],right=0.)
    out={'common_interior_points':len(keys)}
    for label,w in (('whole',np.ones(len(keys))),('localized',h)):
        denominator=float(sum(w*norm3(reference)));numerator=float(sum(w*norm3(difference)))
        out[label+'_relative_tensor_L2']=float(np.sqrt(numerator/denominator)) if denominator else 0.
        out[label+'_tensor_RMS_Pa']=float(np.sqrt(numerator/sum(w)))
    out['phase_Linf']=float(max(abs(left['phi'][ia]-right['phi'][ib])))
    out['normal_max_Pa']=float(max(abs(left['normal_Pa'][ia]-right['normal_Pa'][ib])))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    runs={f.parent.name:json.loads(f.read_text()) for f in a.directory.glob('*/record.json')}
    result={'runs':{},'comparisons':{}}
    for label,record in runs.items():
        entry={k:record[k] for k in ('exit_code','seconds','max_rss_KiB','arguments','artifacts')}
        folder=a.directory/label/'run'
        profiles=sorted(folder.glob('coupled-profile-*.csv'),key=lambda f:int(f.stem.rsplit('-',1)[1]))
        if profiles:
            initial=read(profiles[0]);final=read(profiles[-1]);checks=read(folder/'coupled-checks.csv')
            entry.update(last_step=int(checks['step'][-1]),last_time_s=float(final['time_s'][0]),
                V_range=[float(min(final['V_m_s'])),float(max(final['V_m_s']))],
                V_change_from_initial_max=float(max(abs(final['V_m_s']-initial['V_m_s']))),
                max_initial_raw_filter_difference_Pa=float(max(abs(initial['raw_normal_Pa']-initial['filtered_normal_Pa']))),
                max_raw_filter_difference_Pa=float(max(abs(final['raw_normal_Pa']-final['filtered_normal_Pa']))),
                max_stress_recurrence_Pa=float(max(checks['stress_error_Pa'])),
                max_allowance_fraction=float(max(checks['error_fraction_of_allowance'])),
                max_Z_recurrence_Pa_m=float(max(checks['Z_error_Pa_m'])),
                fixed_particle_count=int(checks['particles'][0]),
                publications=checks['publications'].astype(int).tolist())
            accepted=read(folder/'accepted_steps.csv')
            entry['max_normalized_nonlinear_residual']=float(max(accepted['normalized_nonlinear_residual']))
            entry['max_surface_RMS_Pa']=float(max(accepted['surface_RMS_Pa']))
            entry['max_Theta_relative_error']=float(max(accepted['Theta_relative_error']))
            entry['all_linear_checks_passed']=bool(np.all(accepted['fresh_linear_checks_passed']==1))
        normalization=folder/'native-normalization.csv'
        if record['exit_code']==0 and normalization.exists() and normalization.stat().st_size:
            entry['normalization']={k:float(v[0]) for k,v in read(normalization).items()}
        paired=[]
        for f in folder.glob('paired-history-*.csv'):
            if record['exit_code']==0 and f.stat().st_size:
                c=read(f)
                paired.append({'path':f.name,'stress_error_Pa':float(max(c['stress_error_Pa'])),
                    'relative_stress_error':float(max(c['stress_error_Pa']/c['stress_scale_Pa'])),
                    'weak_error':float(max(c['weak_error'])),'relative_weak_error':float(max(c['weak_error']/c['weak_scale']))})
        if paired:entry['paired']=paired
        result['runs'][label]=entry
    # Every comparison is declared explicitly, keeping the finite budget visible.
    for lhs,rhs in [('dg-10','fe-10'),('particle-10','fe-10'),('dg-q6','fe-q6'),
                    ('dg-10','dg-q6'),('dg-q6','dg-q8'),('dg-q6','dg-half'),('dg-q6','dg-fine')]:
        lf,rf=a.directory/lhs/'run',a.directory/rhs/'run'
        if lhs not in runs or rhs not in runs:continue
        if runs[lhs]['exit_code'] or runs[rhs]['exit_code']:continue
        c={}
        for f in sorted(lf.glob('coupled-profile-*.csv')):
            d=read(f);time=d['time_s'][0]
            other=None
            for g in rf.glob('coupled-profile-*.csv'):
                e=read(g)
                if e['time_s'][0]==time:other=(g,e);break
            if other:
                g,e=other
                entry={'profiles_max_absolute':profile_compare(d,e)}
                tau0=50e6*.025*np.arcsinh(1e-9/(2e-6)*np.exp((.6+.015*np.log(1e-6/1e-9))/.025))+4624440.*1e-9
                # The native steady strengthening reference includes radiation
                # damping; this experiment uses no fitted background offset.
                for k,bg in (('shear_Pa',tau0),('raw_normal_Pa',e['background_Pa']),('filtered_normal_Pa',e['background_Pa'])):
                    scale=float(max(abs(e[k]-bg)))
                    entry.setdefault('traction_perturbation_scales_Pa',{})[k]=scale
                    entry.setdefault('traction_relative_changes',{})[k]=entry['profiles_max_absolute'][k]/scale if scale else 0.
                stress=lf/('authoritative-stress-'+f.stem.split('-')[-1]+'.csv')
                ref=rf/('authoritative-stress-'+g.stem.split('-')[-1]+'.csv')
                if stress.exists() and ref.exists() and lhs!='dg-fine' and rhs!='dg-fine':
                    entry['stress']=stress_compare(read(stress),read(ref))
                c[str(time)]=entry
        result['comparisons'][lhs+'__'+rhs]=c
    physical_profile=a.directory/'operator-nonproportional/run/prescribed-profile.csv'
    if physical_profile.exists():
        profile=read(physical_profile)
        physical_I=read(physical_profile.with_name('prescribed-normalization.csv'))['I_m'][0]
        result['initialization_reference']={'physical_I_m':float(physical_I),'meshes':{}}
        upper=np.array([-523.1751345948129,1137.25]);normal=np.array([np.sqrt(3)/2,.5])
        for label in ('dg-10','dg-fine'):
            f=a.directory/label/'run/authoritative-stress-0.csv'
            if f.exists() and label in runs and runs[label]['exit_code']==0:
                data=read(f);r=(data['x']-upper[0])*normal[0]+(data['y']-upper[1])*normal[1]
                target=np.interp(abs(r),profile['r_m'],profile['phi'],left=profile['phi'][0],right=0.)
                w=data['JxW'];delta=data['phi']-target
                surface=read(f.with_name('coupled-profile-0.csv'))
                result['initialization_reference']['meshes'][label]={
                  'phase_relative_L2_common_Gauss3_sampling':float(np.sqrt(np.sum(w*delta**2)/np.sum(w*target**2))),
                  'phase_sample_Linf':float(max(abs(delta))),
                  'I_relative_Linf_vs_physical_profile':float(max(abs(surface['I_h_m']-physical_I))/physical_I),
                  'I_range_m':[float(min(surface['I_h_m'])),float(max(surface['I_h_m']))]}
    if 'dg-fine' in runs and runs['dg-fine']['exit_code']==0:
        result['spatial_common_cuts']={}
        physical=read(physical_profile)
        for step in range(4):
            lhs=a.directory/'dg-q6/run'/f'bulk-normal-cuts-{step}.csv'
            rhs=a.directory/'dg-fine/run'/f'bulk-normal-cuts-{step}.csv'
            left,right=read(lhs),read(rhs)
            result['spatial_common_cuts'][str(step)]=common_cut_compare(left,right,physical)
            result['spatial_common_cuts'][str(step)]['coarse_jumps']=cut_jumps(left)
            result['spatial_common_cuts'][str(step)]['fine_jumps']=cut_jumps(right)
    if physical_profile.exists():
        result['same_mesh_common_cuts']={}
        physical=read(physical_profile)
        for label,step in (('dg-q8',3),('dg-half',6)):
            lhs=a.directory/'dg-q6/run/bulk-normal-cuts-3.csv'
            rhs=a.directory/label/'run'/f'bulk-normal-cuts-{step}.csv'
            if lhs.exists() and rhs.exists() and label in runs and runs[label]['exit_code']==0:
                result['same_mesh_common_cuts'][label]=common_cut_compare(read(lhs),read(rhs),physical)
    result['total_native_seconds']=sum(r['seconds'] for r in runs.values())
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'runs':len(runs),'total_native_seconds':result['total_native_seconds']}))


if __name__=='__main__':main()
