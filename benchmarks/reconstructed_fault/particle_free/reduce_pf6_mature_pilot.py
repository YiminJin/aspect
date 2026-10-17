#!/usr/bin/env python3
"""Reduce the bounded mature pilot matrix, retaining failed attempts and accuracy limits."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
from analyze_pf6_mature_pilot import compare,physical_states,merged,replicated
from analyze_pf6_mature_resolution import UPPER,NORMAL,partitions,SIGMA0,TAU0,TANGENT
from sample_pf6_mature_dg import sample
ROOT=Path(__file__).resolve().parents[3]
FUNCTIONAL=['dg-life-r1-final','dg-life-r2-owned','dg-life-r4-owned','dg-r1-restart-reference',
 'dg-r1-restart-split','dg-r1-restart-second-leg','dg-r1-to-r4-restart','dg-r4-to-r2-restart',
 'dg-r4-restart-reference','dg-outer-cutback-r2-final','dg-nonlinear-cutback-r4',
 'dg-cutback-matching-time','dg-one-owner-candidate']


def particle_state(run,step):
    paths=list(run.glob(f'audit_particles_{step}_rank*.csv'))
    assert paths,run
    data=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1,ndmin=2) for p in paths])
    assert len(np.unique(data[:,0]))==len(data)
    order=np.lexsort((data[:,2],data[:,1]));return data[order]


def particle_compare(first,second,step,identical_ids):
    a=particle_state(first,step);b=particle_state(second,step)
    assert a.shape==b.shape and np.array_equal(a[:,1:3],b[:,1:3])
    if identical_ids:assert np.array_equal(a[:,0],b[:,0])
    return dict(points=len(a),coordinate_Linf_m=0.,IDs_identical=bool(np.array_equal(a[:,0],b[:,0])),
                property_Linf=[float(x) for x in np.max(abs(a[:,3:]-b[:,3:]),axis=0)])


def paired_time_contrast(fine,half_fine,coarse,half_coarse,step,profile):
    """Change of signed spatial differences; never subtract scalar norms."""
    states=[]
    for folder,n in [(fine,step),(half_fine,2*step)]:
        states.append(merged(folder,f'authoritative-stress-{n}.csv',('cell','qp')))
    x=np.column_stack((states[0]['x'],states[0]['y']))
    states.extend([sample(coarse,step,x),sample(half_coarse,2*step,x)])
    delta={k:(states[1][k]-states[3][k])-(states[0][k]-states[2][k])
           for k in ('tau_xx','tau_yy','tau_zz','tau_xy','p')}
    square=sum(delta[k]**2 for k in ('tau_xx','tau_yy','tau_zz'))+2*delta['tau_xy']**2
    h=np.interp(abs((x-UPPER)@NORMAL),profile['r_m'],profile['h'],right=0.)
    out={}
    profiles=[read(folder/f'coupled-profile-{n}.csv') for folder,n in
              [(fine,step),(half_fine,2*step),(coarse,step),(half_coarse,2*step)]]
    weak=[replicated(folder,f'surface-weak-{n}.csv') for folder,n in
          [(fine,step),(half_fine,2*step),(coarse,step),(half_coarse,2*step)]]
    for region,mask in partitions((x-UPPER)@TANGENT).items():
        entry={}
        for name,w in [('whole',states[0]['JxW']),('localized',states[0]['JxW']*h)]:
            w=w*mask
            entry[name+'_tensor_RMS_Pa']=float(np.sqrt(np.sum(w*square)/np.sum(w)))
            entry[name+'_Cauchy_RMS_Pa']=float(np.sqrt(np.sum(w*(square+3*delta['p']**2))/np.sum(w)))
        smask=partitions(profiles[0]['s_m'])[region]
        for field,bg in [('shear',TAU0),('raw_normal',SIGMA0),('filtered_normal',SIGMA0)]:
            projection=[q[field+'_Pa'] for q in profiles]
            d=(projection[1]-projection[3])-(projection[0]-projection[2])
            entry[field+'_projection_Linf_Pa']=float(max(abs(d[smask])))
            values=[q[field]-bg*(q['mass_diagonal']+q['mass_edge']+np.r_[0.,q['mass_edge'][:-1]]) for q in weak]
            d=(values[1]-values[3])-(values[0]-values[2])
            entry[field+'_own_mass_perturbation_Linf_Pa_m']=float(max(abs(d[smask])))
        out[region]=entry
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=a.directory;run=b/'current/runs';baseline=ROOT/'build-particle-free/mature-resolution/runs/dg-ff/run'
    physical=read(ROOT/'build-particle-free/mature-resolution/inputs-lf/physical-profile.csv')
    result=dict(functional_checks_passed=True,strict_reference_match_passed=False,production_default_changed=False,
      interpretation='DG is an experimental candidate; neither DG nor particles is an accuracy reference.',
      accuracy_limitations=['Spatial stress is not converged.','Endpoint temporal load uncertainty remains comparable to the fixed-profile mechanical change.',
       'Full-geometry mechanics, costs and weakening evolution have not been qualified by the initialization-only check.'])
    records={}
    for directory in (b/'runs',run):
        for record in directory.glob('*/record.json'):
            d=json.loads(record.read_text());records[record.parent.name]={key:d[key] for key in ('exit_code','seconds','max_rss_KiB','ranks','artifacts')}
            for relative,digest in d['artifacts'].items():
                assert hashlib.sha256((record.parent/Path(relative).name).read_bytes()).hexdigest()==digest
            assert hashlib.sha256((record.parent/'input.prm').read_bytes()).hexdigest()==d['input_sha256']
    for name in FUNCTIONAL+['dg-fixed-ff-h8-final','dg-fixed-ff-h8-dt100','dg-ff-dt100-final','dg-ff-dt50-final']:
        assert records[name]['exit_code']==0,name
    result['runs']=records
    result['all_recorded_native_attempt_seconds']=sum(r['seconds'] for r in records.values())
    result['functional_native_seconds']=sum(records[name]['seconds'] for name in FUNCTIONAL)
    result['fixed_FF']=json.loads((b/'fixed-ff-field-verification.json').read_text())
    for name in ['diagnostic-quadrature','initial-profile-error','temporal-initialization']:
        result[name]=json.loads((b/(name+'.json')).read_text())
    result['surface_initialization']=json.loads((b/'surface-independent.json').read_text())
    result['same_build_physics_regression']=json.loads((b/'fresh-regression.json').read_text());assert all(result['same_build_physics_regression'].values())
    result['comparisons']={}
    for name in ['fixed-ff-final','fixed-ff-dt100-final','fine-time-half','mpi-2-vs-1','mpi-4-vs-1','restart-serial','restart-1-to-4','restart-4-to-2','rejection-outer','rejection-nonlinear']:
        result['comparisons'][name]=json.loads((b/(name+'.json')).read_text())
    result['temporal']={}
    for k in (1,2,3):
        half=compare(run/'dg-ff-dt100-final/run',2*k,baseline,k,physical)
        quarter=compare(run/'dg-ff-dt50-final/run',4*k,run/'dg-ff-dt100-final/run',2*k,physical)
        entry=dict(time_s=200000*k,dt_to_half=half,half_to_quarter=quarter,orders={})
        for key,x,y in [('localized_tensor',half['stress']['all']['localized_RMS_Pa'],quarter['stress']['all']['localized_RMS_Pa'])]+[(key,half['surface']['all'][key],quarter['surface']['all'][key]) for key in ('shear_Pa','raw_normal_Pa','filtered_normal_Pa')]:
            order=math.log2(x/y);entry['orders'][key]=dict(order=order,finest_Richardson_estimate=y/(2**order-1))
        result['temporal'][k]=entry
    result['paired_mechanical_time_control']={}
    for k in (1,2,3):
        result['paired_mechanical_time_control'][k]=paired_time_contrast(
            run/'dg-fixed-ff-h8-final/run',run/'dg-fixed-ff-h8-dt100/run',baseline,
            run/'dg-ff-dt100-final/run',k,physical)
    fine_records=[json.loads((run/name/'record.json').read_text()) for name in ('dg-fixed-ff-h8-final','dg-fixed-ff-h8-dt100')]
    result['fine_step_zero_identical']={name:fine_records[0]['csv_hashes'][name]==fine_records[1]['csv_hashes'][name]
      for name in fine_records[0]['csv_hashes'] if name.endswith('-0.csv') and name.startswith(('physical-fe','authoritative-stress','dg-coefficients','coupled-profile','surface-weak'))}
    assert all(result['fine_step_zero_identical'].values())
    result['particle_property_order']=['crack_driving_force','initial theta_initial','initial strengthening','maxwell xx','maxwell yy','maxwell xy']
    result['particle_checks']={}
    pairs=[('restart_serial','dg-r1-restart-second-leg','dg-r1-restart-reference',5,True),
           ('restart_1_to_4','dg-r1-to-r4-restart','dg-r1-restart-reference',5,True),
           ('restart_4_to_2','dg-r4-to-r2-restart','dg-r4-restart-reference',6,True),
           ('outer_cutback','dg-outer-cutback-r2-final','dg-cutback-matching-time',4,True),
           ('nonlinear_cutback','dg-nonlinear-cutback-r4','dg-cutback-matching-time',4,False)]
    for label,first,second,step,ids in pairs:
        result['particle_checks'][label]=particle_compare(run/first/'run',run/second/'run',step,ids)
    stationarity={}
    for name in ['dg-life-r4-owned','dg-outer-cutback-r2-final','dg-nonlinear-cutback-r4','dg-cutback-matching-time']:
        end=3 if name=='dg-life-r4-owned' else 4
        initial=particle_state(run/name/'run',0);final=particle_state(run/name/'run',end)
        assert np.array_equal(initial[:,:3],final[:,:3])
        summary=read(run/name/'run/particle_summary.csv')
        assert np.all(summary['births_since_backup']==0) and np.all(summary['removed_or_exited_since_backup']==0)
        stationarity[name]=dict(IDs_and_coordinates_bit_identical=True,maximum_motion_m=0.,births=0,deletions=0,points=len(final))
    result['stationarity']=dict(integration_scheme='BP3 stationary samples',freeze_environment='unset',measured=stationarity,
      note='Fixed point samples remain diagnostic/publication dependencies; no domain construction or domain reads enter the homogeneous mechanical branch.')
    result['cutback_clocks']={}
    for name in ['dg-outer-cutback-r2-final','dg-nonlinear-cutback-r4']:
        files=list((run/name/'run').glob('rank-*-outer-cutback.csv'))
        rows=[list(csv.DictReader(file.open())) for file in files];assert all(row==rows[0] for row in rows)
        assert len(rows[0])==1 and rows[0][0]['publications']=='1' and rows[0][0]['accepted_unchanged']=='1'
        result['cutback_clocks'][name]=rows[0][0]
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('functional_checks_passed','all_recorded_native_attempt_seconds','functional_native_seconds','stationarity')},indent=2))
if __name__=='__main__':main()
