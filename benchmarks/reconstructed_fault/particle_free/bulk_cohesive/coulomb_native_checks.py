#!/usr/bin/env python3
"""Read native outputs and prepare exact checkpoint forks; no physical evolution."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,struct,zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/coulomb-native'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/coulomb_native'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return list(csv.DictReader(path.open()))
def column(rows,name):return np.array([float(r[name]) for r in rows])
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
def write(name,rows):
    with (OUT/name).open('w') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
def resume(ranks):
    source=WORK/'runs'/f'split-r{ranks}';target=WORK/'runs'/f'resume-r{ranks}'
    assert not target.exists()
    pointer=(source/'restart/last_good_checkpoint.txt').read_text().strip()
    assert len(read(source/'bulk-accepted.csv'))==12
    shutil.copytree(source/'restart',target/'restart')
    (WORK/f'resume-r{ranks}.prm').write_text(
        'include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native.prm\n'
        f'set Output directory = $ASPECT_SOURCE_DIR/build-particle-free/coulomb-native/runs/resume-r{ranks}\n'
        'set Resume computation = true\n')
    record={'parent_publication':12,'accepted_time_s':.15,'pending_dt_s':.0125,
            'pointer':pointer,'files':{str(p.relative_to(source/'restart')):sha(p) for p in (source/'restart').rglob('*') if p.is_file()}}
    (WORK/f'resume-r{ranks}-input.json').write_text(json.dumps(record,indent=2)+'\n')
def field_scale(name):
    if name.endswith('_Pa'):return 10000. if 'pressure' in name or name.startswith(('p0','pn_','pprime','min_plane')) else 1000.
    if name.endswith('_s-1'):return .01
    return .001
def compare_fields(a,b):
    assert len(a)==len(b)
    maximum=0.
    for x,y in zip(a,b):
        assert (x['cell'],x['q'])==(y['cell'],y['q'])
        for k in x:
            if k in ['cell','q']:continue
            if k.startswith('theta_'):
                u,v=float(x[k]),float(y[k])
                assert np.isfinite(u)==np.isfinite(v)
                if not np.isfinite(u):continue
                error=abs(.5*np.arctan2(np.sin(2*(u-v)),np.cos(2*(u-v))))
            else:error=abs(float(x[k])-float(y[k]))/max(field_scale(k),abs(float(y[k])))
            maximum=max(maximum,error)
    return maximum
def verify():
    OUT.mkdir(exist_ok=True,parents=True)
    results={'bounds':{'scaled_mean_point_and_MPI':1e-12,'mixed_FD':1e-6,'mechanical':1e-9,'source':1e-12,'BE_ledger':1e-10},'runs':{},'restart':{}}
    summary=[];contact=[];max_point=0.;max_mean=0.;history_defect=0.
    reference=read(WORK/'point-reference.csv')
    for name in ['homogeneous','nonuniform-r1','nonuniform-r4','split-r1','split-r4','resume-r1','resume-r4']:
        run=WORK/'runs'/name;ledger=read(run/'bulk-accepted.csv');profile=read(run/'bulk-profile.csv')
        phase=sorted(run.glob('phase-*.csv'))
        for path in phase:assert np.max(abs(column(read(path),'phi')-.2))==0.
        assert not (run/'bulk-stability.csv').exists(),'No phase variations/Hessian certificate in this mode'
        assert np.max(column(ledger,'mechanical_residual'))<=1e-9
        assert np.max(column(ledger,'projection_error'))<=1e-12
        assert np.max(abs(column(ledger,'ledger_error'))/np.maximum(1.,abs(column(ledger,'Wext'))))<=1e-10
        for key in ['Db','Dc','Df']:assert np.min(column(ledger,key))>=0.
        gates=read(run/'gates.csv');assert all(r['passed']=='1' for r in gates)
        resources=read(run/'resources.csv');assert len(set(r['particle_fingerprint'] for r in resources))==1
        assert all(r['freeze']=='1' for r in resources)
        previous_reaction=None
        initial_reaction=read((run if (run/'boundary-reactions-0000.csv').exists() else WORK/'runs/nonuniform-r1')/'boundary-reactions-0000.csv')
        by_dof={r['dof']:r for r in initial_reaction}
        if 'resume' not in name:
            for component,coordinate,position,expected in [(0,'x',0.,2500.),(0,'x',.25,-2500.),(1,'y',0.,2500.),(1,'y',.25,-2500.)]:
                reaction=sum(float(r['force']) for r in initial_reaction if int(float(r['component']))==component and float(r[coordinate])==position)
                assert abs(reaction-expected)/2500.<=1e-12,(name,reaction)
        for path in sorted(run.glob('coulomb-[0-9][0-9][0-9][0-9].csv')):
            step=int(path.stem.split('-')[-1]);rows=read(path);weights=column(rows,'JxW_m2');volume=weights.sum()
            mean=lambda k:float(weights@column(rows,k)/volume)
            assert len(rows)==16*36 and abs(volume-.25**2)<1e-15
            beta=1/(1+.0125*1e6/1e8)
            trial_x=.5*(column(rows,'tauxx_Pa')-column(rows,'tauyy_Pa'))+2e6*beta*.0125*column(rows,'Wxx_s-1')
            trial_y=column(rows,'tauxy_Pa')+2e6*beta*.0125*column(rows,'Wxy_s-1')
            trial_min=column(rows,'total_pressure_Pa')-.5*(column(rows,'tauxx_Pa')+column(rows,'tauyy_Pa'))-np.hypot(trial_x,trial_y)
            assert min(trial_min)>0.
            assert min(column(rows,'min_plane_normal_Pa'))>0.
            assert min(column(rows,'total_pressure_Pa')-column(rows,'tauzz_Pa'))>0.
            assert min(column(rows,'pn_plus_Pa'))>0. and min(column(rows,'pn_minus_Pa'))>0.
            assert abs(mean('pprime_Pa'))/10000.<=1e-12
            speedp=column(rows,'w_plus_s-1');speedm=column(rows,'w_minus_s-1')
            activity=(speedp>1e-10).astype(int)+2*(speedm>1e-10).astype(int)
            row={'case':name,'step':step,'time_s':step*.0125,'p0_Pa':10000.,'mean_pprime_Pa':mean('pprime_Pa'),
                 'min_pprime_Pa':float(min(column(rows,'pprime_Pa'))),'max_pprime_Pa':float(max(column(rows,'pprime_Pa'))),
                 'min_trial_all_plane_normal_Pa':float(min(trial_min)),
                 'min_all_plane_normal_Pa':float(min(column(rows,'min_plane_normal_Pa'))),
                 'min_zz_compression_Pa':float(min(column(rows,'total_pressure_Pa')-column(rows,'tauzz_Pa'))),
                 'min_pn_plus_Pa':float(min(column(rows,'pn_plus_Pa'))),'min_pn_minus_Pa':float(min(column(rows,'pn_minus_Pa'))),
                 'mean_tauxx_Pa':mean('tauxx_Pa'),'mean_tauxy_Pa':mean('tauxy_Pa'),
                 **{f'activity_{k}_area_fraction':float(weights[activity==k].sum()/volume) for k in range(4)}}
            summary.append(row)
            if step:
                old_path=run/f'coulomb-{step-1:04}.csv'
                if not old_path.exists():old_path=WORK/'runs'/name.replace('resume','split')/f'coulomb-{step-1:04}.csv'
                previous=read(old_path)
                for suffix in ['xx','xy']:
                    A,K,Q,W=(column(rows,'A'+suffix),column(rows,'K'+suffix),column(rows,'Q'+suffix+'_Pa'),column(rows,'W'+suffix+'_s-1'))
                    for defect in [A-beta*(column(previous,'A'+suffix)+.0125*W),K-column(previous,'K'+suffix)-.0125*W,A-1.5/(2e6)*Q]:
                        history_defect=max(history_defect,float(max(abs(defect)))/.001)
                delta_L=column(rows,'L')-column(previous,'L')-.0125*2*np.hypot(column(rows,'Wxx_s-1'),column(rows,'Wxy_s-1'))
                history_defect=max(history_defect,float(max(abs(delta_L)))/.001)
            if name=='homogeneous':
                r=reference[step]
                for k in ['tauxx_Pa','tauxy_Pa','Axx','Axy','Kxx','Kxy','Qxx_Pa','Qxy_Pa','L','Wxx_s-1','Wxy_s-1']:
                    error=max(abs(column(rows,k)-float(r[k])))/max(field_scale(k),abs(float(r[k])))
                    max_point=max(max_point,float(error))
                    max_mean=max(max_mean,abs(mean(k)-float(r[k]))/max(field_scale(k),abs(float(r[k]))))
                for k,tau,A in [('Txx_Pa','tauxx_Pa','Axx'),('Txy_Pa','tauxy_Pa','Axy')]:
                    error=max(abs(column(rows,tau)+2e6*column(rows,A)-float(r[k])))/max(1000.,abs(float(r[k])))
                    max_point=max(max_point,float(error))
                    max_mean=max(max_mean,abs(mean(tau)+2e6*mean(A)-float(r[k]))/max(1000.,abs(float(r[k]))))
                if step:
                    entry=ledger[step-1]
                    for key,reference_key,multiplier in [('bulk','bulk_J_m3',1),('cohesive','cohesive_J_m3',1),('Db','Db_W_m3',.0125),('Dc','Dc_W_m3',.0125),('Df','Df_W_m3',.0125)]:
                        max_mean=max(max_mean,abs(float(entry[key])/volume-float(r[reference_key])*multiplier)/max(1.,abs(float(r[reference_key])*multiplier)))
            reaction_path=run/f'boundary-reactions-{step:04}.csv'
            reactions=read(reaction_path);work=sum(float(x['force'])*float(x['actual_increment']) for x in reactions)
            background_work=sum(float(by_dof[r['dof']]['force'])*float(r['actual_increment']) for r in reactions)
            assert abs(background_work)<=1e-10
            trapezoid=None if previous_reaction is None else sum(.5*(float(x['force'])+float(y['force']))*float(x['actual_increment']) for x,y in zip(reactions,previous_reaction))
            if step:
                entry=next(x for x in ledger if int(x['step'])==step)
                assert abs(work-float(entry['Wext']))<=1e-10
                plane_friction=.0125*float(weights@(.1*column(rows,'pn_plus_Pa')*speedp+.1*column(rows,'pn_minus_Pa')*speedm))
                assert abs(plane_friction-float(entry['Df']))<=1e-12
                assert abs(float(entry['pressure_work']))<=1e-10
                # Keep the physical quadrature remainders separate from BE closure.
                contact.append({'case':name,'step':step,'time_s':step*.0125,'reaction_endpoint_work_J_m':work,'background_pressure_work_J_m':background_work,'plane_friction_integral_J_m':plane_friction,
                                'reaction_trapezoidal_work_J_m':trapezoid,
                                **{k:float(entry[k]) for k in ['energy_change','Db','Dc','Df','stress_loss','traction_loss','mechanical_work','pressure_work','ledger_error']},
                                'endpoint_physical_remainder_J_m':work-float(entry['energy_change'])-sum(float(entry[k]) for k in ['Db','Dc','Df'])})
            previous_reaction=reactions
        tangent=read(run/'coulomb-tangent.csv') if (run/'coulomb-tangent.csv').exists() else []
        if tangent:assert max(column(tangent,'relative_action_error'))<=1e-6
        results['runs'][name]={'accepted_intervals':len(ledger),'last_publication':int(ledger[-1]['publications']),
            'max_mechanical_residual':float(max(column(ledger,'mechanical_residual'))),'max_source_defect':float(max(column(ledger,'projection_error'))),
            'max_BE_error_J_m':float(max(abs(column(ledger,'ledger_error')))),
            'max_FD_relative':float(max(column(tangent,'relative_action_error'))) if tangent else None,
            'solve_seconds':float(sum(column(profile,'total'))),'assembly_seconds':float(sum(column(profile,'assembly'))),
            'linear_seconds':float(sum(column(profile,'linear'))),'history_seconds':float(sum(column(profile,'history'))),
            'peak_rank_KiB':int(max(column(resources,'max_rank_KiB')))}
    assert history_defect<=1e-12,history_defect;results['native_history_recurrence_defect']=history_defect
    assert max_mean<=1e-12,max_mean;results['homogeneous_mean_point_scaled_defect']=max_mean
    results['homogeneous_pointwise_scaled_defect']=max_point
    results['homogeneous_pointwise_1e12_identity_passed']=max_point<=1e-12
    mpi=0.
    for step in range(25):mpi=max(mpi,compare_fields(read(WORK/f'runs/nonuniform-r1/coulomb-{step:04}.csv'),read(WORK/f'runs/nonuniform-r4/coulomb-{step:04}.csv')))
    assert mpi<=1e-12,mpi;results['MPI_1_4_max_scaled_QP_defect']=mpi
    for ranks in [1,4]:
        control=WORK/'runs'/f'nonuniform-r{ranks}';resumed=WORK/'runs'/f'resume-r{ranks}'
        error=0.;bitwise=[]
        for step in range(13,25):
            error=max(error,compare_fields(read(control/f'coulomb-{step:04}.csv'),read(resumed/f'coulomb-{step:04}.csv')))
            for pattern in ['record-{step:04}.bin','coulomb-{step:04}.csv','phase-{step:04}.csv','boundary-reactions-{step:04}.csv']:
                name=pattern.format(step=step)
                if sha(control/name)==sha(resumed/name):bitwise.append(name)
        assert error<=1e-12,error
        results['restart'][str(ranks)]={'QP_defect':error,'bitwise_files':bitwise,'expected_files':48}
        assert len(bitwise)==48,(ranks,len(bitwise))
    for name in ['nonuniform-r1','nonuniform-r4','resume-r1','resume-r4']:
        gates={r['gate'] for r in read(WORK/'runs'/name/'gates.csv')}
        assert {'trial does not publish','rejection restores all accepted owners','deterministic retry','second rejection unchanged'}<=gates
    legacy=0.;legacy_bitwise=[]
    for step in range(9):
        for pattern in ['record-{step:04}.bin','means-{step:04}.csv','phase-{step:04}.csv']:
            name=pattern.format(step=step)
            a,b=WORK/'runs/legacy-reference'/name,WORK/'runs/legacy-current'/name
            if sha(a)==sha(b):legacy_bitwise.append(name)
            if name.endswith('.bin'):
                x,y=np.fromfile(a,dtype='<f8'),np.fromfile(b,dtype='<f8')
                qp=[.001,.001,.001,1000.,.01]+[.001]*12+[1000.]*6+[.01]*6+[.001]
                scales=np.tile(np.array([1000.]*54+[.001]*18+qp*64),64)
                assert len(x)==len(scales)
                legacy=max(legacy,float(max(abs(x-y)/np.maximum(scales,abs(y)))))
            elif name.startswith('means'):
                x,y=read(a),read(b)
                for k in x[0]:
                    if k=='cell':continue
                    scale=1000. if k.startswith('tau') else (8. if k=='Y' else (.01 if k in ['abs_rate','max_rate','Wxx','Wxy'] else (1. if k in ['bulk','cohesive'] else .001)))
                    legacy=max(legacy,float(max(abs(column(x,k)-column(y,k))/np.maximum(scale,abs(column(y,k))))))
            else:assert sha(a)==sha(b),name
    assert legacy<=1e-12,legacy
    results['legacy_native_scaled_defect']=legacy
    results['legacy_native_bitwise_files']=legacy_bitwise
    phase_log=(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/coulomb-native-release-phase-forbidden.log').read_text()
    assert 'physical phase evolution (or explicitly fixed' in ' '.join(phase_log.split())
    assert not (WORK/'runs/phase-forbidden/bulk-accepted.csv').exists()
    results['phase_evolution_refused']=True
    negatives={}
    for name,marker in [('changed-background','The value of surface pressure that was stored in the checkpoint file is not the same'),('wrong-mode','Bulk cohesive checkpoint mode/version mismatch; conversion is unsupported.')]:
        log=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/f'coulomb-native-{name}.log'
        assert marker in ' '.join(log.read_text().split())
        assert json.loads(log.with_suffix('.json').read_text())['exit_code']==1
        assert not (WORK/'runs'/name/'bulk-accepted.csv').exists()
        for path,digest in json.loads((WORK/'negative-inputs.json').read_text())[name].items():
            assert sha(WORK/'runs'/name/'restart'/path)==digest
        negatives[name]={'expected_error':marker,'exit_code':1,'no_publication':True,'checkpoint_files_unchanged':True}
    save('negative-restart.json',negatives)
    save('verification.json',results);write('histories.csv',summary);write('work.csv',contact)
    shutil.copy2(WORK/'point-reference.csv',OUT/'point-reference.csv')
    for name in ['homogeneous','nonuniform-r1','nonuniform-r4']:
        shutil.copy2(WORK/'runs'/name/'coulomb-tangent.csv',OUT/f'{name}-tangent.csv')
    selected=read(WORK/'runs/nonuniform-r1/coulomb-0016.csv')
    write('quadrature-step16.csv',[{'case':'nonuniform-r1','step':16,'time_s':.2,**row} for row in selected])
    run=WORK/'runs/homogeneous';stack=[];resolved={}
    for line in (run/'parameters.prm').read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('subsection '):stack.append(line[11:].strip())
        elif line=='end':stack.pop()
        elif line.startswith('set '):
            key,value=line[4:].split('=',1);resolved['/'.join(stack+[key.strip()])]=value.strip()
    prefix='Material model/Phase field fault/'
    for key,expected in [('Elastic shear moduli',1e6),('Reference viscosities',1e8),('Cohesions',2000.),('Critical energy release rates',1.)]:
        assert all(float(x)==expected for x in resolved[prefix+key].split(',')),key
    assert float(resolved['Phase field model/Length scale'])==.046875
    assert float(resolved['Surface pressure'])==10000. and resolved[prefix+'Evolve phase field']=='false'
    checkpoint=run/'restart'/str(int((run/'restart/last_good_checkpoint.txt').read_text())).zfill(2)/'resume.z'
    raw=zlib.decompress(checkpoint.read_bytes()[16:]);tag=b'bulk cohesive closed Coulomb fixed phase cell history v1'
    offset=raw.index(tag)+len(tag);q,step,pubs,time=struct.unpack_from('<IIQd',raw,offset)
    assert (q,step,pubs)==(6,24,24) and abs(time-.3)<1e-15
    assert struct.unpack_from('<Q',raw,offset+24)[0]==7
    material=struct.unpack_from('<7d',raw,offset+32)
    assert material==(1e6,1e8,4.,1000.,10000.,.1,0.)
    save('resolved-physics.json',{'active_resolved_parameters':{k:v for k,v in resolved.items() if k.startswith(prefix) or k in ['Surface pressure','Phase field model/Length scale']},
        'accepted_checkpoint':{'quadrature':q,'step':step,'publications':pubs,'time_s':time,'material_fields':['G','eta','m','legacy_reference_R_unused_in_Coulomb','p0','mu','c_r'],'material':material}})
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-resume',type=int,choices=[1,4]);args=parser.parse_args()
    if args.prepare_resume:resume(args.prepare_resume)
    else:verify()
