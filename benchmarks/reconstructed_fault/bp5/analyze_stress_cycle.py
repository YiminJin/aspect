"""Small tensor-level transfer/update audit; no interpolation of plotted CSVs."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

TRANSFER_COLUMNS=('stage step time_s dt cell index field component property dof x y ref_x ref_y value').split()
COMPONENTS=('xx','yy','xy')


def rows(root,pattern,columns=None):
    result=[]
    for file in sorted(root.glob(pattern)):
        with file.open() as stream: result.extend(csv.DictReader(stream,fieldnames=columns))
    return result


def norm(data,key):
    weight=sum(float(r['work_weight']) for r in data)
    return math.sqrt(sum(float(r['work_weight'])*float(r[key])**2 for r in data)/weight)


def audit(path):
    out=path if (path/'normal_summary.csv').exists() else path/'output-normal-diagnostic'
    if (path/'execution.json').exists() and not json.loads((path/'execution.json').read_text())['passed']:
        raise ValueError(f'{path} did not complete; do not interpret its candidate output as accepted')
    # Server copies may contain only ASPECT's output directory, not the local
    # launcher manifest. Require numerical and lifecycle evidence in either case.
    log=(out/'log.txt').read_text()
    residuals=re.findall(r'Relative nonlinear residuals.*?:\s*([\deE+.-]+),\s*([\deE+.-]+)',log)
    linear=re.findall(r'fresh=([\deE+.-]+), target=([\deE+.-]+)',log)
    summaries=rows(out,'normal_summary.csv')
    if (len(summaries)!=1 or not residuals or max(map(float,residuals[-1]))>1e-8
        or not linear or any(float(a)>float(b) for a,b in linear)
        or 'Termination requested by criterion: BP5 normal diagnostic complete' not in log):
        raise ValueError('Missing or unsuccessful numerical convergence/lifecycle evidence')
    before={r['id']:r for r in rows(out,'stress_particles_before_rank*.csv')}
    after={r['id']:r for r in rows(out,'stress_particles_after_rank*.csv')}
    updates=rows(out,'stress_update_*_rank*.csv')
    if not updates or set(before)!=set(after) or {r['particle_id'] for r in updates}!=set(before):
        raise ValueError('Incomplete parent/candidate/committed trace')
    error=0.;sample_error=0.
    for r in updates:
        a=before[r['particle_id']];b=after[r['particle_id']]
        for key in ('x','y'):
            assert float(a[key])==float(b[key])==float(r[key])
            sample_error=max(sample_error,abs(float(r[key])-float(r['sample_'+key])))
        for c in COMPONENTS:
            old=float(r['old_'+c]);new=float(r['new_'+c])
            assert old==float(a['tau_'+c]) and new==float(b['tau_'+c])
            computed=float(r['beta'])*old+2*float(r['kappa'])*(float(r['eps_'+c])-float(r['crack_'+c]))
            error=max(error,abs(new-computed))
    transfer=rows(out,'stress_transfer_*_rank*.csv',TRANSFER_COLUMNS)
    published={(r['cell'],r['index'],r['field'].split('(')[-1].rstrip(')')):float(r['value'])
               for r in transfer if r['stage']=='published_FE'}
    if not published or not any(r['stage']=='support_proposal' for r in transfer):
        raise ValueError('Actual transfer trace is missing')
    qp=rows(out,'normal_qp_*_rank*.csv')
    selected=[r for r in qp if (r['cell'],r['qp'],'tau_xx') in published]
    if not selected: raise ValueError('No matched production QP in cell trace')
    comparisons={}
    for c in COMPONENTS:
        comparisons[c]=dict(
            published_to_working_max=max(abs(published[(r['cell'],r['qp'],'tau_'+c)]-float(r['incoming_FE_'+c])) for r in selected),
            direct_particle_interpolation_to_working_max=max(abs(float(r['particle_interp_'+c])-float(r['incoming_FE_'+c])) for r in selected))
    windows={}
    for name,low,high in [('70-71',70000,71000),('79-80',79000,80000)]:
        samples=[r for r in qp if low<=float(r['down_dip_s_m'])<=high]
        if not samples:raise ValueError(f'Empty {name}')
        windows[name]={key+'_RMS':norm(samples,key) for key in
                       ('d_history','d_update','d_strain','d_slip','grad_xx','grad_xy','grad_yx','grad_yy')}
        windows[name]['max_slip_normal_over_tensor']=max(abs(float(r['d_slip']))/
           max(1e-300,abs(float(r['slip_xx']))+abs(float(r['slip_yy']))+2*abs(float(r['slip_xy']))) for r in samples)
        windows[name]['samples']=len(samples)
    result=dict(particles=len(before),cells=len({r['cell'] for r in before.values()}),
                candidate_formula_max_Pa=error,particle_sample_position_error_m=sample_error,
                transfer_comparisons_Pa=comparisons,windows=windows,
                clock=next(csv.DictReader((out/'stress_cycle_clock.csv').open())),
                constitutive_dt=sorted({float(r['stress_dt']) for r in qp}),
                particle_update_dt=sorted({float(r['dt']) for r in updates}))
    result['convergence']=dict(final_relative_residuals=list(map(float,residuals[-1])),
                               fresh_linear_checks=len(linear),summary=summaries[0])
    assert result['constitutive_dt']==result['particle_update_dt']==[float(result['clock']['actual_dt'])]
    return result,qp


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    output={};data={};inventories=[];faults=[]
    for name,server in (('dt','full'),('half','half'),('quarter','quarter')):
        path=args.directory/name
        if not path.exists(): path=args.directory/('output-'+server)
        output[name],data[name]=audit(path)
        out=path if (path/'normal_summary.csv').exists() else path/'output-normal-diagnostic'
        inventories.append((out/'stress_particles_before_inventory.txt').read_text())
        faults.append((out/'normal_restored_fault.csv').read_bytes())
    if len(set(inventories))!=1 or len(set(faults))!=1:raise ValueError('Incoming particle/fault states differ')
    base={(r['cell'],r['qp']):r for r in data['dt']}
    changes={}
    for name in ('half','quarter'):
        current={(r['cell'],r['qp']):r for r in data[name]}
        assert current.keys()==base.keys()
        for k in base:
            for c in ('x','y','phase','I_h','chi','incoming_FE_xx','incoming_FE_yy','incoming_FE_xy'):
                assert float(current[k][c])==float(base[k][c]),(name,k,c)
        changes[name]={}
        mass=sum(float(r['work_weight']) for r in base.values())
        for field in ('d_update','d_strain','d_slip','grad_xx','grad_xy','grad_yx','grad_yy'):
            changes[name][field+'_difference_RMS']=math.sqrt(sum(float(r['work_weight'])*
                (float(current[k][field])-float(r[field]))**2 for k,r in base.items())/mass)
    output['changes_from_dt']=changes
    text=json.dumps(output,indent=2)+'\n'
    (args.directory/'stress_cycle_summary.json').write_text(text);print(text)
