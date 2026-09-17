"""Same-rank research restart: compare lifecycle inputs AND accepted outputs.

Reuse the established 1e-8 per-component coefficient; frozen data and IDs are
exact. Work-QP data are current constitutive stress, mature_history is committed
particle stress, and continued_source_history records the retained particle
input used in the boundary-source update. None may substitute for another.
"""
import argparse
import json
import hashlib
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from check_first_cycle_restart import difference, convergence, load_parts
from analyze_fully_frictional import samples
from analyze_uniform_sliding import read

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=HERE/'fully-frictional-cleanup-local4')
    parser.add_argument('--run',type=Path,default=HERE/'fully-frictional-restart-qualified-local4')
    parser.add_argument('--prefix-only',action='store_true',help='Audit only accepted states 0–4; never declares restart passed')
    args=parser.parse_args()
    ref,run=args.reference,args.run
    def values(tree,path=''):
        answer={}
        for key,value in tree.items():
            if isinstance(value,dict) and 'value' in value: answer[path+'/'+key]=value['value']
            elif isinstance(value,dict): answer.update(values(value,path+'/'+key))
        return answer
    old,new=[values(json.loads((p/'parameters.json').read_text())) for p in (ref,run)]
    changes={key:[old.get(key),new.get(key)] for key in old.keys()|new.keys() if old.get(key)!=new.get(key)}
    allowed={'/Output directory','/Resume computation','/Termination criteria/End step',
             '/Termination criteria/Wall time','/Termination criteria/Termination criteria',
             '/Fault reconstruction/Prescribed faults file','/Postprocess/BP3/Mature prestress file',
             '/Postprocess/BP3/Bottom normalization completion file'}
    assert set(changes)<=allowed,changes
    for key,(a,b) in changes.items():
        if key.endswith(' file'):
            assert hashlib.sha256(Path(a).read_bytes()).digest()==hashlib.sha256(Path(b).read_bytes()).digest()
    old_provenance=json.loads((ref/'provenance.json').read_text())
    new_provenance=json.loads((run/'run_provenance.json').read_text())
    binary=str(HERE.parents[2]/'build-pf-cpdi/aspect-release')
    assert old_provenance['sha256'][binary]==new_provenance['sha256'][binary]
    result={'coefficient':1e-8,'prefix':convergence(run/'run.log'),
            'resumed':{} if args.prefix_only else convergence(run/'resume.log'),'steps':{},'parameter_changes':changes,
            'release_sha256':new_provenance['sha256'][binary]}
    assert set(result['prefix'])==set(range(5))
    if not args.prefix_only: assert set(result['resumed'])=={5,6,7}
    clock=read(run/'accepted_steps.csv');old_clock=read(ref/'accepted_steps.csv')
    count=5 if args.prefix_only else 8
    np.testing.assert_array_equal(clock['step'],np.arange(count))
    for key in ('time','dt','free','lower_active'):
        np.testing.assert_array_equal(clock[key],old_clock[key][:count])
    np.testing.assert_array_equal(clock['free'],1236)
    np.testing.assert_array_equal(clock['lower_active'],0)
    selected=re.findall(r'BP3 matched timestep: next=5 requested=([^ ]+) selected=([^\n]+)',(run/'run.log').read_text())
    assert len(selected)==1
    assert float(selected[0][0])==float(selected[0][1])==old_clock['dt'][5]
    result['checkpoint_next_dt']=float(selected[0][1])
    previous=None
    preceding_particles=None
    for k in range(count):
        a,b=[read(p/f'fault_{k}.csv') for p in (ref,run)]
        for key in ('x','y','Ih','tau_bg','sigma_n_bg','prescribed'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(b['prescribed'],0)
        entry={'fault':{key:difference(a[key],b[key]) for key in ('V','Theta','slip','C','q')}}
        if previous is not None:
            # Independent split cycle: one aging update and one slip increment.
            dt=float(clock['dt'][k]);x=b['V']*dt/.008
            expected=previous['Theta']*np.exp(-x)-.008/b['V']*np.expm1(-x)
            error=np.max(np.abs(b['Theta']-expected)/np.abs(expected))
            assert error<=1e-12,error
            # The Release accumulation is a fused multiply-add; numpy's two
            # rounded operations can differ by one ulp even in the reference.
            expected_slip=np.array([math.fma(dt,float(v),float(s)) for v,s in zip(b['V'],previous['slip'])])
            np.testing.assert_array_equal(b['slip'],expected_slip)
            entry['aging_relative_error']=float(error)
        previous=b
        a,b=[samples(p,k) for p in (ref,run)]
        for key in ('x','y','JxW','source_active','segment','xi','phi','Ih'):
            np.testing.assert_array_equal(a[key],b[key])
        entry['current_constitutive']={key:difference(a[key],b[key]) for key in
            ('V','chi','p','tau_xx','tau_yy','tau_xy','tauN','sigma_n','q','eps_xx','eps_yy','eps_xy')}
        entry['boundary_source']={}
        for name,mask in [('top',a['y']>98000),('bottom',a['y']<2000)]:
            assert np.any(mask & (a['source_active']>0))
            entry['boundary_source'][name]={key:difference(a[key][mask],b[key][mask])
                                         for key in ('V','chi','tau_xx','tau_yy','tau_xy')}
        a,b=[load_parts(p,f'mature_history_{k}_rank*.csv') for p in (ref,run)]
        np.testing.assert_array_equal(a[:,:2],b[:,:2])
        entry['committed_particle_stress']={str(c):difference(a[:,c],b[:,c]) for c in range(2,5)}
        current_particles=b
        entry['history_inputs_and_publication']={}
        if k>0:
            a,b=[load_parts(p,f'continued_source_history_{k}_rank*.csv') for p in (ref,run)]
            names=(ref/f'continued_source_history_{k}_rank0.csv').open().readline().strip().split(',')
            np.testing.assert_array_equal(a[:,0],b[:,0])
            entry['history_inputs_and_publication']={key:difference(a[:,c],b[:,c]) for c,key in enumerate(names) if c>0}
            old_index=np.searchsorted(preceding_particles[:,0],b[:,0])
            new_index=np.searchsorted(current_particles[:,0],b[:,0])
            np.testing.assert_array_equal(preceding_particles[old_index,0],b[:,0])
            np.testing.assert_array_equal(current_particles[new_index,0],b[:,0])
            np.testing.assert_array_equal(preceding_particles[old_index,2:5],b[:,14:17])
            np.testing.assert_array_equal(current_particles[new_index,2:5],b[:,17:20])
            expected=2*b[:,6,None]*(b[:,8:11]-b[:,11:14])+b[:,7,None]*b[:,14:17]
            entry['boundary_Maxwell_update']={str(c):difference(expected[:,c],b[:,17+c]) for c in range(3)}
        preceding_particles=current_particles
        # Measure the actual source integrals independently of rank ownership.
        def source_integrals(root):
            rows=[read(p) for p in sorted(root.glob(f'bulk_slip_transfer_{k}_rank*.csv'))]
            segment=np.concatenate([r['segment'] for r in rows]).astype(int)
            return {key:np.bincount(segment,weights=np.concatenate([r[key] for r in rows]))
                    for key in ('weight','V_integral','chi_integral','instantaneous_integral',
                                'history_integral','total_integral','stress_coefficient_integral')}
        a,b=source_integrals(ref),source_integrals(run)
        entry['actual_source_integrals']={key:difference(a[key],b[key]) for key in a}
        a,b=[read(p/f'work_weak_{k}.csv') for p in (ref,run)]
        entry['weak_loads']={key:difference(a[key],b[key]) for key in ('weight','p','tauN','q','sigma','bg')}
        entry['bulk_output_exact_files']=[]
        for path in sorted(ref.glob(f'bulk_{k}_*.vtu')):
            x,y=[list(ET.parse(p).iter('DataArray')) for p in (path,run/path.name)]
            assert len(x)==len(y) and len(x)>0
            assert all(a.attrib==b.attrib and a.text==b.text for a,b in zip(x,y)),path.name
            entry['bulk_output_exact_files'].append(path.name)
        result['steps'][k]=entry
    result['prefix_passed']=True
    result['restart_passed']=not args.prefix_only
    name='prefix_equivalence.json' if args.prefix_only else 'restart_equivalence.json'
    (run/name).write_text(json.dumps(result,indent=2)+'\n')
    print('Prefix 0–4 PASSED; restart NOT qualified.' if args.prefix_only else
          'Restart PASSED: first resumed step 5 and steps 6/7; exact clock, no reset/double update.')
    for family in ('fault','current_constitutive','committed_particle_stress','history_inputs_and_publication'):
        worst=max((v['relative'],k,key,v['absolute']) for k,row in result['steps'].items() for key,v in row[family].items())
        print(f'{family}: worst relative, step, component, absolute = {worst}')


if __name__=='__main__':
    main()
