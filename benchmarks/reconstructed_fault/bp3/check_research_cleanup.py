"""Compare the cleaned plugin/core against the preserved seven-step trajectory.

Reuse the existing per-component restart coefficient (1e-8), not a tolerance
fitted to cleanup differences. Geometry, prescribed masks, phase, Ih and fixed
backgrounds must agree exactly. Current stress and published history are
compared separately; neither is substituted for the other.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from check_first_cycle_restart import difference, convergence, load_parts
from analyze_fully_frictional import samples
from analyze_uniform_sliding import read

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=HERE/'fully-frictional-seven-local4')
    parser.add_argument('--run',type=Path,default=HERE/'fully-frictional-cleanup-local4')
    args=parser.parse_args()
    result={'relative_comparison_coefficient':1e-8,
            'convergence':convergence(args.run/'run.log'),'steps':{}}
    def parameter_values(tree,path=''):
        values={}
        for key,value in tree.items():
            if isinstance(value,dict) and 'value' in value:
                values[path+'/'+key]=value['value']
            elif isinstance(value,dict):
                values.update(parameter_values(value,path+'/'+key))
        return values
    old,new=[parameter_values(json.loads((p/'parameters.json').read_text()))
             for p in (args.reference,args.run)]
    changes={key:[old.get(key),new.get(key)] for key in old.keys()|new.keys()
             if old.get(key)!=new.get(key)}
    assert set(changes)<={'/Output directory','/Postprocess/BP3/Fault loading configuration'},changes
    assert new['/Postprocess/BP3/Fault loading configuration']=='fully frictional'
    result['effective_parameter_changes']=changes
    accepted=read(args.run/'accepted_steps.csv')
    np.testing.assert_array_equal(accepted['step'],np.arange(8))
    np.testing.assert_array_equal(accepted['free'],1236)
    np.testing.assert_array_equal(accepted['lower_active'],0)
    old_clock=read(args.reference/'accepted_steps.csv')
    for key in ('time','dt'):
        np.testing.assert_array_equal(accepted[key],old_clock[key])
    for k in range(8):
        a,b=[read(p/f'fault_{k}.csv') for p in (args.reference,args.run)]
        assert len(b['V'])==1236
        for key in ('x','y','Ih','tau_bg','sigma_n_bg','prescribed'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(b['prescribed'],0)
        entry={'fault':{key:difference(a[key],b[key])
                        for key in ('V','Theta','slip','C','q')}}
        a,b=[samples(p,k) for p in (args.reference,args.run)]
        for key in ('x','y','JxW','source_active','segment','xi','phi','Ih'):
            np.testing.assert_array_equal(a[key],b[key])
        entry['current_constitutive']={key:difference(a[key],b[key])
            for key in ('V','chi','p','tau_xx','tau_yy','tau_xy','tauN','sigma_n','q',
                        'eps_xx','eps_yy','eps_xy')}
        a,b=[load_parts(p,f'mature_history_{k}_rank*.csv') for p in (args.reference,args.run)]
        np.testing.assert_array_equal(a[:,:2],b[:,:2])  # stable ID and inert H
        entry['published_particle_stress']={str(c):difference(a[:,c],b[:,c]) for c in range(2,5)}
        # Bulk output is Float32; compare its actual serialized arrays exactly,
        # ignoring only the file-generation timestamp in the XML comment.
        # The separate CSV comparison above retains full-precision stress/strain.
        old_files=sorted(args.reference.glob(f'bulk_{k}_*.vtu'))
        new_files=sorted(args.run.glob(f'bulk_{k}_*.vtu'))
        assert [p.name for p in old_files]==[p.name for p in new_files]
        entry['bulk_output_exact_files']=[]
        for old_file,new_file in zip(old_files,new_files):
            x,y=[list(ET.parse(p).iter('DataArray')) for p in (old_file,new_file)]
            assert len(x)==len(y) and len(x)>0
            assert all(a.attrib==b.attrib and a.text==b.text for a,b in zip(x,y)),old_file.name
            entry['bulk_output_exact_files'].append(new_file.name)
        result['steps'][k]=entry
    result['passed']=True
    (args.run/'cleanup_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    for family in ('fault','current_constitutive','published_particle_stress'):
        worst=max((v['relative'],k,key,v['absolute'])
                  for k,row in result['steps'].items() for key,v in row[family].items())
        print(f'{family}: worst scaled error, step, field, absolute = {worst}')
    print('Cleanup equivalence PASSED: eight accepted states; all nodes free; unchanged clock and split histories.')


if __name__=='__main__':
    main()
