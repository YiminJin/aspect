"""Bounded long-run preparation checks; retain the existing 1e-8 field criterion.

Current work-QP stress, retained bulk FE history and committed particle stress
are different lifecycle quantities and are compared separately.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from check_first_cycle_restart import difference, convergence, load_parts
from analyze_fully_frictional import samples
from analyze_uniform_sliding import read


def native_fault(root, step):
    tree=ET.parse(root/f'reconstructed_faults/reconstructed_faults-{step:05d}.vtu')
    # Compare old evidence with the output-only aliases; omitted implementation
    # data remain covered by checkpoint/history checks, not visualization.
    aliases={'phase field fault state':'slip_state',
             'phase field fault cohesive traction':'cohesive_traction',
             'phase field fault previous I h':'previous_I_h',
             'cumulative_signed_slip_m':'cumulative_slip'}
    excluded={'vertex_id','BP3 fixed shear correction','mature fault reference geometry'}
    result={}
    for a in tree.findall('.//PointData/DataArray'):
        name=a.get('Name')
        if name in excluded: continue
        name=aliases.get(name,name.replace('phase field fault chemical composition ','composition_'))
        result[name.replace(' ','_')]=np.fromstring(a.text,sep=' ')
    return result


def outputs(root):
    heavy=read(root/'heavy_outputs.csv')
    # Read mixed string/numeric profile index independently.
    import csv
    with (root/'profiles.csv').open() as f: index=list(csv.DictReader(f))
    times=[float(r['time_s']) for r in index]
    assert all(a<b for a,b in zip(times,times[1:]))
    for row in index:
        data=read(root/row['file'])
        np.testing.assert_array_equal(data['time_s'],float(row['time_s']))
    collections={}
    sizes={}
    for name in ('solution','particles','reconstructed_faults'):
        nodes=ET.parse(root/f'{name}.pvd').findall('.//DataSet')
        collections[name]=[float(n.get('timestep')) for n in nodes]
        assert all(a<b for a,b in zip(collections[name],collections[name][1:]))
        sizes[name]=[]
        for n in nodes:
            file=root/n.get('file');assert file.is_file()
            pieces=ET.parse(file).findall('.//Piece') if file.suffix=='.pvtu' else []
            sizes[name].append(file.stat().st_size+sum((file.parent/p.get('Source')).stat().st_size for p in pieces))
    assert collections['solution']==collections['particles']==collections['reconstructed_faults']
    np.testing.assert_allclose(collections['solution'],heavy['time_s'],rtol=2e-11,atol=0.)
    return dict(times=collections,bytes=sizes,profile_bytes=[(root/r['file']).stat().st_size for r in index])


def compare(ref,run,steps,same_mesh):
    report={}
    for k in steps:
        entry={}
        b=read(run/f'profiles/fault_{k}.csv')
        if same_mesh:
            a=read(ref/f'fault_{k}.csv')
            mapping={'x':'x_m','y':'y_m','V':'V_m_per_s','Theta':'Theta_s','slip':'slip_m','q':'q_weak_Pa'}
        else:
            a=read(ref/f'profiles/fault_{k}.csv')
            mapping={key:key for key in ('x_m','y_m','V_m_per_s','Theta_s','slip_m','q_weak_Pa','sigma_n_weak_Pa')}
        entry['fault']={key:difference(a[key],b[value]) for key,value in mapping.items()}
        a,b=[samples(root,k) for root in (ref,run)]
        for key in ('x','y','JxW','phi','Ih','segment','xi','source_active'):
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        entry['current_stress_source']={key:difference(a[key],b[key]) for key in
            ('V','chi','p','tau_xx','tau_yy','tau_xy','tauN','sigma_n','q','eps_xx','eps_yy','eps_xy')}
        for name in ('top','bottom'):
            mask=b['y']>98000 if name=='top' else b['y']<2000
            assert np.any(mask & (b['source_active']>0))
            entry[name]={key:difference(a[key][mask],b[key][mask]) for key in ('chi','tau_xx','tau_yy','tau_xy')}
        a,b=[load_parts(root,f'mature_history_{k}_rank*.csv') for root in (ref,run)]
        np.testing.assert_array_equal(a[:,:2],b[:,:2])
        entry['committed_particle_stress']={str(i):difference(a[:,i],b[:,i]) for i in range(2,5)}
        if not same_mesh:
            for kind in ('bulk','particles'):
                a,b=[load_parts(root,f'audit_{kind}_{k}_rank*.csv') for root in (ref,run)]
                np.testing.assert_array_equal(a[:,0],b[:,0])
                if kind=='bulk':
                    np.testing.assert_array_equal(a[:,1],b[:,1])
                    entry['bulk_FE']={str(c):difference(a[a[:,1]==c,2],b[b[:,1]==c,2]) for c in np.unique(a[:,1])}
                else: entry['particle_properties']={str(c):difference(a[:,c],b[:,c]) for c in range(1,a.shape[1])}
            native=[(root/f'reconstructed_faults/reconstructed_faults-{k:05d}.vtu').exists()
                    for root in (ref,run)]
            assert native[0]==native[1], 'Native output schedules differ'
            if native[0]:
                a,b=[native_fault(root,k) for root in (ref,run)]
                assert a.keys()==b.keys()
                entry['native_fault']={key:difference(a[key],b[key]) for key in a}
        a,b=[read(root/f'work_weak_{k}.csv') for root in (ref,run)]
        entry['weak']={key:difference(a[key],b[key]) for key in ('weight','p','tauN','q','sigma','bg')}
        report[k]=entry
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('run',type=Path)
    p.add_argument('--steps',type=int,nargs='+',required=True)
    p.add_argument('--same-mesh',action='store_true')
    p.add_argument('--log',default='resume.log')
    args=p.parse_args()
    result={'coefficient':1e-8,'convergence':convergence(args.run/args.log),
            'outputs':outputs(args.run),'fields':compare(args.reference,args.run,args.steps,args.same_mesh)}
    a,b=[read(root/'accepted_steps.csv') for root in (args.reference,args.run)]
    for step in args.steps:
        for key in ('time','dt','free','lower_active'):
            np.testing.assert_array_equal(a[key][a['step']==step],b[key][b['step']==step])
    result['passed']=True
    (args.run/'long_run_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    worst=max((v['relative'],k,f,key) for k,row in result['fields'].items() for f,fields in row.items() for key,v in fields.items())
    print('PASSED; worst relative, step, family, field:',worst)


if __name__=='__main__':main()
