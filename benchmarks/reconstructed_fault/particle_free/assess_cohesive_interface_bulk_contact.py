#!/usr/bin/env python3
"""Common physical tests and explicit failures for the fixed-path comparison."""
import csv
import json
import numpy as np
from investigate_cohesive_interface_bulk_contact import OUT, WORK, SAVED, save, table


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def main():
    labels=[p.name.removesuffix('-summary.json') for p in OUT.glob('*-summary.json')
            if p.name.startswith(('narrow','wide','fixed','fitted','offset'))]
    reference={r['region']:r for r in read('fitted96-loads.csv')}
    ref_energy=json.loads((OUT/'fitted96-summary.json').read_text())['jump_induced_J_m']
    loads=[];profiles=[];errors=[];volumes=[];cost=[]
    points=None
    for label in sorted(labels):
        r=read(label+'-loads.csv');p=read(label+'-profiles.csv');meta=json.loads((OUT/(label+'-summary.json')).read_text())
        current=np.array([[float(row[k]) for k in ['x_m','y_m','weight_m']] for row in p])
        if points is None:points=current
        assert np.array_equal(points,current),'Physical surface coordinates/weights differ'
        weights=current[:,2].reshape(-1,3);z=(np.polynomial.legendre.leggauss(3)[0]+1)/2;b=4*z*(1-z)
        raw=np.array([(float(row['raw_normal_minus_Pa'])+float(row['raw_normal_plus_Pa']))/2-10000 for row in p])
        projection=np.repeat((weights*b*raw.reshape(-1,3)).sum(axis=1)/(weights*b).sum(axis=1),3)
        for q,(row,v) in enumerate(zip(p,projection)):
            row['raw_normal_same_bubble_Pa']=float(v+10000)
            row['physical_jump_m']=SAVED['histories']['jump'][q]
            row['physical_gamma']=SAVED['gamma'];row['relaxed_affine_coefficient']=SAVED['wall_memory']
        profiles+=p;loads+=r
        for row in r:
            ref=reference[row['region']]
            shear=float(row['tangential_work_conjugate_Pa'])-float(ref['tangential_work_conjugate_Pa'])
            normal=float(row['normal_perturbation_Pa'])-float(ref['normal_perturbation_Pa'])
            relative_shear=abs(shear)/abs(float(ref['tangential_work_conjugate_Pa']))
            relative_normal=abs(normal)/abs(float(ref['normal_perturbation_Pa']))
            errors.append(dict(label=label,region=row['region'],tangential_difference_Pa=shear,tangential_relative=relative_shear,
                normal_difference_Pa=normal,normal_perturbation_relative=relative_normal,
                tangential_target_pass=relative_shear<.02,normal_target_pass=relative_normal<.02,
                jump_storage_difference_J_m=meta['jump_induced_J_m']-ref_energy,
                jump_storage_relative=abs(meta['jump_induced_J_m']-ref_energy)/ref_energy))
        for row in csv.DictReader((WORK/label/'physical-diagnostic.csv').open()):
            row=dict(row);row['label']=label
            row['divergence_RMS']=float(np.sqrt(float(row['divergence_squared_m2'])/float(row['area_m2'])))
            volumes.append(row)
        cost.append(dict(label=label,cells=meta['cells'],bulk_masters=meta['bulk_masters'],jump_modes=meta['jump_modes'],
            degree=meta['degree'],lifting_width_m=meta['lifting_width_m'],normal_test_m=meta['normal_test_width_m'],
            fitted=meta['fitted_mesh'],native_seconds=meta['seconds'],mixed_residual=meta['mixed_residual'],
            energy_check_relative=meta['energy_relative'],normal_work_check=meta['normal_work_check']))
    table(OUT/'frozen-loads.csv',loads);table(OUT/'frozen-profiles.csv',profiles);table(OUT/'frozen-errors.csv',errors)
    table(OUT/'incompressibility.csv',volumes);table(OUT/'native-cost.csv',cost)
    save(OUT/'frozen-assessment.json',dict(reference='Independent fitted96 response, checked by fitted64 and fitted-Q3, not a converted checkpoint',
        common_physical_coordinates_bitwise=True,common_normal_bubble_width_m=.75,
        physical_gamma=SAVED['gamma'],relaxed_affine_coefficient=SAVED['wall_memory'],
        common_jump='Actual saved physical Gauss jump, including the unchanged permanent seed offset; bulk stress depends on the separately fixed relaxed-memory field.',
        tangential_test='Common Q1 trace of sin^2 bump at existing physical nodes; work divided by its physical measure',
        regions=dict(old=[3.75,5.5],developed=[6,6.5],zone=[6.5,7.75],ahead=[7.75,8.4]),
        integration_note='Global/near-fault measures match. Hard x-clipped tip subregion has up to .1% measure uncertainty; report its area.',
        errors=errors,scope='Frozen independently equilibrated responses; pressure p0 unchanged, no accepted history conversion.'))


if __name__=='__main__':main()
