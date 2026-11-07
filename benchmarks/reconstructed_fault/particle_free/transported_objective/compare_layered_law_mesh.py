#!/usr/bin/env python3
"""Mesh sensitivity of the constitutive difference, using four matched runs.

Compare (bulk spin - total spin) on the coarse and fine meshes. Absolute
trajectory mesh errors need not equal the mesh error of this difference.
"""
import argparse
import json
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_layered_campaign import OUT, load_case, state_at, common_bulk_quadrature
from assess_sheared_moving_fault import masks, metric
from layered_fields import dot3, surface, surface_rule


def differences(fields, weights, labels):
    coarse = {k:fields[0][k]-fields[1][k] for k in fields[0] if k!='jacobian'}
    fine = {k:fields[2][k]-fields[3][k] for k in coarse}
    result = {}
    for region, mask in masks(labels).items():
        values = {}
        for key in coarse:
            error = (coarse[key]-fine[key])[mask]
            reference = fine[key][mask]
            if key=='stress':
                num, den = dot3(error,error), dot3(reference,reference)
            elif error.ndim==2:
                num, den = (error*error).sum(axis=1), (reference*reference).sum(axis=1)
            else:
                num, den = error*error, reference*reference
            values[key] = metric(num,den,weights[mask])
        result[region] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bulk_coarse','total_coarse','bulk_fine','total_fine'):
        parser.add_argument(name)
    args = parser.parse_args()
    names = list(vars(args).values())
    cases = [json.loads((OUT/(name+'.json')).read_text()) for name in names]
    for i,c in enumerate(cases):
        assert c['arguments']['spin']==('bulk slip subtracted' if i%2==0 else 'total material')
        for key in ('translation','background_shear','dt','halfwidth','eta'):
            assert c['arguments'][key]==cases[0]['arguments'][key], key
        assert abs(c['records'][-1]['time']-cases[0]['records'][-1]['time'])<1e-13
        assert c['labels']==cases[0]['labels']
    states = []
    for name in names:
        path, command, rows = load_case(name)
        states.append(state_at(path,command,int(rows[-1]['step'])))
    # Each law pair uses the same grid; partition at both spatial resolutions.
    for i,j in ((0,1),(2,3)):
        assert np.array_equal(states[i].shape,states[j].shape)
        assert np.array_equal(states[i].lo,states[j].lo)
        assert np.array_equal(states[i].hi,states[j].hi)
    x,w = common_bulk_quadrature(states[0],states[2],0.)
    delta_coarse = states[0].stress(x)-states[1].stress(x)
    delta_fine = states[2].stress(x)-states[3].stress(x)
    error = delta_coarse-delta_fine
    bulk = metric(dot3(error,error),dot3(delta_fine,delta_fine),w)

    labels,weights = surface_rule(states)
    fields = [surface(s,labels,c['records'][-1]['cumulative_vertices'])
              for s,c in zip(states,cases)]
    final = differences(fields,weights*fields[3]['jacobian'],labels)
    history = []
    by_time = [{round(r['time'],10):r for r in c['records']} for c in cases]
    common_times = sorted(set.intersection(*(set(d) for d in by_time)))
    labels = np.array(cases[0]['labels'])
    for t in common_times:
        fields = [{k:np.array(v) for k,v in d[t]['profile'].items()} for d in by_time]
        history.append(dict(time=t,regions=differences(fields,fields[3]['jacobian']/len(labels),labels)))
    result = dict(cases=names,interpretation='Mesh change of (bulk-spin minus total-spin); fine law-difference norm is the denominator, not the full stress/slip norm.',
                  bulk_final=bulk,surface_final=final,history=history)
    (OUT/'law-difference-mesh.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(bulk=bulk,central=final['central'],ends=final['ends'])),flush=True)


if __name__=='__main__':
    main()
