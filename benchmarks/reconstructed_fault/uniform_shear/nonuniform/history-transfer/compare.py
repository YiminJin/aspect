#!/usr/bin/env python3
"""Compare frozen physical data, realized FE fields, and assembled weak loads."""
import json
import argparse
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prefix', default='')
parser.add_argument('--require-invariance', action='store_true')
args = parser.parse_args()

def read(case, name, keys):
    files = sorted((base/(args.prefix+case)).glob(name+"_rank*.csv"))
    assert files, (case, name)
    data = np.concatenate([np.atleast_1d(np.genfromtxt(p, names=True, delimiter=",")) for p in files])
    order = np.lexsort(tuple(data[key] for key in reversed(keys)))
    data = data[order]
    coordinates = np.column_stack([data[k] for k in keys])
    assert len(np.unique(np.round(coordinates, 13), axis=0)) == len(data)
    return data

def matching(a, b, keys):
    assert len(a) == len(b)
    for key in keys:
        assert np.max(abs(a[key]-b[key])) < 1e-13, key

def difference(a, b, field, weight=None):
    delta = a[field]-b[field]
    if weight is None:
        return dict(l2=float(np.linalg.norm(delta)), maximum=float(max(abs(delta))),
                    relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(a[field]), 1e-300)))
    assert np.array_equal(a[weight], b[weight])
    return dict(rms=float(np.sqrt(np.average(delta**2, weights=a[weight]))),
                maximum=float(max(abs(delta))))

particles = [read(case, "particles", ("x", "y")) for case in ("one", "two")]
matching(*particles, ("x", "y"))
assert np.array_equal(particles[0]["stress"], particles[1]["stress"])
assert np.array_equal(particles[0]["constant"], particles[1]["constant"])
report = dict(particle_count=len(particles[0]), identical_physical_particle_data=True,
              comparisons=[], constraint_corrections=[], approximation=[])
for mode in range(1, 5):
    for case in ("one", "two"):
        data = read(case, f"fields_{mode}", ("x", "y"))
        delta = data["working"]-data["published"]
        report["constraint_corrections"].append(dict(case=case, mode=mode,
            rms_Pa=float(np.sqrt(np.average(delta**2, weights=data["weight"]))),
            maximum_Pa=float(max(abs(delta)))))
        if mode == 3:
            prescribed = 1500+20*np.sin(8*np.pi*data['x'])+10*data['y']
            report['approximation'].append(dict(case=case, **{
                field+'_rms_Pa': float(np.sqrt(np.average((data[field]-prescribed)**2, weights=data['weight'])))
                for field in ('published','working')}))
            # Interior Q2 support nodes have one incident cell: the transfer
            # must retain its original cell mean, independent of shared nodes.
            p=read(case,'particles',('x','y'))
            cell=lambda a: np.floor(a['x']/.03125).astype(int)+8*np.floor((a['y']+.5)/.03125).astype(int)
            pc=cell(p)
            means=np.bincount(pc,weights=p['stress'])/np.bincount(pc)
            center=(abs(data['x']/.03125%1-.5)<1e-12)&(abs((data['y']+.5)/.03125%1-.5)<1e-12)
            assert sum(center)==256
            error=float(max(abs(data['published'][center]-means[cell(data)[center]])))
            assert error<1e-10
            report['approximation'][-1]['single_contribution_center_error_Pa']=error
for label, left, right in (
    ("constant one traversal", ("one", 1), ("one", 2)),
    ("constant two traversal", ("two", 1), ("two", 2)),
    ("nonconstant one traversal", ("one", 3), ("one", 4)),
    ("nonconstant two traversal", ("two", 3), ("two", 4)),
    ("constant forward MPI", ("one", 1), ("two", 1)),
    ("constant reverse MPI", ("one", 2), ("two", 2)),
    ("nonconstant forward MPI", ("one", 3), ("two", 3)),
    ("nonconstant reverse MPI", ("one", 4), ("two", 4))):
    fields = [read(case, f"fields_{mode}", ("x", "y")) for case, mode in (left, right)]
    loads = [read(case, f"load_{mode}", ("x", "y", "component")) for case, mode in (left, right)]
    matching(*fields, ("x", "y"))
    matching(*loads, ("x", "y", "component"))
    result = dict(comparison=label,
                  published=difference(*fields, "published", "weight"),
                  working=difference(*fields, "working", "weight"),
                  load=difference(*loads, "load"))
    if "constant" in label and "nonconstant" not in label:
        assert result["working"]["maximum"] < 1e-10
        assert result["load"]["maximum"] < 1e-9
    report["comparisons"].append(result)
    if args.require_invariance:
        assert result['published']['maximum'] < 1e-9
        assert result['working']['maximum'] < 1e-9
        assert result['load']['l2'] < 1e-9
(base/(args.prefix+"comparison.json")).write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
