"""Compare every keyed donor column that has a counterpart in the target."""
import argparse
import csv
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def read(paths, keys=()):
    paths = sorted(paths)
    if not paths:
        raise ValueError('Missing frame')
    records = []
    header = None
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            if header is None:
                header = fields
            elif fields != header:
                raise ValueError(f'Inconsistent field names in {path}')
            shard = list(reader)
            if not shard:
                raise ValueError(f'Empty shard {path}')
            if any(len(row) != len(header) for row in shard):
                raise ValueError(f'Incomplete row in {path}')
            records.extend(shard)
    data = {k: np.asarray(values, dtype=str if k=='cell' else float)
            for k,values in zip(header,zip(*records))}
    if keys:
        order = np.lexsort(tuple(data[k] for k in reversed(keys)))
        data = {k:v[order] for k,v in data.items()}
        if np.any(np.logical_and.reduce([data[k][1:] == data[k][:-1] for k in keys])):
            raise ValueError(f'Duplicate identity in {paths}')
    return data


def shards(folder, stem, keys):
    files = sorted(folder.glob(stem+'-r*.csv*'))
    if [f.name.split('-r')[-1].split('.')[0] for f in files] != ['0','1']:
        raise ValueError(f'Missing MPI owner for {stem} in {folder}')
    return read(files, keys)


@lru_cache
def field_names(output):
    """Physical component of each exported FE index on this ordinary Stokes layout."""
    parameters = json.loads((output/'parameters.json').read_text())
    if parameters['Dimension']['value'] != '2':
        raise ValueError('The 18-strip comparison requires plane strain')
    compositions = parameters['Compositional fields']['Names of fields']['value']
    return ['velocity_x', 'velocity_y', 'pressure', 'temperature', *[s.strip() for s in compositions.split(',')]]


class Comparison:
    def __init__(self):
        self.rows = []

    def check(self, name, expected, actual, atol, rtol=0., ids=None):
        a,b = np.broadcast_arrays(expected,actual)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f'Nonfinite values for {name}')
        delta = abs(a-b)
        budget = atol+rtol*abs(a)
        ratio = delta / np.maximum(1e-300,budget)
        i = int(np.argmax(ratio))
        self.rows.append((name,int(np.sum(delta>budget)),float(delta.max()),float(ratio.flat[i]),
                          tuple(v[i].item() for v in ids) if ids else i,float(a.flat[i]),float(b.flat[i])))

    def write(self, path):
        with path.open('w') as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(['quantity','failed','max_abs','max_ratio','worst_key','donor','target'])
            writer.writerows(self.rows)
        failed = [r for r in self.rows if r[1]]
        for r in failed:
            print(*r)
        return len(failed)


def aligned(a,b,keys):
    for k in keys:
        np.testing.assert_array_equal(a[k], b[k], err_msg=f'Identity mismatch on {k}')
    return [a[k] for k in keys]


def compare(step, target, donor, replay, cmp):
    tag = f'{step:04}'
    keys = ('cell','component','local')
    a,b = (shards(p,'mesh-'+tag,keys) for p in [replay,target])
    ids = aligned(a,b,keys)
    for k in ('x0','y0','x1','y1'):
        cmp.check('geometry/'+k,a[k],b[k],0.,ids=ids)
    names = field_names(target)
    if names != field_names(replay):
        raise ValueError('Donor and target FE field ordering differs')
    np.testing.assert_array_equal(np.unique(a['component']), np.arange(len(names)))
    for c in np.unique(a['component']).astype(int):
        mask = a['component']==c
        atol = {'velocity_x':1e-10,'velocity_y':1e-10,'pressure':1e-6,'phase_field':1e-9}.get(names[c],1e-12)
        cmp.check(f'FE/{names[c]}',a['value'][mask],b['value'][mask],atol,1e-7,[v[mask] for v in ids])
    keys = ('cell','q')
    a = shards(donor if step else replay,'tip-points-'+tag,keys)
    b = shards(target,'points-'+tag,keys)
    ids = aligned(a,b,keys)
    for k in (k for k in b if k not in keys):
        atol = 0 if k=='section' else 1e-14 if k in ('x','y','weight') else 1e-6 if k.startswith('tau') else 1e-9 if k=='phi' else 1e-10
        cmp.check('point/'+k,a[k],b[k],atol,1e-7 if atol>=1e-10 else 0.,ids)
    if not step:
        # The donor writes the virgin incoming history on its first physical
        # plane table. The unturned seed spans exactly the reference width.
        a=read([donor/'tip-planes-0001.csv'],['section'])
        b=read([target/'sections-0000.csv'],['section'])
        ids=aligned(a,b,['section'])
        for key,expected,atol in [('xi',a['xi'],1e-11),('ds',a['reference_dxi'],1e-11),
                                  ('beta',0.,1e-7),('formed',a['parent_formed'],0.),
                                  ('age',a['parent_age'],1e-11),('slip',a['parent_slip'],1e-11),
                                  ('V',0.,1e-11),('force',0.,1e-6)]:
            cmp.check('seed_strip/'+key,expected,b[key],atol,1e-7 if atol else 0.,ids)
        return
    a = read([donor/f'tip-planes-{tag}.csv'],['section'])
    b = read([target/f'sections-{tag}.csv'],['section'])
    ids = aligned(a,b,['section'])
    rates = read([donor/f'tip-sections-{tag}.csv'])
    np.testing.assert_array_equal(a['xi'],rates['xi'])
    for k in ('xi','formed','ds','beta','age','slip','V','force'):
        atol = {'formed':0.,'beta':1e-7,'force':1e-6}.get(k,1e-11)
        cmp.check('strip/'+k,(a if k in a else rates)[k],b[k],atol,1e-7 if atol else 0.,ids)
    cmp.check('strip/reference_width',a['reference_dxi'],b['ds']*np.cos(b['beta']),1e-14,ids=ids)
    book = read([donor/'bulk-accepted.csv'])
    keep = book['step']==step
    if keep.sum()!=1:
        raise ValueError(f'Missing ledger at {step}')
    b = read([target/f'balance-{tag}.csv'])
    for k in ('time','bulk','cohesive','fracture','Wext','Db','Dc','Df','phase_remainder','phase_work','energy_change'):
        cmp.check('ledger/'+k,book[k][keep],b[k],1e-15 if k=='time' else 1e-11,0 if k=='time' else 1e-6)
    for k,limit in [('mechanical_residual',1e-9),('phase_KKT',1e-7),('adjoint_error',1e-12)]:
        cmp.check('gate/'+k,0.,b[k],limit)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',type=int)
    for name in ('target','donor','replay','report'):
        p.add_argument(name,type=Path)
    a = p.parse_args()
    result = Comparison()
    compare(a.step,a.target,a.donor,a.replay,result)
    raise SystemExit(bool(result.write(a.report)))
