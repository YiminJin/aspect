#!/usr/bin/env python3
"""Read-only bounded-profile lifecycle checks against retained paired exports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
import argparse
import csv
from assess_material_lifecycle import compare,RUNS,ROOT
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--default-only',action='store_true');p.add_argument('--full-support',action='store_true');args=p.parse_args()
    result={}
    prefix="bpf" if args.full_support else "bp"
    for c in ([] if args.default_only else ['c','l']):
        baseline=f'{prefix}-c-p-m-base-live' if c=='c' else f'{prefix}-l-p-m-base'
        result[c+'_restart']=compare(baseline,f'{prefix}-{c}-p-m-restart',[2] if args.full_support else [3,4])
        result[c+'_rejection']=compare(f'{prefix}-{c}-p-m-schedule',f'{prefix}-{c}-p-m-reject',[0,1,2,3])
        counters={}
        for suffix in ['reject','schedule','checkpoint','restart']:
            name=f'{prefix}-{c}-p-m-{suffix}'
            with (RUNS/name/'output/material-summary.csv').open() as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                assert int(row['history_publications'])==int(row['step']), (name,row)
                assert int(row['geometry_publications'])==int(row['step']), (name,row)
            assert int(rows[-1]['retries'])==(1 if suffix=='reject' else 0),(name,rows[-1])
            counters[suffix]=rows
        result[c+'_publication_counters']=counters
        identity={}
        for p in (RUNS/f'{prefix}-{c}-p-m-reject/output').glob('material-*.csv'):
            if p.name!='material-summary.csv':identity[p.name]=p.read_bytes()==(RUNS/f'{prefix}-{c}-p-m-schedule/output'/p.name).read_bytes()
        result[c+'_rejection_byte_identity']=identity
    if (RUNS/'bp-default-control/command.json').exists():
        result['default_path']=compare('matched-positive','bp-default-control',[0,1,2,3,4])
        result['default_byte_identity']={p.name:p.read_bytes()==(RUNS/'bp-default-control/output'/p.name).read_bytes()
            for p in (RUNS/'matched-positive/output').glob('material-*.csv')}
    if (RUNS/'bp-entry-default-control/command.json').exists():
        result['entry_pair_default']=compare('bp-entry-default-control','bp-default-control',[0,1,2,3,4])
        result['entry_pair_byte_identity']={p.name:p.read_bytes()==(RUNS/'bp-default-control/output'/p.name).read_bytes()
            for p in (RUNS/'bp-entry-default-control/output').glob('material-*.csv')}
    result['bound']='Each numeric column <=1e-12*max(1,max(abs(control))); exact labels and point keys.'
    result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in [f'{prefix}-*-reject',f'{prefix}-*-schedule',f'{prefix}-*-restart','bp-default-control','bp-entry-default-control']
        for folder in RUNS.glob(pattern) for p in folder.glob('output/material-*.csv')}
    (OUT/('default-regression.json' if args.default_only else 'full-support-lifecycle.json' if args.full_support else 'lifecycle.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('Bounded-profile rejection/restart/default comparisons pass.')

if __name__=='__main__':main()
