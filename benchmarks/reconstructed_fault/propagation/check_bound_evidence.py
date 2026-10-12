#!/usr/bin/env python3
"""Independent numerical checks; bound KKT success does not imply no healing."""
import csv
import json
import math
from pathlib import Path
import re
import sys


def rows(path, name_key):
    with path.open() as stream:
        data=[{k:v if k==name_key else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    assert data and all(math.isfinite(v) for r in data for k,v in r.items() if k!=name_key)
    return data


def assess_bounds(data):
    assert [r['stage'] for r in data]==['relax_commit','hold','hold_repeat','ramp_05','ramp_05_repeat','ramp_10','unload']
    for r in data:
        assert all(math.isfinite(v) for k,v in r.items() if k!='stage')
        assert min(r[k] for k in ['free_residual','dual_violation','complementarity','kkt','correction'])>=0
        assert 0<=r['node_min']<=r['node_max']<=1
        cpdi_roundoff=r.get('cpdi_roundoff',0)
        assert cpdi_roundoff<=512*math.ulp(1.0)
        assert -cpdi_roundoff<=r['cpdi_min']<=r['cpdi_max']<=1+cpdi_roundoff
        assert max(r['free_residual'],r['dual_violation'],r['complementarity'],r['kkt'])<=r['target']
        assert r['H_increment_min']>=0 and r['equality_residual']<=r['equality_target']
    repeat=max(r['change'] for r in data if r['stage'] in ['hold','hold_repeat','ramp_05_repeat','unload'])
    correction=max(r['correction'] for r in data)
    allowance=10*max(repeat,correction,math.ulp(1.0))
    ramps=[r for r in data if r['stage'] in ['ramp_05','ramp_10']]
    resolved=min(r['change'] for r in ramps)>100*allowance
    healing=max(r['healing'] for r in ramps)
    repeat_consistent=repeat<=10*max(correction,math.ulp(1.0))
    return dict(box_kkt_pass=True,repeat_drift=repeat,correction_estimate=correction,
                phase_allowance=allowance,repeat_consistent=repeat_consistent,
                resolved_evolution=resolved,max_incremental_healing=healing,
                no_resolved_healing=healing<=allowance,
                restricted_prescribed_H_pass=repeat_consistent and resolved and healing<=allowance,
                production_solver_qualified=False,mechanical_H_qualified=False)


def check(evidence):
    result={'reference':{},'bounds':{}}
    for case in ['reference32','reference64','bounds_straight_final','bounds_padded_final']:
        run=json.loads((evidence/(case+'.run.json')).read_text())
        log=(evidence/(case+'.log')).read_text()
        assert run['exit_code']==0 and run['seconds']<120 and run['peak_rss_kib']<2*1024*1024
        assert 'P1 supplied scratch state and normal/exception restoration verified' in log
        if case.startswith('reference'):
            summary=rows(evidence/(case+'.reference-summary.csv'),'name')
            assert all(r['correction']<2e-12 and r['residual']<2e-9 for r in summary)
            assert summary[0]['name']=='cpdi_reduction' and summary[0]['cpdi_node_difference']<1e-11
            cuts=rows(evidence/(case+'.reference-cuts.csv'),None)
            errors={}
            for kind in ['sampled','native']:
                a='fine_'+kind+'_4096';b='fine_'+kind+'_8192'
                errors[kind+'_global_repeat']=max(abs(r[a]-r[b]) for r in cuts)
                errors[kind+'_outer_tail_repeat']=max(abs(r[a]-r[b]) for r in cuts if r['y']<.188664692 or r['y']>.811335308)
            sampled=next(r for r in summary if r['name']=='fine_sampled_8192')['minimum']
            assert -sampled>100*errors['sampled_outer_tail_repeat']
            assert next(r for r in summary if r['name']=='fine_native_8192')['minimum']>=0
            scalars={k:float(v) for k,v in re.findall(r'(\w+)=([\deE+.-]+)',re.search(r'P1 reference cells=.*',log)[0])}
            assert abs(scalars['L_tail']**2-scalars['tail_diffusion']/scalars['tail_tangent'])<1e-15
            result['reference'][case]=dict(scalars,**errors,solves=summary)
        else:
            data=rows(evidence/(case+'.bounds.csv'),'stage')
            result['bounds'][case]=dict(assess_bounds(data),stages=data)
    return result


if __name__=='__main__':
    evidence=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'bounds_evidence'
    result=check(evidence)
    print(json.dumps(result,indent=2))
    sys.exit(0 if all(r['restricted_prescribed_H_pass'] for r in result['bounds'].values()) else 2)
