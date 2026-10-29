#!/usr/bin/env python3
"""Collect bounded native acceptance, publication, cost and physical comparisons."""
from pathlib import Path
import argparse,json,hashlib,csv
import numpy as np
from p5c_analysis import read,save
from p5c_specimen_analysis import grid
ROOT=Path(__file__).resolve().parents[4]
def main(work,out):
    result={'runs':{},'protected':{},'model_difference':{}}
    for name in ['seed-fixed','seed-radial','seed-fixed-half','seed-radial-half']:
        run=work/'runs'/name;a=read(run/'bulk-accepted.csv');s=read(run/'bulk-stability.csv');p=read(run/'bulk-profile.csv');r=read(run/'resources.csv');g=read(run/'gates.csv')
        assert np.array_equal(a['publications'],a['step']) and all(g['passed']==1) and all(s['stable']==1)
        assert max(a['mechanical_residual'])<=1e-9 and max(a['phase_KKT'])<=1e-7 and max(a['projection_error'])<=1e-12
        bound=1e-10*np.maximum.reduce([np.ones(len(a)),abs(a['Wext']),a['bulk']+a['cohesive']+a['fracture']]);assert max(abs(a['ledger_error'])/bound)<=1
        with (run/'resources.csv').open() as stream: fingerprints={row['particle_fingerprint'] for row in csv.DictReader(stream)}
        assert max(s['critical_masters'])<=1024 and len(fingerprints)==1 and all(r['freeze']==1)
        cmd=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('p5c-'+name+'.json');c=json.loads(cmd.read_text());assert c['exit_code']==0
        nonempty=s[s['critical_masters']>0]
        result['runs'][name]={'wall_s':c['seconds'],'accepted_intervals_this_run':len(a),'first_time_s':float(a['time'][0]),'last_time_s':float(a['time'][-1]),'first_publication':int(a['publications'][0]),'last_publication':int(a['publications'][-1]),'max_source_error':float(max(a['projection_error'])),'max_mechanical_residual':float(max(a['mechanical_residual'])),'max_phase_KKT':float(max(a['phase_KKT'])),'max_abs_BE_ledger_error_J_m':float(max(abs(a['ledger_error']))),'max_ledger_bound_ratio':float(max(abs(a['ledger_error'])/bound)),'max_critical_masters':int(max(s['critical_masters'])),'minimum_nonempty_relaxed_curvature':float(min(nonempty['minimum_curvature'])) if len(nonempty) else None,'stability_retries':int(sum(p['stability_retries'])),'native_solve_s':float(sum(p['total'])),'assembly_s':float(sum(p['assembly'])),'communication_s':float(sum(p['communication'])),'linear_s':float(sum(p['linear'])),'history_s':float(sum(p['history'])),'stability_s':float(sum(p['stability'])),'max_rank_peak_KiB':int(max(r['max_rank_KiB'])),'particles':int(r['particles'][0]),'particle_fingerprint':int(next(iter(fingerprints))),'freeze':True}
    base=json.loads((ROOT/'build-particle-free/p5b-native/entry.json').read_text())
    for f,h in base['protected'].items():
        p=ROOT/f;actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==h,f;result['protected'][f]=h
    for step in [107,112,116]:
        aa=read(work/'runs/seed-fixed'/f'means-{step:04}.csv');bb=read(work/'runs/seed-radial'/f'means-{step:04}.csv');assert np.array_equal(aa['cell'],bb['cell']);w=aa['area']
        def metric(a,b,c):return a*a+b*b+(a+b)**2+2*c*c
        n=metric(aa['tauxx']-bb['tauxx'],aa['tauyy']-bb['tauyy'],aa['tauxy']-bb['tauxy']);d=metric(aa['tauxx'],aa['tauyy'],aa['tauxy'])
        item={'full_tensor_stress_relative_L2_difference_vs_fixed':float(np.sqrt(w@n/(w@d))),'Y_relative_L2_difference_vs_fixed':float(np.sqrt(w@(aa['Y']-bb['Y'])**2/(w@(aa['Y']**2))))}
        for mode in ['fixed','radial']:
            a=read(work/'runs'/('seed-'+mode)/f'phase-{step:04}.csv');ini=read(work/'runs'/('seed-'+mode)/'phase-0000.csv');x,y,p=grid(a,'phi');delta=p-grid(ini,'phi')[2];active=delta>1e-8;X,Y=np.meshgrid(x,y)
            item[mode]={'growth_symmetry_max_defect':float(np.max(abs(delta-np.flipud(delta)))),'growing_nodes':int(active.sum()),'growth_x_interval_m':[float(X[active].min()),float(X[active].max())] if active.any() else None,'growth_y_interval_m':[float(Y[active].min()),float(Y[active].max())] if active.any() else None}
        result['model_difference'][str(step)]=item
    save(out/'native-verification.json',result)
    print(json.dumps(result['runs'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.work,a.out)
