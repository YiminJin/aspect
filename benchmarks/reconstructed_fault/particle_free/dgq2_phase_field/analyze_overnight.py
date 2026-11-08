#!/usr/bin/env python3
"""Reduce the four frozen restored-build cases using existing field observers.

Run sequentially after simulations. No solve, parameter adjustment or replay.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):os.environ[k]='1'
import csv
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight'
os.environ['LAYERED_EVIDENCE']=str(OUT)
sys.path.insert(0,str(HERE.parent/'transported_objective'))
import assess_layered_campaign as assess
import compare_coupled as compare
assess.OUT=compare.OUT=OUT
os.environ['LAYERED_EVIDENCE']=str(OUT)
from layered_fields import powers,work_traction,localization,read
from assess_sheared_moving_fault import metric

NAMES={key:'layered-phase-integration-overnight-'+key.lower() for key in 'ABDC'}

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    start=time.monotonic();cases={};summaries={};ledger=[]
    queue=json.loads((OUT/'queue.json').read_text())
    assert queue['state']=='simulations-finished', 'No substantial reduction during simulations'
    for key,name in NAMES.items():
        decision=next((r for r in queue['results'] if r['case']==key),{})
        if not decision.get('completed'):continue
        t=time.monotonic();print('Reduce',key,flush=True)
        if (OUT/(name+'.json')).exists():c=json.loads((OUT/(name+'.json')).read_text())
        else:c=assess.reduce_case(name)
        assert abs(c['records'][-1]['time']-.8)<1e-10
        cases[key]=c;summaries[key]=compare.summarize(c)
        path,cmd,rows=assess.load_case(name)
        logs=(path/'run.log').read_text()
        spin=[{k:float(v) for k,v in re.findall(r'(\w+)=([^\s]+)',line)} for line in logs.splitlines() if 'Accepted bulk spin:' in line]
        native=summaries[key]['native_stage_balance'];assert len(native['records'])==len(rows)-1
        native['internal_dt']=[float(r['dt'])/q['substeps'] for r,q in zip(rows[1:],native['records'])]
        summaries[key]['accepted_bulk_spin']=spin
        summaries[key]['accumulated_dt_bulk_max']=sum(v['dt_bulk_max'] for v in spin)
        summaries[key]['particle_managers_zero']='Native integration: particle managers=0, chemical fields=0' in logs
        assert summaries[key]['particle_managers_zero']
        summaries[key]['rejected_line_search_trials']=sum(int(v) for v in re.findall(r'line search accepted after (\d+) rejected',logs))
        summaries[key]['linear_residuals']=[{k:float(v) for k,v in re.findall(r'(\w+)=([^,\s]+)',line)} for line in logs.splitlines() if 'Fault linear solve:' in line]
        summaries[key]['label_count']=len(c['records'][0]['curve'])
        ledger.append(dict(stage='reduce-'+key,seconds=time.monotonic()-t));save('reduction-ledger.json',ledger)
        save('summary-partial.json',summaries)
    pairs=[]
    for ka,kb in [('A','B'),('B','D'),('A','C')]:
        if ka not in cases or kb not in cases:continue
        a,b=cases[ka],cases[kb];shift=ka=='A' and kb=='C';print('Compare',ka,kb,flush=True);t=time.monotonic()
        pair=dict(a=ka,b=kb,drift_removed_for_comparison=shift,history=assess.compare_records(a,b,shift),selected_times=[])
        for physical in [0.,.2,.4,.6,.8]:
            print('  time',physical,flush=True)
            values=assess.compare_final(a,b,shift,physical)
            pa,ca,ra=assess.load_case(a['case']);pb,cb,rb=assess.load_case(b['case'])
            ar=next(r for r in a['records'] if abs(r['time']-physical)<1e-10);br=next(r for r in b['records'] if abs(r['time']-physical)<1e-10)
            sa=assess.state_at(pa,ca,ar['step']);sb=assess.state_at(pb,cb,br['step'])
            offset=(a['arguments']['translation']-b['arguments']['translation'])*physical if shift else 0.
            x,w=assess.common_bulk_quadrature(sa,sb,offset);xa=x.copy();xa[:,0]+=offset
            chia=localization(sa,xa,sa.at(xa))[2];chib=localization(sb,x,sb.at(x))[2]
            values['localization']=metric((chia-chib)**2,chib**2,w)
            # Scalars retain dimensional differences as well as reference values.
            values['profile_scalars']={}
            for field in ['profile_mass','peak_phase','profile_normal_variance','localization_measure']:
                av,bv=ar['energy'][field],br['energy'][field]
                values['profile_scalars'][field]=dict(a=av,b=bv,difference=av-bv,relative=(av-bv)/bv if bv else None)
            if physical==0 and not shift:
                values['initial_surface_Linf']=float(np.max(abs(sa.surface-sb.surface)))
                if ka=='A':
                    values['initial_coeff_Linf']=float(np.max(abs(sa.coeff-sb.coeff)))
                    assert values['initial_coeff_Linf']<1e-11 and values['initial_surface_Linf']<1e-11
            pair['selected_times'].append(dict(time=physical,values=values))
        pairs.append(pair);save('pair-'+ka+'-'+kb+'.json',pair)
        ledger.append(dict(stage='compare-'+ka+'-'+kb,seconds=time.monotonic()-t));save('reduction-ledger.json',ledger)
    quadrature={}
    for key in ['B','D']:
        if key not in cases:continue
        print('Frozen diagnostic quadrature',key,flush=True);t=time.monotonic();c=cases[key];path,cmd,rows=assess.load_case(c['case'])
        final=assess.state_at(path,cmd,int(rows[-1]['step']));old=assess.state_at(path,cmd,int(rows[-2]['step']));initial=assess.state_at(path,cmd,0)
        data={}
        for order in [8,12,16]:
            e=powers(final,old,initial,order);work,measure=work_traction(final,order)
            data[order]=dict(energy=e,work_traction=work.tolist(),localization_measure=measure)
        quadrature[key]=data
        save('frozen-quadrature.json',quadrature);ledger.append(dict(stage='frozen-quadrature-'+key,seconds=time.monotonic()-t));save('reduction-ledger.json',ledger)
    # The sharp finite localization ends move between the two fields. Check
    # the observer's common-cell integration sensitivity without another solve.
    pair_quadrature={}
    if 'B' in cases and 'D' in cases:
        for key in ['B','D']:
            path,cmd,rows=assess.load_case(cases[key]['case'])
            if key=='B':sa=assess.state_at(path,cmd,int(rows[-1]['step']))
            else:sb=assess.state_at(path,cmd,int(rows[-1]['step']))
        for order in [8,12,16]:
            x,w=assess.common_bulk_quadrature(sa,sb,0.,order)
            ca=localization(sa,x,sa.at(x))[2];cb=localization(sb,x,sb.at(x))[2]
            pair_quadrature[order]=metric((ca-cb)**2,cb**2,w)
        save('localization-pair-quadrature.json',pair_quadrature)
    result=dict(complete=len(cases)==4,cases=summaries,pairs=pairs,quadrature=quadrature,
                localization_pair_quadrature=pair_quadrature,reduction_seconds=time.monotonic()-start)
    save('comparison-summary.json',result)
    compare.write_summary_csv({NAMES[k]:v for k,v in summaries.items()},OUT)
    metrics=[]
    for pair in pairs:
        for entry in pair['selected_times']:
            values=entry['values']
            for region,fields in values['surface_common_quadrature'].items():
                for field,val in fields.items():metrics.append(dict(pair=pair['a']+'/'+pair['b'],time=entry['time'],region=region,field=field,**val))
            for field in ['common_domain','phase','localization','pressure']:
                metrics.append(dict(pair=pair['a']+'/'+pair['b'],time=entry['time'],region='whole',field=field,**values[field]))
    if metrics:
        with (OUT/'comparison-norms.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    from plot_overnight import plot
    plot(cases,result,OUT)
    save('reduction-ledger.json',ledger+[dict(stage='total',seconds=time.monotonic()-start)])

if __name__=='__main__':main()
