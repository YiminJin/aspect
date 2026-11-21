"""Collect timing pairs, endpoint differences and physical geometry checks from native controls."""
import json
import re
from pathlib import Path

import numpy as np

from compare import covariance,read
from localized_analysis import strip_check

root=Path(__file__).resolve().parents[4]/'build-particle-free/bulk-tip/runs'

def run(label):
    return root/('localized-'+label)

def timing(label):
    log=(run(label)/'run.log').read_text(errors='replace')
    rows=re.findall(r'Tip solver timing: step=(\d+), wall=([\d.e+-]+), direction=([\d.e+-]+), scores=(\d+), RSF=(\d+)',log)
    assert rows and len({r[0] for r in rows})==1
    return np.array(rows,dtype=float).sum(axis=0)[1:]

def pair(a,b,step):
    wall,direct,score,rsf=timing(a)
    fixed=timing(b)[0]
    errors=covariance(run(a)/'output',run(b)/'output',0,step)
    return dict(search=wall,fixed=fixed,direct=direct,r_direct=direct/wall,r_overhead=(wall-fixed)/wall,
                scores=score,rsf_solves=rsf,endpoint_errors=errors)

def state(label,step):
    folder=run(label)/'output'
    section=read(folder/f'tip-sections-{step:04}.csv')
    ledger=read(folder/'bulk-accepted.csv')[-1]
    phase0=read(folder/'tip-initial.csv')[0]['fracture']
    return dict(slip=section['ds']@section['slip'],max_rate=abs(section['V']).max(),
                added_fracture=ledger['fracture']-phase0,work=ledger['Wext'],
                bulk=ledger['bulk'],cohesive=ledger['cohesive'],ledger=ledger['ledger_error'])

def screening_results():
    pairs={}
    for i in range(1,5): pairs[f'h{i}']=pair(f'v{i}',f'vf{i}',i)
    for i in range(1,4): pairs[f'o{i}']=pair(f'vo{i}',f'vof{i}',i)
    median=lambda key:float(np.median([r[key] for r in pairs.values()]))
    largest=lambda key:max(r[key] for r in pairs.values())
    out=dict(timing=pairs,typical_direct=median('r_direct'),worst_direct=largest('r_direct'),
             typical_overhead=median('r_overhead'),worst_overhead=largest('r_overhead'))
    out['forced_rollbacks']=pair('vfront','vfrontf',2)
    out['coulomb']={str(i):pair(f'cr{i}',f'crf{i}',i) for i in (2,3)}
    out['coulomb_repeats']={n:pair(a,b,i) for n,a,b,i in (
        ('first2','c2','cf2',2),('first3','c3','cf3',3),('warm2','cc2','ccf2',2),
        ('reverse2','cd2','cdf2',2),('warm3','cd3','cdf3',3))}
    out['repeat']={}
    for i in range(1,5): out['repeat'][f'h{i}']=pair(f'q{i}',f'q{i}f',i)
    for i in range(1,4): out['repeat'][f'o{i}']=pair(f'qo{i}',f'qo{i}f',i)
    out['rotation']={str(a):covariance(run('v1')/'output',run(n)/'output',a,1) for a,n in [(30,'vrot'),(90,'vvert')]}
    out['restart']=covariance(run('vcut')/'output',run('vrestart')/'output',0,3)
    out['strip']=strip_check(run('vcut')/'output',30)
    out['physical']={n:state(n,3) for n in ('vcut','vexact')}
    out['max_power_defect']=max(abs(read(run('vcut')/'output/tip-power.csv')['power_adjoint_error']))
    out['tangent_error']=read(run('vcut')/'output/bulk-tangent.csv')[0]['relative_action_error']
    figures={}
    for n in ('vcut','vexact','v1','vo1'):
        folder=run(n)/'output/restart/02'
        figures[n]=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    out['checkpoint_bytes']=figures
    return out

def single_front_results():
    timing={}
    for i in range(1,5): timing[f'h{i}']=pair(f'one{i}',f'onef{i}',i)
    for i in range(1,4): timing[f'o{i}']=pair(f'oneo{i}',f'oneof{i}',i)
    def output(n): return run(n)/'output'
    return dict(timing=timing,forced_rollbacks=pair('onefront','onefrontf',2),
                repeat_oblique=pair('oneros','onerof',3),
                repeat_rollbacks=[pair('onerfs','onerff',2),pair('retrythird','retrythirdfixed',2)],
                coulomb={str(i):pair(f'onec{i}',f'onecf{i}',i) for i in (2,3)},
                physical={n:state(n,3) for n in ('onecut','oneline')},
                strip=strip_check(output('onecut'),30),
                tangent_error=read(output('onecut')/'bulk-tangent.csv')[0]['relative_action_error'],
                work_adjoint=max(abs(read(output('onecut')/'tip-power.csv')['power_adjoint_error'])),
                restart=covariance(output('onecut'),output('oneresume'),0,3),
                rotation={str(a):covariance(output('one1'),output('one'+str(a)),a,1) for a in (30,90)})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screening',action='store_true',help='Read the earlier two-ended screening controls')
    print(json.dumps(screening_results() if parser.parse_args().screening else single_front_results(),indent=2))
