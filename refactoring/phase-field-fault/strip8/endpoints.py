"""Audit every 5A comparison column, retaining failures from every physical clock."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'benchmarks/growing_fault'))
from compare import rows


def audit(reference, target, label, writer):
    ledger = rows([reference/'bulk-accepted.csv'], ('step',))
    actual = rows(target.glob('balance-000[1-9].csv'), ('step',))
    if set(ledger) != {(1,),(2,),(3,)} or ledger.keys()!=actual.keys():
        raise ValueError(f'{label}: publication clocks differ')
    def column(step, name, keys, a, b, atol, rtol):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if not (len(a)==len(b)==len(keys) and len(a) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError(f'{label}, step {step}, {name}: incomplete or nonfinite column')
        diff=abs(b-a)
        tol=atol+rtol*abs(a)
        bad=diff>tol
        ratio=np.divide(diff,tol,out=np.where(diff==0,0.,np.inf),where=tol>0)
        worst=int(np.argmax(ratio))
        writer.writerow([label,step,name,len(keys),sum(bad),max(diff),ratio[worst],keys[worst],a[worst],b[worst]])
    def compare(step, name, old, new, atol, rtol, new_name=None):
        if old.keys()!=new.keys(): raise ValueError(f'{label}, step {step}, {name}: identities differ')
        keys=sorted(old)
        column(step,name,keys,[old[k][name] for k in keys],[new[k][new_name or name] for k in keys],atol,rtol)
    for step in range(4):
        tag=f'{step:04}'
        old=rows(reference.glob(f'tip-points-{tag}-r*.csv*'),('cell','q'))
        new=rows(target.glob(f'points-{tag}-r*.csv'),('cell','q'))
        for name in ('section','x','y','weight','phi','tauxx','tauyy','tauzz','tauxy',
                     'Axx','Ayy','Azz','Axy','signed_shear','absolute_shear'):
            atol=(1e-6 if name.startswith('tau') else 1e-9 if name=='phi' else
                  1e-10 if name.startswith(('A','signed','absolute')) else 1e-14)
            compare(step,name,old,new,atol,1e-7 if atol>=1e-10 else 0.)
        if step:
            # Extra stage observations where both layouts expose the quantity.
            for key,donor_key,tol in [('phi','stage_phi',1e-9),('tauxy','stage_tauxy',1e-6),('Axy','stage_Axy',1e-10)]:
                ids=sorted(old)
                column(step,donor_key,ids,[old[k][donor_key] for k in ids],
                       [.5*(float(new[k][key])+float(parent[k][key])) for k in ids],tol,1e-7)
        parent=new
        if not step: continue
        planes=rows([reference/f'tip-planes-{tag}.csv'],('section',))
        sections=rows([target/f'sections-{tag}.csv'],('section',))
        incoming=rows([target/f'sections-{step-1:04}.csv'],('section',))
        with (reference/f'tip-sections-{tag}.csv').open() as stream:
            rates={(i,):r for i,r in enumerate(csv.DictReader(stream))}
        compare(step,'parent_formed',planes,incoming,0.,0.,'formed')
        compare(step,'formed',planes,sections,0.,0.)
        for name in ('xi','ds','beta','age','slip'):
            compare(step,name,planes,sections,1e-7 if name=='beta' else 1e-11,1e-7)
        for name,atol in [('V',1e-11),('force',1e-6)]:
            compare(step,name,rates,sections,atol,1e-7)
        for name,other in [('parent_age','age'),('parent_slip','slip')]:
            compare(step,name,planes,incoming,1e-11,1e-7,other)
        key=(step,)
        for name in ('time','bulk','cohesive','Wext','Db','Dc','Df','phase_remainder'):
            compare(step,name,{key:ledger[key]},{key:actual[key]},1e-15 if name=='time' else 1e-11,0 if name=='time' else 1e-6)
        if step>1:
            def added(table):
                return float(table[key]['fracture'])-float(table[step-1,]['fracture'])
            column(step,'added_fracture',[key],[added(ledger)],[added(actual)],1e-11,1e-6)
        for name,tol in [('mechanical_residual',1e-9),('phase_KKT',1e-7),('adjoint_error',1e-12)]:
            column(step,name,[key],[0.],[actual[key][name]],tol,0.)
        def slip(table):
            return sum(float(r['slip'])*float(r['ds']) for r in table.values())
        # Report derived observations without inventing comparison tolerances.
        def observed(name, a, b):
            if not np.isfinite([a,b]).all(): raise ValueError(f'Nonfinite {name}')
            writer.writerow([label,step,name,1,'',abs(b-a),'',key,a,b])
        observed('integrated_slip',slip(planes),slip(sections))
        for name in ('energy_change','phase_work','fracture'):
            observed(name,float(ledger[key][name]),float(actual[key][name]))
        def defect(book):
            r=book[key]
            return float(r['Wext'])-sum(float(r[n]) for n in ('energy_change','Db','Dc','Df'))
        observed('physical_work_defect',defect(ledger),defect(actual))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('old',type=Path)
    ap.add_argument('csv',type=Path)
    args=ap.parse_args()
    with args.csv.open('w') as f:
        w=csv.writer(f)
        w.writerow('layout step column entries failed max_abs max_tolerance_ratio worst_id ref target'.split())
        for ref,case in [('donor-onecut',c) for c in ('review','release1','debug','debug2')]+[('donor-serial','release1')]:
            audit(args.old/ref/'output',args.old/case/'principal/output',ref+'_'+case,w)
