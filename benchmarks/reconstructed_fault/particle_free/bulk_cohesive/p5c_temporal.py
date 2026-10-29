#!/usr/bin/env python3
"""Focused dt/2 comparison after a same-mode accepted common prefix."""
from pathlib import Path
import argparse,csv,json,os
import numpy as np
from p5c_analysis import read,save
from p5c_specimen_analysis import grid,history

def main(work,out):
    summary={}
    for mode in ['fixed','radial']:
        coarse=work/'runs'/f'seed-{mode}';fine=work/'runs'/f'seed-{mode}-half'
        if not (fine/'bulk-accepted.csv').exists():continue
        A=read(coarse/'bulk-accepted.csv');B=read(fine/'bulk-accepted.csv')
        if int(B['step'][-1])!=137:continue
        at={round(float(r['time']),12):int(r['step']) for r in A};bt={round(float(r['time']),12):int(r['step']) for r in B}
        init=read(coarse/'phase-0000.csv');x,y,p=grid(init,'phi');X,Y=np.meshgrid(x,y);mass=np.ones_like(p)*(x[1]-x[0])*(y[1]-y[0]);mass[[0,-1],:]*=.5;mass[:,[0,-1]]*=.5
        rows=[]
        for t in sorted(set(at)&set(bt)):
            ca=read(coarse/f'phase-{at[t]:04}.csv');fb=read(fine/f'phase-{bt[t]:04}.csv');pa=grid(ca,'phi')[2];pb=grid(fb,'phi')[2]
            d=pa-pb;growth=pb-p;norm=np.sqrt(np.sum(mass*growth*growth));maska=X>=1.-.046875
            row={'time_s':t,'dt_coarse_s':.00125,'dt_fine_s':.000625,'growth_relative_L2':float(np.sqrt(np.sum(mass*d*d))/norm) if norm else None,'peak_difference_over_growth_peak':float(np.max(abs(d))/np.max(growth)) if growth.max() else None,'coarse_integrated_cap_growth_m2':float(np.sum(mass*(pa-p)*maska)),'fine_integrated_cap_growth_m2':float(np.sum(mass*growth*maska)),'maximum_growth_coarse':float((pa-p).max()),'maximum_growth_fine':float(growth.max())}
            ma=read(coarse/f'means-{at[t]:04}.csv');mb=read(fine/f'means-{bt[t]:04}.csv');assert np.array_equal(ma['cell'],mb['cell']);area=mb['area']
            def metric(xx,yy,xy):return xx*xx+yy*yy+(xx+yy)**2+2*xy*xy
            num=metric(ma['tauxx']-mb['tauxx'],ma['tauyy']-mb['tauyy'],ma['tauxy']-mb['tauxy']);den=metric(mb['tauxx'],mb['tauyy'],mb['tauxy']);row['full_tensor_stress_relative_L2']=float(np.sqrt(np.dot(area,num)/np.dot(area,den)))
            row['Y_relative_L2']=float(np.sqrt(np.dot(area,(ma['Y']-mb['Y'])**2)/max(np.dot(area,mb['Y']**2),1e-300)))
            rows.append(row)
        with (out/f'{mode}-temporal.csv').open('w') as f:
            c=csv.DictWriter(f,list(rows[0]));c.writeheader();c.writerows(rows)
        combined=work/'analysis'/f'{mode}-half';combined.mkdir(parents=True,exist_ok=True)
        for source in [*coarse.glob('phase-*.csv'),*coarse.glob('means-*.csv'),*coarse.glob('boundary-reactions-*.csv')]:
            if int(source.stem[-4:])<=95:
                target=combined/source.name
                if not target.exists():os.link(source,target)
        for source in [*fine.glob('phase-*.csv'),*fine.glob('means-*.csv'),*fine.glob('boundary-reactions-*.csv'),coarse/'seed-prescription.csv']:
            target=combined/source.name
            if not target.exists():os.link(source,target)
        # Analysis-only combined ledger; accepted clocks/solver files are never edited.
        prefix=(coarse/'bulk-accepted.csv').read_text().splitlines();selected=[prefix[0],*[line for line in prefix[1:] if int(line.split(',')[0])<=95],*(fine/'bulk-accepted.csv').read_text().splitlines()[1:]];(combined/'bulk-accepted.csv').write_text('\n'.join(selected)+'\n')
        h,_=history(combined,out,f'{mode}-half')
        summary[mode]={'common_accepted_prefix_s':.11875,'halved_window_s':[.11875,float(B['time'][-1])],'paired_times':len(rows),'final':rows[-1],'fine_response':h,'maximum_growth_relative_L2':max((r['growth_relative_L2'] or 0) for r in rows),'interpretation_target':.05}
    save(out/'temporal-summary.json',summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.work,a.out)
