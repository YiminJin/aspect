"""Compare frozen-rate budgets on the archived donor acceleration; no trajectory integration."""
import csv
from pathlib import Path
import sys
import numpy as np
from math import inf

def budgets(rate,age,limit):
    steady=.008/abs(rate)
    change=steady/age-1
    with np.errstate(divide='ignore',invalid='ignore'):
        fraction=np.expm1(np.copysign(limit,change))/change
        bound=np.where((fraction>0)&(fraction<1),-steady*np.log1p(-fraction),inf)
    slip=.0008/abs(rate)
    i=np.argmin(np.minimum(slip,bound))
    return i,('age' if bound[i]<slip[i] else 'slip'),min(bound[i],slip[i]),bound.min(),slip.min()

if __name__=='__main__':
    w=csv.writer(sys.stdout,lineterminator="\n")
    w.writerow('step,time_s,peak_m_s,log_budget,criterion,down_dip_id,s_m,proposal_s,min_age_bound_s,min_slip_bound_s'.split(','))
    paths=[p for directory in sys.argv[1:] for p in Path(directory).glob('profiles/fault_*.csv')]
    assert len({p.name for p in paths})==len(paths)
    for path in sorted(paths,key=lambda p:int(p.stem.split('_')[1])):
        data=np.genfromtxt(path,delimiter=',',names=True)
        peak=data['V_m_per_s'].max()
        for budget in (.1,.2):
            i,why,dt,age,slip=budgets(data['V_m_per_s'],data['Theta_s'],budget)
            w.writerow([int(data['step'][0]),data['time_s'][0],peak,budget,why,len(data)-1-i,data['xd_m'][i],dt,age,slip])
