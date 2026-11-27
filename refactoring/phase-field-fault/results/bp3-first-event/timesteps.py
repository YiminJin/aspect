"""Identify every attempt, including a discarded endpoint, from ASPECT's coupled-step log."""
import csv
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'benchmarks/bp3_qd_thrust'))
from check_historical import profile
from history import read_statistics, slip_budget
from acceleration import budgets
import json

if __name__=='__main__':
    directory,log=map(Path,sys.argv[1:3])
    params=json.loads((directory/'parameters.json').read_text())
    budget=float(params['Time stepping']['Fault RSF']['Maximum logarithmic state change']['value'])
    factor=float(params['Time stepping']['Fault RSF'].get('Proposal safety factor',{'value':1.})['value'])
    clocks={float(r["BP3 clock (s)"]) for r in read_statistics(directory/"statistics")}
    profiles={t:data for t,data in (profile(p) for p in directory.glob('fault-profile-*.txt'))}
    cols='attempt_s,dt_s,accepted,proposal_s,criterion,id,s_m,actual_slip_Dc,actual_abs_log_age,max_slip_id,max_age_id,manager_dt_s,manager_limit,retry_reason,retry_id,retry_dt_s,raw_fault_limit_s,limiting_budget'.split(',')
    writer=csv.writer(sys.stdout,lineterminator="\n");writer.writerow(cols)
    row=None
    for line in log.read_text().splitlines():
        if line.startswith('Fault timestep at '):
            t=float(line.split()[3])
            v=dict(re.findall(r'(\w+)=([^ ]+)',line))
            row=[t,float(v['dt']),int(t in clocks and "retry_reason" not in v),float(v['proposal']),v['criterion'],int(v['id']),float(v['station_m']),
                 float(v['slip_fraction']),float(v['abs_log_age']),int(v['slip_id']),int(v['age_id'])]
            if row[2] and t in profiles:
                data=profiles[t]
                caps=np.array([slip_budget(params,s) for s in data['s_m']])
                i,why,dt,*_=budgets(data['rate_m_s'],data['age_s'],budget,caps)
                # Distinguish roundoff-level station ties from a different physical limit.
                assert why==row[4] and np.isclose(factor*dt,row[3],rtol=4e-14,atol=1e-8),(t,dt,row)
                assert abs(data['s_m'][row[5]]-row[6])<1e-7
                if row[1]>0:
                    parent=min(profiles,key=lambda x:abs(x-(t-row[1])))
                    assert abs(parent-(t-row[1]))<1e-8
                    change=np.abs(np.log(data['age_s']/profiles[parent]['age_s']))
                    slip=np.abs((data['slip_m']-profiles[parent]['slip_m'])/.008)
                    slip=np.where(np.isfinite(caps),slip,0.)
                    assert abs(max(change)-row[8])<1e-12 and abs(max(slip)-row[7])<1e-12
                    assert abs(change[row[10]]-row[8])<1e-12 and abs(slip[row[9]]-row[7])<1e-12
        elif line.startswith('Fault next dt='):
            manager=dict(re.findall(r'(\w+)=([^ ]+)',line))
            writer.writerow(row+[manager['dt'],manager['limiter'],v.get('retry_reason',''),v.get('retry_id',''),v.get('retry_dt',''),
                                 v.get('raw_limit',v['proposal']),v.get('limiting_budget','')])
