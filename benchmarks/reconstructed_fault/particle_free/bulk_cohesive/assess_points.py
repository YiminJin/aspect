#!/usr/bin/env python3
"""Independent piecewise-convex return and exact fixed-profile rate solution."""
import argparse
import json
from pathlib import Path
import numpy as np


def h(phi):
    return 128*phi*(1+phi)/(1-phi)**2


def assess(path):
    rows=np.genfromtxt(path,delimiter=',',names=True)
    maxima=dict(return_a=0.,return_q=0.,yield_error=0.,ledger_error=0.)
    temporal={}
    for family in [0,1,2]:
        for dt in [.01,.005]:
            data=rows[(rows['family']==family)&(rows['dt']==dt)]
            a=T=k=L=0.
            exactT=exacta=exactk=0.
            errorT=errora=0.
            for row in data:
                phi=row['phi'];c=h(phi);G=row['G'];eta=row['eta'];R=row['resistance']
                beta=1/(1+dt*G/eta)
                T=beta*(T+G*row['dgamma'])
                abar=beta*a
                if c==0:
                    anew=0.;q=np.sign(T)*max(abs(T)-R,0.)
                else:
                    stiffness=G*(1+1/c)
                    candidates=[abar,(T-R)/stiffness,(T+R)/stiffness]
                    # Independent constrained minimization over the exact
                    # stationary points and the friction kink, not the kernel
                    # branch detector or its scaled-numerator implementation.
                    potential=lambda v:.5*G*(v-T/G)**2+G*v*v/(2*c)+R*abs(v-abar)
                    anew=min(candidates,key=potential);q=G*anew/c
                dk=(1+dt*G/eta)*anew-a
                if abs(anew-abar)<1e-16:dk=0.
                k+=dk;L+=abs(dk)
                maxima['return_a']=max(maxima['return_a'],abs(anew-row['a']))
                maxima['return_q']=max(maxima['return_q'],abs(q-row['q'])/1000)
                maxima['yield_error']=max(maxima['yield_error'],max(0,abs(row['stress']-row['q'])-R)/1000)
                maxima['ledger_error']=max(maxima['ledger_error'],abs(row['ledger_error']))
                assert abs(k-row['k'])<1e-12 and abs(L-row['L'])<1e-12
                a=anew
                if family in [0,1]:
                    decay=np.exp(-dt*G/eta)
                    exactT=decay*exactT+eta*(1-decay)*row['dgamma']/dt
                    if c==0:exacta=0.
                    else:
                        stiffness=G*(1+1/c)
                        exacta=np.clip(decay*exacta,(exactT-R)/stiffness,(exactT+R)/stiffness)
                    errorT=max(errorT,abs(exactT-row['Txy']))
                    errora=max(errora,abs(exacta-row['a']))
            if family in [0,1]:temporal[f'{family}:{dt}']={'stress_Pa':errorT,'a':errora}
    assert max(maxima.values())<1e-10
    ratios={str(family):temporal[f'{family}:0.01']['stress_Pa']/temporal[f'{family}:0.005']['stress_Pa'] for family in [0,1]}
    assert all(1.9<r<2.1 for r in ratios.values())
    virgin=rows[(rows['family']==0)&(abs(rows['Txy'])>2000.)]
    assert len(virgin)>0 and np.max(virgin['driving'])>64.
    return dict(samples=len(rows),independent_maximum_defects=maxima,
                fixed_profile_temporal_errors=temporal,stress_halving_ratios=ratios,
                virgin_above_threshold_positive_driving=True,
                maximum_virgin_driving_Pa=float(np.max(virgin['driving'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=assess(a.csv);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
