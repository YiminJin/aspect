#!/usr/bin/env python3
"""Independent positive-weight moments; no projection or lost residual history."""
import argparse,json
from pathlib import Path
import numpy as np


def assess(path):
    data=np.genfromtxt(path,delimiter=',',names=True);result=[]
    for family in range(3):
        d=data[data['family']==family];w=d['weight'];phi=d['phi']
        h=128*phi*(1+phi)/(1-phi)**2;a=d['a'];q=d['q'];rate=d['w']
        I=w@h;se=w@a
        if I==0:
            assert np.all(a==0) and np.all(d['normalized']==0)
            result.append(dict(family=family,intact_no_normalized_record=True));continue
        rho=a-h/I*se
        positive=h>0
        Ec=np.sum(w[positive]*1e6*a[positive]**2/(2*h[positive]))
        Es=1e6*se**2/(2*I)
        Er=np.sum(w[positive]*1e6*rho[positive]**2/(2*h[positive]))
        Dc=2*1e6/1e8*Ec;Ds=2*1e6/1e8*Es;Dr=2*1e6/1e8*Er
        Df=1000*(w@abs(rate));Fs=1000*abs(w@rate);Fr=Df-Fs
        values=dict(I=I,se=se,Ec=Ec,Es=Es,Er=Er,Dc=Dc,Ds=Ds,Dr=Dr,Df=Df,Fs=Fs,Fr=Fr,moment=w@rho)
        error=max(float(abs(d[k][0]-v)/max(1,abs(v))) for k,v in values.items())
        assert error<1e-10 and abs(Ec-Es-Er)<1e-12 and abs(Dc-Ds-Dr)<1e-12
        if family==1:assert Er>0 and Fr>0
        result.append(dict(family=family,maximum_scaled_defect=error,**{k:float(v) for k,v in values.items()}))
    return dict(profiles=result,surface_energy_added_twice=False,
                identity_measure='raw positive normal weights; not projected nodal I')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=assess(a.csv);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
