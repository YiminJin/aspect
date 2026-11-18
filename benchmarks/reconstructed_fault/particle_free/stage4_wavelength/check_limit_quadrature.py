#!/usr/bin/env python3
"""Integrate the limit on cuts matching physical interpolation/profile knots."""
from shortwave import *
cuts=np.unique(np.r_[np.arange(40,89)/16.,sections['x'][(sections['x']>2.5)&(sections['x']<5.5)],
                    operating['x'][(operating['x']>2.5)&(operating['x']<5.5)]])
records=[]
for order,normal in ((3,10),(6,10),(10,10),(10,16)):
    z,w=np.polynomial.legendre.leggauss(order)
    x=(.5*(cuts[:-1]+cuts[1:])[:,None]+.5*np.diff(cuts)[:,None]*z).ravel()
    weights=(.5*np.diff(cuts)[:,None]*w).ravel()
    C=[];Q2=[]
    for v in x:
        c0,cm,cc=limits(v,128,normal);C.append(cc);Q2.append(c0+cm)
    C=np.array(C);Q2=np.array(Q2)
    measure=weights/np.cos(np.interp(x,sections['x'],sections['angle']))
    for name,values in [('continuum',C),('normal_Q2',Q2)]:
        record=dict(x_order=order,normal_order=normal,x_intervals=len(cuts)-1,model=name,
                    **carrier_root(x,measure,values,values*values));records.append(record)
        print(json.dumps(record),flush=True)
for name in ('continuum','normal_Q2'):
    a=next(r for r in records if r['model']==name and r['x_order']==6)
    b=next(r for r in records if r['model']==name and r['x_order']==10 and r['normal_order']==10)
    c=next(r for r in records if r['model']==name and r['normal_order']==16)
    assert abs(a['growth']-b['growth'])<1e-6
    assert abs(c['growth']-b['growth'])<1e-6
(OUT/'limit-quadrature.json').write_text(json.dumps(records,indent=2)+'\n')
