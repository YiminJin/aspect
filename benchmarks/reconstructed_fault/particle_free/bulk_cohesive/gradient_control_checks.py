#!/usr/bin/env python3
"""Unique-branch controls, kept separate from the preserved ambiguous case."""
import csv,json
import numpy as np
from gradient_point import ROOT,OUT,point,initial,ledger
rows=list(csv.DictReader((ROOT/'build-particle-free/coulomb-localization/point-states.csv').open()))
identity=0.;fd=0.;power=0.;work=0.;active=set();records=[];worst={}
groups=[list(range(10)),[11,12,13],[14,15,16],[17,18,19],[20,21,22],[24,25,26,27],[28,29,30]]
# Retain coulomb_points.cc physical scales and probe sizes, rather than a
# dimensionally mixed norm of derivatives per UNIT strain or G Pa pressure.
scales=np.array([1000,1000,1000,.001,.001,.001,.001,1000,1000,.001,1,
                 1000,1000,1000,.001,.001,.001,8,1,1,.01,.01,.01,1,1,1,1000,1000,1000,1000,.001])
for phi in [0.,.2,.6,.75,.8,.9,.95,.99]:
    for sign in [1.,-1.]:
        old=initial(phi)
        for step in range(1,25):
            rate=np.array([0.,0.,.02]) if step<=8 else (np.array([sign*.02,-sign*.02,0.]) if step<=20 else np.zeros(3))
            de=.0125*rate;r,_=point(old,de,.0125,jac=False)
            ref=next(v for v in rows if v['kind']=='path' and float(v['phi'])==phi and float(v['normal_sign'])==sign and int(v['step'])==step)
            for i,k in [(0,'Txx'),(1,'Tyy'),(2,'Txy'),(3,'Axx'),(4,'Axy'),(7,'Qxx'),(8,'Qxy'),(11,'tauxx'),(12,'tauyy'),(13,'tauxy'),(14,'Wxx'),(15,'Wxy'),(17,'Y')]:identity=max(identity,abs(r[i]-float(ref[k]))/max(1.,abs(float(ref[k]))))
            if phi in [.2,.8,.9,.99] and step in [7,10,16]:
                for X in [np.array([30.,-40.]),np.array([-50.,15.]),np.array([0.,40.])]:records.append((old.copy(),de.copy(),X,step,sign))
            old=r[:11].copy()
for phi in [.2,.8,.9,.99]:records.append((initial(phi),np.zeros(3),np.array([30.,-40.]),0,1.))
evidence=[]
for old,de,X,step,sign in records:
    unchanged=old.tobytes();r,J=point(old,de,.0125,X=X);repeat,_=point(old,de,.0125,X=X,jac=False)
    assert old.tobytes()==unchanged and repeat.tobytes()==r.tobytes()
    active.add(int(r[23]));error=0.
    for c in range(6):
        epsilon=1e-5;probe=.0003;eps=epsilon*probe
        plus=de.copy();minus=de.copy();pp=pm=10000.;xp=X.copy();xm=X.copy()
        if c<3:plus[c]+=eps;minus[c]-=eps
        elif c==3:pp+=1e6*eps;pm-=1e6*eps
        else:xp[c-4]+=1e6*eps;xm[c-4]-=1e6*eps
        rp,_=point(old,plus,.0125,pp,xp,jac=False);rm,_=point(old,minus,.0125,pm,xm,jac=False)
        assert int(rp[23])==int(rm[23])==int(r[23]),('control crossed active set',old[10],step,X,c)
        d=rp-rm;d[24:26]=(d[24:26]+np.pi/2)%np.pi-np.pi/2;d/=2*epsilon
        for group in groups:
            ad=probe*J[group,c];defect=np.max(np.abs(d[group]-ad)/np.maximum(scales[group],np.abs(ad)));error=max(error,defect)
            if defect>fd:fd=defect;worst={'phi':old[10],'step':step,'X':X.tolist(),'column':c,'group':group,'defect':fd}
    h=4*old[10]*(1+old[10])/(1-old[10])**2;beta=1/(1+.0125/100)
    plane_power=np.sum(.1*r[26:28]*h*r[28:30]/(1e6*beta*.0125));pdef=abs(plane_power-r[22])/max(1.,abs(plane_power));power=max(power,pdef)
    account=ledger(old,r,de,.0125,X);work=max(work,abs(account['error']))
    evidence.append({'phi':old[10],'incoming_step':step,'sign':sign,'Xxx_Pa':X[0],'Xxy_Pa':X[1],'active':int(r[23]),
        'complete_FD_relative_error':error,'plane_power_defect':pdef,'micro_work_ledger_defect_Pa':account['error']})
summary={'zero_X_identity_max':identity,'cases':len(records),'active_masks':sorted(active),'complete_FD_relative_error':fd,'plane_power_defect':power,
         'micro_work_ledger_defect_Pa':work,'worst_FD':worst,'whole_candidate_qualified':False,'reason':'Distinct admissible roots at the separately preserved developed-damage negative-X state.'}
summary['FD_units']='Same physical output scales and epsilon=1e-5 as coulomb_points.cc; independent probes .0003 strain,300Pa pressure/backstress; bound1e-6 unchanged.'
(OUT/'unique-control-checks.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'unique-control-checks.csv').open('w') as f:
    w=csv.DictWriter(f,list(evidence[0]));w.writeheader();w.writerows(evidence)
assert identity<=1e-12 and fd<=1e-6 and power<=1e-12 and work<=1e-10,summary
assert active=={0,1,2,3},active
print(json.dumps(summary,indent=2))
