#!/usr/bin/env python3
"""Independent physical force/work and complete fixed-phase action checks."""
import csv,json
import numpy as np
from gradient_point import ROOT,WORK,OUT,NAMES,point,initial,tensor,ledger

rows=list(csv.DictReader((ROOT/'build-particle-free/coulomb-localization/point-states.csv').open()))
identity=0.;fd=0.;power=0.;energy=0.;cases=0;active=set();evidence=[];worst={};oracle=[];boundaries=[]
groups=[list(range(10)),[11,12,13],[14,15,16],[17,18,19],[20,21,22],[24,25,26,27],[28,29,30]]
incoming=[]
for phi in [0.,.2,.6,.75,.8,.9,.95,.99]:
    for sign in [1.,-1.]:
        old=initial(phi)
        for step in range(1,25):
            rate=np.array([0.,0.,.02]) if step<=8 else (np.array([sign*.02,-sign*.02,0.]) if step<=20 else np.zeros(3))
            de=.0125*rate;r,_=point(old,de,.0125,jac=False)
            ref=next(v for v in rows if v['kind']=='path' and float(v['phi'])==phi and float(v['normal_sign'])==sign and int(v['step'])==step)
            for i,key in [(0,'Txx'),(1,'Tyy'),(2,'Txy'),(3,'Axx'),(4,'Axy'),(7,'Qxx'),(8,'Qxy'),(11,'tauxx'),(12,'tauyy'),(13,'tauxy'),(14,'Wxx'),(15,'Wxy'),(17,'Y')]:
                identity=max(identity,abs(r[i]-float(ref[key]))/max(1.,abs(float(ref[key]))))
            if phi>0 and step in [7,10,16,24]:incoming.append((old.copy(),de.copy(),step,sign))
            old=r[:11].copy()
assert identity<=1e-12,identity

for old,de,step,sign in incoming:
    for X in [np.array([30.,-40.]),np.array([-50.,15.]),np.zeros(2)]:
        r,J=point(old,de,.0125,X=X);active.add(int(r[23]));cases+=1;case_fd=0.;needs_oracle=False
        for column in range(6):
            eps=1e-7;plus=de.copy();minus=de.copy();pplus=pminus=10000.;xp=X.copy();xm=X.copy()
            if column<3:plus[column]+=eps;minus[column]-=eps
            elif column==3:pplus+=1e6*eps;pminus-=1e6*eps
            else:xp[column-4]+=1e6*eps;xm[column-4]-=1e6*eps
            try:
                rp,_=point(old,plus,.0125,pplus,xp,jac=False);rm,_=point(old,minus,.0125,pminus,xm,jac=False)
            except RuntimeError as error:
                (OUT/'point-failed-perturbation.json').write_text(json.dumps({'phi':old[10],'step':step,'sign':sign,'old':old.tolist(),
                    'column':column,'base_X':X.tolist(),'plus_X':xp.tolist(),'minus_X':xm.tolist(),'plus_increment':plus.tolist(),
                    'minus_increment':minus.tolist(),'plus_pressure':pplus,'minus_pressure':pminus,'base_result':r.tolist(),'error':str(error)},indent=2)+'\n')
                raise
            if not int(rp[23])==int(rm[23])==int(r[23]):
                boundaries.append({'phi':old[10],'step':step,'sign':sign,'X':X.tolist(),'column':column,'masks':[int(rm[23]),int(r[23]),int(rp[23])]})
                needs_oracle=True;continue
            difference=rp-rm
            difference[24:26]=(difference[24:26]+np.pi/2)%np.pi-np.pi/2
            d=difference/(2*eps)
            for group in groups:
                defect=np.linalg.norm(d[group]-J[group,column])/max(1.,np.linalg.norm(J[group,column]))
                if defect>fd:
                    fd=defect;worst={'phi':old[10],'step':step,'sign':sign,'X':X.tolist(),'column':column,
                        'group':group,'old':old.tolist(),'increment':de.tolist(),'AD':J[group,column].tolist(),'FD':d[group].tolist(),'defect':fd}
                case_fd=max(case_fd,defect)
        beta=1/(1+.0125/100);h=4*old[10]*(1+old[10])/(1-old[10])**2
        speeds=h*r[28:30]/(1e6*beta*.0125)
        diss=np.sum(.1*r[26:28]*speeds)
        defect=abs(diss-r[22])/max(1.,abs(diss));power=max(power,defect)
        account=ledger(old,r,de,.0125,X);energy=max(energy,abs(account['error']))
        evidence.append({'phi':old[10],'path_step':step,'normal_sign':sign,'Xxx_Pa':X[0],'Xxy_Pa':X[1],
            'active':int(r[23]),'friction_power_Pa_s':r[22],'system_power_defect':defect,'ledger_defect_Pa':account['error'],'FD_smooth_columns_max':case_fd,'FD_crosses_active_set':needs_oracle})
        if needs_oracle or case_fd>1e-6 or (old[10] in [.2,.8,.99] and sign==1 and step==10):
            oracle.append({'old':old.tolist(),'increment':de.tolist(),'X':X.tolist(),'result':r.tolist(),'jacobian':J.tolist(),'phi':old[10],'step':step,'sign':sign})
(OUT/'point-FD-diagnostic.json').write_text(json.dumps({'worst':worst,'power':power,'energy':energy,'active':sorted(active)},indent=2)+'\n')
(WORK/'point-oracle-inputs.json').write_text(json.dumps(oracle)+'\n')
(OUT/'FD-active-boundaries.json').write_text(json.dumps(boundaries,indent=2)+'\n')
assert power<=1e-12,power
assert energy<=1e-10,energy
assert active=={0,1,2,3},active
with (OUT/'point-checks.csv').open('w') as f:
    w=csv.DictWriter(f,list(evidence[0]));w.writeheader();w.writerows(evidence)
(OUT/'point-checks.json').write_text(json.dumps({'zero_backstress_identity_max':identity,'point_cases':cases,
 'fixed_phase_complete_FD_relative_error':fd,'ordinary_FD_all_smooth_columns_passed':fd<=1e-6,'higher_precision_reference_required':bool(oracle),
 'FD_active_boundary_probes':len(boundaries),'friction_power_relative_defect':power,'local_microforce_work_ledger_defect_Pa':energy,
 'active_masks':sorted(active),'bounds':{'identity':1e-12,'derivative':1e-6,'power':1e-12,'ledger':1e-10}},indent=2)+'\n')
print((OUT/'point-checks.json').read_text())
