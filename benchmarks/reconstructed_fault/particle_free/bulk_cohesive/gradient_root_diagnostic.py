#!/usr/bin/env python3
"""Independent trigonometric reevaluation of the saved ambiguous point."""
import json
import mpmath as mp
from gradient_point import OUT
mp.mp.dps=70
c=json.loads((OUT/'point-failed-perturbation.json').read_text());a=list(map(lambda v:mp.mpf(str(v)),c['old']))
de=list(map(lambda v:mp.mpf(str(v)),c['minus_increment']));X=mp.matrix([mp.mpf(str(v)) for v in c['minus_X']]);dt=mp.mpf('.0125')
G=mp.mpf('1e6');mu=mp.mpf('.1');beta=1/(1+dt/100);phi=a[10];h=4*phi*(1+phi)/(1-phi)**2
t=mp.matrix([beta*((a[0]-a[1])/2+G*(de[0]-de[1])),beta*(a[2]+G*de[2])]);ref=beta*mp.matrix(a[7:9])
pb=mp.mpf(str(c['minus_pressure']))-beta*(a[0]+a[1]+2*G*(de[0]+de[1])/3)/2
J=lambda v:mp.matrix([-v[1],v[0]])
roots=[]
for mask in [1,2,3]:
    active=[i for i in range(2) if mask&(1<<i)];m=len(active)
    q0=c['base_result'][7:9];gam=c['base_result'][28:30];theta=c['base_result'][24:26]
    guess=[mp.mpf(str(q0[0])),mp.mpf(str(q0[1]))]+[mp.mpf(str(gam[i]*(2 if m==1 else 1))) for i in active]+[mp.mpf(str(theta[i])) for i in active]
    def residual(*values):
        q=mp.matrix(values[:2]);e=t-X-(1+h)*q;d=t-h*q;flow=mp.matrix([0,0]);f=[];orientation=[]
        for j,i in enumerate(active):
            k=1 if i==0 else -1;v=mp.matrix([mp.cos(2*values[2+m+j]),mp.sin(2*values[2+m+j])]);jv=J(v)
            flow+=values[2+j]*k*jv
            f.append((k*mp.fdot(e,jv)-mu*(pb+mp.fdot(d,v)))/1000)
            orientation.append((-k*mp.fdot(e,v)-mu*mp.fdot(d,jv))/1000)
        return tuple([(q[0]-ref[0]-flow[0])/1000,(q[1]-ref[1]-flow[1])/1000]+f+orientation)
    sol=mp.findroot(residual,tuple(guess),tol=mp.mpf('1e-60'),maxsteps=100)
    q=mp.matrix(sol[:2]);e=t-X-(1+h)*q;d=t-h*q
    margins=[]
    for k in [1,-1]:
        z=-k*J(e)-mu*d;margins.append(mp.sqrt(mp.fdot(z,z))-mu*pb)
    gamma=[mp.mpf(0),mp.mpf(0)]
    for j,i in enumerate(active):gamma[i]=sol[2+j]
    norm=lambda v:mp.sqrt(mp.fdot(v,v))
    roots.append({'active':mask,'Qxx_Pa':str(q[0]),'Qxy_Pa':str(q[1]),'tauxx_Pa':str(d[0]),'tauxy_Pa':str(d[1]),
        'gamma_Pa':list(map(str,gamma)),'margins_Pa':list(map(str,margins)),'min_normal_pressure_Pa':str(pb-norm(d)),
        'residual':str(max(map(abs,residual(*sol)))),'admissible':bool(min(gamma)>=0 and max(margins)<mp.mpf('1e-45') and pb>norm(d))})
result={'precision_digits':70,'saved_case':'point-failed-perturbation.json','independent_equations':'Flow, yield and stationarity in plane angles; no quadratic candidate construction.',
        'roots':roots,'distinct_admissible_roots':sum(r['admissible'] for r in roots)}
(OUT/'independent-ambiguous-roots.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
