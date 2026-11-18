#!/usr/bin/env python3
"""Check horizontal-plane decoupling against the retained straight-load fields."""
from observations import *
a='overnight-free-turn-b-prefix';b='resolution60-coarse-growth'
pa,pb=RUNS/a/'output',RUNS/b/'output'
ra,rb=read(pa/'bulk-accepted.csv'),read(pb/'bulk-accepted.csv')
result={'prediction':'Added traceless diagonal strain has S:D=0 at horizontal S; constant-coefficient Maxwell background superposes. Prescribed confinement is unchanged.','endpoints':[]}
for row in ra:
 time=float(row.time);match=rb[abs(rb.time-time)<1e-13]
 if time>.075+1e-13 or not len(match):continue
 r=match[0];step=int(row.step);other=int(r.step)
 # Verify the whole preceding schedule, not just this endpoint time.
 aa=ra[ra.time<=time+1e-13];bb=rb[rb.time<=time+1e-13]
 assert len(aa)==len(bb) and max(abs(aa.dt-bb.dt))<1e-14
 x=field(a,step);y=field(b,other)
 sa,sb=read(pa/f'tip-sections-{step:04d}.csv'),read(pb/f'tip-sections-{other:04d}.csv')
 errors={n:float(max(abs(sa[n]-sb[n]))) for n in ['s','ds','age','slip','V','force'] if n in sa.dtype.names}
 info=dict(time=time,phase_max=float(max(abs(x.coefficients-y.coefficients).ravel())),sections=errors,
     work_error_difference=float((row.Wext-row.energy_change-row.Db-row.Dc-row.Df)-(r.Wext-r.energy_change-r.Db-r.Dc-r.Df)))
 weight=.5*(sa.ds*abs(sa.V)+sb.ds*abs(sb.V));delta=sa.force-sb.force
 info['rate_weighted_force_rms']=float(np.sqrt(sum(weight*delta**2)/sum(weight))) if sum(weight) else None
 info['section_power_difference']=float(sum(sa.ds*sa.V*sa.force)-sum(sb.ds*sb.V*sb.force))
 info['largest_force_differences']=[dict(section=int(k),x=float(sa.s[k]),force_B=float(sa.force[k]),force_straight=float(sb.force[k]),V_B=float(sa.V[k]),V_straight=float(sb.V[k]),slip_B=float(sa.slip[k]),slip_straight=float(sb.slip[k])) for k in np.argsort(abs(delta))[-5:][::-1]]
 if list(pa.glob(f'tip-points-{step:04d}-r*')) and list(pb.glob(f'tip-points-{other:04d}-r*')):
  qa=gather(pa,f'tip-points-{step:04d}-r',['cell','q']);qb=gather(pb,f'tip-points-{other:04d}-r',['cell','q'])
  background=0.
  for t,dt in zip(aa.time,aa.dt):
   h=dt/2;beta=1/(1+h*1e6/1e8);D=-.0041954981558863999 if t>.04 else 0.
   background=(2*beta-1)*background+2*1e6*beta*dt*D
  info['background_tau_xx']=background
  info['tensor_superposition_errors']={n:float(max(abs(qa[n]-qb[n]-bg))) for n,bg in [('tauxx',background),('tauyy',-background),('tauzz',0),('tauxy',0),('Axx',0),('Ayy',0),('Azz',0),('Axy',0)]}
 result['endpoints'].append(info)
(OUT/'B-straight-reduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['endpoints'][-1])
