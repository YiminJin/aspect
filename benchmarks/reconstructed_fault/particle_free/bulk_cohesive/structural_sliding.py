#!/usr/bin/env python3
"""Substantial-slip reference; leaves the qualified short reference untouched."""
import csv,json,time
from pathlib import Path
import numpy as np
from structural_rsf import G,ETA,P0,ELL,ETA_RAD,V0,DC,RELAX,solve_speed
from structural_rsf_band import Band
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding'
WORK=ROOT/'build-particle-free/structural-sliding'


def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def imposed_speed(t):
    return .001 if t<2 else .002 if t<4 else 0. if t<4.5 else -.001


class SlidingBand(Band):
    def __init__(self,cells,peak):
        super().__init__(cells)
        self.peak=peak
        self.phi=np.maximum(np.sqrt(peak)-abs(self.y)/(2*ELL),0.)**2
        self.active=self.phi>0
        p=self.phi[self.active]
        self.ih=np.zeros_like(self.y);self.ih[self.active]=(1-p)**2/(4*p*(1+p))
        self.h=np.zeros_like(self.y);self.h[self.active]=np.inf
        np.divide(1.,self.ih,out=self.h,where=self.ih>0)
        gamma=self.mass*self.phi
        edge=ELL**2*np.diff(self.phi)**2/self.dx/2
        gamma[:-1]+=edge;gamma[1:]+=edge;gamma*=3/(8*ELL)
        self.I=sum(gamma);self.chi=gamma/(self.mass*self.I)
        self.coherent_C=G*self.mass@(self.chi**2*self.ih)


class Interface:
    def __init__(self,C):self.C=C
    def initial(self):return dict(traction=0.,elastic_slip=0.,slip=0.,Theta=1.)
    def trial(self,old,dt,du):
        C=self.C;beta=1/(1+RELAX*dt)
        f=beta*(old['traction']+G/.25*du-C*old['elastic_slip'])
        V,mu,slope,theta,error=solve_speed(abs(f),ETA_RAD+beta*dt*(G/.25+C),P0,old['Theta'],dt)
        V,mu,theta=map(float,[V,mu,theta]);v=np.sign(f)*V
        s=beta*(old['traction']+G/.25*(du-dt*v));e=beta*(old['elastic_slip']+dt*v)
        bulk=.25*s*s/(2*G);coh=C*e*e/2
        old_energy=.25*old['traction']**2/(2*G)+C*old['elastic_slip']**2/2
        energy=dict(work_J_m2=s*du,bulk_J_m2=bulk,cohesive_J_m2=coh,Maxwell_J_m2=RELAX*dt*(.25*s*s/G+C*e*e),
            friction_J_m2=dt*P0*mu*V,radiation_J_m2=dt*ETA_RAD*V*V,
            BE_J_m2=.25*(s-old['traction'])**2/(2*G)+C*(e-old['elastic_slip'])**2/2)
        defect=energy['work_J_m2']-(bulk+coh-old_energy+sum(energy[k] for k in ['Maxwell_J_m2','friction_J_m2','radiation_J_m2','BE_J_m2']))
        assert abs(defect)<1e-10*max(1.,abs(energy['work_J_m2'])),defect
        energy['ledger_error_J_m2']=defect
        return dict(traction=s,elastic_slip=e,slip=old['slip']+dt*v,Theta=theta,v=v,energy=energy,rate_error=error)


def run(cells,peak,dt,mode='band'):
    start=time.perf_counter();band=SlidingBand(cells,peak)
    model=band if mode=='band' else Interface(0. if mode=='interface' else band.coherent_C)
    old=model.initial();records=[];profiles=[];U=0.;absolute_slip=0.
    cumulative={k:0. for k in ['work_J_m2','Maxwell_J_m2','friction_J_m2','radiation_J_m2','BE_J_m2']}
    trap=0.;abs_remainders=0.;max_ledger=0.;max_rate=0.;max_compat=0.;max_iter=0
    for step in range(round(6/dt)):
        t=(step+1)*dt;du=imposed_speed((step+.5)*dt)*dt;U+=du
        new=model.trial(old,dt,du);e=new['energy'];trap_step=.5*(old['traction']+new['traction'])*du
        trap+=trap_step;abs_remainders+=abs(e['BE_J_m2']-.5*(new['traction']-old['traction'])*du)
        for key in cumulative:cumulative[key]+=e[key]
        max_ledger=max(max_ledger,abs(e['ledger_error_J_m2']));max_rate=max(max_rate,new['rate_error'])
        if mode=='band':
            new['publications']+=1;assert new['publications']==step+1
            w=band.chi*new['v'];speed=band.mass@w;abs_speed=band.mass@abs(w);slip=2*band.mass@new['K']
            width=(band.mass@abs(w))**2/(band.mass@(w*w)) if np.any(w) else 0.
            age=band.mass@(band.chi*new['Theta']);minimum=np.min(new['Theta'][band.active]);maximum=np.max(new['Theta'][band.active])
            max_compat=max(max_compat,abs(new['compatibility_m']));max_iter=max(max_iter,new['iterations'])
        else:speed=new['v'];abs_speed=abs(speed);slip=new['slip'];width=0.;age=minimum=maximum=new['Theta']
        absolute_slip+=dt*abs_speed
        physical=cumulative['Maxwell_J_m2']+cumulative['friction_J_m2']+cumulative['radiation_J_m2']
        records.append(dict(mode=mode,peak_phi=peak,cells=cells,dt_s=dt,time_s=t,U_m=U,traction_Pa=new['traction'],
            slip_m=slip,absolute_slip_m=absolute_slip,V_m_s=speed,rate_width_m=width,Theta_chi_mean=age,Theta_min=minimum,Theta_max=maximum,
            bulk_J_m2=e['bulk_J_m2'],cohesive_J_m2=e['cohesive_J_m2'],**cumulative,trap_work_J_m2=trap,
            trap_balance_J_m2=trap-e['bulk_J_m2']-e['cohesive_J_m2']-physical,absolute_step_remainders_J_m2=abs_remainders))
        if mode=='band' and any(abs(t-s)<1e-10 for s in [1.,2.,2.1,3.,4.,4.5,5.,6.]):
            for j,y in enumerate(band.y):
                profiles.append(dict(peak_phi=peak,cells=cells,dt_s=dt,time_s=t,y_m=y,phi=band.phi[j],chi_inv_m=band.chi[j],
                    local_V_m_s=new['v'][j],flow_s_inv=w[j],Theta=new['Theta'][j],T_xy_Pa=new['T'][j],A_xy=new['A'][j],K_xy=new['K'][j],
                    cohesive_traction_Pa=2*G*new['A'][j]*band.ih[j],physical_shear_Pa=new['traction']))
        old=new
    end=records[-1];sliding=cumulative['friction_J_m2']+cumulative['radiation_J_m2']
    meta=dict(mode=mode,peak_phi=peak,cells=cells,dt_s=dt,seconds=time.perf_counter()-start,coherent_C_Pa_m=band.coherent_C,
        absolute_slip_over_Dc=absolute_slip/DC,final_slip_over_Dc=end['slip_m']/DC,Theta_min=min(r['Theta_min'] for r in records),
        max_rate_error_Pa=max_rate,max_ledger_error_J_m2=max_ledger,max_compatibility_m=max_compat,max_Newton_iterations=max_iter,
        trap_balance_over_sliding=abs(end['trap_balance_J_m2'])/sliding,absolute_remainders_over_sliding=abs_remainders/sliding,
        trap_balance_over_work=abs(end['trap_balance_J_m2'])/max(abs(trap),1e-30),absolute_remainders_over_work=abs_remainders/max(abs(trap),1e-30),
        endpoint=end)
    print(json.dumps({k:v for k,v in meta.items() if k!='endpoint'}),flush=True)
    return records,profiles,meta


def compare(a,b,kind):
    ratio=round(a[0]['dt_s']/b[0]['dt_s']);reference=b[ratio-1::ratio];assert len(a)==len(reference)
    rows=[]
    floors=dict(traction_Pa=1e-5,slip_m=1e-10,work_J_m2=1e-10,rate_width_m=1e-7,Theta_chi_mean=1e-7)
    for key,floor in floors.items():
        # Width is undefined at exactly zero rate; exclude only genuinely tiny rates.
        use=[i for i,r in enumerate(reference) if key!='rate_width_m' or abs(r['V_m_s'])>=V0*.01]
        x=np.array([a[i][key] for i in use]);y=np.array([reference[i][key] for i in use]);absolute=max(abs(x-y));scale=max(abs(y))
        passed=absolute<=floor+(.005 if kind=='temporal' else .01)*scale
        rows.append(dict(kind=kind,peak_phi=a[0]['peak_phi'],coarse_cells=a[0]['cells'],fine_cells=b[0]['cells'],coarse_dt_s=a[0]['dt_s'],fine_dt_s=b[0]['dt_s'],
            quantity=key,absolute_error=absolute,relative_error=absolute/max(scale,1e-300),absolute_floor=floor,passed=bool(passed)))
    return rows


def main():
    OUT.mkdir(exist_ok=True,parents=True);allrows=[];allprofiles=[];metas=[];comparisons=[]
    for peak in [.8,1.]:
        runs={}
        for cells,dt in [(64,.002),(128,.002),(256,.002),(256,.001)]:
            rows,profiles,meta=run(cells,peak,dt);runs[(cells,dt)]=rows;allrows+=rows[4::5] if dt==.002 else rows[9::10];allprofiles+=profiles;metas.append(meta)
            (OUT/'sliding_progress.json').write_text(json.dumps(metas,indent=2)+'\n')
        comparisons+=compare(runs[(64,.002)],runs[(128,.002)],'spatial')+compare(runs[(128,.002)],runs[(256,.002)],'spatial')+compare(runs[(256,.002)],runs[(256,.001)],'temporal')
        for mode in ['interface','coherent']:
            rows,profiles,meta=run(256,peak,.001,mode);allrows+=rows[9::10];metas.append(meta)
    write_csv('sliding_histories.csv',allrows);write_csv('sliding_profiles.csv',allprofiles);write_csv('sliding_comparison.csv',comparisons)
    (OUT/'sliding_summary.json').write_text(json.dumps(dict(runs=metas,comparisons=comparisons),indent=2)+'\n')

if __name__=='__main__':main()
