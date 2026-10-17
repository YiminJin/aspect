#!/usr/bin/env python3
"""Common-time native PF6a versus selected 1D fields; no field registration."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from assess_pf6a import samples_at,norm2,grids,strains
from assess_phase_energy import csv,gauss,q1
from assess_pf6a_reference import load,sample,pullback,TARGETS
from pf6a_spatial_reference import A,ELL


def snapshot(path):
    out=SimpleNamespace(path=path,nodes=csv(path/'nodes.csv'),cells=csv(path/'cells.csv'),
                        tensor=csv(path/'tensor.csv'),surface=csv(path/'surface.csv'))
    out.indices=np.stack([out.cells[f'i{i}']for i in range(4)],axis=1).astype(int)
    out.ns=len(out.surface);out.m=float(csv(path/'constants.csv')['m'])
    return out


def fracture_gradient(ref,energy=False):
    r,w=gauss(2);r,s=np.repeat(r,2),np.tile(r,2);w=np.repeat(w,2)*np.tile(w,2)
    N=q1(r,s);p=ref.nodes['phase'][ref.indices]
    dx,dy=ref.cells['dx'],ref.cells['dy'];grad=np.empty((len(p),4,4,2))
    grad[:,:,:,0]=np.array([-(1-s),1-s,-s,s]).T[None]/dx[:,None,None]
    grad[:,:,:,1]=np.array([-(1-r),-r,1-r,r]).T[None]/dy[:,None,None]
    gp=np.sum(p[:,None,:,None]*grad,axis=2)
    g=np.zeros(len(ref.nodes))
    for i in range(4):
        value=dx*dy*np.sum(w[None]*A*(N[None,:,i]+2*ELL**2*np.sum(gp*grad[:,:,i,:],axis=2)),axis=1)
        np.add.at(g,ref.indices[:,i],value)
    phase=p@N.T
    value=float(np.sum((dx*dy)[:,None]*w[None]*A*(phase+ELL**2*np.sum(gp*gp,axis=2))))
    return (g,value) if energy else g


def assess(run,reference):
    record=json.loads((run/'record.json').read_text());metrics=np.atleast_1d(csv(run/'run/metrics.csv'))
    case=load(reference);f=case['fields'];ny=len(f['y'])-1
    assert np.array_equal(metrics['time'],f['time'][:len(metrics)]), 'Native/reference clocks differ'
    z,w=gauss(8);y=(np.arange(1024)[:,None]+z).ravel()/1024;wy=np.tile(w/1024,1024)
    zx,wx=gauss(3);x=(np.arange(8)[:,None]+zx).ravel()/8;wx=np.tile(wx/8,8)
    xx,yy=np.repeat(x,len(y)),np.tile(y,len(x));ww=np.repeat(wx,len(y))*np.tile(wy,len(x))
    rp,rc,rs,rd=sample(case,y)
    refq=np.r_[0,[a['q']for a in case['history']]]
    refenergy=np.r_[float(metrics['bulk'][0]+metrics['cohesive'][0]+metrics['fracture'][0]),[a['bulk']+a['cohesive']+a['fracture']for a in case['history']]]
    scales=dict(stress=max(np.sqrt(2*np.sum(wy*rs**2,axis=1))),phase=max(np.sqrt(np.sum(wy*rp**2,axis=1))),
                q=max(abs(refq)),slip=f['slip'][-1],energy=max(abs(refenergy)),driving=max(abs((f['driving']@pullback(f['y']))[:,0:2].sum(axis=1)).ravel()))
    rows=[];previous=None;slip=np.zeros(9)
    for row in metrics:
        k=int(row['step']);path=run/'run'/f'state-{k}';ref=snapshot(path)
        p,T,tau=samples_at(ref,xx,yy)
        delta=tau.copy();delta[:,2]-=np.tile(rs[k],len(x))
        phase_difference=p-np.tile(rp[k],len(x));weight=np.tile(rc[k],len(x))
        surface=csv(path/'accepted-surface.csv');slip+=float(row['dt'])*surface['V']
        g=csv(path/'gradient.csv');fg,fracture=fracture_gradient(ref,True)
        driving=(g['gradient']-fg)@pullback(ref.nodes['y']);expected=(f['driving'][k,0]+f['driving'][k,1])@pullback(f['y'])
        gradient=np.zeros_like(phase_difference)
        # Q1 phase normal derivative at the same unregistered physical points.
        ix=np.minimum((xx*2).astype(int),1);iy=np.minimum((yy*512).astype(int),511)
        lookup=np.empty((512,2),dtype=int);lookup[np.rint(ref.cells['y0']*512).astype(int),np.rint(ref.cells['x0']*2).astype(int)]=np.arange(len(ref.cells))
        ci=lookup[iy,ix];rr=(xx-ref.cells['x0'][ci])/ref.cells['dx'][ci]
        nodal=ref.nodes['phase'][ref.indices[ci]]
        gradient=((1-rr)*(nodal[:,2]-nodal[:,0])+rr*(nodal[:,3]-nodal[:,1]))/ref.cells['dy'][ci]
        out=dict(step=k,time=float(row['time']),whole_Pa=float(np.sqrt(ww@norm2(delta))),
                 localized_Pa=float(np.sqrt((ww*weight)@norm2(delta)/(ww@weight))),
                 phase_L2_absolute=float(np.sqrt(ww@phase_difference**2)),phase_Linf=float(max(abs(ref.nodes['phase']-np.interp(ref.nodes['y'],f['y'],f['phase'][k])))),
                 phase_gradient_L2=float(np.sqrt(ww@(gradient-np.tile(rd[k],len(x)))**2)),
                 q_Pa=float(max(abs(surface['Z']/surface['I']-refq[k]))),
                 slip_m=float(max(abs(slip-f['slip'][k]))),
                 Z_Pa_m=float(max(abs(surface['Z']-f['Z'][k]))),
                 elastic_slip_m=float(max(abs(surface['Z']-f['Z'][k]))/1e6),
                 V_m_per_s=float(max(abs(surface['V']-(case['history'][k-1]['V'] if k else 0.)))),
                 energy_J_per_m=float(abs(float(row['bulk']+row['cohesive']+row['fracture'])-refenergy[k])),
                 driving_absolute=float(max(abs(driving-expected))),KKT=float(row['KKT']),healing=float(row['healing']),publications=int(row['publications']))
        out.update(whole_stress=out['whole_Pa']/scales['stress'],localized_stress=out['localized_Pa']/scales['stress'],phase_L2=out['phase_L2_absolute']/scales['phase'],q=out['q_Pa']/scales['q'],slip=out['slip_m']/scales['slip'],energy=out['energy_J_per_m']/scales['energy'],driving=out['driving_absolute']/scales['driving'])
        seg=np.minimum((xx*8).astype(int),7);xi=xx*8-seg
        normalization=(1-xi)*surface['I'][seg]+xi*surface['I'][seg+1]
        chi=ref.m*p*(1+p)/(1-p)**2/normalization
        independent_load=np.zeros(9)
        np.add.at(independent_load,seg,ww*(1-xi)*chi*tau[:,2])
        np.add.at(independent_load,seg+1,ww*xi*chi*tau[:,2])
        ks=np.minimum((x*8).astype(int),7);xs=x*8-ks
        zsurf=(1-xs)*surface['Z'][ks]+xs*surface['Z'][ks+1]
        isurf=(1-xs)*surface['I'][ks]+xs*surface['I'][ks+1]
        independent_cohesive=np.zeros(9)
        np.add.at(independent_cohesive,ks,wx*(1-xs)*zsurf/isurf)
        np.add.at(independent_cohesive,ks+1,wx*xs*zsurf/isurf)
        independent_energy=ww@norm2(tau)/4e6+np.sum(wx*zsurf*zsurf/isurf)/2e6+fracture
        native_energy=float(row['bulk']+row['cohesive']+row['fracture'])
        out['same_field_energy_scaled']=float(abs(independent_energy-native_energy)/max(1.,native_energy))
        if k:
            velocities=grids(csv(path/'velocity.csv'),['ux','uy'],len(ref.cells))
            a=np.repeat(np.array([0.,.5,1.]),3);b=np.tile(np.array([0.,.5,1.]),3)
            strain,_=strains(ref,velocities,a,b)
            Tg=grids(ref.tensor,['xx','yy','xy'],len(ref.cells))
            from assess_pf6a import evaluate_grid
            native=evaluate_grid(Tg,a,b)
            old=evaluate_grid(grids(previous.tensor,['xx','yy','xy'],len(ref.cells)),a,b)
            dt=float(row['dt']);exact=np.exp(-dt/100)*old+2*(-1e8*np.expm1(-dt/100))*strain
            out['source_recurrence_Pa']=float(max(abs(native-exact).ravel()))
            out['source_recurrence_scaled']=out['source_recurrence_Pa']/max(1.,float(max(abs(native).ravel())))
            out['Z_recurrence_Pa_m']=float(max(abs(surface['Z']-np.exp(-dt/100)*previous.surface['Z']+1e8*np.expm1(-dt/100)*surface['V'])))
            native_loads=csv(path/'native-loads.csv')
            out['same_field_shear_weak_load_scaled']=float(max(abs(independent_load-native_loads['shear']))/max(1.,float(max(abs(native_loads['shear'])))))
            out['same_field_cohesive_weak_load_scaled']=float(max(abs(independent_cohesive-native_loads['cohesive']))/max(1.,float(max(abs(native_loads['cohesive'])))))
            mass=np.full(9,1/8);mass[[0,-1]]*=.5
            expected_shear=float(wy@(rc[k]*rs[k]))*mass
            out['shear_weak_load_Pa_m']=float(max(abs(native_loads['shear']-expected_shear)))
            out['cohesive_weak_load_Pa_m']=float(max(abs(native_loads['cohesive']-refq[k]*mass)))
            contact=native_loads['residual']/mass
            contact=np.where(surface['V']<=1.0000000001e-20,np.maximum(contact,0.),contact)
            out['fault_KKT_Pa']=float(np.sqrt(mass@(contact**2)))
            mechanical=csv(path/'mechanical.csv');out['mechanical_bulk']=float(mechanical['bulk_norm']);out['mechanical_fault']=float(mechanical['surface_rms'])
        rows.append(out);previous=ref
        print(f"checked step {k}, t={row['time']:.7f}",flush=True)
    completed=record['exit_code']==0 and len(rows)==73 and rows[-1]['time']==.5625
    keys=['whole_stress','localized_stress','phase_L2','phase_Linf','q','slip','energy','driving']
    maximum={k:max(r[k]for r in rows)for k in keys}
    checks={k:max(r.get(k,0.)for r in rows)for k in ['source_recurrence_Pa','source_recurrence_scaled','same_field_energy_scaled','same_field_shear_weak_load_scaled','same_field_cohesive_weak_load_scaled','Z_recurrence_Pa_m','mechanical_bulk','mechanical_fault','fault_KKT_Pa','KKT','healing','shear_weak_load_Pa_m','cohesive_weak_load_Pa_m']}
    onset=next((float(a['time'])for a in metrics if a['phase_max']-metrics['phase_max'][0]>1e-5),None)
    return dict(native_completed=completed,seconds=record['seconds'],max_rss_KiB=record['max_rss_KiB'],states=len(rows),reference=str(reference),
                scales={k:float(v)for k,v in scales.items()},maximum={k:float(v)for k,v in maximum.items()},worst_time={k:max(rows,key=lambda r:r[k])['time']for k in keys},
                implementation_target=1e-6,implementation_target_passes={k:bool(maximum[k]<=1e-6) for k in keys},
                practical_targets=TARGETS,practical_target_passes={k:bool(maximum[k]<=v) for k,v in TARGETS.items()},
                native_onset=onset,reference_onset=case['summary']['onset'],checks=checks,by_time=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=assess(a.run,a.reference);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='by_time'},indent=2))
if __name__=='__main__':main()
