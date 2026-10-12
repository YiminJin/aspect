#!/usr/bin/env python3
"""Common-time fields, independent path verification and compact decision plots."""
import json
import hashlib
from pathlib import Path

import numpy as np
from coupled_branch import E, HERE, START, FINAL, TARGET, Coupled, Reference, G, A, h, path_ledger, write


def read(path):
    return json.loads(path.read_text())


def stress(r, phi, R, Z):
    return G*(r.D@R)-Z*h(r.B@phi)/r.direct(phi)[0]


def totals(rows):
    keys = ['external_work','bulk_work','lifting_work','bulk_change','cohesive_change',
            'fracture_change','bulk_viscous','cohesive_viscous','friction',
            'external_remainder','mechanical_power','friction_residual_power','phase_power',
            'mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute',
            'bulk_profile_power','normalization_power','fracture_power','signed_profile_cost']
    return {k:dict(signed=float(sum(x[k] for x in rows)),
                   absolute=float(sum(abs(x[k]) for x in rows)),
                   maximum=float(max(abs(x[k]) for x in rows))) for k in keys}


def verify(d, f, hist, energies):
    p = Coupled(); results=[]
    for t in [.53125,.546875,FINAL]:
        j=round((t-START)*d)-1
        pa=f['phase'][j]; pb=f['phase'][j+1]; Ra=f['R'][j]; Za=f['Z'][j]
        p.incoming(pa,Ra,Za,hist[j-1]['slip'] if j else float(np.load(E/'start.npz')['slip']),t,1/d)
        value,g=p.evaluate(pb); H=p.hessian(pb)
        free=np.flatnonzero(pb>pa+1e-13)
        eig,vec=np.linalg.eigh(H[np.ix_(free,free)])
        direction=np.zeros(len(pb));direction[free]=vec[:,0]
        assert min(pb-1e-4*abs(direction))>=0 and max(pb+1e-4*abs(direction))<1
        checks=[]
        for eps in [1e-4,1e-5,1e-6]:
            vp,gp=p.evaluate(pb+eps*direction);vm,gm=p.evaluate(pb-eps*direction)
            fd=(gp-gm)/(2*eps)
            checks.append(dict(epsilon=eps,energy_error=float(abs((vp-vm)/(2*eps)-g@direction)),
                               tangent_error=float(max(abs(fd-H@direction))),
                               minimum_mode_FD=float(direction@fd),minimum_mode_exact=float(eig[0])))
        p.evaluate(pb); data=p.last[3]; Rb=p.last[4].copy();Zb=p.last[5];u=p.r.velocity.copy()
        coarse=path_ledger(p.r,pa,pb,Ra,Za,Rb,Zb,u,data['V'],1/d,16)
        fine=path_ledger(p.r,pa,pb,Ra,Za,Rb,Zb,u,data['V'],1/d,32)
        keys=['external_work','bulk_viscous','cohesive_viscous','external_remainder',
              'mechanical_power','friction_residual_power','phase_power']
        uncertainty=max(abs(coarse[k]-fine[k]) for k in keys)
        assert uncertainty<1e-10
        assert coarse['identity_error']<1e-10 and fine['identity_error']<1e-10
        assert min(x['tangent_error'] for x in checks)<1e-7
        # Lowest-mode derivative independently resolves the small Schur margin.
        mode_error=min(abs(x['minimum_mode_FD']-eig[0]) for x in checks)
        assert mode_error<eig[0]*.01
        results.append(dict(time=t,derivatives=checks,mode_uncertainty=mode_error,
                            time_quadrature_uncertainty=uncertainty,
                            identity_error=fine['identity_error']))
    return results


def main():
    out=E/'assessment';out.mkdir(exist_ok=False)
    init=np.load(E/'start.npz'); cases={}; checks=[]; tables=[]
    r=Reference(256,1.,12)  # Independent stress samples, not new assembly rule.
    for d in [256,512,1024]:
        p=E/f'dt-1-{d}';hist=read(p/'history.json');f=np.load(p/'fields.npz');en=read(p/'energy.json')
        summary=read(p/'summary.json');stable=read(p/'stability.json')
        assert summary['completed'] and summary['accepted']==round((FINAL-START)*d)
        assert np.array_equal(f['phase'][0],init['phi']) and np.array_equal(f['R'][0],init['R']) and f['Z'][0]==init['Z']
        assert min(np.diff(f['phase'],axis=0).ravel())>=-4.922e-12
        assert max(x['phase_KKT'] for x in hist)<=TARGET
        assert max(x['force'] for x in hist)<1e-7 and max(x['contact'] for x in hist)<1e-7
        assert all(x['equilibrium'] and not x['admissible_negative'] for x in stable)
        assert max(x['identity_error'] for x in en)<1e-10
        smooth=[x for x in en if x['time']>.53125]
        checks.append(dict(denominator=d,checks=verify(d,f,hist,en)))
        cases[d]=(hist,f,en)
        entry=dict(denominator=d,summary=summary,final=hist[-1],totals=totals(en),
                   onset_window_totals=totals([x for x in en if x['time']>.5234375]),
                   smooth_totals=totals(smooth),onset_totals=totals([x for x in en if x['time']<=.53125]),
                   phase_increment_max=max(x['phase_increment'] for x in hist),
                   endpoint_phase_KKT_max=max(x['phase_KKT'] for x in hist),
                   endpoint_force_max=max(x['force'] for x in hist),endpoint_contact_max=max(x['contact'] for x in hist),
                   minimum_reduced_curvature=min(x['minimum'] for x in stable if x['minimum'] is not None),
                   weak_contacts_max=max(x['weak'] for x in stable),
                   intermediate_max={k:max(x['intermediate_max'][k] for x in en) for k in en[0]['intermediate_max']},
                   smooth_intermediate_max={k:max(x['intermediate_max'][k] for x in smooth) for k in smooth[0]['intermediate_max']},
                   critical_margin_note='finite-step reduced margin, not uniform as dt tends to zero')
        tables.append(entry)
    comparison=[]
    for da,db in [(256,512),(512,1024),(256,1024)]:
        ha,fa,_=cases[da];hb,fb,_=cases[db]
        errors={k:0. for k in ['phase','q','slip','bulk','cohesive','fracture','stress_RMS','localized_stress_RMS','stress_max','common_interval_V']}
        times=[];previous_slip_a=float(init['slip']);previous_slip_b=previous_slip_a
        for j,a in enumerate(ha):
            t=a['time'];k=round((t-START)*db)-1;b=hb[k]
            pa=fa['phase'][j+1];pb=fb['phase'][k+1]
            sa=stress(r,pa,fa['R'][j+1],fa['Z'][j+1]);sb=stress(r,pb,fb['R'][k+1],fb['Z'][k+1])
            delta=sa-sb;chi=h(r.B@pb)/r.direct(pb)[0]
            va=(a['slip']-previous_slip_a)*da;vb=(b['slip']-previous_slip_b)*da
            difference=dict(phase=float(max(abs(pa-pb))),q=abs(a['q']-b['q']),slip=abs(a['slip']-b['slip']),
                            bulk=abs(a['bulk']-b['bulk']),cohesive=abs(a['cohesive']-b['cohesive']),
                            fracture=abs(a['fracture']-b['fracture']),stress_RMS=float(np.sqrt(r.w@(delta*delta))),
                            localized_stress_RMS=float(np.sqrt((r.w*chi)@(delta*delta)/(r.w@chi))),
                            stress_max=float(max(abs(delta))),common_interval_V=abs(va-vb))
            for key,value in difference.items():errors[key]=max(errors[key],value)
            times.append(dict(time=t,**difference));previous_slip_a=a['slip'];previous_slip_b=b['slip']
        comparison.append(dict(coarse=da,fine=db,maximum=errors,by_time=times))
    manifest=read(E/'preservation.json');root=HERE.parents[2]
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    write(out/'tables.json',tables);write(out/'comparison.json',comparison);write(out/'verification.json',checks)
    write(out/'decision.json',dict(complete_short_branch=True,production_unchanged=True,
          original_evidence_unchanged=True,preserved_files=len(manifest),native_adoption=False,P3=False,
          reason='Short onset branch converges; full .625s trajectory/native transfer remain untested'))
    plots(out,cases,r)
    for row in tables:
        print(json.dumps(dict(d=row['denominator'],phase=row['final']['phase_max'],q=row['final']['q'],
          increment=row['phase_increment_max'],energy=row['totals']['external_remainder'],
          smooth=row['smooth_totals']['external_remainder'],residual_work={k:row['totals'][k] for k in ['mechanical_power','friction_residual_power','phase_power']},
          curvature=row['minimum_reduced_curvature'],intermediate=row['smooth_intermediate_max'])),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)


def plots(out,cases,r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(12,7))
    for d,(hist,f,en) in cases.items():
        t=[x['time'] for x in hist];label=f'dt=1/{d} s'
        ax[0,0].plot(t,[x['phase_max'] for x in hist],label=label)
        ax[0,1].plot([x['slip'] for x in hist],[x['q'] for x in hist],label=label)
        ax[0,2].plot(f['y'],f['phase'][-1],label=label)
        ax[1,0].plot(t,np.cumsum([x['external_remainder'] for x in en]),label=label)
        for key,style in [('mechanical_power','-'),('friction_residual_power','--'),('phase_power',':')]:
            ax[1,1].plot(t,np.cumsum([x[key+'_absolute'] for x in en]),style,label=f'{d}: {key}')
        ax[1,2].plot(r.yq,stress(r,f['phase'][-1],f['R'][-1],f['Z'][-1]),label=label)
    for a,x,y in zip(ax.ravel(),['time [s]','cumulative slip [m]','y [m]','time [s]','time [s]','y [m]'],
                      ['maximum phase','cohesive traction [Pa]','final phase','external energy remainder [J/m]',
                       'integral absolute residual power [J/m]','final stress [Pa]']):
        a.set_xlabel(x);a.set_ylabel(y);a.grid(alpha=.2)
    ax[0,0].set_xlim(.5234375,FINAL)
    ax[0,0].legend();ax[1,1].legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'branch.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(12,3.3))
    _,f,_=cases[1024]
    for t in [START,.53125,.546875,FINAL]:
        j=round((t-START)*1024);p=f['phase'][j];I=r.direct(p)[0]
        ax[0].plot(f['y'],p,label=f'{t:g} s');ax[1].plot(f['y'],h(p)/I,label=f'{t:g} s')
        ax[2].plot(r.yq,stress(r,p,f['R'][j],f['Z'][j]),label=f'{t:g} s')
    for a,y in zip(ax,['phase','chi [1/m]','stress [Pa]']):a.set_xlabel('y [m]');a.set_ylabel(y);a.grid(alpha=.2)
    ax[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'profiles.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
