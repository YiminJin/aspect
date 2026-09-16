"""All-QP top comparison and the separate free-endpoint coupling gate."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_uniform_sliding as audit
from analyze_bottom_source import keyed, S, VP
import bottom_completion as geometry

HERE=Path(__file__).resolve().parent
ROOTS=[HERE/f'top-source-{name}-50-local4' for name in ('control','paired')]
OUT=HERE/'top-source-comparison'

def weak_rows(k):
    rows=[]
    for root,tag in zip(ROOTS,('control','paired')):
        _,moments=audit.moments(root,k)
        raw=audit.raw_samples(root,k,audit.read(root/f'fault_{k}.csv'))
        audit.write(OUT/f'{tag}_weak_{k}.csv',moments)
        for region,lo,hi in [('top_0_2km',0,2000),('bottom_last_2km',audit.L-2000,audit.L),
                            ('interior_59_61km',59000,61000)]:
            # The physical top has xd=-7e-12 m from the rotated chart.
            # Include the actual endpoint; this is only a region selector.
            mask=(moments['xd']>=lo-1e-7)&(moments['xd']<=hi+1e-7)
            rr=(raw['xd']>=lo-1e-7)&(raw['xd']<=hi+1e-7)
            row=dict(step=k,case=tag,region=region)
            for field in ('sigma_weak_mean','sigma_Q1','p_weak_mean','minus_tauN_weak_mean'):
                row[field+'_min']=float(np.min(moments[field][mask]))
                row[field+'_max']=float(np.max(moments[field][mask]))
            row['retained_raw_sigma_min']=float(np.min(raw['sigma_n'][rr])) if np.any(rr) else None
            row['retained_raw_sigma_max']=float(np.max(raw['sigma_n'][rr])) if np.any(rr) else None
            rows.append(row)
    return rows

def main():
    OUT.mkdir(exist_ok=True)
    for root in ROOTS:
        audit.RUN=root;audit.OUT=root/'analysis';audit.OUT.mkdir(exist_ok=True)
        if not (audit.OUT/'summary.json').exists():audit.analyze_run()
    summaries=[];weak=[];checks=[];virtual_work=[];histories=[]
    tangent=np.array([.5,geometry.sn]);normal=geometry.normal
    for k in range(3):
        cases=[keyed(root,k) for root in ROOTS]
        cases=[{key:d for key,d in c.items() if d['phi']>0.} for c in cases]
        assert cases[0].keys()==cases[1].keys(),'Use all relevant QPs, not only admitted ones.'
        keys=sorted(cases[0])
        data=[{field:np.array([c[key][field] for key in keys]) for field in c[keys[0]]} for c in cases]
        for field in ('x','y','weight','phi'):np.testing.assert_array_equal(data[0][field],data[1][field])
        faults=[audit.read(root/f'fault_{k}.csv') for root in ROOTS]
        for field in ('x','y','V','Theta','C','time','dt','slip'):
            np.testing.assert_array_equal(faults[0][field],faults[1][field])
        # Observer-only rerun must reproduce the saved bottom-enabled control.
        saved=keyed(HERE/'bottom-source-complete-wedge-50-local4',k)
        saved={key:d for key,d in saved.items() if d['phi']>0.}
        repeat={}
        for field in ('chi','ux','uy','p','tau_xx','tau_yy','tau_xy'):
            old=np.array([d[field] for d in saved.values()])
            new=np.array([cases[0][key][field] for key in saved])
            scale=np.max(abs(old));error=np.max(abs(new-old))
            assert error<=1e-12*max(scale,1e-30),(field,error,scale)
            repeat[field]=float(error/max(scale,1e-30))
        f=faults[1];d=data[1]
        coordinate=(d['x']-f['x'][0])*.5+(d['y']-f['y'][0])*geometry.sn
        fs=(f['x']-f['x'][0])*.5+(f['y']-f['y'][0])*geometry.sn
        Ih=np.interp(coordinate,fs,f['Ih'])
        phi=d['phi'];h=geometry.m*phi*(1+phi)/(1-phi)**2
        ref=h/Ih*VP
        changed=data[1]['source_active']!=data[0]['source_active']
        assert np.all(coordinate[changed]>fs[-1]) and np.all(d['y'][changed]<=100000.)
        np.testing.assert_allclose(d['chi'][changed]*VP,ref[changed],rtol=1e-12,atol=1e-25)
        # Bottom normalization and source policy must not be modified by adding
        # the remote top completion. Solution changes there are measured below.
        bottom=d['y']<2000
        np.testing.assert_allclose(data[0]['chi'][bottom],d['chi'][bottom],rtol=1e-12,atol=1e-24)
        checks.append(dict(step=k,control_repeat_scaled=repeat,added_top_qps=int(changed.sum()),
            added_source_max_error=float(np.max(abs(d['chi'][changed]*VP-ref[changed]))),
            bottom_max_chi_change=float(np.max(abs(data[0]['chi'][bottom]-d['chi'][bottom])))))
        depth=100000-d['y']
        masks=dict(top_0_200m=depth<200,top_200_1000m=(depth>=200)&(depth<1000),
                   top_1000_2000m=(depth>=1000)&(depth<2000),bottom_0_200m=d['y']<200,
                   bottom_200_2000m=(d['y']>=200)&(d['y']<2000),
                   interior_59_61km=(d['xd']>59000)&(d['xd']<61000))
        r=(d['x']-f['x'][0])*normal[0]+d['y']*.5
        reference=audit.read(ROOTS[1]/'analysis/compatible_reference_profile.csv')
        speed=np.sign(r)*np.interp(abs(r),reference['r'],reference['speed_positive_r'],right=VP/2)
        for region,mask in masks.items():
            w=d['weight'][mask];rms=lambda a:float(np.sqrt(np.dot(w,a*a)/w.sum()))
            row=dict(step=k,region=region,points=int(mask.sum()),volume=float(w.sum()))
            for tag,c in zip(('control','paired'),data):
                amplitude=c['chi'][mask]*VP
                mismatch=np.sqrt((c['eps_xx'][mask]-amplitude*S[0,0])**2
                    +(c['eps_yy'][mask]-amplitude*S[1,1])**2
                    +2*(c['eps_xy'][mask]-amplitude*S[0,1])**2)
                row[tag+'_source_error_relative']=rms(amplitude-ref[mask])/rms(ref[mask])
                row[tag+'_source_signed_missing_fraction']=float(np.dot(w,ref[mask]-amplitude)/np.dot(w,ref[mask]))
                row[tag+'_strain_mismatch_rms']=rms(mismatch)
                row[tag+'_strain_over_reference']=rms(mismatch)/(rms(ref[mask])*np.linalg.norm(S))
                row[tag+'_velocity_error_over_Vp']=rms(np.hypot(c['ux'][mask]-.5*speed[mask],
                    c['uy'][mask]-geometry.sn*speed[mask]))/VP
                for field in ('p','tau_xx','tau_yy','tau_xy','tauN','sigma_n'):
                    row[tag+'_'+field+'_min']=float(np.min(c[field][mask]))
                    row[tag+'_'+field+'_max']=float(np.max(c[field][mask]))
            summaries.append(row)
        weak.extend(weak_rows(k))
        # A smooth admissible velocity test: w = Vp*s*X(1-X)*Y. It vanishes
        # on prescribed lateral boundaries and belongs to bulk Q2 exactly.
        # Quantify the EXTRA endpoint B virtual work over the continued wedge.
        X=d['x']/100000.;Y=d['y']/100000.
        grad=np.array([(1-2*X)*Y,X*(1-X)])/100000.*VP
        test_strain=.5*(tangent[:,None,None]*grad[None,:,:]+grad[:,None,:]*tangent[None,:,None])
        S_test=np.einsum('ij,ijn->n',S,test_strain)
        work=2*d['kappa']*d['chi']*VP*S_test*d['weight']
        endpoint_shape=np.where(d['segment']==len(f['V'])-2,d['xi'],0.)
        endpoint_work=float(np.dot(work,endpoint_shape))
        parents=audit.cat(p for p in ROOTS[1].glob(f'top_source_parents_{k}_rank*.csv')
                          if len(p.read_text().splitlines())>1)
        absent=parents['surface_active']==0
        assert np.all(parents['surface_weight'][absent]==0)
        virtual_work.append(dict(step=k,extra_B_work_W_per_m=float(work[changed].sum()),
            extra_B_absolute_work_W_per_m=float(abs(work[changed]).sum()),
            full_endpoint_B_work_W_per_m=endpoint_work,
            extra_fraction_of_endpoint_work=float(work[changed].sum()/endpoint_work),
            extra_QP_measure=float(d['weight'][changed].sum()),
            omitted_surface_parents=int(absent.sum()),
            omitted_full_parent_volume=float(parents['volume'][absent].sum()),
            omitted_surface_weight=float(parents['surface_weight'][absent].sum()),
            omitted_positive_phi_parent_volume=None,omitted_positive_phi_parents=None))
        if k:
            committed=audit.cat(ROOTS[1].glob(f'mature_history_{k}_rank*.csv'))
            lookup={int(i):j for j,i in enumerate(committed['id'])}
            parts=[audit.read(p) for p in ROOTS[1].glob(f'continued_source_history_{k}_rank*.csv')
                   if len(p.read_text().splitlines())>1]
            h={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
            top=h['y']>98000;errors=[]
            for comp in ('xx','yy','xy'):
                expect=2*h['kappa']*(h['eps_'+comp]-h['crack_'+comp])+h['beta']*h['old_'+comp]
                np.testing.assert_allclose(h['new_'+comp],expect,rtol=1e-12,atol=1e-9)
                actual=np.array([committed['tau_'+comp][lookup[int(i)]] for i in h['id']])
                np.testing.assert_array_equal(actual,h['new_'+comp])
                errors.append(float(np.max(abs(expect[top]-actual[top]))))
            histories.append(dict(step=k,top_parents=int(top.sum()),top_positive_phi_parents=int(np.sum(top&(h['phi']>0))),
                                  max_formula_error_Pa=max(errors)))
            positive_ids=set(h['id'][top&(h['phi']>0)].astype(int))
            positive=np.array([int(i) in positive_ids for i in parents['id']])
            assert np.all(parents['surface_active'][positive]==0)
            virtual_work[-1]['omitted_positive_phi_parent_volume']=float(parents['volume'][positive].sum())
            virtual_work[-1]['omitted_positive_phi_parents']=int(positive.sum())
    audit.records(OUT/'all_qp_comparison.csv',summaries)
    audit.records(OUT/'weak_normal_comparison.csv',weak)
    audit.records(OUT/'coupling_measure.csv',virtual_work)
    (OUT/'checks.json').write_text(json.dumps(dict(checks=checks,history=histories,
        endpoint_B=audit.read(ROOTS[1]/'top_endpoint_B_check.csv')['finite_difference_relative'].tolist()),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for root,tag in zip(ROOTS,('control','paired')):
        _,m=audit.moments(root,2);mask=m['xd']<2500
        axes[0].plot(m['xd'][mask],(m['sigma_weak_mean'][mask]-50e6)/1000,label=tag)
        d=keyed(root,2);selected=[v for v in d.values() if v['y']>98000 and v['phi']>0]
        axes[1].scatter([100000-v['y'] for v in selected],[(v['sigma_n']-50e6)/1000 for v in selected],s=3,label=tag)
    axes[0].set(xlabel='Down dip (m)',ylabel='Weak normal traction - 50 MPa (kPa)')
    axes[1].set(xlabel='Depth (m)',ylabel='Raw bulk normal traction - 50 MPa (kPa)')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(OUT/'top_comparison.png',dpi=160)
    print(json.dumps(dict(checks=checks,history=histories,virtual_work=virtual_work),indent=2))

if __name__=='__main__':
    if sys.argv[1:]==['--weak-only']:
        # Reuse the expensive all-QP comparison when only region accounting changes.
        audit.records(OUT/'weak_normal_comparison.csv',[row for k in range(3) for row in weak_rows(k)])
    else:main()
