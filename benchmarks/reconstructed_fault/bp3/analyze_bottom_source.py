"""All-QP paired mechanics comparison; retain raw and weak stresses separately."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_uniform_sliding as audit
import bottom_completion as geometry

HERE=Path(__file__).resolve().parent
S=.5*(np.outer([.5,geometry.sn],geometry.normal)+np.outer(geometry.normal,[.5,geometry.sn]))
VP=1e-9

def keyed(root,k):
    result={}
    for path in root.glob(f'uniform_bulk_{k}_rank*.csv'):
        for row in csv.DictReader(path.open()):
            key=(row.pop('cell'),int(row['qp']))
            assert key not in result
            result[key]={name:float(value) for name,value in row.items()}
    return result

def main():
    old=HERE/'bottom-completion-50-local4'
    roots=[HERE/f'bottom-source-{name}-50-local4' for name in ['control','complete-wedge']]
    out=HERE/'bottom-source-comparison';out.mkdir(exist_ok=True)
    for root in roots:
        audit.RUN=root;audit.OUT=root/'analysis';audit.OUT.mkdir(exist_ok=True)
        if not (audit.OUT/'summary.json').exists():audit.analyze_run()
    summaries=[];checks={};history=[];closure=[]
    for k in range(3):
        a,b=[keyed(root,k) for root in roots]
        a={key:value for key,value in a.items() if value['phi']>0.}
        b={key:value for key,value in b.items() if value['phi']>0.}
        assert a.keys()==b.keys(),'All-QP comparison must use the identical physical set.'
        keys=sorted(a)
        data=[{field:np.array([case[key][field] for key in keys]) for field in a[keys[0]]} for case in [a,b]]
        for field in ('x','y','weight','phi'):np.testing.assert_array_equal(data[0][field],data[1][field])
        original=keyed(old,k)
        original={key:value for key,value in original.items() if value['phi']>0.}
        max_repeat=0.
        for field in next(iter(original.values())):
            if field=='source_active':continue
            values=np.array([row[field] for row in original.values()])
            repeated=np.array([a[key][field] for key in original])
            max_repeat=max(max_repeat,float(np.max(abs(repeated-values)/np.maximum(1.,abs(values)))))
            # A tiny signed stress/pressure component is not its own numerical
            # scale. Check repeatability against this complete field's scale,
            # four orders tighter than the unchanged nonlinear relative target.
            np.testing.assert_allclose(repeated,values,rtol=0,atol=1e-12*np.max(abs(values)))
        checks[f'control_repeat_max_scaled_difference_{k}']=max_repeat
        f=[audit.read(root/f'fault_{k}.csv') for root in roots]
        for field in ('x','y','V','Theta','C','Ih','time','dt','slip'):
            np.testing.assert_array_equal(f[0][field],f[1][field])
        coordinate=(data[0]['x']-f[0]['x'][0])*.5+(data[0]['y']-f[0]['y'][0])*geometry.sn
        fs=(f[0]['x']-f[0]['x'][0])*.5+(f[0]['y']-f[0]['y'][0])*geometry.sn
        Ih=np.interp(coordinate,fs,f[0]['Ih'])
        phi=data[0]['phi'];h=geometry.m*phi*(1+phi)/(1-phi)**2
        ref=h/Ih*VP
        r=(data[0]['x']-f[0]['x'][0])*geometry.normal[0]+data[0]['y']*.5
        masks=dict(bottom_0_200m=data[0]['y']<200,
            bottom_200_1000m=(data[0]['y']>=200)&(data[0]['y']<1000),
            bottom_1000_2000m=(data[0]['y']>=1000)&(data[0]['y']<2000),
            interior_59_61km=(data[0]['xd']>59000)&(data[0]['xd']<61000))
        for name,mask in masks.items():
            mask=mask&(phi>0.)
            w=data[0]['weight'][mask];rms=lambda value:float(np.sqrt(np.sum(w*value**2)/w.sum()))
            result=dict(step=k,region=name,points=int(mask.sum()),volume=float(w.sum()))
            for root,d in zip(roots,data):
                tag='control' if root==roots[0] else 'continued'
                source=d['chi'][mask]*VP
                eps=d['eps_xx'][mask],d['eps_yy'][mask],d['eps_xy'][mask]
                def tensor_norm(amplitude):
                    return np.sqrt((eps[0]-amplitude*S[0,0])**2+(eps[1]-amplitude*S[1,1])**2+2*(eps[2]-amplitude*S[0,1])**2)
                result[tag+'_source_error_relative']=rms(source-ref[mask])/rms(ref[mask])
                result[tag+'_source_missing_fraction']=float(np.dot(w,ref[mask]-source)/np.dot(w,ref[mask]))
                result[tag+'_strain_mismatch_rms']=rms(tensor_norm(source))
                result[tag+'_strain_reference_error_rms']=rms(tensor_norm(ref[mask]))
                result[tag+'_strain_over_reference']=rms(tensor_norm(source))/(rms(ref[mask])*np.linalg.norm(S))
                reference=audit.read(root/'analysis/compatible_reference_profile.csv')
                u=np.sign(r[mask])*np.interp(abs(r[mask]),reference['r'],reference['speed_positive_r'],right=VP/2)
                result[tag+'_velocity_error_over_Vp']=rms(np.hypot(d['ux'][mask]-.5*u,d['uy'][mask]-geometry.sn*u))/VP
                for field in ('p','tau_xx','tau_yy','tau_xy','tauN','sigma_n'):
                    result[tag+'_'+field+'_min']=float(np.min(d[field][mask]))
                    result[tag+'_'+field+'_max']=float(np.max(d[field][mask]))
            summaries.append(result)
        admitted=data[0]['source_active'].astype(bool)
        # Geometry, phase and I_h are identical. After advection, evaluating
        # identical material degradation functions through projected mixtures
        # can change the last bits; retain the measured difference explicitly.
        admitted_difference=float(np.max(abs(data[0]['chi'][admitted]-data[1]['chi'][admitted])))
        np.testing.assert_allclose(data[0]['chi'][admitted],data[1]['chi'][admitted],rtol=1e-12,atol=1e-24)
        changed=data[1]['source_active']!=data[0]['source_active']
        assert np.all(coordinate[changed]<0) and np.all(data[0]['y'][changed]>=0)
        np.testing.assert_allclose(data[1]['chi'][changed]*VP,ref[changed],rtol=1e-12,atol=1e-25)
        closure.append(dict(step=k,added_qps=int(changed.sum()),
            admitted_max_chi_difference=admitted_difference,
            max_added_source_error=float(np.max(abs(data[1]['chi'][changed]*VP-ref[changed])))))
        audit.write(out/f'all_qp_{k}.csv',dict(cell=[key[0] for key in keys],qp=[key[1] for key in keys],
            x=data[0]['x'],y=data[0]['y'],weight=data[0]['weight'],reference=ref,
            source_control=data[0]['chi']*VP,source_continued=data[1]['chi']*VP,
            sigma_control=data[0]['sigma_n'],sigma_continued=data[1]['sigma_n']))
        if k:
            committed=audit.cat(roots[1].glob(f'mature_history_{k}_rank*.csv'))
            lookup={int(id):i for i,id in enumerate(committed['id'])}
            parts=[]
            for path in roots[1].glob(f'continued_source_history_{k}_rank*.csv'):
                if len(path.read_text().splitlines())>1:parts.append(audit.read(path))
            assert parts
            d={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
            max_error=0.
            for comp in ('xx','yy','xy'):
                expected=2*d['kappa']*(d['eps_'+comp]-d['crack_'+comp])+d['beta']*d['old_'+comp]
                np.testing.assert_allclose(d['new_'+comp],expected,rtol=1e-12,atol=1e-9)
                actual=np.array([committed['tau_'+comp][lookup[int(id)]] for id in d['id']])
                np.testing.assert_array_equal(actual,d['new_'+comp])
                max_error=max(max_error,float(np.max(abs(actual-expected))))
                np.testing.assert_allclose(d['crack_'+comp],d['chi']*VP*S[{'xx':(0,0),'yy':(1,1),'xy':(0,1)}[comp]],rtol=1e-12,atol=1e-25)
            history.append(dict(step=k,continued_particles=len(d['id']),max_formula_error_Pa=max_error))
    audit.records(out/'all_qp_comparison.csv',summaries)
    (out/'checks.json').write_text(json.dumps(dict(repeat=checks,closure=closure,history=history),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for root,label in zip(roots,['completed denominator','plus source continuation']):
        f,w=audit.moments(root,2);mask=f['y']<2000
        axes[0].plot(f['y'][mask],(w['sigma_weak_mean'][mask]-50e6)/1000,label=label)
        d=keyed(root,2);rows=[v for v in d.values() if v['y']<500 and v['phi']>0]
        axes[1].scatter([v['y'] for v in rows],[(v['sigma_n']-50e6)/1000 for v in rows],s=5,label=label)
        tag='control' if root==roots[0] else 'continued'
        rows=[v for v in summaries if v['step']==2 and v['region'].startswith('bottom')]
        axes[2].plot([100,600,1500],[v[tag+'_strain_mismatch_rms'] for v in rows],label=label)
    for ax in axes:ax.set_xlabel('Height above bottom (m)');ax.legend(fontsize=7)
    axes[0].set_ylabel('Weak mean sigma_n - 50 MPa (kPa)')
    axes[1].set_ylabel('All-QP sigma_n - 50 MPa (kPa)')
    axes[2].set_ylabel('All-QP strain mismatch RMS (1/s)');axes[2].set_yscale('log')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(checks=checks,closure=closure,history=history,final=summaries[-4:]),indent=2))

if __name__=='__main__':main()
