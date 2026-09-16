"""Read-only uniform-rate diagnostic and earlier normal-traction profiles."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid,simpson
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=HERE/'uniform-sliding-50-local4'
OUT=RUN/'analysis'
SN=np.sqrt(3)/2
VP=1e-9
L=100000/SN

def read(path,skip=()):
    with path.open() as f:names=next(csv.reader(f))
    cols=[j for j,n in enumerate(names) if n not in skip]
    a=np.loadtxt(path,delimiter=',',skiprows=1,usecols=cols,ndmin=2)
    return {names[j]:a[:,i] for i,j in enumerate(cols)}

def cat(paths,skip=()):
    pieces=[read(p,skip) for p in sorted(paths)]
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}

def select(data,mask):return {k:v[mask] for k,v in data.items()}

def write(path,data):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(data);w.writerows(zip(*data.values()))

def records(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def moments(root,k):
    f=read(root/f'fault_{k}.csv');n=len(f['V'])
    parts=cat(root.glob(f'stress_weak_moments_{k}_rank*.csv'))
    w={key:np.bincount(parts['node'].astype(int),weights=parts[key],minlength=n)
       for key in ('weight','p_load','tauN_load','sigma_load','tensile_weight')}
    w.update(node=np.arange(n),xd=f['xd'])
    band=np.zeros((3,n));band[1]=f['mass_diagonal']
    band[0,1:]=f['mass_upper'][:-1];band[2,:-1]=f['mass_upper'][:-1]
    for name,load in [('p','p_load'),('minus_tauN','tauN_load'),('sigma','sigma_load')]:
        sign=-1 if name=='minus_tauN' else 1
        w[name+'_weak_mean']=sign*w[load]/w['weight']
        w[name+'_Q1']=solve_banded((1,1),band,sign*w[load])
    for suffix in ('weak_mean','Q1'):
        np.testing.assert_allclose(w['sigma_'+suffix],50e6+w['p_'+suffix]+w['minus_tauN_'+suffix],rtol=1e-12)
    return f,w

def raw_samples(root,k,f):
    raw=cat(root.glob(f'stress_samples_{k}_rank*.csv'),('selection','cell'))
    # A retained extreme can appear in multiple support classes. Count once.
    keys=np.column_stack([raw[k] for k in ('rank','particle','domain_q','segment','xi')])
    _,indices=np.unique(keys,axis=0,return_index=True);raw=select(raw,indices)
    j=raw['segment'].astype(int);z=raw['xi']
    raw['xd']=(1-z)*f['xd'][j]+z*f['xd'][j+1];raw['minus_tauN']=-raw['delta_tau_N']
    np.testing.assert_allclose(raw['sigma_n'],raw['sigma_bg']+raw['delta_p']+raw['minus_tauN'],rtol=1e-14)
    return raw

def extract_existing():
    result=[]
    for case,steps in [('fault-grid-50-local4',[0,10,12,13]),('mature-fault-50-local4',[0,2,10])]:
        root=HERE/case
        for k in steps:
            f,w=moments(root,k);raw=raw_samples(root,k,f)
            fig,axes=plt.subplots(2,2,figsize=(11,7))
            for col,(name,lo,hi) in enumerate([('transition',13000,20000),('junction',37000,43000)]):
                s=select(w,(w['xd']>=lo)&(w['xd']<=hi));s=select(s,np.argsort(s['xd']))
                r=select(raw,(raw['xd']>=lo)&(raw['xd']<=hi));r=select(r,np.argsort(r['xd']))
                write(OUT/f'existing_{case}_{k}_{name}_weak.csv',s)
                write(OUT/f'existing_{case}_{k}_{name}_raw_extrema.csv',r)
                for suffix,ax in [('weak_mean',axes[0,col]),('Q1',axes[1,col])]:
                    for field,label in [('p','delta p'),('minus_tauN','-delta tau:N'),('sigma','sigma_n-50 MPa')]:
                        y=s[field+'_'+suffix]-(50e6 if field=='sigma' else 0)
                        ax.plot(s['xd']/1000,y/1e6,label=label)
                    for mark in (15,18,40):
                        if lo/1000<=mark<=hi/1000:ax.axvline(mark,color='k',ls=':',lw=.7)
                    ax.set(xlabel='Down dip (km)',ylabel='MPa',title=name+' / '+suffix);ax.legend(fontsize=7)
                if len(r['xd']):axes[0,col].scatter(r['xd']/1000,(r['sigma_n']-50e6)/1e6,s=6,c='k')
                d=dict(case=case,step=k,time_yr=float(f['time'][0]/31557600),region=name,raw_count=len(r['xd']))
                for field in ('p','minus_tauN','sigma'):
                    for suffix in ('weak_mean','Q1'):d[field+'_'+suffix+'_ptp_Pa']=float(np.ptp(s[field+'_'+suffix]))
                d['raw_sigma_min_Pa']=float(np.min(r['sigma_n'])) if len(r['xd']) else None
                d['raw_sigma_max_Pa']=float(np.max(r['sigma_n'])) if len(r['xd']) else None
                result.append(d)
            fig.suptitle(f'{case}, step {k}; raw dots are retained extrema only')
            fig.tight_layout();fig.savefig(OUT/f'existing_{case}_{k}.png',dpi=160);plt.close(fig)
    records(OUT/'existing_profile_summary.csv',result)

def analyze_run():
    clock=read(RUN/'accepted_steps.csv')
    np.testing.assert_array_equal(clock['step'],[0,1,2])
    assert np.all(clock['free']==0) and np.all(clock['lower_active']==0)
    log=(RUN/'run.log').read_text()
    pairs=[tuple(map(float,x)) for x in re.findall(r'Fault linear solve: iterations=\d+, estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)]
    assert pairs and all(a<=b for a,b in pairs)
    residuals=[];parts=re.split(r'\*\*\* Timestep (\d+):',log)
    for i in range(1,len(parts),2):
        k=int(parts[i]);r=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',parts[i+1])[-1]
        assert all(float(v)<1e-8 for v in r)
        residuals.append(dict(step=k,bulk=float(r[0]),surface=float(r[1])))
    reference=read(RUN/'uniform_reference_profile.csv')
    _,index=np.unique(reference['r'],return_index=True);reference=select(reference,index)
    # Independently integrate the exported prescribed distance profile. This
    # never substitutes for production I_h or removes physical truncation.
    Isimpson=2*simpson(reference['h'],x=reference['r']);Iref=2*np.trapezoid(reference['h'],reference['r'])
    integral=cumulative_trapezoid(reference['h'],reference['r'],initial=0)
    reference['chi_ref']=reference['h']/Iref;reference['speed_positive_r']=VP*integral/Iref
    write(OUT/'compatible_reference_profile.csv',reference)
    summary=dict(execution=json.loads((RUN/'execution.json').read_text()),fresh_linear_checks=len(pairs),
        worst_fresh_ratio=max(a/b for a,b in pairs),nonlinear=residuals,I_reference_m=Iref,
        I_trapezoid_difference_m=Iref-Isimpson,profile_radius_m=float(reference['r'].max()),
        boundary_min_abs_r_m=.5*100000*(SN-.5))
    assert summary['boundary_min_abs_r_m']>reference['r'].max()
    # The integral of piecewise-linear h has a piecewise-quadratic primitive.
    # Its analytic derivative and finite differences reproduce chi Vp S.
    def plate(point):
        r=float(np.dot(point,[-SN,.5]));a=abs(r)
        j=min(np.searchsorted(reference['r'],a,side='right')-1,len(integral)-2)
        if a>=reference['r'][-1]:primitive=Iref/2
        else:
            dr=a-reference['r'][j]
            slope=(reference['h'][j+1]-reference['h'][j])/(reference['r'][j+1]-reference['r'][j])
            primitive=integral[j]+reference['h'][j]*dr+.5*slope*dr*dr
        return np.array([.5,SN])*np.sign(r)*VP*primitive/Iref
    S=.5*(np.outer([.5,SN],[-SN,.5])+np.outer([-SN,.5],[.5,SN]))
    derivative_error=[]
    for r in (10.,75.,200.,400.,700.):
        p=np.array([-SN,.5])*r;step=.001;gradient=np.zeros((2,2))
        for j in range(2):
            e=np.eye(2)[j]*step;gradient[:,j]=(plate(p+e)-plate(p-e))/(2*step)
        exact=VP*np.interp(r,reference['r'],reference['h'])/Iref*S
        derivative_error.append(float(np.linalg.norm(.5*(gradient+gradient.T)-exact)/np.linalg.norm(exact)))
    assert max(derivative_error)<1e-6
    summary['reference_symmetric_gradient_relative_error']=max(derivative_error)
    boundary=read(RUN/'velocity_constraints.csv',('side',))
    assert np.max(boundary['max_actual_error'])==0
    summary['physical_lateral_constraint_error']=float(np.max(boundary['max_actual_error']))
    regions=[('bottom_0_200m',lambda d:d['y']<200),('bottom_200_1000m',lambda d:(d['y']>=200)&(d['y']<1000)),
        ('bottom_1000_2000m',lambda d:(d['y']>=1000)&(d['y']<2000)),
        ('interior_59_61km',lambda d:(d['xd']>59000)&(d['xd']<61000)),
        ('junction_37_43km',lambda d:(d['xd']>37000)&(d['xd']<43000)),
        ('transition_13_20km',lambda d:(d['xd']>13000)&(d['xd']<20000))]
    rows=[];extrema=[];normal=[];previous=None;initial_history=None;initial_fault=None
    for k in range(3):
        f,w=moments(RUN,k)
        assert np.all(f['V']==VP) and np.all(f['prescribed']==1) and np.all(f['C']==0)
        if initial_fault is None:initial_fault=f
        for key in ('x','y','Ih'):np.testing.assert_array_equal(f[key],initial_fault[key])
        if previous is not None:
            x=VP*clock['dt'][k]/.008
            expected=previous['Theta']*np.exp(-x)-.008/VP*np.expm1(-x)
            np.testing.assert_allclose(f['Theta'],expected,rtol=1e-12,atol=0)
            np.testing.assert_allclose(f['slip'],previous['slip']+VP*clock['dt'][k],rtol=1e-13,atol=1e-15)
        history=cat(RUN.glob(f'mature_history_{k}_rank*.csv'))
        history=select(history,np.argsort(history['id']))
        assert len(np.unique(history['id']))==len(history['id'])
        if initial_history is None:
            initial_history=history
            for key in ('tau_xx','tau_yy','tau_xy'):assert np.max(abs(history[key]))==0
        for key in ('id','H_inert'):np.testing.assert_array_equal(history[key],initial_history[key])
        previous=f
        write(OUT/f'uniform_surface_{k}.csv',w)
        d=cat(RUN.glob(f'uniform_bulk_{k}_rank*.csv'),('cell',))
        if k==0:assert max(np.max(abs(d[key])) for key in ('old_xx','old_yy','old_xy'))==0
        np.testing.assert_allclose(d['sigma_n'],50e6+d['p']-d['tauN'],rtol=1e-14)
        # The first observer used deviator(eps), while production uses raw
        # sym(grad u): incompressibility holds weakly, not pointwise. Correct
        # ONLY the diagnostic using the saved strain/kappa; no new solve and
        # no Maxwell history update. Preserve original CSVs and label outputs.
        provenance=json.loads((RUN/'provenance.json').read_text())
        old_observer=provenance['sha256'][str(HERE/'uniform_sliding.h')]=='d614627eb512d4ed94cdae2a66dfff2b350f4e29508312db1406dfbc86c3f945'
        correction=d['kappa']*(d['eps_xx']+d['eps_yy']) if old_observer else np.zeros(len(d['x']))
        for key in ('tau_xx','tau_yy','tauN'):
            d['original_deviatorized_'+key]=d[key].copy();d[key]=d[key]+correction
        d['original_deviatorized_sigma_n']=d['sigma_n'].copy()
        d['sigma_n']=50e6+d['p']-d['tauN']
        d['h_ref']=np.interp(abs(d['r']),reference['r'],reference['h'],left=reference['h'][0],right=0)
        d['chi_ref']=d['h_ref']/Iref
        d['u_ref_s']=np.sign(d['r'])*VP*np.interp(abs(d['r']),reference['r'],integral,right=Iref/2)/Iref
        d['u_reference_error']=np.hypot(d['ux']-.5*d['u_ref_s'],d['uy']-SN*d['u_ref_s'])
        for name,mask in regions:
            a=select(d,mask(d));weight=a['weight'];volume=weight.sum()
            rms=lambda value:float(np.sqrt(np.sum(weight*value**2)/volume))
            cracknorm=np.sqrt(a['crack_xx']**2+a['crack_yy']**2+2*a['crack_xy']**2)
            row=dict(step=k,region=name,points=len(weight),volume_m2=float(volume),
                elastic_strain_rms=rms(a['elastic_norm']),crack_strain_rms=rms(cracknorm),
                elastic_over_crack_rms=rms(a['elastic_norm'])/rms(cracknorm),
                u_reference_error_rms_over_Vp=rms(a['u_reference_error'])/VP,
                chi_min=float(a['chi'].min()),chi_max=float(a['chi'].max()),Ih_min=float(a['Ih'].min()),Ih_max=float(a['Ih'].max()),
                p_min=float(a['p'].min()),p_max=float(a['p'].max()),sigma_min=float(a['sigma_n'].min()),sigma_max=float(a['sigma_n'].max()),
                chi_difference_rms=rms(a['chi']-a['chi_ref']))
            rows.append(row)
            for kind,col in [('minimum','sigma_n'),('maximum','sigma_n'),('elastic','elastic_norm')]:
                index=np.argmin(a[col]) if kind=='minimum' else np.argmax(a[col])
                rec={key:float(value[index]) for key,value in a.items()};rec.update(step=k,region=name,selection=kind);extrema.append(rec)
        raw=raw_samples(RUN,k,f);write(OUT/f'uniform_raw_extrema_{k}.csv',raw)
        for name,lo,hi in [('bottom',L-2000,L+1),('transition',13000,20000),('junction',37000,43000)]:
            a=select(w,(w['xd']>=lo)&(w['xd']<=hi));b=select(raw,(raw['xd']>=lo)&(raw['xd']<=hi))
            normal.append(dict(step=k,region=name,weak_min=float(a['sigma_weak_mean'].min()),weak_max=float(a['sigma_weak_mean'].max()),
                Q1_min=float(a['sigma_Q1'].min()),Q1_max=float(a['sigma_Q1'].max()),
                raw_min=float(b['sigma_n'].min()) if len(b['xd']) else None,raw_max=float(b['sigma_n'].max()) if len(b['xd']) else None))
        write(OUT/f'uniform_bulk_sample_{k}.csv',select(d,np.arange(0,len(d['x']),50)))
        write(OUT/f'uniform_bottom_bulk_{k}.csv',select(d,d['y']<2000))
    records(OUT/'uniform_bulk_summary.csv',rows);records(OUT/'uniform_bulk_extrema.csv',extrema)
    records(OUT/'uniform_normal_summary.csv',normal);summary['normal_traction']=normal
    summary['bulk_diagnostic_note']=('Raw run used deviator(eps). Offline CSVs add kappa*trace(eps)*Identity to match production; raw surface diagnostics are unchanged.'
        if old_observer else 'Observer uses raw symmetric strain, matching production; no offline trace correction applied.')
    summary['stable_particle_ids_H_verified']=len(initial_history['id'])
    # Plot the endpoint I_h and actual raw versus weak constitutive traction.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for k in range(3):
        f,w=moments(RUN,k);mask=f['y']<2000
        axes[0].plot(f['y'][mask],f['Ih'][mask],label=f'step {k}')
        axes[1].plot(f['y'][mask],(w['sigma_weak_mean'][mask]-50e6)/1e3,label=f'step {k} weak mean')
        r=raw_samples(RUN,k,f);mask=r['surface_y']<2000
        axes[1].scatter(r['surface_y'][mask],(r['sigma_n'][mask]-50e6)/1e3,s=8)
        b=read(OUT/f'uniform_bottom_bulk_{k}.csv')
        edges=np.array([0,100,200,400,800,1200,2000]);values=[]
        for a,z in zip(edges[:-1],edges[1:]):
            mask=(b['y']>=a)&(b['y']<z);weight=b['weight'][mask]
            values.append(np.sqrt(np.sum(weight*b['elastic_norm'][mask]**2)/weight.sum()))
        axes[2].plot(.5*(edges[:-1]+edges[1:]),values,label=f'step {k}')
    axes[0].axhline(Iref,color='k',ls=':',label='infinite reference')
    for ax in axes:ax.set_xlabel('Height above bottom (m)');ax.legend(fontsize=7)
    axes[0].set_ylabel('I_h (m)');axes[1].set_ylabel('sigma_n - 50 MPa (kPa)')
    axes[2].set_ylabel('RMS |sym grad u - chi Vp S| (1/s)');axes[2].set_yscale('log')
    fig.tight_layout();fig.savefig(OUT/'uniform_bottom.png',dpi=160);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);extract_existing()
    if (RUN/'execution.json').exists():analyze_run()
