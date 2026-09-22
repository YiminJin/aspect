"""Offline native-work traction decomposition; never advances a simulation."""
import json
from pathlib import Path
import numpy as np
from analyze_loading_startup import OUT, table, raw_qps
from startup_30km import parameters

DEST=OUT/'traction-audit'
CASES={'initial':('startup',0),'first':('startup',1),'full':('startup',2),
       'last':('startup',3),'half':('half',3)}


def chord(z):
    # Diagnostic only. Original curves are never filtered or replaced.
    v=np.full_like(z,np.nan);v[1:-1]=z[1:-1]-.5*(z[:-2]+z[2:]);return v


def metric(z,w):
    return dict(max=float(max(abs(z))),RMS=float(np.sqrt(np.dot(w,z*z)/sum(w))))


def collect(label):
    case,k=CASES[label]; path=OUT/case
    raw=np.concatenate([raw_qps(p) for p in sorted(path.glob(f'work_qp_{k}_rank*.csv'))])
    raw=raw[(raw['source_active']==1)&(raw['chi']>0)]
    j=raw['segment'].astype(int);xi=raw['xi']; W=raw['JxW']*raw['chi']
    native=table(path/f'work_weak_{k}.csv');state=table(path/f'state_work_{k}.csv')
    profile=table(path/f'profiles/fault_{k}.csv'); init=table(OUT/'startup/steady_initialization.csv')
    for other in (state,profile,init):np.testing.assert_array_equal(native['node'],other['node'])
    assert np.all(np.diff(native['xd'])<0), 'Stored fault ordering changed'
    np.testing.assert_allclose(native['xd'],profile['xd_m'],rtol=0,atol=1e-9)
    prm=parameters((path/'parameters.prm').read_text());section=('Material model','Phase field fault')
    assert float(prm[section+('Thermal viscosity exponents',)])==0
    eta=float(prm[section+('Reference viscosities',)]);G=float(prm[section+('Elastic shear moduli',)])
    assert float(prm[section+('Minimum viscosity',)])<eta<float(prm[section+('Maximum viscosity',)])
    dt=float(prm[section+('Initial time step',)]) if k==0 else state['dt'][0]
    kappa=-eta*np.expm1(-dt*G/eta);beta=np.exp(-dt*G/eta)
    # Same tangent and normal as the observer; symmetric tensor contraction
    # counts off-diagonal components twice. S:S=1/2 and S:N=0.
    tangent=np.array([profile['x_m'][1]-profile['x_m'][0],profile['y_m'][1]-profile['y_m'][0]])
    tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
    S=.5*(np.outer(tangent,normal)+np.outer(normal,tangent));N=np.outer(normal,normal)
    def contract(prefix,tensor):
        return raw[prefix+'_xx']*tensor[0,0]+raw[prefix+'_yy']*tensor[1,1]+2*raw[prefix+'_xy']*tensor[0,1]
    epsS=contract('eps',S);tauS=contract('tau',S)
    tauN=contract('tau',N);epsN=contract('eps',N)
    bg=(1-xi)*init['tau_bg'][j]+xi*init['tau_bg'][j+1]
    strain=2*kappa*epsS;slip=-2*kappa*raw['chi']*raw['V']*np.sum(S*S)
    history=tauS-strain-slip
    assert max(abs(tauN-raw['tauN']))<1e-7
    assert max(abs(bg+tauS-raw['q']))<1e-6
    if k in (0,1) and case=='startup':assert max(abs(history))<1e-6
    def integrate(z):
        result=np.zeros(len(native))
        for end,shape in [(0,1-xi),(1,xi)]:
            result+=np.bincount(j+end,weights=W*shape*z,minlength=len(native))
        return result
    mass=integrate(np.ones(len(raw)))
    complete=abs(mass/native['weight']-1)<1e-12
    interior=complete&(native['xd']>=20000)&(native['xd']<=42000)
    def average(z):return integrate(z)/native['weight']
    alternating=(-1.)**np.arange(len(native))
    mass_alternating=integrate((1-xi)*alternating[j]+xi*alternating[j+1])*alternating/native['weight']
    P=average(raw['p']);D=average(-tauN);sumN=P+D
    qstrain=average(strain);qslip=average(slip);qhistory=average(history)
    checks=dict(pressure=max(abs(P[interior]-native['p'][interior]/native['weight'][interior])),
                deviatoric=max(abs(D[interior]+native['tauN'][interior]/native['weight'][interior])),
                normal_sum=max(abs(sumN[interior]-(native['sigma'][interior]/native['weight'][interior]-5e7))),
                shear_sum=max(abs((qstrain+qslip+qhistory)[interior]-(native['q'][interior]-native['bg'][interior])/native['weight'][interior])))
    # Compare loads, not point values: M times the saved consistent Q1 field.
    for key,column,nload in [('reconstructed_q','q_weak_Pa','q'),('reconstructed_sigma','sigma_n_weak_Pa','sigma')]:
        projected=profile[column]; recon=integrate((1-xi)*projected[j]+xi*projected[j+1])
        checks[key]=max(abs(recon[interior]-native[nload][interior])/native['weight'][interior])
    assert max(checks.values())<1e-5,checks
    # Rows outside exported support are deliberately NaN, never treated as zero.
    partial=dict(strain=qstrain,slip=qslip,history=qhistory,
                 normal_strain=average(-2*kappa*epsN),normal_history=average(-tauN+2*kappa*epsN),
                 chi_work=average(raw['chi']))
    for z in partial.values():z[~complete]=np.nan
    result=dict(xd=native['xd'],weight=native['weight'],**partial,
                P=native['p']/native['weight'],D=-native['tauN']/native['weight'],
                normal=(native['p']-native['tauN'])/native['weight'],
                q=(native['q']-native['bg'])/native['weight'],
                q_total=native['q']/native['weight'],sigma=native['sigma']/native['weight'],
                q_reconstructed=profile['q_weak_Pa']-init['tau_bg'],
                sigma_reconstructed=profile['sigma_n_weak_Pa']-5e7,
                V=state['V'],Theta_in=state['Theta_in'],Theta_out=state['Theta_out'],
                Ih=average(raw['Ih']),mass_alternating_ratio=mass_alternating,complete=complete.astype(float))
    for name in ('q','q_reconstructed','normal','P','D','strain','slip','history','chi_work','weight','V'):
        result['chord_'+name]=chord(result[name])
    np.savetxt(DEST/f'{label}.csv',np.column_stack(list(result.values())),delimiter=',',
               header=','.join(result),comments='')
    return result,dict(time=float(state['time'][0]),dt=float(dt),kappa=float(kappa),beta=float(beta),
                       rows=int(sum(interior)),checks_Pa={k:float(v) for k,v in checks.items()})


def run():
    DEST.mkdir(exist_ok=True)
    data={};info={}
    for label in CASES:data[label],info[label]=collect(label)
    x=data['full']['xd'];w=data['full']['weight']
    windows={'VW_27_29.8':(27000,29800),'transition_30.2_32.8':(30200,32800),
             'VS_33.2_36':(33200,36000)}
    comparison={}
    for name,(left,right) in windows.items():
        mask=(x>=left)&(x<=right);record={}
        for field in ('P','D','normal','q','q_reconstructed','strain','slip','history','chi_work','V'):
            a,b=data['full'][field],data['half'][field]
            ca,cb=chord(a)[mask],chord(b)[mask]
            corr=float(np.corrcoef(ca,cb)[0,1])
            record[field]=dict(full_chord=metric(ca,w[mask]),half_chord=metric(cb,w[mask]),
                correlation=corr,half_over_full_RMS=metric(cb,w[mask])['RMS']/metric(ca,w[mask])['RMS'],
                difference=metric((b-a)[mask],w[mask]),
                opposite_chord_sign_count=int(sum(ca*cb<0)),nodes=int(sum(mask)),
                initial_chord=metric(chord(data['initial'][field])[mask],w[mask]))
        # Signed projection onto the total sawtooth: additive component shares
        # retain cancellation and sum to 1, unlike separate absolute magnitudes.
        for label in ('full','half'):
            total=chord(data[label]['q'])[mask];norm=np.dot(w[mask]*total,total)
            record[label+'_signed_shear_shares']={f:float(np.dot(w[mask]*total,chord(data[label][f])[mask])/norm)
                for f in ('strain','slip','history')}
        comparison[name]=record
    report=dict(states=info,windows=comparison,
                fault_spacing_m=float(np.median(abs(np.diff(x)))),
                history_note='beta*tau_old_FE inferred from saved current tensor minus current strain/slip terms; not newly committed particle stress.')
    report['geometry_correlations']={}
    for name,(left,right) in windows.items():
        mask=(x>=left)&(x<=right)
        report['geometry_correlations'][name]={label:{
            field:float(np.corrcoef(data[label]['chord_chi_work'][mask],data[label]['chord_'+field][mask])[0,1])
            for field in ('q','V')} for label in ('initial','full','half')}
    mask=(x>=27000)&(x<=29800)
    report['VW_geometry']=dict(local_fault_spacing_m=[float(min(abs(np.diff(x[mask])))),
                                                      float(max(abs(np.diff(x[mask]))))],
        mass_alternating_ratio_range=[float(min(data['full']['mass_alternating_ratio'][mask])),
                                      float(max(data['full']['mass_alternating_ratio'][mask]))],
        chi_work_peak_to_peak_over_mean=float(np.ptp(data['full']['chi_work'][mask])/np.mean(data['full']['chi_work'][mask])))
    (DEST/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plots(data)
    print(json.dumps(report,indent=2))


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for label,style in [('initial',':'),('full','-'),('half','--')]:
        d=data[label];order=np.argsort(d['xd'])
        for ax,name in zip(axes,('P','D','normal')):
            ax.plot(d['xd'][order]/1000,d[name][order],style,label=label)
    for ax,title in zip(axes,('Pressure (Pa)','-Deviatoric normal (Pa)','Their sum: normal perturbation (Pa)')):
        ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Down-dip distance (km)');fig.tight_layout()
    fig.savefig(DEST/'whole-fault-normal.png',dpi=180);plt.close(fig)
    # Every curve is an unsmoothed native row value in physical node order.
    for window,limits in [('transition',(27000,36000)),('VW_closeup',(27000,29800))]:
        x=data['full']['xd']; ids=np.argsort(x);ids=ids[(x[ids]>=limits[0])&(x[ids]<=limits[1])]
        xx=x[ids]/1000
        fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
        for col,label in enumerate(('full','half')):
            d=data[label]
            for f,text in [('P','pressure'),('D','-deviatoric normal'),('normal','sum')]:
                axes[0,col].plot(xx,d[f][ids],label=text)
            for f,text in [('strain','strain'),('slip','current slip'),('history','incoming history'),('q','sum')]:
                axes[1,col].plot(xx,d[f][ids],label=text)
            axes[2,col].plot(xx,d['q'][ids],label='native load / row mass')
            axes[2,col].plot(xx,d['q_reconstructed'][ids],label='consistent Q1 projection',ls='--')
            axes[0,col].set_title(label+'; common time 2,086,054.244 s')
        for row,title in enumerate(('Normal stress perturbation (Pa)','Shear perturbation components (Pa)',
                                     'Net shear perturbation (Pa)')):
            # Identical axes for both time discretizations, without smoothing.
            lo=min(ax.get_ylim()[0] for ax in axes[row]);hi=max(ax.get_ylim()[1] for ax in axes[row])
            for ax in axes[row]:ax.set_ylim(lo,hi);ax.set_ylabel(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
        fig.tight_layout();fig.savefig(DEST/f'components-{window}.png',dpi=180);plt.close(fig)

        fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
        for label,style in [('full','-'),('half','--')]:
            d=data[label];initial=data['initial']
            series=[(d['P']-initial['P']),(d['D']-initial['D']),d['normal']-initial['normal'],
                    d['q']-initial['q'],d['V']/1e-9,d['Theta_in']/1e8]
            for ax,z in zip(axes.flat,series):ax.plot(xx,z[ids],style,label=label,marker='.',ms=2)
        titles=['Pressure change (Pa)','-Deviatoric normal change (Pa)','Normal sum change (Pa)',
                'Native shear change (Pa)','Accepted V/Vp','Mechanics incoming Theta / 1e8 s']
        for ax,title in zip(axes.flat,titles):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
        fig.tight_layout();fig.savefig(DEST/f'matched-{window}.png',dpi=180);plt.close(fig)

    # Near-neighbor chord departure exposes grid-scale structure without
    # replacing original fields by a fitted, detrended or smoothed curve.
    d=data['full'];ids=np.argsort(d['xd']);ids=ids[(d['xd'][ids]>=27000)&(d['xd'][ids]<=29800)]
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for label in ('initial','full','half'):
        for ax,f in zip(axes,('normal','q','V')):
            scale=1e9 if f=='V' else 1
            ax.plot(d['xd'][ids]/1000,chord(data[label][f])[ids]*scale,'.-',label=label)
    for ax,title in zip(axes,('Normal chord departure (Pa)','Shear chord departure (Pa)','V/Vp chord departure')):
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend()
    axes[-1].set_xlabel('Down-dip distance (km)');fig.tight_layout()
    fig.savefig(DEST/'neighbor-chords.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    for label,style in [('full','.-'),('half','.--')]:
        for ax,f in zip(axes.flat,('P','D','normal','strain','slip','history')):
            ax.plot(d['xd'][ids]/1000,chord(data[label][f])[ids],style,label=label)
    for ax,title in zip(axes.flat,('pressure','-deviatoric normal','normal sum',
                                  'strain shear','current-slip shear','incoming-history shear')):
        ax.set_ylabel(title+' chord departure (Pa)');ax.grid(alpha=.2);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
    fig.suptitle('Unfiltered nearest-neighbor diagnostic; signed components retain cancellation')
    fig.tight_layout();fig.savefig(DEST/'component-chords.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for col,label in enumerate(('full','half')):
        for ax,field,projected in [(axes[0,col],'q','q_reconstructed'),
                                   (axes[1,col],'normal','sigma_reconstructed')]:
            ax.plot(d['xd'][ids]/1000,data[label][field][ids],'.-',label='native load / row mass')
            ax.plot(d['xd'][ids]/1000,data[label][projected][ids],'.--',label='consistent Q1 projection')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0,col].set_title(label)
    for row,title in [(0,'Shear perturbation (Pa)'),(1,'Normal perturbation (Pa)')]:
        lo=min(ax.get_ylim()[0] for ax in axes[row]);hi=max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:ax.set_ylim(lo,hi);ax.set_ylabel(title)
    for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
    fig.tight_layout();fig.savefig(DEST/'native-versus-Q1.png',dpi=180);plt.close(fig)


if __name__=='__main__':run()
