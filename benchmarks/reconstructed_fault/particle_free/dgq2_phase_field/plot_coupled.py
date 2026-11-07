"""Compact plots of native values and common-quadrature comparison measures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'plus':'#0072B2','plus-half':'#009E73','minus':'#D55E00','plus-fine':'#CC79A7'}

def short(name):
    return name.replace('layered-dgphase-','DGQ2 ').replace('layered-bulk-','Q1 ').replace('-cartesian','')

def plot(cases,summary,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    caption='Positive translation, h = .125; plus: Δt = .0125, plus-half: Δt = .00625'
    if not summary['complete']:fig.suptitle(caption,fontsize=12)
    for name,c in cases.items():
        dg=name.startswith('layered-dgphase-');key=short(name).split(' ',1)[1]
        col=COLORS[key];ls='-' if dg else '--'
        t=np.array([r['time'] for r in c['records']]);e=[r['energy'] for r in c['records']]
        mass=np.array([v['profile_mass'] for v in e]);p0=mass[0]
        axs[0,0].plot(t,100*(mass/p0-1),ls,color=col,label=short(name))
        axs[0,1].plot(t,[v['peak_phase'] for v in e],ls,color=col)
        axs[1,0].plot(t,np.sqrt([v['profile_normal_variance'] for v in e]),ls,color=col)
        axs[1,1].plot(t,100*np.array([v['normalization_relative'] for v in e]),ls,color=col)
    for ax,title,ylabel in zip(axs.flat,['Phase integral: source, flux and numerical change','Profile peak (volume quadrature samples)','Profile width about the mean curve normal','Localization integral versus curve length'],['Change from initialization (%)','Phase','Weighted standard deviation','Relative difference (%)']):
        ax.set(title=title,xlabel='Physical time',ylabel=ylabel);ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    if not summary['complete']:
        native=[summary['cases'][n]['phase_integral_relative']*100 for n in ('layered-dgphase-plus','layered-dgphase-plus-half') if n in cases]
        axs[0,0].text(.48,.12,'DGQ2 final: '+', '.join(f'{v:.4f}%' for v in native),transform=axs[0,0].transAxes,fontsize=9)
    fig.savefig(out/'profile-comparison.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    if not summary['complete']:fig.suptitle(caption,fontsize=12)
    for name,c in cases.items():
        dg=name.startswith('layered-dgphase-');key=short(name).split(' ',1)[1]
        col=COLORS[key];ls='-' if dg else '--'
        t=np.array([r['time'] for r in c['records']]);e=c['energy'];steps=[r['energy'] for r in c['records'][1:]]
        signed=np.r_[0,np.cumsum([r['endpoint_remainder'] for r in steps])]
        absolute=np.r_[0,np.cumsum([abs(r['endpoint_remainder']) for r in steps])]
        axs[0,0].plot(t,signed,ls,color=col,label=short(name))
        axs[0,1].plot(t,absolute,ls,color=col)
        axs[1,0].plot(t,np.r_[0,np.cumsum([r['trapezoid_remainder'] for r in steps])],ls,color=col)
        axs[1,1].plot(t,np.r_[0,np.cumsum([abs(r['trapezoid_remainder']) for r in steps])],ls,color=col)
    for ax,title in zip(axs.flat,['Endpoint signed cumulative remainder','Endpoint accumulated absolute remainder','Trapezoidal signed cumulative remainder','Trapezoidal accumulated absolute remainder']):
        ax.set(title=title,xlabel='Physical time',ylabel='Energy per out-of-plane length');ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.savefig(out/'energy-comparison.png',dpi=170);plt.close(fig)

    # Final material-label traces distinguish central and finite-end behavior.
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    if not summary['complete']:fig.suptitle(caption,fontsize=12)
    fields=['shear_traction','normal_traction','slip_rate','cumulative_slip','age','I']
    titles=['Shear traction','Normal traction (includes pressure)','Slip rate','Cumulative slip','Surface age','Normalization I']
    for name,c in cases.items():
        dg=name.startswith('layered-dgphase-');key=short(name).split(' ',1)[1]
        col=COLORS[key];ls='-' if dg else '--';r=c['records'][-1]
        for ax,k in zip(axs.flat,fields):ax.plot(c['labels'],r['profile'][k],ls,color=col,label=short(name))
    for ax,title in zip(axs.flat,titles):
        ax.set(title=title,xlabel='Material label');ax.grid(alpha=.2)
        ax.axvspan(0,.2,color='.8',alpha=.25);ax.axvspan(.8,1,color='.8',alpha=.25)
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.savefig(out/'surface-comparison.png',dpi=170);plt.close(fig)

    names=list(cases);x=np.arange(len(names));values=summary['cases']
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    if not summary['complete']:fig.suptitle(caption,fontsize=12)
    axs[0].bar(x,[values[n]['wall_s']/60 for n in names])
    axs[1].bar(x,[values[n]['phase_dofs'] for n in names])
    axs[2].bar(x,[values[n]['cost']['stokes_iterations'] for n in names])
    for ax,title in zip(axs,['Measured runtime (minutes)','Phase degrees of freedom','Total linear solver iterations']):
        ax.set(title=title,xticks=x,xticklabels=[short(n) for n in names]);ax.tick_params(axis='x',rotation=60);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'cost-comparison.png',dpi=170);plt.close(fig)

    if any(p['a']=='layered-dgphase-plus' and p['b']=='layered-dgphase-plus-half' for p in summary['pairs']):
        def pair(items,a,b):
            return next(p['final'] for p in items if p['a']==a and p['b']==b)
        old=[pair(summary['retained_q1_comparisons'],'layered-bulk-plus','layered-bulk-plus-half')]
        new=[pair(summary['pairs'],'layered-dgphase-plus','layered-dgphase-plus-half')]
        labels=['Time halving']
        if summary['complete']:
            old.append(pair(summary['retained_q1_comparisons'],'layered-bulk-plus-half','layered-bulk-plus-fine-cartesian'))
            new.append(pair(summary['pairs'],'layered-dgphase-plus-half','layered-dgphase-plus-fine'))
            labels.append('Mesh refinement')
        metrics=[('Phase field',lambda p:p['phase']['relative_L2']),
                 ('Whole-domain stress',lambda p:p['common_domain']['relative_L2']),
                 ('Central cumulative slip',lambda p:p['surface_common_quadrature']['central']['cumulative_slip']['relative_L2']),
                 ('Endpoint cumulative slip',lambda p:p['surface_common_quadrature']['ends']['cumulative_slip']['relative_L2'])]
        fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for ax,(title,measure) in zip(axs.flat,metrics):
            x=np.arange(len(labels))
            ax.bar(x-.18,[100*measure(p) for p in old],.36,label='Q1',color='#0072B2')
            ax.bar(x+.18,[100*measure(p) for p in new],.36,label='DGQ2',color='#E69F00')
            ax.set(title=title,xticks=x,xticklabels=labels,
                   ylabel='Relative L2 difference (%)');ax.grid(axis='y',alpha=.2)
        axs[0,0].legend()
        fig.suptitle(('Sensitivity to timestep and mesh refinement' if summary['complete'] else 'Positive translation: timestep sensitivity')+' — 20 fixed material labels')
        fig.savefig(out/'refinement-comparison.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[4]
    out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled'
    old=out.parent.parent/'transported_objective/layered_bulk_campaign'
    summary=json.loads((out/'comparison-summary.json').read_text())
    cases={n:json.loads(((out if n.startswith('layered-dgphase-') else old)/(n+'.json')).read_text()) for n in summary['cases']}
    plot(cases,summary,out)
