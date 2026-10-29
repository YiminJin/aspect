#!/usr/bin/env python3
"""Plots of measured native exact-jump operators, traces and accepted histories."""
import os,json,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-interface-exact')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_cohesive_interface_exact import OUT,OLD,read,traces,write,reference

def array(rows,key):return np.array([r[key] for r in rows])
def save(fig,name,title):
    fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/name,dpi=160);plt.close(fig)

def main():
    reference.OUT=OUT
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    summary=json.loads((OUT/'summary.json').read_text());modes=summary['operator_modes'][:4]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    old=json.loads((OLD/'summary.json').read_text())['operator_modes'];old=[r for r in old if r['width_m']==.025]
    axes[0,0].semilogy([r['n'] for r in old],[100*abs(r['K_width_difference']) for r in old],'o--',label='source-only, retained')
    axes[0,0].semilogy([r['segments'] for r in modes],[100*r['K_relative_error'] for r in modes],'o-',label='exact jump')
    axes[0,0].set(xlabel='Surface segments',ylabel='Shear error against sharp (%)');axes[0,0].legend()
    axes[0,1].semilogy([r['segments'] for r in modes],[100*r['N_relative_error'] for r in modes],'s-',label='normal coupling error')
    axes[0,1].semilogy([r['segments'] for r in modes],[100*r['normal_trace_jump_over_sharp_K'] for r in modes],'o-',label='two-side mismatch / sharp K')
    axes[0,1].set(xlabel='Surface segments',ylabel='Resolved 1 m mode (%)');axes[0,1].legend(fontsize=8)
    n=32;Q,w,x,P=reference.surface_rule(n);z=1e-4*(1+.3*np.cos(2*np.pi*np.arange(n)/n));u=2e-4;K,N,_=reference.operators('operator-32');ts0,ts1,tn0,tn1=traces('operator-32');sharp=reference.sharp_q1(n);v=np.exp(2j*np.pi*np.arange(n)/n)
    sref=P@(np.full(n,reference.ref.G/.125*(u-1e-4))-(sharp[0]*3e-5*v).real)
    nref=10000+P@(sharp[1]*3e-5*v).real
    slo=reference.ref.G/.125*u+P@ts0@z;sup=reference.ref.G/.125*u+P@ts1@z
    nlo=10000+P@tn0@z;nup=10000+P@tn1@z
    axes[1,0].plot(x,slo,label='lower trace');axes[1,0].plot(x,sup,'--',label='upper trace');axes[1,0].plot(x,sref,':',label='sharp same-space reference')
    axes[1,0].set(xlabel='Along-fault x (m)',ylabel='Physical shear (Pa)');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(x,nlo,label='lower trace');axes[1,1].plot(x,nup,'--',label='upper trace');axes[1,1].plot(x,nref,':',label='sharp reference')
    axes[1,1].set(xlabel='Along-fault x (m)',ylabel='Normal pressure (Pa)');axes[1,1].legend(fontsize=8)
    write('frozen_trace_profiles.csv',[dict(x_m=float(x[i]),lower_shear_Pa=float(slo[i]),upper_shear_Pa=float(sup[i]),sharp_shear_Pa=float(sref[i]),lower_pn_Pa=float(nlo[i]),upper_pn_Pa=float(nup[i]),sharp_pn_Pa=float(nref[i])) for i in range(len(x))])
    save(fig,'operators_and_traces.png','Exact jump removes width bias; physical traces approach the same-space sharp reference')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    with (OUT/'native_reference_histories.csv').open() as f:comparisons=[{k:(v if k=='case' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
    for name in ['formation','formation-half']:
        r=[r for r in comparisons if r['case']==name];axes[0,0].plot(array(r,'time_s'),array(r,'mean_damage'),label=name)
    axes[0,0].axvline(.3757048877712288,color='k',ls=':',label='analytic admission');axes[0,0].set(xlim=(.36,.42),xlabel='Time (s)',ylabel='Surface damage a');axes[0,0].legend(fontsize=8)
    r=read(OUT/'mature/states.csv');times=np.unique(array(r,'time_s'));V=[];slip=[]
    for t in times:
        rr=[a for a in r if a['time_s']==t];V.append(np.mean(array(rr,'V_m_s')));slip.append(np.mean(array(rr,'absolute_slip_m'))/.001)
    axes[0,1].plot(times,np.array(V)*1000,label='signed geometric rate');axes[0,1].axhline(0,color='.5',lw=.7);axes[0,1].set(xlabel='Time (s)',ylabel='Slip velocity (mm/s)')
    sec=axes[0,1].twinx();sec.plot(times,slip,ls='--',color='tab:orange');sec.set_ylabel('Absolute slip / Dc',color='tab:orange')
    for name,folder in [('exact jump',OUT),('source-only',OLD)]:
        r=read(folder/'nonuniform/states.csv');r=[a for a in r if a['time_s']==max(array(r,'time_s'))]
        axes[1,0].plot(array(r,'x_m'),array(r,'pn_Pa'),label=name)
    axes[1,0].set(xlabel='Along-fault x (m)',ylabel='Current projected normal pressure (Pa)');axes[1,0].legend()
    labels=[];errors=[]
    for r in summary['runs']:labels.append(r['case']);errors.append(100*r['absolute_work_error_over_sliding'])
    axes[1,1].bar(labels,errors);axes[1,1].set(yscale='log',ylabel='Σ |physical work error| / sliding loss (%)');axes[1,1].tick_params(axis='x',labelrotation=15)
    save(fig,'response_and_energy.png','Same law and history lifecycle: intact admission, signed reversal and energy errors')
    exact=read(OUT/'operator-32/frozen_fields.csv');old=read(OLD/'operator-32/frozen_fields.csv')
    assert all(a['x_m']==b['x_m'] and a['y_m']==b['y_m'] for a,b in zip(exact,old))
    fig,axes=plt.subplots(3,1,figsize=(11,5.5));x=array(exact,'x_m');y=array(exact,'y_m')
    for ax,data,label in [(axes[0],array(exact,'tau_xy_Pa'),'Exact-jump physical τxy (Pa)'),(axes[1],array(exact,'p_perturbation_Pa'),'Exact-jump p′ (Pa)'),(axes[2],array(exact,'tau_xy_Pa')-array(old,'tau_xy_Pa'),'τxy exact − source-only (Pa)')]:
        im=ax.scatter(x,y,c=data,cmap='coolwarm',s=5,marker='s');ax.axhline(.05,color='k',lw=.8);ax.set(xlim=(0,1),ylim=(0,.125),ylabel='y (m)',aspect='equal');fig.colorbar(im,ax=ax,label=label,fraction=.02,pad=.02)
    axes[-1].set_xlabel('x (m)')
    save(fig,'frozen_fields.png','Same nonuniform memory probe: reconstructed bulk stress and pressure, not a propagation run')

if __name__=='__main__':main()
