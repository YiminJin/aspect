#!/usr/bin/env python3
"""Compact physical-energy and response evidence; never correct the raw error."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cohesive_interface_energy as e

OUT=e.OUT

def read(name):return np.genfromtxt(OUT/name,delimiter=',',names=True)
def compare(x,y):return abs(x-y)/max(abs(y),1e-300)
def main():
    names=['coarse','fine','selected','replay'];rows={n:read(n+'-history.csv') for n in names}
    sums={n:json.loads((OUT/(n+'-summary.json')).read_text()) for n in names}
    summaries=[]
    for n in names:
        r=sums[n];summaries.append(dict(case=n,dt_s=r.get('dt_s',float(rows[n]['dt_s'].max())),absolute_remainder_J_m=r['absolute_remainder_J_m'],signed_remainder_J_m=r['physical_remainder_J_m'],fracture_J_m=r['fracture_increment_J_m'],sliding_J_m=r['sliding_loss_J_m'],maxwell_J_m=r['maxwell_loss_J_m'],remainder_over_fracture=r['remainder_over_fracture'],remainder_over_sliding=r['remainder_over_sliding'],peak_damage=r['final']['peak_damage'],integrated_damage_m=r['final']['integrated_damage_m'],integrated_absolute_jump_m2=r['final']['integrated_absolute_jump_m2']))
    e.table('comparison.csv',summaries)
    segments=[]
    for n in names:
        r=rows[n]
        for label,mask in [('loading',r['time_s']<=.65+1e-12),('hold',r['time_s']>.65+1e-12)]:
            q=r[mask]
            d=dict(case=n,segment=label,start_s=float(q['time_s'][0]-q['dt_s'][0]),end_s=float(q['time_s'][-1]))
            for key in ['physical_remainder_J_m','trapezoid_work_J_m','endpoint_work_J_m','maxwell_loss_J_m','sliding_loss_J_m','fracture_increment_J_m','maxwell_remainder_J_m','surface_time_remainder_J_m']:
                d[key]=float(sum(q[key]))
            d['absolute_remainder_J_m']=float(sum(abs(q['physical_remainder_J_m'])))
            segments.append(d)
    e.table('loading-hold.csv',segments)
    full=rows['selected'];half=rows['replay'];suffix=full[full['time_s']>.58+1e-12]
    response=[]
    fields=['integrated_damage_m','peak_damage','integrated_absolute_jump_m2','peak_absolute_jump_m','bulk_J_m','bond_J_m']
    for t in [.585,.6,.625,.65,.66]:
        r=full[np.argmin(abs(full['time_s']-t))];q=half[np.argmin(abs(half['time_s']-t))]
        assert abs(r['time_s']-t)<1e-10 and abs(q['time_s']-t)<1e-10
        for field in fields:response.append(dict(time_s=t,quantity=field,selected=float(r[field]),half=float(q[field]),absolute_difference=float(abs(r[field]-q[field])),relative_difference=float(compare(r[field],q[field]))))
    e.table('response-comparison.csv',response)
    sliding=compare(sum(suffix['sliding_loss_J_m']),sum(half['sliding_loss_J_m']))
    profiles=[];a=json.loads((OUT/'selected-snapshots.json').read_text());b=json.loads((OUT/'replay-snapshots.json').read_text());k=e.a.Kernel('native-16-axial')
    profile_errors=[]
    for t in [.585,.6,.625,.65,.66]:
        s=a[f'{t:.6f}'];h=b[f'{t:.6f}']
        for key in ['damage','jump','theta']:
            v=np.array(s['histories'][key]);w=np.array(h['histories'][key])
            profile_errors.append(dict(time_s=t,quantity=key,weighted_L2_relative=float(np.sqrt(k.w@((v-w)**2)/max(k.w@(w*w),1e-300)))))
        for i,qp in enumerate(k.s):profiles.append(dict(time_s=t,x_m=float(qp['x_m']),y_m=float(qp['y_m']),weight_m=float(k.w[i]),selected_damage=s['histories']['damage'][i],half_damage=h['histories']['damage'][i],selected_jump_m=s['histories']['jump'][i],half_jump_m=h['histories']['jump'][i],selected_theta=s['histories']['theta'][i],half_theta=h['histories']['theta'][i]))
    e.table('profiles.csv',profiles);e.table('profile-errors.csv',profile_errors)
    gates=dict(full_raw_sliding=sums['selected']['remainder_over_sliding']<.01,full_raw_fracture=sums['selected']['remainder_over_fracture']<.01,replay_raw_sliding=sums['replay']['remainder_over_sliding']<.01,replay_raw_fracture=sums['replay']['remainder_over_fracture']<.01,sliding_difference=bool(sliding<.01),response_difference=bool(max(r['relative_difference'] for r in response if r['quantity'] in ['integrated_damage_m','integrated_absolute_jump_m2'])<.02))
    verification=dict(gates=gates,all_pass=all(gates.values()),sliding_window_relative_difference=float(sliding),selected_window_sliding_J_m=float(sum(suffix['sliding_loss_J_m'])),half_window_sliding_J_m=float(sum(half['sliding_loss_J_m'])),selected_window_absolute_remainder_J_m=float(sum(abs(suffix['physical_remainder_J_m']))),profile_errors=profile_errors)
    e.save('verification.json',verification)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'figure.dpi':150})
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors={'coarse':'#929292','fine':'#ce7b00','selected':'#0072b2','replay':'#009e73'}
    labels={'coarse':'Δt = 0.005 s','fine':'Δt = 0.0025 s','selected':'Δt = 1.953125e−5 s','replay':'Replay: Δt/2, from 0.58 s'}
    for n in names:
        r=rows[n];t=r['time_s'];c=colors[n];lab=labels[n]
        ax[0,0].plot(t,np.cumsum(abs(r['physical_remainder_J_m'])),label=lab,color=c)
        ax[0,1].plot(t,np.cumsum(r['physical_remainder_J_m']),label=lab,color=c)
    ax[0,0].set(yscale='log',ylabel='Cumulative absolute remainder [J/m]',title='Raw physical work error (no subtraction)')
    ax[0,1].set(ylabel='Signed cumulative remainder [J/m]',title='Signed error (compare with the absolute sum)')
    for key,lab in [('maxwell_remainder_J_m','Maxwell time term'),('surface_time_remainder_J_m','Surface time term'),('physical_remainder_J_m','Raw physical remainder')]:ax[1,0].plot(full['time_s'],np.cumsum(full[key]),label=lab)
    ax[1,0].set(ylabel='Selected run: cumulative terms [J/m]',title='Independent decomposition')
    analytic=read('analytic-maxwell.csv')
    for key,label in [('maxwell_error_J_m','Maxwell loss error'),('work_error_J_m','Trapezoidal work error'),('absolute_remainder_J_m','Absolute balance remainder')]:ax[1,1].loglog(analytic['dt_s'],abs(analytic[key]),'o-',label=label)
    ax[1,1].set(xlabel='Δt [s]',ylabel='Absolute error [J/m]',title='Independent analytic tied-Maxwell control')
    for x in ax.flat:x.legend(fontsize=8)
    for x in [ax[0,0],ax[0,1],ax[1,0]]:x.set_xlabel('Physical time [s]');x.axvline(.65,color='k',lw=.7,ls=':')
    fig.savefig(OUT/'energy.png');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for n in names:
        r=rows[n];t=r['time_s'];c=colors[n];lab=labels[n]
        ax[0,0].plot(t,r['integrated_damage_m'],color=c,label=lab)
        ax[0,1].plot(t,r['peak_damage'],color=c,label=lab)
        ax[1,0].plot(t,np.cumsum(r['sliding_loss_J_m']),color=c,label=lab)
    ax[0,0].set(ylabel='∫a ds [m]',xlim=(.58,.66),title='Prescribed-append damage')
    ax[0,1].set(ylabel='Peak surface damage a',xlim=(.58,.66),title='No fully decohered region')
    ax[1,0].set(ylabel='Sliding dissipation [J/m]',xlim=(.58,.66),title='Replay starts before any sliding')
    for field,label in [('integrated_damage_m','Integrated damage / slip'),('peak_damage','Peak damage')]:
        rr=[r for r in response if r['quantity']==field];ax[1,1].plot([r['time_s'] for r in rr],[100*r['relative_difference'] for r in rr],'o-',label=label)
    ax[1,1].set(ylabel='Selected vs half-step difference [%]',title='Common physical times; hold begins at 0.65 s')
    for x in ax.flat:x.set_xlabel('Physical time [s]');x.axvline(.65,color='k',ls=':',lw=.7);x.legend(fontsize=8)
    fig.savefig(OUT/'response.png');plt.close(fig)
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
