"""Matched timestep-zero native-work comparison; no smoothing or time evolution."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_normal_control import BASE, OUT
from analyze_loading_startup import table, raw_qps
from analyze_loading_tractions import chord, metric
from check_first_cycle_restart import convergence
from startup_30km import parameters


def read(path):
    state=table(path/'state_work_0.csv'); native=table(path/'work_weak_0.csv')
    profile=table(path/'profiles/fault_0.csv')
    raw=np.concatenate([raw_qps(p) for p in sorted(path.glob('work_qp_0_rank*.csv'))])
    raw=np.sort(raw,order=['cell','qp'])
    w=native['weight']
    data=dict(xd=state['xd'],weight=w,V=state['V'],Theta=state['Theta_in'],
              q=(native['q']-native['bg'])/w,q_total=native['q']/w,
              P=native['p']/w,D=-native['tauN']/w,
              mechanical_normal=(native['p']-native['tauN'])/w,
              mechanical_total_normal=native['sigma']/w,
              friction_total_normal=state['weak_sigma']/w,
              q_Q1=profile['q_weak_Pa'],friction_normal_Q1=profile['sigma_n_weak_Pa'])
    for name in ('q','q_total','V','P','D','mechanical_normal'):
        data['chord_'+name]=chord(data[name])
    assert np.all(state['slip']==0)
    np.testing.assert_array_equal(state['Theta_in'],state['Theta_out'])
    clock=table(path/'accepted_steps.csv')
    row=clock[clock['step']==0][0]
    assert row['max_committed_stress_Pa']==0 and row['Theta_relative_error']==0
    assert row['fresh_linear_checks_passed']==1 and row['lower_active']==0
    # Independent QP check with incoming Q1 state and the configured pressure
    # mode. Keep the prescribed friction load separate from actual bulk stress.
    init=table(BASE/'steady_initialization.csv')
    r=raw[(raw['source_active']==1)&(raw['chi']>0)]
    j=r['segment'].astype(int);xi=r['xi']
    f=np.clip((1-xi)*init['projected_chemical'][j]+xi*init['projected_chemical'][j+1],0,1)
    aq=.004*(1-f)+.04*f
    theta=(1-xi)*state['Theta_in'][j]+xi*state['Theta_in'][j+1]
    mu=aq*np.arcsinh(r['V']/(2e-6)*np.exp((.6+.03*np.log(1e-6*theta/.1))/aq))
    prm=parameters((path/'parameters.prm').read_text())
    prescribed=prm['Material model','Phase field fault','Use adiabatic pressure in fault friction']=='true'
    sigma=np.full(len(r),5e7) if prescribed else r['sigma_n']
    load=np.zeros(len(state));mass=np.zeros(len(state))
    for end,shape in ((0,1-xi),(1,xi)):
        weight=r['JxW']*r['chi']*shape
        load+=np.bincount(j+end,weights=weight*sigma*mu,minlength=len(state))
        mass+=np.bincount(j+end,weights=weight,minlength=len(state))
    window=(state['xd']>=27000)&(state['xd']<=36000)
    np.testing.assert_allclose(mass[window],w[window],rtol=1e-12,atol=0)
    friction_error=float(max(abs(load[window]-state['weak_friction'][window])/w[window]))
    assert friction_error<1e-5
    tangent=np.array([profile['x_m'][1]-profile['x_m'][0],profile['y_m'][1]-profile['y_m'][0]])
    tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
    S=.5*(np.outer(tangent,normal)+np.outer(normal,tangent))
    kappa=-1e26*np.expm1(-1e6*32038120320/1e26)
    history_error=max(float(max(abs(r['tau_'+name]-2*kappa*(r['eps_'+name]-r['chi']*r['V']*S[a,b]))))
                      for name,a,b in [('xx',0,0),('yy',1,1),('xy',0,1)])
    assert history_error<1e-6
    return data,raw,dict(convergence=convergence(path/'run.log')[0],
                         accepted={n:float(row[n]) for n in row.dtype.names},
                         QP_weak_friction_error_Pa=friction_error,
                         recovered_incoming_history_max_Pa=history_error)


def main():
    control=OUT/'initial'; dest=OUT/'comparison';dest.mkdir(exist_ok=True)
    assert json.loads((control/'execution.json').read_text())['passed']
    a,ra,ia=read(BASE); b,rb,ib=read(control)
    assert len(table(control/'accepted_steps.csv'))==1
    assert len(list(control.glob('state_work_*.csv')))==1
    for baseline_mesh in BASE.glob('initial_mesh_*.csv'):
        assert baseline_mesh.read_bytes()==(control/baseline_mesh.name).read_bytes()
    for key in ('cell','qp','x','y','source_active','segment','xi','phi'):
        np.testing.assert_array_equal(ra[key],rb[key])
    agreement={}
    agreement['owned_initial_mesh_exports_byte_identical']=True
    for key in ('Ih','chi','JxW'):
        np.testing.assert_allclose(ra[key],rb[key],rtol=1e-12,atol=0)
        agreement[key+'_max_abs_difference']=float(max(abs(rb[key]-ra[key])))
    for key in ('xd','Theta'):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(a['weight'],b['weight'],rtol=1e-13,atol=0)
    assert (control/'steady_initialization.csv').read_bytes()==(BASE/'steady_initialization.csv').read_bytes()
    np.testing.assert_allclose(b['friction_total_normal'],5e7,rtol=0,atol=1e-5)
    for label,data in [('feedback',a),('prescribed50MPa',b)]:
        np.savetxt(dest/f'{label}_native.csv',np.column_stack(list(data.values())),
                   header=','.join(data),delimiter=',',comments='')
    for label,raw in [('feedback',ra),('prescribed50MPa',rb)]:
        active=(raw['source_active']==1)&(raw['chi']>0)
        r=raw[active]
        mechanical=r['p']-r['tauN']
        friction=5e7+mechanical if label=='feedback' else np.full(len(r),5e7)
        cols=[r[n] for n in ('x','y','xd','r','JxW','chi','p','tauN')]
        cols += [mechanical,5e7+mechanical,friction,r['q'],r['V']]
        np.savetxt(dest/f'{label}_raw.csv',np.column_stack(cols),delimiter=',',comments='',
                   header='x,y,xd,r,JxW,chi,p,tau_nn,mechanical_p_minus_tau_nn,mechanical_total_sigma_n,friction_total_sigma_n,q,V')
    x=a['xd']; windows={'primary_27_29.8km':(27000,29800),'transition_30.2_32.8km':(30200,32800)}
    report=dict(reference=ia,control=ib,matched_input_checks=agreement,windows={})
    for name,(lo,hi) in windows.items():
        mask=(x>=lo)&(x<=hi);w=a['weight'][mask]; results={}
        for field in ('q','q_total','V','P','D','mechanical_normal'):
            ca=a['chord_'+field][mask];cb=b['chord_'+field][mask]
            results[field]=dict(feedback=metric(ca,w),prescribed=metric(cb,w),
                ratio=metric(cb,w)['RMS']/metric(ca,w)['RMS'],
                correlation=float(np.corrcoef(ca,cb)[0,1]),
                feedback_peak_xd_m=float(x[mask][np.argmax(abs(ca))]),
                prescribed_peak_xd_m=float(x[mask][np.argmax(abs(cb))]),
                opposite_chord_signs=int(sum(ca*cb<0)),
                matched_difference=metric((b[field]-a[field])[mask],w))
        results['nodes']=int(sum(mask))
        results['mechanical_normal_range_Pa']={label:[float(min(d['mechanical_normal'][mask])),
                float(max(d['mechanical_normal'][mask]))] for label,d in [('feedback',a),('prescribed',b)]}
        report['windows'][name]=results
    report['execution']=json.loads((control/'execution.json').read_text())
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    colors=['#2455a4','#c53e20']
    # Identical axes for both cases, and the original neighboring-chord window.
    fig,axs=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for label,d,color in zip(('Normal feedback','Friction normal = 50 MPa'),(a,b),colors):
        mask=(d['xd']>=27000)&(d['xd']<=29800);xx=d['xd'][mask]/1000
        for row,key,scale,ylabel in [(0,'q',1,'Native perturbation shear [Pa]'),
                                    (1,'V',1e9,'V / Vp'),
                                    (2,'mechanical_normal',1,'Mechanical p - tau_nn [Pa]')]:
            axs[row,0].plot(xx,d[key][mask]*scale,label=label,color=color,lw=1)
            axs[row,1].plot(xx,d['chord_'+key][mask]*scale,label=label,color=color,lw=1,marker='.',ms=3)
            axs[row,0].set_ylabel(ylabel);axs[row,1].set_ylabel('Neighbor-chord departure')
    for ax in axs.flat:ax.set_xlim(27,29.8);ax.grid(alpha=.25)
    axs[0,0].legend();axs[2,0].set_xlabel('Down-dip distance [km]');axs[2,1].set_xlabel('Down-dip distance [km]')
    fig.suptitle('Initialization only: identical history, background, geometry and work weights')
    fig.tight_layout();fig.savefig(dest/'initial-teeth.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey='row')
    for col,(label,d) in enumerate(zip(('Normal feedback','Friction normal = 50 MPa'),(a,b))):
        mask=(d['xd']>=27000)&(d['xd']<=29800);xx=d['xd'][mask]/1000
        for field,text in [('P','p'),('D','-tau_nn'),('mechanical_normal','p - tau_nn')]:
            axs[0,col].plot(xx,d[field][mask],label=text,lw=1)
            axs[1,col].plot(xx,d['chord_'+field][mask],label=text,lw=1)
        axs[0,col].plot(xx,d['friction_total_normal'][mask]-5e7,ls='--',color='black',label='Friction normal - 50 MPa')
        axs[0,col].set_title(label);axs[0,col].legend()
    for ax in axs.flat:ax.set_xlim(27,29.8);ax.grid(alpha=.25)
    axs[0,0].set_ylabel('Native weak contribution [Pa]');axs[1,0].set_ylabel('Neighbor-chord departure [Pa]')
    fig.tight_layout();fig.savefig(dest/'normal-decomposition.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
