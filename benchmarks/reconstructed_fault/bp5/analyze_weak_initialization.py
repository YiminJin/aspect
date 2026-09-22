"""Production-QP and native-row checks for the initialization-only experiment."""
import json
import numpy as np
import matplotlib.pyplot as plt
from analyze_clean_background import audit, table, OUT, mu, VP, SIGMA, DAMPING
from analyze_length_coupled import raw_qps
from check_first_cycle_restart import convergence
from run_weak_initialization import LABEL


def main():
    baseline,old,old_rows,old_fields=audit('clean-background-init')
    result,new,rows,fields=audit(LABEL, 'state_relative_to_physical_mixture_excess')
    directory=OUT/LABEL
    initial=table(directory/'weak_initialization.csv')
    native=table(directory/'work_weak_0.csv')
    np.testing.assert_array_equal(initial['Theta_weak'],new['Theta_in'])
    np.testing.assert_array_equal(new['Theta_in'],new['Theta_out'])
    np.testing.assert_array_equal(old['xd'],new['xd'])
    for key in ['composition_strengthening','previous_I_h','background_tractions']:
        np.testing.assert_array_equal(old_fields[key],fields[key])
    mass_error=float(max(abs(initial['weight']/native['weight']-1)))
    assert mass_error<1e-12, 'Initialization and production work measures differ'
    assert max(abs(initial['weak_friction_excess_Pa']))<1e-5
    assert min(initial['Theta_weak'])>0
    # Recompute both intermediate and final initialization at actual production
    # quadrature locations, independently of the C++ initialization row assembly.
    raw=np.concatenate([raw_qps(p) for p in sorted(directory.glob('work_qp_0_rank*.csv'))])
    raw=raw[(raw['source_active']==1)&(raw['chi']>0)]
    j=raw['segment'].astype(int);xi=raw['xi']
    def interp(x):return (1-xi)*x[j]+xi*x[j+1]
    f=np.clip(interp(fields['composition_strengthening']),0,1)
    target=float(table(directory/'friction_configuration.csv')['mu_target'][0])
    nodal_inverse_error=float(max(abs(mu(VP,initial['Theta_projected_inverse'],
                                         np.clip(fields['composition_strengthening'],0,1))-target))*SIGMA)
    assert nodal_inverse_error<1e-7
    mask=(new['xd']>=13000)&(new['xd']<=20000)
    qp_mask=(raw['xd']>=13000)&(raw['xd']<=20000)
    qp_results={}
    for name,field in [('projected_inverse','Theta_projected_inverse'),('weak','Theta_weak')]:
        excess=SIGMA*(mu(VP,interp(initial[field]),f)-target)
        load=np.zeros(len(new))
        for end,basis in [(0,1-xi),(1,xi)]:np.add.at(load,j+end,raw['JxW']*raw['chi']*basis*excess)
        scaled=load/native['weight']
        saved=initial['nodal_friction_excess_Pa' if name=='projected_inverse' else 'weak_friction_excess_Pa']
        assert max(abs(scaled[mask]-saved[mask]))<1e-6
        qp_results[name]=dict(weak_max_Pa=float(max(abs(scaled[mask]))),
            pointwise_range_Pa=[float(min(excess[qp_mask])),float(max(excess[qp_mask]))],
            reproduced_row_error_Pa=float(max(abs(scaled[mask]-saved[mask]))))
    accepted=table(directory/'accepted_steps.csv')
    assert list(accepted['step'])==[0]
    assert accepted['fresh_linear_checks_passed'][0]==1
    assert accepted['Theta_relative_error'][0]<1e-12
    assert accepted['max_committed_stress_Pa'][0]==0
    assert accepted['free'][0]==1156 and accepted['lower_active'][0]==0
    velocities={}
    for label,state in [('clean_original_state',old),('weak_state',new)]:
        e=state['V']/VP-1
        velocities[label]=dict(max_absolute=float(max(abs(e[mask]))),
            weighted_rms=float(np.sqrt(np.dot(native['weight'][mask],e[mask]**2)/sum(native['weight'][mask]))))
    output=dict(initialization=qp_results,nodal_inverse_error_Pa=nodal_inverse_error,
                production=result,baseline=baseline,
                mass_relative_error=mass_error,velocity=velocities,
                convergence=convergence(directory/'run.log'),
                execution=json.loads((directory/'execution.json').read_text()))
    (OUT/'weak_initialization_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(4,1,figsize=(10,12),sharex=True)
    order=np.argsort(new['xd'][mask]);x=new['xd'][mask][order]/1000
    for state,label in [(old,'Uniform background, old state'),(new,'Weak initial state')]:
        axes[0].plot(x,state['V'][mask][order]/VP,label=label)
    for field,label in [('Theta_original','Physical nodal inverse'),('Theta_projected_inverse','Projected-material nodal inverse'),('Theta_weak','Weak initial state')]:
        axes[1].plot(x,initial[field][mask][order],label=label)
    axes[1].set_yscale('log')
    for field in ['nodal_friction_excess_Pa','weak_friction_excess_Pa']:
        axes[2].plot(x,initial[field][mask][order]/1000,label=field)
    for data,label in [(old_rows,'Old state'),(rows,'Weak state')]:
        axes[3].plot(x,data['delta_shear'][mask][order]/1000,label=label+' bulk shear')
        axes[3].plot(x,-data['normal_feedback'][mask][order]/1000,'--',label=label+' normal feedback')
    for ax,y in zip(axes,['V/Vp','Theta [s]','Vinit weak friction excess [kPa]','Mechanical terms [kPa]']):
        ax.set_ylabel(y);ax.grid();ax.legend(fontsize=8)
        for s in [15,18]:ax.axvline(s,color='gray',linewidth=.7)
    axes[-1].set_xlabel('Down-dip distance [km]');fig.tight_layout()
    fig.savefig(OUT/'weak_initialization_comparison.png',dpi=180)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
