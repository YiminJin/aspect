"""Read-only quadrature reconstruction of the initialization friction budget.

Uses exported owned production QPs, their chi*JxW weights, and the actual
projected Q1 composition/state. Does not replace them with nodal friction.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clean_background import OUT, LABEL
from run_short import parameters
from analyze_length_coupled import raw_qps
from check_first_cycle_restart import convergence

VP, DC, SIGMA, V0 = 1e-9, .1, 50e6, 1e-6
DAMPING = 2670.*3464./2


def table(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def mu(v, theta, fraction):
    a = .004+.036*fraction
    return a*np.arcsinh(v/(2*V0)*np.exp((.6+.03*np.log(theta*V0/DC))/a))


def audit(label, state_excess_name='state_interpolation_excess'):
    directory = OUT/label
    state = table(directory/'state_work_0.csv')
    weak = table(directory/'work_weak_0.csv')
    arrays = {a.attrib['Name']: np.fromstring(a.text, sep=' ') for a in
              ET.parse(directory/'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
    np.testing.assert_array_equal(arrays['slip_state'], state['Theta_in'])
    np.testing.assert_array_equal(state['Theta_in'], state['Theta_out'])
    raw = np.concatenate([raw_qps(p) for p in sorted(directory.glob('work_qp_0_rank*.csv'))])
    raw = raw[(raw['source_active'] == 1) & (raw['chi'] > 0)]
    assert len(set(zip(raw['cell'],raw['qp']))) == len(raw)
    j = raw['segment'].astype(int)
    xi = raw['xi']
    def interp(values):
        return (1-xi)*values[j]+xi*values[j+1]
    theta = interp(state['Theta_in'])
    # MaterialUtilities::compute_composition_fractions clips the one chemical
    # fraction before creating its complementary background fraction.
    fraction = np.clip(interp(arrays['composition_strengthening']), 0, 1)
    v = interp(state['V'])
    np.testing.assert_allclose(v, raw['V'], rtol=2e-15, atol=0)
    s = interp(state['xd'])
    physical_fraction = np.clip((s-15000)/3000, 0, 1)
    nominal = float(table(directory/'friction_configuration.csv')['nominal_q_Pa'][0])
    target_mu = (nominal-DAMPING*VP)/SIGMA
    accepted_mu = mu(v, theta, fraction)
    intended_mu = mu(VP, theta, fraction)
    physical_mu = mu(VP, theta, physical_fraction)
    # Diagnostic inverse at each actual QP mixture, never projected or committed.
    a = .004+.036*fraction
    z = target_mu/a
    theta_equilibrated = DC/V0*np.exp((a*(np.log(V0/VP)+z+np.log(-np.expm1(-2*z)))-.6)/.03)
    inverse_error = float(np.max(abs(mu(VP,theta_equilibrated,fraction)-target_mu))*SIGMA)
    assert inverse_error < 1e-7
    weight = raw['JxW']*raw['chi']
    config = parameters((directory/'run.prm').read_text())
    prestress = np.loadtxt(config['Postprocess','BP3','Mature prestress file'], skiprows=1)
    bg = interp(prestress[:,2])-interp(prestress[:,4])-interp(prestress[:,5])/interp(prestress[:,6])
    # Full Vinit residual at the *accepted* bulk iterate is a distinct diagnostic:
    # changing V also changes current shear by -kappa*chi*(Vinit-V), since 2 S:S=1.
    kappa = -1e26*np.expm1(-4e6*32038120320/1e26)
    frozen_bulk_q_at_Vinit = raw['q']-kappa*raw['chi']*(VP-v)
    data = dict(weight=np.ones(len(raw)), accepted_friction=accepted_mu*raw['sigma_n'],
                accepted_shear=raw['q'], accepted_damping=DAMPING*v,
                accepted_residual=raw['q']-accepted_mu*raw['sigma_n']-DAMPING*v,
                background=bg, delta_shear=raw['q']-bg,
                normal_feedback=accepted_mu*(raw['sigma_n']-SIGMA),
                intended_friction=intended_mu*SIGMA,
                intended_residual=nominal-intended_mu*SIGMA-DAMPING*VP,
                fixed_bulk_residual_at_Vinit=frozen_bulk_q_at_Vinit-intended_mu*raw['sigma_n']-DAMPING*VP,
                composition_projection_excess=(intended_mu-physical_mu)*SIGMA)
    data[state_excess_name]=(physical_mu-target_mu)*SIGMA
    accumulated = {k: np.zeros(len(state)) for k in data}
    for end,basis in [(0,1-xi),(1,xi)]:
        for k,values in data.items():
            np.add.at(accumulated[k], j+end, weight*basis*values)
    # All source QPs are exported over 4--28 km. Restrict reconstruction checks
    # to rows whose full two-element support lies strictly inside that window.
    covered = (state['xd'] >= 5000) & (state['xd'] <= 27000)
    errors = {}
    for computed,saved in [('weight','weight'),('accepted_shear','q')]:
        errors[computed] = float(np.max(abs(accumulated[computed][covered]-weak[saved][covered])/weak['weight'][covered]))
    errors['accepted_friction'] = float(np.max(abs(accumulated['accepted_friction'][covered]-state['weak_friction'][covered])/weak['weight'][covered]))
    errors['accepted_residual'] = float(np.max(abs(accumulated['accepted_residual'][covered]-state['weak_residual'][covered])/weak['weight'][covered]))
    assert max(errors.values()) < 1e-5, errors
    rows = {k: np.divide(v, weak['weight']) for k,v in accumulated.items()}
    mask = (state['xd'] >= 13000) & (state['xd'] <= 20000)
    def extrema(values, selection=mask):
        ix = np.flatnonzero(selection)
        lo,hi = ix[np.argmin(values[ix])],ix[np.argmax(values[ix])]
        return dict(min=float(values[lo]), min_xd=float(state['xd'][lo]),
                    max=float(values[hi]), max_xd=float(state['xd'][hi]))
    result = dict(reconstruction_error_Pa=errors, qp_inverse_error_Pa=inverse_error,
                  V_over_Vp=extrema(state['V']/VP),
                  native_terms_Pa={k:extrema(v) for k,v in rows.items() if k!='weight'},
                  qp_intended_friction_error_Pa=[float(np.min((intended_mu-target_mu)[(s>=13000)&(s<=20000)]*SIGMA)),
                                                 float(np.max((intended_mu-target_mu)[(s>=13000)&(s<=20000)]*SIGMA))])
    keys = ['xd','V_over_Vp','Theta']+list(rows)
    output = np.column_stack([state['xd'],state['V']/VP,state['Theta_in']]+[rows[k] for k in rows])
    np.savetxt(directory/'initial_friction_weak_audit.csv',output[covered],delimiter=',',header=','.join(keys),comments='')
    qp_mask = (s>=13000)&(s<=20000)
    qp = np.column_stack([raw['x'],raw['y'],s,j,xi,weight,theta,fraction,physical_fraction,v,
                          accepted_mu,intended_mu,raw['sigma_n'],bg,
                          (intended_mu-target_mu)*SIGMA,theta_equilibrated])
    np.savetxt(directory/'initial_friction_qp_audit.csv',qp[qp_mask],delimiter=',',
        header='x,y,xd,segment,xi,JxW_chi,Theta_Q1,strengthening_Q1,strengthening_physical,V,mu_accepted,mu_at_Vinit,sigma_n,background,friction_excess_at_Vinit_Pa,Theta_equilibrated_QP_diagnostic',comments='')
    return result,state,rows,arrays


def main():
    old,so,ro,ao = audit('candidate-six')
    new,sn,rn,an = audit(LABEL)
    for field in ['slip_state','composition_strengthening','previous_I_h']:
        np.testing.assert_array_equal(ao[field],an[field])
    np.testing.assert_array_equal(so['xd'],sn['xd'])
    np.testing.assert_allclose(ro['weight'],rn['weight'],rtol=1e-14,atol=1e-14)
    accepted=table(OUT/LABEL/'accepted_steps.csv')
    assert list(accepted['step'])==[0] and accepted['fresh_linear_checks_passed'][0]==1
    assert accepted['max_committed_stress_Pa'][0]==0 and accepted['Theta_relative_error'][0]<1e-12
    conv=convergence(OUT/LABEL/'run.log')
    result=dict(baseline=old,clean=new,convergence=conv,unchanged_state_composition_Ih_geometry=True,
                execution=json.loads((OUT/LABEL/'execution.json').read_text()))
    (OUT/'clean_background_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(4,1,figsize=(10,12),sharex=True)
    for state,rows,label in [(so,ro,'Inherited background'),(sn,rn,'Uniform effective background')]:
        order=np.argsort(state['xd']);order=order[(state['xd'][order]>=13000)&(state['xd'][order]<=20000)]
        x=state['xd'][order]/1000
        axes[0].plot(x,state['V'][order]/VP,label=label)
        axes[1].plot(x,(rows['background'][order]-26546122.365139291)/1000,label=label)
        axes[2].plot(x,rows['delta_shear'][order],label=label+' shear')
        axes[2].plot(x,-rows['normal_feedback'][order],linestyle='--',label=label+' normal contribution')
    order=np.argsort(sn['xd']);order=order[(sn['xd'][order]>=13000)&(sn['xd'][order]<=20000)]
    x=sn['xd'][order]/1000
    for k in ['state_interpolation_excess','composition_projection_excess','intended_residual']:
        axes[3].plot(x,rn[k][order]/1000,label=k.replace('_',' '))
    for ax,ylabel in zip(axes,['V / Vp','Background - nominal [kPa]','Mechanical traction terms [Pa]','Vinit friction audit [kPa]']):
        ax.set_ylabel(ylabel);ax.set_xlim(13,20);ax.grid();ax.legend(fontsize=8)
        for xd in [15,18]:ax.axvline(xd,color='gray',linewidth=.7)
    axes[-1].set_xlabel('Down-dip distance [km]')
    fig.tight_layout();fig.savefig(OUT/'clean_background_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
