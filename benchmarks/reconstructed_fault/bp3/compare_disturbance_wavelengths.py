"""Compare the saved 200/600-m positive branches on their own Q1 pattern masses.

No trajectories are run. All norms use the native consistent work mass, and
each branch is compared with reference32 at exactly matching accepted times.
"""
import csv
import json
from pathlib import Path

import numpy as np
from analyze_disturbance import ROOT, table, quadratic

OUT = ROOT/'wavelength-comparison'
BRANCHES = {200: 'plus32', 600: 'plus60032'}
YEAR = 31557600.


def write_csv(name, rows):
    with (OUT/name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    histories, profiles, verified = [], [], {}
    initial_fields, field_cache = {}, {}
    reference_launch = json.loads((ROOT/'reference32/launch.json').read_text())
    for wavelength, branch in BRANCHES.items():
        launch = json.loads((ROOT/branch/'launch.json').read_text())
        execution = json.loads((ROOT/branch/'execution.json').read_text())
        assert execution['returncode'] == 0
        assert launch['steps'] == 32 and launch['epsilon'] == 1e-4
        assert launch.get('wavelength_m',200) == wavelength
        for key in ('start_s','final_s','steps','dt','checkpoint_source'):
            assert launch[key] == reference_launch[key], key
        accepted = table(ROOT/branch/'accepted_steps.csv')
        accepted = accepted[accepted['step'] > 11]
        assert np.array_equal(accepted['step'],np.arange(12,44))
        assert np.all(accepted['dt'] == launch['dt'])
        assert np.max(accepted['max_step_slip_over_Dc']) <= .125
        assert np.all(accepted['fresh_linear_checks_passed'] == 1)
        assert np.all(accepted['free'] == 1156) and np.all(accepted['lower_active'] == 0)
        assert np.max(accepted['Theta_relative_error']) < 1e-12
        initial = None
        previous = None
        for k, step in enumerate(range(12,44)):
            r = table(ROOT/'reference32'/f'disturbance_nodes_{step}.csv')
            p = table(ROOT/branch/f'disturbance_nodes_{step}.csv')
            assert np.array_equal(p['node'],r['node']) and np.array_equal(p['xd'],r['xd'])
            assert np.array_equal(p['time'],r['time']) and np.array_equal(p['dt'],r['dt'])
            assert p['time'][0] == accepted['time'][k]
            for key in ('mass_diagonal','mass_upper'):
                assert np.max(abs(p[key]-r[key]))/np.max(abs(r[key])) < 1e-12
            w = p['w']
            xd = p['xd']
            expected = np.where(abs(xd-16500) < 1500,
                np.cos(np.pi*(xd-16500)/3000)**2*np.cos(2*np.pi*(xd-16500)/wavelength),0.)
            assert np.max(abs(w-expected)) < 1e-13
            if previous is not None:
                assert np.array_equal(p['Theta_in'],previous)
            previous = p['Theta_out'].copy()
            one = np.ones(len(r))
            mass = quadratic(w,w,r)
            measure = quadratic(one,one,r)
            pattern_rms = np.sqrt(mass/measure)
            norm = lambda f: float(np.sqrt(max(0.,quadratic(f,f,r))/measure))
            projection = lambda f: float(quadratic(w,f,r)/mass)
            if initial is None:
                initial = p['Theta_in']-r['Theta_in']
                initial_reference = r['Theta_in'].copy()
                initial_relative = initial/initial_reference
                assert np.max(abs(np.log(p['Theta_in']/r['Theta_in'])-launch['epsilon']*w)) < 5e-16
                initial_fields[wavelength] = dict(w=w.copy(),xd=xd.copy(),absolute=initial.copy(),relative=initial_relative.copy())
                initial_norm = norm(initial)
                initial_relative_norm = norm(initial_relative)
                initial_projection = projection(initial)
            absolute = p['Theta_out']-r['Theta_out']
            relative = absolute/r['Theta_out']
            log_relative = np.log(p['Theta_out']/r['Theta_out'])
            velocity = p['V']-r['V']
            relative_velocity = velocity/r['V']
            if k == 0:
                first_v_norm = norm(velocity)
                first_relative_v_norm = norm(relative_velocity)
            row = dict(wavelength_m=wavelength,step=step,time_s=float(r['time'][0]),
                       elapsed_years=float((r['time'][0]-launch['start_s'])/YEAR),
                       mode_mass_m=float(mass),work_measure_m=float(measure),pattern_RMS=float(pattern_rms),
                       initial_absolute_state_RMS_s=initial_norm,initial_relative_state_RMS=initial_relative_norm,
                       absolute_state_RMS_s=norm(absolute),relative_state_RMS=norm(relative),log_state_RMS=norm(log_relative),
                       absolute_velocity_RMS_m_s=norm(velocity),relative_velocity_RMS=norm(relative_velocity),
                       absolute_state_gain=norm(absolute)/initial_norm,
                       relative_state_gain=norm(relative)/initial_relative_norm,
                       absolute_velocity_gain=norm(velocity)/first_v_norm,
                       relative_velocity_gain=norm(relative_velocity)/first_relative_v_norm,
                       relative_state_scaled_RMS=norm(relative)/(launch['epsilon']*pattern_rms),
                       absolute_velocity_scaled_RMS=norm(velocity)/(launch['epsilon']*1e-9*pattern_rms),
                       relative_velocity_scaled_RMS=norm(relative_velocity)/(launch['epsilon']*pattern_rms),
                       absolute_state_projection_s=projection(absolute),
                       absolute_state_projection_gain=projection(absolute)/initial_projection,
                       relative_state_projection_over_epsilon=projection(relative)/launch['epsilon'],
                       velocity_projection_over_epsilon_Vp=projection(velocity)/(launch['epsilon']*1e-9),
                       relative_velocity_projection_over_epsilon=projection(relative_velocity)/launch['epsilon'],
                       frozen_initial_denominator_gain=norm(absolute/initial_reference)/initial_relative_norm,
                       denominator_only_factor=norm(relative)/norm(absolute/initial_reference))
            histories.append(row)
            field_cache[(wavelength,step)] = dict(xd=xd.copy(),absolute=absolute.copy(),relative=relative.copy(),
                                                 velocity=velocity.copy(),relative_velocity=relative_velocity.copy())
            for j in np.flatnonzero((xd >= 14000)&(xd <= 19000)):
                profiles.append(dict(wavelength_m=wavelength,step=step,elapsed_years=row['elapsed_years'],
                                     node=int(p['node'][j]),xd_m=xd[j],pattern=w[j],
                                     initial_delta_Theta_s=initial[j],delta_Theta_s=absolute[j],
                                     relative_Theta=relative[j],delta_V_m_s=velocity[j],relative_V=relative_velocity[j],
                                     Theta_reference_s=r['Theta_out'][j],V_reference_m_s=r['V'][j]))
        log = (ROOT/branch/'run.log').read_text()
        audits = [line for line in log.splitlines() if line.startswith('Disturbance audit:')]
        assert len(audits) == 32
        assert all(float(line.split('incoming-state weak error=')[1].split()[0]) < 1e-5 for line in audits)
        verified[str(wavelength)] = dict(steps=32,free=1156,lower_active=0,
            max_step_slip_over_Dc=float(np.max(accepted['max_step_slip_over_Dc'])),
            max_theta_relative_error=float(np.max(accepted['Theta_relative_error'])),
            max_surface_RMS_Pa=float(np.max(accepted['surface_RMS_Pa'])),
            max_normalized_nonlinear_residual=float(np.max(accepted['normalized_nonlinear_residual'])),
            fresh_linear_checks=True,matching_clock=True,exact_incoming_state_chain=True,
            max_diagnostic_weak_error_Pa=max(float(line.split('incoming-state weak error=')[1].split()[0]) for line in audits),
            execution=execution)
    write_csv('norms.csv',histories)
    write_csv('profiles.csv',profiles)
    summary = dict(first={str(w):next(r for r in histories if r['wavelength_m']==w) for w in BRANCHES},
                   final={str(w):[r for r in histories if r['wavelength_m']==w][-1] for w in BRANCHES},
                   verification=verified)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(histories,initial_fields,field_cache)
    print(json.dumps(summary,indent=2))


def plots(rows, initial, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    keys=('absolute_state_gain','relative_state_gain','absolute_velocity_gain','relative_velocity_gain')
    titles=('Absolute state difference / initial norm','Relative state difference / initial norm',
            'Absolute velocity difference / first response','Relative velocity difference / first response')
    for wavelength in BRANCHES:
        data=[r for r in rows if r['wavelength_m']==wavelength]
        for ax,key,title in zip(axes.flat,keys,titles):
            t=[r['elapsed_years'] for r in data]; y=[r[key] for r in data]
            if key in keys[:2]:t=[0.]+t;y=[1.]+y
            ax.plot(t,y,label=f'{wavelength} m');ax.set_title(title);ax.grid(alpha=.3)
    for ax in axes.flat:ax.legend();ax.axhline(1,color='0.6',lw=.7)
    for ax in axes[-1]:ax.set_xlabel('Years after accepted step 11')
    fig.tight_layout();fig.savefig(OUT/'norm_gains.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    keys=('absolute','relative','velocity','relative_velocity')
    scales=(1.,1e-4,1e-13,1e-4)
    labels=('delta Theta (s)','(delta Theta / Theta_ref) / epsilon','delta V / (epsilon Vp)','(delta V / V_ref) / epsilon')
    for column,wavelength in enumerate(BRANCHES):
        for step,label in ((12,'first response'),(27,'0.5 yr'),(43,'1 yr')):
            f=fields[(wavelength,step)];order=np.argsort(f['xd']);x=f['xd'][order]/1000
            for ax,key,scale,title in zip(axes[:,column],keys,scales,labels):
                ax.plot(x,f[key][order]/scale,label=label);ax.set_ylabel(title);ax.grid(alpha=.3);ax.set_xlim(14.8,18.2)
        f=initial[wavelength];order=np.argsort(f['xd']);x=f['xd'][order]/1000
        for row,key,scale in ((0,'absolute',1.),(1,'relative',1e-4)):
            axes[row,column].plot(x,f[key][order]/scale,'k:',label='initial')
        axes[0,column].set_title(f'{wavelength}-m disturbance');axes[0,column].legend(fontsize=8)
        axes[-1,column].set_xlabel('Down dip (km)')
    fig.tight_layout();fig.savefig(OUT/'spatial_profiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True)
    for wavelength in BRANCHES:
        data=[r for r in rows if r['wavelength_m']==wavelength]
        for ax,key,title in zip(axes,('absolute_velocity_scaled_RMS','relative_state_projection_over_epsilon'),
                               ('V RMS / (epsilon Vp RMS(pattern))','Relative-state pattern projection / epsilon')):
            ax.plot([r['elapsed_years'] for r in data],[r[key] for r in data],label=f'{wavelength} m')
            ax.set_title(title);ax.grid(alpha=.3);ax.set_xlabel('Elapsed years');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'pattern_normalized.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
