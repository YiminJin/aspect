"""Final matched-clock, amplitude, feedback and solver-noise comparisons."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from analyze_disturbance import ROOT, HERE, table, quadratic


def field(name,step):return table(ROOT/name/f'disturbance_nodes_{step}.csv')


def inner_norm(v,m):return np.sqrt(max(0.,quadratic(v,v,m)))


def odd(plus,minus,step,key):
    p,m=field(plus,step),field(minus,step)
    return .5*(np.log(p['Theta_out']/m['Theta_out']) if key=='logTheta' else p[key]-m[key])


def main():
    names={'plus16':'reference16-ready','minus16':'reference16-ready',
           'plus32':'reference32','minus32':'reference32',
           'state32':'reference32','normal32':'reference32','double4':'reference32'}
    for name,ref in names.items():
        execution=json.loads((ROOT/name/'execution.json').read_text());assert execution['returncode']==0
        launch=json.loads((ROOT/name/'launch.json').read_text());comparison=ROOT/name/'comparison.json'
        completed=json.loads(comparison.read_text()) if comparison.exists() else []
        reusable=(len(completed)==launch['steps'] and completed[-1]['step']==11+launch['steps']
                  and comparison.stat().st_mtime>=(HERE/'analyze_disturbance.py').stat().st_mtime)
        if not reusable:
            with (ROOT/name/'analysis.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'analyze_disturbance.py'),ref,name],stdout=log,stderr=subprocess.STDOUT,check=True)
    results={n:json.loads((ROOT/n/'comparison.json').read_text()) for n in names}
    temporal=[];symmetry=[];amplitude=[];background_temporal=[]
    for coarse_step in range(12,28):
        fine_step=11+2*(coarse_step-11);m=field('reference32',fine_step)
        row={'elapsed_years':(coarse_step-11)/16}
        for key in ('V','Theta_out','logTheta'):
            c=odd('plus16','minus16',coarse_step,key);f=odd('plus32','minus32',fine_step,key)
            row[key+'_relative_field_change']=float(inner_norm(c-f,m)/inner_norm(f,m))
            row[key+'_projection_relative_change']=float(quadratic(m['w'],c-f,m)/quadratic(m['w'],f,m))
        temporal.append(row)
        c=field('reference16-ready',coarse_step)
        background_temporal.append(dict(elapsed_years=row['elapsed_years'],**{
            key+'_relative_field_change':float(inner_norm(c[key]-m[key],m)/inner_norm(m[key],m))
            for key in ('V','Theta_out')}))
    for level in (16,32):
        for step in range(12,12+level):
            ref='reference16-ready' if level==16 else 'reference32'
            r,p,m=field(ref,step),field('plus'+str(level),step),field('minus'+str(level),step)
            row={'steps':level,'elapsed_years':(step-11)/level}
            for key in ('V','Theta_out','logTheta'):
                dp=np.log(p['Theta_out']/r['Theta_out']) if key=='logTheta' else p[key]-r[key]
                dm=np.log(m['Theta_out']/r['Theta_out']) if key=='logTheta' else m[key]-r[key]
                row[key+'_even_over_odd']=float(inner_norm(dp+dm,r)/inner_norm(dp-dm,r))
            symmetry.append(row)
    for step in range(12,16):
        r,p,d=field('reference32',step),field('plus32',step),field('double4',step)
        row={'elapsed_years':(step-11)/32}
        for key in ('V','Theta_out','logTheta'):
            dp=np.log(p['Theta_out']/r['Theta_out']) if key=='logTheta' else p[key]-r[key]
            dd=np.log(d['Theta_out']/r['Theta_out']) if key=='logTheta' else d[key]-r[key]
            row[key+'_relative_scaling_error']=float(inner_norm(dd/2-dp,r)/inner_norm(dp,r))
        amplitude.append(row)
    r,z,p=field('reference32',12),field('normal-zero1',12),field('plus32',12)
    noise={key:float(inner_norm(z[key]-r[key],r)/inner_norm(p[key]-r[key],r)) for key in ('V','Theta_out')}
    noise['weighted_V_RMS_m_s']=float(inner_norm(z['V']-r['V'],r)/inner_norm(np.ones(len(r)),r))
    c=field('state32',12)
    first_state_mechanics=float(inner_norm(c['V']-p['V'],r)/inner_norm(p['V']-r['V'],r))
    # With the same imposed reference rate in both aging maps, the absolute
    # state difference decays exactly by exp(-V_ref dt/Dc) at every node.
    # This independent identity checks the control's full history, not just
    # its final plot or the production aging audit.
    expected=r['Theta_in']*np.expm1(1e-4*r['w']);state_history=[]
    for step in range(12,44):
        r,c=field('reference32',step),field('state32',step)
        expected*=np.exp(-r['V']*r['dt']/.008)
        error=float(inner_norm(c['Theta_out']-r['Theta_out']-expected,r)/inner_norm(expected,r))
        assert error<1e-7
        state_history.append(dict(step=step,relative_difference_error=error))
    verification={}
    for name in ['reference16-ready','reference32',*names,'normal-zero1']:
        rows=table(ROOT/name/'accepted_steps.csv');rows=rows[rows['step']>11]
        launch=json.loads((ROOT/name/'launch.json').read_text())
        assert len(rows)==launch['steps'] and rows['step'][-1]==11+launch['steps']
        assert np.max(rows['max_step_slip_over_Dc'])<=launch['ratio_limit']
        assert np.all(rows['fresh_linear_checks_passed']==1)
        assert np.max(rows['Theta_relative_error'])<1e-12
        assert np.all(rows['free']==1156) and np.all(rows['lower_active']==0)
        verification[name]=dict(steps=len(rows),ratio_max=float(np.max(rows['max_step_slip_over_Dc'])),
                                surface_RMS_max_Pa=float(np.max(rows['surface_RMS_Pa'])),
                                nonlinear_residual_max=float(np.max(rows['normalized_nonlinear_residual'])),
                                theta_audit_max=float(np.max(rows['Theta_relative_error'])),
                                all_fresh_checks=bool(np.all(rows['fresh_linear_checks_passed']==1)),
                                all_nodes_free=bool(np.all(rows['free']==1156) and np.all(rows['lower_active']==0)),
                                execution=json.loads((ROOT/name/'execution.json').read_text()))
    summary=dict(temporal=temporal,background_temporal=background_temporal,
                 antisymmetry=symmetry,amplitude=amplitude,zero_normal_control_noise=noise,
                 state_control_exact_difference=state_history,
                 state_control_first_mechanics_relative_difference=first_state_mechanics,
                 final={n:rows[-1] for n,rows in results.items()},verification=verification)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
    keys=['V_projection_over_eps_Vp','V_norm_over_eps_Vp_pattern_norm','Theta_projection','Theta_norm',
          'logTheta_projection_gain','logTheta_norm_gain']
    labels=['200-m projection delta V / (epsilon Vp)','Total V RMS / (epsilon Vp RMS(w))',
            '200-m projection delta Theta (s)','Total delta Theta RMS (s)',
            '200-m log-state projection / epsilon','Log-state RMS / initial RMS']
    colors={'plus16':'tab:blue','plus32':'tab:blue','state32':'tab:green','normal32':'tab:red'}
    for name,style,label in [('plus16','--','Full, 16 steps'),('plus32','-','Full, 32 steps'),
                              ('state32','-','Reference V in state update'),('normal32','-','Reference normal in friction')]:
        rows=results[name];t=[r['elapsed_years'] for r in rows]
        for ax,key,ylabel in zip(axes.flat,keys,labels):
            ax.plot(t,[r[key] for r in rows],style,color=colors[name],label=label);ax.set_ylabel(ylabel);ax.grid(alpha=.25)
    for ax in axes[-1]:ax.set_xlabel('Years after accepted step 11')
    initial=json.loads((ROOT/'plus32/initial_disturbance.json').read_text())
    for ax,value in [(axes[1,0],initial['Theta_projection']),(axes[1,1],initial['Theta_norm']),
                     (axes[2,0],1.),(axes[2,1],1.)]:
        ax.axhline(value,color='0.4',linestyle=':',linewidth=1)
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'disturbance_growth.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for name,label in [('plus32','Full'),('state32','Reference V in aging'),('normal32','Reference normal in friction')]:
        r,p=field('reference32',43),field(name,43);order=np.argsort(r['xd']);x=r['xd'][order]/1000
        for ax,y in zip(axes,[(p['V']-r['V'])/1e-13,(p['Theta_out']-r['Theta_out']),np.log(p['Theta_out']/r['Theta_out'])/1e-4]):
            ax.plot(x,y[order],color=colors[name],label=label);ax.grid(alpha=.25);ax.set_xlim(14.5,18.5)
    axes[0].set_ylabel('delta V / (epsilon Vp)');axes[1].set_ylabel('delta Theta (s)');axes[2].set_ylabel('delta log(Theta) / epsilon')
    axes[2].plot(x,r['w'][order],'k:',label='Initial relative-state pattern')
    axes[0].legend();axes[2].legend();axes[2].set_xlabel('Down dip (km)');fig.tight_layout();fig.savefig(ROOT/'disturbance_final_profiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey='row')
    for column,name in enumerate(['plus32','state32','normal32']):
        rows=results[name]
        for ax,keys in zip(axes[:,column],[('state_friction','rate_friction'),('shear','normal_friction')]):
            for key in keys:
                ax.plot([r['elapsed_years'] for r in rows],[r[key+'_projection_Pa'] for r in rows],label=key)
            ax.grid(alpha=.25)
        axes[1,column].plot([r['elapsed_years'] for r in rows],
                            [r['state_friction_projection_Pa']+r['rate_friction_projection_Pa'] for r in rows],
                            '--',label='state + rate friction')
        axes[0,column].set_title(name);axes[1,column].set_xlabel('Elapsed years')
    for row in axes:
        row[0].set_ylabel('Signed 200-m force projection (Pa)');row[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'disturbance_force_budget.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('antisymmetry','verification')},indent=2))


if __name__=='__main__':main()
