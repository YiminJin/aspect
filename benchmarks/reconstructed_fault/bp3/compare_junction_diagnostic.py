"""Compare the one frozen-history junction probe with the accepted baseline."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_normal_stress import rows, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-unverified-rollback',action='store_true')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    base=here/'normal-stress-complete-local4'
    probe=here/'normal-stress-junction35-local4'
    result=json.loads((probe/'execution.json').read_text())
    assert all(result[k] for k in ['converged',
                                  'source_checkpoint_unchanged','diagnostic_checkpoint_unchanged'])
    assert result['rollback_verified'] or args.allow_unverified_rollback
    assert not result['accepted_output_written']
    original=rows(base/'analysis/step12/projected_full.csv')
    changed=rows(probe/'analysis/projected_full.csv')
    retained=rows(probe/'noncommitting_history.csv') if (probe/'noncommitting_history.csv').exists() else []
    before=rows(base/'stress_projected_11.csv')
    before_fault={(float(r['x']),float(r['y'])):r for r in rows(base/'fault_11.csv')}
    errors={k:0. for k in ['V_committed','Theta_retained','C_retained','Ih_retained','bg_shear','bg_normal']}
    for i,r in enumerate(retained):
        np.testing.assert_array_equal([float(r['x']),float(r['y'])],[float(before[i]['x']),float(before[i]['y'])])
        f=before_fault[(float(r['x']),float(r['y']))]
        expected=dict(V_committed=float(before[i]['V']),Theta_retained=float(before[i]['Theta_committed']),
                      C_retained=float(f['C']),Ih_retained=float(f['Ih']),bg_shear=float(f['tau_bg']),bg_normal=float(f['sigma_n_bg']))
        for key,value in expected.items():
            errors[key]=max(errors[key],abs(float(r[key])-value))
            np.testing.assert_allclose(float(r[key]),value,rtol=1e-13,atol=0.)
    dt_record=rows(base/'accepted_steps.csv')[-1]
    state=rows(probe/'noncommitting_surface.csv')
    np.testing.assert_equal(float(state[0]['time']),float(dt_record['time']))
    before_mass=rows(base/'stress_projected_12.csv')
    mass_error=max(abs(float(s[k])-float(b[k])) for s,b in zip(state,before_mass)
                   for k in ['mass_diagonal','mass_upper'])
    comparison=[]
    for a,b in zip(original,changed):
        assert a['node']==b['node']
        row=dict(node=a['node'],xd=float(a['xd']))
        for field in ['V','delta_p','minus_delta_tau_N','sigma_n','q','prescribed','lower_active']:
            row[field+'_40km']=float(a[field]);row[field+'_35km']=float(b[field])
            row[field+'_change']=float(b[field])-float(a[field])
        comparison.append(row)
    write(probe/'comparison.csv',comparison)
    report=dict(rollback_verified=result['rollback_verified'],
                retained_history_max_abs_errors=errors if retained else None,mass_max_absolute_difference=mass_error,
                time=float(state[0]['time']),regions={})
    for label,lo,hi in [('new_junction',34000,36000),('old_junction',39000,41000),('bottom',113000,116000)]:
        region=[r for r in comparison if lo<=r['xd']<=hi]
        report['regions'][label]={}
        for variant in ['40km','35km']:
            report['regions'][label][variant]={}
            for f in ['delta_p','minus_delta_tau_N','sigma_n','q']:
                y=np.array([r[f+'_'+variant] for r in region])
                report['regions'][label][variant][f]=dict(min=float(y.min()),max=float(y.max()),ptp=float(np.ptp(y)))
    fig,axes=plt.subplots(4,3,figsize=(14,12))
    for col,(lo,hi) in enumerate([(33,37),(38,42),(114.8,115.5)]):
        for j,(f,scale) in enumerate([('delta_p',1e6),('minus_delta_tau_N',1e6),('sigma_n',1e6),('V',1e-9)]):
            for variant,style in [('40km','-'),('35km','--')]:
                axes[j,col].plot([r['xd']/1000 for r in comparison],
                                 [r[f+'_'+variant]/scale for r in comparison],style,label=variant)
            axes[j,col].set_xlim(lo,hi);axes[j,col].grid(alpha=.25)
            axes[j,col].axvline(35 if col==0 else 40 if col==1 else 115.470054,color='grey',ls=':')
            axes[j,col].set_ylabel(f+(' / Vp' if f=='V' else ' [MPa]'))
            axes[j,col].legend(fontsize=8)
        axes[-1,col].set_xlabel('down-dip coordinate [km]')
    fig.suptitle('Same step-11 histories and step-12 dt: accepted 40-km baseline vs noncommitting 35-km probe')
    fig.tight_layout();fig.savefig(probe/'comparison.png',dpi=150)
    (probe/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
