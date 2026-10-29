#!/usr/bin/env python3
"""Independent physical-field check of the selected zero-increment hold."""
import argparse,json
import numpy as np
from continuation_analysis import ROOT, BASE, OUT, read


def check(name):
    info=json.load(open(BASE/(name+'-input.json')))
    parent=(ROOT/info['source']).parents[1]
    a=read(parent/f"means-{info['accepted_step']:04d}.csv");a.sort(order=['y','x'])
    p=read(parent/f"phase-{info['accepted_step']:04d}.csv");p.sort(order='dof')
    resource=read(BASE/'runs'/name/'resources.csv')
    rows=[];factor=1.
    for step in resource['step'].astype(int):
        factor/=1+info['dt']*.01
        b=read(BASE/'runs'/name/f'means-{step:04d}.csv');b.sort(order=['y','x'])
        q=read(BASE/'runs'/name/f'phase-{step:04d}.csv');q.sort(order='dof')
        error=max(float(np.max(abs(b[key]-factor*a[key]))) for key in ['tauxx','tauyy','tauxy'])
        pressure_error=float(np.max(abs(b['pressure']-factor*a['pressure'])))
        scale=max(1.,float(np.max(abs(a['tauxy']))))
        # The virgin multiplier includes the unscaled residual strength R;
        # its driving can decrease faster than beta squared. Do not force the
        # damaged-state q scaling onto those intact points.
        excess=float(np.max(b['Y']-factor**2*a['Y']))
        result=dict(step=int(step),factor=factor,maximum_deviatoric_stress_error_Pa=error,
                    maximum_pressure_error_Pa=pressure_error,
                    relative_error=error/scale,phase_bitwise_equal=bool(np.array_equal(q['phi'],p['phi'])),
                    maximum_abs_w_s_inv=float(b['max_rate'].max()),maximum_Y_above_beta_squared_parent_Pa=excess)
        assert result['phase_bitwise_equal'] and result['maximum_abs_w_s_inv']==0
        assert excess<1e-10
        if not rows:
            # Only the deliberately Maxwell-scaled first guess has this
            # roundoff-level identity. Later default pressure guesses are
            # reused until the unchanged mechanical residual triggers a solve.
            assert result['relative_error']<1e-12 and pressure_error/scale<1e-12
        rows.append(result)
    (OUT/(name+'-exact-relaxation.json')).write_text(json.dumps(dict(case=name,states=rows,
        interpretation='First-hold stress/pressure identity is checked at roundoff. Later discrepancies are measured against the analytic solution, not assigned the independent source-projection tolerance. Native mechanical acceptance remains unchanged.'),indent=2)+'\n')
    print(name,max(r['relative_error'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:check(name)
