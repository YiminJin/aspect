"""Reduce opt-in pre-publication stress diagnostics from a converged BP3 replay.

Requires the completed BP3 projected output and a genuine accepted-state log
message. Raw extrema and nodal projections are never interchanged.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, data):
    with path.open("x", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def down_dip(x,y):
    sine=np.sqrt(3.)/2
    return (.5e5*(1+.5/sine)-float(x))*.5+(1e5-float(y))*sine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--step",type=int,default=12)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--noncommitting",action="store_true",
                        help="Require converged diagnostic plus verified rollback; never infer acceptance.")
    parser.add_argument("--allow-unverified-rollback",action="store_true",
                        help="Analyze converged pre-rollback exports, explicitly recording incomplete rollback verification.")
    args=parser.parse_args()
    root=args.directory
    if args.allow_unverified_rollback and not args.noncommitting:
        raise SystemExit('Unverified rollback analysis is only valid for a noncommitting diagnostic.')
    import re
    log=(root/"log.txt").read_text()
    if args.noncommitting:
        if ('Noncommitting fault diagnostic converged:' not in log
            or ('BP3 noncommitting rollback verified:' not in log and not args.allow_unverified_rollback)):
            raise SystemExit('Noncommitting solve lacks convergence or verified rollback.')
    elif not re.search(r"BP3 accepted state\s+"+str(args.step)+r"\b",log):
        raise SystemExit("No fully accepted BP3 state: do not interpret a trial diagnostic as accepted.")
    ranks=int(re.search(r"running with (\d+) MPI process",log)[1])
    if args.noncommitting:
        projected=rows(root/'noncommitting_surface.csv')
        retained=rows(root/'noncommitting_history.csv') if (root/'noncommitting_history.csv').exists() else None
        n=len(projected)
        band=np.zeros((3,n));band[1]=[float(r['mass_diagonal']) for r in projected]
        band[0,1:]=band[2,:-1]=[float(r['mass_upper']) for r in projected[:-1]]
        q,sigma=solve_banded((1,1),band,np.array([[float(r['weak_q']),float(r['weak_sigma'])] for r in projected])).T
        for i,r in enumerate(projected):
            assert int(r['fault'])==0 and int(r['step'])==args.step
            if retained is not None: assert int(retained[i]['node'])==i
            r.update(q=q[i],sigma_n=sigma[i],xd=down_dip(r['x'],r['y']),
                     Theta_retained=retained[i]['Theta_retained'] if retained is not None else float('nan'))
    else:
        projected=rows(root/f"stress_projected_{args.step}.csv")
    n=len(projected)
    assert [int(r['node']) for r in projected]==list(range(n))
    moments=np.zeros((n,10))
    fields=['weight','p_load','tauN_load','bg_load','sigma_load','friction_load',
            'abs_friction_load','tensile_weight','tensile_friction_load','tensile_abs_friction_load']
    raw={}
    bulk={}
    for rank in range(ranks):
        for r in rows(root/f"stress_weak_moments_{args.step}_rank{rank}.csv"):
            assert int(r['fault'])==0
            i=int(r['node'])
            np.testing.assert_allclose(float(r['V']),float(projected[i]['V']),rtol=1e-13,atol=0.)
            moments[i]+=np.array([float(r[f]) for f in fields])
        for r in rows(root/f"stress_samples_{args.step}_rank{rank}.csv"):
            key=(r['rank'],r['fault'],r['particle'],r['domain_q'])
            r.pop('selection')
            r['xd']=down_dip(r['surface_x'],r['surface_y'])
            np.testing.assert_allclose(float(r['sigma_n']),float(r['sigma_bg'])+
                                       float(r['delta_p'])-float(r['delta_tau_N']),rtol=1e-13,atol=1e-7)
            raw[key]=r
        for r in rows(root/f"bulk_slip_transfer_{args.step}_rank{rank}.csv"):
            i=int(r['segment']); assert int(r['fault'])==0
            np.testing.assert_allclose([float(r['V0']),float(r['V1'])],
                [float(projected[i]['V']),float(projected[i+1]['V'])],rtol=1e-13,atol=0.)
            sums=['weight','V_integral','chi_integral','instantaneous_integral','history_integral',
                  'total_integral','stress_coefficient_integral','qp_count']
            b=bulk.setdefault(i,dict(segment=i,xd0=down_dip(r['x0'],r['y0']),
                                     xd1=down_dip(r['x1'],r['y1']),**{f:0. for f in sums}))
            for f in sums:b[f]+=float(r[f])
    band=np.zeros((3,n))
    band[1]=[float(r['mass_diagonal']) for r in projected]
    band[0,1:]=band[2,:-1]=[float(r['mass_upper']) for r in projected[:-1]]
    p,tau,bg,sigma,friction=solve_banded((1,1),band,moments[:,1:6]).T
    np.testing.assert_allclose(sigma,[float(r['sigma_n']) for r in projected],rtol=1e-10,atol=1e-6)
    profile=[]
    for i,r in enumerate(projected):
        state_name='Theta_retained' if args.noncommitting else 'Theta_committed'
        profile.append(dict(node=i,xd=float(r['xd']),V=float(r['V']),
          **({} if args.noncommitting else dict(slip=float(r['slip']))),
          prescribed=int(r['prescribed']),lower_active=int(r['lower_active']),
          delta_p=p[i],minus_delta_tau_N=-tau[i],sigma_bg=bg[i],
          sigma_n=sigma[i],q=float(r['q']),**{state_name:float(r[state_name])},
          friction_projected=friction[i],tensile_weight=moments[i,7],
          tensile_friction_weak_load=moments[i,8],
          marker_km=next((x for x in (15,18,40) if abs(float(r['xd'])-x*1000)<1e-7),'')))
    unprescribed=np.array([int(r['prescribed'])==0 for r in projected])
    lower_active=np.array([int(r['lower_active'])!=0 for r in projected])
    free=unprescribed & ~lower_active
    summary={'noncommitting':args.noncommitting,
             'rollback_verified':('BP3 noncommitting rollback verified:' in log) if args.noncommitting else None,
             'history_snapshot_available':(retained is not None) if args.noncommitting else True}
    for label,mask in [('all',np.ones(n,dtype=bool)),('unprescribed',unprescribed),
                       ('free',free),('lower_active',lower_active),('prescribed',~unprescribed)]:
        m=moments[mask]
        summary[label]=dict(weight=float(m[:,0].sum()),tensile_weight=float(m[:,7].sum()),
            tensile_friction_signed_load=float(m[:,8].sum()),
            tensile_absolute_friction_load=float(m[:,9].sum()),
            total_absolute_friction_load=float(m[:,6].sum()),
            tensile_to_total_weak_friction_l2=float(np.linalg.norm(m[:,8])/max(np.linalg.norm(m[:,5]),1e-300)))
    ordered=sorted(raw.values(),key=lambda r:float(r['sigma_n']))
    summary['minimum']=ordered[0];summary['maximum']=ordered[-1]
    summary['raw_selection']='20 minima/maxima per rank per free/prescribed/mixed support class; not all samples'
    summary['regions']={}
    for lo,hi in [(13000,20000),(37000,43000)]:
        selected=[r for r in profile if lo-1e-7<=r['xd']<=hi+1e-7]
        summary['regions'][f'{lo}-{hi}']={f:float(np.ptp([r[f] for r in selected]))
                                         for f in ['delta_p','minus_delta_tau_N','sigma_n']}
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'projected_full.csv',profile)
    write(args.output/'raw_extrema.csv',ordered[:20]+ordered[-20:])
    write(args.output/'raw_selected.csv',ordered)
    write(args.output/'bulk_transfer.csv',list(bulk.values()))
    with (args.output/'summary.json').open('x') as stream:
        json.dump(summary,stream,indent=2);stream.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
