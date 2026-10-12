#!/usr/bin/env python3
"""Approach the bracketed release at fixed incoming history; no publication."""
import hashlib
import numpy as np
import asymmetric_branch as a


def main():
    rows=[]
    for n in [256,512]:
        base=a.search_path(n,1024);f=np.load(base/'incoming.npz')
        bracket=a.read(base/'release-bracket.json');p=a.model(n)
        for multiple in [0,5,20]:
            t=bracket['first_unstable_by']+multiple*bracket['width']
            p.incoming(f['phi'],f['R'],float(f['Z']),float(f['slip']),t,t-bracket['incoming_time'])
            x,kkt,_=p.solve();value,_=p.evaluate(x);spec,v=a.spectrum(p,x)
            assert kkt<=a.TARGET and spec['admissible_negative']
            for sign in [-1,1]:
                trial=x+sign*.5*spec['amplitude_two_sided']*v
                assert min(trial-p.lower)>=0
                y,status=a.minimize_phase(p,trial,sign)
                val,g=p.evaluate(y);mechanical=p.last[3].copy()
                qualified=bool(status['stable'] and status['KKT']<=a.TARGET
                     and mechanical['force']<1e-7 and mechanical['contact']<1e-7)
                allowance=a.A*32*np.finfo(float).eps*max(1.,abs(value))
                gain=float(a.A*(value-val))
                row=dict(n=n,time=t,sign=sign,bracket_widths_above_upper=multiple,
                    qualified=qualified,stability=status['stability'],
                    KKT=status['KKT'],curvature=status['stability']['minimum'],
                    centroid=a.centroid(p.r,y),phase_difference_from_symmetric=float(max(abs(y-x))),
                    phase_L2_difference=float(np.sqrt((y-x)@a.mass(p.r)@(y-x))),
                    total_increment=float(max(y-p.lower)),potential_reduction=gain,
                    arithmetic_allowance=float(allowance),energy_reduction_resolved=bool(gain>allowance),
                    mechanical=mechanical)
                rows.append(row);print(a.json.dumps(row),flush=True)
                if not qualified:
                    np.savez_compressed(a.E/f'release-unqualified-n{n}-m{multiple}-{sign:+d}.npz',
                         incoming_phase=p.lower,Ra=p.Ra,Za=p.Za,trial_phase=y,symmetric=x)
                a.write(a.E/'release-approach.json',dict(rows=rows,
                    restriction='Fixed original incoming R,Z,slip,obstacle; probes only, including unqualified trials'))
    a.write(a.E/'release-approach.json',dict(rows=rows,
        restriction='Fixed original incoming R,Z,slip,obstacle; fractional-step diagnostic probes only',
        source_sha256=hashlib.sha256((a.HERE/'probe_asymmetric_release.py').read_bytes()).hexdigest()))


if __name__=='__main__':main()
