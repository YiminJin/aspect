#!/usr/bin/env python3
"""Saved-data K2.2 spatial audit; no simulation, fitting or acceptance changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rms(values, weights):
    return float(np.sqrt(np.average(values**2, weights=weights)))


def main():
    base = Path(__file__).resolve().parent
    out = base/'refinement'
    directories = [out/'space32-measurements', base/'output-measurements',
                   out/'space128-measurements']
    fields = ('particle_q_Q1', 'C_evaluated_Q1', 'friction_Q1',
              'radiation_Q1', 'F_Q1')
    gauss, weights = np.polynomial.legendre.leggauss(8)
    read = lambda directory, k: np.genfromtxt(directory/f'surface_balance_{k}.csv',
                                            delimiter=',', names=True)
    report = dict(status='spatial traction plateau: temporal runs held for review',
                  initialization=[], response=[], adjacent_surface_differences=[])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for n, directory in zip((32, 64, 128), directories):
        data = read(directory, 0)
        s = data['s']
        x = ((s[:-1, None]+s[1:, None])/2 + np.diff(s)[:, None]*gauss/2).ravel()
        w = (np.diff(s)[:, None]*weights/2).ravel()
        theta = np.interp(x, s, data['Theta'])
        analytic = 200+10*np.where(abs(x-.125)<.0625,
                                   (1+np.cos(np.pi*(x-.125)/.0625))**2/4, 0)
        report['initialization'].append(dict(nx=n, theta_projection_rms_s=rms(theta-analytic,w),
                                             theta_min_s=float(min(data['Theta'])),
                                             theta_max_s=float(max(data['Theta']))))
        axes[0, 0].plot(s, data['Theta']-200, '.-', label=str(n))
        for k in range(5):
            data = read(directory, k)
            v = np.interp(x, s, data['V'])
            q = np.interp(x, s, data['particle_q_Q1'])
            report['response'].append(dict(nx=n, time_s=.5*k,
                V_min_m_s=float(min(data['V'])), V_max_m_s=float(max(data['V'])),
                V_anomaly_rms_m_s=rms(v-np.average(v,weights=w),w),
                q_anomaly_rms_Pa=rms(q-np.average(q,weights=w),w)))
        axes[0, 1].plot(s, data['V']-np.average(v,weights=w), '.-', label=str(n))
        axes[1, 0].plot(s, data['particle_q_Q1']-np.average(q,weights=w), '.-', label=str(n))
    for k in range(5):
        data = [read(directory, k) for directory in directories]
        for label, c, r in zip(('32-64', '64-128'), data, data[1:]):
            s = np.unique(np.r_[c['s'],r['s']])
            x = ((s[:-1,None]+s[1:,None])/2+np.diff(s)[:,None]*gauss/2).ravel()
            w = (np.diff(s)[:,None]*weights/2).ravel()
            interior = (x>.0625)&(x<.1875)
            row = dict(pair=label, time_s=.5*k, terms={})
            for field in fields:
                d = np.interp(x,c['s'],c[field])-np.interp(x,r['s'],r[field])
                mean = np.average(d,weights=w)
                d -= mean
                row['terms'][field] = dict(mean_Pa=float(mean), anomaly_rms_Pa=rms(d,w),
                    inside_bump_rms_Pa=rms(d[interior],w[interior]),
                    outside_bump_rms_Pa=rms(d[~interior],w[~interior]))
                if k==4 and field=='particle_q_Q1':
                    dv = np.interp(s,c['s'],c[field])-np.interp(s,r['s'],r[field])-mean
                    row['terms'][field]['max_error_location_m']=float(s[np.argmax(abs(dv))])
                    row['terms'][field]['max_error_Pa']=float(max(abs(dv)))
                    axes[1,1].plot(s,dv,'.-',label=label)
            report['adjacent_surface_differences'].append(row)
    for ax, title, ylabel in zip(axes.flat,
        ('Realized initial state', 'Slip-rate anomaly at 2 s',
         'Actual particle/Q1 traction anomaly at 2 s', 'Adjacent traction-anomaly differences at 2 s'),
        ('Theta - 200 (s)', 'V - mean(V) (m/s)', 'q - mean(q) (Pa)', 'difference (Pa)')):
        ax.set(title=title,xlabel='fault coordinate (m)',ylabel=ylabel)
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'spatial_surface_audit.png',dpi=160)
    (out/'spatial_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
