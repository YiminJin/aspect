#!/usr/bin/env python3
"""Plot only the accepted part of the stopped K1 convergence attempt."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
data = json.loads((base/'space16_dt05-errors.json').read_text())
rows = data['steps']
fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
for axis,field in zip(axes,['V','Theta','C']):
    axis.plot([r['time_s'] for r in rows],[r[field]['mean'] for r in rows],'o-',label='Accepted ASPECT')
    axis.plot([r['time_s'] for r in rows],[r[field]['reference'] for r in rows],'--',label='Own discrete reference')
    axis.set(xlabel='Time [s]',ylabel={'V':'V [m/s]','Theta':'Theta [s]','C':'C [Pa]'}[field])
axes[0].legend(fontsize=8)
fig.suptitle('Incomplete dt=0.5 s run: accepted through 4 s; failed at 4.5 s')
fig.savefig(base/'partial_histories.png',dpi=160); plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
for step in [0,8]:
    profile = np.genfromtxt(base/'space16_dt05-errors'/f'transverse_{step}.csv',names=True,delimiter=',')
    axes[0].plot(profile['y'],profile['ux_mean']-profile['reference_ux'],label=f't={rows[step]["time_s"]:g} s')
    axes[1].plot(profile['y'],profile['q_mean']-profile['reference_q'],label=f't={rows[step]["time_s"]:g} s')
    axes[1].fill_between(profile['y'],profile['q_min']-profile['reference_q'],profile['q_max']-profile['reference_q'],alpha=.2)
axes[0].set(xlabel='y [m]',ylabel='Velocity profile error [m/s]')
axes[1].set(xlabel='y [m]',ylabel='Raw stress error [Pa]')
for axis in axes:
    axis.legend(fontsize=8)
fig.suptitle('Native QP samples; stress is not smoothed')
fig.savefig(base/'partial_profiles.png',dpi=160); plt.close(fig)
