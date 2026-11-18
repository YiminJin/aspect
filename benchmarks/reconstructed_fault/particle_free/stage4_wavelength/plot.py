#!/usr/bin/env python3
"""Compact spatial-decision figures, from completed analyses only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-wavelength-mpl')
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'evidence/stage4_wavelength'
records=[json.loads(f.read_text()) for f in sorted(OUT.glob('spectrum-*.json'))]
assert records
plt.rcParams.update({'font.size':9})
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
for color,record in enumerate(records):
 nx=record['nx'];hx=8/nx;rows=record['rows'];c=f'C{color}'
 for law,style in [('weakening','-'),('matched_b_a','--'),('original_Dc',':')]:
  pairs=sorted([r for r in rows if r['law']==law and r['space'].startswith('pair')],key=lambda r:r['shortest_trial_wavelength'])
  axs[0].plot([r['shortest_trial_wavelength'] for r in pairs],[r['growth'] for r in pairs],style+'o',color=c,ms=4,label=f'{nx}: {law}')
 singles=[r for r in rows if r['law']=='weakening' and r['space'].startswith('single_') and r['shortest_trial_wavelength']<.3]
 axs[0].scatter([r['shortest_trial_wavelength'] for r in singles],[r['growth'] for r in singles],marker='x',s=18,color=c,alpha=.5)
 pairs=sorted([r for r in rows if r['law']=='weakening' and r['space'].startswith('pair')],key=lambda r:r['shortest_trial_wavelength'])
 axs[1].plot([r['shortest_trial_wavelength'] for r in pairs],[r['minimum_stiffness']/1e6 for r in pairs],'o-',color=c,label=str(nx))
 cumulative=[r for r in rows if r['law']=='weakening' and r['space'].startswith('coupled_through')]
 axs[2].plot([r['shortest_trial_wavelength'] for r in cumulative],[r['growth'] for r in cumulative],'o-',color=c,label=f'{nx} columns')
 for ax in axs:ax.axvline(2*hx,color=c,lw=.8,ls=':',alpha=.7)
for ax in axs:
 ax.set_xscale('log');ax.set_xlabel('Physical wavelength (m)');ax.grid(alpha=.2)
axs[0].set(title='Pair-space spectra; x = single patterns',ylabel='Largest real growth (1/s)')
axs[0].legend(fontsize=6,ncol=1)
axs[1].set(title='Sine/cosine restoring space',ylabel='Minimum effective stiffness (MPa/m)');axs[1].legend(title='Columns')
axs[2].set(title='Coupled six smooth + all longer pairs',ylabel='Largest real growth (1/s)',xlabel='Shortest included trial wavelength (m)');axs[2].legend(fontsize=8)
fig.suptitle('Frozen .075 specimen; vertical dotted lines = two cells / wavelength',fontsize=12)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'wavelength-spectrum.{ext}',dpi=180)
fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=False)
for color,record in enumerate(records):
 nx=record['nx'];rows=[r for r in record['rows'] if r['law']=='weakening' and r['space'].startswith('coupled_through')]
 r=min(rows,key=lambda r:r['shortest_trial_wavelength']);d=np.genfromtxt(OUT/f"mode-{nx}-{r['space']}.csv",delimiter=',',names=True)
 for ax in axs:ax.plot(d['x'],d['rate_real'],lw=.8,color=f'C{color}',label=f"{nx}: {r['growth']:.2f}/s; carrier {r['signed_rate_wavelength']:.4f} m")
axs[0].set(xlim=(2.5,5.5),ylabel='Signed rate, real part',title='Leading cumulative-space rate; common cosine phase reference at x=4 m')
axs[1].set(xlim=(3.7,4.3),xlabel='Physical x (m)',ylabel='Signed rate, real part',title='Central detail (unit maximum complex modulus)')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'leading-signed-modes.{ext}',dpi=180)
fig,axs=plt.subplots(1,3,figsize=(12,3.6))
for color,record in enumerate(records):
 nx=record['nx'];directory=OUT/f'native-{nx}'
 cross=np.genfromtxt(directory/'cross-section.csv',delimiter=',',names=True)
 op=np.genfromtxt(directory/'operating.csv',delimiter=',',names=True)
 raw=np.genfromtxt(directory/'restoring-modes.csv',delimiter=',',names=True);smooth=raw[raw['mode']==0]
 style='-' if color==0 else '--'
 axs[0].plot(cross['y'],cross['phi'],style,color=f'C{color}',label=f'{nx} columns')
 axs[1].plot(op['x'],op['V']*1e4,style,color=f'C{color}')
 axs[2].plot(smooth['x'],smooth['slip_response_per_force_coefficient']*1e6,style,color=f'C{color}')
axs[0].set(xlim=(1.8,2.2),xlabel='y (m), at x=4.01 m',ylabel='Frozen phase',title='Exact polynomial subdivision');axs[0].legend(fontsize=8)
axs[1].set(xlim=(1.5,6.5),xlabel='x (m)',ylabel='Operating rate (1e-4 m/s)',title='Same V(x), uniform age 1.61056 ms')
axs[2].set(xlim=(1.5,6.5),xlabel='x (m)',ylabel='Smooth response (1e-6 m/Pa/sqrt(m))',title='First native restoring action')
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'transfer-checks.{ext}',dpi=180)
allrows=[r for rec in records for r in rec['rows']]
with (OUT/'comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in allrows for k in row)));writer.writeheader();writer.writerows(allrows)
print('Wrote spectrum, signed-mode overlays, and complete table')
