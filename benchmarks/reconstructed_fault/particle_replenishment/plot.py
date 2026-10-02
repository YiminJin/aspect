#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent/'results'
a=list(csv.DictReader((r/'stage_fields.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for c in ['regular','random-5432','random-5433','random-5434']:
 b=[x for x in a if x['case']==c and x['stage']=='transport' and x['site']=='Q2_gauss3' and x['family']=='2']
 t=[int(x['step'])*.2 for x in b]
 axes[0].plot(t,[float(x['min_ratio']) for x in b],label=c)
 axes[1].plot(t,[float(x['rms_error_over_1e8']) for x in b],label=c)
axes[0].set(ylabel='Minimum singular-value ratio',xlabel='Time (s)')
axes[1].set(ylabel='Curved Q2 RMS error / 1e8',xlabel='Time (s)',title='Unlimited native LLS')
axes[0].legend(fontsize=8)
fig.savefig(r/'conditioning_and_error.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for c in ['regular','random-5432','random-5433','random-5434']:
 for family,ax in [('1',axes[0]),('2',axes[1])]:
  b=[x for x in a if x['case']==c+'-limited' and x['stage']=='transport' and x['site']=='Q2_gauss3' and x['family']==family]
  ax.plot([int(x['step'])*.2 for x in b],[float(x['rms_error_over_1e8']) for x in b],label=c)
axes[0].set(ylabel='Affine Q2 RMS error / 1e8',xlabel='Time (s)',title='Native limiter: inflow accuracy cost')
axes[1].set(ylabel='Curved Q2 RMS error / 1e8',xlabel='Time (s)',title='Native limiter: bounded extrapolation')
axes[0].legend(fontsize=8);fig.savefig(r/'limiter_tradeoff.png',dpi=160)
