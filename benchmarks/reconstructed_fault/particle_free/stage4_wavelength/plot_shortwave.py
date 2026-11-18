#!/usr/bin/env python3
"""Matched native patterns and independently compatible short-wave reference."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-shortwave-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'evidence/stage4_shortwave'
old=json.loads((HERE.parent/'evidence/stage4_wavelength/spectrum-256.json').read_text())['rows']
new=json.loads((OUT/'spectrum-512.json').read_text())['rows']
ref=json.loads((OUT/'compatible-limit.json').read_text())
def get(rows,name):return next(r for r in rows if r['space']==name and r['law']=='weakening')
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for column,key,label,factor in [(0,'minimum_stiffness','Restoring stiffness (MPa/m)',1e-6),(1,'growth','Frozen growth rate (1/s)',1)]:
 ax=axes[0,column]
 for shift,marker,label2 in [(0,'o','Sine'),(1,'s','Cosine')]:
  vals=[get(old,f'single_96_{shift}')[key],get(new,f'single_96_{shift}')[key],get(new,f'single_192_{shift}')[key]]
  ax.plot(range(3),np.array(vals)*factor,marker=marker,label=label2)
 ax.set_xticks(range(3),['256: λ=.0625\n2 cells','512: λ=.0625\n4 cells','512: λ=.03125\n2 cells'])
 ax.axvline(1.5,color='.65',ls=':');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
axes[0,0].set_title('Same disturbance first; shorter disturbance second')
axes[0,1].set_title('Individual force patterns, identical operating point')
rows=[r for r in ref['carrier_rows'] if r['model']=='finite_compatible_WKB']
ax=axes[1,0];ax.semilogx([r['wavelength'] for r in rows],[r['growth'] for r in rows],'o-',label='Compatible finite-k, slow-x reference')
limit=next(r for r in ref['carrier_rows'] if r['model']=='continuum_resolved' and r['quadrature']==10)
if (OUT/'limit-quadrature.json').exists():
 limit=next(r for r in json.loads((OUT/'limit-quadrature.json').read_text())
            if r['model']=='continuum' and r['x_order']==10 and r['normal_order']==16)
ax.axhline(limit['growth'],color='k',ls='--',label=f"Fixed-patch limit {limit['growth']:.2f}/s")
ax.plot([.0625,.03125],[get(new,'pair96')['growth'],get(new,'pair192')['growth']],'s',label='Native 512 pairs (4 / 2 cells)')
ax.invert_xaxis();ax.set_xlabel('Wavelength (m), decreasing →');ax.set_ylabel('Frozen growth rate (1/s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
ax.set_title('A finite plateau; no selected patch spacing')
ax=axes[1,1]
for name,label in [('smooth6_pair96','Six smooth + λ=.0625'),('smooth6_pair192','Six smooth + λ=.03125')]:
 d=np.genfromtxt(OUT/f'mode-512-{name}.csv',delimiter=',',names=True)
 ax.plot(d['x'],d['rate_real'],label=label,lw=1)
ax.set_xlim(3.8,4.2);ax.set_xlabel('Along-fault x (m)');ax.set_ylabel('Normalized signed rate mode');ax.grid(alpha=.2);ax.legend(fontsize=8)
ax.set_title('Central signed modes (not amplitude envelopes)')
for ext in ('png','pdf'):fig.savefig(OUT/f'matched-shortwave.{ext}',dpi=170)
plt.close(fig)
