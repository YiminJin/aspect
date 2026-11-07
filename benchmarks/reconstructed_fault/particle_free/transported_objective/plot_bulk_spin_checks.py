#!/usr/bin/env python3
"""Extract tested error tables and display the isolated spin correction."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-bulk-spin-mpl')
import re,csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin'
if __name__=='__main__':
    log=(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence/bulk-spin-units4.log').read_text()
    profiles=[dict(width=float(a),h_over_width=float(b),residual_L2=float(c),residual_Linf=float(d)) for a,b,c,d in re.findall(r'Bulk spin Q2 profile: width=(.*?), h/width=(.*?), residual_L2=(.*?), residual_Linf=(.*?)\n',log)]
    observer=[dict(steps=int(a),error=float(b)) for a,b in re.findall(r'Bulk spin observer Lie: steps=(\d+), error=(.*?)\n',log)]
    for name,rows in [('profile-spin',profiles),('observer-spin',observer)]:
        with (OUT/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    d=json.loads((OUT/'coupled.json').read_text());fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    widths=np.linspace(.05,.4,250);old=-.3/(2*widths)
    ax[0,0].plot(widths,old,label='total spin');ax[0,0].plot(widths,0*widths,label='bulk spin')
    ax[0,0].set(title='Pure slip: analytic angle, accumulated slip 0.3 m',xlabel='width (m)',ylabel='rotation angle (rad)');ax[0,0].legend()
    for width in [.1,.2,.4]:
        rows=[r for r in profiles if abs(r['width']-width)<1e-14]
        ax[0,1].loglog([r['h_over_width'] for r in rows],[r['residual_Linf'] for r in rows],'o-',label=f'width {width} m')
    ax[0,1].set(title='Q2 derivative vs smooth localized jump',xlabel='h / width',ylabel='absolute residual spin (1/s)');ax[0,1].legend(fontsize=8)
    ax[1,0].loglog([.8/r['steps'] for r in observer],[r['error'] for r in observer],'o-')
    ax[1,0].set(title='Time-dependent observer: accepted-spin Lie',xlabel='dt (s)',ylabel='completed stress error')
    labels=[];values=[]
    for r in d['comparisons'][:4]:labels.append(('constant' if '-c' in r['left'] else 'layered')+'\n'+('direct' if 'd' in r['left'].split('-')[1] else 'separated'));values.append(r['stress_relative_L2'])
    ax[1,1].bar(labels,values);ax[1,1].set(title='Coupled bulk-spin vs total-spin stress',xlabel='constant/layered; direct/separated',ylabel='relative L2 difference')
    from matplotlib.ticker import NullFormatter
    ax[0,1].set_xticks([.03125,.0625,.125,.25],['1/32','1/16','1/8','1/4'])
    ax[1,0].set_xticks([.01,.02,.04,.08],['.01','.02','.04','.08'])
    ax[0,1].xaxis.set_minor_formatter(NullFormatter());ax[1,0].xaxis.set_minor_formatter(NullFormatter())
    for a in ax.ravel():a.grid(alpha=.2);a.title.set_fontsize(11);a.tick_params(labelsize=9)
    fig.savefig(OUT/'spin-checks.png',dpi=160)
