"""Plot the saved common-step localization audit; no simulations."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


here = Path(__file__).resolve().parent
figure,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
for col,n in enumerate((128,256)):
    data = np.genfromtxt(here/f'spatial0375_n{n}/identity_audit_3.csv',delimiter=',',names=True)
    x = data['x']
    for field,label in (('actual_signed_defect','Actual supported defect'),
                        ('full_signed_defect','Full-profile identity defect'),
                        ('omitted_signed_tail','Omitted tail contribution'),
                        ('bulk_quadrature_signed_defect','Bulk quadrature contribution')):
        axes[0,col].plot(x,data[field],label=label)
    axes[0,col].axhline(1e-4,color='black',linestyle='--',label='Unchanged limit')
    axes[0,col].set_title(f'32x{n}, t=1.125 s: signed defect')
    axes[0,col].legend(fontsize=7)
    axes[1,col].plot(x,1e6*(data['Ih_independent']/data['Ih_projected']-1),label='Current relative I_h error')
    axes[1,col].plot(x,1e6*(data['previous_Ih_independent']/data['previous_Ih_projected']-1),label='Previous relative I_h error')
    axes[1,col].set_ylabel('Column integral / Q1 I_h - 1 (ppm)')
    axes[1,col].legend(fontsize=8)
for axis in axes.flat:
    axis.set_xlabel('x (m)')
    axis.grid(alpha=.2)
figure.suptitle('Same timestep/history sequence; full-profile and tail defects kept separate')
figure.savefig(here/'common0375-identity.png',dpi=160)
