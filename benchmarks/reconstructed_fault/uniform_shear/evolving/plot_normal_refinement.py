"""Plot saved normal-refinement evidence only; no simulations or parameter search."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reference import Reference, read_parameters

directory=Path(__file__).resolve().parent
figure,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
audit={}
for case,label in [('smoke','32x128'),('normal256','32x256')]:
    path=directory/case
    phi=np.genfromtxt(path/'comparison_phase_2.csv',delimiter=',',names=True)
    H=np.genfromtxt(path/'comparison_H_1.csv',delimiter=',',names=True)
    normalization=np.genfromtxt(path/'crack_integrals_2.csv',delimiter=',',names=True)
    axes[0,0].plot(phi['y'],phi['production_mean_phi'],label=label)
    axes[0,1].plot(phi['y'],phi['increment'],label=label)
    axes[1,0].plot(H['y'],H['increment'],label=label)
    axes[1,1].plot(normalization['x'],normalization['normalization_error'],label=label)
    model=Reference(read_parameters(path/'parameters.prm'),2048,6,.00225)
    particles=np.genfromtxt(path/'particles_0.csv',delimiter=',',names=True)
    prescribed=model.stationary(particles['y'])
    g,_=model.degradation(prescribed)
    _,hc=model.degradation(model.core)
    expected=np.where(prescribed>model.activation,model.Ec*model.core/(hc*g*g),model.Hc)
    error=particles['H']-expected
    audit[case]=dict(initial_H_pointwise_max_abs_Pa=float(max(abs(error))),
                    initial_H_pointwise_max_relative=float(max(abs(error)/expected)))
ref=np.loadtxt(directory/'normal256-reference/phase-2.csv',delimiter=',',skiprows=1)
ref1=np.loadtxt(directory/'normal256-reference/phase-1.csv',delimiter=',',skiprows=1)
axes[0,0].plot(ref[:,0],ref[:,1],'k--',label='Independent reference')
axes[0,1].plot(ref[:,0],ref[:,1]-ref1[:,1],'k--',label='Independent reference')
href=np.loadtxt(directory/'normal256-reference/history-1.csv',delimiter=',',skiprows=1)
h0=np.loadtxt(directory/'normal256-reference/history-0.csv',delimiter=',',skiprows=1)
axes[1,0].plot(href[:,0],href[:,1]-h0[:,1],'k--',label='Independent reference')
axes[1,1].axhline(1e-4,color='red',linestyle='--',label='Unchanged limit')
for axis,title,xlabel in zip(axes.flat,
    ('Full transverse phase at step 2','Phase feedback: phi2 - phi1','History feedback: H1 - H0 (Pa)',
     'Total supported normalization error at step 2'),('y (m)','y (m)','y (m)','x (m)')):
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
figure.suptitle('Fixed ell, fixed tangential/fault discretization; own CFL timestep sequence per mesh')
figure.savefig(directory/'normal256-comparison.png',dpi=160)
(directory/'normal256-initial-H-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
