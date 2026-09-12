"""Plot saved coupled fault-resolution evidence at common physical times."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


here = Path(__file__).resolve().parent
summaries = json.loads((here/'coupled-fault-resolution.json').read_text())
common_step = min(s['final']['step'] for s in summaries)
figure,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
for normal in (128,256):
    for fault in (16,32):
        name = f'spatial0375_n{normal}'+('_f32' if fault==32 else '')
        identity = json.loads((here/name/'identity_audit.json').read_text())
        guards = [json.loads((here/name/f'guard_{k}.json').read_text()) for k in range(len(identity))]
        times = .375*np.arange(len(identity))
        label = f'32x{normal}, fault{fault}'
        axes[0,0].semilogy(times,[r['max_full_defect'] for r in identity],'.-',label=label)
        axes[0,1].plot(times,[g['max_supported_normalization_error'] for g in guards],'.-',label=label)
    name = f'spatial0375_n{normal}_f32'
    initial = np.genfromtxt(here/name/'surface_0.csv',delimiter=',',names=True)
    current = np.genfromtxt(here/name/f'surface_{common_step}.csv',delimiter=',',names=True)
    axes[1,0].plot(current['x'],current['Ih']-initial['Ih'],label=f'32x{normal}, fault32')
    phi0 = np.genfromtxt(here/name/'comparison_phase_0.csv',delimiter=',',names=True)
    phi = np.genfromtxt(here/name/f'comparison_phase_{common_step}.csv',delimiter=',',names=True)
    axes[1,1].plot(phi['y'],phi['production_mean_phi']-phi0['production_mean_phi'],label=f'32x{normal}, fault32')
reference = json.loads((here/'spatial0375_n128_f32-reference/report.json').read_text())
axes[1,0].axhline(reference['steps'][common_step-1]['Ih']-reference['initial']['Ih'],color='k',linestyle='--',label='Independent reference')
ref = np.loadtxt(here/f'spatial0375_n128_f32-reference/phase-{common_step}.csv',delimiter=',',skiprows=1)
ref0 = np.loadtxt(here/'spatial0375_n128_f32-reference/phase-0.csv',delimiter=',',skiprows=1)
axes[1,1].plot(ref[:,0],ref[:,1]-ref0[:,1],'k--',label='Independent reference')
axes[0,1].axhline(1e-4,color='k',linestyle='--',label='Unchanged normalization limit')
titles = ('Maximum full-profile identity defect','Actual supported normalization error',
          f'I_h increment at common t={common_step*.375:g} s (m)',
          f'Mean transverse phase increment at t={common_step*.375:g} s')
for axis,title,xlabel in zip(axes.flat,titles,('time (s)','time (s)','x (m)','y (m)')):
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(alpha=.2)
    axis.legend(fontsize=7)
figure.suptitle('Coupled fault refinement: normalization improvement and retained homogeneity limitation')
figure.savefig(here/'coupled-fault-resolution.png',dpi=160)
