"""Plot consumed pressures and accepted rates from the two dipping fixtures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('trace', type=Path, help='mature affine output directory')
parser.add_argument('strips', type=Path, help='growing affine output directory')
parser.add_argument('figure', type=Path)
args = parser.parse_args()
with (args.strips/'friction-pressure-0001.csv').open() as f:
    growth = [(float(r['depth']),float(r['compression']),float(r['V'])) for r in csv.DictReader(f)]
mature = []
for f in args.trace.glob('trace-pressure-1-r*.csv'):
    with f.open() as stream:
        mature.extend((1-float(r['foot_y']),float(r['N']),float(r['V'])) for r in csv.DictReader(stream))
if len(growth)!=9 or not mature:
    raise ValueError('Missing accepted stations')
fig, axes = plt.subplots(2,1,figsize=(7.2,5.8),sharex=True,layout='constrained')
depth = np.linspace(0,1)
axes[0].plot(depth,200+50*depth,color='#333333',label='200 + 50 × depth')
for values,label,marker,color in [(mature,'Mature Q1 trace · 12.5 ms','.','#216e88'),
                                   (growth,'Growing strips · 5 ms','o','#b64a25')]:
    z,p,v=np.array(sorted(values)).T
    axes[0].scatter(z,p,s=14 if marker=='.' else 40,marker=marker,facecolors='none' if marker=='o' else color,
                    edgecolors=color,label=label)
    axes[1].scatter(z,v,s=14 if marker=='.' else 34,marker=marker,color=color)
axes[0].set_ylabel('Compression used by friction (Pa)')
axes[0].legend(frameon=False)
axes[1].set_ylabel('Slip rate (m/s)')
axes[1].set_yscale('log')
axes[1].set_xlabel('Physical depth on the fault (m)')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.set_xlim(-.02,1.02)
fig.savefig(args.figure,dpi=160)
