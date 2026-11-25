"""Plot the first accepted physical state of the nine-strip wall-loading controls."""
import argparse
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def at_five_ms(root, case):
    output = root/case/'output'
    with (output/'balance-0001.csv').open() as stream:
        ledger, = csv.DictReader(stream)
    if float(ledger['time']) != .005:
        raise ValueError('Expected the 5 ms accepted endpoint')
    with (output/'points-0001-r0.csv').open() as stream:
        shear = max(abs(float(p['tauxy'])) for p in csv.DictReader(stream))
    return float(ledger['bulk']), shear


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','figure'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    cases = ['half','ramp','full']
    figure, axes = plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
    for i,(root,label,color) in enumerate([(args.before,'6B','#989da5'),(args.after,'Stage A','#246b82')]):
        values = [at_five_ms(root,case) for case in cases]
        for ax, series in zip(axes,zip(*values)):
            ax.bar(np.arange(3)+(i-.5)*.34,series,width=.34,label=label,color=color)
    for ax,title in zip(axes,['Bulk elastic energy (J/m)','Maximum native |τxy| (Pa)']):
        ax.set_ylabel(title)
        ax.set_xticks(range(3),['Half load','Linear ramp','Full load'])
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle('Nine-strip fault at 5 ms')
    figure.savefig(args.figure,dpi=160)
