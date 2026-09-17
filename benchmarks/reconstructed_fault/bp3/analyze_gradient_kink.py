"""Summarize existing manufactured runs without starting simulations."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-kink-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'gradient-kink'


def profile(label):
    return np.genfromtxt(OUT/label/'profile.csv', delimiter=',', names=True)


def main():
    summaries = []
    for p in sorted(OUT.glob('*/summary.json')):
        row = json.loads(p.read_text())
        if 'nx' not in row:
            continue
        row['label'] = p.parent.name
        summaries.append(row)
    keys = sorted(set().union(*(row.keys() for row in summaries)))
    with (OUT/'bulk_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, name in [('bulk32', 'bulk h=0.25, split'),
                        ('bulk32-native-load', 'bulk h=0.25, native'),
                        ('normal64', 'bulk h=0.125'),
                        ('bulk128-normal64', 'bulk h=0.0625')]:
        p = profile(label)
        ax[0].plot(p['s'], p['FE'], '.-', ms=3, label=name)
        ax[1].plot(p['s'], p['error'], '.-', ms=3, label=name)
    ax[0].plot(p['s'], p['target'], 'k--', label='continuum manufactured target')
    ax[0].set_xlim(1.4, 2.6)
    ax[0].set_ylim(-.65, .045)
    ax[1].set_xlim(1.2, 4)
    ax[0].set_ylabel('Rate perturbation / manufactured amplitude')
    ax[1].set_ylabel('Rate error / manufactured amplitude')
    for a in ax:
        a.set_xlabel('s / ell')
        a.grid(True, alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT/'kink_refinement.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [('bulk32', 'bulk h=0.25'), ('normal64', 'bulk h=0.125'),
                        ('bulk128-normal64', 'bulk h=0.0625')]:
        p = profile(label)
        ax.plot((p['s']-4)/.125, p['impulse'], '.-', label=name)
    ax.plot((p['s']-4)/.125, p['reference_impulse'], 'k--', label='exact bulk, same Q1 surface')
    ax.set_xlim(-7, 1)
    ax.set_xlabel('(s - junction) / h_Gamma')
    ax.set_ylabel('Rate response / prescribed impulse')
    ax.grid(True, alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'resolved_impulse.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nf in (32, 64, 128, 256):
        p = np.genfromtxt(OUT/'trace-comparison'/f'trace_{nf}_2048.csv', delimiter=',', names=True)
        ax[0].plot(p['s'], p['continuous'], '.-', ms=3, label=f'continuous, h={8/nf:g}')
        ax[1].plot(p['s'], p['independent_trace']-p['target_free_trace'], '.-', ms=3,
                   label=f'independent trace, h={8/nf:g}')
    ax[0].plot(p['s'], p['target_free_trace'], 'k--', label='exact free-side target')
    ax[0].plot([4, 4.3], [0, 0], 'k:', label='prescribed side (unchanged)')
    ax[0].set_xlim(3, 4.25)
    ax[0].set_ylabel('Rate perturbation / trace jump amplitude')
    ax[1].set_xlim(3, 4)
    ax[1].set_ylabel('Independent-trace error / jump amplitude')
    for a in ax:
        a.set_xlabel('s / ell')
        a.grid(True, alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT/'trace_remedy.png', dpi=170)
    plt.close(fig)

    paths = [HERE/name for name in ['manufactured_gradient_kink.py', 'manufactured_trace_comparison.py',
                                    'check_gradient_kink_nonlinear.py',
                                    'test_manufactured_gradient_kink.py', 'analyze_gradient_kink.py']]
    paths += [ROOT/name for name in ['doc/reconstructed_fault/current_design.md',
                                     'doc/reconstructed_fault/specification.tex',
                                     'source/simulator/assemblers/stokes.cc',
                                     'source/reconstructed_fault/surface_system.cc']]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    traces = json.loads((OUT/'trace-comparison'/'summary.json').read_text())
    nonlinear_traces = json.loads((OUT/'trace-nonlinear'/'summary.json').read_text())
    nonlinear_kink = json.loads((OUT/'nonlinear-friction'/'summary.json').read_text())
    manifest = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    input_hashes=hashes,
                    total_recorded_calculation_seconds=sum(r['seconds'] for r in summaries)+
                    sum(r['seconds'] for r in traces['cases'])+traces['reference_check']['seconds']+
                    sum(r['seconds'] for r in nonlinear_traces['cases'])+
                    nonlinear_traces['reference_check']['seconds']+nonlinear_kink['seconds'],
                    peak_RSS_KiB=max(r['peak_RSS_KiB'] for r in summaries),
                    scope='Independent diagnostic; no ASPECT run, production change, history advance or MPI test.')
    (OUT/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
