#!/usr/bin/env python3
"""Saved-field time sampling, distinct from common native-time agreement."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuation_analysis import OUT, read


def main():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for prefix, label, color in [
        ('stable-coarse-half', r'$\Delta t$ vs $\Delta t/2$', '#222222'),
        ('stable-half-quarter', r'$\Delta t/2$ vs $\Delta t/4$', '#0072b2'),
    ]:
        a = read(OUT / (prefix + '-sampling.csv'))
        f = read(OUT / (prefix + '-sampling-fronts.csv'))
        axs[0, 0].plot(a['time_s'], a['max_absolute_phase_difference'], '.-', ms=2, color=color, label=label)
        axs[0, 1].plot(a['time_s'], 100*a['interval_growth_relative_L2'], '.-', ms=2, color=color, label=label)
        axs[1, 0].plot(a['time_s'], 100*a['interval_growth_relative_peak'], '.-', ms=2, color=color, label=label)
        for level, style in [(.001, ':'), (.1, '-'), (.5, '--')]:
            q = f[f['phi_contour'] == level]
            axs[1, 1].plot(q['time_s'], 1000*q['difference_m'], ls=style, color=color, label=label+f', phi={level:g}')
    axs[0, 0].set_ylabel('Maximum absolute phase discrepancy')
    axs[0, 1].set_ylabel('Relative Q1 $L^2$ discrepancy [% of growth to date]')
    axs[1, 0].set_ylabel('Relative peak discrepancy [% of growth to date]')
    axs[1, 1].set_ylabel('Centerline first-exit difference [mm]')
    for ax in [axs[0, 1], axs[1, 0]]:
        ax.axhline(5, color='tab:red', ls=':', label='Declared 5% target')
    for sign in [-1, 1]:
        axs[1, 1].axhline(sign*15.625, color='tab:red', ls=':', lw=.8)
    for ax in axs.flat:
        ax.axvline(.175, color='.6', ls=':')
        ax.set_xlabel('Time [s]')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Linear interpolation of saved physical phase only; no interpolated constitutive state')
    fig.savefig(OUT/'temporal-sampling.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
