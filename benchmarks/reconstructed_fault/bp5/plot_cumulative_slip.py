"""Plot recorded BP5 cumulative slip, reusing the validated BP3 CSV reader."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bp3'))
from plot_cumulative_slip import read_profiles, select_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--increment-mm', type=float, default=0.025)
    parser.add_argument('--output', type=Path,
                        help='Output stem (default: input directory/plots/cumulative_slip)')
    args = parser.parse_args()
    if not np.isfinite(args.increment_mm) or args.increment_mm <= 0:
        parser.error('Slip increment must be finite and positive')
    source = args.input / 'cumulative_slip.csv' if args.input.is_dir() else args.input
    xd, profiles, summary = select_profiles(
        read_profiles(source), increment=args.increment_mm / 1000,
        xd_min=0, xd_max=float('inf'))
    summary.update(source=str(source.resolve()),
                   final_time_days=profiles[-1]['time_s'] / 86400,
                   maximum_final_slip_mm=float(np.max(profiles[-1]['slip'])) * 1000,
                   xd_window_km=[float(xd[0]) / 1000, float(xd[-1]) / 1000])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    days = [p['time_s'] / 86400 for p in profiles]
    norm = Normalize(days[0], days[-1] if days[-1] > days[0] else days[0] + 1)
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.8), layout='constrained')
    for ax, depth, title in zip(axes, (45, xd[-1] / 1000),
                                ('Shallow fault and transition', 'Entire fault')):
        for profile, day in zip(profiles, days):
            ax.plot(profile['slip'] * 1000, xd / 1000,
                    color=cmap(norm(day)), lw=0.75, alpha=0.9)
        ax.plot(profiles[-1]['slip'] * 1000, xd / 1000, color='black',
                lw=1.4, label=f'Final: {days[-1]:.3f} days')
        for boundary in (30, 33):
            ax.axhline(boundary, color='0.5', ls='--', lw=0.7)
            ax.text(0.98, boundary, f'{boundary} km', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=8, color='0.35')
        ax.set(xlabel='Cumulative slip (mm)', ylabel='Down-dip distance (km)',
               ylim=(depth, 0), title=title)
        ax.margins(x=0.025)
        ax.grid(alpha=0.15)
        ax.legend(loc='lower left', fontsize=8)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label='Physical time (days)', fraction=0.035, pad=0.03)
    fig.suptitle('Modified BP3 / BP5 friction: recorded cumulative slip\n'
                 f'{days[0]:g}–{days[-1]:.3f} days; '
                 f'{args.increment_mm:g} mm slip-increment selection', fontsize=13)
    axes[0].text(0, -0.14, 'Recorded nodal profiles; no smoothing or time interpolation.\n'
                 'Dashed lines: 30–33 km friction transition.',
                 transform=axes[0].transAxes, fontsize=8)
    output = args.output or source.parent / 'plots' / 'cumulative_slip'
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=220)
    plt.close(fig)
    columns = ['step', 'time_s', 'regime', 'max_rate_m_s']
    with output.with_suffix('.profiles.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: p[key] for key in columns} for p in profiles)
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
