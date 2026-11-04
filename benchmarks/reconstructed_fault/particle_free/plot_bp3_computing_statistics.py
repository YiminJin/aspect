#!/usr/bin/env python3
"""Plot BP3 server log timers and solver convergence; never run the solver.

Combine the last cumulative timer table from each process invocation once.
Draw one four-panel figure with unified accepted-state histories. Timed sections can
nest, so show independent percentages, not a purported additive pie chart.
Only explicit 'BP3 accepted state' records enter convergence statistics.
Dependencies: numpy, matplotlib. Outputs: PNG/PDF plus CSV/JSON provenance.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / 'benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event'
NUMBER = r'[-+\d.eE]+'
STEP = re.compile(rf'\*\*\* Timestep (\d+):\s+t=({NUMBER}) seconds, dt=({NUMBER}) seconds')
RESIDUAL = re.compile(rf'after nonlinear iteration\s+(\d+): ({NUMBER}), ({NUMBER})')
LINEAR = re.compile(rf'Fault linear solve: iterations=(\d+), fresh=({NUMBER}), target=({NUMBER})')
TIMER = re.compile(rf'\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*({NUMBER})s\s*\|\s*({NUMBER})%\s*\|')


def parse_log(path):
    """Keep attempts separate, including incomplete/rejected trailing attempts."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    text = raw.decode()
    attempts, tables = [], []
    current = table = None
    for line in text.splitlines():
        if match := STEP.search(line):
            current = dict(step=int(match[1]), time_s=float(match[2]), dt_s=float(match[3]),
                           accepted=False, residuals=[], linear=[], line_search_rejections=0)
            attempts.append(current)
        elif match := re.search(r'BP3 accepted state\s+(\d+)', line):
            if current is None or current['step'] != int(match[1]):
                raise ValueError(f'Acceptance without matching timestep: {path}: {line}')
            current['accepted'] = True
        elif current is not None and (match := RESIDUAL.search(line)):
            current['residuals'].append(dict(iteration=int(match[1]), bulk=float(match[2]), fault=float(match[3])))
        elif current is not None and (match := LINEAR.search(line)):
            current['linear'].append(dict(iterations=int(match[1]), fresh=float(match[2]), target=float(match[3])))
        elif current is not None and (match := re.search(r'accepted after (\d+) rejected candidates', line)):
            current['line_search_rejections'] += int(match[1])
        elif line.startswith('| Total wallclock time elapsed since start'):
            elapsed = float(line.split('|')[2].strip().removesuffix('s'))
            table = dict(step=current['step'] if current else None, wall_s=elapsed, sections=[])
        elif table is not None and (match := TIMER.fullmatch(line)):
            table['sections'].append(dict(section=match[1].strip(), calls=int(match[2]),
                                          wall_s=float(match[3]), printed_percent=float(match[4])))
        elif match := re.search(r'Total wallclock time elapsed including restarts: (\d+)s', line):
            if table is None or not table['sections']:
                raise ValueError(f'Missing timer table before cumulative clock: {path}')
            table['including_restarts_s'] = int(match[1])
            tables.append(table)
            table = None
    accepted = [a for a in attempts if a['accepted']]
    if not accepted or not tables:
        raise ValueError(f'No accepted steps or complete timer tables: {path}')
    if any(not a['residuals'] for a in accepted):
        raise ValueError(f'Accepted step lacks residual observations: {path}')
    tolerances = {float(t) for t in re.findall(r'"Nonlinear solver tolerance":\s*"([^"]+)"', text)}
    if len(tolerances) != 1:
        raise ValueError(f'Cannot identify unique logged nonlinear tolerance: {path}')
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), segment=path.parent.name,
                ranks=int(re.search(r'running with (\d+) MPI processes', text)[1]),
                tolerance=tolerances.pop(), attempts=attempts, timers=tables)


def accepted_rows(run):
    return [dict(segment=run['segment'], step=a['step'], time_s=a['time_s'], dt_s=a['dt_s'],
                 residual_evaluations=len(a['residuals']), linear_solves=len(a['linear']),
                 linear_iterations=sum(l['iterations'] for l in a['linear']),
                 final_bulk=a['residuals'][-1]['bulk'], final_fault=a['residuals'][-1]['fault'],
                 line_search_rejections=a['line_search_rejections'])
            for a in run['attempts'] if a['accepted']]


def timer_windows(run):
    """Difference only adjacent clocks in the same invocation (integer seconds).

    Never difference rounded per-section times or clocks across a restart. The
    first startup/resume interval is omitted because it has no prior timestamp.
    """
    rows = []
    for left, right in zip(run['timers'], run['timers'][1:]):
        elapsed = right['including_restarts_s'] - left['including_restarts_s']
        count = sum(a['accepted'] and left['step'] < a['step'] <= right['step'] for a in run['attempts'])
        if elapsed <= 0 or count == 0:
            raise ValueError('Non-increasing clock or empty accepted-step timer window')
        rows.append(dict(segment=run['segment'], from_step=left['step'], through_step=right['step'],
                         accepted_steps=count, wall_s=elapsed, wall_s_per_accepted_step=elapsed/count))
    return rows


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def combined_timers(runs):
    """Sum final invocation tables once, weighting shares by elapsed wall time."""
    wall_s = sum(r['timers'][-1]['wall_s'] for r in runs)
    sections = {}
    for run in runs:
        for section in run['timers'][-1]['sections']:
            name = section['section']
            sections[name] = sections.get(name, 0.) + section['wall_s']
    return dict(wall_s=wall_s, sections=[dict(section=name, wall_s=seconds,
                                           percent=100*seconds/wall_s)
                                       for name, seconds in sorted(sections.items(), key=lambda x: x[1], reverse=True)])


def draw(runs, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    blue, orange = '#286ca8', '#d05c2c'
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    timer_ax, dt_ax, residual_ax, effort_ax = axes.flat
    totals = combined_timers(runs)
    sections = totals['sections'][:8]
    labels = {'Fault: condensed linear solve': 'Condensed linear solve', 'Create snapshot': 'Checkpoint writing',
              'Assemble Stokes system rhs': 'Stokes RHS assembly', 'Assemble Stokes system Picard': 'Stokes matrix assembly',
              'Build Stokes preconditioner': 'Stokes preconditioner', 'Fault: Property preparation': 'Fault property preparation',
              'Native history candidate/check': 'History candidate / check'}
    bars = timer_ax.barh(np.arange(len(sections)), [s['percent'] for s in sections], color=blue, height=.65)
    timer_ax.bar_label(bars, labels=[f"{s['percent']:.1f}%" for s in sections], padding=4, fontsize=10)
    timer_ax.set_yticks(np.arange(len(sections)), [labels.get(s['section'], s['section']) for s in sections])
    timer_ax.invert_yaxis()
    timer_ax.set_xlim(0, max(s['percent'] for s in sections)*1.2)
    timer_ax.set_xlabel('Section wall time / total reported wall time (%)')
    timer_ax.set_title('A  Largest logged timer sections', loc='left', weight='bold')

    rows = sorted((row for run in runs for row in accepted_rows(run)), key=lambda r: r['step'])
    steps = [r['step'] for r in rows]
    dt_ax.plot(steps, [r['dt_s'] if r['dt_s'] > 0 else np.nan for r in rows], color=blue, lw=1.5)
    dt_ax.set(yscale='log', xlabel='Accepted step', ylabel='Timestep (s)')
    dt_ax.set_title('B  Adaptive physical timestep', loc='left', weight='bold')
    residual_ax.plot(steps, [r['final_bulk'] for r in rows], '.', ms=2, color=blue, label='Bulk')
    residual_ax.plot(steps, [r['final_fault'] for r in rows], '.', ms=2, color=orange, label='Fault')
    for tol in sorted({r['tolerance'] for r in runs}):
        residual_ax.axhline(tol, color='black', ls='--', lw=1, label=f'Logged tolerance {tol:g}')
    residual_ax.set(yscale='log', xlabel='Accepted step', ylabel='Normalized nonlinear residual')
    residual_ax.set_title('C  Final residuals of accepted steps', loc='left', weight='bold')
    residual_ax.legend(fontsize=9, loc='lower center', ncol=3, markerscale=2)
    effort_ax.plot(steps, [r['linear_iterations'] for r in rows], '.', ms=2.5, color=blue)
    effort_ax.set(xlabel='Accepted step', ylabel='Sum of linear iterations per accepted step')
    effort_ax.set_title('D  Linear solver effort', loc='left', weight='bold')
    for ax in (dt_ax, residual_ax, effort_ax):
        ax.grid(alpha=.18)
    rank_label = '/'.join(str(n) for n in sorted({r['ranks'] for r in runs}))
    fig.suptitle(f'BP3-derived server run · computing statistics · {rank_label} MPI ranks', fontsize=16, weight='bold')
    excluded = sum(not a['accepted'] for r in runs for a in r['attempts'])
    fig.text(.5, .025, f"Timers cover ~{totals['wall_s']/3600:.1f} reported wall hours; rounded sections may overlap and omit unreported tails.\n"
             f"Accepted-state histories only; {excluded} unaccepted attempts excluded. Residual convergence is not a physical-error certificate.",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2.4, w_pad=2.5)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'computing_statistics.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='*', type=Path, default=[SERVER / d / 'log.txt' for d in ('output_start', 'output_resumed')])
    parser.add_argument('--output', type=Path, default=ROOT / 'build-particle-free/bp3-computing-statistics')
    args = parser.parse_args()
    runs = [parse_log(p) for p in args.logs]
    if len({r['sha256'] for r in runs}) != len(runs):
        raise ValueError('Duplicate input log: do not count stdout copies twice')
    seen = set()
    for run in runs:
        for row in accepted_rows(run):
            if row['step'] in seen:
                raise ValueError('Overlapping accepted steps; choose nonoverlapping invocation logs')
            seen.add(row['step'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for run in runs for row in accepted_rows(run)]
    write_csv(output / 'accepted_steps.csv', rows)
    write_csv(output / 'timing_windows.csv', [w for r in runs for w in timer_windows(r)])
    write_csv(output / 'timer_sections.csv', [dict(segment=r['segment'], through_step=t['step'], invocation_wall_s=t['wall_s'], **s)
                                            for r in runs for t in r['timers'] for s in t['sections']])
    write_csv(output / 'residuals.csv', [dict(segment=r['segment'], step=a['step'], accepted=a['accepted'], **v)
                                      for r in runs for a in r['attempts'] for v in a['residuals']])
    draw(runs, output)
    summaries = []
    for r in runs:
        accepted = accepted_rows(r)
        summaries.append(dict(segment=r['segment'], path=r['path'], sha256=r['sha256'], ranks=r['ranks'], tolerance=r['tolerance'],
                              accepted_count=len(accepted), first_accepted=accepted[0]['step'], last_accepted=accepted[-1]['step'],
                              unaccepted_attempts=[a['step'] for a in r['attempts'] if not a['accepted']],
                              timer_tables=len(r['timers']), final_timer=r['timers'][-1],
                              residual_evaluations_range=[min(a['residual_evaluations'] for a in accepted), max(a['residual_evaluations'] for a in accepted)],
                              linear_iterations_range=[min(a['linear_iterations'] for a in accepted), max(a['linear_iterations'] for a in accepted)],
                              median_linear_iterations=float(np.median([a['linear_iterations'] for a in accepted])),
                              max_final_bulk=max(a['final_bulk'] for a in accepted), max_final_fault=max(a['final_fault'] for a in accepted),
                              line_search_rejections=sum(a['line_search_rejections'] for a in accepted),
                              linear_fresh_over_target_max=max(l['fresh']/l['target'] for a in r['attempts'] if a['accepted'] for l in a['linear']),
                              median_window_wall_s_per_step=float(np.median([w['wall_s_per_accepted_step'] for w in timer_windows(r)]))))
    summary = dict(command=sys.argv, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=summaries, combined_timers=combined_timers(runs),
                   caveats=['Timers are cumulative within each invocation, rounded, and potentially nested.',
                            'No timer coverage after the last complete table; no fabricated full-job partition.',
                            'No CPU-hours, rank scaling or per-Krylov-iteration residual history inferred.',
                            'Production mature DG run; no cohesive-interface or objective-spin qualification.'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), accepted_steps=len(rows), summaries=[{k:v for k,v in r.items() if k not in ('final_timer',)} for r in summaries]), indent=2))


if __name__ == '__main__':
    main()
