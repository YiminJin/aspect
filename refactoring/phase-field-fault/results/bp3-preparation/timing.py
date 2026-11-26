"""Price three-event schedules using measured 160-rank BP3 timer windows."""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(windows, times):
    count = sum(int(w['accepted_steps']) for w in windows)
    wall = sum(int(w['wall_s']) for w in windows)
    first, last = int(windows[0]['from_step']), int(windows[-1]['through_step'])
    if last - first != count:
        raise ValueError('Timing intervals must cover contiguous accepted steps')
    elapsed = times[last] - times[first]
    costs = [float(w['wall_s_per_accepted_step']) for w in windows]
    print(f'{first}–{last}: {count} steps, {elapsed:.9g} physical s, {wall} wall s, '
          f'{wall/count:.6f} wall s/step (100-step range {min(costs):.2f}–{max(costs):.2f})')
    return wall/count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('counters', type=Path, help='output of the donor computing-statistics parser')
    parser.add_argument('start_history', type=Path, help='accepted_steps.csv from job 3610007')
    parser.add_argument('--event-steps', nargs='+', type=int, default=[4000, 8000],
                        help='assumed advances after each upward crossing through recovery')
    parser.add_argument('--resume-step', type=int,
                        help='also price the continuation from an accepted step in this first episode')
    args = parser.parse_args()
    windows = rows(args.counters/'timing_windows.csv')
    histories = rows(args.counters/'accepted_steps.csv')
    times = {int(r['step']):float(r['time_s']) for r in histories}
    if len(histories) != len(times):
        raise ValueError('Repeated accepted step in the two jobs')
    runs = {r['segment']:r for r in json.loads((args.counters/'summary.json').read_text())['runs']}
    if set(runs) != {'output_start', 'output_resumed'} or any(r['ranks'] != 160 for r in runs.values()):
        raise ValueError('Expected the two 160-rank server invocations')
    startup = runs['output_start']
    initial_windows = [r for r in windows if r['segment'] == 'output_start']
    resumed_windows = [r for r in windows if r['segment'] == 'output_resumed']
    initial = rows(args.start_history)
    acceleration = next(int(r['step']) for r in initial if float(r['max_V']) > 1e-8)
    onset = next(int(r['step']) for r in initial if float(r['max_V']) >= 1e-3)
    print(f'First crossing of 1e-3 m/s: step {onset}, {times[onset]/31557600:.9f} years')
    summarize([w for w in initial_windows if int(w['through_step']) < acceleration], times)
    # The resumed job samples the accelerating first event. Its native profiles
    # are intermittent; the timer counts every accepted step between reports.
    event_cost = summarize(resumed_windows, times)
    bracket, = [w for w in initial_windows if int(w['from_step']) < onset <= int(w['through_step'])]
    lo, hi = int(bracket['from_step']), int(bracket['through_step'])
    upper = startup['final_timer']['including_restarts_s'] - sum(
        int(w['wall_s']) for w in initial_windows if int(w['from_step']) >= hi)
    lower = upper - int(bracket['wall_s'])
    loading_hours = (lower + (upper-lower)*(onset-lo)/(hi-lo))/3600
    print(f'Initial loading to onset: {lower/3600:.2f}–{upper/3600:.2f} h; '
          f'linear interpolation in this timer window: {loading_hours:.2f} h')
    observed = runs['output_resumed']['last_accepted'] - onset
    print(f'{observed} further advances after the first crossing, with recovery unobserved')
    if args.resume_step is not None and not onset <= args.resume_step <= onset + observed:
        raise ValueError('The assumed restart step is outside the observed first episode')
    for steps in args.event_steps:
        if steps <= observed:
            raise ValueError('This scenario must allow the first event to continue through recovery')
        event_hours = steps*event_cost/3600
        print(f'Assume three loading periods of {loading_hours:.2f} h and {steps} steps/event: '
              f'{3*loading_hours + 2*event_hours:.2f} h to third onset, '
              f'{3*loading_hours + 3*event_hours:.2f} h through third recovery')
        if args.resume_step is not None:
            completed_hours = loading_hours + (args.resume_step-onset)*event_cost/3600
            print(f'If parent {args.resume_step} is recovered, on that schedule: '
                  f'{3*loading_hours + 2*event_hours - completed_hours:.2f} h to third onset, '
                  f'{3*loading_hours + 3*event_hours - completed_hours:.2f} h through third recovery')


if __name__ == '__main__':
    main()
