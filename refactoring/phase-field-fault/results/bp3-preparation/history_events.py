"""Count recovered BP3 slip episodes in one or more ordered donor output directories.

Each directory must contain accepted_steps.csv and stations.csv from the same run.
Pass the original output and all its restarts. Exact overlap at checkpoint clocks
is allowed. Gaps, conflicting observations and a run beginning after step zero
are errors. Units are metres, seconds and metres per second.
"""
import argparse
import csv
from dataclasses import dataclass
from itertools import groupby, zip_longest
import json
from math import isclose, isfinite, ulp
from pathlib import Path


# Proposed classification for this 60-degree, 50 MPa, Dc=8 mm case.
PLATE_RATE = 1e-9
ONSET = 1e-3                         # threshold used by the donor observer
RECOVERY = 10 * PLATE_RATE
SLIP_DISTANCE = .008
QUIET_DURATION = SLIP_DISTANCE / PLATE_RATE
STATIONS = (0., 2500., 5000., 7500., 10000., 12500., 15000.)


@dataclass(frozen=True)
class Frame:
    step: int
    time: float
    dt: float
    vmax: float
    xd_at_max: float
    speeds: tuple
    slips: tuple


def read_frames(directory):
    with (directory/'accepted_steps.csv').open() as speed_file, (directory/'stations.csv').open() as station_file:
        grouped = groupby(csv.DictReader(station_file), key=lambda row: int(row['step']))
        for speed, station_group in zip_longest(csv.DictReader(speed_file), grouped):
            if speed is None or station_group is None or int(speed['step']) != station_group[0]:
                raise ValueError(f'{directory}: speed and station frames have different clocks')
            time, dt, vmax, peak_xd = (float(speed[key]) for key in ('time', 'dt', 'max_V', 'xd_at_max_V'))
            if not all(map(isfinite, (time, dt, vmax, peak_xd))) or min(time, dt, vmax) < 0:
                raise ValueError(f'{directory}: invalid speed or clock at {speed["step"]}')
            by_distance = {}
            for row in station_group[1]:
                xd, velocity, slip = (float(row[name]) for name in ('xd', 'V', 'slip'))
                if float(row['time']) != time or float(row['dt']) != dt or not all(map(isfinite, (xd, velocity, slip))):
                    raise ValueError(f'{directory}: invalid station data or time at {speed["step"]}')
                if xd in by_distance or velocity < 0 or velocity > vmax*(1+1e-12):
                    raise ValueError(f'{directory}: duplicate station or unexpected speed at {xd} m')
                by_distance[xd] = velocity, slip
            if not set(STATIONS).issubset(by_distance):
                raise ValueError(f'{directory}: missing a weakening-zone station at {speed["step"]}')
            velocities, slips = zip(*(by_distance[xd] for xd in STATIONS))
            yield Frame(int(speed['step']), time, dt, vmax, peak_xd, velocities, slips)


def unique_frames(directories):
    # Keep accepted observations by step to verify overlaps across restart files.
    seen = []
    for directory in directories:
        count = 0
        for f in read_frames(directory):
            count += 1
            if 0 <= f.step < len(seen) and f == seen[f.step]:
                continue
            if f.step != len(seen):
                raise ValueError(f'{directory}: missing or conflicting accepted step {f.step}')
            if seen:
                p = seen[-1]
                if f.time <= p.time or not isclose(f.time-p.time, f.dt, rel_tol=0, abs_tol=4*ulp(f.time)):
                    raise ValueError(f'{directory}: inconsistent accepted interval at {f.step}')
                # Q1 interpolation at a stationary point commutes with the
                # donor's endpoint update S_new = S_old + dt*V_new.
                for xd, before, after, v in zip(STATIONS, p.slips, f.slips, f.speeds):
                    if abs(after-before-f.dt*v) > 128*ulp(max(1., abs(after), abs(before), abs(f.dt*v))):
                        raise ValueError(f'{directory}: stored slip increment at {xd:g} m, step {f.step}')
            elif f.time != 0 or f.dt != 0:
                raise ValueError(f'{directory}: initial clock must be zero')
            seen.append(f)
            yield f
        if count == 0:
            raise ValueError(f'{directory}: no accepted observations')


def catalogue(frames):
    completed = []
    small = []
    active = None
    previous = None
    quiet_start = None
    armed = False
    for f in frames:
        if previous is None:
            if f.vmax > RECOVERY:
                raise ValueError('Initial state already exceeds the quiet sliding speed')
            armed = True
        if previous is not None and armed and previous.vmax < ONSET <= f.vmax:
            active = {'onset_step': f.step, 'preceding_time_s': previous.time, 'onset_time_s': f.time,
                      'peak_speed_m_s': 0., 'station_fast_excess_m': [0.] * len(STATIONS)}
            armed = False
        if active is not None and f.vmax >= ONSET:
            if f.vmax > active['peak_speed_m_s']:
                active.update(peak_speed_m_s=f.vmax, peak_time_s=f.time, peak_xd_m=f.xd_at_max)
            for j, v in enumerate(f.speeds):
                if v > RECOVERY:
                    # Measure the actual increment at one persistent station.
                    # Subtract the slip steady plate motion would give there.
                    active['station_fast_excess_m'][j] += f.slips[j]-previous.slips[j]-PLATE_RATE*f.dt
        if f.vmax <= RECOVERY:
            if quiet_start is None:
                quiet_start = f
            if f.step-quiet_start.step >= 4 and f.time-quiet_start.time >= QUIET_DURATION:
                if active is not None:
                    slip = max(active['station_fast_excess_m'])
                    active.update(recovered_step=f.step, recovered_time_s=f.time,
                                  quiet_from_s=quiet_start.time, largest_fast_excess_m=slip,
                                  confirming_xd_m=STATIONS[active['station_fast_excess_m'].index(slip)])
                    (completed if slip >= SLIP_DISTANCE else small).append(active)
                    active = None
                armed = True
        else:
            quiet_start = None
        previous = f
    if previous is None:
        raise ValueError('No observations')
    return {'last_accepted_step': previous.step, 'last_accepted_time_s': previous.time,
            'onset_m_s': ONSET, 'quiet_limit_m_s': RECOVERY, 'quiet_duration_s': QUIET_DURATION,
            'slip_distance_m': SLIP_DISTANCE, 'station_distances_m': STATIONS,
            'completed_count': len(completed),
            'completed': completed, 'sub_Dc_episodes': small, 'ongoing': active}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(catalogue(unique_frames(args.outputs)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
