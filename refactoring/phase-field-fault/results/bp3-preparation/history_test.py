"""Exercise event separation, checkpoint overlap and corrupt/missing history."""
import csv
import tempfile
from pathlib import Path

from history_events import Frame, STATIONS, PLATE_RATE, catalogue, unique_frames


def fixture():
    frames = []

    def add(dt, vmax, station_speed):
        p = frames[-1] if frames else None
        speeds = tuple(station_speed if xd == 10000. else PLATE_RATE for xd in STATIONS)
        slips = tuple((p.slips[j] if p else 0.) + dt*v for j, v in enumerate(speeds))
        frames.append(Frame(len(frames), (p.time if p else 0.)+dt, dt, vmax, 10000., speeds, slips))

    def reload():
        for _ in range(6):
            add(2e6, PLATE_RATE, PLATE_RATE)

    add(0., PLATE_RATE, PLATE_RATE)
    reload()
    for _ in range(3):
        add(1., .002, .002)
        add(1., .015, .012)
        # A brief return to quiet must not re-arm the counter. Nor may five
        # clocks of moderate afterslip, all below the legacy onset threshold.
        for _ in range(5):
            add(.1, PLATE_RATE, PLATE_RATE)
        for _ in range(6):
            add(10., 5e-5, 1e-5)
        add(1., .018, .01)
        reload()
    # A large rate at a vertex without a measured station slip pulse remains
    # a sub-Dc candidate; six rapid samples do not certify quiet duration.
    add(.05, .02, .0001)
    for _ in range(6):
        add(.1, PLATE_RATE, PLATE_RATE)
    reload()
    return frames


def write(directory, frames, damage=None):
    directory.mkdir()
    with (directory/'accepted_steps.csv').open('w') as sf, (directory/'stations.csv').open('w') as xf:
        speed = csv.writer(sf)
        station = csv.writer(xf)
        speed.writerow(['step','time','dt','max_V','xd_at_max_V'])
        station.writerow(['step','time','dt','xd','V','slip'])
        for f in frames:
            speed.writerow([f.step,f.time,f.dt,f.vmax,f.xd_at_max])
            for xd, v, slip in zip(STATIONS, f.speeds, f.slips):
                if damage == 'missing' and xd == 5000. and f == frames[0]:
                    continue
                if damage == 'slip' and xd == 5000. and f == frames[0]:
                    slip += .002
                station.writerow([f.step,f.time,f.dt,xd,v,slip])


def main():
    frames = fixture()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write(root/'full', frames)
        write(root/'first', frames[:15])
        write(root/'resume', frames[10:])
        clean = catalogue(unique_frames([root/'full']))
        if clean['completed_count'] != 3 or len(clean['sub_Dc_episodes']) != 1 or clean['ongoing'] is not None:
            raise ValueError(f'Wrong event count {clean}')
        if any(event['confirming_xd_m'] != 10000. or event['largest_fast_excess_m'] <= .0239
               for event in clean['completed']):
            raise ValueError('Incorrect same-station slip excess')
        if catalogue(unique_frames([root/'first', root/'resume'])) != clean:
            raise ValueError('Restart changed the event catalogue')
        if catalogue(unique_frames([root/'first']))['ongoing'] is None:
            raise ValueError('A short quiet period closed the first event')
        for label, start, corruption, earlier, expected in [
                ('gap', 16, None, ['first'], 'missing or conflicting accepted step'),
                ('fork', 10, 'slip', ['first'], 'missing or conflicting accepted step'),
                ('slip', 15, 'slip', ['first'], 'stored slip increment'),
                ('missing', 15, 'missing', ['first'], 'missing a weakening-zone station')]:
            write(root/label, frames[start:], corruption)
            try:
                catalogue(unique_frames([root/path for path in earlier+[label]]))
            except ValueError as e:
                if expected not in str(e):
                    raise
                print(label, 'rejected:', e)
            else:
                raise ValueError(f'{label} corruption passed')
    print('Clean and checkpoint-overlap: three recovered episodes, one sub-Dc candidate; '
          'brief relocking and intermediate afterslip pulses merged.')


if __name__ == '__main__':
    main()
