"""Independent pointwise strip-8 objective on the solved first midpoint field.

All fixture inputs are read as binary64 before promotion, including the midpoint
duration and material constants of the 8 x 4 nine-strip specimen. The reference
does not call the compiled selector, its moment formula or its scalar root solver.
"""
import argparse
import csv
from pathlib import Path

import mpmath as mp


def captured_double(token):
    return mp.mpf(float(token))


def section(directory, step):
    with (directory / f'sections-{step:04d}.csv').open() as stream:
        return {int(row['section']): row for row in csv.DictReader(stream)}[8]


def native_points(directory):
    points = {}
    for shard in sorted(directory.glob('turn-points-*')):
        for line in shard.read_text().splitlines():
            cell, local, area, kernel, fraction, u, v, *_ = line.split()
            key = cell, int(local)
            if key in points:
                raise ValueError(f'Duplicated native point {key}')
            points[key] = tuple(map(captured_double, (area, kernel, fraction, u, v)))
    if len(points) != 180:
        raise ValueError(f'{directory}: found {len(points)} of 180 strip-8 native points')
    return [points[key] for key in sorted(points)]


class PointScore:
    """Full-strip signed RSF root and the unnormalized disk score (Pa² m²)."""

    def __init__(self, directory):
        parent = section(directory, 0)
        if parent['formed'] != '0' or captured_double(parent['beta']) != 0:
            raise ValueError('Expected the unformed zero-angle parent of the first turn')
        self.width = captured_double(parent['ds'])
        self.age = captured_double(parent['age'])
        self.pressure = captured_double((directory / 'root-input').read_text().split()[-1])
        self.h, G, eta, self.dc, self.vref, self.a, self.b, self.mu, self.damping = map(
            captured_double, (.0025, 1e6, 1e8, .01, .001, .015, .02, .6, 1000.))
        self.GBh = G * self.h / (1 + self.h * G / eta)
        if self.damping < self.pressure * self.h * self.b / self.dc:
            raise ValueError('Signed friction needs a monotone branch in this fixture')

        points = native_points(directory)
        # All points, including those outside the disk, enter the section solve.
        self.integral_u = mp.fsum(area * F * u for area, _, F, u, _ in points)
        self.integral_v = mp.fsum(area * F * v for area, _, F, _, v in points)
        self.integral_F = mp.fsum(area * F for area, _, F, _, _ in points)
        self.score_points = [(area * kernel * (1-F)**2, u, v)
                             for area, kernel, F, u, v in points if kernel and F != 1]
        self.update_threshold = 128 * mp.mpf(2)**-52 * mp.fsum(
            w * (u*u + v*v) for w, u, v in self.score_points)

    def resistance(self, speed):
        x = abs(speed) * self.h / self.dc
        stage_age = self.age * mp.exp(-x) + self.h * (-mp.expm1(-x)/x if x else 1)
        log_factor = (self.mu + self.b * mp.log(self.vref * stage_age / self.dc)) / self.a
        return (self.pressure * self.a * mp.asinh(speed * mp.exp(log_factor) / (2*self.vref))
                + self.damping * speed)

    def value(self, angle):
        """Return (rate in m/s, resistance in Pa, direct score in Pa² m²)."""
        s, c = mp.sin(2*angle), mp.cos(2*angle)
        scale = self.GBh * self.width / mp.cos(angle)
        free = (self.integral_u * s + self.integral_v * c) / scale
        mobility = self.integral_F / scale
        if not free:
            rate = force = mp.mpf(0)
        else:
            # The odd signed law has a unique root between zero and the free
            # rate. Bracket its magnitude on a log scale for cold creep.
            target = abs(free)

            def residual(log_speed):
                speed = mp.exp(log_speed)
                return (speed + mobility * self.resistance(speed)) / target - 1

            hi = mp.log(target)
            lo = hi-8
            while residual(lo) > 0:
                lo -= 8
            log_rate = mp.findroot(residual, (lo, hi), solver='anderson',
                                   maxsteps=150, tol=mp.eps*128)
            rate = mp.sign(free) * mp.exp(log_rate)
            force = self.resistance(rate)
            if not (lo <= log_rate <= hi and abs(rate + mobility*force - free) < mp.eps*512*target):
                raise ValueError('Signed friction root failed its bracket or relative residual check')
        score = mp.fsum(weight * (u*s + v*c - force)**2 for weight, u, v in self.score_points)
        return rate, force, score


def independent_peak(objective):
    """Scan the full sector, then locate stationary points by a centered difference."""
    dense_grid = mp.linspace(-mp.pi/4, mp.pi/4, 513)
    dense_scores = [objective.value(angle)[2] for angle in dense_grid]
    dense_peaks = [i for i in range(1, 512)
                   if dense_scores[i] > max(dense_scores[i-1], dense_scores[i+1])]
    grid, scores = dense_grid[::4], dense_scores[::4]
    best = max(range(129), key=scores.__getitem__)
    peaks = [i for i in range(1, 128) if scores[i] > max(scores[i-1], scores[i+1])]
    if best not in peaks:
        raise ValueError('Best scan sample is at a sector boundary or a tied sample')

    def refine(i, step):
        def derivative(angle):
            return (objective.value(angle+step)[2] - objective.value(angle-step)[2]) / (2*step)
        lo, hi = grid[i-1], grid[i+1]
        if derivative(lo) <= 0 or derivative(hi) >= 0:
            raise ValueError(f'Unbracketed stationary point at sample {i}')
        root = mp.findroot(derivative, (lo, hi), solver='anderson', tol=mp.mpf('1e-49'))
        if not lo < root < hi:
            raise ValueError(f'Stationary point left bracket {i}')
        return root

    coarse = refine(best, mp.mpf('1e-7'))
    fine = refine(best, mp.mpf('1e-9'))
    # Centered differentiation has O(step²) bias. Use a third step and compare
    # two extrapolations so the residual error has an empirical scale in rad.
    finer = refine(best, mp.mpf('1e-11'))
    extrapolated = finer + (finer-fine)/9999
    previous = fine + (fine-coarse)/9999
    if abs(extrapolated-previous) > mp.mpf('1e-22'):
        raise ValueError('Step-refined peak angle did not stabilize')
    maximum = objective.value(extrapolated)[2]
    if len(dense_peaks) != len(peaks) or any(abs(j/4-i) >= 1 for i, j in zip(peaks, dense_peaks)):
        raise ValueError('Additional peak bracket discovered on the independent 512-interval scan')
    competitors = [scores[0], scores[-1]]
    for i in peaks:
        if i != best:
            competitors.append(objective.value(refine(i, mp.mpf('1e-9')))[2])
    if max(competitors) >= maximum:
        raise ValueError('A competing bracket or sector boundary beats the peak')

    h = mp.mpf('1e-9')
    curvature = (objective.value(extrapolated+h)[2] + objective.value(extrapolated-h)[2]-2*maximum)/h**2
    return (best, grid[best-1], grid[best+1], extrapolated, peaks, dense_peaks,
            fine-coarse, extrapolated-previous, curvature, maximum-max(competitors))


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: mp.nstr(value, 32) if isinstance(value, mp.mpf) else value
                             for key, value in row.items()})


def case_results(name, directory, donor_plane=None):
    objective = PointScore(directory)
    bin_index, lo, hi, peak, bins, dense_bins, drift, extrapolation, curvature, gap = independent_peak(objective)
    maximum = objective.value(peak)[2]
    rows = [row for row in read_rows(directory/'turn-decisions.csv')
            if int(row['step']) == 1 and int(row['k']) == 8 and captured_double(row['incoming']) == 0]
    if [int(row['policy']) for row in rows] != [0, 1]:
        raise ValueError(f'Missing paired decisions on identical incoming state: {directory}')
    names = ['production', 'candidate']
    probes = read_rows(directory / 'turn-scores.csv')
    if [(int(r['policy']), r['where']) for r in probes] != [
            (policy, where) for policy in (0, 1) for where in ['incoming', 'off_peak', *names]]:
        raise ValueError('Missing a common-angle probe')

    selected = {}
    angles = {}
    for method, row in zip(names, rows):
        selected[method] = row
        angles[method] = captured_double(row['angle'])
        if int(row['bin']) != bin_index or max(abs(captured_double(row['lo'])-lo),
                                               abs(captured_double(row['hi'])-hi)) > mp.mpf('2e-16'):
            raise ValueError(f'{method} selected a different sector bracket')
        if captured_double(row['pressure']) != objective.pressure or row['accepted'] != '1':
            raise ValueError('Captured update, compression and first-turn input disagree')
        if not lo < angles[method] < hi:
            raise ValueError('Proposal outside its selected bracket')

    angles.update(independent=peak, incoming=mp.mpf(0), off_peak=captured_double(.23))
    if donor_plane is not None:
        donor, = [row for row in read_rows(donor_plane) if int(row['section']) == 8]
        if captured_double(donor['reference_dxi']) != objective.width or donor['formed'] != '1':
            raise ValueError('Donor plane has different reference width or formation status')
        angles['historical_donor_angle'] = captured_double(donor['beta'])
    values = {key: objective.value(angle) for key, angle in angles.items()}
    summary = []
    for method in angles:
        rate, force, score = values[method]
        observed = selected.get(method)
        peak_choice = method in [*names, 'independent']
        summary.append(dict(case=name, method=method, bin=bin_index if peak_choice else '',
                            bracket_lo_rad=lo if peak_choice else '', bracket_hi_rad=hi if peak_choice else '',
                            angle_rad=angles[method], angle_minus_peak_rad=angles[method]-peak,
                            direct_score_Pa2_m2=score, direct_minus_peak_Pa2_m2=score-maximum,
                            selector_score_Pa2_m2=observed['score'] if observed else '',
                            score_evaluation_error_Pa2_m2=captured_double(observed['score'])-score if observed else '',
                            resistance_Pa=force, rate_m_s=rate, length_m=objective.width/mp.cos(angles[method])))

    audit = []
    for row in probes:
        method, where = names[int(row['policy'])], row['where']
        rate, force, score = values[where]
        if captured_double(row['angle']) != angles[where]:
            raise ValueError('Probes are not on the common input angles')
        if method == where and row['score'] != selected[method]['score']:
            raise ValueError('Peak proposal and common-angle replay disagree')
        observed_force, observed_rate, observed_score = map(captured_double, (row['resistance'], row['rate'], row['score']))
        if abs(observed_force - force) > mp.mpf('1e-10') or abs(observed_rate/rate-1) > mp.mpf('1e-10'):
            raise ValueError('The evaluated signed law disagrees with the independent root')
        if where in ('incoming', 'off_peak') and (
                captured_double(selected[method]['score']) - observed_score <= objective.update_threshold):
            raise ValueError('Compiled selector does not distinguish the off-peak probe at its update threshold')
        audit.append(dict(case=name, method=method, probe=where, angle_rad=angles[where],
                          observed_resistance_Pa=observed_force, direct_resistance_Pa=force,
                          observed_rate_m_s=observed_rate, direct_rate_m_s=rate,
                          observed_score_Pa2_m2=observed_score, direct_score_Pa2_m2=score,
                          score_evaluation_error_Pa2_m2=observed_score-score))

    loss = {key: maximum-values[key][2] for key in values}
    if not max(loss['production'], loss['candidate']) < objective.update_threshold < min(
            loss['off_peak'], loss['incoming'], gap):
        raise ValueError('The off-peak control and the two nearby maxima are not distinguished by the existing threshold')
    diagnostics = dict(case=name, sector_local_bins_128=' '.join(map(str, bins)),
                       sector_local_bins_512=' '.join(map(str, dense_bins)),
                       coarse_to_fine_rad=drift, extrapolation_change_rad=extrapolation,
                       curvature_Pa2_m2_rad2=curvature, gap_to_next_sector_peak_or_edge_Pa2_m2=gap,
                       existing_update_threshold_Pa2_m2=objective.update_threshold,
                       width_m=objective.width, age_s=objective.age, pressure_Pa=objective.pressure)
    curves = []
    wide = max(abs(angles[method]-peak) for method in names)
    narrow = max(abs(angles['candidate']-peak), mp.mpf('1e-10'))
    for window, span in [('wide', wide), ('inset', narrow)]:
        for offset in mp.linspace(-mp.mpf('1.4')*span, mp.mpf('1.4')*span, 65):
            curves.append(dict(case=name, window=window, offset_rad=offset,
                               direct_minus_peak_Pa2_m2=objective.value(peak+offset)[2]-maximum))
    return summary, curves, audit, diagnostics


def archive_comparison(archive, case, current):
    objective = PointScore(archive)
    archived_bin, _, _, peak, *_ = independent_peak(objective)
    selected = {row['method']: row for row in current}
    if archived_bin != selected['independent']['bin']:
        raise ValueError('Fresh and archived fields select different direct peak brackets')
    if captured_double(section(archive, 1)['beta']) != selected['production']['angle_rad']:
        raise ValueError('Archived and fresh runs returned different production angles')
    fresh_peak = selected['independent']['angle_rad']
    maximum = objective.value(peak)[2]
    rows = []
    for name in ('production', 'candidate', 'independent', 'off_peak'):
        angle = peak if name == 'independent' else selected[name]['angle_rad']
        rate, force, score = objective.value(angle)
        rows.append(dict(case=case, probe=name, angle_on_archive_rad=angle,
                         archive_peak_rad=peak, fresh_minus_archive_peak_rad=fresh_peak-peak,
                         angle_minus_archive_peak_rad=angle-peak, archive_resistance_Pa=force,
                         archive_rate_m_s=rate, archive_score_Pa2_m2=score,
                         archive_score_minus_archive_peak_Pa2_m2=score-maximum))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('cases', nargs='+', metavar='NAME=CAPTURED_OUTPUT_DIR',
                        help='Each output includes both compiled proposals on the same first-turn field.')
    parser.add_argument('--frozen-donor-plane', type=Path, metavar='TIP_PLANES_0001_CSV',
                        help='Also evaluate the observed 5 ms donor angle on the frozen target field.')
    parser.add_argument('--archived-affine', type=Path, metavar='BUILD_INTERFACE_C2',
                        help='Directory containing the original cap1/output and cap2/output point captures.')
    args = parser.parse_args()
    mp.mp.dps = 65
    results = [case_results(name, Path(directory), args.frozen_donor_plane if name == 'frozen-2' else None)
               for name, directory in
               (case.split('=', 1) for case in args.cases)]
    args.destination.mkdir(parents=True, exist_ok=True)
    for filename, rows in zip(['peak-summary', 'peak-curves', 'common-angles'], zip(*[r[:3] for r in results])):
        write_table(args.destination / (filename+'.csv'), [row for dataset in rows for row in dataset])
    write_table(args.destination / 'peak-convergence.csv', [r[3] for r in results])
    if args.archived_affine is not None:
        summaries = {summary[0]['case']: summary for summary, _, _, _ in results}
        historical = []
        for ranks in (1, 2):
            case = f'affine-{ranks}'
            historical.extend(archive_comparison(args.archived_affine/f'cap{ranks}'/'output',
                                                 case, summaries[case]))
        write_table(args.destination / 'archive-peaks.csv', historical)


if __name__ == '__main__':
    main()
