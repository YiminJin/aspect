"""Measure slip, phase change and physical work on the 18-strip, 5–10 ms window."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import fields, read


def reduce_window(parent_dir, output, parent_fields, n):
    initial, = read(parent_dir/'bulk-accepted.csv')
    ledger = read(output/'bulk-accepted.csv')
    assert len(ledger) == n and initial['time'] == .005
    np.testing.assert_allclose(ledger['dt'], .005/n, atol=1e-16, rtol=0)
    np.testing.assert_allclose(ledger['time'], initial['time']+np.cumsum(ledger['dt']), atol=1e-16, rtol=0)
    np.testing.assert_array_equal(ledger['publications'], np.arange(2,2+n))
    assert abs(ledger[-1]['time']-.01) < 1e-16

    prev_planes = read(parent_dir/'tip-planes-0001.csv')
    prev_energy = initial['fracture']
    area = parent_fields['weight']
    assert len(prev_planes) == 18 and np.isclose(area.sum(), 2.)
    phase = parent_fields['phi']
    steps = []
    for row in ledger:
        step = row['step']
        planes = read(output/f'tip-planes-{step:04}.csv')
        rates = read(output/f'tip-sections-{step:04}.csv')
        p = fields(output, step)
        for key in ('cell','q','section','x','y','weight'):
            np.testing.assert_array_equal(p[key], parent_fields[key])
        for key in ('slip','age','formed'):
            np.testing.assert_array_equal(planes['parent_'+key], prev_planes[key])
        slip = planes['ds'] @ (planes['slip']-prev_planes['slip'])
        np.testing.assert_allclose(slip, row['dt']*(rates['ds']@rates['V']), rtol=1e-12)
        res = row['Wext']-row['energy_change']-row['Db']-row['Dc']-row['Df']
        remainder = row['phase_remainder']
        other = (row['stress_loss']+row['traction_loss']+row['fracture_loss']
                 -row['phase_work']-row['mechanical_work']+row['ledger_error'])
        assert abs(res-remainder-other) < 1e-12
        inc = p['phi']-phase
        steps.append(dict(step=int(step), time_s=row['time'], signed_slip_m2=slip,
                          stage_peak_abs_rate_m_s=max(abs(rates['V'])), fracture_change_J_m=row['fracture']-prev_energy,
                          phase_increment_L2_m=np.sqrt(area @ inc**2), work_residual_J_m=res,
                          midpoint_remainder_J_m=remainder, phase_work_J_m=row['phase_work'],
                          mechanical_work_J_m=row['mechanical_work'], ledger_error_J_m=row['ledger_error'],
                          registrations=int(sum(planes['formed']-planes['parent_formed']))))
        prev_energy, prev_planes, phase = row['fracture'], planes, p['phi']

    delta = phase-parent_fields['phi']
    residuals = np.array([s['work_residual_J_m'] for s in steps])
    added = ledger[-1]['fracture']-initial['fracture']
    return dict(steps=steps, signed_slip_m2=sum(s['signed_slip_m2'] for s in steps),
                end_integrated_slip_m2=prev_planes['ds'] @ prev_planes['slip'],
                added_fracture_J_m=added, phase_increment_L2_m=np.sqrt(area @ delta**2),
                phase_increment_sample_max=np.max(delta), signed_residual_J_m=sum(residuals),
                absolute_residual_J_m=sum(abs(residuals)), abs_residual_over_added=sum(abs(residuals))/added,
                signed_midpoint_remainder_J_m=sum(ledger['phase_remainder']),
                absolute_midpoint_remainder_J_m=sum(abs(ledger['phase_remainder']))), delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','reference','half'):
        parser.add_argument(arg, type=Path, help='Native output directory')
    args = parser.parse_args()
    parent = fields(args.parent, 1)
    coarse, a = reduce_window(args.parent, args.reference, parent, 1)
    fine, b = reduce_window(args.parent, args.half, parent, 2)
    print(json.dumps(dict(reference=coarse, half=fine,
                          phase_increment_field_difference_L2_m=np.sqrt(parent['weight'] @ (b-a)**2),
                          phase_increment_field_difference_sample_max=np.max(abs(b-a))), indent=2))
