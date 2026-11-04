"""Sampling/history checks for the server-slip plot; no solver or graphics tests."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_bp3_cumulative_slip import load_history, select_timed_profiles
from plot_cumulative_slip import Profile, YEAR


class SavedTimeSelection(unittest.TestCase):
    @staticmethod
    def profile(step, time, velocity, slip=0.):
        return Profile(step, time, np.arange(3), np.array([0., 20000., 50000.]),
                       np.array([slip, .5*slip, .25*slip]), np.array(velocity))

    def test_gap_does_not_manufacture_intermediate_rupture_profiles(self):
        profiles = [self.profile(10, 5e9, [.1, .1, 0.], slip=0.),
                    self.profile(50, 5e9+3., [.2, .1, 0.], slip=2.)]
        chosen, targets, summary = select_timed_profiles(profiles)
        self.assertEqual([r['step'] for r in chosen], [10, 50])
        self.assertEqual([r['target_offset_s'] for r in targets if r['status']=='omitted'], [1., 2.])
        self.assertEqual(summary['omitted_targets'], 2)
        np.testing.assert_array_equal(profiles[1].slip, [2., 1., .5])

    def test_full_fault_instantaneous_rate_and_threshold_equality(self):
        # Only the node outside the default 0–40 km display window is seismic.
        # Average slip/time is deliberately tiny and cannot replace saved V.
        profiles = [self.profile(0, 0., [1e-9, 1e-9, 1e-9]),
                    self.profile(7, 100., [1e-9, 1e-9, -1e-3], slip=1e-10)]
        chosen, _, summary = select_timed_profiles(profiles)
        self.assertEqual([r['regime'] for r in chosen], ['interseismic', 'coseismic'])
        self.assertEqual(summary['sampled_maximum_V_m_per_s'], 1e-3)
        self.assertEqual(summary['sampled_regime_transitions'][0]['unsampled_gap_s'], 100.)

    def test_nearest_saved_year_targets_and_boundary_are_distinct(self):
        profiles = [self.profile(i, t*YEAR, [1e-9]*3, slip=i*.1)
                    for i, t in enumerate([0., 4.8, 5.2, 9.9, 12.])]
        chosen, targets, summary = select_timed_profiles(profiles)
        self.assertEqual([r['step'] for r in chosen], [0, 1, 3, 4])
        self.assertEqual([r['selected_step'] for r in targets], [0, 1, 3])
        self.assertEqual(chosen[-1]['selection_kind'], 'boundary')
        self.assertEqual(summary['omitted_targets'], 0)

    def test_epoch_seconds_retain_subsecond_target_offsets(self):
        profiles = [self.profile(i, 5.172e9+t, [.1]*3) for i, t in enumerate([0., .4, .95, 1.5, 2.02])]
        chosen, targets, _ = select_timed_profiles(profiles)
        self.assertEqual([r['selected_step'] for r in targets], [0, 2, 4])
        self.assertAlmostEqual(targets[1]['time_error_s'], -.05, places=5)
        self.assertEqual([r['step'] for r in chosen], [0, 2, 4])

    def test_restart_merge_rejects_conflicting_history_or_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write_segment(name, steps, changed_slip=False, changed_geometry=False):
                path = root / name
                (path / 'profiles').mkdir(parents=True, exist_ok=True)
                with (path / 'profiles.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow(['step', 'time_s', 'file'])
                    for step in steps:
                        writer.writerow([step, step*10., f'profiles/fault_{step}.csv'])
                        with (path / f'profiles/fault_{step}.csv').open('w', newline='') as out:
                            w = csv.writer(out)
                            w.writerow(['fault', 'node', 'step', 'time_s', 'xd_m', 'x_m', 'y_m', 'slip_m', 'V_m_per_s'])
                            for node in range(3):
                                w.writerow([0, node, step, step*10., node*1000., node*1000.,
                                            float(changed_geometry), step*.01+float(changed_slip), 1e-9])
                return path
            first = write_segment('first', [0, 3])
            second = write_segment('second', [3, 7])
            profiles, _, _, duplicates = load_history([second, first])
            self.assertEqual([p.step for p in profiles], [0, 3, 7])
            self.assertEqual(duplicates, 1)
            write_segment('second', [3, 7], changed_slip=True)
            with self.assertRaisesRegex(ValueError, 'Conflicting overlapping'):
                load_history([first, second])
            write_segment('second', [3, 7], changed_geometry=True)
            with self.assertRaisesRegex(ValueError, 'geometry differs'):
                load_history([first, second])


if __name__ == '__main__':
    unittest.main()
