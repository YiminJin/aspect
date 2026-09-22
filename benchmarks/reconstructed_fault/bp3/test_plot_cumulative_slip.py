"""Small plotting-data checks; no ASPECT simulation or graphical backend needed."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_cumulative_slip import HEADER, Profile, read_profiles, select_profiles


class SlipContours(unittest.TestCase):
    def profile(self, step, time, slip):
        return Profile(step, time, np.arange(3), np.array([50000., 20000., 0.]), np.array(slip))

    def test_increment_not_time_or_change_of_maximum(self):
        profiles = [self.profile(0, 0., [0., 0., 0.]),
                    self.profile(1, 1., [0., 1., 0.]),
                    self.profile(2, 2., [0., 1., 0.15]),
                    self.profile(3, 100., [0., 1., 0.16])]
        _, chosen, _ = select_profiles(profiles, increment=.1, seismic_rate=10.)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 2, 3])

    def test_seismic_classification_uses_full_fault_and_keeps_transition(self):
        profiles = [self.profile(0, 0., [0., 0., 0.]),
                    self.profile(1, 100., [.01, .01, .01]),
                    self.profile(2, 200., [.02, .02, .02]),
                    self.profile(3, 201., [.03, .021, .021]),
                    self.profile(4, 301., [.04, .022, .022])]
        _, chosen, summary = select_profiles(profiles, increment=1.)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 2, 3, 4])
        self.assertEqual(chosen[3]['regime'], 'coseismic')
        self.assertEqual(summary['coseismic_accepted_profiles'], 1)
        self.assertAlmostEqual(chosen[3]['max_rate_m_s'], .01)

    def test_skips_small_changes_but_keeps_last_and_exact_recorded_values(self):
        profiles = [self.profile(i, i*1000., [i*.04, i*.04, i*.02]) for i in range(7)]
        xd, chosen, _ = select_profiles(profiles, increment=.1)
        self.assertEqual([p['step'] for p in chosen], [0, 1, 4, 6])
        np.testing.assert_array_equal(xd, [0., 20000.])
        np.testing.assert_array_equal(chosen[-1]['slip'], [.12, .24])

    def test_stream_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'slip.csv'
            rows = [[k, k*100, 0, n, n*10, 20-n*10, k*.1] for k in range(3) for n in range(3)]
            def write(data):
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(HEADER)
                    writer.writerows(data)
            write(rows)
            self.assertEqual(len(list(read_profiles(path))), 3)
            write(rows[:3] + rows)
            with self.assertWarnsRegex(UserWarning, 'identical duplicate'):
                self.assertEqual(len(list(read_profiles(path))), 3)
            conflicting = [r.copy() for r in rows[:3]]
            conflicting[0][-1] = 4.
            write(conflicting + rows)
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                list(read_profiles(path))
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete profile'):
                list(read_profiles(path))
            write(rows[:3] + rows[6:])
            with self.assertRaisesRegex(ValueError, 'consecutive steps'):
                list(read_profiles(path))
            write(rows + rows[3:6])
            with self.assertRaisesRegex(ValueError, 'consecutive steps'):
                list(read_profiles(path))


if __name__ == '__main__':
    unittest.main()
