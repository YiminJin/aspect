"""Offline profile-index/geometry checks, with no simulation or plotting backend."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_fault_evolution import YEAR, edges, read_profiles


class FaultEvolution(unittest.TestCase):
    def make_data(self, run):
        header = ['fault', 'node', 'step', 'time_s', 'xd_m', 'x_m', 'y_m',
                  'slip_m', 'V_m_per_s', 'Theta_s', 'q_weak_Pa', 'sigma_n_weak_Pa']
        for step in range(2):
            with (run/f'fault_{step}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                for node in range(3):
                    writer.writerow([0, node, step, step*YEAR, 2000-node*1000, node, node,
                                     step*(node+1), 1e-9, YEAR, 3e7, 5e7])
        with (run/'profiles.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step', 'time_s', 'file', 'max_slip_change_m'])
            writer.writerow([0, 0, 'fault_0.csv', 0])
            writer.writerow([0, 0, 'fault_0.csv', 0])
            writer.writerow([1, YEAR, 'fault_1.csv', 1])

    def test_saved_times_sorting_and_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_data(run)
            geometry, times, steps, fields, duplicates, missing = read_profiles(run)
            np.testing.assert_array_equal(times, [0, 1])
            np.testing.assert_array_equal(steps, [0, 1])
            np.testing.assert_array_equal(geometry[:, 1], [0, 1000, 2000])
            np.testing.assert_array_equal(fields['slip_m'][1], [3, 2, 1])
            self.assertEqual(duplicates, 1)
            self.assertEqual(missing, [])

    def test_missing_file_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_data(run)
            with (run/'profiles.csv').open('a') as stream:
                stream.write(f'2,{2*YEAR},absent.csv,1\n')
            with self.assertRaisesRegex(ValueError, 'Missing 1'):
                read_profiles(run)
            with self.assertWarns(UserWarning):
                result = read_profiles(run, skip_missing=True)
            self.assertEqual(result[-1], ['absent.csv'])
            self.assertEqual(result[1][-1], 1)

    def test_changed_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_data(run)
            path = run/'fault_1.csv'
            text = path.read_text().replace(',2000,', ',2001,')
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'Changed geometry'):
                read_profiles(run)

    def test_irregular_time_bins(self):
        np.testing.assert_array_equal(edges(np.array([0., 1., 5.])), [0., .5, 3., 5.])


if __name__ == '__main__':
    unittest.main()
