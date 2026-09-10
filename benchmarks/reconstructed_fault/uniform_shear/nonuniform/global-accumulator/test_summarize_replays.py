"""Small analytic checks for the bounded replay comparison (no ASPECT run)."""
import unittest

import numpy as np

from summarize_replays import compare, q1_errors, q1_mean


class ReplayComparison(unittest.TestCase):
    def test_piecewise_linear_moments(self):
        x = np.array([0., .2, 1.])
        self.assertAlmostEqual(q1_mean(x, 2*x+1), 2.)
        result = q1_errors(x, 2*x+1)
        self.assertAlmostEqual(result['rms'], np.sqrt(13/3))
        self.assertAlmostEqual(result['anomaly_rms'], 1/np.sqrt(3))
        self.assertAlmostEqual(result['anomaly_maximum'], 1.)

    def test_initialization_offset_is_not_mechanical_change(self):
        def case(x, offset, evolution):
            rows = {}
            names = ('particle_q_Q1', 'V', 'Theta', 'C_retained', 'slip')
            for t in (0., .5, 1.):
                surface = np.zeros(len(x), dtype=[(name, float) for name in names])
                for name in names:
                    surface[name] = x+offset+evolution*t
                rows[t] = dict(x=x, surface=surface)
            return rows
        rows = compare(case(np.array([0., .5, 1.]), 2., 3.),
                       case(np.array([0., .2, .6, 1.]), 0., 1.))
        for row in rows:
            for field in row['fields'].values():
                self.assertAlmostEqual(field['total']['mean'], 2.+2.*row['time_s'])
                self.assertAlmostEqual(field['total']['anomaly_rms'], 0.)
                self.assertAlmostEqual(field['change_from_initial_difference']['mean'], 2.*row['time_s'])


if __name__ == '__main__':
    unittest.main()
