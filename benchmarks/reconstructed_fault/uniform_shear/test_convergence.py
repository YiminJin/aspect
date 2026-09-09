"""Independent integration and temporal-reference checks, not ASPECT tests."""
import unittest

import numpy as np
from scipy.integrate import quad

from analyze import h
from analyze_convergence import profile_primitive, scalar_limit
from reference import Histories, ideal_profile


class ConvergenceChecks(unittest.TestCase):
    def test_profile_primitive_retains_full_domain(self):
        ys = np.array([-.5, -.2, 0., .2, .5])
        phi = np.array([0., .2, .6, .2, 0.])
        samples = np.array([-.5, -.3, 0., .13, .5])
        values, integral = profile_primitive(ys, phi, samples)
        for y, value in zip(samples, values):
            expected = quad(lambda z: float(h(np.interp(z, ys, phi))), -.5, y,
                            points=[v for v in ys if -.5 < v < y], epsabs=1e-10)[0]
            self.assertAlmostEqual(value, expected, delta=1e-9)
        self.assertEqual(values[0], 0.)
        self.assertAlmostEqual(values[-1], integral, delta=1e-12)

    def test_constant_profile_and_zero_tail(self):
        ys = np.array([-.5, 0., .5])
        points = np.array([-.5, -.1, .5])
        for value in [0., .3]:
            primitive, integral = profile_primitive(ys, np.full(3,value), points)
            np.testing.assert_allclose(primitive, h(value)*(points+.5), atol=1e-12)
            self.assertAlmostEqual(integral, h(value), delta=1e-12)

    def test_temporal_limit_does_not_mutate_initial_histories(self):
        initial = Histories()
        integral = ideal_profile()[1]
        coarse = scalar_limit(integral, initial, .125)
        fine = scalar_limit(integral, initial, .0625)
        finer = scalar_limit(integral, initial, .03125)
        self.assertEqual(initial, Histories())
        self.assertEqual([r['time_s'] for r in fine], [2.,4.,6.])
        self.assertGreater(abs(fine[0]['Theta']-initial.theta), 1.)
        self.assertLess(abs(finer[-1]['Theta']-fine[-1]['Theta']),
                        abs(fine[-1]['Theta']-coarse[-1]['Theta']))


if __name__ == '__main__':
    unittest.main()
