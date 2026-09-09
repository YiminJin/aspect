"""Isolated checks of the K1 diagnosis, not a change to production acceptance."""
import math
import unittest

import numpy as np

from analyze_line_search import friction_remainder, strong_rms


class DiagnosisChecks(unittest.TestCase):
    def test_gauss_585_mode_has_zero_constant_and_linear_moments(self):
        coordinates, weights = np.polynomial.legendre.leggauss(3)
        values = 9.2830745568*(1-3*coordinates**2)
        self.assertLess(abs(weights@values), 1e-13)
        self.assertLess(abs(weights@(coordinates*values)), 1e-13)
        np.testing.assert_allclose(values/values[1], [-.8, 1., -.8], atol=1e-14)
        # The same polynomial is not annihilated by equal-weight particle
        # sampling at cell thirds. This does NOT interpolate actual stress.
        particle_points = np.array([-2/3, 0., 2/3])
        self.assertGreater(np.mean(1-3*particle_points**2), .1)

    def test_consistent_mass_norm_recovers_constant_density(self):
        mass = np.array([[2., 1.], [1., 2.]])/6
        self.assertAlmostEqual(strong_rms(mass@np.array([7., 7.]), mass), 7.)

    def test_friction_remainder_is_second_order(self):
        V, direction = .00031731792608273, 9.09883e-6
        errors = [float(friction_remainder(V, a*direction)) for a in [1e-2, 5e-3, 2.5e-3]]
        for first, second in zip(errors, errors[1:]):
            self.assertAlmostEqual(first/second, 4., delta=4e-4)

    def test_exact_newton_direction_still_exhausts_tiny_surface_scale(self):
        # Two-block local model, with the surface initially exactly balanced.
        # The bulk decreases exactly as 1-alpha and the surface's first
        # derivative vanishes. Only the physical friction curvature remains.
        V, direction = .00031731792608273, 9.09883e-6
        scale = 1.6161383225535839e-5
        for reduction in range(6):
            alpha = (2/3)**reduction
            surface = float(friction_remainder(V, alpha*direction))
            merit = .5*((1-alpha)**2+(surface/scale)**2)
            self.assertGreater(merit, (1-1e-4*alpha)*.5)
        # A characteristic (not roundoff-suppressed) surface scale permits
        # the same direction. This is proposal evidence, not an approved
        # production scale or a claim of a completed FE timestep.
        characteristic = scale/math.sqrt(np.finfo(float).eps)
        merit = .5*(float(friction_remainder(V, direction))/characteristic)**2
        self.assertLess(merit, (1-1e-4)*.5)


if __name__ == "__main__":
    unittest.main()
