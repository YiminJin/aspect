"""Small reference checks; these are not production benchmark comparisons."""
import math
import unittest

from reference import Histories, ideal_profile, residual, solve_step


class ReferenceChecks(unittest.TestCase):
    def test_initial_zero_previous_profile_has_no_interior_root(self):
        _, integral = ideal_profile()
        with self.assertRaisesRegex(ValueError, "No interior root"):
            solve_step(Histories(), 2., 1e-4, integral, 0., initial=True)
        samples = [1e-12, 1e-8, 1e-6, 1e-4, 1e-2]
        values = [residual(v, Histories(), 2., 1e-4, integral, 0.) for v in samples]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertLess(values[0], -32000.)

    def test_initial_evaluation_is_not_history_evolution(self):
        _, integral = ideal_profile()
        supplied = Histories()
        evaluated, retained = solve_step(supplied, 2., 1e-4, integral,
                                          integral, initial=True)
        self.assertIs(retained, supplied)
        self.assertNotEqual(evaluated["stress"], supplied.stress)
        self.assertNotEqual(evaluated["cohesive"], supplied.cohesive)

    def test_real_steps_advance_the_reference_own_history(self):
        _, integral = ideal_profile()
        previous = Histories()
        for loading in (1.1e-4, 1.2e-4, 1.2e-4):
            evaluated, current = solve_step(previous, 2., loading, integral, integral)
            v = evaluated["V"]
            self.assertLess(abs(residual(v, previous, 2., loading, integral, integral)), 1e-7)
            # Integrating dTheta/dt = 1 - V Theta / D_c at constant V.
            steady = 0.001/v
            expected = steady + (previous.theta - steady)*math.exp(-2./steady)
            self.assertAlmostEqual(current.theta, expected, delta=1e-12)
            self.assertGreater(abs(current.theta - previous.theta), 100.*2e-8)
            previous = current


if __name__ == "__main__":
    unittest.main()
