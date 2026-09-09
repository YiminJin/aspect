"""Checks of the separate diagnostic; never relax the approved reference."""
import math
import unittest

from reference import Histories, solve_step
from support_resolution import retained_step


class RetainedFractionChecks(unittest.TestCase):
    def test_unit_retention_is_the_approved_reference(self):
        histories = Histories(1500.,313.5117601471167,200.)
        for initial in (True,False):
            expected, history = solve_step(histories,2.,1e-4,108.0722382,108.0722382,initial)
            actual, diagnostic_history = retained_step(histories,2.,1e-4,108.0722382,1.,initial)
            for field in expected:
                self.assertAlmostEqual(actual[field],expected[field],delta=1e-10)
            self.assertEqual(history,diagnostic_history)

    def test_full_integral_stays_in_cohesive_law(self):
        initial = Histories(1500.,313.5117601471167,200.)
        full, dt = 108.0722382, 2.
        response, history = retained_step(initial,dt,1e-4,full,.999943508245,True)
        beta, kappa = math.exp(-dt/100), -1e8*math.expm1(-dt/100)
        self.assertAlmostEqual(response["cohesive"],beta*initial.cohesive+kappa*response["V"]/full,delta=1e-12)
        self.assertIs(history,initial)

    def test_self_advanced_real_state_matches_aging_solution(self):
        initial = Histories(1500.,313.5117601471167,200.)
        for load in (1.1e-4,1.2e-4):
            response, history = retained_step(initial,2.,load,108.0722382,.999943508245)
            steady = .001/response["V"]
            expected = steady+(initial.theta-steady)*math.exp(-2./steady)
            self.assertAlmostEqual(history.theta,expected,delta=1e-12)
            initial = history


if __name__ == "__main__":
    unittest.main()
