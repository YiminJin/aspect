"""Cheap reference-only checks; never loads the ASPECT benchmark plugin."""
import json
from pathlib import Path
import unittest

import numpy as np

from reference import Reference, read_parameters
from assess_adjustment import assess

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE.parent/'residual-floor/convergence/space32_dt05/parameters.prm'


class ReferenceChecks(unittest.TestCase):
    def test_configured_activation_and_intact_equilibrium(self):
        model = Reference(read_parameters(PARAMETERS), 128, 6)
        self.assertEqual(model.activation, .1)
        residual = model.residual(np.zeros_like(model.y), np.full_like(model.H0, model.Hc))
        self.assertLess(np.linalg.norm(residual), 1e-13)

    def test_phase_directional_jacobian(self):
        model = Reference(read_parameters(PARAMETERS), 128, 6)
        phi = .2+.1*np.cos(2*np.pi*model.y)
        direction = np.cos(3*np.pi*model.y)
        _, band = model.residual(phi, model.H0, True)
        action = band[1]*direction
        action[:-1] += band[0, 1:]*direction[1:]
        action[1:] += band[2, :-1]*direction[:-1]
        epsilon = 1e-6
        difference = (model.residual(phi+epsilon*direction, model.H0)
                      -model.residual(phi-epsilon*direction, model.H0))/(2*epsilon)
        self.assertLess(np.linalg.norm(action-difference)/np.linalg.norm(action), 1e-8)

    def test_saved_preflight_feedback_and_failed_budget(self):
        report = json.loads((HERE/'preflight/report.json').read_text())
        self.assertEqual(report['first_timestep']['selected_s'], 2.)
        self.assertFalse(report['smoke_budget_pass'])
        self.assertTrue(report['steps'][0]['normalization_pass'])
        self.assertFalse(report['steps'][1]['normalization_pass'])
        for step in report['steps']:
            self.assertTrue(step['containment_pass'])
            self.assertTrue(step['phi_envelope_pass'])
            self.assertGreater(step['F_at_Vmin'], 0)
            self.assertLess(abs(step['residual']), 1e-7)
            self.assertLess(step['full_slip_normalization_error'], 1e-12)
        self.assertGreater(report['steps'][0]['H_max_ratio'], 1.2)
        self.assertGreater(report['feedback']['phi2_minus_phi0_max'], .03)

    def test_saved_reference_accuracy(self):
        coarse = json.loads((HERE/'preflight/report.json').read_text())
        fine = json.loads((HERE/'accuracy-check/report.json').read_text())
        for k in range(3):
            a = np.loadtxt(HERE/f'preflight/phase-{k}.csv', delimiter=',', skiprows=1)
            b = np.loadtxt(HERE/f'accuracy-check/phase-{k}.csv', delimiter=',', skiprows=1)
            self.assertLess(np.max(abs(a[:, 1]-np.interp(a[:, 0], b[:, 0], b[:, 1]))), 1e-6)
        for a, b in zip(coarse['steps'], fine['steps']):
            for field in ('Ih', 'V', 'C', 'H_max_ratio'):
                self.assertLess(abs(a[field]/b[field]-1), 1e-5)
        self.assertFalse(fine['smoke_budget_pass'])

    def test_bounded_candidate_decision(self):
        rejected = json.loads((HERE/'ramp-0045/report.json').read_text())
        self.assertFalse(rejected['smoke_budget_pass'])
        for suffix in ('', '-conditional'):
            result = assess(HERE/f'ramp-00225{suffix}', HERE/f'ramp-00225{suffix}-accuracy')
            self.assertTrue(result['passed'])
            self.assertGreaterEqual(min(result['signal_to_noise'].values()), 10)

    def test_conditional_initialization_is_not_re_equilibrated(self):
        initial = HERE.parent/'nonuniform/global-accumulator/k1_short'
        model = Reference(read_parameters(PARAMETERS), 128, 6, .00225, initial)
        self.assertGreater(min(model.dy), 1e-8)
        np.testing.assert_array_equal(model.H0,
            model.history_values[np.searchsorted(model.history_edges[1:], model.qy)])
        report = json.loads((HERE/'ramp-00225-conditional/report.json').read_text())
        self.assertEqual(report['initial']['phase_residual_history'], [])
        self.assertAlmostEqual(report['initial']['retained_histories']['q'], 1500)
        self.assertAlmostEqual(report['initial']['retained_histories']['Theta'], 200)
        self.assertGreater(report['steps'][0]['phi_change_max'], 1e-4)

    def test_explicit_sequence_and_empty_prefix(self):
        saved = json.loads((HERE/'ramp-00225/report.json').read_text())
        sequence = [dict(time=s['time_s'],dt=s['dt_s'],U=s['U']) for s in saved['steps']]
        model = Reference(read_parameters(PARAMETERS),2048,6,.00225)
        report, _ = model.run(sequence)
        for actual, expected in zip(report['steps'],saved['steps']):
            self.assertEqual(actual['time_s'],expected['time_s'])
            self.assertEqual(actual['dt_s'],expected['dt_s'])
            self.assertEqual(actual['U'],expected['U'])
            self.assertAlmostEqual(actual['V'],expected['V'],places=13)
        empty, _ = model.run([])
        self.assertEqual(empty['completed_real_steps'],0)
        self.assertFalse(empty['smoke_budget_pass'])


if __name__ == '__main__':
    unittest.main()
