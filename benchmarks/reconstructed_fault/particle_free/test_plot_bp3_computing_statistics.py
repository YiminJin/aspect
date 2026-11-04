"""Checks for acceptance, cumulative timers, and restart accounting."""
from pathlib import Path
import tempfile
import unittest

from plot_bp3_computing_statistics import accepted_rows, parse_log, timer_windows


class LogStatisticsTests(unittest.TestCase):
    def test_timer_differences_stay_within_invocation(self):
        run = dict(segment='resume', timers=[
            dict(step=100, including_restarts_s=10000),
            dict(step=102, including_restarts_s=10120)], attempts=[
                dict(step=100, accepted=True), dict(step=101, accepted=False),
                dict(step=101, accepted=True), dict(step=102, accepted=True)])
        row, = timer_windows(run)
        self.assertEqual(row['wall_s'], 120)
        self.assertEqual(row['accepted_steps'], 2)
        self.assertEqual(row['wall_s_per_accepted_step'], 60)

    def test_unaccepted_tail_and_cumulative_tables(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.txt'
            p.write_text('''running with 160 MPI processes
"Nonlinear solver tolerance": "1e-8"
*** Timestep 0:  t=0 seconds, dt=0 seconds
Fault linear solve: iterations=3, fresh=2e-9, target=3e-9
Relative nonlinear residuals (bulk, fault) after nonlinear iteration  0: 1e-10, 2e-10
BP3 accepted state 0
| Total wallclock time elapsed since start       |  10s |            |
| Fault: condensed linear solve | 1 | 6s | 60% |
-- Total wallclock time elapsed including restarts: 10010s
*** Timestep 1:  t=1 seconds, dt=1 seconds
Fault linear solve: iterations=4, fresh=2e-9, target=3e-9
Relative nonlinear residuals (bulk, fault) after nonlinear iteration  0: 1e-10, 2e-10
BP3 accepted state 1
| Total wallclock time elapsed since start       |  20s |            |
| Fault: condensed linear solve | 2 | 12s | 60% |
-- Total wallclock time elapsed including restarts: 10020s
*** Timestep 2:  t=2 seconds, dt=1 seconds
Relative nonlinear residuals (bulk, fault) after nonlinear iteration  0: 1e-10, 2e-10
''')
            run = parse_log(p)
            self.assertEqual([a['step'] for a in accepted_rows(run)], [0, 1])
            self.assertFalse(run['attempts'][-1]['accepted'])
            self.assertEqual(run['timers'][-1]['sections'][0]['wall_s'], 12)
            self.assertEqual(timer_windows(run)[0]['wall_s_per_accepted_step'], 10)
            self.assertEqual(accepted_rows(run)[-1]['linear_iterations'], 4)
            self.assertEqual(accepted_rows(run)[-1]['residual_evaluations'], 1)

    def test_decreasing_clock_rejected(self):
        run = dict(segment='bad', timers=[dict(step=0, including_restarts_s=20),
                                         dict(step=1, including_restarts_s=10)],
                   attempts=[dict(step=1, accepted=True)])
        with self.assertRaises(ValueError):
            timer_windows(run)


if __name__ == '__main__':
    unittest.main()
