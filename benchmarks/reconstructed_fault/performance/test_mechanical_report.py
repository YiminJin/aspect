"""A restart attempt must not be counted as a returned linear direction."""
import json
from pathlib import Path
import tempfile
import unittest
from mechanical_report import summarize


class FreshChecks(unittest.TestCase):
    def evaluate(self, checks, status=0):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run'
            lines=['Fault linear profile: step=0, newton=0, other_s=1, elapsed=1']
            lines += [f'Fault linear solve: iterations={n}, estimated=.5, fresh={r}, target={t}'
                      for n,r,t in checks]
            path.with_suffix('.log').write_text('\n'.join(lines))
            path.with_suffix('.resources.json').write_text(json.dumps(dict(status=status,wall_seconds=2)))
            return summarize(path)

    def test_replacement_within_budget(self):
        result=self.evaluate([(33,1e11,1.4),(66,6e4,1.4),(74,.98,1.4),(17,.01,.1)])
        self.assertEqual(result['fresh_checks'],2)
        self.assertEqual(result['linear_iterations'],91)
        self.assertEqual(len(result['rejected_fresh_attempts']),2)

    def test_exhausted_is_not_a_return(self):
        with self.assertRaises(AssertionError): self.evaluate([(33,1e11,1.4)])

    def test_changed_target_is_not_replacement(self):
        with self.assertRaises(AssertionError): self.evaluate([(33,1e11,1.4),(74,1,2)])

    def test_reset_budget_is_not_replacement(self):
        with self.assertRaises(AssertionError): self.evaluate([(33,1e11,1.4),(12,1,1.4)])

    def test_failure_status_is_retained(self):
        with self.assertRaises(AssertionError): self.evaluate([(12,1,1.4)],status=1)


if __name__=='__main__': unittest.main()
