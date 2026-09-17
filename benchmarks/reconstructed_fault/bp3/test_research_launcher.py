"""Preparation-only checks: default GMG, explicit backends, no simulation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class ResearchLauncher(unittest.TestCase):
    def test_preconditioner_selection(self):
        for selection in (None, 'gmg', 'amg'):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory(prefix='bp3-launcher-') as temp:
                output = Path(temp)/'prepared'
                command = [sys.executable, str(HERE/'run_research.py'),
                           '--prepare-only', '--output', str(output)]
                if selection:
                    command += ['--velocity-preconditioner', selection]
                # Explicit AMG must not inherit an externally enabled GMG or
                # reference-comparison callback from a previous experiment.
                environment = dict(os.environ, ASPECT_FAULT_VELOCITY_GMG='1',
                                   ASPECT_FAULT_GMG_HIERARCHY='1',
                                   ASPECT_FAULT_COMPARE_COUPLING='1')
                subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
                record = json.loads((output/'provenance.json').read_text())
                flags = record['environment']
                for name in ('ASPECT_FAULT_VELOCITY_GMG', 'ASPECT_FAULT_GMG_HIERARCHY'):
                    if selection == 'amg':
                        self.assertNotIn(name, flags)
                    else:
                        self.assertEqual(flags[name], '1')
                self.assertNotIn('ASPECT_FAULT_COMPARE_COUPLING', flags)
                self.assertIn('bp3_modified_wide.prm', (output/'run.prm').read_text())
                self.assertFalse((output/'run.log').exists())


if __name__ == '__main__':
    unittest.main()
