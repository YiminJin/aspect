"""Cheap restart-prefix and sourceable-environment tests; no ASPECT execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from slip_history import HEADER, restore_prefix

HERE=Path(__file__).resolve().parent


class SlipHistory(unittest.TestCase):
    def test_checkpoint_prefix_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cumulative_slip.csv'
            rows=[f'{step},{step*.5},0,{node},{node*2.},{10-node*2.},{step*.1}\n'
                  for step in range(4) for node in range(3)]
            path.write_text(HEADER+'\n'+''.join(rows))
            restore_prefix(path,1)
            self.assertEqual(path.read_text(),HEADER+'\n'+''.join(rows[:6]))
            with path.open('a') as out: out.writelines(rows[6:])
            self.assertEqual(path.read_text(),HEADER+'\n'+''.join(rows))

    def test_missing_checkpoint_fails_without_changing_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cumulative_slip.csv'
            initial=HEADER+'\n0,0,0,0,0,10,0\n'
            path.write_text(initial)
            with self.assertRaises(ValueError): restore_prefix(path,2)
            self.assertEqual(path.read_text(),initial)

    def test_environment(self):
        for shell in ('bash','zsh'):
            env=dict(os.environ,ASPECT_COMPARE_COUPLING='1',ASPECT_FAULT_GMG_HIERARCHY='stale',
                     ASPECT_FAULT_VELOCITY_GMG='1',ASPECT_FAULT_SURFACE_SOLVER='umfpack')
            command='source "$1" && python3 -c "import os,json; print(json.dumps(dict(os.environ)))"'
            result=json.loads(subprocess.check_output(
                [shell,'-c',command,'test',str(HERE/'environment.sh')],env=env))
            self.assertNotIn('ASPECT_COMPARE_COUPLING',result)
            self.assertEqual(result['ASPECT_FAULT_EXPLICIT_B'],'1')
            self.assertEqual(result['OMP_NUM_THREADS'],'1')
            self.assertNotIn('ASPECT_FAULT_GMG_HIERARCHY',result)
            self.assertNotIn('ASPECT_FAULT_VELOCITY_GMG',result)
            self.assertNotIn('ASPECT_FAULT_SURFACE_SOLVER',result)


if __name__=='__main__': unittest.main()
