"""Cheap gate regressions from saved K1 data, in a disposable copy only."""
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from check_smoke import check


class SmokeGate(unittest.TestCase):
    def test_mixed_phase_residual_gate(self):
        source=Path(__file__).resolve().parent/'spatial0375_n128_f32_periodic'
        with tempfile.TemporaryDirectory(prefix='k3-mixed-gate-') as temporary:
            path=Path(temporary)
            for step in (0,1):
                for name in ('surface','phase','bulk','particles','segments'):
                    shutil.copy2(source/f'{name}_{step}.csv',path/f'{name}_{step}.csv')
            # A saved-state parser test, not a rerun or alteration of production
            # evidence. Verify both a permitted precision floor and a larger
            # residual, independently of the relative field in the exit file.
            np.savetxt(path/'phase_probe_1.csv',[[1e-7,1e-7,1,1,1e-12]],delimiter=',',
                       header='R_old_phi_H0,R_old_phi_Hprevious,forced_failure_restored,stable_ids,roundoff',comments='')
            for residual,expected in ((4e-13,True),(2e-12,False)):
                np.savetxt(path/'phase_probe_exit_1.csv',[[residual,residual/1e-7,1e-12,1e-12]],delimiter=',',
                           header='R_new_phi_Hprevious,relative,absolute_target,roundoff',comments='')
                self.assertEqual(check(path,1)['checks']['phase_converged'],expected)

    def test_initial_and_nonzero_history_contribution(self):
        source=Path(__file__).resolve().parent.parent/'nonuniform/global-accumulator/k1_short'
        with tempfile.TemporaryDirectory(prefix='k3-gate-') as temporary:
            path=Path(temporary)
            for name in ('surface','phase','bulk','particles','segments'):
                shutil.copy2(source/f'{name}_0.csv',path/f'{name}_0.csv')
            baseline=check(path,0)
            self.assertTrue(baseline['passed'])
            # A history perturbation must fail the total normalization check
            # even though phi and its omitted h fraction are unchanged.
            bulk=np.genfromtxt(path/'bulk_0.csv',delimiter=',',names=True)
            names=bulk.dtype.names
            bulk['history'][bulk['active']>0]=1e-5
            np.savetxt(path/'bulk_0.csv',np.column_stack([bulk[n] for n in names]),
                       delimiter=',',header=','.join(names),comments='')
            altered=check(path,0)
            self.assertTrue(altered['checks']['containment'])
            self.assertFalse(altered['checks']['supported_normalization'])
            self.assertFalse(altered['passed'])


if __name__=='__main__':
    unittest.main()
