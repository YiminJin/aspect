"""Cheap consistency checks of the saved K4.1 reference-only calculations."""
import importlib.util
import json
import sys
import unittest

import numpy as np

from reference_check import HERE, Y, compare

spec=importlib.util.spec_from_file_location('k1_scalar',HERE.parent/'reference.py')
k1=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=k1
spec.loader.exec_module(k1)


class Checks(unittest.TestCase):
    def test_scalar_history_timeline_against_k1(self):
        for name in ('ell0','half'):
            path=HERE/'k41'/f'{name}-4096'
            initial=json.loads((path/'initialization.json').read_text())['retained_initial']
            old=k1.Histories(initial['q'],initial['C'],initial['Theta'])
            for row in json.loads((path/'dt0.125.json').read_text()):
                step=row['step']
                value,next_old=k1.solve_step(old,.125 if step else 2.,row['U'],
                                            initial['Ih'],initial['Ih'],initial=step==0)
                np.testing.assert_allclose([value['V'],value['stress'],value['cohesive']],
                                           [row['V'],row['q'],row['C']],rtol=1e-12,atol=1e-12)
                self.assertAlmostEqual(next_old.theta,row['Theta'],places=10)
                if step==0: self.assertEqual(old,next_old)
                old=next_old

    def test_calibration_support_and_history_localization(self):
        for name,m in (('ell0',128),('half',256)):
            path=HERE/'k41'/f'{name}-4096'
            data=json.loads((path/'initialization.json').read_text())
            self.assertAlmostEqual(data['m'],m)
            self.assertEqual(data['activation'],.1)
            self.assertTrue(data['guards_pass'])
            phi=np.load(path/'profile.npz')['phi']
            np.testing.assert_allclose(phi,phi[::-1],rtol=0,atol=1e-11)
            for row in json.loads((path/'dt0.125.json').read_text()):
                self.assertEqual(row['history_integral'],0)
                self.assertLess(row['supported_normalization_error'],1e-4)
                self.assertLess(abs(row['crack_integral']-row['V']),1e-14)

    def test_comparison_and_stopping(self):
        path=HERE/'k41/ell0-4096'
        rows=json.loads((path/'dt0.125.json').read_text())
        velocity=np.load(path/'dt0.125.npz')['velocity']
        self.assertEqual(compare(rows,rows,velocity,velocity)['max_ratio'],0)
        for row,u in zip(rows,velocity):
            self.assertAlmostEqual(u[0],-row['U']/2,places=12)
            self.assertAlmostEqual(u[-1],row['U']/2,places=12)
        decision=json.loads((HERE/'k41/decision.json').read_text())
        self.assertFalse(decision['ready'])
        self.assertIsNone(decision['selected_common_dt_s'])
        self.assertTrue(all(x['max_ratio']<1 for x in decision['spatial'].values()))


if __name__=='__main__': unittest.main()
