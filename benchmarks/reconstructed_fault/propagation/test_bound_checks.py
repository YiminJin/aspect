"""Reject false qualification from solver-only or numerically unresolved evidence."""
from copy import deepcopy
from pathlib import Path
import unittest
from check_bound_evidence import assess_bounds, rows

class BoundChecks(unittest.TestCase):
    def setUp(self):
        self.data=rows(Path(__file__).with_name('bounds_evidence')/'bounds_padded_final.bounds.csv','stage')

    def test_saved_trajectory(self):
        self.assertTrue(assess_bounds(self.data)['restricted_prescribed_H_pass'])

    def test_significant_violations(self):
        for key,value in [('node_min',-1e-3),('node_max',1.001),('cpdi_min',-1e-3),
                          ('free_residual',1e-3),('dual_violation',1e-3),('complementarity',1e-3),
                          ('H_increment_min',-1),('equality_residual',1),('correction',float('nan'))]:
            with self.subTest(key=key):
                data=deepcopy(self.data);data[3][key]=value
                with self.assertRaises(AssertionError):assess_bounds(data)

    def test_healing_is_separate_but_disqualifies(self):
        self.data[3]['healing']=1e-4
        result=assess_bounds(self.data)
        self.assertTrue(result['box_kkt_pass'])
        self.assertFalse(result['no_resolved_healing'])
        self.assertFalse(result['restricted_prescribed_H_pass'])

    def test_unresolved_repeat_disqualifies(self):
        self.data[4]['change']=1e-4
        result=assess_bounds(self.data)
        self.assertFalse(result['repeat_consistent'])
        self.assertFalse(result['restricted_prescribed_H_pass'])

if __name__=='__main__':unittest.main()
