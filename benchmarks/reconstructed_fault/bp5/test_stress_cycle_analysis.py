"""Small offline checks; no simulation or checkpoint mutation."""
import unittest
import numpy as np
from analyze_stress_cycle_cells import q2, stats


class StressCycleAnalysis(unittest.TestCase):
    def test_q2_recovery_with_shuffled_support_order(self):
        support=np.array([(x,y) for x in (0.,.5,1.) for y in (0.,.5,1.)])[[4,8,0,6,2,1,7,3,5]]
        gauss=np.polynomial.legendre.leggauss(3)[0]/2+.5
        quadrature=np.array([(x,y) for x in gauss for y in gauss])
        parents=np.array([(x,y) for x in (.16,.49,.83) for y in (.19,.52,.85)])
        def values(x):
            a,b=x.T
            return np.column_stack((3+2*a-4*b+7*a*a*b*b,-7+5*a*a*b,11-8*a*b*b))
        recovered=np.linalg.solve(q2(quadrature,support),values(quadrature))
        np.testing.assert_allclose(q2(parents,support)@recovered,values(parents),rtol=1e-14,atol=1e-14)

    def test_distinct_measures(self):
        self.assertEqual(stats([1.,3.])['mean'],2.)
        self.assertEqual(stats([1.,3.],[3.,1.])['mean'],1.5)
        with self.assertRaises(AssertionError):stats([np.nan])


if __name__=='__main__':unittest.main()
