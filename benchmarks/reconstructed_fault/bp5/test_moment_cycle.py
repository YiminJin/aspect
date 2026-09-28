"""Mathematical targets, independent of the C++ publication implementation."""
import unittest
import numpy as np

class MomentTargets(unittest.TestCase):
    def test_constant_and_linear_moments(self):
        x,w=np.polynomial.legendre.leggauss(3);x=x/2;w=w/2
        # Orthogonal projection via weighted QR, not the implementation's 2x2 formula.
        design=np.column_stack((np.ones(3),x))
        for field in (np.array([3.,3.,3.]),2+5*x,np.array([494.1645261204,507.2960583940,494.1645261204])):
            coeff=np.linalg.lstsq(design*np.sqrt(w[:,None]),field*np.sqrt(w),rcond=None)[0]
            projected=design@coeff
            np.testing.assert_allclose(design.T@(w*(projected-field)),0,atol=2e-13)
        self.assertAlmostEqual(coeff[0],500.0007626864444,places=9)
        self.assertAlmostEqual(coeff[1],0.,places=10)

    def test_reported_mode_has_zero_resolved_shear_moments(self):
        x,w=np.polynomial.legendre.leggauss(3)
        mode=np.array([-1.,1.25,-1.])
        self.assertLess(abs(w@mode),1e-15)
        self.assertLess(abs(w@(x*mode)),1e-15)
        self.assertGreater(w@(mode*mode),1.)

    def test_norms_need_cross_term(self):
        a=np.array([1.,2.]);b=np.array([-1.,3.])
        self.assertAlmostEqual(np.dot(a+b,a+b),np.dot(a,a)+np.dot(b,b)+2*np.dot(a,b))

if __name__=='__main__':unittest.main()
