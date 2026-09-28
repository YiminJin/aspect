"""Cheap checks of the independent Cartesian weak-load integrator."""
import unittest
import numpy as np
from analyze_inclined_moment import independent_jump


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q,w=np.polynomial.legendre.leggauss(3);q=(q+1)/2;w=w/2
        x=(np.arange(64)[:,None]+q)/64-.5
        weights=np.tile(w/64,64)
        xx,yy=np.meshgrid(x.ravel(),x.ravel(),indexing='ij')
        cls.data=np.column_stack([xx.ravel(),yy.ravel(),np.outer(weights,weights).ravel()])

    def test_constant_tensor(self):
        tensor=np.tile([2.,-2.,3.],(len(self.data),1))
        self.assertLess(independent_jump(self.data,tensor),2e-14)

    def test_linear_tensor(self):
        # div([[x,y],[y,-x]])=(2,0): integration by parts gives -2 int N.
        x,y=self.data[:,:2].T;tensor=np.column_stack([x,-x,y])
        m=np.array([2/3 if i%2 else 1/3 for i in range(1,128)])/64
        expected=2*np.linalg.norm(np.outer(m,m))
        self.assertAlmostEqual(independent_jump(self.data,tensor),expected,places=13)


if __name__=='__main__':unittest.main()
