"""Checks that the DG trace diagnostic preserves, rather than averages, jumps."""
import unittest
import numpy as np
from analyze import traces


class Traces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q = (np.polynomial.legendre.leggauss(3)[0]+1)/2
        x = ((np.arange(64)[:,None]+q)/64-.5).ravel()
        xx,yy = np.meshgrid(x,x,indexing='ij')
        cls.data = np.column_stack((xx.ravel(),yy.ravel()))

    def test_continuous_quadratic(self):
        x,y=self.data.T
        v=np.column_stack((x*x+y, x*y+y*y, 2+x*x*y*y))
        self.assertLess(traces(self.data,v)['max_abs_component_face_jump'],1e-11)

    def test_discontinuous_cell_offsets(self):
        x,y=self.data.T
        v=np.column_stack((x+2*np.floor((x+.5)*64), y*0, x*0))
        result=traces(self.data,v)
        self.assertAlmostEqual(result['x_faces'],2.,places=10)
        self.assertLess(result['y_faces'],1e-10)
        self.assertAlmostEqual(result['interior_max_component'],2.,places=10)


if __name__=='__main__': unittest.main()
