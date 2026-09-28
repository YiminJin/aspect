"""Independent weak-load check on the small fixture's exact Q2 test space."""
import unittest
import numpy as np
from analyze_clean_stress_cycle import weak_load

class CleanCycleAnalysis(unittest.TestCase):
    def test_constant_tensor_has_no_free_weak_load(self):
        points,weights=np.polynomial.legendre.leggauss(3)
        points=(points+1)/2;weights=weights/2
        qps={};stress={};h=1/64
        for i in range(16):
            for j in range(64):
                for a in range(3):
                    for b in range(3):
                        key=(str((i,j)),str(3*a+b))
                        qps[key]=dict(cell=key[0],x=(i+points[a])*h,y=(j+points[b])*h-.5,
                                      JxW=weights[a]*weights[b]*h*h)
                        stress[key]=np.array([3.,-7.,11.])
        free,local=weak_load(qps,stress,True)
        self.assertGreater(local,1.)
        self.assertLess(max(abs(x) for x in free.values()),1e-12)

if __name__=='__main__':unittest.main()
