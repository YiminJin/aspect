import unittest
import numpy as np

from diagnose_endpoint import align_history, cell_average_transfer


class EndpointAudit(unittest.TestCase):
    def test_history_join_uses_ids_not_particle_order(self):
        a=np.array([(2,20.),(1,10.)],dtype=[('id',int),('tau_xy',float)])
        b=np.array([(1,3.),(2,4.)],dtype=a.dtype)
        current,previous=align_history(a,b)
        np.testing.assert_array_equal(current['tau_xy'],[10.,20.])
        np.testing.assert_array_equal(previous['tau_xy'],[3.,4.])
        with self.assertRaises(ValueError):
            align_history(a,b[:1])

    def fixture(self):
        g=(np.polynomial.legendre.leggauss(3)[0]+1)/2
        points=[]
        for i in range(2):
            for j in range(8):
                points.extend([((i+x)*.125,(j+y)/8-.5) for x in g for y in g])
        bulk=np.array(points,dtype=[('x',float),('y',float)])
        a=np.zeros(len(bulk),dtype=[('x',float),('y',float),('tau_xx',float),('tau_yy',float),('tau_xy',float)])
        a['x']=bulk['x'];a['y']=bulk['y'];a['tau_xy']=1500
        return a,bulk

    def test_constant_history_survives_shared_node_writes(self):
        a,b=self.fixture()
        nodal,_,counts=cell_average_transfer(a,a,b,2)
        np.testing.assert_array_equal(nodal[:,:,2],1500)
        np.testing.assert_array_equal(counts,9)

    def test_nonconstant_shared_node_depends_on_last_cell(self):
        a,b=self.fixture()
        a['tau_xy']=np.repeat(np.arange(16),9)
        forward,_,_=cell_average_transfer(a,a,b,2)
        reverse,_,_=cell_average_transfer(a,a,b.reshape(-1,9)[::-1].ravel(),2)
        self.assertEqual(forward[2,2,2],9)
        self.assertEqual(reverse[2,2,2],0)
        # A cell-center DoF is unshared and independent of traversal.
        self.assertEqual(forward[1,1,2],reverse[1,1,2])


if __name__=='__main__':
    unittest.main()
