import unittest
import numpy as np
from compare_cases import interpolate_q2
from measure_case import project


class Diagnostics(unittest.TestCase):
    def test_exact_q2_sampling_preserves_cellwise_polynomial(self):
        g=(np.polynomial.legendre.leggauss(3)[0]+1)/2
        x=np.ravel((np.arange(2)[:,None]+g)*.125)
        y=np.ravel((np.arange(8)[:,None]+g)/8-.5)
        xx,yy=np.meshgrid(x,y,indexing='ij')
        data=np.zeros(xx.size,dtype=[('x',float),('y',float)])
        data['x']=xx.ravel(); data['y']=yy.ravel()
        polynomial=lambda x,y: 3+2*x-7*y+5*x*x*y*y
        points=np.random.default_rng(42).uniform([0,-.5],[.25,.5],size=(200,2))
        actual=interpolate_q2(data,polynomial(data['x'],data['y']),points)
        np.testing.assert_allclose(actual,polynomial(points[:,0],points[:,1]),rtol=0,atol=1e-13)

    def test_q1_particle_projection_reproduces_affine_traction(self):
        s=np.repeat(np.arange(4),3); xi=np.tile([.2,.5,.8],4)
        weight=np.linspace(.7,1.3,len(s))
        values=2+3*(s+xi)
        mass,rhs,nodal=project(s,xi,weight,values[:,None],5)
        np.testing.assert_allclose(nodal[:,0],2+3*np.arange(5),atol=1e-13,rtol=0)
        np.testing.assert_allclose(mass@nodal,rhs,atol=1e-13,rtol=0)


if __name__=='__main__':
    unittest.main()
