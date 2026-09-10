import unittest
import numpy as np

from endpoint_moments import (AMPLITUDE, BASE_STRESS, HALF_WIDTH, LENGTH,
                              assemble_domains, assemble_point, cloud,
                              evaluate, polygon_moments, voronoi_cells)
from measure_case import project


class EndpointMoments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quadratic=[evaluate(n,r) for n,r in ((8,.1),(16,.2),(32,.4))]
        cls.linear=evaluate(16,.4,stress_mode='linear',check_periodic=True)

    def test_polygon_moments_against_exact_rectangle(self):
        p=np.array([[2.,3.],[5.,3.],[5.,7.],[2.,7.]])
        actual=polygon_moments([p,p[::-1]],np.array([[2.,3.],[2.,3.]]))
        np.testing.assert_allclose(actual,[[12,18,36],[12,18,36]],rtol=0,atol=1e-13)

    def test_current_point_rule_matches_existing_projection_kernel(self):
        points,ax,ay=cloud(8,.4)
        n=5;ds=LENGTH/(n-1)
        volume=np.full(len(points),ax*ay)
        stress=AMPLITUDE*(points[:,1]/HALF_WIDTH)
        segment=(points[:,0]/ds).astype(int)
        xi=points[:,0]/ds-segment
        expected_mass,expected_load,_=project(segment,xi,volume,stress[:,None],n)
        mass,load=assemble_point(points,volume,stress,ds,n)
        np.testing.assert_allclose(mass,expected_mass,rtol=0,atol=1e-15)
        np.testing.assert_allclose(load,expected_load[:,0],rtol=0,atol=1e-15)

    def test_constant_reproduction_does_not_certify_weak_moments(self):
        # Assemble the constant load independently: b=M1 is the expected
        # identity, not a replacement for exercising the load assembly.
        points,ax,ay=cloud(8,.4)
        polygons=voronoi_cells(points,ax,ay)
        areas=polygon_moments(polygons,points)[:,0]
        constant=np.full(len(points),BASE_STRESS)
        for mass,load in (
            assemble_point(points,areas,constant,LENGTH/4,5),
            assemble_domains(polygons,points,constant,LENGTH/4,5,
                             np.ones(len(points),dtype=bool))):
            np.testing.assert_allclose(load,BASE_STRESS*mass.sum(axis=1),rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(np.linalg.solve(mass,load),BASE_STRESS,rtol=0,atol=1e-9)
        case=self.quadratic[-1]
        for method in case['methods'].values():
            self.assertLess(method['constant_reproduction_error_Pa'],1e-9)
        self.assertGreater(case['methods']['point_wall']['relative_mass_error'],1e-3)
        self.assertLess(case['methods']['domain_integrated']['relative_mass_error'],1e-11)
        self.assertLess(abs(case['area_error_m2']),1e-11)
        self.assertGreater(case['min_area_m2'],0)

    def test_nonconstant_field_is_decisive_at_fixed_displacement(self):
        cases=self.quadratic
        np.testing.assert_allclose([c['max_displacement_m'] for c in cases],cases[0]['max_displacement_m'])
        old=[c['methods']['point_wall']['traction_max_error_Pa'] for c in cases]
        integrated=[c['methods']['domain_integrated']['traction_max_error_Pa'] for c in cases]
        self.assertGreater(old[-1],4*old[0])
        for fine,coarse in zip(integrated[1:],integrated):
            self.assertLess(fine,.4*coarse)
        self.assertGreater(old[-1],100*integrated[-1])

    def test_periodic_area_change_alone_fails_linear_transverse_field(self):
        case=self.linear
        self.assertLess(case['periodic_audit']['max_area_difference_from_interior_relative'],1e-10)
        self.assertLess(case['periodic_audit']['wrapped_relative_mass_error'],1e-11)
        methods=case['methods']
        self.assertGreater(methods['periodic_domain_area_point']['traction_max_error_Pa'],
                           5*methods['point_wall']['traction_max_error_Pa'])
        self.assertLess(methods['domain_integrated']['traction_max_error_Pa'],
                        .01*methods['point_wall']['traction_max_error_Pa'])


if __name__=='__main__':
    unittest.main()
