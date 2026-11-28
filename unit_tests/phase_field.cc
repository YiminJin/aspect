/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include "common.h"
#include <aspect/phase_field/crack_geometry.h>
#include <aspect/phase_field/degradation.h>
#include <aspect/fe_dgq_bernstein.h>
#include <aspect/phase_field/midpoint_fracture.h>
#include <deal.II/base/quadrature_lib.h>

TEST_CASE("Mature energetic degradation uses the physical phase limits")
{
  using aspect::PhaseField::excess_compliance;
  CHECK(excess_compliance(0.,2.) == 0.);
  CHECK(excess_compliance(1.e-20,2.)/2.e-20 == Approx(1.));
  CHECK(excess_compliance(.5,2.) == Approx(6.));
  CHECK(std::isinf(excess_compliance(1.,2.)));
}

TEST_CASE("Crack geometry selection, variations and surface normalization")
{
  using aspect::PhaseField::CrackGeometry;
  const std::vector<std::string> names = {"AT1","AT2","CZM"};
  const double phases[] = {0., .25, .75, 1.};
  REQUIRE(CrackGeometry::selection_pattern() == "AT1|AT2|CZM");
  REQUIRE_THROWS(CrackGeometry::select("unknown"));

  for (const auto &name : names)
    {
      CAPTURE(name);
      const auto &law = CrackGeometry::select(name);
      for (const double phase : phases)
        {
          CAPTURE(phase);
          if (name == "AT1")
            {
              CHECK(law.value(phase) == Approx(phase));
              CHECK(law.first_derivative(phase) == Approx(1.));
              CHECK(law.second_derivative(phase) == Approx(0.));
            }
          else if (name == "AT2")
            {
              CHECK(law.value(phase) == Approx(phase*phase));
              CHECK(law.first_derivative(phase) == Approx(2.*phase));
              CHECK(law.second_derivative(phase) == Approx(2.));
            }
          else
            {
              CHECK(law.value(phase) == Approx(1.-(1.-phase)*(1.-phase)));
              CHECK(law.first_derivative(phase) == Approx(2.*(1.-phase)));
              CHECK(law.second_derivative(phase) == Approx(-2.));
            }
        }

      // Integrate the defining c_0 integral independently. Substituting
      // phase = u^2 removes the square-root endpoint singularity.
      const dealii::QGauss<1> rule(24);
      double integral = 0.;
      for (unsigned int q=0; q<rule.size(); ++q)
        {
          const double u = rule.point(q)[0];
          integral += rule.weight(q) * 8.*u*std::sqrt(law.value(u*u));
        }
      CHECK(std::abs(law.normalization()-integral) < 5.e-14);
    }
}

TEST_CASE("Midpoint inequality rows respect the signed stage obstacle")
{
  using aspect::PhaseField::projected_gradient;
  // A physical endpoint in [parent,1] gives a midpoint coefficient
  // in [parent,(1+parent)/2]; an admissible Bernstein parent can have
  // a negative coefficient.
  const double lo = -.008;
  const double hi = .5*(1.+lo);
  const double mass = .03;
  REQUIRE(projected_gradient(lo,2.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(hi,-2.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(.2,0.,mass,lo,hi)==0.);
  REQUIRE(projected_gradient(lo,-2.,mass,lo,hi)<0.);
  REQUIRE(projected_gradient(hi,2.,mass,lo,hi)>0.);
  REQUIRE(projected_gradient(.2,.001,mass,lo,hi)==Approx(.001/mass));
}


TEST_CASE("Bernstein phase prolongation preserves an irreversible off-node polynomial")
{
  const aspect::FE_DGQBernstein<2> phase(2);
  dealii::Vector<double> parent(9), next(9);
  for (unsigned int i=0; i<9; ++i)
    {
      // A nonseparable quadratic with a large middle coefficient. The
      // physical polynomial is bounded despite the coefficient above one.
      parent[i] = .13 + .01*i + (i==4 ? 1.1 : 0.);
      next[i] = parent[i] + .03 + .002*i;
    }

  for (unsigned int child=0; child<4; ++child)
    {
      dealii::Vector<double> p(9), q(9);
      const auto &restriction_to_child = phase.get_prolongation_matrix(child);
      restriction_to_child.vmult(p,parent);
      restriction_to_child.vmult(q,next);
      const dealii::QGauss<2> probes(6);
      for (const auto &point : probes.get_points())
        {
          const dealii::Point<2> parent_point((point[0]+(child%2))*.5,(point[1]+(child/2))*.5);
          double reference=0., restricted=0., increment=0.;
          for (unsigned int j=0; j<9; ++j)
            {
              reference += parent[j] * phase.shape_value(j,parent_point);
              restricted += p[j] * phase.shape_value(j,point);
              increment += (q[j]-p[j]) * phase.shape_value(j,point);
            }
          REQUIRE(std::abs(reference-restricted) < 2.e-14);
          REQUIRE(restricted >= 0.);
          REQUIRE(restricted+increment <= 1.);
          REQUIRE(increment >= .02999999999999);
        }
    }
}
