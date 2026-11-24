/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include "common.h"
#include <aspect/material_model/growing_fault/section_response.h>
#include <aspect/reconstructed_fault/material_sections.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

TEST_CASE("Material sections preserve the signed work of an oblique distributed strip")
{
  using namespace aspect;
  using namespace GrowingFault;
  const MPI_Comm comm=MPI_COMM_WORLD;
  parallel::distributed::Triangulation<2> mesh(comm);
  GridGenerator::subdivided_hyper_rectangle(mesh, {2,1},Point<2>(),Point<2>(2.,1.));
  const MaterialSections geometry(mesh,comm,45.,4);
  REQUIRE(geometry.edges.size()==4);
  SectionTransfer rates(3), perturbed(3);
  double total=0., area=0., moment=0., point_work=0.;
  unsigned int active=0;
  const std::array<double,3> forces= {{2.,-3.,1.}};
  for (const auto &cell:mesh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto &rule=geometry.cells.at(cell->id());
        const auto first_shared=std::find(rule.section.begin(),rule.section.end(),1u)-rule.section.begin();
        for (unsigned int q=0; q<rule.quadrature.size(); ++q)
          {
            const auto x=cell->vertex(0)+rule.quadrature.point(q);
            const double J=rule.quadrature.weight(q); // unit squares
            const auto k=rule.section[q];
            REQUIRE(k==static_cast<unsigned int>(std::floor(x[0]+x[1])));
            total+=J;
            if (k==0)
              {
                area+=J;
                moment+=J*x[0];
              }
            if (k==1) active=1;
            const double w=x[0]-2.*x[1], dn=J/geometry.length(k,.1*k);
            rates.add(k,dn,w,.01);
            perturbed.add(k,dn,w+(cell->center()[0]<1. && q==first_shared?.1:0.),.01);
            point_work+=J*w*forces[k];
          }
      }
  REQUIRE(dealii::Utilities::MPI::sum(total,comm)==Approx(2.));
  REQUIRE(dealii::Utilities::MPI::sum(area,comm)==Approx(.5));
  REQUIRE(dealii::Utilities::MPI::sum(moment,comm)==Approx(1./6.));
  REQUIRE(dealii::Utilities::MPI::sum(active,comm)==std::min(2u,dealii::Utilities::MPI::n_mpi_processes(comm)));
  rates.sum(comm);
  perturbed.sum(comm);
  double strip_work=0.;
  for (unsigned int k=0; k<3; ++k) strip_work+=geometry.length(k,.1*k)*forces[k]*rates.free_rate[k];
  REQUIRE(std::abs(strip_work-dealii::Utilities::MPI::sum(point_work,comm))<1.e-13);
  const Friction law= {.6,.015,.02,.001,.01,1000.};
  const auto force=[&](const SectionTransfer &t)
  {
    return solve_section(law,t.free_rate[1],t.mobility[1],10.,.0025,200.).resistance;
  };
  REQUIRE(std::abs(force(perturbed)-force(rates))>1.e-6);
  const auto positive=solve_section(law,.1,.0003,10.,.0025,200.);
  REQUIRE(positive.rate+positive.resistance*.0003==Approx(.1));
  REQUIRE(solve_section(law,-.1,.0003,10.,.0025,200.).rate==Approx(-positive.rate));
  REQUIRE(solve_section(law,0.,.0003,10.,.0025,200.).resistance==0.);
  const double d=1.e-6;
  const auto plus=solve_section(law,.1+d,.0003+d*.001,10.,.0025,200.);
  const auto minus=solve_section(law,.1-d,.0003-d*.001,10.,.0025,200.);
  REQUIRE((plus.resistance-minus.resistance)/(2.*d)==Approx(positive.slope*(1.-.001*positive.resistance)).epsilon(1.e-8));
}

TEST_CASE("Intact cohesive Maxwell response has the analytic locked and phase derivative limits")
{
  using namespace aspect::GrowingFault;
  const CohesiveMaxwell law
  {
    1.e6,1.e8,80.
  };
  const double h=.0025;
  Stress plane, increment;
  plane[0][1]=.5;
  increment[0][1]=.0001;
  increment[2][2]=-.00003;
  increment[0][0]=.00003;
  PointHistory old;
  old.stress[0][1]=-20.;
  const auto response=law.response(old,increment,0.,h,plane,30.);
  const auto predictor=(old.stress+2.*law.G*increment)/(1.+h*law.G/law.eta);
  const double traction=predictor*plane-30.;
  REQUIRE((response.stress-predictor).norm()<1.e-13);
  REQUIRE(response.rate==0.);
  REQUIRE(response.cohesive_strain.norm()==0.);
  REQUIRE(law.cohesive_energy(response.cohesive_strain,0.)==0.);
  REQUIRE(response.driving==Approx(law.m*traction*traction/(2.*law.G)));
  const auto derivative=law.derivative(response, {},1.,h,plane,0.);
  REQUIRE(derivative.rate==Approx(law.m*traction*(1.+h*law.G/law.eta)/(law.G*h)));
}
