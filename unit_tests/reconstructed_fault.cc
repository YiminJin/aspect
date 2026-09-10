/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/reconstructed_fault/fault.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator/solver/reconstructed_fault_nonlinear.h>
#include <aspect/simulator/solver/reconstructed_fault_linear.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <aspect/utilities.h>

#include <limits>
#include <sstream>

namespace
{
  class ThrowOnDealIIException
  {
    public:
      ThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::disable_abort_on_exception();
      }

      ~ThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::enable_abort_on_exception();
      }
  };
}


TEST_CASE("ReconstructedFault domain quadrature covers nodes and tips", "[fault_domain_quadrature]")
{
  for (const double angle : {0.0, 0.37})
    {
      const auto rotate = [angle](double x, double y)
      {
        return dealii::Point<2>(2+x*std::cos(angle)-y*std::sin(angle),
                               3+x*std::sin(angle)+y*std::cos(angle));
      };
      const aspect::ReconstructedFault<2> fault(
        {rotate(0,0),rotate(.3,0),rotate(.7,0),rotate(1,0)});
      const std::vector<dealii::Point<2>> polygon =
        {rotate(-.1,-1),rotate(1.1,-1),rotate(1.1,1),rotate(-.1,1)};
      const auto quadrature = aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault);
      dealii::FullMatrix<double> mass(4,4), reference(4,4);
      std::vector<double> load(4,0.0);
      double area=0;
      for (const auto &q : quadrature)
        {
          REQUIRE(q.weight>0);
          REQUIRE(q.xi>=0);
          REQUIRE(q.xi<=1);
          area+=q.weight;
          const double shape[2]={1-q.xi,q.xi};
          for (unsigned int i=0;i<2;++i)
            {
              load[q.segment_index+i]+=q.weight*shape[i]*7;
              for (unsigned int j=0;j<2;++j)
                mass(q.segment_index+i,q.segment_index+j)+=q.weight*shape[i]*shape[j];
            }
        }
      REQUIRE(area==Approx(2.4).margin(1e-13));
      for (unsigned int s=0;s<3;++s)
        {
          const double ds=fault.vertex(s).distance(fault.vertex(s+1));
          reference(s,s)+=2*ds/3;
          reference(s+1,s+1)+=2*ds/3;
          reference(s,s+1)+=ds/3;
          reference(s+1,s)+=ds/3;
        }
      reference(0,0)+=.2;
      reference(3,3)+=.2;
      for (unsigned int i=0;i<4;++i)
        {
          double row=0;
          for (unsigned int j=0;j<4;++j)
            {
              REQUIRE(mass(i,j)==Approx(reference(i,j)).margin(1e-13));
              row+=reference(i,j);
            }
          REQUIRE(load[i]==Approx(7*row).margin(1e-12));
        }
    }
}

TEST_CASE("ReconstructedFault nonlinear domain quadrature accuracy", "[fault_domain_quadrature]")
{
  const aspect::ReconstructedFault<2> fault({{0,0},{.3,0},{.7,0},{1,0}});
  // A slanted convex domain exercises nonconstant transverse width. Compare
  // smooth nonlinear loads separately from exact polynomial geometry moments.
  const std::vector<dealii::Point<2>> polygon={{0,-1},{1,-.8},{.9,1},{.1,.8}};
  std::vector<std::vector<double>> loads;
  for (const unsigned int order : {3,5,7})
    {
      std::vector<double> load(4,0.0);
      for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault,order))
        {
          const double s=(1-q.xi)*fault.vertex(q.segment_index)[0]
                         +q.xi*fault.vertex(q.segment_index+1)[0];
          const double response=std::log(2+.5*s);
          load[q.segment_index]+=q.weight*(1-q.xi)*response;
          load[q.segment_index+1]+=q.weight*q.xi*response;
        }
      loads.push_back(load);
    }
  for (unsigned int i=0;i<4;++i)
    {
      REQUIRE(std::abs(loads[0][i]-loads[2][i])<1e-8);
      REQUIRE(std::abs(loads[1][i]-loads[2][i])<1e-12);
    }
}

TEST_CASE("ReconstructedFault bent-domain first and second moments", "[fault_domain_quadrature]")
{
  // For this right-angle bend the finite-projection overlap is split by y=-x.
  // The lower-right quadrant is a constant-vertex corner region. Integrating
  // its rectangles/triangles analytically gives the fractions below.
  const aspect::ReconstructedFault<2> fault({{-1,0},{0,0},{0,1}});
  const std::vector<dealii::Point<2>> polygon={{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
  const auto quadrature=aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault);
  dealii::FullMatrix<double> mass(3,3);
  std::vector<double> first(3,0), corner(2,0);
  for (const auto &q : quadrature)
    {
      REQUIRE(q.weight>0);
      const double shape[2]={1-q.xi,q.xi};
      for (unsigned int i=0;i<2;++i)
        {
          first[q.segment_index+i]+=q.weight*shape[i];
          for (unsigned int j=0;j<2;++j)
            mass(q.segment_index+i,q.segment_index+j)+=q.weight*shape[i]*shape[j];
        }
      if (q.segment_index==0 && q.xi==1) corner[0]+=q.weight;
      if (q.segment_index==1 && q.xi==0) corner[1]+=q.weight;
    }
  const double exact_first[3]={5./48,19./24,5./48};
  const double exact_mass[3][3]={{7./192,13./192,0},{13./192,21./32,13./192},{0,13./192,7./192}};
  for (unsigned int i=0;i<3;++i)
    {
      REQUIRE(first[i]==Approx(exact_first[i]).margin(1e-13));
      for (unsigned int j=0;j<3;++j)
        REQUIRE(mass(i,j)==Approx(exact_mass[i][j]).margin(1e-13));
    }
  REQUIRE(corner[0]==Approx(.125).margin(1e-13));
  REQUIRE(corner[1]==Approx(.125).margin(1e-13));

  // Off-origin tip pieces have no finite orthogonal candidate. They retain
  // their whole measure and the sole incident endpoint frame.
  for (const bool last : {false,true})
    {
      const std::vector<dealii::Point<2>> tip=last ?
        std::vector<dealii::Point<2>>{{.1,1.1},{.2,1.1},{.2,1.5},{.1,1.5}} :
        std::vector<dealii::Point<2>>{{-1.5,-.2},{-1.1,-.2},{-1.1,-.1},{-1.5,-.1}};
      double area=0;
      for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(tip,fault))
        {
          REQUIRE(q.segment_index==(last ? 1 : 0));
          REQUIRE(q.xi==(last ? 1.0 : 0.0));
          area+=q.weight;
        }
      REQUIRE(area==Approx(.04).margin(1e-13));
    }
}

TEST_CASE("ReconstructedFault polyline straight limit and tiny cuts", "[fault_domain_quadrature]")
{
  const std::vector<dealii::Point<2>> polygon={{-.2,-.3},{.2,-.3},{.2,.3},{-.2,.3}};
  std::vector<std::vector<double>> moments;
  for (const double bend : {0.,1e-5,1e-7,1e-12})
    {
      const aspect::ReconstructedFault<2> fault({{-1,0},{0,bend},{1,0}});
      std::vector<double> entries(9,0);
      double area=0;
      for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault))
        {
          const double shape[2]={1-q.xi,q.xi};
          area+=q.weight;
          for (unsigned int i=0;i<2;++i)
            for (unsigned int j=0;j<2;++j)
              entries[3*(q.segment_index+i)+q.segment_index+j]+=q.weight*shape[i]*shape[j];
        }
      REQUIRE(area==Approx(.24).margin(1e-13));
      moments.push_back(entries);
    }
  for (unsigned int k=1;k<moments.size();++k)
    for (unsigned int i=0;i<9;++i)
      REQUIRE(std::abs(moments[k][i]-moments[0][i])<1e-5);
  for (unsigned int i=0;i<9;++i)
    REQUIRE(std::abs(moments[3][i]-moments[0][i])<1e-12);
}

TEST_CASE("ReconstructedFault captured near-straight polyline", "[fault_domain_quadrature]")
{
  // Captured verbatim from phase_field_fault_condensed_adiabatic's reconstructed
  // geometry, not its prescribed horizontal reference or a flattened fit.
  const aspect::ReconstructedFault<2> fault({
    {.20000000000000001,.49999985615271447},
    {.28571428571428575,.49999970654504250},
    {.37142857142857144,.49999960406507321},
    {.45714285714285718,.49999955150387382},
    {.54285714285714293,.49999954964845694},
    {.62857142857142867,.49999959839597602},
    {.71428571428571441,.49999969640130554},
    {.80000000000000004,.49999983934469361}});
  for (unsigned int vertex=0;vertex<fault.n_vertices();++vertex)
    {
      const double x=fault.vertex(vertex)[0];
      const std::vector<dealii::Point<2>> polygon={{x-.003,.45},{x+.003,.45},{x+.003,.55},{x-.003,.55}};
      std::vector<std::vector<double>> matrices;
      for (const unsigned int order : {3,7})
        {
          std::vector<double> matrix(64,0);
          double area=0;
          for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault,order))
            {
              area+=q.weight;
              const double shape[2]={1-q.xi,q.xi};
              for (unsigned int i=0;i<2;++i)
                for (unsigned int j=0;j<2;++j)
                  matrix[8*(q.segment_index+i)+q.segment_index+j]+=q.weight*shape[i]*shape[j];
            }
          REQUIRE(area==Approx(.0006).margin(1e-14));
          matrices.push_back(matrix);
        }
      for (unsigned int i=0;i<64;++i)
        REQUIRE(matrices[0][i]==Approx(matrices[1][i]).margin(1e-14));
    }
}

TEST_CASE("ReconstructedFault polyline near-degenerate tip cut", "[fault_domain_quadrature]")
{
  const aspect::ReconstructedFault<2> fault({{-1,0},{0,0},{0,1}});
  for (const double delta : {1e-8,1e-13})
    {
      const double left=-1-delta, right=-1+delta;
      const std::vector<dealii::Point<2>> polygon={{left,-.2},{right,-.2},{right,-.1},{left,-.1}};
      double area=0,tip=0;
      for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault))
        {
          area+=q.weight;
          if (q.segment_index==0 && q.xi==0) tip+=q.weight;
        }
      REQUIRE(area==Approx(.1*(right-left)).epsilon(1e-10));
      REQUIRE(tip==Approx(.1*(-1-left)).epsilon(1e-10));
      REQUIRE(tip>0);
    }
}

TEST_CASE("ReconstructedFault bent nonlinear quadrature accuracy", "[fault_domain_quadrature]")
{
  const aspect::ReconstructedFault<2> fault({{-1,0},{0,0},{0,1}});
  const std::vector<dealii::Point<2>> polygon={{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}};
  std::vector<std::vector<double>> loads;
  for (const unsigned int order : {3,5,7})
    {
      std::vector<double> load(3,0);
      for (const auto &q : aspect::ReconstructedFaultUtilities::domain_quadrature(polygon,fault,order))
        {
          const double response=std::log(2+.5*(q.segment_index+q.xi));
          load[q.segment_index]+=q.weight*(1-q.xi)*response;
          load[q.segment_index+1]+=q.weight*q.xi*response;
        }
      loads.push_back(load);
    }
  for (unsigned int i=0;i<3;++i)
    {
      REQUIRE(std::abs(loads[0][i]-loads[2][i])<1e-8);
      REQUIRE(std::abs(loads[1][i]-loads[2][i])<1e-12);
    }
}

TEST_CASE("ReconstructedFault empty geometry")
{
  const aspect::ReconstructedFault<2> fault;

  REQUIRE(fault.empty());
  REQUIRE(fault.n_vertices() == 0);
  REQUIRE(fault.n_cells() == 0);
  REQUIRE(fault.get_vertices().empty());
  REQUIRE(fault.geometry_version() == 0);
#ifndef NDEBUG
  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(fault.vertex(0));
#endif
}


TEST_CASE("ReconstructedFault construction and ordered access")
{
  const std::vector<dealii::Point<2>> vertices =
  {
    dealii::Point<2>(0.0, 1.0),
    dealii::Point<2>(2.0, 3.0),
    dealii::Point<2>(5.0, 8.0)
  };
  const aspect::ReconstructedFault<2> fault(vertices);

  REQUIRE_FALSE(fault.empty());
  REQUIRE(fault.n_vertices() == 3);
  REQUIRE(fault.n_cells() == 2);
  REQUIRE(fault.geometry_version() == 0);
  REQUIRE(fault.get_vertices() == vertices);
  REQUIRE(fault.vertex(0) == vertices[0]);
  REQUIRE(fault.vertex(2) == vertices[2]);
#ifndef NDEBUG
  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(fault.vertex(3));
#endif
}


TEST_CASE("ReconstructedFault append-only updates")
{
  aspect::ReconstructedFault<2> fault;
  const dealii::Point<2> first(1.0, 2.0);
  const dealii::Point<2> second(3.0, 4.0);
  const dealii::Point<2> third(5.0, 6.0);

  fault.append_vertex(first);
  REQUIRE(fault.n_vertices() == 1);
  REQUIRE(fault.n_cells() == 0);
  REQUIRE(fault.geometry_version() == 1);

  fault.append_vertices({second, third});
  REQUIRE(fault.n_vertices() == 3);
  REQUIRE(fault.n_cells() == 2);
  REQUIRE(fault.geometry_version() == 2);
  REQUIRE(fault.vertex(0) == first);
  REQUIRE(fault.vertex(1) == second);
  REQUIRE(fault.vertex(2) == third);

  fault.append_vertices({});
  REQUIRE(fault.n_vertices() == 3);
  REQUIRE(fault.geometry_version() == 2);

  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(fault.append_vertex(third));
  REQUIRE_THROWS(fault.append_vertex(
    dealii::Point<2>(std::numeric_limits<double>::infinity(), 0.0)));
  REQUIRE_THROWS(fault.append_vertices(
    {dealii::Point<2>(4,0), dealii::Point<2>(4,0)}));
  REQUIRE(fault.n_vertices() == 3);
  REQUIRE(fault.geometry_version() == 2);
}


TEST_CASE("ReconstructedFaultManager owns the shared vertex property schema")
{
  aspect::ReconstructedFaultManager<2> manager;

  const unsigned int scalar = manager.register_property("scalar", 1);
  const unsigned int vector = manager.register_property("vector", 2);

  REQUIRE(scalar == 0);
  REQUIRE(vector == 1);
  REQUIRE(manager.has_property("scalar"));
  REQUIRE_FALSE(manager.has_property("missing"));
  REQUIRE(manager.get_property_index("vector") == vector);
  REQUIRE(manager.get_property_information().size() == 2);
  REQUIRE(manager.get_property_information()[0].position == 0);
  REQUIRE(manager.get_property_information()[1].name == "vector");
  REQUIRE(manager.get_property_information()[1].n_components == 2);
  REQUIRE(manager.get_property_information()[1].position == 1);

  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(manager.register_property("", 1));
  REQUIRE_THROWS(manager.register_property("zero components", 0));
  REQUIRE_THROWS(manager.register_property("scalar", 1));
  REQUIRE_THROWS(manager.register_property("slip_rate", 1));
  REQUIRE_THROWS(manager.get_property_index("missing"));

  manager.add_reconstructed_fault({dealii::Point<2>(0,0), dealii::Point<2>(1,0)},
                                  {0.5, 0.5});
  REQUIRE_THROWS(manager.add_reconstructed_fault(
    {dealii::Point<2>(0,0), dealii::Point<2>(0,0)}, {0.5, 0.5}));
  REQUIRE_THROWS(manager.add_reconstructed_fault(
    {dealii::Point<2>(0,0), dealii::Point<2>(1,0)}, {0.5}));
  const unsigned int scalar_position = manager.get_property_information()[scalar].position;
  REQUIRE_FALSE(manager.get_fault(0).property_value_is_initialized(0, scalar_position));
  manager.get_fault(0).get_properties(0)[scalar_position] = 3.0;
  REQUIRE(manager.get_fault(0).property_value_is_initialized(0, scalar_position));
}


TEST_CASE("ReconstructedFaultManager slip-rate lifecycle and interpolation")
{
  aspect::ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault({dealii::Point<2>(0,0),
                                   dealii::Point<2>(2,0),
                                   dealii::Point<2>(4,0)},
                                  {1.0, 1.0, 1.0});

  REQUIRE_FALSE(manager.slip_rates_are_initialized());
  manager.initialize_slip_rate(0, {1.0, 2.0, 4.0});
  REQUIRE(manager.slip_rates_are_initialized());
  REQUIRE(manager.interpolate_slip_rate(0, 1, 0.25) == Approx(2.5));

  manager.begin_slip_rate_nonlinear_solve();
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{2.0, -2.0, 4.0}}, 0.5);
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(2.0));
  REQUIRE(manager.get_slip_rate(0)[1] == Approx(1.0));
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[0] == Approx(1.0));

  // A second candidate is formed from the current Newton iterate, not the first candidate.
  manager.set_slip_rate_trial({{2.0, -2.0, 4.0}}, 0.25);
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(1.5));
  REQUIRE(manager.get_slip_rate(0)[1] == Approx(1.5));
  manager.rollback_slip_rate_trial();
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(1.0));

  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{2.0, -2.0, 4.0}}, 0.25);
  manager.accept_slip_rate_trial();
  REQUIRE(manager.get_slip_rate(0)[2] == Approx(5.0));
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[2] == Approx(4.0));

  // A new line-search trial starts from the accepted Newton iterate.
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{2.0, -2.0, 4.0}}, 0.25);
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(2.0));
  manager.rollback_slip_rate_trial();
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(1.5));

  manager.commit_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[2] == Approx(5.0));

  // Rejecting a later nonlinear solve restores and preserves timestep state.
  manager.begin_slip_rate_nonlinear_solve();
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{1.0, 1.0, 1.0}}, 1.0);
  manager.accept_slip_rate_trial();
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(2.5));
  manager.rollback_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_slip_rate(0)[0] == Approx(1.5));
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[0] == Approx(1.5));

#ifndef NDEBUG
  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(manager.initialize_slip_rate(0, {1.0, 2.0, 3.0}));
  REQUIRE_THROWS(manager.interpolate_slip_rate(0, 0, 1.1));
  REQUIRE_THROWS(manager.set_slip_rate_trial({{0.0, 0.0, 0.0}}, 1.0));
#endif
}


TEST_CASE("ReconstructedFaultManager validates slip-rate initialization")
{
  aspect::ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault({dealii::Point<2>(0,0), dealii::Point<2>(1,0)},
                                  {0.5, 0.5});
  const ThrowOnDealIIException throw_on_dealii_exception;
#ifndef NDEBUG
  REQUIRE_THROWS(manager.initialize_slip_rate(0, {1.0}));
#endif
  REQUIRE_THROWS(manager.initialize_slip_rate(
    0, {1.0, std::numeric_limits<double>::quiet_NaN()}));
  REQUIRE_THROWS(manager.initialize_slip_rate(0, {-1.0, 2.0}));
#ifndef NDEBUG
  REQUIRE_THROWS(manager.begin_slip_rate_trial());
#endif

  manager.initialize_slip_rate(0, {0.0, 2.0});
  manager.begin_slip_rate_nonlinear_solve();
  manager.begin_slip_rate_trial();
#ifndef NDEBUG
  REQUIRE_THROWS(manager.set_slip_rate_trial({}, 1.0));
  REQUIRE_THROWS(manager.set_slip_rate_trial({{1.0}}, 1.0));
#endif
  REQUIRE_THROWS(manager.set_slip_rate_trial(
    {{0.0, std::numeric_limits<double>::infinity()}}, 1.0));
  REQUIRE_THROWS(manager.set_slip_rate_trial({{-2.0, 0.0}}, 1.0));
  manager.rollback_slip_rate_trial();
  manager.rollback_slip_rate_nonlinear_solve();
}


TEST_CASE("Stage-I lower-bound tolerance is local")
{
  constexpr double minimum = 1e-12;
  const double epsilon = std::numeric_limits<double>::epsilon();
  REQUIRE(aspect::internal::reconstructed_fault_locally_at_lower_bound(
            minimum, minimum));
  REQUIRE(aspect::internal::reconstructed_fault_locally_at_lower_bound(
            minimum*(1.0+50.0*epsilon), minimum));
  REQUIRE_FALSE(aspect::internal::reconstructed_fault_locally_at_lower_bound(
                  minimum*(1.0+200.0*epsilon), minimum));
}


TEST_CASE("Stage-I active set releases on a later Newton iteration")
{
  constexpr double minimum = 1e-12;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const aspect::ReconstructedFaultVector slip_rate =
  {
    {minimum*(1.0+50.0*epsilon), 1e8}
  };
  const aspect::ReconstructedFaultVector outward_direction =
  {
    {-1.0, -1.0}
  };
  aspect::ReconstructedFaultActiveSet active_set =
    aspect::internal::make_reconstructed_fault_inactive_set(slip_rate);

  REQUIRE(aspect::internal::update_reconstructed_fault_active_set(
            slip_rate, outward_direction, minimum, active_set) == 1);
  REQUIRE(active_set[0][0]);
  REQUIRE_FALSE(active_set[0][1]);

  // A new outer Newton iteration starts from an empty set. An inward
  // direction releases the vertex that was active in the previous iteration.
  active_set = aspect::internal::make_reconstructed_fault_inactive_set(slip_rate);
  const aspect::ReconstructedFaultVector inward_direction =
  {
    {1.0, -1.0}
  };
  REQUIRE(aspect::internal::update_reconstructed_fault_active_set(
            slip_rate, inward_direction, minimum, active_set) == 0);
  REQUIRE_FALSE(active_set[0][0]);
}


TEST_CASE("Stage-I fraction-to-boundary permits exact contact")
{
  const aspect::ReconstructedFaultVector slip_rate = {{1.0, 1.5}};
  const aspect::ReconstructedFaultVector direction = {{-100.0, -2.0}};
  const aspect::ReconstructedFaultActiveSet active_set = {{true, false}};

  const double step =
    aspect::internal::reconstructed_fault_maximum_step_length(
      slip_rate, direction, active_set, 1.0);
  REQUIRE(step == Approx(0.25));
  REQUIRE(slip_rate[0][1] + step*direction[0][1] == Approx(1.0));
  REQUIRE(step != Approx(0.99*0.25));
}


TEST_CASE("Stage-I Armijo search rejects twice before acceptance")
{
  std::vector<double> evaluated_steps;
  bool accept_called = false;
  double accepted_step = 0.0;
  const auto result = aspect::internal::reconstructed_fault_armijo_line_search(
    1.0,
    4,
    1.0,
    [&](const double step)
    {
      evaluated_steps.push_back(step);
      return evaluated_steps.size() <= 2 ? 1.0 : 0.5;
    },
    [&](const double step)
    {
      accept_called = true;
      accepted_step = step;
    });

  REQUIRE(result.accepted);
  REQUIRE(result.rejected_candidates == 2);
  REQUIRE(accept_called);
  REQUIRE(evaluated_steps.size() == 3);
  REQUIRE(evaluated_steps[0] == Approx(1.0));
  REQUIRE(evaluated_steps[1] == Approx(2.0/3.0));
  REQUIRE(evaluated_steps[2] == Approx(4.0/9.0));
  REQUIRE(accepted_step == Approx(4.0/9.0));
}


TEST_CASE("Stage-I Armijo exhaustion never accepts the last candidate")
{
  unsigned int evaluations = 0;
  double accepted_state = 7.0;
  const auto result = aspect::internal::reconstructed_fault_armijo_line_search(
    1.0,
    2,
    1.0,
    [&](const double)
    {
      ++evaluations;
      return 1.0;
    },
    [&](const double step)
    {
      accepted_state = step;
    });

  REQUIRE_FALSE(result.accepted);
  REQUIRE(result.rejected_candidates == 3);
  REQUIRE(evaluations == 3);
  REQUIRE(accepted_state == Approx(7.0));
}


TEST_CASE("Stage-I pressure complement agrees with an explicit full coupled solve")
{
  using namespace dealii;
  using namespace aspect::internal;
  FullMatrix<double> full(6), C(4);
  const double A[4][4] = {{4,1,1,-1}, {1,3,2,-2}, {1,2,0,0}, {-1,-2,0,0}};
  const double B[4] = {1,.5,0,0}, G[4] = {2,-1,0,0};
  Vector<double> q(4), expected(6), rhs(6), reference(6), b(4), x(4), residual(4);
  q[2] = q[3] = 1./std::sqrt(2.);
  expected[0]=1e-12; expected[1]=-2e-12;
  expected[2]=3e-12; expected[3]=-3e-12; expected[4]=4e-12;
  for (unsigned int i=0; i<4; ++i)
    {
      for (unsigned int j=0; j<4; ++j)
        {
          full(i,j)=A[i][j];
          C(i,j)=A[i][j]-B[i]*G[j]/3.;
        }
      full(i,4)=-B[i]; full(4,i)=G[i];
      full(i,5)=full(5,i)=q[i];
    }
  full(4,4)=-3.;
  full.vmult(rhs,expected);
  for (unsigned int i=0; i<4; ++i) b[i]=rhs[i]-B[i]*rhs[4]/3.;
  full.gauss_jordan();
  full.vmult(reference,rhs);

  Vector<double> check(4);
  C.vmult(check,q); REQUIRE(check.l2_norm()<1e-15);
  C.Tvmult(check,q); REQUIRE(check.l2_norm()<1e-15);
  const PreconditionIdentity identity;
  const FaultPressureComplementOperator<FullMatrix<double>,Vector<double>> op{C,q};
  const FaultPressureComplementOperator<PreconditionIdentity,Vector<double>> preconditioner{identity,q};
  SolverControl control(30,1e-9*b.l2_norm());
  SolverFGMRES<Vector<double>> solver(control);
  solver.solve(op,x,b,preconditioner);
  project_fault_pressure(q,x);
  for (unsigned int i=0; i<4; ++i) REQUIRE(std::abs(x[i]-reference[i])<1e-24);
  double V=-rhs[4]/3.;
  for (unsigned int i=0; i<4; ++i) V+=G[i]*x[i]/3.;
  REQUIRE(std::abs(V-reference[4])<1e-24);

  double raw, component;
  REQUIRE(fault_true_linear_residual(C,q,x,b,residual,raw,component)<control.tolerance());
  // An optimistic estimated residual must not certify a bad returned vector.
  x[0]+=1e-6;
  REQUIRE(fault_true_linear_residual(C,q,x,b,residual,raw,component)>control.tolerance());
}

TEST_CASE("Stage-I pressure compatibility rejects a significant RHS before projection")
{
  using namespace dealii;
  using namespace aspect::internal;
  const ThrowOnDealIIException throw_on_dealii_exception;
  Vector<double> q(3), rhs(3);
  q[1]=q[2]=1./std::sqrt(2.);
  rhs[0]=1.; rhs.add(1e-6,q);
  const Vector<double> saved(rhs);
  REQUIRE_THROWS(project_compatible_fault_rhs(q,1e-14,rhs));
  rhs-=saved; REQUIRE(rhs.l2_norm()==0.);
  rhs[0]=1.; rhs.add(1e-16,q);
  REQUIRE(project_compatible_fault_rhs(q,1e-14,rhs)==Approx(1e-16));
  REQUIRE(std::abs(q*rhs)<1e-30);
  rhs.add(147.,q);
  project_fault_pressure(q,rhs);
  REQUIRE(std::abs(q*rhs)<1e-25);
}

TEST_CASE("Stage-I bulk residual scale handles a zero initial block")
{
  const double scale = aspect::internal::reconstructed_fault_residual_scale(
    0.0, 1e-280, 1e-8);
  REQUIRE(std::isfinite(scale));
  REQUIRE(scale/1e-288 == Approx(1.0));
  REQUIRE(aspect::internal::normalized_reconstructed_fault_residual(
            0.0, scale, "bulk") == Approx(0.0));
}


TEST_CASE("Stage-I bulk precision scale is independent of the stalled residual")
{
  using namespace dealii;
  using namespace aspect::internal;
  const MPI_Comm comm = MPI_COMM_WORLD;
  const std::vector<IndexSet> partition = {
    Utilities::MPI::create_evenly_distributed_partitioning(comm,2),
    Utilities::MPI::create_evenly_distributed_partitioning(comm,1)};
  BlockDynamicSparsityPattern sparsity(2,2);
  for (unsigned int b=0; b<2; ++b)
    for (unsigned int c=0; c<2; ++c)
      {
        sparsity.block(b,c).reinit(partition[b].size(),partition[c].size());
        for (unsigned int i=0; i<partition[b].size(); ++i)
          for (unsigned int j=0; j<partition[c].size(); ++j)
            sparsity.block(b,c).add(i,j);
      }
  sparsity.collect_sizes();
  aspect::LinearAlgebra::BlockSparseMatrix matrix;
  matrix.reinit(partition,sparsity,comm);
  const double entries[3][3] = {{4.,-2.,3.},{-2.,5.,-1.},{3.,-1.,0.}};
  aspect::LinearAlgebra::BlockVector state(partition,comm), perturbation(partition,comm), action(partition,comm);
  const double x[3] = {2.,-1.,7.};
  const double eps = std::numeric_limits<double>::epsilon();
  for (unsigned int b=0; b<2; ++b)
    for (const auto i : partition[b])
      {
        const unsigned int row = b == 0 ? i : 2;
        state.block(b)[i] = x[row];
        perturbation.block(b)[i] = eps*(b == 0 ? 2. : 7.);
        for (unsigned int c=0; c<2; ++c)
          for (unsigned int j=0; j<partition[c].size(); ++j)
            matrix.block(b,c).set(i,j,entries[row][c == 0 ? j : 2]);
      }
  matrix.compress(VectorOperation::insert);
  state.compress(VectorOperation::insert);
  perturbation.compress(VectorOperation::insert);
  const double precision = reconstructed_fault_bulk_precision_scale(matrix,state,comm);
  REQUIRE(precision/eps == Approx(std::sqrt(33.*33.+21.*21.+8.*8.)));
  matrix.vmult(action,perturbation);
  REQUIRE(action.l2_norm() <= precision);
  state *= 2.;
  REQUIRE(reconstructed_fault_bulk_precision_scale(matrix,state,comm)/precision == Approx(2.));
  state = 0.;
  REQUIRE(reconstructed_fault_bulk_precision_scale(matrix,state,comm) == 0.);

  // Mixed acceptance is a fixed absolute plus relative target, not permission
  // to accept stagnation. A materially larger residual still fails both tests.
  const double relative_scale = 1e-10, tolerance = 1e-8;
  const double mixed_scale = relative_scale+precision/tolerance;
  const double large = normalized_reconstructed_fault_residual(100.*precision,mixed_scale,"bulk");
  REQUIRE(large > tolerance);
  const auto result = reconstructed_fault_armijo_line_search(
    1.,5,large*large/2.,[&](double) { return large*large/2.; },
    [&](double) { FAIL("Stagnation above the mixed target was accepted."); });
  REQUIRE_FALSE(result.accepted);
  REQUIRE(result.rejected_candidates == 6);
}


TEST_CASE("Stage-I bulk residual floor handles a tiny initial block")
{
  const double scale = aspect::internal::reconstructed_fault_residual_scale(
    1e-300, 1e-290, 1e-8);
  REQUIRE(std::isfinite(scale));
  REQUIRE(scale/1e-298 == Approx(1.0));
  REQUIRE(aspect::internal::normalized_reconstructed_fault_residual(
            1e-300, scale, "surface") == Approx(1e-2));

  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE(aspect::internal::normalized_reconstructed_fault_residual(
            0.0, 0.0, "surface") == Approx(0.0));
  REQUIRE_THROWS(aspect::internal::normalized_reconstructed_fault_residual(
    1e-300, 0.0, "surface"));
}


TEST_CASE("ReconstructedFaultManager checkpoint restores committed slip rate")
{
  aspect::ReconstructedFaultManager<2> manager;
  const unsigned int cohesive =
    manager.register_property("phase field fault cohesive traction", 1);
  const unsigned int previous_I_h =
    manager.register_property("phase field fault previous I h", 1);
  manager.add_reconstructed_fault({dealii::Point<2>(0,0), dealii::Point<2>(1,0)},
                                  {0.5, 0.75});
  manager.initialize_slip_rate(0, {2.0, 3.0});
  const auto &property_information = manager.get_property_information();
  manager.get_fault(0).get_properties(0)[property_information[cohesive].position] = 7.0;
  manager.get_fault(0).get_properties(0)[property_information[previous_I_h].position] = 2.5;
  REQUIRE(manager.get_fault(0).property_value_is_initialized(
    0, property_information[cohesive].position));
  REQUIRE_FALSE(manager.get_fault(0).property_value_is_initialized(
    1, property_information[cohesive].position));
  manager.begin_slip_rate_nonlinear_solve();
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{10.0, 10.0}}, 1.0);
  manager.accept_slip_rate_trial();
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{10.0, 10.0}}, 1.0);

  std::stringstream storage;
  {
    aspect::oarchive archive(storage);
    archive << manager;
  }

  aspect::ReconstructedFaultManager<2> restored;
  {
    aspect::iarchive archive(storage);
    archive >> restored;
  }

  REQUIRE(restored.get_faults().size() == 1);
  REQUIRE(restored.get_fault(0).vertex(1) == dealii::Point<2>(1,0));
  const auto &restored_information = restored.get_property_information();
  REQUIRE(restored.get_fault(0).get_properties(0)[restored_information[
            restored.get_property_index("phase field fault cohesive traction")].position]
          == Approx(7.0));
  REQUIRE(restored.get_fault(0).get_properties(0)[restored_information[
            restored.get_property_index("phase field fault previous I h")].position]
          == Approx(2.5));
  REQUIRE(restored.get_fault(0).property_value_is_initialized(
    0, restored_information[restored.get_property_index(
      "phase field fault cohesive traction")].position));
  REQUIRE_FALSE(restored.get_fault(0).property_value_is_initialized(
    1, restored_information[restored.get_property_index(
      "phase field fault cohesive traction")].position));
  // Checkpoints contain timestep state, not an accepted Newton iterate or trial candidate.
  REQUIRE(restored.get_slip_rate(0)[0] == Approx(2.0));
  REQUIRE(restored.interpolate_slip_rate(0, 0, 0.5) == Approx(2.5));
  REQUIRE(restored.has_property("phase field fault cohesive traction"));
  REQUIRE(restored.has_property("phase field fault previous I h"));
}


TEST_CASE("Fault normal-profile projection respects finite segments and varying widths")
{
  const std::vector<aspect::ReconstructedFault<2>> faults =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0),
                                   dealii::Point<2>(2,0)})
  };
  const std::vector<std::vector<double>> widths = {{0.5, 1.5}};

  const auto interior =
    aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
      faults, widths, dealii::Point<2>(1.0, 0.9));
  REQUIRE(interior.active);
  REQUIRE(interior.fault_index == 0);
  REQUIRE(interior.segment_index == 0);
  REQUIRE(interior.xi == Approx(0.5));
  REQUIRE(interior.signed_distance == Approx(0.9));

  const auto outside_width =
    aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
      faults, widths, dealii::Point<2>(0.2, 0.7));
  REQUIRE_FALSE(outside_width.active);

  const auto beyond_open_tip =
    aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
      faults, widths, dealii::Point<2>(-0.1, 0.1));
  REQUIRE_FALSE(beyond_open_tip.active);
}


TEST_CASE("Fault normal-profile projection selects an incident segment near an internal bend")
{
  const std::vector<aspect::ReconstructedFault<2>> faults =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0),
                                   dealii::Point<2>(1,0),
                                   dealii::Point<2>(2,0.2)})
  };
  const auto projection =
    aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
      faults, {{0.4, 0.4, 0.4}}, dealii::Point<2>(1.5, 0.25));
  REQUIRE(projection.active);
  REQUIRE(projection.segment_index == 1);
  REQUIRE(projection.xi > 0.0);
  REQUIRE(projection.xi < 1.0);
}


TEST_CASE("Fault normal-profile projection rejects unsupported topology and overlap")
{
  const ThrowOnDealIIException throw_on_dealii_exception;
  const std::vector<aspect::ReconstructedFault<2>> overlapping_faults =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0), dealii::Point<2>(2,0)}),
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0.1), dealii::Point<2>(2,0.1)})
  };
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
    overlapping_faults, {{0.2, 0.2}, {0.2, 0.2}}, dealii::Point<2>(1,0.05)));

  const std::vector<aspect::ReconstructedFault<2>> closed_fault =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0),
                                   dealii::Point<2>(1,0),
                                   dealii::Point<2>(0,0)})
  };
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
    closed_fault, {{0.2, 0.2, 0.2}}, dealii::Point<2>(0.5,0.1)));
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
    overlapping_faults, {{0.2, 0.2}}, dealii::Point<2>(1,0.05)));
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::project_to_normal_profiles(
    {overlapping_faults.front()}, {{0.2, 0.2}},
    dealii::Point<2>(std::numeric_limits<double>::infinity(), 0.05)));
}


TEST_CASE("Fault projection tridiagonal solve")
{
  const std::vector<double> solution =
    aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(
      {2.0, 2.0}, {1.0}, {3.0, 3.0});
  REQUIRE(solution[0] == Approx(1.0));
  REQUIRE(solution[1] == Approx(1.0));

  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(
    {1.0, 1.0}, {1.0}, {1.0, 1.0}));
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(
    {2.0, 2.0}, {1.0}, {3.0}));
  REQUIRE_THROWS(aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(
    {2.0, 2.0}, {1.0}, {3.0, std::numeric_limits<double>::infinity()}));
}


TEST_CASE("Fault projection MPI reduction reproduces a constant field")
{
  const double rank_weight = 1.0 + dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<double> local_values(5, 0.0);
  for (const double xi : {0.25, 0.75})
    {
      const double shape[2] = {1.0-xi, xi};
      local_values[0] += rank_weight * shape[0] * shape[0];
      local_values[1] += rank_weight * shape[1] * shape[1];
      local_values[2] += rank_weight * shape[0] * shape[1];
      local_values[3] += rank_weight * shape[0] * 2.0;
      local_values[4] += rank_weight * shape[1] * 2.0;
    }

  std::vector<double> global_values(local_values.size());
  aspect::Utilities::MPI::sum(local_values, MPI_COMM_WORLD, global_values);
  const std::vector<double> solution =
    aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(
      {global_values[0], global_values[1]}, {global_values[2]},
      {global_values[3], global_values[4]});
  REQUIRE(solution[0] == Approx(2.0));
  REQUIRE(solution[1] == Approx(2.0));
}
