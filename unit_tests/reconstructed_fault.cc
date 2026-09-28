/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"
#include "../source/reconstructed_fault/surface_direct_internal.h"
#include "../tests/fault_surface_reference.h"
#include "../source/reconstructed_fault/normal_filter_internal.h"

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
#include <aspect/phase_field.h>

#include <limits>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

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

TEST_CASE("Fault shear sense preserves positive rates and survives restart", "[fault_shear_sense]")
{
  ThrowOnDealIIException exceptions;
  aspect::ReconstructedFaultManager<2> manager;
  const double c=.5, s=std::sqrt(3.)/2;
  manager.add_reconstructed_fault({dealii::Point<2>(c,0),dealii::Point<2>(0,s)}, {1.,1.});
  REQUIRE(manager.get_shear_sense(0)==1);
  manager.set_shear_sense(0,-1);
  manager.set_shear_sense(0,-1);
  REQUIRE_THROWS(manager.set_shear_sense(0,1));
  REQUIRE_THROWS(manager.set_shear_sense(0,0));
  manager.initialize_slip_rate(0,{1e-9,1e-9});
  const dealii::Tensor<1,2> tangent({-c,s}),normal({-s,-c});
  const auto S=manager.get_shear_sense(0)*dealii::symmetrize(dealii::outer_product(tangent,normal));
  const auto N=dealii::symmetrize(dealii::outer_product(normal,normal));
  const dealii::Tensor<1,2> down({c,-s}),right({s,c});
  const auto required=-dealii::symmetrize(dealii::outer_product(down,right));
  REQUIRE((S-required).norm()==0.);
  REQUIRE(std::abs(S*N)<1e-15);
  REQUIRE(S*S==Approx(.5));
  std::stringstream storage;
  {aspect::oarchive archive(storage);archive<<manager;}
  aspect::ReconstructedFaultManager<2> restored;
  {aspect::iarchive archive(storage);archive>>restored;}
  REQUIRE(restored.get_shear_sense(0)==-1);
  REQUIRE(restored.get_slip_rate(0)[0]==1e-9);
  restored.set_shear_sense(0,-1);
  REQUIRE_THROWS(restored.set_shear_sense(0,1));
}

TEST_CASE("Restored BP3 stationary profile matches independent completion table", "[.][bp3_restore_profile]")
{
  const aspect::PhaseField::GeometricFunction geometry(20.,1.,8./3.);
  const double m=1e5/((8./3.)*20.*(1e12/(2.*32038120320.)));
  const aspect::PhaseField::DegradationFunction degradation(1.,m);
  const aspect::PhaseField::PhaseFieldProfile profile(geometry,degradation,.6);
  std::ifstream in(aspect::Utilities::expand_ASPECT_SOURCE_DIR(
    "$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/fixtures/bp3_150x50/profile.txt"));
  unsigned int n=0;double stored_m=0.;
  REQUIRE(static_cast<bool>(in>>n>>stored_m));
  REQUIRE(stored_m==Approx(m));
  double error=0.;
  for(unsigned int i=0;i<n;++i)
    {
      double r,p,integral;
      REQUIRE(static_cast<bool>(in>>r>>p>>integral));
      error=std::max(error,std::abs(p-profile.value(r)));
    }
  std::cout<<"Restored BP3 completion profile max phase error="<<error<<std::endl;
  REQUIRE(error<2e-11);
}

TEST_CASE("Normal filter preserves work mean and attenuates generalized modes", "[fault_normal_filter]")
{
  ThrowOnDealIIException exceptions;
  const unsigned int n=9;const double h=100.,L=200.;
  std::vector<double> m(n,2*h/3),me(n-1,h/6),k(n,2/h),ke(n-1,-1/h);
  m.front()=m.back()=h/3;k.front()=k.back()=1/h;
  auto multiply=[&](const auto &d,const auto &e,const auto &x)
  {
    std::vector<double> y(n);
    for (unsigned int i=0;i<n;++i)
      {y[i]=d[i]*x[i];if(i)y[i]+=e[i-1]*x[i-1];if(i+1<n)y[i]+=e[i]*x[i+1];}
    return y;
  };
  aspect::internal::FaultNormalFilter filter(m,me,k,ke,L,0);
  REQUIRE(filter.matches(m,me,k,ke,L));REQUIRE_FALSE(filter.matches(m,me,k,ke,L/2));
  auto changed_mass=m;changed_mass[3]*=1.01;
  REQUIRE_FALSE(filter.matches(changed_mass,me,k,ke,L));
  const auto zero=multiply(k,ke,std::vector<double>(n,1.));
  for (double v:zero) REQUIRE(v==0.);
  for (const double constant:{1.,5e7})
    {
      const auto rhs=multiply(m,me,std::vector<double>(n,constant));
      const auto z=filter.solve(rhs);
      double error=0.;for(double v:z) error=std::max(error,std::abs(v-constant));
      std::cout<<"Normal filter constant="<<constant<<" max absolute error="<<std::setprecision(17)<<error<<std::endl;
      for (double v:z) REQUIRE(std::abs(v-constant)<2e-14*constant);
    }
  dealii::FullMatrix<double> dense(n),inverse(n);
  for (unsigned int i=0;i<n;++i)
    {dense(i,i)=m[i]+L*L*k[i];if(i+1<n)dense(i,i+1)=dense(i+1,i)=me[i]+L*L*ke[i];}
  inverse.invert(dense);
  double previous=1.;
  for (const unsigned int mode:{1u,4u,8u})
    {
      const double angle=mode*dealii::numbers::PI/(n-1),lambda=6*(1-std::cos(angle))/(h*h*(2+std::cos(angle)));
      const double attenuation=1/(1+L*L*lambda);REQUIRE(attenuation<previous);previous=attenuation;
      std::vector<double> v(n);for(unsigned int i=0;i<n;++i)v[i]=5e7+1000*std::cos(angle*i);
      const auto rhs=multiply(m,me,v),z=filter.solve(rhs),mz=multiply(m,me,z);
      double mean_error=0.;
      double mode_error=0.,dense_error=0.;
      for(unsigned int i=0;i<n;++i)
        {
          REQUIRE(std::abs(z[i]-5e7-1000*attenuation*std::cos(angle*i))<2e-7);
          double reference=0;for(unsigned int j=0;j<n;++j)reference+=inverse(i,j)*rhs[j];
          REQUIRE(std::abs(reference-z[i])<2e-7);
          mode_error=std::max(mode_error,std::abs(z[i]-5e7-1000*attenuation*std::cos(angle*i)));
          dense_error=std::max(dense_error,std::abs(reference-z[i]));
          mean_error+=mz[i]-rhs[i];
        }
      REQUIRE(std::abs(mean_error)<1e-4);
      std::cout<<"Normal filter mode="<<mode<<" attenuation="<<attenuation<<" mode error Pa="<<mode_error
               <<" dense error Pa="<<dense_error<<" mean integral error Pa m="<<mean_error<<std::endl;
    }
  m[3]=0.;REQUIRE_THROWS(aspect::internal::FaultNormalFilter(m,me,k,ke,L,0));
}


TEST_CASE("Prescribed fault V is private until commit and cannot be released",
          "[fault_prescribed_v]")
{
  ThrowOnDealIIException exceptions;
  aspect::ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault({{0,0},{1,0},{2,0}}, {1,1,1});
  manager.initialize_slip_rate(0, {2,2,2});
  manager.set_prescribed_slip_rates({{{2,3}}});
  manager.begin_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_slip_rate(0)[2] == 3);
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[2] == 2);
  auto active = manager.prescribed_slip_rate_mask();
  REQUIRE(active[0] == std::vector<bool>{false,false,true});
  aspect::internal::update_reconstructed_fault_active_set(
    {{2,2,3}}, {{1,1,1}}, 1., active);
  REQUIRE(active[0][2]);
  manager.begin_slip_rate_trial();
  REQUIRE_THROWS(manager.set_slip_rate_trial({{1,1,1}}, 1.));
  manager.rollback_slip_rate_trial();
  manager.rollback_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[2] == 2);
  manager.begin_slip_rate_nonlinear_solve();
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial({{1,1,0}}, .5);
  manager.accept_slip_rate_trial();
  manager.validate_slip_rate_nonlinear_commit();
  manager.commit_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_timestep_committed_slip_rate(0) == std::vector<double>{2.5,2.5,3});
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


TEST_CASE("Stage-I captured BP3 contact retains absolute evaluated and accepted rates")
{
  ThrowOnDealIIException exceptions;
  using namespace aspect;
  constexpr double base = 1.018971212482027e-9;
  constexpr double direction = -8.573688417268437e-9;
  constexpr double minimum = 1e-20;
  const double alpha = aspect::internal::reconstructed_fault_maximum_step_length(
    {{base,2e-9}}, {{direction,0.}}, {{false,true}}, minimum);
  REQUIRE(base+alpha*direction < minimum);
  REQUIRE(base+(minimum-base) < minimum);
  REQUIRE(aspect::internal::reconstructed_fault_trial_value(base,direction,alpha,minimum) == minimum);
  REQUIRE_THROWS(aspect::internal::reconstructed_fault_trial_value(base,direction,alpha*1.01,minimum));
  // The follow-up replay reaches a tip whose right node is exactly in contact.
  // Even endpoint interpolation must not reconstruct that node by subtract/add.
  constexpr double tip_left = 4.879984001041503e-13;
  ReconstructedFaultManager<2> tip;
  tip.add_reconstructed_fault({{0,0},{1,0}}, {1,1});
  tip.initialize_slip_rate(0, {tip_left,minimum});
  REQUIRE(tip_left+(minimum-tip_left) < minimum);
  REQUIRE(tip.interpolate_slip_rate(0,0,0.) == tip_left);
  REQUIRE(tip.interpolate_slip_rate(0,0,1.) == minimum);
  REQUIRE(tip.interpolate_slip_rate(0,0,.37) == Approx(tip_left+.37*(minimum-tip_left)).epsilon(1e-15));
  REQUIRE(dealii::Utilities::MPI::min(tip.interpolate_slip_rate(0,0,1.),MPI_COMM_WORLD) == minimum);
  ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault({{0,0},{1,0}}, {1,1});
  manager.initialize_slip_rate(0, {base,2e-9});
  manager.set_prescribed_slip_rates({{{1,2e-9}}});
  manager.begin_slip_rate_nonlinear_solve();
  ReconstructedFaultVector evaluated;
  unsigned int evaluations = 0;
  const auto evaluate = [&](const double step)
  {
    evaluated = {{aspect::internal::reconstructed_fault_trial_value(base,direction,step,minimum),2e-9}};
    manager.begin_slip_rate_trial();
    manager.set_slip_rate_trial_values(evaluated);
    REQUIRE(manager.get_slip_rate(0) == evaluated[0]);
    REQUIRE(dealii::Utilities::MPI::min(manager.get_slip_rate(0)[0], MPI_COMM_WORLD)
            == dealii::Utilities::MPI::max(evaluated[0][0], MPI_COMM_WORLD));
    manager.rollback_slip_rate_trial();
    REQUIRE(manager.get_slip_rate(0)[0] == base);
    return ++evaluations <= 2 ? 1.0 : .5;
  };
  const auto accept = [&](const double)
  {
    manager.begin_slip_rate_trial();
    manager.set_slip_rate_trial_values(evaluated);
    manager.accept_slip_rate_trial();
    REQUIRE(manager.get_slip_rate(0) == evaluated[0]);
  };
  auto result = aspect::internal::reconstructed_fault_armijo_line_search(alpha,4,1.,evaluate,accept);
  REQUIRE(result.accepted);
  REQUIRE(result.rejected_candidates == 2);
  REQUIRE(evaluated[0][0] == base+result.step_length*direction);
  manager.rollback_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[0] == base);
  manager.begin_slip_rate_nonlinear_solve();
  evaluations = 0;
  result = aspect::internal::reconstructed_fault_armijo_line_search(alpha,1,1.,evaluate,accept);
  REQUIRE_FALSE(result.accepted);
  REQUIRE(manager.get_slip_rate(0)[0] == base);
  manager.begin_slip_rate_trial();
  manager.set_slip_rate_trial_values({{minimum,2e-9}});
  REQUIRE_THROWS(manager.set_slip_rate_trial_values({{minimum,3e-9}}));
  REQUIRE(manager.get_slip_rate(0)[0] == minimum);
  manager.accept_slip_rate_trial();
  REQUIRE(manager.get_slip_rate(0)[0] == minimum);
  auto active = manager.prescribed_slip_rate_mask();
  aspect::internal::update_reconstructed_fault_active_set({{minimum,2e-9}},{{-minimum,0.}},minimum,active);
  REQUIRE(active[0][0]);
  active = manager.prescribed_slip_rate_mask();
  aspect::internal::update_reconstructed_fault_active_set({{minimum,2e-9}},{{minimum,0.}},minimum,active);
  REQUIRE_FALSE(active[0][0]);
  REQUIRE(aspect::internal::reconstructed_fault_trial_value(minimum,minimum,1.,minimum)==2*minimum);
  manager.validate_slip_rate_nonlinear_commit();
  manager.commit_slip_rate_nonlinear_solve();
  REQUIRE(manager.get_timestep_committed_slip_rate(0)[0] == minimum);
}


TEST_CASE("Stage-I slip-rate interpolation preserves nodal bounds", "[fault_slip_interpolation]")
{
  using aspect::ReconstructedFaultUtilities::interpolate_slip_rate;
  constexpr double minimum=1e-20, captured_xi=0.07893139508566324;
  REQUIRE((1.-captured_xi)*minimum+captured_xi*minimum < minimum);
  REQUIRE(interpolate_slip_rate(minimum,minimum,captured_xi)==minimum);
  const double near=std::nextafter(minimum,std::numeric_limits<double>::infinity());
  for (const auto endpoints:std::vector<std::pair<double,double>>{
         {minimum,minimum},{minimum,near},{near,minimum},{minimum,1e-9},{1e-9,minimum}})
    {
      aspect::ReconstructedFaultManager<2> manager;
      manager.add_reconstructed_fault({{0,0},{1,0}}, {1,1});
      manager.initialize_slip_rate(0,{endpoints.first,endpoints.second});
      for (unsigned int i=0;i<=1000;++i)
        {
          const double xi=double(i)/1000.;
          const double actual=interpolate_slip_rate(endpoints.first,endpoints.second,xi);
          const long double exact=(1.L-xi)*endpoints.first+static_cast<long double>(xi)*endpoints.second;
          REQUIRE(actual>=std::min(endpoints.first,endpoints.second));
          REQUIRE(actual<=std::max(endpoints.first,endpoints.second));
          REQUIRE(std::abs(actual-exact)<=4*std::numeric_limits<double>::epsilon()*exact);
          REQUIRE(manager.interpolate_slip_rate(0,0,xi)==actual);
        }
      REQUIRE(interpolate_slip_rate(endpoints.first,endpoints.second,0.)==endpoints.first);
      REQUIRE(interpolate_slip_rate(endpoints.first,endpoints.second,1.)==endpoints.second);
    }
  // Invalid nodal values are not silently raised to the constitutive bound.
  REQUIRE(interpolate_slip_rate(.5*minimum,.5*minimum,captured_xi)<minimum);
  REQUIRE(dealii::Utilities::MPI::min(interpolate_slip_rate(minimum,minimum,captured_xi),MPI_COMM_WORLD)==minimum);
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

TEST_CASE("Stage-I compatibility uses uncancelled operations and still rejects physical flux")
{
  using namespace dealii;
  using namespace aspect::internal;
  const ThrowOnDealIIException throw_on_dealii_exception;
  Vector<double> q(3),rhs(3);
  q[1]=q[2]=1./std::sqrt(2.);
  // The surviving residual can be tiny despite O(1) terms in its assembly.
  const double cap=1.75e-10;
  const double bound=fault_pressure_compatibility_bound(2.,64.,1e-13,12.,cap);
  REQUIRE(bound<cap);
  REQUIRE(bound>1.6047464209644725e-14);
  rhs[0]=1e-11;rhs.add(1.6047464209644725e-14,q);
  const double removed=project_compatible_fault_rhs(q,bound,rhs);
  REQUIRE(removed==Approx(1.6047464209644725e-14).epsilon(1e-12));
  REQUIRE(std::abs(q*rhs)<1e-29);
  rhs.add(1e-6,q);
  const Vector<double> saved(rhs);
  REQUIRE_THROWS(project_compatible_fault_rhs(q,bound,rhs));
  rhs-=saved;REQUIRE(rhs.l2_norm()==0.);
  REQUIRE(fault_pressure_compatibility_bound(1e20,64.,0.,12.,cap)==cap);
  REQUIRE(fault_pressure_compatibility_bound(0.,64.,0.,12.,cap)==0.);
}

TEST_CASE("Stage-I nonsymmetric pressure compatibility requires the left nullspace")
{
  using namespace dealii;
  using namespace aspect::internal;
  const ThrowOnDealIIException throw_on_dealii_exception;
  FullMatrix<double> C(2);C(0,0)=1.;C(0,1)=1.;
  Vector<double> right(2),left(2),rhs(2),check(2);
  right[0]=1./std::sqrt(2.);right[1]=-right[0];left[1]=1.;
  C.vmult(check,right);REQUIRE(check.l2_norm()==0.);
  C.Tvmult(check,left);REQUIRE(check.l2_norm()==0.);
  rhs[0]=rhs[1]=1.;
  REQUIRE(right*rhs==0.); // A right-null test alone would accept this RHS.
  REQUIRE_THROWS(project_compatible_fault_rhs(left,1e-14,rhs));
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

TEST_CASE("Mature fixed prestress and geometry survive manager checkpoint", "[mature_fault]")
{
  aspect::ReconstructedFaultManager<2> manager;
  manager.register_property("mature fault reference geometry",2);
  manager.register_property("background tractions",2);
  manager.register_property("BP3 fixed shear correction",3);
  manager.register_property("phase field fault cohesive traction",1);
  manager.add_reconstructed_fault({dealii::Point<2>(0,0),dealii::Point<2>(1,0)},{.5,.5});
  manager.initialize_slip_rate(0,{1.,2.});
  for (unsigned int v=0;v<2;++v)
    {
      auto p=manager.get_fault(0).get_properties(v);
      const std::vector<double> data={double(v),0.,20.+v,50.,1.+v,2.+v,4.+v,0.};
      std::copy(data.begin(),data.end(),p.begin());
    }
  std::stringstream storage;
  {aspect::oarchive archive(storage);archive<<manager;}
  aspect::ReconstructedFaultManager<2> restored;
  {aspect::iarchive archive(storage);archive>>restored;}
  REQUIRE(restored.has_property("mature fault reference geometry"));
  for (unsigned int v=0;v<2;++v)
    for (unsigned int c=0;c<8;++c)
      CHECK(restored.get_fault(0).get_properties(v)[c]==manager.get_fault(0).get_properties(v)[c]);
  // The restored coefficients retain a rational interior background, not
  // the linear interpolation of the endpoint background values.
  const double rational=20.5-(1.5+2.5/4.5);
  CHECK(std::abs(rational-(.5*(20.-1.-2./4.)+.5*(21.-2.-3./5.)))>1e-3);
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


TEST_CASE("Bottom bulk source continuation preserves surface admission", "[fault_bottom_source]")
{
  aspect::ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault({dealii::Point<2>(2,0),dealii::Point<2>(3,2),dealii::Point<2>(4,4)},
                                 {1.,1.,1.});
  const dealii::Point<2> wedge(1.8,.01), admitted(2.5,1.);
  REQUIRE_FALSE(manager.project_to_bulk_source(wedge).active);
  const auto before=manager.project_to_bulk_source(admitted);
  manager.enable_bottom_source_continuation(0,dealii::Point<2>(0,0),dealii::Point<2>(6,4));
  const auto continued=manager.project_to_bulk_source(wedge);
  REQUIRE(continued.active);
  REQUIRE(continued.segment_index==0);
  REQUIRE(continued.xi==0.);
  REQUIRE_FALSE(manager.project_to_normal_profiles(wedge).active);
  const auto after=manager.project_to_bulk_source(admitted);
  REQUIRE(after.active==before.active);
  REQUIRE(after.segment_index==before.segment_index);
  REQUIRE(after.xi==before.xi);
  REQUIRE(after.signed_distance==before.signed_distance);
  REQUIRE_FALSE(manager.project_to_bulk_source(dealii::Point<2>(1.8,-.01)).active);
  REQUIRE(manager.project_to_bulk_source(dealii::Point<2>(0,.01)).active);
  REQUIRE_FALSE(manager.project_to_bulk_source(dealii::Point<2>(-.01,.01)).active);
  REQUIRE_FALSE(manager.project_to_bulk_source(dealii::Point<2>(0,3.)).active);
  REQUIRE_FALSE(manager.project_to_bulk_source(dealii::Point<2>(4.2,4.1)).active);

  const dealii::Point<2> top_wedge(4.2,3.99);
  REQUIRE_FALSE(manager.project_to_bulk_source(top_wedge).active);
  manager.enable_top_source_continuation();
  const auto top=manager.project_to_bulk_source(top_wedge);
  REQUIRE(top.active);
  REQUIRE(top.segment_index==1);
  REQUIRE(top.xi==1.);
  REQUIRE_FALSE(manager.project_to_normal_profiles(top_wedge).active);
  REQUIRE_FALSE(manager.project_to_bulk_source(dealii::Point<2>(4.2,4.01)).active);
  REQUIRE(manager.project_to_bulk_source(wedge).xi==0.);
  REQUIRE(manager.project_to_bulk_source(admitted).xi==before.xi);
}


TEST_CASE("Saved bulk quadrature support audit", "[.fault_saved_support]")
{
  // Frozen-data audit only: exercise the same projection used by the Stokes
  // QP cache, including points missing from associated-only diagnostic exports.
  const char *directory=std::getenv("ASPECT_SAVED_FAULT_SUPPORT_AUDIT");
  REQUIRE(directory!=nullptr);
  const std::string root=std::string(directory)+"/";
  std::ifstream geometry(root+"geometry.txt");
  unsigned int n=0;double width=0.;
  REQUIRE(static_cast<bool>(geometry>>n>>width));
  std::vector<dealii::Point<2>> vertices(n);
  for (auto &p:vertices) REQUIRE(static_cast<bool>(geometry>>p[0]>>p[1]));
  aspect::ReconstructedFaultManager<2> manager;
  manager.add_reconstructed_fault(vertices,std::vector<double>(n,width));
  std::ifstream input(root+"points.txt");
  REQUIRE(input.good());
  std::ofstream output(root+"production_projection.csv");
  output<<std::setprecision(17)<<"id,active,fault,segment,xi,distance\n";
  unsigned int id,count=0;dealii::Point<2> point;
  while (input>>id>>point[0]>>point[1])
    {
      const auto p=manager.project_to_normal_profiles(point);
      output<<id<<','<<p.active<<','<<p.fault_index<<','<<p.segment_index<<','
            <<(p.active ? p.xi : 0.)<<','<<(p.active ? p.signed_distance : 0.)<<'\n';
      ++count;
    }
  REQUIRE(input.eof());
  REQUIRE(count>0);
  REQUIRE(output.good());
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

TEST_CASE("Pivoted surface inverse preserves indefinite free blocks", "[fault_surface_direct]")
{
  ThrowOnDealIIException exceptions;
  using FaultSurfaceDirect = aspect::Testing::FaultSurfaceInverse;
  const std::vector<std::vector<double>> diagonals={{4,3,2,4,3}, {-2,3,-4,5,-6}, {0,0}, {0,0,1}};
  const std::vector<std::vector<double>> edges={{1,.5,1,.5}, {1,2,1,.5}, {1}, {1,1}};
  for (unsigned int example=0;example<diagonals.size();++example)
    {
      const unsigned int n=diagonals[example].size();
      for (unsigned int pattern=0;pattern<(example<2 ? 5U : 1U);++pattern)
        {
          std::vector<bool> active(n,false);
          if (pattern==1) active.front()=active.back()=true;
          if (pattern==2) active[n/2]=true;
          if (pattern==3) for (unsigned int i=0;i<n;++i) active[i]=(i%2==1);
          if (pattern==4) active.assign(n,true);
          FaultSurfaceDirect pivot(diagonals[example],edges[example],active,example,true);
          FaultSurfaceDirect umf(diagonals[example],edges[example],active,example,false);
          for (unsigned int load=0;load<4;++load)
            {
              std::vector<double> rhs(n),actual,reference;
              for (unsigned int i=0;i<n;++i)
                rhs[i]=load==0 ? 0. : (load==1 ? 1. : (i%2 ? -1. : 2.))*(i+1)*load;
              // Active RHS entries are deliberately enormous and must be ignored.
              for (unsigned int i=0;i<n;++i) if (active[i]) rhs[i]=1e200;
              pivot.solve(rhs,actual); umf.solve(rhs,reference);
              for (unsigned int i=0;i<n;++i)
                {
                  REQUIRE(std::abs(actual[i]-reference[i])<=2e-12*std::max(1.,std::abs(reference[i])));
                  if (active[i]) REQUIRE(actual[i]==0.);
                  REQUIRE(dealii::Utilities::MPI::min(actual[i],MPI_COMM_WORLD)
                          ==dealii::Utilities::MPI::max(actual[i],MPI_COMM_WORLD));
                }
            }
        }
    }
  // A scalar free block is valid even when negative; a zero pivot is not.
  FaultSurfaceDirect scalar({-2},{},{false},7,true);
  std::vector<double> result;
  scalar.solve({4},result);
  REQUIRE(result[0]==-2.);
  REQUIRE_THROWS(FaultSurfaceDirect({1,1},{1},{false,false},8,true));
  REQUIRE_THROWS(FaultSurfaceDirect({0},{},{false},9,true));
  REQUIRE_THROWS(scalar.solve({std::numeric_limits<double>::quiet_NaN()},result));
  REQUIRE_THROWS(FaultSurfaceDirect({std::numeric_limits<double>::infinity()},{},{false},10,true));
}

TEST_CASE("Pivoted surface inverse retains nonsymmetric state columns", "[fault_surface_direct]")
{
  using FaultSurfaceDirect = aspect::Testing::FaultSurfaceInverse;
  const std::vector<double> diagonal={0,3,-4,5,-6},upper={1,2,.5,1},lower={-2,.25,3,-.5};
  for (const auto active:{std::vector<bool>{false,false,false,false,false},
                          std::vector<bool>{true,false,true,false,true}})
    {
      FaultSurfaceDirect pivot(diagonal,upper,active,0,true,lower);
      FaultSurfaceDirect reference(diagonal,upper,active,0,false,lower);
      std::vector<double> a,b;pivot.solve({1,2,3,4,5},a);reference.solve({1,2,3,4,5},b);
      for(unsigned int i=0;i<a.size();++i)
        {
          REQUIRE(a[i]==Approx(b[i]).epsilon(1e-12));
          if(active[i]) REQUIRE(a[i]==0.);
          else
            {
              double value=diagonal[i]*a[i];
              if(i>0)value+=lower[i-1]*a[i-1];
              if(i+1<a.size())value+=upper[i]*a[i+1];
              REQUIRE(value==Approx(i+1.).epsilon(1e-12));
            }
        }
    }
}
