/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "common.h"
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/utilities.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <limits>

namespace
{
  std::string material_geometry_archive(const aspect::ReconstructedFaultManager<2> &manager)
  {
    std::ostringstream out;
    {aspect::oarchive archive(out);archive<<manager;}
    return out.str();
  }

  void make_material_geometry(
    aspect::ReconstructedFaultManager<2> &manager)
  {
    manager.register_property("material age, slip and auxiliary memory",3);
    manager.add_reconstructed_fault({{0.,.4},{.8,.7},{2.,1.6}},{.2,.2,.2});
    manager.set_shear_sense(0,-1);
    manager.initialize_slip_rate(0,{1e-8,3e-8,2e-8});
    for(unsigned int i=0;i<3;++i)
      {
        auto p=manager.get_fault(0).get_properties(i);
        p[0]=20.+3.*i; p[1]=.01*i*i; p[2]=1e6*(1.+i);
      }
  }
}

TEST_CASE("Material geometry stages translation with unchanged label histories", "[fault_material_motion]")
{
  using namespace dealii;
  aspect::ReconstructedFaultManager<2> manager;
  make_material_geometry(manager);
  const auto before=material_geometry_archive(manager);
  const auto vertices=manager.get_fault(0).get_vertices();
  // Populate the old spatial index before either trial. A stale association
  // must not follow a copied owner or survive publication/rollback.
  REQUIRE(manager.project_to_normal_profiles({.4,.55}).active);
  for(const double direction:{-1.,1.})
    {
      auto shifted=vertices;
      const Tensor<1,2> displacement({direction*4.35,.125});
      for(auto &x:shifted)x+=displacement;
      auto candidate=manager.stage_material_motion({shifted},{{.2,.2,.2}});
      REQUIRE(material_geometry_archive(manager)==before);
      REQUIRE(candidate->get_fault(0).geometry_version()==manager.get_fault(0).geometry_version()+1);
      REQUIRE(candidate->get_shear_sense(0)==-1);
      REQUIRE(candidate->get_slip_rate(0)==manager.get_slip_rate(0));
      for(unsigned int j=0;j<2;++j)
        for(const double xi:{.21,.63})
          {
            auto tangent=vertices[j+1]-vertices[j];tangent/=tangent.norm();
            const Tensor<1,2> normal({-tangent[1],tangent[0]});
            const Point<2> x=shifted[j]+xi*(shifted[j+1]-shifted[j])+.03*normal;
            const auto p=candidate->project_to_normal_profiles(x);
            REQUIRE(p.active);
            REQUIRE(p.segment_index==j);
            REQUIRE(p.xi==Approx(xi).margin(2e-14));
            REQUIRE(p.signed_distance==Approx(.03).margin(2e-14));
          }
      for(unsigned int i=0;i<3;++i)
        for(unsigned int c=0;c<3;++c)
          REQUIRE(candidate->get_fault(0).get_properties(i)[c]==manager.get_fault(0).get_properties(i)[c]);
      const auto staged=material_geometry_archive(*candidate);
      manager.swap_prescribed_geometry(*candidate);
      REQUIRE(material_geometry_archive(manager)==staged);
      REQUIRE(manager.project_to_normal_profiles(shifted[0]+.3*(shifted[1]-shifted[0])).active);
      REQUIRE_FALSE(manager.project_to_normal_profiles({.4,.55}).active);
      manager.swap_prescribed_geometry(*candidate);
      REQUIRE(material_geometry_archive(manager)==before);
      REQUIRE(manager.project_to_normal_profiles({.4,.55}).active);
      aspect::ReconstructedFaultManager<2> restored;
      std::istringstream in(staged);
      {aspect::iarchive archive(in);archive>>restored;}
      REQUIRE(material_geometry_archive(restored)==staged);
      REQUIRE(restored.get_fault(0).get_vertices()==shifted);
      REQUIRE(restored.get_slip_rate(0)==manager.get_slip_rate(0));
    }
  auto identity=manager.stage_material_motion({vertices},{{.2,.2,.2}});
  REQUIRE(material_geometry_archive(*identity)==before);
}

TEST_CASE("Material geometric normals follow deformation rather than stress spin", "[fault_material_motion]")
{
  using namespace dealii;
  aspect::ReconstructedFaultManager<2> manager;
  make_material_geometry(manager);
  auto moved=manager.get_fault(0).get_vertices();
  constexpr double shear=.65;
  for(auto &x:moved)x[0]+=shear*x[1];
  auto candidate=manager.stage_material_motion({moved},{{.3,.3,.3}});
  double max_normal_error=0.;
  for(unsigned int j=0;j<2;++j)
    {
      auto t=manager.get_fault(0).vertex(j+1)-manager.get_fault(0).vertex(j);t/=t.norm();
      // Independent F^{-T} n for F=[[1,shear],[0,1]]. This is a material
      // surface map, not exp(dt*skew(L)).
      Tensor<1,2> n({-t[1],t[0]});
      n[1]-=shear*n[0];n/=n.norm();
      auto current_t=moved[j+1]-moved[j];current_t/=current_t.norm();
      const Tensor<1,2> current_n({-current_t[1],current_t[0]});
      max_normal_error=std::max(max_normal_error,(current_n-n).norm());
      const Point<2> x=moved[j]+.37*(moved[j+1]-moved[j])+.02*n;
      const auto p=candidate->project_to_normal_profiles(x);
      REQUIRE(p.active);
      REQUIRE(p.segment_index==j);
      REQUIRE(p.xi==Approx(.37).margin(2e-14));
      REQUIRE(p.signed_distance==Approx(.02).margin(2e-14));
    }
  REQUIRE(max_normal_error<5e-15);
  std::cout<<"Material geometry shear: max inverse-transpose normal error="<<max_normal_error<<std::endl;
}

TEST_CASE("Invalid or nonlinear material motion cannot change accepted geometry", "[fault_material_motion]")
{
  using namespace dealii;
  aspect::ReconstructedFaultManager<2> manager;
  make_material_geometry(manager);
  const auto before=material_geometry_archive(manager);
  const auto v=manager.get_fault(0).get_vertices();
  deal_II_exceptions::disable_abort_on_exception();
  REQUIRE_THROWS(manager.stage_material_motion({},{}));
  REQUIRE_THROWS(manager.stage_material_motion({{v[0],v[1]}},{{.2,.2}}));
  REQUIRE_THROWS(manager.stage_material_motion({v},{{.2,0.,.2}}));
  auto invalid=v;invalid[1]=invalid[0];
  REQUIRE_THROWS(manager.stage_material_motion({invalid},{{.2,.2,.2}}));
  invalid=v;invalid[2][1]=std::numeric_limits<double>::infinity();
  REQUIRE_THROWS(manager.stage_material_motion({invalid},{{.2,.2,.2}}));
  REQUIRE(material_geometry_archive(manager)==before);
  manager.begin_slip_rate_nonlinear_solve();
  REQUIRE_THROWS(manager.stage_material_motion({v},{{.2,.2,.2}}));
  manager.rollback_slip_rate_nonlinear_solve();
  REQUIRE(material_geometry_archive(manager)==before);
  deal_II_exceptions::enable_abort_on_exception();
}
