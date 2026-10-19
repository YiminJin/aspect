/* Copyright (C) 2026 by the authors of ASPECT. GPL-2.0-or-later. */
#include "common.h"
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/reconstructed_fault/normal_filter_internal.h>
#include <aspect/reconstructed_fault/sparse_coupling.h>
#include <deal.II/base/quadrature_lib.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

TEST_CASE("Indexed normal-profile search preserves exact native admission", "[fault_projection_index]")
{
  using namespace dealii;
  using namespace aspect;
  const auto compare=[](const std::vector<ReconstructedFault<2>> &faults,
                        const std::vector<std::vector<double>> &widths,
                        const std::vector<Point<2>> &points)
  {
    ReconstructedFaultUtilities::NormalProfileIndex<2> index(faults,widths);
    for(const auto &p:points)
      {
        const auto a=ReconstructedFaultUtilities::project_to_normal_profiles(faults,widths,p),b=index.project(p);
        REQUIRE(a.active==b.active);
        if(a.active)
          {REQUIRE(a.fault_index==b.fault_index);REQUIRE(a.segment_index==b.segment_index);REQUIRE(a.xi==b.xi);REQUIRE(a.signed_distance==b.signed_distance);}
      }
  };
  // Nonuniform spacing, a bend, endpoints and one-ULP neighbors at knots/widths.
  std::vector<ReconstructedFault<2>> faults{ReconstructedFault<2>({Point<2>(0,0),Point<2>(.01,.03),Point<2>(.8,2.4),Point<2>(2,3.)})};
  std::vector<std::vector<double>> widths{{.2,.3,.1,.4}};
  std::vector<Point<2>> points;
  for(const auto &x:faults[0].get_vertices())
    for(int d=0;d<2;++d)
      for(double sign:{-1.,0.,1.})
        {auto p=x;if(sign)p[d]=std::nextafter(p[d],sign*std::numeric_limits<double>::infinity());points.push_back(p);}
  std::mt19937 gen(713);std::uniform_real_distribution<double> uniform(-.5,3.5);
  for(unsigned int i=0;i<4000;++i)points.emplace_back(uniform(gen),uniform(gen));
  compare(faults,widths,points);
  // Exact width edges and arithmetic fallback for extremely distant points.
  compare({ReconstructedFault<2>({Point<2>(0,0),Point<2>(1,0),Point<2>(4,0)})},{{1.,1.,1.}},
          {Point<2>(.2,1),Point<2>(.2,std::nextafter(1.,2.)),Point<2>(1,0),Point<2>(4,0),Point<2>(4.01,0),Point<2>(1e100,1e100)});
  // The same overlap rejection is mandatory; the tree must not hide a fault.
  deal_II_exceptions::disable_abort_on_exception();
  faults={ReconstructedFault<2>({Point<2>(0,0),Point<2>(1,0)}),ReconstructedFault<2>({Point<2>(0,.1),Point<2>(1,.1)})};
  widths={{1,1},{1,1}};
  ReconstructedFaultUtilities::NormalProfileIndex<2> index(faults,widths);
  REQUIRE_THROWS(ReconstructedFaultUtilities::project_to_normal_profiles(faults,widths,Point<2>(.5,.05)));
  REQUIRE_THROWS(index.project(Point<2>(.5,.05)));
  deal_II_exceptions::enable_abort_on_exception();
}

TEST_CASE("Stationary BP3 full-node projection and filtered coupling kernels", "[stationary_dg_kernel]")
{
  using namespace dealii;using namespace aspect;
  using Clock=std::chrono::steady_clock;
  const auto seconds=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
  // Actual full-geometry anchors and the manager's segment-by-segment 10 m
  // resampling, preserving the 15/18 km material anchors. No bulk mesh/solve.
  const std::vector<Point<2>> anchors{{28867.513459481292,0},{20000,15358.983848622454},{9000,34411.542731880108},{7500,37009.618943233421},{0,50000}};
  std::vector<Point<2>> vertices;
  for(unsigned int s=0;s+1<anchors.size();++s)
    {
      const auto v=ReconstructedFaultUtilities::resample_reference_fault<2>({anchors[s],anchors[s+1]},10.);
      vertices.insert(vertices.end(),v.begin()+(s>0),v.end());
    }
  REQUIRE(vertices.size()==5776);
  const std::vector<ReconstructedFault<2>> faults{ReconstructedFault<2>(vertices)};
  const std::vector<std::vector<double>> widths{std::vector<double>(vertices.size(),45.)};
  auto start=Clock::now();ReconstructedFaultUtilities::NormalProfileIndex<2> index(faults,widths);
  const double setup=seconds(start);
  std::vector<Point<2>> points;
  for(unsigned int j=0;j<vertices.size();j+=2)
    {
      const auto edge=vertices[std::min<unsigned int>(j+1,vertices.size()-1)]-vertices[j>0 && j+1==vertices.size() ? j-1 : j];
      const auto tangent=edge/edge.norm();const Tensor<1,2> normal({-tangent[1],tangent[0]});
      for(double offset:{0.,44.999,45.001})points.push_back(vertices[j]+offset*normal);
    }
  std::vector<ReconstructedFaultUtilities::NormalProfileProjection> reference;
  start=Clock::now();for(const auto &p:points)reference.push_back(ReconstructedFaultUtilities::internal::project_to_normal_profiles_unchecked(faults,widths,p));
  const double reference_seconds=seconds(start);
  start=Clock::now();std::vector<ReconstructedFaultUtilities::NormalProfileProjection> fast;
  for(const auto &p:points)fast.push_back(index.project(p));
  const double indexed_seconds=seconds(start);
  for(unsigned int i=0;i<points.size();++i)
    {REQUIRE(reference[i].active==fast[i].active);if(reference[i].active){REQUIRE(reference[i].segment_index==fast[i].segment_index);REQUIRE(reference[i].xi==fast[i].xi);REQUIRE(reference[i].signed_distance==fast[i].signed_distance);}}

  // Manufactured positive work samples exercise the EXACT filtered algebra at
  // the full node count. These are not full-grid native quadrature/accuracy data;
  // the independent native frozen B/G/C fixture checks its actual work measure.
  const unsigned int n=vertices.size();std::vector<double> mass(n),edge(n-1),stiff(n),stiff_edge(n-1),mu_mass(n),mu_edge(n-1);
  aspect::internal::FaultSparseCoupling::Entries shear_entries,normal_entries;
  struct Sample{unsigned int j;double xi,w,mu,shear,normal;};std::vector<Sample> samples;
  const QGauss<1> gauss(3);
  for(unsigned int j=0;j+1<n;++j)
    for(unsigned int q=0;q<gauss.size();++q)
      {
        const double xi=gauss.point(q)[0],a=1-xi,b=xi,length=vertices[j].distance(vertices[j+1]),w=gauss.weight(q)*length;
        const double mu=.6+.07*std::sin(.013*(j+xi)),s=std::cos(.008*(j+xi)),r=std::sin(.007*(j+xi));
        samples.push_back({j,xi,w,mu,s,r});
        mass[j]+=w*a*a;mass[j+1]+=w*b*b;edge[j]+=w*a*b;
        stiff[j]+=w/(length*length);stiff[j+1]+=w/(length*length);stiff_edge[j]-=w/(length*length);
        mu_mass[j]+=w*mu*a*a;mu_mass[j+1]+=w*mu*b*b;mu_edge[j]+=w*mu*a*b;
        for(unsigned int end=0;end<2;++end)
          {const double weight=w*(end ? b : a);shear_entries[{j+end,3*j+q}]+=weight*s;normal_entries[{j+end,3*j+q}]+=weight*r;}
      }
  start=Clock::now();aspect::internal::FaultSparseCoupling S,N;S.build(shear_entries);N.build(normal_entries);
  aspect::internal::FaultNormalFilter filter(mass,edge,stiff,stiff_edge,20.,0);const double coupling_setup=seconds(start);
  std::vector<double> x(3*(n-1));for(unsigned int i=0;i<x.size();++i)x[i]=std::cos(.0013*i);
  std::vector<double> normal(n),direct(n),factored(n),tmp(n);N.add(x,normal);const auto z=filter.solve(normal);
  start=Clock::now();
  for(unsigned int i=0;i<samples.size();++i)
    {
      const auto &p=samples[i];const double a=1-p.xi,b=p.xi;
      const double value=p.shear*x[i]-p.mu*(a*z[p.j]+b*z[p.j+1]);
      direct[p.j]+=p.w*a*value;direct[p.j+1]+=p.w*b*value;
    }
  const double direct_seconds=seconds(start);start=Clock::now();S.add(x,factored);
  for(unsigned int j=0;j<n;++j)
    {factored[j]-=mu_mass[j]*z[j];if(j)factored[j]-=mu_edge[j-1]*z[j-1];if(j+1<n)factored[j]-=mu_edge[j]*z[j+1];}
  const double factored_seconds=seconds(start);
  double error=0.,scale=0.;for(unsigned int j=0;j<n;++j){error=std::max(error,std::abs(direct[j]-factored[j]));scale=std::max(scale,std::abs(direct[j]));}
  REQUIRE(error<=2e-11*scale); // Frozen-action diagnostic tolerance; production acceptance is unchanged.
  std::cout<<std::setprecision(17)<<"Stationary DG kernel: nodes="<<n<<", queries="<<points.size()<<", projection_setup_s="<<setup
    <<", reference_s="<<reference_seconds<<", indexed_s="<<indexed_seconds<<", coupling_setup_s="<<coupling_setup
    <<", sampled_product_s="<<direct_seconds<<", factored_product_s="<<factored_seconds<<", filtered_relative_error="<<error/scale<<std::endl;
}
