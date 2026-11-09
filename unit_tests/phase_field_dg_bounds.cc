/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "common.h"
#include <aspect/phase_field/dg_bounds.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/quadrature_lib.h>

TEST_CASE("DGQ2 phase bounds detect an interior undershoot invisible at nodes")
{
  dealii::FE_DGQ<2> fe(2);
  aspect::PhaseField::Q2Bounds<2> bounds(fe);
  dealii::Vector<double> values(9);
  for (unsigned int i = 0; i < 9; ++i)
    {
      const auto x = fe.get_unit_support_points()[i];
      values[i] = std::pow(x[0] - .25, 2) - .01;
      REQUIRE(values[i] > 0.);
    }
  const auto range = bounds.enclosure(values, 0., 1.);
  REQUIRE(range.first <= -.01);
  const dealii::QGauss<2> rule(4);
  const auto mean = [&]()
  {
    double result = 0.;
    for (unsigned int q = 0; q < rule.size(); ++q)
      for (unsigned int i = 0; i < 9; ++i) result += rule.weight(q) * fe.shape_value(i, rule.point(q)) * values[i];
    return result;
  };
  const double before = mean();
  REQUIRE(bounds.limit(values, before, 0., 1.) < 1.);
  REQUIRE(std::abs(mean() - before) < 2e-15);
  for (unsigned int i = 0; i <= 100; ++i)
    for (unsigned int j = 0; j <= 10; ++j)
      {
        const dealii::Point<2> p(i / 100., j / 10.);
        double value = 0.;
        for (unsigned int k = 0; k < 9; ++k) value += values[k] * fe.shape_value(k, p);
        REQUIRE(value >= -2e-15);
        REQUIRE(value <= 1. + 2e-15);
      }
}

TEST_CASE("DGQ2 phase positive-basis roundtrip and inactive smooth bounds")
{
  dealii::FE_DGQ<2> fe(2);
  aspect::PhaseField::Q2Bounds<2> bounds(fe);
  dealii::Vector<double> original(9), restored(9);
  for (unsigned int i = 0; i < 9; ++i)
    {
      const auto x = fe.get_unit_support_points()[i];
      original[i] = .4 - .1 * std::pow(x[0] - .5, 2) - .1 * std::pow(x[1] - .5, 2);
    }
  const auto coefficients = bounds.coefficients(original);
  bounds.expansion.vmult(restored, coefficients);
  restored -= original;
  REQUIRE(restored.linfty_norm() < 1e-15);
  auto values = original;
  REQUIRE(bounds.limit(values, .4 - .2 / 12., 0., 1.) == 1.);
  REQUIRE(values == original);
  dealii::deal_II_exceptions::disable_abort_on_exception();
  REQUIRE_THROWS(bounds.limit(values, -.01, 0., 1.));
  dealii::deal_II_exceptions::enable_abort_on_exception();
}

TEST_CASE("Positive Q2 evaluation represents the same native polynomial")
{
  dealii::FE_DGQ<2> fe(2);
  aspect::PhaseField::Q2Bounds<2> basis(fe);
  dealii::Vector<double> coefficients(9),nodal(9);
  for(unsigned int i=0;i<9;++i)coefficients[i]=i%3==0?0.:std::pow(1e-3,int(i));
  basis.expansion.vmult(nodal,coefficients);
  const dealii::QGauss<2> rule(6);
  for(const auto &point:rule.get_points())
    {
      double positive=0.,native=0.,partition=0.;
      for(unsigned int i=0;i<9;++i)
        {
          const double shape=basis.basis_value(i,point);
          REQUIRE(shape>=0.);partition+=shape;
          positive+=shape*coefficients[i];native+=fe.shape_value(i,point)*nodal[i];
        }
      REQUIRE(positive>=0.);
      REQUIRE(std::abs(partition-1.)<5e-16);
      REQUIRE(std::abs(positive-native)<2e-18);
    }
}

#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

TEST_CASE("Native DG Bernstein interpolation FEValues traces and transfer")
{
  using namespace dealii;
  for (unsigned int degree : {0u,1u,2u,3u})
    {
      aspect::FE_DGQBernstein<2> bernstein(degree);
      FE_DGQ<2> nodal(degree);
      const unsigned int n=nodal.n_dofs_per_cell();
      REQUIRE(bernstein.n_dofs_per_face()==0);
      REQUIRE(bernstein.conforms(FiniteElementData<2>::L2));
      REQUIRE(bernstein.get_unit_support_points().empty());
      REQUIRE(bernstein.clone()->get_name()==bernstein.get_name());
      REQUIRE(FETools::get_fe_by_name<2>(bernstein.get_name())->get_name()==bernstein.get_name());
      FullMatrix<double> E(n,n), inverse(n,n), product(n,n);
      nodal.get_interpolation_matrix(bernstein,E);
      bernstein.get_interpolation_matrix(nodal,inverse);
      inverse.mmult(product,E);
      for(unsigned int i=0;i<n;++i)for(unsigned int j=0;j<n;++j)
        REQUIRE(std::abs(product(i,j)-(i==j))<2e-13);
      Vector<double> b(n),a(n),restored(n);
      for(unsigned int i=0;i<n;++i)b[i]=.2+.01*i;
      E.vmult(a,b);
      std::vector<Vector<double>> samples(n,Vector<double>(1));
      std::vector<double> coefficients(n);
      for(unsigned int i=0;i<n;++i)samples[i][0]=a[i];
      bernstein.convert_generalized_support_point_values_to_dof_values(samples,coefficients);
      for(unsigned int i=0;i<n;++i)REQUIRE(std::abs(coefficients[i]-b[i])<2e-13);
      const auto modes=bernstein.get_constant_modes();
      for(unsigned int i=0;i<n;++i)REQUIRE(modes.first(0,i));
      const QGauss<2> child_rule(4);
      for(unsigned int child=0;child<4;++child)
        {
          Vector<double> fine(n),coarse(n);
          bernstein.get_prolongation_matrix(child).vmult(fine,b);
          for(const auto &point:child_rule.get_points())
            {
              Point<2> parent_point;
              for(unsigned int d=0;d<2;++d)parent_point[d]=.5*(point[d]+((child>>d)&1));
              double child_value=0.,parent_value=0.;
              for(unsigned int j=0;j<n;++j)
                {child_value+=fine[j]*bernstein.shape_value(j,point);parent_value+=b[j]*bernstein.shape_value(j,parent_point);}
              REQUIRE(std::abs(child_value-parent_value)<3e-13);
            }
          bernstein.get_restriction_matrix(child).vmult(coarse,fine);
          restored+=coarse;
        }
      restored-=b;
      REQUIRE(restored.linfty_norm()<3e-12);
      // FEValues must evaluate the same polynomial and physical gradient on
      // affine and bilinear cells; no nodal interpretation of Bernstein DoFs.
      for(bool skew : {false,true})
        {
          Triangulation<2> mesh;GridGenerator::hyper_cube(mesh);
          if(skew){mesh.begin_active()->vertex(3)[0]+=.2;mesh.begin_active()->vertex(3)[1]+=.1;}
          MappingQ1<2> mapping;
          QGauss<2> rule(5);
          FEValues<2> fb(mapping,bernstein,rule,update_values|update_gradients|update_JxW_values);
          FEValues<2> fn(mapping,nodal,rule,update_values|update_gradients|update_JxW_values);
          fb.reinit(mesh.begin_active());fn.reinit(mesh.begin_active());
          for(unsigned int q=0;q<rule.size();++q)
            {
              double vb=0.,vn=0.,one=0.;Tensor<1,2> gb,gn;
              for(unsigned int i=0;i<n;++i)
                {vb+=b[i]*fb.shape_value(i,q);vn+=a[i]*fn.shape_value(i,q);one+=fb.shape_value(i,q);
                 gb+=b[i]*fb.shape_grad(i,q);gn+=a[i]*fn.shape_grad(i,q);}
              REQUIRE(std::abs(vb-vn)<3e-13);REQUIRE((gb-gn).norm()<3e-12);REQUIRE(std::abs(one-1.)<3e-14);
            }
          FEFaceValues<2> face_b(mapping,bernstein,QGauss<1>(5),update_values);
          FEFaceValues<2> face_n(mapping,nodal,QGauss<1>(5),update_values);
          for(unsigned int face=0;face<4;++face)
            {
              face_b.reinit(mesh.begin_active(),face);face_n.reinit(mesh.begin_active(),face);
              for(unsigned int q=0;q<5;++q)
                {double vb=0.,vn=0.;for(unsigned int i=0;i<n;++i)
                   {vb+=b[i]*face_b.shape_value(i,q);vn+=a[i]*face_n.shape_value(i,q);
                    if(!bernstein.has_support_on_face(i,face))REQUIRE(face_b.shape_value(i,q)==0.);}
                 REQUIRE(std::abs(vb-vn)<3e-13);}
            }
        }
    }
}

TEST_CASE("Native DG Bernstein stable endpoints mass and unchanged limiter")
{
  using namespace dealii;
  aspect::FE_DGQBernstein<2> bernstein(2);FE_DGQ<2> nodal(2);
  aspect::PhaseField::Q2Bounds<2> native_bounds(bernstein),legacy_bounds(nodal);
  FullMatrix<double> E(9,9);nodal.get_interpolation_matrix(bernstein,E);
  std::vector<Point<2>> points;
  for(double x:{0.,1e-16,.5,1.-1e-12,1.-1e-16,1.})points.emplace_back(x,x);
  Triangulation<2> mesh;GridGenerator::hyper_cube(mesh);
  FEValues<2> fe(bernstein,Quadrature<2>(points),update_values|update_gradients);
  fe.reinit(mesh.begin_active());
  for(unsigned int q=0;q<points.size();++q)for(unsigned int j=0;j<9;++j)
    {
      const double expected=legacy_bounds.basis_value(j,points[q]);
      REQUIRE(fe.shape_value(j,q)>=0.);
      REQUIRE(std::abs(fe.shape_value(j,q)-expected)<=1e-14*std::max(1e-300,expected));
    }
  FullMatrix<double> MN(9,9),MB(9,9),temp(9,9),congruent(9,9);
  for(unsigned int q=0;q<QGaussLobatto<2>(4).size();++q)
    {
      const QGaussLobatto<2> rule(4);const auto x=rule.point(q);
      for(unsigned int i=0;i<9;++i)for(unsigned int j=0;j<9;++j)
        {MN(i,j)+=rule.weight(q)*nodal.shape_value(i,x)*nodal.shape_value(j,x);
         MB(i,j)+=rule.weight(q)*bernstein.shape_value(i,x)*bernstein.shape_value(j,x);}
    }
  MN.mmult(temp,E);E.Tmmult(congruent,temp);congruent.add(-1.,MB);
  REQUIRE(congruent.frobenius_norm()<3e-16);
  for(bool overshoot:{false,true})
    {
      Vector<double> a(9),b(9),roundtrip(9);
      for(unsigned int i=0;i<9;++i)
        {const auto x=nodal.get_unit_support_points()[i];a[i]=1.-.5*std::pow(x[0]-.5,2)+(overshoot?.02:0.);}
      b=legacy_bounds.coefficients(a);
      REQUIRE(*std::max_element(b.begin(),b.end())>1.);
      const double mean=1.-.5/12.+(overshoot?.02:0.);
      const double an=legacy_bounds.limit(a,mean,0.,1.);
      const double ab=native_bounds.limit(b,mean,0.,1.);
      REQUIRE(std::abs(an-ab)<2e-14);
      if(!overshoot)REQUIRE(ab==1.);
      else REQUIRE(ab<1.);
      E.vmult(roundtrip,b);roundtrip-=a;REQUIRE(roundtrip.linfty_norm()<2e-15);
      double integrated=0.;const QGauss<2> rule(4);
      for(unsigned int q=0;q<rule.size();++q)for(unsigned int i=0;i<9;++i)
        integrated+=rule.weight(q)*bernstein.shape_value(i,rule.point(q))*b[i];
      REQUIRE(std::abs(integrated-mean)<2e-15);
    }
}
