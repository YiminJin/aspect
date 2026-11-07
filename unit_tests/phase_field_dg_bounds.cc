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
