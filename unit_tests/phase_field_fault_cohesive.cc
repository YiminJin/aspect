/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/material_model/phase_field_fault.h>

#include <deal.II/base/quadrature_lib.h>

namespace
{
  using TestAccess =
    aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;
}



TEST_CASE("Common cohesive law evaluates traction and fixed-profile localization",
          "[phase_field_fault_cohesive]")
{
  const double beta = 0.75;
  const double kappa = 4.0;
  const double I_h = 2.5;
  const double previous_traction = 3.0;
  const double slip_rate = 1.25;
  const double h = 0.8;

  const auto response = TestAccess::compute_cohesive_response(
    beta, kappa, I_h, I_h, previous_traction, slip_rate, h, h);

  CHECK(response.cohesive_traction
        == Approx((kappa*slip_rate + beta*I_h*previous_traction)/I_h));
  CHECK(response.localization_factor == Approx(h/I_h));
  CHECK(response.history_correction == Approx(0.0).margin(1.e-14));
  CHECK(response.crack_strain_rate == Approx(h/I_h*slip_rate));
}



TEST_CASE("Exact cohesive history correction preserves integrated slip",
          "[phase_field_fault_cohesive]")
{
  const double beta = 0.83;
  const double kappa = 7.5;
  const double previous_traction = 11.0;
  const double slip_rate = 2.25;
  const double current_amplitude = 3.0;
  const double current_width = 0.7;
  const double previous_amplitude = 1.8;
  const double previous_width = 1.1;
  const double current_I_h = current_amplitude * current_width
                             * std::sqrt(dealii::numbers::PI);
  const double previous_I_h = previous_amplitude * previous_width
                              * std::sqrt(dealii::numbers::PI);

  const dealii::QGauss<1> quadrature(100);
  const double half_width = 10.0 * std::max(current_width, previous_width);
  double integrated_history = 0.0;
  double integrated_crack_strain_rate = 0.0;
  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const double zeta = half_width * (2.0*quadrature.point(q)[0]-1.0);
      const double weight = 2.0*half_width*quadrature.weight(q);
      const double current_h = current_amplitude
                               * std::exp(-zeta*zeta/(current_width*current_width));
      const double previous_h = previous_amplitude
                                * std::exp(-zeta*zeta/(previous_width*previous_width));
      const auto response = TestAccess::compute_cohesive_response(
        beta, kappa, current_I_h, previous_I_h, previous_traction,
        slip_rate, current_h, previous_h);
      integrated_history += weight * response.history_correction;
      integrated_crack_strain_rate += weight * response.crack_strain_rate;
    }

  CHECK(std::abs(integrated_history) <= 2.e-10);
  CHECK(integrated_crack_strain_rate == Approx(slip_rate).epsilon(2.e-10));

  const double zeta = 0.37;
  const double current_h = current_amplitude
                           * std::exp(-zeta*zeta/(current_width*current_width));
  const double previous_h = previous_amplitude
                            * std::exp(-zeta*zeta/(previous_width*previous_width));
  const double perturbation = 1.e-6;
  const double plus = TestAccess::compute_cohesive_response(
    beta, kappa, current_I_h, previous_I_h, previous_traction,
    slip_rate+perturbation, current_h, previous_h).crack_strain_rate;
  const double minus = TestAccess::compute_cohesive_response(
    beta, kappa, current_I_h, previous_I_h, previous_traction,
    slip_rate-perturbation, current_h, previous_h).crack_strain_rate;
  CHECK((plus-minus)/(2.0*perturbation)
        == Approx(current_h/current_I_h).epsilon(1.e-9));
}



TEST_CASE("Common cohesive law rejects inadmissible persistent state",
          "[phase_field_fault_cohesive]")
{
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    1.1, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0));
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0));
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0));
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    0.5, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0));
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    0.5, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0));
  CHECK_THROWS(TestAccess::compute_cohesive_response(
    0.5, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0));
}
