/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/reconstructed_fault.h>

namespace
{
  class PrescribedFaultThrowOnDealIIException
  {
    public:
      PrescribedFaultThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::disable_abort_on_exception();
      }

      ~PrescribedFaultThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::enable_abort_on_exception();
      }
  };
}


TEST_CASE("Prescribed initial fault closest-point projection and core interpolation")
{
  const aspect::PrescribedInitialFault<2> fault =
  {
    {dealii::Point<2>(0.0, 0.0),
     dealii::Point<2>(2.0, 0.0),
     dealii::Point<2>(2.0, 2.0)},
    {0.2, 0.6, 0.8}
  };

  const auto first_segment =
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      fault, dealii::Point<2>(1.0, 0.5));
  REQUIRE(first_segment.first == Approx(0.5));
  REQUIRE(first_segment.second == Approx(0.4));

  const auto second_segment =
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      fault, dealii::Point<2>(2.5, 1.0));
  REQUIRE(second_segment.first == Approx(0.5));
  REQUIRE(second_segment.second == Approx(0.7));

  const auto endpoint =
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      fault, dealii::Point<2>(-1.0, 0.0));
  REQUIRE(endpoint.first == Approx(1.0));
  REQUIRE(endpoint.second == Approx(0.2));
}


TEST_CASE("Prescribed initial fault rejects invalid geometry and core values")
{
  const PrescribedFaultThrowOnDealIIException throw_on_dealii_exception;

  REQUIRE_THROWS_WITH(
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      aspect::PrescribedInitialFault<2>{{dealii::Point<2>(0.0, 0.0)}, {0.5}},
      dealii::Point<2>()),
    Contains("at least two vertices"));

  REQUIRE_THROWS_WITH(
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      aspect::PrescribedInitialFault<2>{{dealii::Point<2>(0.0, 0.0),
                                         dealii::Point<2>(1.0, 0.0)},
                                        {0.5}},
      dealii::Point<2>()),
    Contains("core phase-field value"));

  REQUIRE_THROWS_WITH(
    aspect::ReconstructedFaultUtilities::closest_point_distance_and_core_phase_field(
      aspect::PrescribedInitialFault<2>{{dealii::Point<2>(0.0, 0.0),
                                         dealii::Point<2>(0.0, 0.0)},
                                        {0.5, 0.5}},
      dealii::Point<2>()),
    Contains("segment 0 is degenerate"));
}
