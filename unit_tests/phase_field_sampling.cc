/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/reconstructed_fault.h>

namespace
{
  struct RestoreDealIIExceptions
  {
    RestoreDealIIExceptions()
    {
      dealii::deal_II_exceptions::disable_abort_on_exception();
    }

    ~RestoreDealIIExceptions()
    {
      dealii::deal_II_exceptions::enable_abort_on_exception();
    }
  };
}


TEST_CASE("Fault reconstruction resamples nonuniform input by arc length")
{
  const std::vector<dealii::Point<2>> input =
  {
    dealii::Point<2>(0.0, 0.0),
    dealii::Point<2>(0.1, 0.0),
    dealii::Point<2>(1.0, 0.0)
  };

  const auto points =
    aspect::ReconstructedFaultUtilities::resample_reference_fault(input, 0.26);

  REQUIRE(points.size() == 5);
  for (unsigned int i = 0; i < points.size(); ++i)
    {
      REQUIRE(points[i][0] == Approx(0.25 * i));
      REQUIRE(points[i][1] == Approx(0.0));
    }
}


TEST_CASE("Fault reconstruction parameters have documented defaults")
{
  dealii::ParameterHandler prm;
  aspect::ReconstructedFaultManager<2>::declare_parameters(prm);

  prm.enter_subsection("Fault reconstruction");
  REQUIRE(prm.get_double("Structural point spacing") == Approx(1000.0));
  REQUIRE(prm.get_double("Ridge coefficient") == Approx(1.0));
  prm.leave_subsection();
}


TEST_CASE("Fault reconstruction ridge solve preserves a constant normal shift")
{
  const std::vector<double> matrix =
  {
    2.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 0.0,
    0.0, 0.0, 0.0, 2.0
  };
  const std::vector<double> rhs(4, 0.6);
  const auto offsets =
    aspect::ReconstructedFaultUtilities::solve_normal_offsets(matrix, rhs, 2.0, 1.0);
  for (const double offset : offsets)
    REQUIRE(offset == Approx(0.3));
}


TEST_CASE("Fault reconstruction normalization is insensitive to phase-field maturity")
{
  const std::vector<double> mature_matrix = {2.0, 0.0, 0.0, 2.0};
  const std::vector<double> mature_rhs = {0.4, 0.4};
  const std::vector<double> partial_matrix = {0.5, 0.0, 0.0, 0.5};
  const std::vector<double> partial_rhs = {0.1, 0.1};
  const auto mature = aspect::ReconstructedFaultUtilities::solve_normal_offsets(
    mature_matrix, mature_rhs, 2.0, 1.0);
  const auto partial = aspect::ReconstructedFaultUtilities::solve_normal_offsets(
    partial_matrix, partial_rhs, 0.5, 1.0);
  REQUIRE(partial == mature);
  REQUIRE(partial[0] == Approx(0.2));
}


TEST_CASE("Prescribed-fault file separates connected faults")
{
  const std::string contents =
    "# x y phi_hat\n"
    "0 0 0.6\n"
    "1 0 0.7 # first fault\n"
    "---\n"
    "2 1 0.4\n"
    "3 1 0.5\n"
    "4 2 0.6\n";

  const auto faults =
    aspect::ReconstructedFaultUtilities::parse_prescribed_faults<2>(contents,
                                                                     "test.faults");
  REQUIRE(faults.size() == 2);
  REQUIRE(faults[0].vertices.size() == 2);
  REQUIRE(faults[1].vertices.size() == 3);
  REQUIRE(faults[0].vertices[1] == dealii::Point<2>(1.0, 0.0));
  REQUIRE(faults[1].core_phase_field_values[2] == Approx(0.6));
}


TEST_CASE("Prescribed-fault file rejects empty and underspecified faults")
{
  const RestoreDealIIExceptions restore_exceptions;
  REQUIRE_THROWS_WITH(
    aspect::ReconstructedFaultUtilities::parse_prescribed_faults<2>(
      "0 0 0.5\n---\n1 0 0.5\n2 0 0.5\n", "bad.faults"),
    Contains("contains a fault"));
  REQUIRE_THROWS_WITH(
    aspect::ReconstructedFaultUtilities::parse_prescribed_faults<2>(
      "0 0\n1 0 0.5\n", "bad.faults"),
    Contains("Line 1 of prescribed-fault file"));
}
