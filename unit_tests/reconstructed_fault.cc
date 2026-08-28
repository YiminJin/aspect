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
#include <aspect/utilities.h>

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


TEST_CASE("ReconstructedFault empty geometry")
{
  const aspect::ReconstructedFault<2> fault;

  REQUIRE(fault.empty());
  REQUIRE(fault.n_vertices() == 0);
  REQUIRE(fault.n_cells() == 0);
  REQUIRE(fault.get_vertices().empty());
  REQUIRE(fault.geometry_version() == 0);
  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(fault.vertex(0));
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
  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(fault.vertex(3));
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
  REQUIRE_THROWS(manager.get_property_index("missing"));
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
