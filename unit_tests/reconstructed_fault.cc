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

  const unsigned int scalar = manager.register_vertex_property("scalar", 1);
  const unsigned int vector = manager.register_vertex_property("vector", 2);

  REQUIRE(scalar == 0);
  REQUIRE(vector == 1);
  REQUIRE(manager.has_vertex_property("scalar"));
  REQUIRE_FALSE(manager.has_vertex_property("missing"));
  REQUIRE(manager.get_vertex_property_index("vector") == vector);
  REQUIRE(manager.get_vertex_property_information().size() == 2);
  REQUIRE(manager.get_vertex_property_information()[1].name == "vector");
  REQUIRE(manager.get_vertex_property_information()[1].n_components == 2);

  const ThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(manager.register_vertex_property("", 1));
  REQUIRE_THROWS(manager.register_vertex_property("zero components", 0));
  REQUIRE_THROWS(manager.register_vertex_property("scalar", 1));
  REQUIRE_THROWS(manager.get_vertex_property_index("missing"));
}
