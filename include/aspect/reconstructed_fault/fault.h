/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_fault_h
#define _aspect_reconstructed_fault_fault_h

#include <aspect/global.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <cstdint>
#include <vector>

namespace aspect
{
  template <int dim>
  class ReconstructedFaultManager;

  /**
   * An application-owned representation of a reconstructed fault.
   *
   * In two dimensions, the fault is an ordered polyline. Consecutive
   * vertices define the fault cells implicitly: cell @p i connects vertices
   * @p i and @p i+1. Committed vertices are append-only and can only be
   * accessed through const interfaces.
   *
   * The container is independent of the bulk mesh, particles, phase-field
   * reconstruction, and particular constitutive models. Runtime-defined
   * vertex properties can store material-independent fault data.
   */
  template <int dim>
  class ReconstructedFault
  {
    public:
      /** Construct an empty reconstructed fault. */
      ReconstructedFault() = default;

      /** Construct a fault from an ordered sequence of committed vertices. */
      explicit ReconstructedFault(const std::vector<Point<dim>> &vertices);

      /** Return whether the fault contains no vertices. */
      bool
      empty() const;

      /** Return the number of fault vertices. */
      unsigned int
      n_vertices() const;

      /** Return the number of fault cells. */
      unsigned int
      n_cells() const;

      /** Return vertex @p index. */
      const Point<dim> &
      vertex(const unsigned int index) const;

      /** Return the complete ordered sequence of fault vertices. */
      const std::vector<Point<dim>> &
      get_vertices() const;

      /** Return all property components stored at vertex @p vertex_index. */
      ArrayView<double>
      get_properties(const unsigned int vertex_index);

      /** Return all property components stored at vertex @p vertex_index. */
      ArrayView<const double>
      get_properties(const unsigned int vertex_index) const;

      /**
       * Return whether one generic property component has been assigned a
       * value rather than retaining the container's initialization sentinel.
       */
      bool
      property_value_is_initialized(const unsigned int vertex_index,
                                    const unsigned int component_index) const;

      /** Append one committed vertex to the fault. */
      void
      append_vertex(const Point<dim> &vertex);

      /**
       * Append an ordered sequence of committed vertices to the fault.
       * Appending an empty sequence does not change the geometry version.
       */
      void
      append_vertices(const std::vector<Point<dim>> &new_vertices);

      /**
       * Return the geometry version. Each non-empty append operation advances
       * this counter once.
       */
      std::uint64_t
      geometry_version() const;

    private:
      friend class ReconstructedFaultManager<dim>;
      friend class boost::serialization::access;

      template <class Archive>
      void serialize(Archive &ar, const unsigned int)
      {
        ar &vertices;
        ar &n_property_components;
        ar &property_values;
        ar &current_geometry_version;
      }

      void initialize_properties(const unsigned int n_components);

      std::vector<Point<dim>> vertices;
      unsigned int n_property_components = 0;
      std::vector<double> property_values;
      std::uint64_t current_geometry_version = 0;
  };
}

#endif
