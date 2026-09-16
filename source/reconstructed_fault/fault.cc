/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/fault.h>

#include <deal.II/base/signaling_nan.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace aspect
{
  namespace
  {
    bool
    is_uninitialized_property_sentinel(const double value)
    {
      static_assert(sizeof(double) == sizeof(std::uint64_t));
      static_assert(std::numeric_limits<double>::is_iec559);

      std::uint64_t bits;
      std::uint64_t sentinel_bits;
      const double sentinel = numbers::signaling_nan<double>();
      std::memcpy(&bits, &value, sizeof(bits));
      std::memcpy(&sentinel_bits, &sentinel, sizeof(sentinel_bits));
      return bits == sentinel_bits;
    }


    template <int dim>
    void
    validate_fault_vertex(const Point<dim> &vertex)
    {
      for (unsigned int d = 0; d < dim; ++d)
        AssertThrow(std::isfinite(vertex[d]),
                    ExcMessage("Reconstructed-fault vertices must be finite."));
    }


    template <int dim>
    void
    validate_fault_segment(const Point<dim> &first,
                           const Point<dim> &second)
    {
      const double length_squared = (second-first).norm_square();
      AssertThrow(std::isfinite(length_squared) && length_squared > 0.0,
                  ExcMessage("A reconstructed fault contains a degenerate segment."));
    }
  }

  template <int dim>
  ReconstructedFault<dim>::ReconstructedFault(const std::vector<Point<dim>> &initial_vertices)
    : vertices(initial_vertices)
  {
    for (const Point<dim> &vertex : vertices)
      validate_fault_vertex(vertex);
    for (unsigned int segment = 0; segment + 1 < vertices.size(); ++segment)
      validate_fault_segment(vertices[segment], vertices[segment+1]);
  }


  template <int dim>
  bool
  ReconstructedFault<dim>::empty() const
  {
    return vertices.empty();
  }


  template <int dim>
  unsigned int
  ReconstructedFault<dim>::n_vertices() const
  {
    return vertices.size();
  }


  template <int dim>
  unsigned int
  ReconstructedFault<dim>::n_cells() const
  {
    return vertices.empty() ? 0 : vertices.size() - 1;
  }


  template <int dim>
  const Point<dim> &
  ReconstructedFault<dim>::vertex(const unsigned int index) const
  {
    AssertIndexRange(index, vertices.size());
    return vertices[index];
  }


  template <int dim>
  const std::vector<Point<dim>> &
  ReconstructedFault<dim>::get_vertices() const
  {
    return vertices;
  }


  template <int dim>
  void
  ReconstructedFault<dim>::initialize_properties(const unsigned int n_components)
  {
    n_property_components = n_components;
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
  }


  template <int dim>
  ArrayView<double>
  ReconstructedFault<dim>::get_properties(const unsigned int vertex_index)
  {
    AssertIndexRange(vertex_index, vertices.size());
    return make_array_view(property_values.begin() + vertex_index * n_property_components,
                           property_values.begin() + (vertex_index + 1) * n_property_components);
  }


  template <int dim>
  ArrayView<const double>
  ReconstructedFault<dim>::get_properties(const unsigned int vertex_index) const
  {
    AssertIndexRange(vertex_index, vertices.size());
    return make_array_view(property_values.cbegin() + vertex_index * n_property_components,
                           property_values.cbegin() + (vertex_index + 1) * n_property_components);
  }


  template <int dim>
  bool
  ReconstructedFault<dim>::property_value_is_initialized(
    const unsigned int vertex_index,
    const unsigned int component_index) const
  {
    AssertIndexRange(component_index, n_property_components);
    return !is_uninitialized_property_sentinel(
             get_properties(vertex_index)[component_index]);
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertex(const Point<dim> &new_vertex)
  {
    validate_fault_vertex(new_vertex);
    if (!vertices.empty())
      validate_fault_segment(vertices.back(), new_vertex);

    vertices.push_back(new_vertex);
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
    ++current_geometry_version;
  }


  template <int dim>
  void
  ReconstructedFault<dim>::append_vertices(const std::vector<Point<dim>> &new_vertices)
  {
    if (new_vertices.empty())
      return;

    for (const Point<dim> &vertex : new_vertices)
      validate_fault_vertex(vertex);
    if (!vertices.empty())
      validate_fault_segment(vertices.back(), new_vertices.front());
    for (unsigned int segment = 0; segment + 1 < new_vertices.size(); ++segment)
      validate_fault_segment(new_vertices[segment], new_vertices[segment+1]);

    vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
    property_values.resize(vertices.size() * n_property_components,
                           numbers::signaling_nan<double>());
    ++current_geometry_version;
  }


  template <int dim>
  std::uint64_t
  ReconstructedFault<dim>::geometry_version() const
  {
    return current_geometry_version;
  }


}

namespace aspect
{
#define INSTANTIATE(dim) template class ReconstructedFault<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
