/*
  Copyright (C) 2024 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */

#include <aspect/particle/interpolator/distance_weighted_average.h>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/grid/grid_tools.h>

namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template <int dim>
      void
      DistanceWeightedAverage<dim>::initialize()
      {
        grid_cache = std::make_unique<GridTools::Cache<dim>>(this->get_triangulation(), this->get_mapping());
      }



      template <int dim>
      std::vector<std::vector<double>>
      DistanceWeightedAverage<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
                                                         const std::vector<Point<dim>> &positions,
                                                         const ComponentMask &selected_properties,
                                                         const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const
      {
        const unsigned int n_interpolate_positions = positions.size();
        const unsigned int n_particle_properties = particle_handler.n_properties_per_particle();

        // Create with signaling NaNs
        std::vector<std::vector<double>> cell_properties(n_interpolate_positions,
                                                          std::vector<double>(n_particle_properties,
                                                                              numbers::signaling_nan<double>()));

        // Set requested properties to 0.0
        for (unsigned int index_positions = 0; index_positions < n_interpolate_positions; ++index_positions)
          for (unsigned int index_properties = 0; index_properties < n_particle_properties; ++index_properties)
            if (selected_properties[index_properties])
              cell_properties[index_positions][index_properties] = 0.0;

        std::set<typename Triangulation<dim>::active_cell_iterator> cell_and_neighbors;

        const auto &vertex_to_cell_map = grid_cache->get_vertex_to_cell_map();

        for (const auto v : cell->vertex_indices())
          {
            const unsigned int vertex_index = cell->vertex_index(v);
            cell_and_neighbors.insert(vertex_to_cell_map[vertex_index].begin(),
                                      vertex_to_cell_map[vertex_index].end());
          }

        // Average over all particles that are within half a cell diameter.
        // This distance strikes a balance between required number of particles per
        // cell and accuracy of the interpolation. This also assumes we can find
        // most particles within this distance in neighbor cells. If we do not find
        // some particles in range (e.g. because the neighbor is refined) this
        // will slightly affect the accuracy of the interpolation, but we accept that.
        //
        // TODO: This could be made dependent on the number of particles per cell, the more
        // particles, the smaller the interpolation range to increase accuracy.
        const double interpolation_range = 0.5 * cell->diameter();
        const double epsilon = regularization_factor * interpolation_range;

        std::vector<double> integrated_weight(n_interpolate_positions,0.0);

        for (const auto &current_cell: cell_and_neighbors)
          {
            const typename ParticleHandler<dim>::particle_iterator_range particle_range =
              particle_handler.particles_in_cell(current_cell);

            for (const auto &particle: particle_range)
              {
                const ArrayView<const double> particle_properties = particle.get_properties();
                unsigned int index_positions = 0;

                for (const auto &interpolation_point: positions)
                  {
                    const double distance = particle.get_location().distance(interpolation_point);
                    const double weight = compute_weight(distance, interpolation_range, epsilon);

                    for (unsigned int index_properties = 0; index_properties < particle_properties.size(); ++index_properties)
                      if (selected_properties[index_properties])
                        cell_properties[index_positions][index_properties] += weight * particle_properties[index_properties];

                    integrated_weight[index_positions] += weight;
                    ++index_positions;
                  }
              }
          }

        for (unsigned int index_positions = 0; index_positions < n_interpolate_positions; ++index_positions)
          {
            AssertThrow(integrated_weight[index_positions] > 0.0, ExcInternalError());

            for (unsigned int index_properties = 0; index_properties < n_particle_properties; ++index_properties)
              if (selected_properties[index_properties])
                cell_properties[index_positions][index_properties] /= integrated_weight[index_positions];
          }

        return cell_properties;
      }



        template <int dim>
        double
        DistanceWeightedAverage<dim>::compute_weight(const double distance, 
                                                     const double interpolation_range,
                                                     const double epsilon) const
        {
          switch (weight_type)
            {
              case DistanceWeightedAverage<dim>::linear:
                return std::max(1.0 - (distance / interpolation_range), 0.0);
              case DistanceWeightedAverage<dim>::reciprocal:
                return (distance < interpolation_range ? 1.0 / (distance + epsilon) : 0.0);
              case DistanceWeightedAverage<dim>::squared_reciprocal:
                return (distance < interpolation_range ? 1.0 / (distance * distance + epsilon * epsilon) : 0.0);
              case DistanceWeightedAverage<dim>::modified_shephard:
                return (distance < interpolation_range ? 
                        Utilities::fixed_power<2, double>((1.0 - (distance * distance / (interpolation_range * interpolation_range)))) 
                        / (distance * distance + epsilon * epsilon) : 
                        0.0);
              default:
                Assert(false, ExcNotImplemented());
            }

          return numbers::signaling_nan<double>();
        }



      template <int dim>
      void
      DistanceWeightedAverage<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Interpolator");
        {
          prm.enter_subsection("Distance weighted average");
          {
            prm.declare_entry("Weight type", "linear",
                              Patterns::Selection("linear|reciprocal|squared reciprocal|modified shephard"),
                              "Weight type in the distance weighted average interpolation. "
                              "The options are:\n"
                              "`linear': $w = 1 - (d / r)$;\n"
                              "`reciprocal': $w = \\frac{1}{(1 + e)r}$;\n"
                              "`squared reciprocal': $w = \\frac{1}{(1 + e^2)r^2}$;\n"
                              "`modified shephard': $w = \\frac{(1 - (d / r)^2)^2}{(1 + e^2)r^2}$.\n"
                              "In the above expressions, $d$ is the distance between the particle and the "
                              "target point, $r$ is half of the diameter of the host cell, and $e$ is "
                              "the distance regularization factor. If $r > h$, then the weight is zero.");
            prm.declare_entry("Distance regularization factor", "0.1",
                              Patterns::Double(0),
                              "If `reciprocal', `squared reciprocal' or `modified shephard' is selected as "
                              "the weight type, then we need to regularize the denominator to prevent "
                              "one-point dominance. This parameter is the ratio between the regularization "
                              "parameter and the interpolation range (half of the diameter of the host cell). "
                              "The larger this parameter is, the smoother (and more diffusive) the interpolation "
                              "will be.");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      DistanceWeightedAverage<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Interpolator");
        {
          prm.enter_subsection("Distance weighted average");
          {
            const std::string type = prm.get("Weight type");
            if (type == "linear")
              weight_type = linear;
            else if (type == "reciprocal")
              weight_type = reciprocal;
            else if (type == "squared reciprocal")
              weight_type = squared_reciprocal;
            else if (type == "modified shephard")
              weight_type = modified_shephard;
            else
              AssertThrow(false, ExcNotImplemented());

            regularization_factor = prm.get_double("Distance regularization factor");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(DistanceWeightedAverage,
                                            "distance weighted average",
                                            "Interpolates particle properties onto a vector of points using a "
                                            "distance weighed averaging method.")
    }
  }
}
