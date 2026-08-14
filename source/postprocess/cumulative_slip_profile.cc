/*
  Copyright (C) 2018 - 2025 by the authors of the ASPECT code.

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

#include <aspect/postprocess/cumulative_slip_profile.h>
#include <aspect/material_model/phase_field_rsf.h>

#include <deal.II/grid/grid_tools.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    void
    CumulativeSlipProfile<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Cumulative slip profile");
        {
          prm.declare_entry("Sample points file", "",
                            Patterns::FileName(),
                            "Name of the file storing the sample points along the profile.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    CumulativeSlipProfile<dim>::parse_parameters(ParameterHandler &prm)
    {
      std::string input_filename;

      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Cumulative slip profile");
        {
          input_filename = prm.get("Sample points file");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Only store the sample points in root process
      sample_points.clear();
      std::string error_message;

      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          try
            {
              std::ifstream input(input_filename);
              AssertThrow(input,
                          ExcMessage("Could not open sample-point file <"
                                     + input_filename + ">."));

              std::string line;
              unsigned int line_number = 0;

              while (std::getline(input, line))
                {
                  ++line_number;

                  // Remove comments
                  const std::size_t comment_position = line.find('#');
                  if (comment_position != std::string::npos)
                    line.erase(comment_position);

                  std::istringstream line_stream(line);

                  Point<dim> point;

                  // Empty/comment-only line
                  if (!(line_stream >> point[0]))
                    continue;

                  for (unsigned int d = 1; d < dim; ++d)
                    AssertThrow(line_stream >> point[d],
                                ExcMessage("Could not read sample point file <"
                                           + input_filename
                                           + ">. Line "
                                           + Utilities::int_to_string(line_number)
                                           + " contains fewer than "
                                           + Utilities::int_to_string(dim)
                                           + " coordinates."));

                  std::string extra_entry;
                  AssertThrow(!(line_stream >> extra_entry),
                              ExcMessage("Could not read sample point file <"
                                         + input_filename
                                         + ">. Line "
                                         + Utilities::int_to_string(line_number)
                                         + " contains more than "
                                         + Utilities::int_to_string(dim)
                                         + " entries."));

                  sample_points.push_back(point);
                }

              AssertThrow(!sample_points.empty(),
                ExcMessage("The sample-point file <"
                           + input_filename
                           + "> does not contain any sample points."));
            }
          catch (const std::exception &exc)
            {
              error_message = exc.what();
            }
        }

      error_message = Utilities::MPI::broadcast(this->get_mpi_communicator(),
                                                error_message, 0);

      AssertThrow(error_message.empty(), ExcMessage(error_message));
    }



    template <int dim>
    void CumulativeSlipProfile<dim>::initialize()
    {
      AssertThrow(Plugins::plugin_type_matches<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model()),
                  ExcMessage("Postprocess plugin 'cumulative slip profile' only works when the material model is "
                             "'phase field rsf'."));

      this->get_signals().post_refinement_load_user_data.connect(
        [&](typename parallel::distributed::Triangulation<dim> &)
      {
        this->update_point_locations();
      });

      // Write headers for the output file
      filename = this->get_output_directory() + "cumulative_slip_profile.txt";

      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) != 0)
        return;

      std::ofstream output(filename);
      AssertThrow(output, ExcMessage("Could not open file <" + filename + "> for writing."));

      output << "# time[s] timestep";
      for (unsigned int p = 0; p < sample_points.size(); ++p)
        output << " slip_" << p << "[m]";
      output << '\n';
    }



    template <int dim>
    void 
    CumulativeSlipProfile<dim>::update_point_locations()
    {
      // Skip the pre-refinement steps
      if (this->get_timestep_number() == 0 &&
          this->get_parameters().initial_adaptive_refinement > 0 &&
          this->get_pre_refinement_step() < this->get_parameters().initial_adaptive_refinement - 1)
        return;

      // Create the description of locally owned mesh portions
      const auto local_bboxes =
        GridTools::compute_mesh_predicate_bounding_box(
          this->get_triangulation(),
          [] (const auto &cell)
          {
            return cell->is_locally_owned();
          });

      const auto global_bboxes =
        GridTools::exchange_local_bounding_boxes(
          local_bboxes,
          this->get_mpi_communicator());

      // Compute the point locations
      std::vector<std::vector<Point<dim>>> reference_points;
      std::vector<std::vector<unsigned int>> point_owners;

      std::tie(host_cells, 
               reference_points, 
               locally_owned_sample_point_indices,
               locally_owned_sample_points,
               point_owners)
        = 
        GridTools::distributed_compute_point_locations(
          this->get_phase_field_handler().get_grid_cache(),
          sample_points,
          global_bboxes,
          /*tolerance=*/1.e-8);
    }



    template <int dim>
    std::pair<std::string, std::string>
    CumulativeSlipProfile<dim>::execute(TableHandler &)
    {
      // Prepare to interpolate the slip rate from particles onto sample points
      const MaterialModel::PhaseFieldRSF<dim> &material_model = 
        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model());
      const unsigned int V_property_index = material_model.get_index_cache().particle_properties.slip_rate;

      const Particles::ParticleHandler<dim> &particle_handler = 
        this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();

      std::vector<bool> selected_properties(particle_handler.n_properties_per_particle(), false);
      selected_properties[V_property_index] = true;

      const GridTools::Cache<dim> &grid_cache = this->get_phase_field_handler().get_grid_cache();

      const MPI_Comm comm = this->get_mpi_communicator();
      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

      const unsigned int n_sample_points = Utilities::MPI::broadcast(comm, sample_points.size(), 0);

      std::vector<double>       local_values(n_sample_points, 0.);
      std::vector<unsigned int> local_counts(n_sample_points, 0);

      for (unsigned int c = 0; c < host_cells.size(); ++c)
        {
          const auto &cell = host_cells[c];

          const std::vector<Point<dim>> &sample_points_in_cell  = locally_owned_sample_points[c];
          const std::vector<unsigned int> &sample_point_indices = locally_owned_sample_point_indices[c];

          const auto proportions_and_values =
            PhaseFieldUtilities::interpolate_from_particles_in_crack_zone(
              particle_handler,
              sample_points_in_cell,
              ComponentMask(selected_properties),
              cell, grid_cache,
              [&] (const ArrayView<const double> &properties) -> bool
              {
                return !numbers::is_nan(properties[V_property_index]);
              });

          for (unsigned int p = 0; p < sample_points_in_cell.size(); ++p)
            {
              const unsigned int global_index = sample_point_indices[p];

              // If none of the particles around the sample point are inside the crack zone,
              // then leave the cumulative slip as 0
              if (proportions_and_values[p].second.size() > 0)
                local_values[global_index] = proportions_and_values[p].second[V_property_index];

              local_counts[global_index] += 1;
            }
        }

      // Gather the data to process 0
      std::vector<double> global_values(n_sample_points, 0.);
      std::vector<unsigned int> counts(n_sample_points, 0);

      MPI_Reduce(local_values.data(),
                 global_values.data(),
                 n_sample_points,
                 MPI_DOUBLE,
                 MPI_SUM,
                 0,
                 comm);

      MPI_Reduce(local_counts.data(),
                 counts.data(),
                 n_sample_points,
                 MPI_UNSIGNED,
                 MPI_SUM,
                 0,
                 comm);

      // Append the data to the output file
      if (my_rank == 0)
        {
          if (this->get_timestep_number() == 0)
            {
              cumulative_slip.resize(n_sample_points, 0.);
            }
          else
            {
              const double dt = (this->get_timestep_number() > 0 ? this->get_timestep() : 0);

              for (unsigned int i = 0; i < n_sample_points; ++i)
                {
                  AssertThrow(counts[i] == 1,
                              ExcMessage("Expected exactly one value for cumulative-slip sample point "
                                         + Utilities::int_to_string(i)
                                         + ", but received "
                                         + Utilities::int_to_string(counts[i])
                                         + "."));

                  cumulative_slip[i] += global_values[i] * dt;
                }
            }

          std::ofstream output(filename, std::ios::out | std::ios::app);

          AssertThrow(output, 
                      ExcMessage("Could not open file <" + filename + "> for writing."));

          output << std::scientific
                 << std::setprecision(std::numeric_limits<double>::max_digits10);

          output << this->get_time() << ' ' << this->get_timestep_number();

          for (unsigned int i = 0; i < n_sample_points; ++i)
            output << ' ' << cumulative_slip[i];

          output << '\n';
        }

      return std::make_pair(std::string("Writing cumulative slip profile:"), filename);
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CumulativeSlipProfile,
                                  "cumulative slip profile",
                                  "")
  }
}
