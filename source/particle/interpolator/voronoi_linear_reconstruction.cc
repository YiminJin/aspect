/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

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

#include <aspect/particle/interpolator/voronoi_linear_reconstruction.h>
#include <aspect/particle/particle_domain.h>

#include <deal.II/grid/grid_tools.h>

namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template <int dim>
      VoronoiLinearReconstruction<dim>::VoronoiLinearReconstruction()
        : grid_cache(nullptr)
        , particle_domain_handler(nullptr)
      {}



      template <int dim>
      void
      VoronoiLinearReconstruction<dim>::initialize()
      {
        grid_cache = std::make_unique<GridTools::Cache<dim>>(this->get_triangulation(),
                                                             this->get_mapping());
      }



      namespace
      {
        template <int dim>
        void
        evaluate_gradients_and_limiters(const typename ParticleHandler<dim>::particle_iterator &particle,
                                        const ParticleDomainAccessor<dim> &particle_domain,
                                        const ComponentMask               &selected_properties,
                                        std::vector<Tensor<1, dim>>       &gradients,
                                        std::vector<double>               &limiters)
        {
          // Get the face data of the particle domain
          const unsigned int n_faces = particle_domain.n_faces();
          std::vector<double> face_measures(n_faces);
          std::vector<typename ParticleHandler<dim>::particle_iterator> neighbor_particles(n_faces);
          for (unsigned int f = 0; f < n_faces; ++f)
            {
              face_measures[f] = particle_domain.face_measure(f);
              neighbor_particles[f] = particle_domain.neighbor_particle(f);
            }

          // Initialize the output vectors
          const unsigned int n_properties = selected_properties.size();
          gradients.resize(n_properties, Tensor<1, dim>());
          limiters.resize(n_properties, 1.0);

          // Record the local minimum and maximum
          std::vector<double> local_minimum(n_properties, numbers::signaling_nan<double>());
          std::vector<double> local_maximum(n_properties, numbers::signaling_nan<double>());

          const Point<dim> &particle_location = particle->get_location();
          const ArrayView<const double> particle_properties = particle->get_properties();
          for (unsigned int i = 0; i < selected_properties.size(); ++i)
            if (selected_properties[i])
              {
                local_minimum[i] = particle_properties[i];
                local_maximum[i] = particle_properties[i];
              }

          // Loop over the neighbor particles and compute the gradient using
          // the least-squares method
          Tensor<2, dim> lsq_matrix;
          std::vector<Tensor<1, dim>> lsq_vectors(n_properties);
          for (unsigned int f = 0; f < n_faces; ++f)
            if (neighbor_particles[f]->state() == IteratorState::valid)
              {
                const Tensor<1, dim> d = neighbor_particles[f]->get_location() - particle_location;
                const double weight = face_measures[f] / d.norm_square();
                lsq_matrix += weight * outer_product(d, d);

                const ArrayView<const double> neighbor_properties = neighbor_particles[f]->get_properties();
                for (unsigned int i = 0; i < n_properties; ++i)
                  if (selected_properties[i])
                    {
                      lsq_vectors[i] += weight * (neighbor_properties[i] - particle_properties[i]) * d;

                      local_minimum[i] = std::min(local_minimum[i], neighbor_properties[i]);
                      local_maximum[i] = std::max(local_maximum[i], neighbor_properties[i]);
                    }
            }

          const Tensor<2, dim> lsq_matrix_inv = invert(lsq_matrix);
          for (unsigned int i = 0; i < selected_properties.size(); ++i)
            if (selected_properties[i])
              gradients[i] = lsq_matrix_inv * lsq_vectors[i];

          // Compute the limiters
          for (unsigned int f = 0; f < n_faces; ++f)
            if (neighbor_particles[f]->state() == IteratorState::valid)
              {
                // Take the face centroid as control point
                const Tensor<1, dim> d = (neighbor_particles[f]->get_location() - particle_location) * 0.5;
                for (unsigned int i = 0; i < n_properties; ++i)
                  if (selected_properties[i])
                    {
                      const double delta = gradients[i] * d;
                      if (std::abs(delta) > std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(particle_properties[i])))
                        limiters[i] = std::min(limiters[i], (delta > 0 ? (local_maximum[i] - particle_properties[i]) / delta
                                                                       : (local_minimum[i] - particle_properties[i]) / delta));
                    }
              }
        }
      }



      template <int dim>
      std::vector<std::vector<double>>
      VoronoiLinearReconstruction<dim>::
      properties_at_points(const ParticleHandler<dim> &particle_handler,
                           const std::vector<Point<dim>> &positions,
                           const ComponentMask &selected_properties,
                           const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const
      {
        // If it is the first time this function is called, try to find the particle domain handler
        // corresponding to the input particle handler
        if (particle_domain_handler == nullptr)
          {
            for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
              {
                const Particle::Manager<dim> &particle_manager = this->get_particle_manager(i);
                if (particle_manager.particle_domains_requested())
                  {
                    const ParticleDomainHandler<dim> &pdh = particle_manager.get_particle_domain_handler();
                    if (&pdh.get_particle_handler() == &particle_handler)
                      {
                        AssertThrow(pdh.face_data_requested(),
                                    ExcMessage("Particle interpolator `voronoi linear reconstruction' requires "
                                               "particle domain face data."));
                        particle_domain_handler = &pdh;
                        break;
                      }
                  }
              }

            AssertThrow(particle_domain_handler != nullptr,
                        ExcMessage("Particle interpolator `voronoi linear reconstruction' requires the "
                                   "corresponding particle manager to generate particle domains."));
          }

        // Collect the one-loop patch around the target cell
        const auto &vertex_to_cell_map = grid_cache->get_vertex_to_cell_map();
        std::set<typename Triangulation<dim>::active_cell_iterator> patch;
        for (const unsigned int v : cell->vertex_indices())
          {
            const unsigned int vertex_index = cell->vertex_index(v);
            patch.insert(vertex_to_cell_map[vertex_index].begin(),
                         vertex_to_cell_map[vertex_index].end());
          }

        // Find the nearest particle for each point
        const unsigned int n_points = positions.size();
        std::vector<double> shortest_distances(n_points, std::numeric_limits<double>::max());
        std::vector<typename ParticleHandler<dim>::particle_iterator> nearest_particles(n_points);
        for (const auto &cell : patch)
          {
            const auto particles_in_cell = particle_handler.particles_in_cell(cell);
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const Point<dim> &particle_location = particle->get_location();
                for (unsigned int i = 0; i < n_points; ++i)
                  {
                    const double distance = particle_location.distance(positions[i]);
                    if (distance < shortest_distances[i])
                      {
                        shortest_distances[i] = distance;
                        nearest_particles[i] = particle;
                      }
                  }
              }
          }

        // Collect the relevant particles without duplication
        std::vector<typename ParticleHandler<dim>::particle_iterator> relevant_particles;
        std::vector<unsigned int> point_to_particle(n_points);
        for (unsigned int i = 0; i < n_points; ++i)
          {
            const auto vit = std::find(relevant_particles.begin(), relevant_particles.end(), nearest_particles[i]);
            if (vit == relevant_particles.end())
              {
                point_to_particle[i] = relevant_particles.size();
                relevant_particles.push_back(nearest_particles[i]);
              }
            else
              point_to_particle[i] = std::distance(relevant_particles.begin(), vit);
          }

        // Evaluate the gradients and limiters for each selected property 
        const unsigned int n_relevant_particles = relevant_particles.size();
        std::vector<std::vector<Tensor<1, dim>>> gradients(n_relevant_particles);
        std::vector<std::vector<double>> limiters(n_relevant_particles);
        for (unsigned int p = 0; p < n_relevant_particles; ++p)
          {
            const ParticleDomainAccessor<dim> particle_domain 
              = particle_domain_handler->get_particle_domain(relevant_particles[p]->get_local_index());

            evaluate_gradients_and_limiters(relevant_particles[p],
                                            particle_domain,
                                            selected_properties,
                                            gradients[p],
                                            limiters[p]);
          }

        const unsigned int n_properties = particle_handler.n_properties_per_particle();
        AssertDimension(selected_properties.size(), n_properties);
        std::vector<std::vector<double>> cell_properties(n_points, std::vector<double>(n_properties, numbers::signaling_nan<double>()));
        for (unsigned i = 0; i < n_points; ++i)
          {
            const auto &particle = relevant_particles[point_to_particle[i]];
            const std::vector<Tensor<1, dim>> &particle_gradients = gradients[point_to_particle[i]];
            const std::vector<double> &particle_limiters = limiters[point_to_particle[i]];
            const Tensor<1, dim> d = positions[i] - particle->get_location();
            const ArrayView<const double> particle_properties = particle->get_properties();
            for (unsigned int j = 0; j < n_properties; ++j)
              if (selected_properties[j])
                cell_properties[i][j] = particle_properties[j] + particle_limiters[j] * (particle_gradients[j] * d);
          }

        return cell_properties;
      }
    }
  }
}

// explicit instantiation
namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(VoronoiLinearReconstruction,
                                            "voronoi linear reconstruction",
                                            "")
    }
  }
}
