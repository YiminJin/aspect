/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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

#include <aspect/simulator.h>
#include <aspect/geometry_model/box.h>

#include <voro++.hh>

namespace aspect
{
  namespace internal
  {
    namespace CPDI
    {
      template <int dim>
      struct VoronoiCellInformation
      {
        VoronoiCellInformation() = default;

        void clear()
        {
          vertices.clear();
          faces.clear();
          neighbors.clear();
        }

        std::vector<Point<dim>> vertices;

        std::vector<std::vector<unsigned int>> faces;

        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> neighbors;
      };



      template <int dim>
      std::array<int, 3>
      compute_optimal_block_numbers(const Point<dim> &corner1,
                                    const Point<dim> &corner2,
                                    const unsigned int n_particles)
      {
        static constexpr double optimal_particles = (dim > 2 ? 5.6 : 3.4);

        const double dx = corner2[0] - corner1[0];
        const double dy = corner2[1] - corner1[1];
        const double dz = (dim > 2 ? corner2[2] - corner1[2] : 1.0);
        const double ilscale = std::pow(n_particles / (optimal_particles * dx * dy * dz), 1.0 / 3.0);

        std::array<int, 3> n_blocks;
        n_blocks[0] = int(dx * ilscale + 1);
        n_blocks[1] = int(dy * ilscale + 1);
        n_blocks[2] = (dim > 2 ? int(dz * ilscale + 1) : 1);

        return n_blocks;
      }


      template <int dim>
      bool
      face_is_at_boundary(const Point<3>         &p,
                          const Tensor<1, 3>     &n,
                          const BoundingBox<dim> &box)
      {
        const double n_norm = n.norm();
        Assert(n_norm > 0, ExcInternalError());

        const Tensor<1, 3> n_unit = n / n_norm;
        const double tol1 = 1.e-3;
        const double tol2 = std::sqrt(n_norm) * 1.e-3;

        if (1.0 - std::abs(n_unit[0]) < tol1)
          {
            if (n_unit[0] < 0)
              return (std::abs(p[0] - box.get_boundary_points().first[0]) < tol2);
            else
              return (std::abs(p[0] - box.get_boundary_points().second[0]) < tol2);
          }
        else if (1.0 - std::abs(n_unit[1]) < tol1)
          {
            if (n_unit[1] < 0)
              return (std::abs(p[1] - box.get_boundary_points().first[1]) < tol2);
            else
              return (std::abs(p[1] - box.get_boundary_points().second[1]) < tol2);
          }
        else if (1.0 - std::abs(n_unit[2]) < tol1)
          {
            AssertThrow(dim == 3, ExcInternalError());
            if (n_unit[2] < 0)
              return (std::abs(p[2] - box.get_boundary_points().first[2]) < tol2);
            else
              return (std::abs(p[2] - box.get_boundary_points().second[2]) < tol2);
          }
        else
          {
            AssertThrow(false, ExcInternalError());
          }

        return false;
      }



      template <int dim>
      void
      collect_voronoi_cell_information(const std::vector<double>   &vertices,
                                       const std::vector<int>      &face_info,
                                       const std::vector<int>      &neighbors,
                                       const BoundingBox<dim>      &box,
                                       VoronoiCellInformation<dim> &vorocell_info);



      template <>
      void
      collect_voronoi_cell_information(const std::vector<double> &vertices,
                                       const std::vector<int>    &face_info,
                                       const std::vector<int>    &neighbors,
                                       const BoundingBox<2>      &box,
                                       VoronoiCellInformation<2> &vorocell_info)
      {
        vorocell_info.clear();

        unsigned int target_face = numbers::invalid_unsigned_int;
        for (unsigned int i = 0, j = 0; i < neighbors.size(); ++i)
          {
            if (neighbors[i] >= 0)
              continue;

            // Check if the face is parallel to the 2D plane: pick the first three
            // vertices and calculate the normal vector
            Point<3> v[3];
            for (unsigned int k = 0; k < 3; ++k)
              {
                unsigned int l = 3 * face_info[j + k + 1];
                for (unsigned int d = 0; d < 3; ++d)
                  v[k][d] = vertices[l + d];
              }

            const Tensor<1, 3> n = cross_product_3d(v[1] - v[0], v[2] - v[1]);

            // If the normal vector is parallel to z-axis, then we have found the
            // target face; otherwise, check if the face is at the boundary of the
            // geometry model
            if (n[2] * n[2] > (n[0] * n[0] + n[1] * n[1]) * 1.e2)
              target_face = j;
            else
              AssertThrow(face_is_at_boundary(v[0], n, box),
                          ExcMessage("One of the particle domains crosses the boundary of "
                                     "the bounding box of its surrounding cells. In this case, "
                                     "it is impossible to get all the vertices of the particle "
                                     "domain. Please increase the lower limit of particles per cell."));
          }

        // Collect the vertices of the target face
        vorocell_info.vertices.clear();
        for (int k = 0; k < face_info[target_face]; ++k)
          {
            unsigned int l = 3 * face_info[target_face + k + 1];
            AssertIndexRange(l + 2, vertices.size());

            vorocell_info.vertices.push_back(Point<2>(vertices[l], vertices[l + 1]));
          }

        // TODO: collect the neighbors
      }



      template <>
      void
      collect_voronoi_cell_information(const std::vector<double> &vertices,
                                       const std::vector<int>    &face_info,
                                       const std::vector<int>    &neighbors,
                                       const BoundingBox<3>      &box,
                                       VoronoiCellInformation<3> &vorocell_info)
      {
        vorocell_info.clear();

        // Collect the vertices
        for (unsigned int l = 0; l < vertices.size(); l += 3)
          vorocell_info.vertices.push_back(Point<3>(vertices[l], vertices[l + 1], vertices[l + 2]));

        // Collect the faces
        for (unsigned int i = 0, j = 0; i < neighbors.size(); ++i)
          {
            const unsigned int n_vertices = face_info[j];
            std::vector<unsigned int> face(n_vertices);
            for (unsigned int k = 0; k < n_vertices; ++k)
              face[k] = face_info[j + k + 1];

            // If the face is at the boundary of the container, check if it is at
            // the boundary of the geometry model
            if (neighbors[i] < 0)
              {
                Point<3> v[3];
                for (unsigned int k = 0; k < 3; ++k)
                  v[k] = vorocell_info.vertices[face[k]];

                const Tensor<1, 3> n = cross_product_3d(v[1] - v[0], v[2] - v[1]);

                AssertThrow(face_is_at_boundary(v[0], n, box),
                            ExcMessage("One of the particle domains crosses the boundary of "
                                       "the bounding box of its surrounding cells. In this case, "
                                       "it is impossible to get all the vertices of the particle "
                                       "domain. Please increase the lower limit of particles per cell."));
              }

            vorocell_info.faces.emplace_back(face);
            j += face_info[j] + 1;

            // TODO: collect the neighbors
          }
      }



      template <int dim>
      void
      compute_generalized_basis_function_values(const typename Triangulation<dim>::active_cell_iterator           &cell,
                                                const std::set<typename Triangulation<dim>::active_cell_iterator> &neighbors,
                                                const Particles::ParticleHandler<dim>                             &particle_handler,
                                                const FiniteElement<dim>                                          &fe,
                                                const Mapping<dim>                                                &mapping,
                                                const BoundingBox<dim>                                            &box,
                                                std::vector<std::vector<double>>                                  &phi)
      {
        // Create the voro container, which is the bounding box of the current cell and
        // its neighbors
        Point<dim> corner1 = cell->vertex(0);
        Point<dim> corner2 = cell->vertex((1<<dim) - 1);
        unsigned int n_particles = 0;
        for (const auto &neighbor : neighbors)
          {
            for (const auto v : neighbor->vertex_indices())
              {
                const Point<dim> &vertex = neighbor->vertex(v);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    corner1[d] = std::min(corner1[d], vertex[d]);
                    corner2[d] = std::max(corner2[d], vertex[d]);
                  }
              }
            n_particles += particle_handler.n_particles_in_cell(neighbor);
          }

        const double diameter = corner2.distance(corner1);

        const std::array<int, 3> n_blocks = compute_optimal_block_numbers(corner1,
                                                                          corner2,
                                                                          n_particles);

        voro::container container(/*ax=*/corner1[0],
                                         /*bx=*/corner2[0],
                                         /*ay=*/corner1[1],
                                         /*by=*/corner2[1],
                                         /*az=*/(dim > 2 ? corner1[2] : -diameter * 0.01),
                                         /*bz=*/(dim > 2 ? corner2[2] :  diameter * 0.01),
                                         /*nx=*/n_blocks[0],
                                         /*ny=*/n_blocks[1],
                                         /*nz=*/n_blocks[2],
                                         /*xperiodic=*/false,
                                         /*yperiodic=*/false,
                                         /*zperiodic=*/false,
                                         /*init_mem=*/8);

        // Put all the particles in the current cell and its neighbors into the container
        for (const auto &neighbor : neighbors)
          for (const auto &particle : particle_handler.particles_in_cell(neighbor))
            {
              const Point<dim> &location = particle.get_location();
              container.put(particle.get_local_index(), location[0], location[1], (dim > 2 ? location[2] : 0.0));
            }

        // Collect the local indices of the particles in the current cell
        std::set<int> local_indices;
        for (const auto &particle : particle_handler.particles_in_cell(cell))
          local_indices.insert(static_cast<int>(particle.get_local_index()));

        // Loop over the particles in the current cell and compute each Voronoi cell
        voro::voronoicell_neighbor voro_cell;
        std::vector<double> voro_vertices;
        std::vector<int> voro_face_vertices;
        std::vector<int> voro_neighbors;
        VoronoiCellInformation<dim> vorocell_info;

        voro::c_loop_all loop(container);
        AssertThrow(loop.start(), ExcMessage("An error occurs when calling voro::c_loop_all::start()."));

        do
          {
            const int vorocell_id = loop.pid();
            if (local_indices.find(vorocell_id) == local_indices.end())
              continue;

            AssertThrow(container.compute_cell(voro_cell, loop),
                        ExcMessage("An error occurs when calling voro::container::compute_cell()."));

            double x, y, z;
            loop.pos(x, y, z);
            voro_cell.vertices(x, y, z, voro_vertices);
            voro_cell.face_vertices(voro_face_vertices);
            voro_cell.neighbors(voro_neighbors);
            collect_voronoi_cell_information(voro_vertices,
                                             voro_face_vertices,
                                             voro_neighbors,
                                             box,
                                             vorocell_info);
          }
        while (loop.inc());

        // Compute phi_Ip for each dof I and particle p
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_vertices = vorocell_info.vertices.size();
        AssertDimension(phi.size(), dofs_per_cell);

        std::vector<std::vector<double>> shape_values(dofs_per_cell);

        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            // Compute the values of the shape function at the vertices
            // of the particle domain
            shape_values[i].resize(n_vertices);
            for (unsigned int v = 0; v < n_vertices; ++v)
              {
                const Point<dim> p_unit = mapping.transform_real_to_unit_cell(cell, vorocell_info.vertices[v]);
                shape_values[i][v] = fe.shape_value(i, p_unit);
              }
          }
      }
    }
  }



  template <int dim>
  void
  Simulator<dim>::
  perform_convected_particle_domain_interpolation(const std::vector<AdvectionField> &advection_fields)
  {
    computing_timer.enter_subsection("Particles: CPDI");

    // The container of voro++ is a box, so we can only use the box model without initial topography
    // and mesh deformation.
    AssertThrow(Plugins::plugin_type_matches<GeometryModel::Box<dim>>(*geometry_model),
                ExcMessage("Compositional field method ``cpdi'' only works when the geometry model is ``box''."));
    AssertThrow(Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(*initial_topography_model),
                ExcMessage("Compositional field method ``cpdi'' only works when the initial topography model is ``zero topography''."));
    AssertThrow(mesh_deformation->get_active_mesh_deformation_models().size() == 0,
                ExcMessage("Compositional field method ``cpdi'' only works when mesh deformation is inactive."));

    // If the fields to be interpolated with CPDI method are handled by different particle managers,
    // then we will need more than one matrix block to perform the projection. Currently we do not
    // allow such settings.
    Particle::Manager<dim> *cpdi_particle_manager = nullptr;

    if (parameters.mapped_particle_properties.size() != 0)
      {
        cpdi_particle_manager = &particle_managers[0];
      }
    else
      {
        // Find which particle manager handles field 0
        for (auto &manager : particle_managers)
          {
            const std::string &field0_name = parameters.mapped_particle_properties[advection_fields[0].compositional_variable].first;
            if (manager.get_property_manager().get_data_info().fieldname_exists(field0_name))
              {
                cpdi_particle_manager = &manager;
                break;
              }
          }
      }

    AssertThrow(cpdi_particle_manager != nullptr, ExcInternalError());

    // Check if all the input fields are handled by the same particle manager
    if (parameters.mapped_particle_properties.size() != 0)
      {
        for (const auto &field : advection_fields)
          {
            const std::string &field_name = parameters.mapped_particle_properties[field.compositional_variable].first;
            AssertThrow(cpdi_particle_manager->get_property_manager().get_data_info().fieldname_exists(field_name),
                        ExcMessage("All the compositional fields advected by the CPDI method must be handled by the same particle manager."));
          }
      }

    // We have checked in source/simulator/parameters.cc that all the CPDI fields are
    // discretized by Q1 element. So all the input fields share the same sparsity block.
    const unsigned int sparsity_block_idx = advection_fields[0].sparsity_pattern_block_index(introspection);
    system_matrix.block(sparsity_block_idx, sparsity_block_idx) = 0;
    for (const auto &field : advection_fields)
      {
        const unsigned int block_idx = field.block_index(introspection);
        system_rhs.block(block_idx) = 0;
      }

    // Create the vertex-to-cell map
    GridTools::Cache<dim> grid_cache(triangulation, *mapping);
    const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

    // Get the bounding box of the computational domain
    const GeometryModel::Box<dim> &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(*geometry_model);
    const Point<dim> corner1 = box.get_origin();
    const Point<dim> corner2 = corner1 + box.get_extents();
    const BoundingBox<dim> bounding_box(std::make_pair(corner1, corner2));

    const Particles::ParticleHandler<dim> &particle_handler = cpdi_particle_manager->get_particle_handler();
    const FiniteElement<dim> &fe = finite_element.base_element(advection_fields[0].base_element(introspection));

    std::vector<std::vector<double>> phi;

    // Now loop over the locally owned active cells and assemble the CPDI systems
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // Find the cells neighboring the current cell
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cells;
          for (const auto v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cells.insert(vertex_to_cell_map[vertex_index].begin(),
                                       vertex_to_cell_map[vertex_index].end());
            }

          internal::CPDI::compute_generalized_basis_function_values(cell,
                                                                    neighboring_cells,
                                                                    particle_handler,
                                                                    fe,
                                                                    *mapping,
                                                                    bounding_box,
                                                                    phi);
        }
  }
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  template void Simulator<dim>::perform_convected_particle_domain_interpolation(const std::vector<AdvectionField> &);

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
