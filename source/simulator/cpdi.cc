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
            return (std::abs(p[0] - box.get_boundary_points().first[0]) < tol2 ||
                    std::abs(p[0] - box.get_boundary_points().second[0]) < tol2);
          }
        else if (1.0 - std::abs(n_unit[1]) < tol1)
          {
            return (std::abs(p[1] - box.get_boundary_points().first[1]) < tol2 ||
                    std::abs(p[1] - box.get_boundary_points().second[1]) < tol2);
          }
        else if (1.0 - std::abs(n_unit[2]) < tol1)
          {
            AssertThrow(dim == 3, ExcInternalError());
            return (std::abs(p[2] - box.get_boundary_points().first[2]) < tol2 ||
                    std::abs(p[2] - box.get_boundary_points().second[2]) < tol2);
          }
        else
          {
            AssertThrow(false, ExcInternalError());
          }

        return false;
      }



      template <int dim>
      struct VoronoiCell
      {
        VoronoiCell(const std::vector<double> &voro_vertices,
                    const std::vector<int>    &voro_faces,
                    const std::vector<int>    &voro_neighbors,
                    const BoundingBox<dim>    &box);

        std::vector<Point<dim>> vertices;

        std::vector<std::vector<unsigned int>> faces;

        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> neighbors;
      };



      template <>
      VoronoiCell<2>::
      VoronoiCell(const std::vector<double> &voro_vertices,
                  const std::vector<int>    &voro_faces,
                  const std::vector<int>    &voro_neighbors,
                  const BoundingBox<2>      &box)
      {
        unsigned int target_face = numbers::invalid_unsigned_int;
        for (unsigned int i = 0, j = 0; i < voro_neighbors.size(); ++i)
          {
            if (voro_neighbors[i] >= 0)
              {
                j += voro_faces[j] + 1;
                continue;
              }

            // Check if the face is parallel to the 2D plane: pick the first three
            // vertices and calculate the normal vector
            Point<3> v[3];
            for (unsigned int k = 0; k < 3; ++k)
              {
                unsigned int l = 3 * voro_faces[j + k + 1];
                for (unsigned int d = 0; d < 3; ++d)
                  v[k][d] = voro_vertices[l + d];
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

            j += voro_faces[j] + 1;
          }

        // Collect the vertices of the target face
        vertices.clear();
        for (int k = 0; k < voro_faces[target_face]; ++k)
          {
            unsigned int l = 3 * voro_faces[target_face + k + 1];
            AssertIndexRange(l + 2, voro_vertices.size());

            vertices.emplace_back(Point<2>(voro_vertices[l], voro_vertices[l + 1]));
          }

        // TODO: collect the neighbors
      }



      template <>
      VoronoiCell<3>::
      VoronoiCell(const std::vector<double> &voro_vertices,
                  const std::vector<int>    &voro_faces,
                  const std::vector<int>    &voro_neighbors,
                  const BoundingBox<3>      &box)
      {
        // Collect the vertices
        for (unsigned int l = 0; l < voro_vertices.size(); l += 3)
          vertices.push_back(Point<3>(voro_vertices[l],
                                      voro_vertices[l + 1],
                                      voro_vertices[l + 2]));

        // Collect the faces
        for (unsigned int i = 0, j = 0; i < voro_neighbors.size(); ++i)
          {
            const unsigned int n_vertices = voro_faces[j];
            std::vector<unsigned int> face(n_vertices);
            for (unsigned int k = 0; k < n_vertices; ++k)
              face[k] = voro_faces[j + k + 1];

            // If the face is at the boundary of the container, check if it is at
            // the boundary of the geometry model
            if (voro_neighbors[i] < 0)
              {
                Point<3> v[3];
                for (unsigned int k = 0; k < 3; ++k)
                  v[k] = vertices[face[k]];

                const Tensor<1, 3> n = cross_product_3d(v[1] - v[0], v[2] - v[1]);

                AssertThrow(face_is_at_boundary(v[0], n, box),
                            ExcMessage("One of the particle domains crosses the boundary of "
                                       "the bounding box of its surrounding cells. In this case, "
                                       "it is impossible to get all the vertices of the particle "
                                       "domain. Please increase the lower limit of particles per cell."));
              }

            faces.emplace_back(face);
            j += voro_faces[j] + 1;
          }

        // TODO: collect the neighbors
      }



      template <int dim>
      class GeneralizedBasisData
      {
        public:
          GeneralizedBasisData(const unsigned int dofs_per_cell,
                               const bool evaluate_gradients);

          void resize(const unsigned int n_particles);

          bool evaluate_gradients() const;

          double get_volume(const unsigned int p) const;

          double get_value(const unsigned int i,
                           const unsigned int p) const;

          const Tensor<1, dim> &
          get_gradient(const unsigned int i,
                       const unsigned int p) const;

          void set_volume(const unsigned int p,
                          const double       vol);

          void set_value(const unsigned int i,
                         const unsigned int p,
                         const double       val);

          void set_gradient(const unsigned int    i,
                            const unsigned int    p,
                            const Tensor<1, dim> &grad);

        private:
          std::vector<std::vector<double>>         values;

          std::vector<std::vector<Tensor<1, dim>>> gradients;

          std::vector<double>                      volumes;
      };



      template <int dim>
      GeneralizedBasisData<dim>::
      GeneralizedBasisData(const unsigned int dofs_per_cell,
                           const bool evaluate_gradients)
        : values(dofs_per_cell)
        , gradients(evaluate_gradients ? dofs_per_cell : 0)
      {}



      template <int dim>
      void
      GeneralizedBasisData<dim>::resize(const unsigned int n_particles)
      {
        volumes.resize(n_particles);
        std::fill(volumes.begin(), volumes.end(), 0.0);

        for (unsigned int i = 0; i < values.size(); ++i)
          {
            values[i].resize(n_particles);
            std::fill(values[i].begin(), values[i].end(), 0.0);

            if (gradients.size() > 0)
              {
                gradients[i].resize(n_particles);
                std::fill(gradients[i].begin(), gradients[i].end(), Tensor<1, dim>());
              }
          }
      }



      template <int dim>
      bool
      GeneralizedBasisData<dim>::evaluate_gradients() const
      {
        return (gradients.size() > 0);
      }



      template <int dim>
      double
      GeneralizedBasisData<dim>::get_volume(const unsigned int p) const
      {
        AssertIndexRange(p, volumes.size());
        return volumes[p];
      }



      template <int dim>
      double
      GeneralizedBasisData<dim>::get_value(const unsigned int i,
                                           const unsigned int p) const
      {
        AssertIndexRange(i, values.size());
        AssertIndexRange(p, values[i].size());
        return values[i][p];
      }



      template <int dim>
      const Tensor<1, dim> &
      GeneralizedBasisData<dim>::get_gradient(const unsigned int i,
                                              const unsigned int p) const
      {
        AssertThrow(gradients.size() > 0,
                    ExcMessage("Cannot get basis gradient because gradients are not requested "
                               "when initializing the GeneralizedBasisData object."));

        AssertIndexRange(i, gradients.size());
        AssertIndexRange(p, gradients[i].size());
        return gradients[i][p];
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_volume(const unsigned int p,
                                            const double       vol)
      {
        AssertIndexRange(p, volumes.size());
        volumes[p] = vol;
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_value(const unsigned int i,
                                           const unsigned int p,
                                           const double       val)
      {
        AssertIndexRange(i, values.size());
        AssertIndexRange(p, values[i].size());
        values[i][p] = val;
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_gradient(const unsigned int    i,
                                              const unsigned int    p,
                                              const Tensor<1, dim> &grad)
      {
        AssertThrow(gradients.size() > 0,
                    ExcMessage("Cannot set basis gradient because gradients are not requested "
                               "when initializing the GeneralizedBasisData object."));

        AssertIndexRange(i, gradients.size());
        AssertIndexRange(p, gradients[i].size());
        gradients[i][p] = grad;
      }



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
      double
      simplex_measure(const std::array<Point<dim>, dim + 1> &vertices)
      {
        static constexpr double dim_factorial = (dim == 2 ? 2.0 : 6.0);

        Tensor<2, dim> T;
        for (unsigned int d = 0; d < dim; ++d)
          T[d] = vertices[d] - vertices[dim];

        return std::abs(determinant(T)) / dim_factorial;
      }



      Tensor<1, 2>
      compute_generalized_basis_gradient(const std::array<Point<2>, 3> &v,
                                         const std::array<double,   3> &N)
      {
        Tensor<1, 2> t;
        t[0] = (N[0] * (v[1][1] - v[2][1]) + N[1] * (v[2][1] - v[0][1]) + N[2] * (v[0][1] - v[1][1])) * 0.5;
        t[1] = (N[0] * (v[2][0] - v[1][0]) + N[1] * (v[0][0] - v[2][0]) + N[2] * (v[1][0] - v[0][0])) * 0.5;

        return t;
      }



      Tensor<1, 3>
      compute_generalized_basis_gradient(const std::array<Point<3>, 4> &v,
                                         const std::array<double,   4> &N)
      {
        Tensor<1, 3> t;

      }



      template <int dim>
      void
      do_evaluation(const typename Triangulation<dim>::active_cell_iterator &cell,
                    const std::vector<VoronoiCell<dim>>                     &voronoi_cells,
                    const FiniteElement<dim>                                &fe,
                    const Mapping<dim>                                      &mapping,
                    GeneralizedBasisData<dim>                               &data);



      template <>
      void
      do_evaluation<2>(const Triangulation<2>::active_cell_iterator &cell,
                       const std::vector<VoronoiCell<2>>            &voronoi_cells,
                       const FiniteElement<2>                       &fe,
                       const Mapping<2>                             &mapping,
                       GeneralizedBasisData<2>                      &data)
      {
        const unsigned int n_particles = voronoi_cells.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        data.resize(n_particles);
        std::vector<std::vector<double>> N_v(dofs_per_cell);
        for (unsigned int p = 0; p < n_particles; ++p)
          {
            double area  = 0.0;
            std::vector<double> values(dofs_per_cell);
            std::vector<Tensor<1, 2>> gradients(dofs_per_cell);

            const VoronoiCell<2> &voronoi_cell = voronoi_cells[p];
            const unsigned int n_vertices = voronoi_cell.vertices.size();

            // Evaluate the values of the shape functions at the vertices
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                N_v[i].resize(n_vertices);
                std::fill(N_v[i].begin(), N_v[i].end(), 0.0);
              }

            for (unsigned int v = 0; v < n_vertices; ++v)
              {
                const Point<2> vertex_unit = mapping.transform_real_to_unit_cell(cell, voronoi_cell.vertices[v]);
                if (GeometryInfo<2>::is_inside_unit_cell(vertex_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[i][v] = fe.shape_value(i, vertex_unit);
              }

            if (n_vertices > 3)
              {
                // Divide the voronoi cell into n_vertices triangles, each triangle
                // composed of two adjacent vertices and the barycenter of the Voronoi
                // cell
                Point<2> center;
                for (unsigned int v = 0; v < n_vertices; ++v)
                  center += voronoi_cell.vertices[v];
                center /= n_vertices;

                std::vector<double> N_c(dofs_per_cell);
                const Point<2> center_unit = mapping.transform_real_to_unit_cell(cell, center);
                if (GeometryInfo<2>::is_inside_unit_cell(center_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_c[i] = fe.shape_value(i, center_unit);

                for (unsigned int v = 0; v < n_vertices; ++v)
                  {
                    const unsigned int v1 = v;
                    const unsigned int v2 = (v + 1) % n_vertices;

                    // Calculate the area of the triangle
                    std::array<Point<2>, 3> verts;
                    verts[0] = voronoi_cell.vertices[v1];
                    verts[1] = voronoi_cell.vertices[v2];
                    verts[2] = center;
                    const double triangle_area = simplex_measure(verts);
                    area += triangle_area;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        values[i] += (N_v[i][v1] + N_v[i][v2] + N_c[i]) * (triangle_area / 3.0);
                        if (data.evaluate_gradients())
                        {
                          std::array<double, 3> vals;
                          vals[0] = N_v[i][v1];
                          vals[1] = N_v[i][v2];
                          vals[2] = N_c[i];

                          gradients[i] += compute_generalized_basis_gradient(verts, vals);
                        }
                      }
                  }

                data.set_volume(p, area);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    data.set_value(i, p, values[i] / area);
                    if (data.evaluate_gradients())
                      data.set_gradient(i, p, gradients[i] / area);
                  }
              }
            else
              {
                std::array<Point<2>, 3> verts;
                verts[0] = voronoi_cell.vertices[0];
                verts[1] = voronoi_cell.vertices[1];
                verts[2] = voronoi_cell.vertices[2];
                data.set_volume(p, simplex_measure(verts));

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    data.set_value(i, p, (N_v[i][0] + N_v[i][1] + N_v[i][2]) / 3.0);
                    if (data.evaluate_gradients())
                    {
                      std::array<double, 3> vals;
                      vals[0] = N_v[i][0];
                      vals[1] = N_v[i][1];
                      vals[2] = N_v[i][2];

                      data.set_gradient(i, p, compute_generalized_basis_gradient(verts, vals) / data.get_volume(p));
                    }
                  }
              }
          }
      }



      template <>
      void
      do_evaluation(const Triangulation<3>::active_cell_iterator &cell,
                    const std::vector<VoronoiCell<3>>            &voronoi_cells,
                    const FiniteElement<3>                       &fe,
                    const Mapping<3>                             &mapping,
                    GeneralizedBasisData<3>                      &data)
      {
        const unsigned int n_particles = voronoi_cells.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        data.resize(n_particles);
        std::vector<std::vector<double>> N(dofs_per_cell);
        for (unsigned int p = 0; p < n_particles; ++p)
          {
            double volume = 0.0;
            double value  = 0.0;
            Tensor<1, 2> gradient;

            const VoronoiCell<3> &voronoi_cell = voronoi_cells[p];
            const unsigned int n_vertices = voronoi_cell.vertices.size();
            const unsigned int n_faces    = voronoi_cell.faces.size();

            // Calculate the barycenter of the Voronoi cell
            Point<3> volume_barycenter;
            for (unsigned int v = 0; v < n_vertices; ++v)
              volume_barycenter += voronoi_cell.vertices[v];
            volume_barycenter /= n_vertices;

            // Calculate the barycenters of the faces of the Voronoi cell
            std::vector<Point<3>> face_barycenters;
            for (unsigned int f = 0; f < n_faces; ++f)
              {
                Point<3> face_barycenter;
                for (unsigned int v = 0; v < voronoi_cell.faces[f].size(); ++v)
                  face_barycenter += voronoi_cell.vertices[voronoi_cell.faces[f][v]];
                face_barycenter /= voronoi_cell.faces[f].size();
                face_barycenters.emplace_back(face_barycenter);
              }

            // Evaluate the values of the shape functions at the vertices and the barycenters
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                N[i].resize(n_vertices + n_faces + 1);
                std::fill(N[i].begin(), N[i].end(), 0.0);
              }

            for (unsigned int v = 0; v < n_vertices; ++v)
              {
                const Point<3> vertex_unit = mapping.transform_real_to_unit_cell(cell, voronoi_cell.vertices[v]);
                if (GeometryInfo<3>::is_inside_unit_cell(vertex_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N[i][v] = fe.shape_value(i, vertex_unit);
              }

            for (unsigned int f = 0; f < n_faces; ++f)
              {
                const Point<3> fc_unit = mapping.transform_real_to_unit_cell(cell, face_barycenters[f]);
                if (GeometryInfo<3>::is_inside_unit_cell(fc_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N[i][n_vertices + f] = fe.shape_value(i, fc_unit);
              }

            const Point<3> vc_unit = mapping.transform_real_to_unit_cell(cell, volume_barycenter);
            if (GeometryInfo<3>::is_inside_unit_cell(vc_unit))
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                N[i][n_vertices + n_faces] = fe.shape_value(i, vc_unit);

          }
      }



      template <int dim>
      void
      evaluate(const typename Triangulation<dim>::active_cell_iterator           &cell,
               const std::set<typename Triangulation<dim>::active_cell_iterator> &neighbors,
               const Particles::ParticleHandler<dim>                             &particle_handler,
               const FiniteElement<dim>                                          &fe,
               const Mapping<dim>                                                &mapping,
               const BoundingBox<dim>                                            &box,
               GeneralizedBasisData<dim>                                         &data)
      {
        // Create the voro container, which is the bounding box of the current cell and
        // its neighbors
        Point<dim> corner1 = cell->vertex(0);
        Point<dim> corner2 = cell->vertex(GeometryInfo<dim>::vertices_per_cell - 1);
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

        std::vector<VoronoiCell<dim>> voronoi_cells;

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

            voronoi_cells.emplace_back(VoronoiCell<dim>(voro_vertices,
                                                        voro_face_vertices,
                                                        voro_neighbors,
                                                        box));
          }
        while (loop.inc());

        // Compute the volume and the values (and gradients) of the generalized basis functions
        // for each particle domain
        do_evaluation(cell, voronoi_cells, fe, mapping, data);
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
    AssertThrow(mesh_deformation == nullptr,
                ExcMessage("Compositional field method ``cpdi'' only works when mesh deformation is not enabled."));

    for (const auto &field : advection_fields)
      {
        AssertThrow(field.polynomial_degree(introspection) == 1 && field.is_discontinuous(introspection) == false,
                    ExcMessage("The CPDI method can only be applied to compositional fields discretized by standard Q1 elements."));
      }

    // Find the particle manager handling the input fields
    const Particle::Manager<dim> *cpdi_particle_manager = nullptr;
    if (parameters.mapped_particle_properties.size() == 0)
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
            AssertThrow(cpdi_particle_manager->get_property_manager().get_data_info().fieldname_exists(field_name), ExcInternalError());
          }
      }

    // We have checked that all the input fields are discretized by Q1 element,
    // so all the input fields share the same sparsity block.
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

    internal::CPDI::GeneralizedBasisData<dim> data(fe.dofs_per_cell, false);

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

          internal::CPDI::evaluate(cell,
                                   neighboring_cells,
                                   particle_handler,
                                   fe,
                                   *mapping,
                                   bounding_box,
                                   data);
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
