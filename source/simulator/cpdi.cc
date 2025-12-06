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

#ifdef ASPECT_WITH_VORO
#include <voro++.hh>
#endif

namespace aspect
{
#ifdef ASPECT_WITH_VORO

  namespace internal
  {
    namespace CPDI
    {
      /**
       * This function determines whether a planar face (defined by a point and
       * a normal vector) is at the boundary of the bounding box of the
       * geometry model.
       *
       * @param[in] p A point at the face.
       * @param[in] n The normal vector (not necessarily unit) of the face. If
       *  @p dim equals 2, then n is supposed to be in the x-y plane, or the
       *  function will throw an error.
       * @param[in] box The bounding box of the geometry model.
       */
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



      /**
       * Structure for holding information about a Voronoi cell.
       */
      template <int dim>
      struct VoronoiCell
      {
        using particle_iterator = typename Particles::ParticleHandler<dim>::particle_iterator;
        /**
         * Constructor.
         *
         * @param[in] particle_index The local particle ID.
         *
         * @param[in] particle_map Map from particle indices to particle
         *  iterators.
         *
         * @param[in] voro_vertices An array of float numbers representing the
         *  coordinates of the vertices of the Voronoi cell. It is supposed to
         *  be the output of function voro::voronoicell_neighbor::vertices().
         *
         * @param[in] voro_faces An array of integers containing the vertices
         *  of each face. It is supposed to be the output of function
         *  voro::voronoicell_neighbor::face_vertices().
         *
         * @param[in] voro_neighbors An array of integers representing the
         *  neighbor ids of the Voronoi cell. It is supposed to be the output of
         *  function voro::voronoicell_neighbor::neighbors().
         *
         * @param[in] box The bounding box of the geometry model. It is used for
         *  checking if the Voronoi cell crosses the boundary of the surrounding
         *  FE cells, in which case the CPDI algorithm fails.
         */
        VoronoiCell(const types::particle_index                               particle_index,
                    const std::map<types::particle_index, particle_iterator> &particle_map,
                    const std::vector<double>                                &voro_vertices,
                    const std::vector<int>                                   &voro_faces,
                    const std::vector<int>                                   &voro_neighbors,
                    const BoundingBox<dim>                                   &box);

        particle_iterator particle;

        std::vector<particle_iterator> neighbor_particles;

        std::vector<Point<dim>> vertices;

        std::vector<std::vector<unsigned int>> faces;

      };



      template <>
      VoronoiCell<2>::
      VoronoiCell(const types::particle_index                               particle_index,
                  const std::map<types::particle_index, particle_iterator> &particle_map,
                  const std::vector<double>                                &voro_vertices,
                  const std::vector<int>                                   &voro_faces,
                  const std::vector<int>                                   &voro_neighbors,
                  const BoundingBox<2>                                     &box)
      {
        neighbor_particles.clear();
        vertices.clear();
        faces.clear();

        unsigned int target_face = numbers::invalid_unsigned_int;
        for (unsigned int i = 0, j = 0; i < voro_neighbors.size(); ++i)
          {
            if (voro_neighbors[i] >= 0)
              {
                // We find a neighboring Voronoi cell
                const auto mit = particle_map.find(voro_neighbors[i]);
                AssertThrow(mit != particle_map.end(), ExcInternalError());
                neighbor_particles.push_back(mit->second);

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
        for (int k = 0; k < voro_faces[target_face]; ++k)
          {
            unsigned int l = 3 * voro_faces[target_face + k + 1];
            AssertIndexRange(l + 2, voro_vertices.size());

            vertices.emplace_back(Point<2>(voro_vertices[l], voro_vertices[l + 1]));
          }

        // Finally, find the particle owned by the Voronoi cell
        const auto mit = particle_map.find(particle_index);
        AssertThrow(mit != particle_map.end(), ExcInternalError());
        particle = mit->second;
      }



      template <>
      VoronoiCell<3>::
      VoronoiCell(const types::particle_index                               particle_index,
                  const std::map<types::particle_index, particle_iterator> &particle_map,
                  const std::vector<double>                                &voro_vertices,
                  const std::vector<int>                                   &voro_faces,
                  const std::vector<int>                                   &voro_neighbors,
                  const BoundingBox<3>                                     &box)
      {
        neighbor_particles.clear();
        vertices.clear();
        faces.clear();

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

            // Store the neighbor particle
            const auto mit = particle_map.find(voro_neighbors[i]);
            AssertThrow(mit != particle_map.end(), ExcInternalError());
            neighbor_particles.push_back(mit->second);
          }

        // Finally, find the particle owned by the Voronoi cell
        const auto mit = particle_map.find(particle_index);
        AssertThrow(mit != particle_map.end(), ExcInternalError());
        particle = mit->second;
      }



      /**
       * Class holding the data for generalized interpolation in each Voronoi
       * cell.
       */
      template <int dim>
      class VoronoiData
      {
        public:
          /**
           * Default constructor.
           */
          VoronoiData() = default;

          /**
           * Add the value and gradient of the GIMP weighting function
           * corresponding to the given @param vertex_index.
           */
          void
          add(const unsigned int           vertex_index,
              const double                 value,
              const Tensor<1, dim>        &gradient);

          /**
           * Set the volume of the Voronoi cell.
           */
          void
          set_volume(const double volume);

          /**
           * Get the values and gradients of the GIMP weighting functions
           * corresponding to related DoFs (vertices).
           */
          const std::vector<std::pair<unsigned int, std::pair<double, Tensor<1, dim>>>> &
          get_weighting_function_values_and_gradients() const;

          /**
           * Get the volume of particle domain corresponding to the given
           * @param particle_index.
           */
          double get_volume() const;

        private:
          double volume;

          std::vector<std::pair<unsigned int, std::pair<double, Tensor<1, dim>>>>
          weighting_function_values_and_gradients;
      };



      template <int dim>
      void
      VoronoiData<dim>::add(const unsigned int    vertex_index,
                            const double          value,
                            const Tensor<1, dim> &gradient)
      {
        for (auto &function : weighting_function_values_and_gradients)
          if (function.first == vertex_index)
            {
              function.second.first  += value;
              function.second.second += gradient;
              return;
            }

        weighting_function_values_and_gradients.emplace_back(
          std::make_pair(vertex_index, std::make_pair(value, gradient)));
      }



      template <int dim>
      void
      VoronoiData<dim>::set_volume(const double vol)
      {
        volume = vol;
      }



      template <int dim>
      const std::vector<std::pair<unsigned int, std::pair<double, Tensor<1, dim>>>> &
      VoronoiData<dim>::get_weighting_function_values_and_gradients() const
      {
        return weighting_function_values_and_gradients;
      }



      template <int dim>
      double
      VoronoiData<dim>::get_volume() const
      {
        return volume;
      }



      /**
       * Class handling the integration of linear functions on simplex.
       */
      template <int dim>
      class SimplexIntegrator
      {
        public:
          static constexpr unsigned int n_vertices = dim + 1;
          static constexpr double normalization_factor = 1.0 / (dim * (dim - 1));

          /**
           * Constructor.
           *
           * @param[in] vertices Vertices of the simplex.
           */
          SimplexIntegrator(const std::array<Point<dim>, n_vertices> &vertices);

          /**
           * Returns the integration of the value and gradient of the input
           * linear function.
           *
           * @param[in] values Values of the linear function at the vertices.
           */
          std::pair<double, Tensor<1, dim> >
          integrate_linear_function(const std::array<double, n_vertices> &values) const;

          /**
           * Return the volume of the simplex.
           */
          double get_volume() const;

        private:
          std::array<Point<dim>, n_vertices> vertices;

          std::array<Tensor<1, dim>, n_vertices> faces;

          double signed_volume;
      };




      template <int dim>
      SimplexIntegrator<dim>::
      SimplexIntegrator(const std::array<Point<dim>, n_vertices> &verts)
        : vertices(verts)
      {
        for (unsigned int f = 0; f < n_vertices; ++f)
          {
            if constexpr (dim == 2)
              faces[f] = cross_product_2d(vertices[(2 + f) % 3] - vertices[(1 + f) % 3]);
            else
              faces[f] = cross_product_3d((vertices[(3 - f) % 4] - vertices[(1 - f) % 4]),
                                          (vertices[(2 + f) % 4] - vertices[(1 + f) % 4]));
          }

        signed_volume = (faces[0] * (vertices[0] - vertices[1])) * normalization_factor;
      }



      template <int dim>
      std::pair<double, Tensor<1, dim> >
      SimplexIntegrator<dim>::
      integrate_linear_function(const std::array<double, n_vertices> &N) const
      {
        double value = 0;
        for (unsigned int v = 0; v < n_vertices; ++v)
          value += N[v];

        value *= std::abs(signed_volume) / n_vertices;

        Tensor<1, dim> gradient;
        for (unsigned int v = 0; v < n_vertices; ++v)
          gradient += N[v] * faces[v];

        gradient *= normalization_factor;

        return std::make_pair(value, gradient);
      }



      template <int dim>
      double SimplexIntegrator<dim>::get_volume() const
      {
        return std::abs(signed_volume);
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



      /**
       * Compute the area and the area-weighted centroid of a polygon in 2D.
       *
       * @param[in] vertices The vertices of the polygon.
       */
      std::pair<double, Point<2> >
      area_and_centroid(const std::vector<Point<2>> &vertices)
      {
        const unsigned int n_vertices = vertices.size();
        AssertThrow(n_vertices >= 3, ExcInternalError());

        double A  = 0.0; // signed area
        double Cx = 0.0;
        double Cy = 0.0;

        for (unsigned int i = 0, j = n_vertices - 1; i < n_vertices; j = i++)
          {
            const double xi = vertices[i][0], yi = vertices[i][1],
                         xj = vertices[j][0], yj = vertices[j][1];

            const double cross_product = xj * yi - xi * yj;
            A  += cross_product;
            Cx += (xj + xi) * cross_product;
            Cy += (yj + yi) * cross_product;
          }
        A *= 0.5;

        const double one_over_6A = 1.0 / (6.0 * A);
        Cx *= one_over_6A;
        Cy *= one_over_6A;

        return std::make_pair(A, Point<2>(Cx, Cy));
      }



      /**
       * Compute the area and the area-weighted centroid of a polygon in 3D.
       * The polygon is supposed to be a face of a polyhedron.
       *
       * @param[in] points Vertices of the polyhedron.
       * @param[in] vertices An array of integers indicating the vertices of
       *  the selected face of the polyhedron.
       */
      std::pair<double, Point<3> >
      area_and_centroid(const std::vector<Point<3>>     &points,
                        const std::vector<unsigned int> &vertices)
      {
        const unsigned int n_vertices = vertices.size();
        Assert(n_vertices >= 3 && n_vertices <= points.size(),
               ExcInternalError());

        const Point<3> ref = points[vertices[0]];
        Point<3> centroid_sum;
        double area_sum = 0.0;

        for (unsigned int i = 1; i < n_vertices - 1; ++i)
          {
            const Point<3> &v1 = points[vertices[i]];
            const Point<3> &v2 = points[vertices[i + 1]];

            // Triangle (ref, v1, v2)
            const Tensor<1, 3> a = v1 - ref;
            const Tensor<1, 3> b = v2 - ref;

            // Triangle area = 0.5 * |a x b|
            const double A = 0.5 * cross_product_3d(a, b).norm();
            area_sum += A;

            // Triangle centroid
            Point<3> C = (ref + v1 + v2) * (1.0 / 3.0);

            // Weighted sum
            centroid_sum += C * A;
          }

        return std::make_pair(area_sum, centroid_sum / area_sum);
      }



      /**
       * Compute the volume and the volume-weighted centroid of a polyhedron.
       *
       * @param[in] vertices Vertices of the polyhedron.
       * @param[in] faces Faces of the polyhedron.
       */
      std::pair<double, Point<3> >
      volume_and_centroid(const std::vector<Point<3>>                  &vertices,
                          const std::vector<std::vector<unsigned int>> &faces)
      {
        Assert(vertices.size() >= 4, ExcInternalError());
        Assert(faces.size() >= 4, ExcInternalError());

        // Compute the reference point
        Point<3> ref;
        for (const auto &vertex : vertices)
          ref += vertex;
        ref /= vertices.size();

        Point<3> centroid_sum;
        double volume_sum = 0.0;

        for (const auto &face : faces)
          {
            const Point<3> &v0 = vertices[face[0]];

            for (unsigned int i = 1; i < face.size() - 1; ++i)
              {
                const Point<3> &v1 = vertices[face[i]];
                const Point<3> &v2 = vertices[face[i + 1]];

                const Tensor<1, 3> a = v0 - ref;
                const Tensor<1, 3> b = v1 - ref;
                const Tensor<1, 3> c = v2 - ref;

                // Volume of tetrahedron (ref, v0, v1, v2)
                const double V = a * cross_product_3d(b, c) / 6.0;

                // Barycenter of tetrahedron (ref, v0, v1, v2)
                const Point<3> C = (ref + v0 + v1 + v2) * 0.25;

                centroid_sum += C * V;
                volume_sum   += V;
              }
          }

        return std::make_pair(volume_sum, centroid_sum / volume_sum);
      }



      template <int dim>
      bool
      is_inside_unit_cell(const typename Triangulation<dim>::active_cell_iterator &cell,
                          const Point<dim>                                        &p,
                          const double                                             eps)
      {
        for (const unsigned int f : cell->face_indices())
          {
            if (f % 2 == 0)
              {
                if (p[f / 2] < 0.0 - (cell->at_boundary(f) ? eps : 0.0))
                  return false;
              }
            else
              {
                if (p[f / 2] >= 1.0 + (cell->at_boundary(f) ? eps : 0.0))
                  return false;
              }
          }
        return true;
      }



      void
      do_evaluation(const std::vector<Triangulation<2>::active_cell_iterator> &cells,
                    const VoronoiCell<2>                                      &voronoi_cell,
                    const FE_Q<2>                                             &fe,
                    const Mapping<2>                                          &mapping,
                    VoronoiData<2>                                            &data)
      {
        // Tolerance parameter for function is_inside_unit_cell()
        constexpr double eps = 1.e-12;
        constexpr unsigned int dofs_per_cell = 4;
        const unsigned int n_cells = cells.size();

        const std::vector<Point<2>> &voro_vertices = voronoi_cell.vertices;
        const unsigned int n_voro_vertices = voro_vertices.size();

        const auto A_and_C = area_and_centroid(voro_vertices);
        const double area = std::abs(A_and_C.first);
        const Point<2> center = A_and_C.second;

        data.set_volume(area);

        // Arrays storing the values of shape functions at the Voronoi vertices
        std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
        std::array<double, dofs_per_cell> N_c;

        // Create a simplex integrator for each triangle
        std::vector<SimplexIntegrator<2>> simplex_integrators;
        if (n_voro_vertices > 3)
          {
            // If the Voronoi cell has more than 3 vertices, then we need to
            // divide it into n_voro_vertices triangles, the apex of which
            // is the centroidof the Voronoi cell
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                std::array<Point<2>, 3> vertices;
                vertices[0] = voro_vertices[v];
                vertices[1] = voro_vertices[(v + 1) % n_voro_vertices];
                vertices[2] = center;
                simplex_integrators.emplace_back(SimplexIntegrator(vertices));
              }
          }
        else
          {
            std::array<Point<2>, 3> vertices;
            for (unsigned int v = 0; v < 3; ++v)
              vertices[v] = voro_vertices[v];
            simplex_integrators.emplace_back(SimplexIntegrator(vertices));
          }

        for (unsigned int c = 0; c < n_cells; ++c)
          {
            const auto &cell = cells[c];

            // Evaluate the shape functions at the vertices
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                const Point<2> vertex_unit = mapping.transform_real_to_unit_cell(cell, voro_vertices[v]);
                if (is_inside_unit_cell(cell, vertex_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[v][i] = fe.shape_value(i, vertex_unit);
                else
                  std::fill(N_v[v].begin(), N_v[v].end(), 0.0);
              }

            if (n_voro_vertices > 3)
              {
                // Evaluate the shape functions at the centroid
                const Point<2> center_unit = mapping.transform_real_to_unit_cell(cell, center);
                if (is_inside_unit_cell(cell, center_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_c[i] = fe.shape_value(i, center_unit);
                else
                  std::fill(N_c.begin(), N_c.end(), 0.0);

                for (unsigned int v = 0; v < n_voro_vertices; ++v)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        std::array<double, 3> N;
                        N[0] = N_v[v][i];
                        N[1] = N_v[(v + 1) % n_voro_vertices][i];
                        N[2] = N_c[i];

                        // Integrate the values and/or gradients of the generalized basis function
                        // if the basis function is nonzero in this triangle
                        if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0)
                          {
                            const auto value_and_gradient = simplex_integrators[v].integrate_linear_function(N);
                            data.add(cell->vertex_index(i),
                                     value_and_gradient.first / area,
                                     value_and_gradient.second / area);
                          }
                      }
                  }
              }
            else // n_voro_vertices == 3
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    std::array<double, 3> N;
                    for (unsigned int v = 0; v < 3; ++v)
                      N[v] = N_v[v][i];

                    // Integrate the values and/or gradients of the generalized basis function
                    // if the basis function is nonzero in this triangle
                    if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0)
                      {
                        const auto value_and_gradient = simplex_integrators[0].integrate_linear_function(N);
                        data.add(cell->vertex_index(i),
                                 value_and_gradient.first / area,
                                 value_and_gradient.second / area);
                      }
                  }
              }
          }
      }



      void
      do_evaluation(const std::vector<Triangulation<3>::active_cell_iterator> &cells,
                    const VoronoiCell<3>                                      &voronoi_cell,
                    const FE_Q<3>                                             &fe,
                    const Mapping<3>                                          &mapping,
                    VoronoiData<3>                                            &data)
      {
        // Tolerance parameter for function is_inside_unit_cell()
        constexpr double eps = 1.e-12;
        constexpr unsigned int dofs_per_cell = 8;
        const unsigned int n_cells = cells.size();

        const std::vector<Point<3>> &voro_vertices = voronoi_cell.vertices;
        const std::vector<std::vector<unsigned int>> &voro_faces = voronoi_cell.faces;
        const unsigned int n_voro_vertices = voro_vertices.size();
        const unsigned int n_voro_faces    = voro_faces.size();

        const auto V_and_C = volume_and_centroid(voro_vertices, voro_faces);
        const double volume = std::abs(V_and_C.first);
        const Point<3> voro_center = V_and_C.second;

        data.set_volume(volume);

        // Arrays storing the values of shape functions at the Voronoi vertices
        std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
        std::vector<std::array<double, dofs_per_cell>> N_fc(n_voro_faces);
        std::array<double, dofs_per_cell> N_vc;

        // Create a simplex integrator for each tetrahedron
        std::vector<SimplexIntegrator<3>> simplex_integrators;
        std::vector<Point<3>> face_centers(n_voro_faces);
        if (n_voro_vertices > 4)
          {
            // If the Voronoi cell has more than 4 vertices, then we need to
            // divide it into n_voro_faces prisms, the apex of which is the
            // centroid of the Voronoi cell
            for (unsigned int f = 0; f < n_voro_faces; ++f)
              {
                const unsigned int n_face_vertices = voro_faces[f].size();
                if (n_face_vertices > 3)
                  {
                    // If the face has more than 3 vertices, then we need to
                    // divide it into n_face_vertices triangles, the apex of
                    // which is the centroid of the face
                    face_centers[f] = area_and_centroid(voro_vertices, voro_faces[f]).second;
                    for (unsigned int fv = 0; fv < n_face_vertices; ++fv)
                      {
                        std::array<Point<3>, 4> vertices;
                        vertices[0] = voro_vertices[voro_faces[f][fv]];
                        vertices[1] = voro_vertices[voro_faces[f][(fv + 1) % n_face_vertices]];
                        vertices[2] = face_centers[f];
                        vertices[3] = voro_center;
                        simplex_integrators.emplace_back(SimplexIntegrator(vertices));
                      }
                  }
                else // n_face_vertices == 3
                  {
                    std::array<Point<3>, 4> vertices;
                    for (unsigned int fv = 0; fv < 3; ++fv)
                      vertices[fv] = voro_vertices[voro_faces[f][fv]];
                    vertices[3] = voro_center;
                    simplex_integrators.emplace_back(SimplexIntegrator(vertices));
                  }
              }
          }
        else // n_voro_vertices == 4
          {
            std::array<Point<3>, 4> vertices;
            for (unsigned int v = 0; v < 4; ++v)
              vertices[v] = voro_vertices[v];
            simplex_integrators.emplace_back(SimplexIntegrator(vertices));
          }

        for (unsigned int c = 0; c < n_cells; ++c)
          {
            const auto &cell = cells[c];

            // Evaluate the values of shape functions at the vertices
            unsigned int integrator = 0;
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                const Point<3> vertex_unit = mapping.transform_real_to_unit_cell(cell, voro_vertices[v]);
                if (is_inside_unit_cell(cell, vertex_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[v][i] = fe.shape_value(i, vertex_unit);
                else
                  std::fill(N_v[v].begin(), N_v[v].end(), 0.0);
              }

            if (n_voro_vertices > 4)
              {
                // Evaluate the shape functions at the centroid of the cell
                const Point<3> voro_center_unit = mapping.transform_real_to_unit_cell(cell, voro_center);
                if (is_inside_unit_cell(cell, voro_center_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_vc[i] = fe.shape_value(i, voro_center_unit);
                else
                  std::fill(N_vc.begin(), N_vc.end(), 0.0);

                for (unsigned int f = 0; f < n_voro_faces; ++f)
                  {
                    const unsigned int n_face_vertices = voro_faces[f].size();
                    if (n_face_vertices > 3)
                      {
                        // Evaluate the shape functions at the centroid of the face
                        const Point<3> face_center_unit = mapping.transform_real_to_unit_cell(cell, face_centers[f]);
                        if (is_inside_unit_cell(cell, face_center_unit, eps))
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            N_fc[f][i] = fe.shape_value(i, face_center_unit);
                        else
                          std::fill(N_fc[f].begin(), N_fc[f].end(), 0.0);

                        for (unsigned int fv = 0; fv < n_face_vertices; ++fv)
                          {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                std::array<double, 4> N;
                                N[0] = N_v[voro_faces[f][fv]][i];
                                N[1] = N_v[voro_faces[f][(fv + 1) % n_face_vertices]][i];
                                N[2] = N_fc[f][i];
                                N[3] = N_vc[i];

                                // Integrate the values and/or gradients of the generalized basis function
                                // if the basis function is nonzero in this tetrahedron
                                if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0 || N[3] != 0.0)
                                  {
                                    const auto value_and_gradient = simplex_integrators[integrator++].integrate_linear_function(N);
                                    data.add(cell->vertex_index(i),
                                             value_and_gradient.first / volume,
                                             value_and_gradient.second / volume);
                                  }
                              }
                          }
                      }
                    else // n_face_vertices == 3
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            std::array<double, 4> N;
                            for (unsigned int fv = 0; fv < 3; ++fv)
                              N[fv] = N_v[voro_faces[f][fv]][i];
                            N[3] = N_vc[i];

                            // Integrate the values and/or gradients of the generalized basis function
                            // if the basis function is nonzero in this tetrahedron
                            if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0 || N[3] != 0.0)
                              {
                                const auto value_and_gradient = simplex_integrators[integrator++].integrate_linear_function(N);
                                data.add(cell->vertex_index(i),
                                         value_and_gradient.first / volume,
                                         value_and_gradient.second / volume);
                              }
                          }
                      }
                  }

                AssertDimension(integrator, simplex_integrators.size());
              }
            else // n_voro_vertices == 4
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    std::array<double, 4> N;
                    for (unsigned int v = 0; v < 4; ++v)
                      N[v] = N_v[v][i];

                    // Integrate the values and/or gradients of the generalized basis function
                    // if the basis function is nonzero in this triangle
                    if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0 || N[3] != 0.0)
                      {
                        const auto value_and_gradient = simplex_integrators[0].integrate_linear_function(N);
                        data.add(cell->vertex_index(i),
                                 value_and_gradient.first / volume,
                                 value_and_gradient.second / volume);
                      }
                  }
              }
          }
      }



      /**
       * Calculate the volume and the values/gradients of the generalized basis
       * functions for particles in a given cell.
       *
       * @param[in] cell The target cell
       * @param[in] neighbors Collection of the surrounding cells of all the
       *  vertices of the target cell (including itself).
       * @param[in] particle_handler The ParticleHandler object that handles
       *  the particles.
       * @param[in] fe The finite element that defines the basis functions of
       *  the fields to be interpolated.
       * @param[in] mapping Used for transforming a point from real cell to
       *  reference cell.
       * @param[in] box Bounding box of the geometry model. It is used for
       *  checking if a particle domain is out of reach.
       * @param[in,out] data Holding the calculated quantities.
       */
      template <int dim>
      std::vector<std::vector<Point<dim>>>
      evaluate(const typename Triangulation<dim>::active_cell_iterator              &cell,
               const std::vector<typename Triangulation<dim>::active_cell_iterator> &neighbors,
               const Particles::ParticleHandler<dim>                                &particle_handler,
               const FE_Q<dim>                                                      &fe,
               const Mapping<dim>                                                   &mapping,
               const BoundingBox<dim>                                               &box,
               std::vector<VoronoiData<dim>>                                        &data)
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

        voro::container container(corner1[0],                                // ax
                                  corner2[0],                                // bx
                                  corner1[1],                                // ay
                                  corner2[1],                                // by
                                  (dim > 2 ? corner1[2] : -diameter * 0.01), // az
                                  (dim > 2 ? corner2[2] :  diameter * 0.01), // bz
                                  n_blocks[0],                               // nx
                                  n_blocks[1],                               // ny
                                  n_blocks[2],                               // nz
                                  false,                                     // xperiodic
                                  false,                                     // yperiodic
                                  false,                                     // zperiodic
                                  8);                                        // init_mem

        // Put all the particles in the current cell and its neighbors into the container,
        // and build a map from particle indices to particle iterators
        std::map<types::particle_index, typename Particles::ParticleHandler<dim>::particle_iterator> particle_map;
        for (const auto &neighbor : neighbors)
          {
            const auto &particles_in_cell = particle_handler.particles_in_cell(neighbor);
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const Point<dim> &location = particle->get_location();
                container.put(particle->get_local_index(), location[0], location[1], (dim > 2 ? location[2] : 0.0));
                particle_map.insert(std::make_pair(particle->get_local_index(), particle));
              }
          }

        // Collect the local indices of the particles in the current cell
        std::set<types::particle_index> particles_in_this_cell;
        for (const auto &particle : particle_handler.particles_in_cell(cell))
          particles_in_this_cell.insert(particle.get_local_index());

        // Loop over the particles in the current cell and compute each Voronoi cell
        voro::voronoicell_neighbor voro_cell;
        std::vector<double> voro_vertices;
        std::vector<int> voro_face_vertices;
        std::vector<int> voro_neighbors;
        std::vector<double> voro_volumes;

        std::vector<VoronoiCell<dim>> voronoi_cells;

        voro::c_loop_all loop(container);
        AssertThrow(loop.start(), ExcMessage("An error occurs when calling voro::c_loop_all::start()."));

        do
          {
            const types::particle_index particle_index = static_cast<types::particle_index>(loop.pid());
            if (particles_in_this_cell.find(particle_index) == particles_in_this_cell.end())
              continue;

            AssertThrow(container.compute_cell(voro_cell, loop),
                        ExcMessage("An error occurs when calling voro::container::compute_cell()."));

            double x, y, z;
            loop.pos(x, y, z);
            voro_cell.vertices(x, y, z, voro_vertices);
            voro_cell.face_vertices(voro_face_vertices);
            voro_cell.neighbors(voro_neighbors);
                                                        
            voronoi_cells.emplace_back(VoronoiCell<dim>(particle_index,
                                                        particle_map,
                                                        voro_vertices,
                                                        voro_face_vertices,
                                                        voro_neighbors,
                                                        box));

          }
        while (loop.inc());

        // Do the actual evaluation
        for (const auto &voronoi_cell : voronoi_cells)
          do_evaluation(neighbors, voronoi_cell, fe, mapping, data[voronoi_cell.particle->get_local_index()]);

        std::vector<std::vector<Point<dim>>> vorocells;
        for (const auto &vorocell : voronoi_cells)
          vorocells.push_back(vorocell.vertices);
        return vorocells;
      }
    }
  }
#endif /*ASPECT_WITH_VORO*/



  template <int dim>
  void
  Simulator<dim>::
  perform_convected_particle_domain_interpolation(const std::vector<AdvectionField> &advection_fields)
  {
    if (advection_fields.size() == 0)
      return;

#ifdef ASPECT_WITH_VORO
    computing_timer.enter_subsection("Particles: CPDI");

    // The container of voro++ is a box, so we can only use the box model without initial topography
    // and mesh deformation.
    AssertThrow(Plugins::plugin_type_matches<GeometryModel::Box<dim>>(*geometry_model),
                ExcMessage("Compositional field method ``cpdi'' only works when the geometry model is ``box''."));
    AssertThrow(Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(*initial_topography_model),
                ExcMessage("Compositional field method ``cpdi'' only works when the initial topography model is ``zero topography''."));
    AssertThrow(mesh_deformation == nullptr,
                ExcMessage("Compositional field method ``cpdi'' only works when mesh deformation is not enabled."));

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

    // Create a map from field index to particle property index
    std::map<unsigned int, unsigned int> field_to_property_map;
    if (parameters.mapped_particle_properties.size() != 0)
      {
        const Particle::Property::ParticlePropertyInformation &property_info
          = cpdi_particle_manager->get_property_manager().get_data_info();

        for (const auto &field : advection_fields)
          {
            unsigned int property_index = numbers::invalid_unsigned_int;

            if (parameters.mapped_particle_properties.size() != 0)
              {
                const std::string &field_name = parameters.mapped_particle_properties[field.compositional_variable].first;
                AssertThrow(property_info.fieldname_exists(field_name),
                            ExcMessage("The particle properties to be interpolated by the CPDI method are not handled by "
                                       "the same particle manager. Currently this is not supported."));

                const std::pair<std::string, unsigned int> property_and_component
                  = parameters.mapped_particle_properties[field.compositional_variable];

                property_index = property_info.get_position_by_field_name(property_and_component.first) + property_and_component.second;
              }
            else
              {
                property_index = std::count(introspection.compositional_field_methods.begin(),
                                            introspection.compositional_field_methods.begin() + field.compositional_variable,
                                            Parameters<dim>::AdvectionFieldMethod::particles)
                                 +
                                 std::count(introspection.compositional_field_methods.begin(),
                                            introspection.compositional_field_methods.begin() + field.compositional_variable,
                                            Parameters<dim>::AdvectionFieldMethod::cpdi);

                AssertThrow(property_index < property_info.n_components(),
                            ExcMessage("Can not automatically match particle properties to fields, because there are "
                                       "more fields that are marked as particle/cpdi than particle properties."));
              }

            field_to_property_map.insert(std::make_pair(field.compositional_variable, property_index));
          }
      }

    // We have checked that all the input fields are discretized by Q1 element in
    // source/simulator/parameters.cc, so all the input fields share the same sparsity block.
    const unsigned int sparsity_block_idx = advection_fields[0].sparsity_pattern_block_index(introspection);
    const unsigned int block0_idx         = advection_fields[0].block_index(introspection);
    if (sparsity_block_idx != block0_idx)
      system_matrix.block(block0_idx, block0_idx).reinit(system_matrix.block(sparsity_block_idx, sparsity_block_idx));

    system_matrix.block(block0_idx, block0_idx) = 0;
    for (const auto &field : advection_fields)
      {
        const unsigned int block_idx = field.block_index(introspection);
        system_rhs.block(block_idx) = 0;
      }

    LinearAlgebra::BlockVector lumped_mass_matrix(introspection.index_sets.system_partitioning, mpi_communicator);

    // Create the vertex-to-cell map
    GridTools::Cache<dim> grid_cache(triangulation, *mapping);
    const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

    // Get the bounding box of the computational domain
    const GeometryModel::Box<dim> &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(*geometry_model);
    const BoundingBox<dim> bounding_box(std::make_pair(box.get_origin(), box.get_origin() + box.get_extents()));

    const Particles::ParticleHandler<dim> &particle_handler = cpdi_particle_manager->get_particle_handler();
    const FE_Q<dim> &field_fe = dynamic_cast<const FE_Q<dim>&>(finite_element.base_element(advection_fields[0].base_element(introspection)));

    const unsigned int n_fields = advection_fields.size();

    // Now loop over the locally owned active cells and collect the CPDI data
    std::vector<internal::CPDI::VoronoiData<dim>> data(particle_handler.get_max_local_particle_index());
    double local_volume = 0.0;
    std::vector<std::vector<Point<dim>>> vorocells;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // Find the cells neighboring the current cell
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cell_set;
          for (const unsigned int v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cell_set.insert(vertex_to_cell_map[vertex_index].begin(),
                                          vertex_to_cell_map[vertex_index].end());
            }

          std::vector<typename Triangulation<dim>::active_cell_iterator>
          neighboring_cells(neighboring_cell_set.begin(), neighboring_cell_set.end());

          const std::vector<std::vector<Point<dim>>> local_vorocells =
          internal::CPDI::evaluate(cell,
                                   neighboring_cells,
                                   particle_handler,
                                   field_fe,
                                   *mapping,
                                   bounding_box,
                                   data);
          vorocells.insert(vorocells.end(), local_vorocells.begin(), local_vorocells.end());

#if DEBUG
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            local_volume += data[particle.get_local_index()].get_volume();
#endif
        }

    if (pre_refinement_step == 0)
    {
      std::ofstream out("voronoi-" + Utilities::int_to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".vtk");
      out << "# vtk DataFile Version 3.0\n";
      out << "Voro++ Voronoi diagram\n";
      out << "ASCII\n";
      out << "DATASET POLYDATA\n";

      unsigned int n_points = 0;
      for (const auto &vorocell : vorocells)
        n_points += vorocell.size();
      out << "POINTS " << n_points << " float\n";
      for (const auto &vorocell : vorocells)
        for (const auto &vertex : vorocell)
          out << vertex[0] << ' ' << vertex[1] << " 0.0\n";

      out << "POLYGONS " << vorocells.size() << ' ' << vorocells.size() + n_points << "\n";
      unsigned int point_index = 0;
      for (const auto &vorocell : vorocells)
      {
        out << vorocell.size();
        for (unsigned int v = 0; v < vorocell.size(); ++v)
          out << ' ' << point_index++;
        out << "\n";
      }

    }

    // Build a map from vertex indices to the corresponding DoF indices
    std::vector<std::vector<types::global_dof_index>> vertex_to_dof_indices(triangulation.n_vertices());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (!cell->is_artificial())
        for (const unsigned int v : cell->vertex_indices())
          {
            const unsigned int vertex_index = cell->vertex_index(v);
            if (vertex_to_dof_indices[vertex_index].size() == 0)
              {
                vertex_to_dof_indices[vertex_index].resize(n_fields);
                for (unsigned int field_index = 0; field_index < advection_fields.size(); ++field_index)
                  vertex_to_dof_indices[vertex_index][field_index] =
                    cell->vertex_dof_index(v, advection_fields[field_index].component_index(introspection));
              }
          }

    // Assemble the CPDI system
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            {
              const types::particle_index particle_index = particle.get_local_index();
              const auto &particle_data = data[particle_index];
              const double V_p = particle_data.get_volume();
              const std::vector<std::pair<unsigned int, std::pair<double, Tensor<1, dim>>>> &
              weighting_functions = particle_data.get_weighting_function_values_and_gradients();
              const ArrayView<const double> particle_properties = particle.get_properties();

              const unsigned int n_dofs_per_field = weighting_functions.size();
              FullMatrix<double> particle_matrix(n_dofs_per_field, n_dofs_per_field);
              Vector<double> particle_lumped_mass_matrix(n_dofs_per_field);
              std::vector<Vector<double>> particle_rhs(n_fields, Vector<double>(n_dofs_per_field));
              std::vector<std::vector<types::global_dof_index>> particle_dofs(n_fields, std::vector<types::global_dof_index>(n_dofs_per_field));

              for (unsigned int i = 0; i < n_dofs_per_field; ++i)
                {
                  const unsigned int vertex_index = weighting_functions[i].first;
                  AssertDimension(vertex_to_dof_indices[vertex_index].size(), n_fields);

                  const double phi_ip = weighting_functions[i].second.first;
                  particle_lumped_mass_matrix(i) = V_p * phi_ip;

                  for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                    {
                      particle_dofs[field_index][i] = vertex_to_dof_indices[vertex_index][field_index];

                      const double property = particle_properties[field_to_property_map[field_index]];
                      particle_rhs[field_index](i) = V_p * phi_ip * property;
                    }

                  for (unsigned int j = 0; j < n_dofs_per_field; ++j)
                    {
                      const double phi_jp = weighting_functions[j].second.first;
                      particle_matrix(i, j) = V_p * phi_ip * phi_jp;
                    }
                }

              current_constraints.distribute_local_to_global(particle_matrix,
                                                             particle_dofs[0],
                                                             system_matrix);

              current_constraints.distribute_local_to_global(particle_lumped_mass_matrix,
                                                             particle_dofs[0],
                                                             lumped_mass_matrix);

              for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                current_constraints.distribute_local_to_global(particle_rhs[field_index],
                                                               particle_dofs[field_index],
                                                               system_rhs);
            }
        }

    system_matrix.compress(VectorOperation::add);
    lumped_mass_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

#if DEBUG
    const double total_volume = Utilities::MPI::sum(local_volume, mpi_communicator);
    const double tria_volume  = GridTools::volume(triangulation, *mapping);
    Assert(std::abs(total_volume - tria_volume) / tria_volume < 1.e-4,
           ExcMessage("The total volume of particle domains ("
                      + Utilities::to_string(total_volume)
                      + ") does not match the volume of the triangulation ("
                      + Utilities::to_string(tria_volume)
                      + ")."));
#else
    (void)local_volume;
#endif /*DEBUG*/

    // Set the preconditioner
    LinearAlgebra::PreconditionAMG preconditioner;
    LinearAlgebra::PreconditionAMG::AdditionalData amg_data;

#if DEAL_II_VERSION_GTE(9,7,0)
    amg_data.constant_modes = DoFTools::extract_constant_modes(
                                dof_handler,
                                introspection.component_masks.compositional_fields[advection_fields[0].compositional_variable]);
#else
    std::vector<std::vector<bool>> constant_modes;
    DoFTools::extract_constant_modes(
      dof_handler,
      introspection.component_masks.compositional_fields[advection_fields[0].compositional_variable],
      constant_modes);
    amg_data.constant_modes = constant_modes;
#endif

    amg_data.elliptic = true;
    amg_data.higher_order_elements = false;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    preconditioner.initialize(system_matrix.block(sparsity_block_idx,
                                                  sparsity_block_idx),
                              amg_data);

    // Create a distributed vector
    LinearAlgebra::BlockVector distributed_solution(introspection.index_sets.system_partitioning,
                                                    mpi_communicator);

    for (const auto &field : advection_fields)
      {
        const IndexSet locally_owned_field_dofs =
          dof_handler.locally_owned_dofs() &                                        
          Utilities::extract_locally_active_dofs_with_component(dof_handler, introspection.component_masks.compositional_fields[field.compositional_variable]);
        for (auto eit = locally_owned_field_dofs.begin(); eit != locally_owned_field_dofs.end(); ++eit)
          if (!current_constraints.is_constrained(*eit))
            distributed_solution[*eit] = system_rhs[*eit] / lumped_mass_matrix[*eit];
        /*
        pcout << "   Solving CPDI system for " << field.name(introspection)
              << "... " << std::flush;

        const unsigned int block_idx = field.block_index(introspection);
        distributed_solution.block(block_idx) = current_linearization_point.block(block_idx);
        current_constraints.set_zero(distributed_solution);

        SolverControl solver_control(1000, 1.e-12 * system_rhs.block(block_idx).l2_norm());
        SolverCG<LinearAlgebra::Vector> solver(solver_control);

        try
          {
            solver.solve(system_matrix.block(sparsity_block_idx, sparsity_block_idx),
                         distributed_solution.block(block_idx),
                         system_rhs.block(block_idx),
                         preconditioner);
          }
        catch (const std::exception &exc)
          {
            // if the solver fails, report the error from processor 0 with some additional
            // information about its location, and throw a quiet exception on all other
            // processors
            Utilities::throw_linear_solver_failure_exception("iterative CPDI solver",
                                                             "perform_convected_particle_domain_interpolation()",
                                                             std::vector<SolverControl> {solver_control},
                                                             exc,
                                                             mpi_communicator);
          }

        pcout << solver_control.last_step() << " iterations." << std::endl;*/
      }

    current_constraints.distribute(distributed_solution);
    for (const auto &field : advection_fields)
      {
        const unsigned int block_idx = field.block_index(introspection);
        solution.block(block_idx) = distributed_solution.block(block_idx);
      }

    computing_timer.leave_subsection("Particles: CPDI");

#endif /*ASPECT_WITH_VORO*/
  }



  template <int dim>
  void
  Simulator<dim>::
  make_cpdi_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &sp,
                             const AffineConstraints<double> &constraints) const
  {
    // Find the first compositional field advected by the CPDI method
    unsigned int first_cpdi_field_component = numbers::invalid_unsigned_int;
    for (unsigned int c = 0; c < introspection.n_compositional_fields; ++c)
      if (introspection.compositional_field_methods[c] == Parameters<dim>::AdvectionFieldMethod::cpdi)
        {
          first_cpdi_field_component = introspection.component_indices.compositional_fields[c];
          break;
        }
    AssertThrow(first_cpdi_field_component != numbers::invalid_unsigned_int, ExcInternalError());

    // Create the vertex-to-cell map
    GridTools::Cache<dim> grid_cache(triangulation, *mapping);
    const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

    std::vector<types::global_dof_index> coupled_dofs;

    // Loop over the locally owned cells and add the nonzero entries of CPDI system
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // All the DoFs in the neighborhood of a cell corresponding to the CPDI field
          // are possible to be coupled
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cells;
          for (const unsigned int v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cells.insert(vertex_to_cell_map[vertex_index].begin(),
                                       vertex_to_cell_map[vertex_index].end());
            }

          // Since the CPDI method requires the fields to be discretized by FE_Q(1) element,
          // we only need to loop over the vertices and extract the DoFs corresponding to
          // the first CPDI field
          std::set<types::global_dof_index> coupled_dofs;
          for (const auto &neighbor : neighboring_cells)
            {
              typename DoFHandler<dim>::active_cell_iterator dof_cell(&triangulation,
                                                                      neighbor->level(),
                                                                      neighbor->index(),
                                                                      &dof_handler);

              for (const unsigned int v : dof_cell->vertex_indices())
                coupled_dofs.insert(dof_cell->vertex_dof_index(v, first_cpdi_field_component));

              constraints.add_entries_local_to_global(std::vector<types::global_dof_index>(coupled_dofs.begin(),
                                                                                           coupled_dofs.end()),
                                                      sp, true);
            }
        }
  }
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  template void Simulator<dim>::perform_convected_particle_domain_interpolation(const std::vector<AdvectionField> &); \
  template void Simulator<dim>::make_cpdi_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &, \
                                                           const AffineConstraints<double> &) const;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
