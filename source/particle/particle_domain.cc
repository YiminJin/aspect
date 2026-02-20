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

#include <aspect/particle/particle_domain.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_q.h>

#ifdef ASPECT_WITH_VORO
#include <voro++.hh>
#endif

namespace aspect
{
  namespace Particle
  {
    namespace ParticleDomain
    {
      /*-------------------------- class FaceData --------------------------*/

      template <int dim>
      void
      FaceData<dim>::reinit(const types::particle_index max_local_particle_index)
      {
        // Clear the data
        handles.clear();
        n_faces_per_vorocell.clear();
        face_measures.clear();
        neighbor_particles.clear();

        // Resize the arrays storing the data handles and the face numbers
        handles.resize(max_local_particle_index, numbers::invalid_unsigned_int);
        n_faces_per_vorocell.resize(max_local_particle_index, std::numeric_limits<std::uint16_t>::max());
      }



      template <int dim>
      void
      FaceData<dim>::
      push_back(const types::particle_index                              particle_index,
                const std::vector<std::pair<double, particle_iterator>> &face_data,
                const ParticleHandler<dim>                              &particle_handler)
      {
        AssertIndexRange(particle_index, handles.size());
        Assert(handles[particle_index] == numbers::invalid_unsigned_int,
               ExcMessage("Face data of particle " 
                          + Utilities::int_to_string(particle_index)
                          + " already exists!"));

        const unsigned int n_faces = face_data.size();
        n_faces_per_vorocell[particle_index] = static_cast<std::uint16_t>(n_faces);

        unsigned int start_index = face_measures.size();
        handles[particle_index] = start_index;
        face_measures.resize(start_index + n_faces);
        neighbor_particles.resize(start_index + n_faces);

        for (unsigned int f = 0; f < n_faces; ++f)
          {
            const unsigned int index = start_index + f;
            face_measures[index] = face_data[f].first;

            const particle_iterator &particle = face_data[f].second;
            if (particle->state() != IteratorState::valid)
              continue;
           
            unsigned int particle_index_within_cell = numbers::invalid_unsigned_int;
            const auto &cell = particle->get_surrounding_cell();
            const auto &particles_in_cell = particle_handler.particles_in_cell(cell);
            auto p = particles_in_cell.begin();
            for (unsigned int np = 0; p != particles_in_cell.end(); ++p, ++np)
              if (face_data[f].second == p)
                {
                  particle_index_within_cell = np;
                  break;
                }
            Assert(particle_index_within_cell != numbers::invalid_unsigned_int,
                   ExcInternalError());

            auto &particle_indicator = neighbor_particles[index];
            particle_indicator.cell_index = static_cast<std::uint32_t>(cell->index());
            particle_indicator.cell_level = static_cast<std::uint16_t>(cell->level());
            particle_indicator.particle_index_within_cell
              = static_cast<std::uint16_t>(particle_index_within_cell);
          }
      }



      template <int dim>
      unsigned int
      FaceData<dim>::
      n_faces(const types::particle_index particle_index) const
      {
        AssertIndexRange(particle_index, n_faces_per_vorocell.size());
        return n_faces_per_vorocell[particle_index];
      }



      template <int dim>
      double
      FaceData<dim>::
      face_measure(const types::particle_index particle_index,
                   const unsigned int          face_index) const
      {
        AssertIndexRange(particle_index, handles.size());
        AssertIndexRange(face_index, n_faces_per_vorocell[particle_index]);

        return face_measures[handles[particle_index] + face_index];
      }



      template <int dim>
      typename FaceData<dim>::particle_iterator
      FaceData<dim>::
      neighbor_particle(const types::particle_index  particle_index,
                        const unsigned int           face_index,
                        const ParticleHandler<dim>  &particle_handler,
                        const Triangulation<dim>    &triangulation) const
      {
        AssertIndexRange(particle_index, handles.size());
        AssertIndexRange(face_index, n_faces_per_vorocell[particle_index]);

        const auto &particle_indicator = neighbor_particles[handles[particle_index] + face_index];

        // If the face is at boundary, return an invalid iterator
        if (particle_indicator.particle_index_within_cell
            == std::numeric_limits<std::uint16_t>::max())
          return typename ParticleHandler<dim>::particle_iterator();

        typename Triangulation<dim>::cell_iterator cell(&triangulation,
                                                        particle_indicator.cell_index,
                                                        particle_indicator.cell_level);

        AssertIndexRange(particle_indicator.particle_index_within_cell,
                         particle_handler.n_particles_in_cell(cell));
        particle_iterator neighbor = particle_handler.particles_in_cell(cell).begin();
        std::advance(neighbor, particle_indicator.particle_index_within_cell);
        return neighbor;
      }



      /*-------------------------- class CPDIData --------------------------*/

      template <int dim>
      void
      CPDIData<dim>::reinit(const types::particle_index max_local_particle_index)
      {
        // Clear the data
        handles.clear();
        n_relevant_vertices_per_vorocell.clear();
        relevant_vertices.clear();
        weighting_function_data.clear();

        // Resize the arrays storing the data handles and the relevant vertex numbers
        handles.resize(max_local_particle_index, numbers::invalid_unsigned_int);
        n_relevant_vertices_per_vorocell.resize(max_local_particle_index, std::numeric_limits<std::uint8_t>::max());
      }



      template <int dim>
      void
      CPDIData<dim>::
      push_back(const types::particle_index                                      particle_index,
                const std::map<unsigned int, std::pair<double, Tensor<1, dim>>> &local_weighting_functions)
      {
        AssertIndexRange(particle_index, handles.size());
        Assert(handles[particle_index] == numbers::invalid_unsigned_int,
               ExcMessage("CPDI data of particle "
                          + Utilities::int_to_string(particle_index)
                          + " already exists!"));

        const unsigned int n_relevant_vertices = local_weighting_functions.size();
        n_relevant_vertices_per_vorocell[particle_index] = static_cast<std::uint8_t>(n_relevant_vertices);

        unsigned int start_index = relevant_vertices.size();
        handles[particle_index] = start_index;
        n_relevant_vertices_per_vorocell.resize(start_index + n_relevant_vertices);
        weighting_function_data.resize(start_index + n_relevant_vertices * (1 + dim));

        auto weighting_function = local_weighting_functions.begin();
        for (unsigned int v = 0; v < n_relevant_vertices; ++v, ++weighting_function)
          {
            const unsigned int index1 = start_index + v;
            n_relevant_vertices_per_vorocell[index1] = weighting_function->first;

            const unsigned int index2 = index1 * (1 + dim);
            weighting_function_data[index2] = weighting_function->second.first;
            for (unsigned int d = 0; d < dim; ++d)
              weighting_function_data[index2 + 1 + d] = weighting_function->second.second[d];
          }
      }



      template <int dim>
      unsigned int
      CPDIData<dim>::
      n_relevant_vertices(const types::particle_index particle_index) const
      {
        AssertIndexRange(particle_index, n_relevant_vertices_per_vorocell.size());
        return n_relevant_vertices_per_vorocell[particle_index];
      }



      template <int dim>
      unsigned int
      CPDIData<dim>::
      relevant_vertex_index(const types::particle_index particle_index,
                            const unsigned int          i) const
      {
        AssertIndexRange(particle_index, handles.size());
        AssertIndexRange(i, n_relevant_vertices_per_vorocell[particle_index]);

        return relevant_vertices[handles[particle_index] + i];
      }



      template <int dim>
      double
      CPDIData<dim>::
      weighting_function_value(const types::particle_index particle_index,
                               const unsigned int          i) const
      {
        AssertIndexRange(particle_index, handles.size());
        AssertIndexRange(i, n_relevant_vertices_per_vorocell[particle_index]);

        return weighting_function_data[(handles[particle_index] + i) * (1 + dim)];
      }



      template <int dim>
      Tensor<1, dim>
      CPDIData<dim>::
      weighting_function_gradient(const types::particle_index particle_index,
                                  const unsigned int          i) const
      {
        AssertIndexRange(particle_index, handles.size());
        AssertIndexRange(i, n_relevant_vertices_per_vorocell[particle_index]);

        Tensor<1, dim> gradient;
        const unsigned int start_index = (handles[particle_index] + i) * (1 + dim) + 1;
        for (unsigned int d = 0; d < dim; ++d)
          gradient[d] = weighting_function_data[start_index + d];

        return gradient;
      }
    }

      

    /*------------------- Helper functions and structures -------------------*/

    namespace internal
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
        /**
         * Constructor.
         *
         * @param[in] particle_index The local particle ID.
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
         * @param[in] bounding_box The bounding box of the geometry model. It is 
         *  used for checking if the Voronoi cell crosses the boundary of the 
         *  surrounding FE cells, in which case the CPDI algorithm fails.
         */
        VoronoiCell(const types::particle_index  particle_index,
                    const std::vector<double>   &voro_vertices,
                    const std::vector<int>      &voro_faces,
                    const std::vector<int>      &voro_neighbors,
                    const BoundingBox<dim>      &bounding_box);

        types::particle_index particle_index;

        std::vector<Point<dim>> vertices;

        std::vector<std::vector<unsigned int>> faces;

        std::vector<types::particle_index> neighbors;
      };



      template <>
      VoronoiCell<2>::
      VoronoiCell(const types::particle_index  particle_index,
                  const std::vector<double>   &voro_vertices,
                  const std::vector<int>      &voro_faces,
                  const std::vector<int>      &voro_neighbors,
                  const BoundingBox<2>        &bounding_box)
        : particle_index(particle_index)
      {
        vertices.clear();
        faces.clear();
        neighbors.clear();

        std::vector<std::vector<unsigned int>> faces_3d;
        unsigned int target_face = numbers::invalid_unsigned_int;
        for (unsigned int i = 0, j = 0; i < voro_neighbors.size(); ++i)
          {
            if (voro_neighbors[i] >= 0)
              {
                const unsigned int n_vertices = voro_faces[j];
                AssertDimension(n_vertices, 4);
                std::vector<unsigned int> face_3d(n_vertices);
                for (unsigned int k = 0; k < n_vertices; ++k)
                  face_3d[k] = voro_faces[j + k + 1];
                faces_3d.push_back(face_3d);

                neighbors.push_back(static_cast<types::particle_index>(voro_neighbors[i]));
              }
            else
              {
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
                  {
                    AssertThrow(face_is_at_boundary(v[0], n, bounding_box),
                                ExcMessage("One of the particle domains crosses the boundary of "
                                           "the bounding box of its surrounding cells. In this case, "
                                           "it is impossible to get all the vertices of the particle "
                                           "domain. Please increase the lower limit of particles per cell."));

                    const unsigned int n_vertices = voro_faces[j];
                    AssertDimension(n_vertices, 4);
                    std::vector<unsigned int> face_3d(n_vertices);
                    for (unsigned int k = 0; k < n_vertices; ++k)
                      face_3d[k] = voro_faces[j + k + 1];
                    faces_3d.push_back(face_3d);

                    neighbors.push_back(numbers::invalid_unsigned_int);
                  }
              }
            j += voro_faces[j] + 1;
          }

        // Collect the vertices of the target face
        std::vector<unsigned int> vertex_indices(voro_faces[target_face]);
        for (int k = 0; k < voro_faces[target_face]; ++k)
          {
            vertex_indices[k] = voro_faces[target_face + k + 1];
            const unsigned int l = 3 * vertex_indices[k];
            AssertIndexRange(l + 2, voro_vertices.size());

            vertices.emplace_back(voro_vertices[l], voro_vertices[l + 1]);
          }

        // Collect the edges of the target face
        for (unsigned int f = 0; f < faces_3d.size(); ++f)
          {
            std::vector<unsigned int> face;
            for (unsigned int v = 0; v < faces_3d[f].size(); ++v)
              if (std::find(vertex_indices.begin(), vertex_indices.end(), faces_3d[f][v])
                  != vertex_indices.end())
                face.push_back(faces_3d[f][v]);
            AssertDimension(face.size(), 2);
            faces.push_back(face);
          }
      }



      template <>
      VoronoiCell<3>::
      VoronoiCell(const types::particle_index  particle_index,
                  const std::vector<double>   &voro_vertices,
                  const std::vector<int>      &voro_faces,
                  const std::vector<int>      &voro_neighbors,
                  const BoundingBox<3>        &bounding_box)
        : particle_index(particle_index)
      {
        vertices.clear();
        faces.clear();
        neighbors.clear();

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

                AssertThrow(face_is_at_boundary(v[0], n, bounding_box),
                            ExcMessage("One of the particle domains crosses the boundary of "
                                       "the bounding box of its surrounding cells. In this case, "
                                       "it is impossible to get all the vertices of the particle "
                                       "domain. Please increase the lower limit of particles per cell."));

                neighbors.push_back(numbers::invalid_unsigned_int);
              }
            else
              neighbors.push_back(static_cast<types::particle_index>(voro_neighbors[i]));

            faces.push_back(face);
            j += voro_faces[j] + 1;
          }
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
       * @param[in] vertices The vertices of the polygon.
       */
      std::pair<double, Point<2>>
      area_and_centroid_2d(const std::vector<Point<2>> &vertices)
      {
        const unsigned int n_vertices = vertices.size();
        AssertThrow(n_vertices >= 3, ExcInternalError());

        double area = 0.0;
        Point<2> centroid;

        // Compute the reference point
        Point<2> ref;
        for (const auto &vertex : vertices)
          ref += vertex;
        ref /= n_vertices;

        for (unsigned int i = 0, j = n_vertices - 1; i < n_vertices; j = i++)
          {
            const double ax = vertices[i][0] - ref[0], 
                         ay = vertices[i][1] - ref[1],
                         bx = vertices[j][0] - ref[0], 
                         by = vertices[j][1] - ref[1];

            const double cross_product = bx * ay - ax * by;
            area     += cross_product;
            centroid += (ref + vertices[i] + vertices[j]) * cross_product;
          }

        centroid *= 1.0 / (3.0 * area);
        area     *= 0.5;

        return std::make_pair(area, centroid);
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
      area_and_centroid_3d(const std::vector<Point<3>>     &points,
                           const std::vector<unsigned int> &vertices)
      {
        const unsigned int n_vertices = vertices.size();
        Assert(n_vertices >= 3 && n_vertices <= points.size(),
               ExcInternalError());

        double A = 0.0;
        Point<3> C;

        // Compute the reference point
        Point<3> ref;
        for (const unsigned int vertex : vertices)
          ref += points[vertex];
        ref /= n_vertices;

        for (unsigned int i = 0, j = n_vertices - 1; i < n_vertices; j = i++)
          {
            const Point<3> &vi = points[vertices[i]];
            const Point<3> &vj = points[vertices[j]];

            const Tensor<1, 3> a = vi - ref;
            const Tensor<1, 3> b = vj - ref;

            const double cross_product = cross_product_3d(a, b).norm();
            A += cross_product;
            C += (ref + vi + vj) * cross_product;
          }

        C *= 1.0 / (3.0 * A);
        A *= 0.5;

        return std::make_pair(A, C);
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

        Point<3> C;
        double V = 0.0;

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

                const double cross_product = a * cross_product_3d(b, c);
                V += cross_product;
                C += (ref + v0 + v1 + v2) * cross_product;
              }
          }

        C /= 4.0 * V;
        V /= 6.0;

        return std::make_pair(V, C);
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



      template <int dim>
      std::vector<typename ParticleHandler<dim>::particle_iterator>
      collect_neighbor_particles(const VoronoiCell<dim>                                            &voronoi_cell,
                                 const std::set<typename Triangulation<dim>::active_cell_iterator> &neighbor_cells,
                                 const ParticleHandler<dim>                                        &particle_handler)
      {
        // Map particle indices to particle iterators
        std::map<types::particle_index, typename ParticleHandler<dim>::particle_iterator> index_to_particles;
        for (const auto &cell : neighbor_cells)
          {
            const auto &particles_in_cell = particle_handler.particles_in_cell(cell);
            for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
              {
                const types::particle_index particle_index = particle->get_local_index();
                if (index_to_particles.find(particle_index) == index_to_particles.end())
                  index_to_particles.insert(std::make_pair(particle_index, particle));
              }
          }

        std::vector<typename ParticleHandler<dim>::particle_iterator> neighbor_particles;
        for (unsigned int f = 0; f < voronoi_cell.faces.size(); ++f)
          {
            if (voronoi_cell.neighbors[f] != numbers::invalid_unsigned_int)
              {
                const auto mit = index_to_particles.find(voronoi_cell.neighbors[f]);
                AssertThrow(mit != index_to_particles.end(), ExcInternalError());
                neighbor_particles.push_back(mit->second);
              }
            else
              neighbor_particles.push_back(particle_handler.end());
          }

        return neighbor_particles;
      }



      double
      compute_voronoi_cell(const VoronoiCell<2>                                   &voronoi_cell,
                           const std::set<Triangulation<2>::active_cell_iterator> &cells,
                           const FiniteElement<2>                                 &fe,
                           const Mapping<2>                                       &mapping,
                           const ParticleHandler<2>                               &particle_handler,
                           const bool                                              compute_face_data,
                           const bool                                              compute_cpdi_data,
                           ParticleDomain::FaceData<2>                            &face_data,
                           ParticleDomain::CPDIData<2>                            &cpdi_data)
      {
        // Calculate the area of the Voronoi polygon
        const auto A_and_C = area_and_centroid_2d(voronoi_cell.vertices);
        const double area = std::abs(A_and_C.first);

        if (compute_face_data)
          {
            const std::vector<ParticleHandler<2>::particle_iterator>
            neighbor_particles = collect_neighbor_particles(voronoi_cell, cells, particle_handler);

            std::vector<std::pair<double, ParticleHandler<2>::particle_iterator>> measure_and_neighbor;
            for (unsigned int f = 0; f < voronoi_cell.faces.size(); ++f)
              {
                const double length = ( voronoi_cell.vertices[voronoi_cell.faces[f][0]] -
                                        voronoi_cell.vertices[voronoi_cell.faces[f][1]] ).norm();
                measure_and_neighbor.emplace_back(length, neighbor_particles[f]);
              }

            face_data.push_back(voronoi_cell.particle_index, measure_and_neighbor, particle_handler);
          }

        if (compute_cpdi_data)
          {
            // Tolerance parameter for function is_inside_unit_cell()
            constexpr double eps = 1.e-12;
            constexpr unsigned int dofs_per_cell = 4;

            const std::vector<Point<2>> &voro_vertices = voronoi_cell.vertices;
            const unsigned int n_voro_vertices = voro_vertices.size();

            const Point<2> center = A_and_C.second;

            // Arrays storing the values of shape functions at the Voronoi vertices
            std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
            std::array<double, dofs_per_cell> N_c;

            // Map storing the weighting function values and gradients for each
            // relevant vertex
            std::map<unsigned int, std::pair<double, Tensor<1, 2>>> weighting_function_data;

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
                    simplex_integrators.emplace_back(vertices);
                  }
              }
            else
              {
                std::array<Point<2>, 3> vertices;
                for (unsigned int v = 0; v < 3; ++v)
                  vertices[v] = voro_vertices[v];
                simplex_integrators.emplace_back(vertices);
              }

            for (const auto &cell : cells)
              {
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
                                const std::pair<double, Tensor<1, 2>> value_and_gradient 
                                  = simplex_integrators[v].integrate_linear_function(N);
                                const unsigned int vertex_index = cell->vertex_index(i);
                                const auto data_it = weighting_function_data.find(vertex_index);
                                if (data_it == weighting_function_data.end())
                                  weighting_function_data.insert(std::make_pair(vertex_index, value_and_gradient));
                                else
                                  {
                                    data_it->second.first  += value_and_gradient.first;
                                    data_it->second.second += value_and_gradient.second;
                                  }
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
                            const std::pair<double, Tensor<1, 2>> value_and_gradient 
                              = simplex_integrators[0].integrate_linear_function(N);
                            const unsigned int vertex_index = cell->vertex_index(i);
                            const auto data_it = weighting_function_data.find(vertex_index);
                            if (data_it == weighting_function_data.end())
                              weighting_function_data.insert(std::make_pair(vertex_index, value_and_gradient));
                            else
                            {
                              data_it->second.first  += value_and_gradient.first;
                              data_it->second.second += value_and_gradient.second;
                            }
                          }
                      }
                  }
              }

            // Normalize the weighting function data by the area of the Voronoi cell
            for (auto &vertex_data : weighting_function_data)
              {
                vertex_data.second.first  /= area;
                vertex_data.second.second /= area;
              }

            cpdi_data.push_back(voronoi_cell.particle_index, weighting_function_data);
          }

        return area;
      }



      double
      compute_voronoi_cell(const VoronoiCell<3>                                   &voronoi_cell,
                           const std::set<Triangulation<3>::active_cell_iterator> &cells,
                           const FiniteElement<3>                                 &fe,
                           const Mapping<3>                                       &mapping,
                           const ParticleHandler<3>                               &particle_handler,
                           const bool                                              compute_face_data,
                           const bool                                              compute_cpdi_data,
                           ParticleDomain::FaceData<3>                            &face_data,
                           ParticleDomain::CPDIData<3>                            &cpdi_data)
      {
        // Calculate the volume of the Voronoi polyhedron
        const auto V_and_C = volume_and_centroid(voronoi_cell.vertices, voronoi_cell.faces);
        const double volume = std::abs(V_and_C.first);

        // Calculate the face areas and centroids beforehand, if either face data
        // or CPDI data is requested
        std::vector<std::pair<double, Point<3>>> face_areas_and_centroids;
        if (compute_face_data || compute_cpdi_data)
          for (unsigned int f = 0; f < voronoi_cell.faces.size(); ++f)
            face_areas_and_centroids.push_back(area_and_centroid_3d(voronoi_cell.vertices,
                                                                    voronoi_cell.faces[f]));

        if (compute_face_data)
          {
            const std::vector<ParticleHandler<3>::particle_iterator>
            neighbor_particles = collect_neighbor_particles(voronoi_cell, cells, particle_handler);

            std::vector<std::pair<double, ParticleHandler<3>::particle_iterator>> measure_and_neighbor;
            for (unsigned int f = 0; f < voronoi_cell.faces.size(); ++f)
              {
                const double area = face_areas_and_centroids[f].first;
                measure_and_neighbor.emplace_back(area, neighbor_particles[f]);
              }

            face_data.push_back(voronoi_cell.particle_index, measure_and_neighbor, particle_handler);
          }

        if (compute_cpdi_data)
          {
            // Tolerance parameter for function is_inside_unit_cell()
            constexpr double eps = 1.e-12;
            constexpr unsigned int dofs_per_cell = 8;

            const std::vector<Point<3>> &voro_vertices = voronoi_cell.vertices;
            const std::vector<std::vector<unsigned int>> &voro_faces = voronoi_cell.faces;
            const unsigned int n_voro_vertices = voro_vertices.size();
            const unsigned int n_voro_faces    = voro_faces.size();

            const Point<3> voro_center = V_and_C.second;

            // Arrays storing the values of shape functions at the Voronoi vertices
            std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
            std::vector<std::array<double, dofs_per_cell>> N_fc(n_voro_faces);
            std::array<double, dofs_per_cell> N_vc;

            // Map storing the weighting function values and gradients for each
            // relevant vertex
            std::map<unsigned int, std::pair<double, Tensor<1, 3>>> weighting_function_data;

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
                        face_centers[f] = face_areas_and_centroids[f].second;
                        for (unsigned int fv = 0; fv < n_face_vertices; ++fv)
                          {
                            std::array<Point<3>, 4> vertices;
                            vertices[0] = voro_vertices[voro_faces[f][fv]];
                            vertices[1] = voro_vertices[voro_faces[f][(fv + 1) % n_face_vertices]];
                            vertices[2] = face_centers[f];
                            vertices[3] = voro_center;
                            simplex_integrators.emplace_back(vertices);
                          }
                      }
                    else // n_face_vertices == 3
                      {
                        std::array<Point<3>, 4> vertices;
                        for (unsigned int fv = 0; fv < 3; ++fv)
                          vertices[fv] = voro_vertices[voro_faces[f][fv]];
                        vertices[3] = voro_center;
                        simplex_integrators.emplace_back(vertices);
                      }
                  }
              }
            else // n_voro_vertices == 4
              {
                std::array<Point<3>, 4> vertices;
                for (unsigned int v = 0; v < 4; ++v)
                  vertices[v] = voro_vertices[v];
                simplex_integrators.emplace_back(vertices);
              }

            for (const auto &cell : cells)
              {
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
                                        const std::pair<double, Tensor<1, 3>> value_and_gradient 
                                          = simplex_integrators[integrator].integrate_linear_function(N);
                                        const unsigned int vertex_index = cell->vertex_index(i);
                                        const auto data_it = weighting_function_data.find(vertex_index);
                                        if (data_it == weighting_function_data.end())
                                          weighting_function_data.insert(std::make_pair(vertex_index, value_and_gradient));
                                        else
                                          {
                                            data_it->second.first  += value_and_gradient.first;
                                            data_it->second.second += value_and_gradient.second;
                                          }
                                      }
                                  }
                                ++integrator;
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
                                    const std::pair<double, Tensor<1, 3>> value_and_gradient 
                                      = simplex_integrators[integrator].integrate_linear_function(N);
                                    const unsigned int vertex_index = cell->vertex_index(i);
                                    const auto data_it = weighting_function_data.find(vertex_index);
                                    if (data_it == weighting_function_data.end())
                                      weighting_function_data.insert(std::make_pair(vertex_index, value_and_gradient));
                                    else
                                      {
                                        data_it->second.first  += value_and_gradient.first;
                                        data_it->second.second += value_and_gradient.second;
                                      }
                                  }
                              }
                            ++integrator;
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
                            const std::pair<double, Tensor<1, 3>> value_and_gradient 
                              = simplex_integrators[0].integrate_linear_function(N);
                            const unsigned int vertex_index = cell->vertex_index(i);
                            const auto data_it = weighting_function_data.find(vertex_index);
                            if (data_it == weighting_function_data.end())
                              weighting_function_data.insert(std::make_pair(vertex_index, value_and_gradient));
                            else
                              {
                                data_it->second.first  += value_and_gradient.first;
                                data_it->second.second += value_and_gradient.second;
                              }
                          }
                      }
                  }
              }

            // Normalize the weighting function data by the volume of the Voronoi cell
            for (auto &vertex_data : weighting_function_data)
              {
                vertex_data.second.first  /= volume;
                vertex_data.second.second /= volume;
              }

            cpdi_data.push_back(voronoi_cell.particle_index, weighting_function_data);       
          }

        return volume;
      }
    }



    /*-------------------- class ParticleDomainHandler --------------------*/

    template <int dim>
    ParticleDomainHandler<dim>::ParticleDomainHandler()
      : particle_handler()
      , triangulation()
      , mapping()
      , generate_face_data(false)
      , generate_cpdi_data(false)
    {}



    template <int dim>
    ParticleDomainHandler<dim>::
    ParticleDomainHandler(const Particles::ParticleHandler<dim> &particle_handler,
#if !DEAL_II_VERSION_GTE(9,8,0)
                          const Triangulation<dim>              &triangulation,
                          const Mapping<dim>                    &mapping,
#endif
                          const bool                             generate_face_data,
                          const bool                             generate_cpdi_data)
      : particle_handler(&particle_handler, typeid(*this).name())
#if DEAL_II_VERSION_GTE(9,8,0)
      , triangulation(&particle_handler.get_triangulation(), typeid(*this).name())
      , mapping(&particle_handler.get_mapping(), typeid(*this).name())
#else
      , triangulation(&triangulation, typeid(*this).name())
      , mapping(&mapping, typeid(*this).name())
#endif
      , generate_face_data(generate_face_data)
      , generate_cpdi_data(generate_cpdi_data)
    {}


    template <int dim>
    void
    ParticleDomainHandler<dim>::generate_particle_domains()
    {
#ifdef ASPECT_WITH_VORO
      // Create the vertex-to-cell map
      GridTools::Cache<dim> grid_cache(*triangulation, *mapping);
      const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

      // Get the bounding box of the triangulation
      const BoundingBox<dim> bounding_box = GridTools::compute_bounding_box(*triangulation);
      const double tria_volume = GridTools::volume(*triangulation, *mapping);
      AssertThrow(std::abs(tria_volume - bounding_box.volume()) < tria_volume * 1.e-8,
          ExcMessage("Particle domains can be generated only when the model geometry is "
                     "hyper-rectangle."));

      // The FE space of the background mesh
      FE_Q<dim> fe(1);

      // Reinit the data arrays
      const types::particle_index max_local_particle_index 
        = particle_handler->get_max_local_particle_index();
      volumes.resize(max_local_particle_index);
      face_data.reinit(max_local_particle_index);
      cpdi_data.reinit(max_local_particle_index);

      double local_volume = 0.0;
      for (const auto &cell : triangulation->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // Find the cells neighboring the current cell
            std::set<typename Triangulation<dim>::active_cell_iterator> neighbor_cells;
            for (const unsigned int v : cell->vertex_indices())
              {
                const unsigned int vertex_index = cell->vertex_index(v);
                neighbor_cells.insert(vertex_to_cell_map[vertex_index].begin(),
                                      vertex_to_cell_map[vertex_index].end());
              }

            // Create the voro container, which is the bounding box of the current cell and
            // its neighbors
            Point<dim> corner1 = cell->vertex(0);
            Point<dim> corner2 = cell->vertex(GeometryInfo<dim>::vertices_per_cell - 1);
            unsigned int n_particles = 0;
            for (const auto &neighbor : neighbor_cells)
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
                n_particles += particle_handler->n_particles_in_cell(neighbor);
              }

            const std::array<int, 3> n_blocks 
              = internal::compute_optimal_block_numbers(corner1,
                                                        corner2,
                                                        n_particles);
            const double diameter = corner2.distance(corner1);

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

            // Put all the particles in the current cell and its neighbors into the container
            for (const auto &neighbor : neighbor_cells)
              {
                const auto &particles_in_cell = particle_handler->particles_in_cell(neighbor);
                for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
                  {
                    const Point<dim> &location = particle->get_location();
                    container.put(particle->get_local_index(), location[0], location[1], (dim > 2 ? location[2] : 0.0));
                  }
              }

            // Collect the local indices of the particles in the current cell
            std::set<types::particle_index> particle_indices;
            for (const auto &particle : particle_handler->particles_in_cell(cell))
              particle_indices.insert(particle.get_local_index());

            // Loop over the particles in the current cell and compute each Voronoi cell
            voro::voronoicell_neighbor voro_cell;
            std::vector<double> voro_vertices;
            std::vector<int>    voro_face_vertices;
            std::vector<int>    voro_neighbors;

            voro::c_loop_all loop(container);
            AssertThrow(loop.start(), ExcMessage("An error occurs when calling voro::c_loop_all::start()."));
            
            do
              {
                const types::particle_index particle_index = loop.pid();
                if (particle_indices.find(particle_index) == particle_indices.end())
                  continue;

                AssertThrow(container.compute_cell(voro_cell, loop),
                            ExcMessage("An error occurs when calling voro::container::compute_cell()."));

                double x, y, z;
                loop.pos(x, y, z);
                voro_cell.vertices(x, y, z, voro_vertices);
                voro_cell.face_vertices(voro_face_vertices);
                voro_cell.neighbors(voro_neighbors);

                internal::VoronoiCell<dim> voronoi_cell(particle_index, 
                                                        voro_vertices,
                                                        voro_face_vertices,
                                                        voro_neighbors,
                                                        bounding_box);

                volumes[voronoi_cell.particle_index]
                  = internal::compute_voronoi_cell(voronoi_cell,
                                                   neighbor_cells,
                                                   fe, 
                                                   *mapping,
                                                   *particle_handler,
                                                   generate_face_data,
                                                   generate_cpdi_data,
                                                   face_data, 
                                                   cpdi_data);

#if DEBUG
                local_volume += volumes[voronoi_cell.particle_index];
#endif
              }
            while (loop.inc());
          }

#if DEBUG
#if DEAL_II_VERSION_GTE(9,8,0)
    double total_volume = Utilities::MPI::sum(local_volume, triangulation->get_mpi_communicator());
#else
    double total_volume = Utilities::MPI::sum(local_volume, triangulation->get_communicator());
#endif
    Assert(std::abs(total_volume - tria_volume) / tria_volume < 1.e-6,
           ExcMessage("The total volume of particle domains ("
                      + Utilities::to_string(total_volume)
                      + ") does not match the volume of the triangulation ("
                      + Utilities::to_string(tria_volume)
                      + ")."));
#else /*DEBUG*/
    (void)local_volume;
#endif /*DEBUG*/

#endif /*ASPECT_WITH_VORO*/
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
#define INSTANTIATE(dim) \
    namespace ParticleDomain \
    { \
      template class FaceData<dim>; \
      template class CPDIData<dim>; \
    } \
    template class ParticleDomainHandler<dim>; \
    template class ParticleDomainAccessor<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
