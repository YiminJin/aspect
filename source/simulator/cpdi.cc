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
        /**
         * Constructor.
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
         * @param[in] box The bounding box of the geometry model. It is used to
         *  check if the Voronoi cell crosses the boundary of the surrounding
         *  FE cells, in which case the CPDI algorithm fails.
         */
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



      /**
       * Class for holding information of the generalized basis functions.
       */
      template <int dim>
      class GeneralizedBasisData
      {
        public:
          GeneralizedBasisData(const unsigned int                     dofs_per_cell,
                               const EvaluationFlags::EvaluationFlags integration_flags);

          void resize(const unsigned int n_particles);

          EvaluationFlags::EvaluationFlags get_integration_flags() const;

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

          const unsigned int                       dofs_per_cell;

          const EvaluationFlags::EvaluationFlags   integration_flags;
      };



      template <int dim>
      GeneralizedBasisData<dim>::
      GeneralizedBasisData(const unsigned int                     dofs,
                           const EvaluationFlags::EvaluationFlags flags)
        : values(flags & EvaluationFlags::values ? dofs : 0)
        , gradients(flags & EvaluationFlags::gradients ? dofs : 0)
        , dofs_per_cell(dofs)
        , integration_flags(flags)
      {}



      template <int dim>
      void
      GeneralizedBasisData<dim>::resize(const unsigned int n_particles)
      {
        volumes.resize(n_particles);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            if (integration_flags & EvaluationFlags::values)
              values[i].resize(n_particles);

            if (integration_flags & EvaluationFlags::gradients)
              gradients[i].resize(n_particles);
          }
      }



      template <int dim>
      EvaluationFlags::EvaluationFlags
      GeneralizedBasisData<dim>::get_integration_flags() const
      {
        return integration_flags;
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
        Assert(integration_flags & EvaluationFlags::values,
               ExcInternalError());
        AssertIndexRange(i, dofs_per_cell);
        AssertIndexRange(p, values[i].size());

        return values[i][p];
      }



      template <int dim>
      const Tensor<1, dim> &
      GeneralizedBasisData<dim>::get_gradient(const unsigned int i,
                                              const unsigned int p) const
      {
        Assert(integration_flags & EvaluationFlags::gradients,
               ExcInternalError());
        AssertIndexRange(i, dofs_per_cell);
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
        Assert(integration_flags & EvaluationFlags::values,
               ExcInternalError());
        AssertIndexRange(i, dofs_per_cell);
        AssertIndexRange(p, values[i].size());

        values[i][p] = val;
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_gradient(const unsigned int    i,
                                              const unsigned int    p,
                                              const Tensor<1, dim> &grad)
      {
        Assert(integration_flags & EvaluationFlags::gradients,
               ExcInternalError());
        AssertIndexRange(i, dofs_per_cell);
        AssertIndexRange(p, gradients[i].size());

        gradients[i][p] = grad;
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
           * @param[in] integration_flag Quantities required to be integrated.
           */
          SimplexIntegrator(const std::array<Point<dim>, n_vertices> &vertices,
                            const EvaluationFlags::EvaluationFlags    integration_flags);

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

          EvaluationFlags::EvaluationFlags integration_flags;
      };




      template <int dim>
      SimplexIntegrator<dim>::
      SimplexIntegrator(const std::array<Point<dim>, n_vertices> &verts,
                        const EvaluationFlags::EvaluationFlags    flags)
        : vertices(verts)
        , integration_flags(flags)
      {
        // If the integration of gradients are required, then calculate the
        //normal vectors of faces, and then calculate the volume based on them
        if (integration_flags & EvaluationFlags::gradients)
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
        else
          {
            Tensor<2, dim> T;
            for (unsigned int d = 0; d < dim; ++d)
              T[d] = vertices[d] - vertices[dim];

            signed_volume = determinant(T) * normalization_factor;
          }
      }



      template <int dim>
      std::pair<double, Tensor<1, dim> >
      SimplexIntegrator<dim>::
      integrate_linear_function(const std::array<double, n_vertices> &N) const
      {
        double value            = numbers::signaling_nan<double>();
        Tensor<1, dim> gradient = numbers::signaling_nan<Tensor<1, dim>>();

        if (integration_flags & EvaluationFlags::values)
          {
            value = 0;
            for (unsigned int v = 0; v < n_vertices; ++v)
              value += N[v];

            value *= std::abs(signed_volume) / n_vertices;
          }

        if (integration_flags & EvaluationFlags::gradients)
          {
            gradient = 0;
            for (unsigned int v = 0; v < n_vertices; ++v)
              gradient += N[v] * faces[v];

            gradient *= normalization_factor;
          }

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
        const EvaluationFlags::EvaluationFlags flags = data.get_integration_flags();

        data.resize(n_particles);
        std::vector<std::vector<double>> N_v(dofs_per_cell);
        for (unsigned int p = 0; p < n_particles; ++p)
          {
            double area = 0.0;
            std::vector<double> values(dofs_per_cell, 0.0);
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
                // Divide the voronoi cell into n_vertices triangles. Each triangle
                // consists of two adjacent vertices and the barycenter of the polygon
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

                    // Create a simplex integrator
                    std::array<Point<2>, 3> verts;
                    verts[0] = voronoi_cell.vertices[v1];
                    verts[1] = voronoi_cell.vertices[v2];
                    verts[2] = center;
                    SimplexIntegrator<2> simplex_integrator(verts, flags);
                    area += simplex_integrator.get_volume();

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        std::array<double, 3> N;
                        N[0] = N_v[i][v1];
                        N[1] = N_v[i][v2];
                        N[2] = N_c[i];

                        const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);

                        if (flags & EvaluationFlags::values)
                          values[i] += value_and_gradient.first;
                        if (flags & EvaluationFlags::gradients)
                          gradients[i] += value_and_gradient.second;
                      }
                  }

                data.set_volume(p, area);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (flags & EvaluationFlags::values)
                      data.set_value(i,p, values[i] / area);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(i, p, gradients[i] / area);
                  }
              }
            else
              {
                // No need to divide the Voronoi cell
                std::array<Point<2>, 3> verts;
                for (unsigned int v = 0; v < 3; ++v)
                  verts[v] = voronoi_cell.vertices[v];

                SimplexIntegrator<2> simplex_integrator(verts, flags);

                const double area = simplex_integrator.get_volume();
                data.set_volume(p, area);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    std::array<double, 3> N;
                    for (unsigned int v = 0; v < 3; ++v)
                      N[v] = N_v[i][v];
                    const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                    if (flags & EvaluationFlags::values)
                      data.set_value(i, p, value_and_gradient.first / area);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(i, p, value_and_gradient.second / area);
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
        const EvaluationFlags::EvaluationFlags flags = data.get_integration_flags();

        data.resize(n_particles);
        std::vector<std::vector<double>> N_v(dofs_per_cell);
        for (unsigned int p = 0; p < n_particles; ++p)
          {
            double volume = 0.0;
            std::vector<double> values(dofs_per_cell, 0.0);
            std::vector<Tensor<1, 3>> gradients(dofs_per_cell);

            const VoronoiCell<3> &voronoi_cell = voronoi_cells[p];
            const unsigned int n_vertices = voronoi_cell.vertices.size();
            const unsigned int n_faces    = voronoi_cell.faces.size();

            // Evaluate the values of the shape functions at the vertices
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                N_v[i].resize(n_vertices);
                std::fill(N_v[i].begin(), N_v[i].end(), 0.0);
              }

            for (unsigned int v = 0; v < n_vertices; ++v)
              {
                const Point<3> vertex_unit = mapping.transform_real_to_unit_cell(cell, voronoi_cell.vertices[v]);
                if (GeometryInfo<3>::is_inside_unit_cell(vertex_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[i][v] = fe.shape_value(i, vertex_unit);
              }

            if (n_vertices > 4)
              {
                // Divide the voronoi cell into n_faces prisms. The apex of the prisms
                // is the barycenter of the polyhedron
                Point<3> volume_center;
                for (unsigned int v = 0; v < n_vertices; ++v)
                  volume_center += voronoi_cell.vertices[v];
                volume_center /= n_vertices;

                std::vector<double> N_c(dofs_per_cell);
                const Point<3> vc_unit = mapping.transform_real_to_unit_cell(cell, volume_center);
                if (GeometryInfo<3>::is_inside_unit_cell(vc_unit))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_c[i] = fe.shape_value(i, vc_unit);

                for (unsigned int f = 0; f < n_faces; ++f)
                  {
                    const unsigned int n_face_vertices = voronoi_cell.faces[f].size();
                    if (n_face_vertices > 3)
                      {
                        // Divide the face into n_face_vertices triangles. Each triangle
                        // consists of two adjacent vertices and the barycenter of the polygon
                        Point<3> face_center;
                        for (unsigned int v = 0; v < n_face_vertices; ++v)
                          face_center += voronoi_cell.vertices[voronoi_cell.faces[f][v]];
                        face_center /= n_face_vertices;

                        std::vector<double> N_f(dofs_per_cell);
                        const Point<3> fc_unit = mapping.transform_real_to_unit_cell(cell, face_center);
                        if (GeometryInfo<3>::is_inside_unit_cell(fc_unit))
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            N_f[i] = fe.shape_value(i, fc_unit);

                        for (unsigned int v = 0; v < n_face_vertices; ++v)
                          {
                            const unsigned int v1 = v;
                            const unsigned int v2 = (v + 1) % n_face_vertices;

                            // Create a simplex integrator
                            std::array<Point<3>, 4> verts;
                            verts[0] = voronoi_cell.vertices[voronoi_cell.faces[f][v1]];
                            verts[1] = voronoi_cell.vertices[voronoi_cell.faces[f][v2]];
                            verts[2] = face_center;
                            verts[3] = volume_center;
                            SimplexIntegrator<3> simplex_integrator(verts, flags);
                            volume += simplex_integrator.get_volume();

                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                std::array<double, 4> N;
                                N[0] = N_v[i][voronoi_cell.faces[f][v1]];
                                N[1] = N_v[i][voronoi_cell.faces[f][v2]];
                                N[2] = N_f[i];
                                N[3] = N_c[i];

                                const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);

                                if (flags & EvaluationFlags::values)
                                  values[i] += value_and_gradient.first;
                                if (flags & EvaluationFlags::gradients)
                                  gradients[i] += value_and_gradient.second;
                              }
                          }
                      }
                    else
                      {
                        // No need to divide the face
                        std::array<Point<3>, 4> verts;
                        for (unsigned int fv = 0; fv < 3; ++fv)
                          verts[fv] = voronoi_cell.vertices[voronoi_cell.faces[f][fv]];
                        verts[3] = volume_center;

                        SimplexIntegrator<3> simplex_integrator(verts, flags);
                        volume += simplex_integrator.get_volume();

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            std::array<double, 4> N;
                            for (unsigned int fv = 0; fv < 3; ++fv)
                              N[fv] = N_v[i][voronoi_cell.faces[f][fv]];
                            N[3] = N_c[i];
                            const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                            if (flags & EvaluationFlags::values)
                              values[i] += value_and_gradient.first;
                            if (flags & EvaluationFlags::gradients)
                              gradients[i] += value_and_gradient.second;
                          }
                      }
                  }

                data.set_volume(p, volume);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (flags & EvaluationFlags::values)
                      data.set_value(i, p, values[i] / volume);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(i, p, gradients[i] / volume);
                  }
              }
            else
              {
                // No need to divide the Voronoi cell
                std::array<Point<3>, 4> verts;
                for (unsigned int v = 0; v < 4; ++v)
                  verts[v] = voronoi_cell.vertices[v];

                SimplexIntegrator<3> simplex_integrator(verts, flags);

                const double volume = simplex_integrator.get_volume();
                data.set_volume(p, volume);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    std::array<double, 4> N;
                    for (unsigned int v = 0; v < 4; ++v)
                      N[v] = N_v[i][v];
                    const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                    if (flags & EvaluationFlags::values)
                      data.set_value(i, p, value_and_gradient.first / volume);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(i, p, value_and_gradient.second / volume);
                  }
              }
          }
      }



      /**
       * Calculate the volume and the values/gradients of the generalized basis
       * functions for each particle domain.
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

        // Do the actual evaluation
        do_evaluation(cell, voronoi_cells, fe, mapping, data);

#if DEBUG
        // Check if the volumes of the Voronoi cells are correct.
        const auto particles_in_cell = particle_handler.particles_in_cell(cell);
        for (auto particle = particles_in_cell.begin();
             particle != particles_in_cell.end(); ++particle)
          {
            const unsigned int p = std::distance(particles_in_cell.begin(), particle);
            const double volume = data.get_volume(p) * (dim > 2 ? 1. : diameter * 0.02);
            const double voro_volume = voro_cell.volume();

            if (std::abs(volume - voro_volume) > volume * 1.e-6)
              {
                std::stringstream error_message;
                error_message << "The volume of the Voronoi cell at ("
                              << particle->get_location()
                              << ") should be "
                              << voro_volume
                              << ", but the value we get is "
                              << volume;

                Assert(false, ExcMessage(error_message.str()));
              }
          }
#endif /*DEBUG*/
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
    const BoundingBox<dim> bounding_box(std::make_pair(box.get_origin(), box.get_origin() + box.get_extents()));

    const Particles::ParticleHandler<dim> &particle_handler = cpdi_particle_manager->get_particle_handler();
    const FiniteElement<dim> &field_fe = finite_element.base_element(advection_fields[0].base_element(introspection));

    const unsigned int field_dofs_per_cell = field_fe.dofs_per_cell;
    const unsigned int n_fields            = advection_fields.size();

    // stuff for assembling the CPDI system
    FullMatrix<double> cell_matrix(field_dofs_per_cell, field_dofs_per_cell);
    std::vector<Vector<double>> cell_rhs(n_fields, Vector<double>(field_dofs_per_cell));

    std::vector<types::global_dof_index> cell_dof_indices(finite_element.dofs_per_cell);
    std::vector<std::vector<types::global_dof_index>> field_dof_indices(n_fields, std::vector<types::global_dof_index>(field_dofs_per_cell));

    internal::CPDI::GeneralizedBasisData<dim> data(field_dofs_per_cell, EvaluationFlags::values);

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
                                   field_fe,
                                   *mapping,
                                   bounding_box,
                                   data);

          const auto particles_in_cell = particle_handler.particles_in_cell(cell);

          // Assemble the cell matrix and cell vectors
          cell_matrix = 0;
          for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
            cell_rhs[field_index] = 0;

          for (auto particle = particles_in_cell.begin(); particle != particles_in_cell.end(); ++particle)
            {
              const unsigned int p = std::distance(particles_in_cell.begin(), particle);
              const double Vp = data.get_volume(p);

              const ArrayView<const double> particle_properties = particle->get_properties();

              for (unsigned int i = 0; i < field_dofs_per_cell; ++i)
                {
                  for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                    {
                      const double property_value = particle_properties[field_to_property_map[field_index]];
                      cell_rhs[field_index](i) += Vp * data.get_value(i, p) * property_value;
                    }

                  for (unsigned int j = 0; j < field_dofs_per_cell; ++j)
                    cell_matrix(i, j) += Vp * data.get_value(i, p) * data.get_value(j, p);
                }
            }

          // Collect the dof indices for each field
          cell->get_dof_indices(cell_dof_indices);
          for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
            {
              const unsigned int field_component = advection_fields[field_index].component_index(introspection);
              for (unsigned int i = 0, i_field = 0; i_field < field_dofs_per_cell; /*increment at end of loop*/)
                {
                  if (finite_element.system_to_component_index(i).first == field_component)
                    {
                      field_dof_indices[field_index][i_field] = cell_dof_indices[i];
                      ++i_field;
                    }
                  ++i;
                }
            }

          // Copy cell matrix and vectors to the corresponding entries of system matrix and vectors
          current_constraints.distribute_local_to_global(cell_matrix,
                                                         field_dof_indices[0],
                                                         system_matrix);

          for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
            current_constraints.distribute_local_to_global(cell_rhs[field_index],
                                                           field_dof_indices[field_index],
                                                           system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    // Solve for each field
    SolverControl solver_control(1000, 1.e-8);
    SolverCG<LinearAlgebra::Vector> solver(solver_control);

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
        pcout << "   Solving CPDI system for " << field.name(introspection)
              << "... " << std::flush;

        const unsigned int block_idx = field.block_index(introspection);
        distributed_solution.block(block_idx) = current_linearization_point.block(block_idx);
        current_constraints.set_zero(distributed_solution);

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

        pcout << solver_control.last_step() << " iterations." << std::endl;
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
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  template void Simulator<dim>::perform_convected_particle_domain_interpolation(const std::vector<AdvectionField> &);

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
