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
         * @param[in] box The bounding box of the geometry model. It is used to
         *  check if the Voronoi cell crosses the boundary of the surrounding
         *  FE cells, in which case the CPDI algorithm fails.
         */
        VoronoiCell(const types::particle_index  particle_index,
                    const std::vector<double>   &voro_vertices,
                    const std::vector<int>      &voro_faces,
                    const std::vector<int>      &voro_neighbors,
                    const BoundingBox<dim>      &box);

        VoronoiCell(VoronoiCell &&) noexcept = default;

        VoronoiCell &operator=(VoronoiCell &&) = default;

        types::particle_index particle_index;

        std::vector<Point<dim>> vertices;

        std::vector<std::vector<unsigned int>> faces;

        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> neighbors;
      };



      template <>
      VoronoiCell<2>::
      VoronoiCell(const types::particle_index  particle_index,
                  const std::vector<double>   &voro_vertices,
                  const std::vector<int>      &voro_faces,
                  const std::vector<int>      &voro_neighbors,
                  const BoundingBox<2>        &box)
        : particle_index(particle_index)
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
      VoronoiCell(const types::particle_index  particle_index,
                  const std::vector<double>   &voro_vertices,
                  const std::vector<int>      &voro_faces,
                  const std::vector<int>      &voro_neighbors,
                  const BoundingBox<3>        &box)
        : particle_index(particle_index)
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



      void write_vtk(const std::vector<VoronoiCell<2>> &voronoi_cells,
                     std::ofstream        &out)
      {
        unsigned int total_points = 0;
        unsigned int total_ints_in_polygons = 0;

        for (const auto &poly : voronoi_cells)
          {
            total_points += poly.vertices.size();
            total_ints_in_polygons += poly.vertices.size() + 1;
          }

        out << "# vtk DataFile Version 3.0\n";
        out << "Voronoi polygons\n";
        out << "ASCII\n";
        out << "DATASET POLYDATA\n";

        out << "POINTS " << total_points <<" float\n";
        for (const auto &poly : voronoi_cells)
          for (const auto &p : poly.vertices)
            out << p[0] << ' ' << p[1] << " 0.0\n";

        out << "POLYGONS " << voronoi_cells.size() << " " << total_ints_in_polygons << "\n";
        int index = 0;
        for (const auto &poly : voronoi_cells)
          {
            out << poly.vertices.size();
            for (unsigned int i = 0; i < poly.vertices.size(); ++i)
              out << ' ' << index++;
            out << "\n";
          }
      }



      /**
       * Class for holding information of the generalized basis functions.
       */
      template <int dim>
      class GeneralizedBasisData
      {
        public:
          static constexpr unsigned int n_vertices = GeometryInfo<dim>::vertices_per_cell;

          GeneralizedBasisData(const EvaluationFlags::EvaluationFlags integration_flags)
            : integration_flags(integration_flags)
          {}

          void resize(const unsigned int n_cells);

          EvaluationFlags::EvaluationFlags get_integration_flags() const;

          double get_volume() const;

          double get_value(const unsigned int cell,
                           const unsigned int vertex) const;

          const Tensor<1, dim> &
          get_gradient(const unsigned int cell,
                       const unsigned int vertex) const;

          void set_volume(const double       vol);

          void set_value(const unsigned int cell,
                         const unsigned int vertex,
                         const double       value);

          void set_gradient(const unsigned int    cell,
                            const unsigned int    vertex,
                            const Tensor<1, dim> &gradient);

        private:
          std::vector<std::array<double, n_vertices>>         values;

          std::vector<std::array<Tensor<1, dim>, n_vertices>> gradients;

          double                                              volume;

          EvaluationFlags::EvaluationFlags                    integration_flags;
      };



      template <int dim>
      void
      GeneralizedBasisData<dim>::resize(const unsigned int n_cells)
      {
        if (integration_flags & EvaluationFlags::values)
          {
            values.resize(n_cells);
            for (unsigned int c = 0; c < n_cells; ++c)
              std::fill(values[c].begin(), values[c].end(), 0.0);
          }

        if (integration_flags & EvaluationFlags::gradients)
          {
            gradients.resize(n_cells);
            for (unsigned int c = 0; c < n_cells; ++c)
              std::fill(gradients[c].begin(), gradients[c].end(), Tensor<1,dim>());
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
      GeneralizedBasisData<dim>::get_volume() const
      {
        return volume;
      }



      template <int dim>
      double
      GeneralizedBasisData<dim>::get_value(const unsigned int c,
                                           const unsigned int v) const
      {
        Assert(integration_flags & EvaluationFlags::values,
               ExcInternalError());
        AssertIndexRange(c, values.size());
        AssertIndexRange(v, n_vertices);

        return values[c][v];
      }



      template <int dim>
      const Tensor<1, dim> &
      GeneralizedBasisData<dim>::get_gradient(const unsigned int c,
                                              const unsigned int v) const
      {
        Assert(integration_flags & EvaluationFlags::gradients,
               ExcInternalError());
        AssertIndexRange(c, gradients.size());
        AssertIndexRange(v, n_vertices);

        return gradients[c][v];
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_volume(const double       vol)
      {
        volume = vol;
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_value(const unsigned int c,
                                           const unsigned int v,
                                           const double       val)
      {
        Assert(integration_flags & EvaluationFlags::values,
               ExcInternalError());
        AssertIndexRange(c, values.size());
        AssertIndexRange(v, n_vertices);

        values[c][v] = val;
      }



      template <int dim>
      void
      GeneralizedBasisData<dim>::set_gradient(const unsigned int    c,
                                              const unsigned int    v,
                                              const Tensor<1, dim> &grad)
      {
        Assert(integration_flags & EvaluationFlags::gradients,
               ExcInternalError());
        AssertIndexRange(c, gradients.size());
        AssertIndexRange(v, n_vertices);

        gradients[c][v] = grad;
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



      std::pair<double, Point<2> >
      area_and_barycenter(const std::vector<Point<2>> &vertices)
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



      std::pair<double, Point<3> >
      area_and_barycenter(const std::vector<Point<3>>     &points,
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



      std::pair<double, Point<3> >
      volume_and_barycenter(const std::vector<Point<3>>                  &vertices,
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



      unsigned int
      dof_index_in_cell(const Triangulation<2>::active_cell_iterator &cell,
                        const unsigned int i)
      {
        const unsigned int cell_index = cell->active_cell_index();
        switch (cell_index)
        {
          case 0:
            if (i == 3)
              return 0;
            break;
            
          case 1:
            if (i == 2)
              return 0;
            if (i == 3)
              return 1;
            break;

          case 2:
            if (i == 2)
              return 1;
            break;

          case 3:
            if (i == 1)
              return 0;
            if (i == 3)
              return 2;
            break;

          case 4:
            return i;

          case 5:
            if (i == 0)
              return 1;
            if (i == 2)
              return 3;
            break;

          case 6:
            if (i == 1)
              return 2;
            break;

          case 7:
            if (i == 0)
              return 2;
            if (i == 1)
              return 3;
            break;

          case 8:
            if (i == 0)
              return 3;
            break;

          default:
            Assert(false, ExcInternalError());
        }

        return numbers::invalid_unsigned_int;
      }



      std::vector<std::array<Point<3>, 3>>
      do_evaluation(const std::vector<Triangulation<2>::active_cell_iterator> &cells,
                    const VoronoiCell<2>                                      &voronoi_cell,
                    const FiniteElement<2>                                    &fe,
                    const Mapping<2>                                          &mapping,
                    GeneralizedBasisData<2>                                   &data)
      {
        Assert((dynamic_cast<const FE_Q<2>*>(&fe) ||
                dynamic_cast<const FE_DGQ<2>*>(&fe))
               && fe.degree == 1,
               ExcInternalError());

        // Tolerance parameter for function GeometryInfo::is_inside_unit_cell()
        constexpr double eps = 1.e-12;
        constexpr unsigned int dofs_per_cell = 4;

        const unsigned int n_cells = cells.size();
        const EvaluationFlags::EvaluationFlags flags = data.get_integration_flags();

        const std::vector<Point<2>> &voro_vertices = voronoi_cell.vertices;
        const unsigned int n_voro_vertices = voro_vertices.size();

        const auto A_and_C = area_and_barycenter(voro_vertices);
        const double area = std::abs(A_and_C.first);
        const Point<2> center = A_and_C.second;

        data.resize(n_cells);
        data.set_volume(area);

        // Arrays storing the values of shape functions at the Voronoi vertices
        std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
        std::array<double, dofs_per_cell> N_c;

        // Arrays storing the values and gradients of the generalized basis functions
        // for each cell
        std::vector<double>       values(dofs_per_cell);
        std::vector<Tensor<1, 2>> gradients(dofs_per_cell);

        std::vector<std::array<Point<3>, 3>> basis_functions;

        // Create a simplex integrator for each triangle
        std::vector<SimplexIntegrator<2>> simplex_integrators;
        if (n_voro_vertices > 3)
          {
            // If the Voronoi cell has more than 3 vertices, then we need to
            // divide it into n_voro_vertices triangles, the apex of which
            // is the barycenter of the Voronoi cell
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                std::array<Point<2>, 3> vertices;
                vertices[0] = voro_vertices[v];
                vertices[1] = voro_vertices[(v + 1) % n_voro_vertices];
                vertices[2] = center;
                simplex_integrators.push_back(SimplexIntegrator(vertices, flags));

                basis_functions.push_back({{Point<3>(vertices[0][0], vertices[0][1], 0.0), 
                                            Point<3>(vertices[1][0], vertices[1][1], 0.0),
                                            Point<3>(vertices[2][0], vertices[2][1], 0.0)}});
              }
          }
        else
          {
            std::array<Point<2>, 3> vertices;
            for (unsigned int v = 0; v < 3; ++v)
              vertices[v] = voro_vertices[v];
            simplex_integrators.push_back(SimplexIntegrator(vertices, flags));

            basis_functions.push_back({{Point<3>(vertices[0][0], vertices[0][1], 0.0), 
                                        Point<3>(vertices[1][0], vertices[1][1], 0.0),
                                        Point<3>(vertices[2][0], vertices[2][1], 0.0)}});
          }

        for (unsigned int c = 0; c < n_cells; ++c)
          {
            std::fill(values.begin(), values.end(), 0.0);
            std::fill(gradients.begin(), gradients.end(), Tensor<1,2>());

            const auto &cell = cells[c];
            std::cout << "   cell " << cell->active_cell_index() << std::endl;

            // Evaluate the values of shape functions at the vertices
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                const Point<2> vertex_unit = mapping.transform_real_to_unit_cell(cell, voro_vertices[v]);
                if (GeometryInfo<2>::is_inside_unit_cell(vertex_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[v][i] = fe.shape_value(i, vertex_unit);
                else
                  std::fill(N_v[v].begin(), N_v[v].end(), 0.0);
              }

            if (n_voro_vertices > 3)
              {
                // Evaluate the values of shape functions at the barycenter
                const Point<2> center_unit = mapping.transform_real_to_unit_cell(cell, center);
                if (GeometryInfo<2>::is_inside_unit_cell(center_unit, eps))
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

                        if (dof_index_in_cell(cell, i) == 0)
                          std::cout << "      triangle {(" 
                                    << voro_vertices[v] << ") ("
                                    << voro_vertices[(v + 1) % n_voro_vertices] << ") ("
                                    << center << ")}: "
                                    << N[0] << ", " << N[1] << ", " << N[2]
                                    << " (i = " << i << ")"
                                    << std::endl;

                        // Integrate the values and/or gradients of the generalized basis function
                        // if the basis function is nonzero in this triangle
                        if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0)
                          {
                            const auto value_and_gradient = simplex_integrators[v].integrate_linear_function(N);
                            if (flags & EvaluationFlags::values)
                              values[i] += value_and_gradient.first;
                            if (flags & EvaluationFlags::gradients)
                              gradients[i] += value_and_gradient.second;

                            const unsigned int dof = dof_index_in_cell(cell, i);
                            if (dof != 4)
                              for (unsigned int k = 0; k < 3; ++k)
                                basis_functions[v][k][2] += N[k];
                          }
                      }
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (flags & EvaluationFlags::values)
                      data.set_value(c, i, values[i] / area);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(c, i, gradients[i] / area);
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
                        if (flags & EvaluationFlags::values)
                          data.set_value(c, i, value_and_gradient.first / area);
                        if (flags & EvaluationFlags::gradients)
                          data.set_gradient(c, i, value_and_gradient.second / area);

                        const unsigned int dof = dof_index_in_cell(cell, i);
                        if (dof != 4)
                          for (unsigned int v = 0; v < 3; ++v)
                            basis_functions[0][v][2] += N[v];
                      }
                  }
              }
          }

        return basis_functions;
      }



      std::vector<std::array<Point<3>, 3>>
      do_evaluation(const std::vector<Triangulation<3>::active_cell_iterator> &cells,
                    const VoronoiCell<3>                                      &voronoi_cell,
                    const FiniteElement<3>                                    &fe,
                    const Mapping<3>                                          &mapping,
                    GeneralizedBasisData<3>                                   &data)
      {
        Assert((dynamic_cast<const FE_Q<3>*>(&fe) ||
                dynamic_cast<const FE_DGQ<3>*>(&fe))
               && fe.degree == 1,
               ExcInternalError());

        // Tolerance parameter for function GeometryInfo::is_inside_unit_cell()
        constexpr double eps = 1.e-12;
        constexpr unsigned int dofs_per_cell = 8;

        const unsigned int n_cells = cells.size();
        const EvaluationFlags::EvaluationFlags flags = data.get_integration_flags();

        const std::vector<Point<3>> &voro_vertices = voronoi_cell.vertices;
        const std::vector<std::vector<unsigned int>> &voro_faces = voronoi_cell.faces;
        const unsigned int n_voro_vertices = voro_vertices.size();
        const unsigned int n_voro_faces    = voro_faces.size();

        const auto V_and_C = volume_and_barycenter(voro_vertices, voro_faces);
        const double volume = std::abs(V_and_C.first);
        const Point<3> voro_center = V_and_C.second;

        std::vector<Point<3>> face_centers(n_voro_faces);
        for (unsigned int f = 0; f < n_voro_faces; ++f)
          {
            const unsigned int n_face_vertices = voro_faces[f].size();
            if (n_face_vertices > 3)
              face_centers[f] = area_and_barycenter(voro_vertices, voro_faces[f]).second;
          }

        data.resize(n_cells);
        data.set_volume(volume);

        // Arrays storing the values of shape functions at the Voronoi vertices
        std::vector<std::array<double, dofs_per_cell>> N_v(n_voro_vertices);
        std::vector<std::array<double, dofs_per_cell>> N_fc(n_voro_faces);
        std::array<double, dofs_per_cell> N_vc;

        // Arrays storing the values and gradients of the generalized basis functions
        // for each cell
        std::vector<double>       values(dofs_per_cell);
        std::vector<Tensor<1, 3>> gradients(dofs_per_cell);

        for (unsigned int c = 0; c < n_cells; ++c)
          {
            std::fill(values.begin(), values.end(), 0.0);
            std::fill(gradients.begin(), gradients.end(), Tensor<1,3>());

            const auto &cell = cells[c];

            // Evaluate the values of shape functions at the vertices
            for (unsigned int v = 0; v < n_voro_vertices; ++v)
              {
                const Point<3> vertex_unit = mapping.transform_real_to_unit_cell(cell, voro_vertices[v]);
                if (GeometryInfo<3>::is_inside_unit_cell(vertex_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_v[v][i] = fe.shape_value(i, vertex_unit);
                else
                  std::fill(N_v[v].begin(), N_v[v].end(), 0.0);
              }

            if (n_voro_vertices > 4)
              {
                // If the Voronoi cell has more than 4 faces, then we need to
                // divide it into n_voro_faces tetrahedra, the apex of which
                // is the barycenter of the Voronoi cell
                const Point<3> voro_center_unit = mapping.transform_real_to_unit_cell(cell, voro_center);
                if (GeometryInfo<3>::is_inside_unit_cell(voro_center_unit, eps))
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    N_vc[i] = fe.shape_value(i, voro_center_unit);
                else
                  std::fill(N_vc.begin(), N_vc.end(), 0.0);

                for (unsigned int f = 0; f < n_voro_faces; ++f)
                  {
                    const unsigned int n_face_vertices = voro_faces[f].size();
                    if (n_face_vertices > 3)
                      {
                        // If the face has more than 3 vertices, then we need to
                        // divide it into n_face_vertices triangles, the apex of
                        // which is the barycenter of the face
                        const Point<3> face_center_unit = mapping.transform_real_to_unit_cell(cell, face_centers[f]);
                        if (GeometryInfo<3>::is_inside_unit_cell(face_center_unit, eps))
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            N_fc[f][i] = fe.shape_value(i, face_center_unit);
                        else
                          std::fill(N_fc[f].begin(), N_fc[f].end(), 0.0);

                        for (unsigned int v = 0; v < n_face_vertices; ++v)
                          {
                            const unsigned int v1 = v;
                            const unsigned int v2 = (v + 1) % n_face_vertices;

                            std::array<Point<3>, 4> verts;
                            verts[0] = voronoi_cell.vertices[voronoi_cell.faces[f][v1]];
                            verts[1] = voronoi_cell.vertices[voronoi_cell.faces[f][v2]];
                            verts[2] = face_centers[f];
                            verts[3] = voro_center;
                            SimplexIntegrator<3> simplex_integrator(verts, flags);

                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                std::array<double, 4> N;
                                N[0] = N_v[voro_faces[f][v1]][i];
                                N[1] = N_v[voro_faces[f][v2]][i];
                                N[2] = N_fc[f][i];
                                N[3] = N_vc[i];

                                // Integrate the values and/or gradients of the generalized basis function
                                // if the basis function is nonzero in this triangle
                                if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0 || N[3] != 0.0)
                                  {
                                    const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                                    if (flags & EvaluationFlags::values)
                                      values[i] += value_and_gradient.first;
                                    if (flags & EvaluationFlags::gradients)
                                      gradients[i] += value_and_gradient.second;
                                  }
                              }
                          }
                      }
                    else
                      {
                        // No need to divide the face
                        std::array<Point<3>, 4> verts;
                        for (unsigned int fv = 0; fv < 3; ++fv)
                          verts[fv] = voro_vertices[voro_faces[f][fv]];
                        verts[3] = voro_center;

                        SimplexIntegrator<3> simplex_integrator(verts, flags);
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
                                const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                                if (flags & EvaluationFlags::values)
                                  values[i] += value_and_gradient.first;
                                if (flags & EvaluationFlags::gradients)
                                  gradients[i] += value_and_gradient.second;
                              }
                          }
                      }
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (flags & EvaluationFlags::values)
                      data.set_value(c, i, values[i] / volume);
                    if (flags & EvaluationFlags::gradients)
                      data.set_gradient(c, i, gradients[i] / volume);
                  }
              }
            else
              {
                // No need to divide the Voronoi cell
                std::array<Point<3>, 4> verts;
                for (unsigned int v = 0; v < 4; ++v)
                  verts[v] = voro_vertices[v];

                SimplexIntegrator<3> simplex_integrator(verts, flags);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    std::array<double, 4> N;
                    for (unsigned int v = 0; v < 4; ++v)
                      N[v] = N_v[v][i];

                    // Integrate the values and/or gradients of the generalized basis function
                    // if the basis function is nonzero in this triangle
                    if (N[0] != 0.0 || N[1] != 0.0 || N[2] != 0.0 || N[3] != 0.0)
                      {
                        const auto value_and_gradient = simplex_integrator.integrate_linear_function(N);
                        if (flags & EvaluationFlags::values)
                          data.set_value(c, i, value_and_gradient.first / volume);
                        if (flags & EvaluationFlags::gradients)
                          data.set_gradient(c, i, value_and_gradient.second / volume);
                      }
                  }
              }
          }

        return std::vector<std::array<Point<3>, 3>>();
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
      std::vector<std::array<Point<3>, 3>>
      evaluate(const typename Triangulation<dim>::active_cell_iterator              &cell,
               const std::vector<typename Triangulation<dim>::active_cell_iterator> &neighbors,
               const Particles::ParticleHandler<dim>                                &particle_handler,
               const FiniteElement<dim>                                             &fe,
               const Mapping<dim>                                                   &mapping,
               const BoundingBox<dim>                                               &box,
               std::vector<GeneralizedBasisData<dim>>                               &data)
      {
        std::cout << "Main cell: " << cell->active_cell_index() << std::endl;
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
        std::set<types::particle_index> particle_indices;
        for (const auto &particle : particle_handler.particles_in_cell(cell))
          particle_indices.insert(particle.get_local_index());

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
            if (particle_indices.find(particle_index) == particle_indices.end())
              continue;

            AssertThrow(container.compute_cell(voro_cell, loop),
                        ExcMessage("An error occurs when calling voro::container::compute_cell()."));

            double x, y, z;
            loop.pos(x, y, z);
            voro_cell.vertices(x, y, z, voro_vertices);
            voro_cell.face_vertices(voro_face_vertices);
            voro_cell.neighbors(voro_neighbors);

            voronoi_cells.emplace_back(VoronoiCell<dim>(particle_index,
                                                        voro_vertices,
                                                        voro_face_vertices,
                                                        voro_neighbors,
                                                        box));

          }
        while (loop.inc());

        // Now the sequence of Voronoi cells are determined by voro::container.
        // We need to make the sequence of Voronoi cells match that of particles.
        unsigned int local_index = 0;
        for (const auto &particle : particle_handler.particles_in_cell(cell))
          {
            const types::particle_index particle_index = particle.get_local_index();
            for (unsigned int i = local_index; i < voronoi_cells.size(); ++i)
              if (voronoi_cells[i].particle_index == particle_index)
                {
                  std::swap(voronoi_cells[local_index], voronoi_cells[i]);
                  ++local_index;
                  continue;
                }
          }
        AssertDimension(local_index, voronoi_cells.size());

        // Do the actual evaluation
        data.resize(voronoi_cells.size(), EvaluationFlags::values);
        std::vector<std::array<Point<3>, 3>> basis_functions;
        for (unsigned int i = 0; i < voronoi_cells.size(); ++i)
        {
          std::cout << "***Particle " << i << std::endl;
          std::vector<std::array<Point<3>, 3>> particle_basis_functions =
          do_evaluation(neighbors, voronoi_cells[i], fe, mapping, data[i]);
          basis_functions.insert(basis_functions.end(), particle_basis_functions.begin(), particle_basis_functions.end());
        }

        return basis_functions;
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

    std::vector<internal::CPDI::GeneralizedBasisData<dim>> data;

    double local_volume = 0.0;

    // Now loop over the locally owned active cells and assemble the CPDI systems
    std::vector<std::array<Point<3>, 3>> basis_functions;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // Find the cells neighboring the current cell
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cell_set;
          for (const auto v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cell_set.insert(vertex_to_cell_map[vertex_index].begin(),
                                          vertex_to_cell_map[vertex_index].end());
            }
          std::vector<typename Triangulation<dim>::active_cell_iterator>
          neighboring_cells(neighboring_cell_set.begin(), neighboring_cell_set.end());

          const std::vector<std::array<Point<3>, 3>> cell_basis_functions =
          internal::CPDI::evaluate(cell,
                                   neighboring_cells,
                                   particle_handler,
                                   field_fe,
                                   *mapping,
                                   bounding_box,
                                   data);
          basis_functions.insert(basis_functions.end(), cell_basis_functions.begin(), cell_basis_functions.end());

#if DEBUG
          for (unsigned int p = 0; p < particle_handler.n_particles_in_cell(cell); ++p)
            local_volume += data[p].get_volume();
#endif

          // Assemble the cell matrix and cell vectors for each neighboring cell
          for (unsigned int c = 0; c < neighboring_cells.size(); ++c)
            {
              cell_matrix = 0;
              for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                cell_rhs[field_index] = 0;

              const auto particles_in_cell = particle_handler.particles_in_cell(cell);
              for (auto pit = particles_in_cell.begin(); pit != particles_in_cell.end(); ++pit)
                {
                  const unsigned int p = std::distance(particles_in_cell.begin(), pit);
                  const internal::CPDI::GeneralizedBasisData<dim> &particle_data = data[p];

                  const double Vp = particle_data.get_volume();
                  const ArrayView<const double> particle_properties = pit->get_properties();

                  for (unsigned int i = 0; i < field_dofs_per_cell; ++i)
                    {
                      for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                        {
                          const double property_value = particle_properties[field_to_property_map[field_index]];
                          cell_rhs[field_index](i) += Vp * particle_data.get_value(c, i) * property_value;
                        }

                      for (unsigned int j = 0; j < field_dofs_per_cell; ++j)
                        cell_matrix(i, j) += Vp * particle_data.get_value(c, i) * particle_data.get_value(c, j);
                    }
                }

              // Collect the dof indices for each field
              const typename DoFHandler<dim>::active_cell_iterator
              dof_cell(&triangulation,
                       neighboring_cells[c]->level(),
                       neighboring_cells[c]->index(),
                       &dof_handler);
              dof_cell->get_dof_indices(cell_dof_indices);
              std::cout << "assemble cell " << dof_cell->active_cell_index() << std::endl;

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
              std::cout << "cell_matrix: " << std::endl;
              cell_matrix.print_formatted(std::cout);
              std::cout << "cell_rhs: " << std::endl;
              cell_rhs[0].print(std::cout);
              current_constraints.distribute_local_to_global(cell_matrix,
                                                             field_dof_indices[0],
                                                             system_matrix);

              for (unsigned int field_index = 0; field_index < n_fields; ++field_index)
                current_constraints.distribute_local_to_global(cell_rhs[field_index],
                                                               field_dof_indices[field_index],
                                                               system_rhs);
            }
        }

    std::ofstream out("voronoi.vtk");
    out << "# vtk DataFile Version 3.0\n";
    out << "Generalized basis function\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";

    out << "POINTS " << basis_functions.size() * 3 << " float\n";
    for (unsigned int t = 0; t < basis_functions.size(); ++t)
      for (unsigned int v = 0; v < 3; ++v)
      {
        for (unsigned int d = 0; d < 3; ++d)
          out << basis_functions[t][v][d] << ' ';
        out << "\n";
      }

    out << "POLYGONS " << basis_functions.size() << ' ' << basis_functions.size() * 4 << "\n";
    unsigned int point_index = 0;
    for (unsigned int t = 0; t < basis_functions.size(); ++t)
      out << "3 " << point_index++ << ' ' << point_index++ << ' ' << point_index++ << "\n";

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    std::cout << "system_matrix: " << std::endl;
    system_matrix.block(sparsity_block_idx, sparsity_block_idx).print(std::cout);
    std::cout << "system_rhs: " << std::endl;
    system_rhs.block(advection_fields[0].block_index(introspection)).print(std::cout);

#if DEBUG
    const double total_volume = Utilities::MPI::sum(local_volume, mpi_communicator);
    const double tria_volume  = GridTools::volume(triangulation, *mapping);
    std::cout << "total_volume = " << total_volume << ", tria_volume = " << tria_volume << std::endl;
    Assert(std::abs(total_volume - tria_volume) / tria_volume < 1.e-3,
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
