/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_sections_h
#define _aspect_material_sections_h

#include <aspect/global.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/quadrature.h>

namespace aspect
{
  namespace GrowingFault
  {
    /**
     * A material section is a strip between two fixed projections on the seed
     * tangent. Each owned cell stores a reference quadrature and one strip
     * index per point. The pair (CellId, quadrature index) identifies material
     * through retries and continuation on this fixed mesh.
     *
     * The structural plane is the local Mode II plane at angle beta to the
     * seed. A prospective plane can turn. A formed plane retains its angle
     * because it carries physical shear history. Turning changes length and
     * shear orientation while the integration points and areas stay fixed.
     */
    struct MaterialSections
    {
      struct CellRule
      {
        Quadrature<2> quadrature;
        std::vector<unsigned int> section;
      };

      /** Collective construction on a conforming Cartesian mesh. Only strip
       * boundaries and the material frame are replicated. */
      MaterialSections(const Triangulation<2> &mesh, MPI_Comm comm,
                       double seed_angle_degrees, unsigned int gauss_order);

      double coordinate(const Point<2> &point) const;
      double width(unsigned int section) const;
      double length(unsigned int section, double beta) const;
      SymmetricTensor<2,3> shear(double beta) const;

      /** Lower box corner. Stations are tangent projections relative to this origin. */
      Point<2> origin;
      Tensor<1,2> tangent;
      std::vector<double> edges;
      std::map<CellId,CellRule> cells;
    };

    /** Rank-local weighted strip integrals followed by a collective sum. The
     * adjoint return at point q is the resistance of its unique material strip. */
    struct SectionTransfer
    {
      explicit SectionTransfer(unsigned int n) : free_rate(n), mobility(n) {}
      void add(unsigned int strip, double area_over_length, double rate, double point_mobility);
      void sum(MPI_Comm);
      std::vector<double> free_rate, mobility;
    };
  }
}
#endif
