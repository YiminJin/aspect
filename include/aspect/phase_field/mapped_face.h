/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_MAPPED_FACE_H
#define ASPECT_PHASE_FIELD_MAPPED_FACE_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace PhaseField
  {
    /** Mapping-aware face scratch shared by transport and fracture assembly.
     * A coarse row visits its subfaces; a fine row visits its entire face.
     * Neighbor traces are evaluated at the very same physical quadrature
     * locations, independently of reference-face orientation and MPI ownership.
     */
    template <int dim>
    class MappedFaceScratch
    {
      public:
        MappedFaceScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &element,
                          const Quadrature<dim-1> &rule)
          : face(mapping, element, rule, update_values | update_gradients |
                 update_quadrature_points | update_normal_vectors | update_JxW_values),
          subface(mapping, element, rule, update_values | update_gradients |
                  update_quadrature_points | update_normal_vectors | update_JxW_values)
        {}

        template <class Iterator>
        const FEFaceValuesBase<dim> &reinit(const Iterator &cell, unsigned int f,
                                            unsigned int sub = numbers::invalid_unsigned_int)
        {
          const typename Triangulation<dim>::cell_iterator native(cell);
          if (sub == numbers::invalid_unsigned_int)
            {
              face.reinit(native, f);
              return face;
            }
          subface.reinit(native, f, sub);
          return subface;
        }

      private:
        FEFaceValues<dim> face;
        FESubfaceValues<dim> subface;
    };

    /** Lower Jacobian determinant on a bilinear quadrilateral. In 2-D the
     * determinant is affine on the reference cell, so its minimum is attained
     * at a corner. This is not a bound for higher-order curved mappings. */
    template <int dim, class Iterator>
    double minimum_cell_jacobian(const Mapping<dim> &mapping,
                                 const FiniteElement<dim> &element, const Iterator &cell)
    {
      QGaussLobatto<dim> corners(2);
      FEValues<dim> values(mapping, element, corners, update_jacobians);
      values.reinit(typename Triangulation<dim>::cell_iterator(cell));
      double minimum = std::numeric_limits<double>::max();
      for (unsigned int q = 0; q < corners.size(); ++q)
        minimum = std::min(minimum, values.jacobian(q).determinant());
      AssertThrow(minimum > 0., ExcMessage("Native phase requires an orientation-preserving bilinear cell map."));
      return minimum;
    }
  }
}
#endif
