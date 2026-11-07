/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_MAPPED_FACE_H
#define ASPECT_PHASE_FIELD_MAPPED_FACE_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace PhaseField
  {
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
