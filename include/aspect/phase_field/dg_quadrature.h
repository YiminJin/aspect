/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_DG_QUADRATURE_H
#define ASPECT_PHASE_FIELD_DG_QUADRATURE_H
namespace aspect
{
  namespace PhaseField
  {
    /** Explicit assembly policy for static Q1 maps. Lobatto n integrates degree
     * 2n-3. Products contain two phase factors, the velocity and mapping
     * cofactors. For p=p_v=2 in 2-D this retains mass4/transport5.
     * Mapped diffusion is rational: fracture5 is a retained oversampling policy,
     * not a claim of polynomial exactness on bilinear cells. */
    template <int dim>
    struct DGQuadrature
    {
      DGQuadrature(unsigned int phase_degree, unsigned int velocity_degree = 2)
        : mass_lobatto((2*phase_degree + dim-1 + 4)/2),
          transport_lobatto((2*phase_degree + velocity_degree + dim-1 + 4)/2),
          fracture_gauss(2*phase_degree + 1) {}
      unsigned int mass_lobatto, transport_lobatto, fracture_gauss;
    };
  }
}
#endif
