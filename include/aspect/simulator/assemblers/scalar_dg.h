/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_ASSEMBLERS_SCALAR_DG_H
#define ASPECT_ASSEMBLERS_SCALAR_DG_H

namespace aspect
{
  namespace Assemblers
  {
    /** Symmetric interior-penalty/Nitsche scalar face kernel. The caller
     * supplies oriented jumps and averaged diffusive normal fluxes. Thermal
     * prefactors, physical time integration and JxW are deliberately external.
     * With boundary traces this includes both consistency terms and the penalty.
     */
    inline double scalar_symmetric_penalty(const double test_jump,
                                           const double test_normal_flux,
                                           const double trial_jump,
                                           const double trial_normal_flux,
                                           const double penalty)
    {
      return -test_normal_flux * trial_jump - trial_normal_flux * test_jump
             + penalty * test_jump * trial_jump;
    }
  }
}
#endif
