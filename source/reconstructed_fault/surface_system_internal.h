/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_reconstructed_fault_surface_system_internal_h
#define _aspect_reconstructed_fault_surface_system_internal_h

#include <aspect/reconstructed_fault/surface_system.h>

namespace aspect
{
  template <int dim>
  struct ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  {
    struct CouplingPoint
    {
      Point<dim> position;
      unsigned int parent_index;
      unsigned int fault_index;
      unsigned int segment_index;
      double xi;
      double particle_domain_volume;
      double eta_ve;
      double friction_coefficient;
      SymmetricTensor<2,dim> slip_tensor;
      SymmetricTensor<2,dim> normal_tensor;
      bool uses_adiabatic_friction_pressure;
    };

    ReconstructedFaultSurfaceResidual residual;
    std::vector<std::vector<double>> diagonal;
    std::vector<std::vector<double>> off_diagonal;
    std::vector<std::vector<double>> mass_diagonal;
    std::vector<std::vector<double>> mass_off_diagonal;
    std::vector<CouplingPoint> coupling_points;
    std::shared_ptr<NormalTractionDiagnostic> normal_diagnostic;
    std::vector<std::shared_ptr<const internal::FaultNormalFilter>> normal_filters;
  };


}

#endif
