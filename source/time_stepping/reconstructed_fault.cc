/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/time_stepping/reconstructed_fault.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>

namespace aspect
{
  namespace TimeStepping
  {
    template <int dim>
    void
    ReconstructedFault<dim>::initialize()
    {
      phase_field_fault =
        &Plugins::get_plugin_as_type<
          const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
    }



    template <int dim>
    double
    ReconstructedFault<dim>::execute()
    {
      Assert(phase_field_fault != nullptr, ExcInternalError());
      return phase_field_fault->compute_reconstructed_fault_time_step(
        this->get_parameters().CFL_number);
    }



    ASPECT_REGISTER_TIME_STEPPING_MODEL(
      ReconstructedFault,
      "reconstructed fault time step",
      "Compute the operator-split law-specific timestep restriction from "
      "the timestep-committed reconstructed-fault slip rate and surface "
      "material mixture. This model is opt-in and uses ASPECT's global CFL "
      "number.")
  }
}
