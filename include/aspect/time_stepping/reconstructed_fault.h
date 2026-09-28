/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_time_stepping_reconstructed_fault_h
#define _aspect_time_stepping_reconstructed_fault_h

#include <aspect/time_stepping/interface.h>
#include <limits>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldFault;
  }

  namespace TimeStepping
  {
    /** Law-specific restriction and optional committed-state aging predictor. */
    template <int dim>
    class ReconstructedFault : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm) override;

        void initialize() override;

        double execute() override;

      private:
        const MaterialModel::PhaseFieldFault<dim> *phase_field_fault = nullptr;
        double maximum_logarithmic_state_change = std::numeric_limits<double>::max();
    };
  }
}

#endif
