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

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldFault;
  }

  namespace TimeStepping
  {
    /** Return the law-specific timestep restriction of a reconstructed fault. */
    template <int dim>
    class ReconstructedFault : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        double execute() override;

      private:
        const MaterialModel::PhaseFieldFault<dim> *phase_field_fault = nullptr;
    };
  }
}

#endif
