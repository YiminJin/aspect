/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

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

#ifndef _aspect_time_stepping_rsf_time_step_h
#define _aspect_time_stepping_rsf_time_step_h

#include <aspect/time_stepping/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace TimeStepping
  {
template <int dim>
    class RSFTimeStep : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        double execute() override;

        static
        void 
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        double CFL_number;

        struct ParticlePropertyIndices
        {
          unsigned int slip_rate;
          std::vector<unsigned int> chemical_fields;
        };

        ParticlePropertyIndices particle_property_indices;

        const Particles::ParticleHandler<dim> *particle_handler;
    };
  }
}

#endif
