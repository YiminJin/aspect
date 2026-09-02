/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/particle/manager.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyMaxwellStressInitialization : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());

          const Particle::Manager<dim> &particle_manager =
            this->get_particle_manager(0);
          const unsigned int stress_position =
            particle_manager.get_property_manager().get_data_info()
            .get_position_by_field_name("maxwell stress");

          double local_maximum_error = 0.0;
          for (const auto &particle : particle_manager.get_particle_handler())
            {
              const Point<dim> &position = particle.get_location();
              const ArrayView<const double> properties = particle.get_properties();
              local_maximum_error = std::max(
                local_maximum_error,
                std::abs(properties[stress_position] - (position[0] + 2.0 * position[1])));
              local_maximum_error = std::max(
                local_maximum_error,
                std::abs(properties[stress_position + 1] - (3.0 * position[0] - position[1])));
              local_maximum_error = std::max(
                local_maximum_error,
                std::abs(properties[stress_position + 2] - 5.0));
            }

          const double maximum_error = Utilities::MPI::max(
            local_maximum_error,
            this->get_mpi_communicator());
          AssertThrow(maximum_error < 1.e-12,
                      ExcMessage("Maxwell stress particle initialization does not match "
                                 "the mapped initial composition fields."));

          return {"Maxwell stress initialization:", "verified"};
        }
    };



    ASPECT_REGISTER_POSTPROCESSOR(VerifyMaxwellStressInitialization,
                                  "verify maxwell stress initialization",
                                  "Verify the Stage 2 Maxwell stress initialization test data.")
  }
}
