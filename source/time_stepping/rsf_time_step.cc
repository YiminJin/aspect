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

#include <aspect/time_stepping/rsf_time_step.h>
#include <aspect/material_model/phase_field_rsf.h>

namespace aspect
{
  namespace TimeStepping
  {
    template <int dim>
    void RSFTimeStep<dim>::initialize()
    {
      // Find the particle manager that handles the slip rate
      particle_handler = nullptr;
      for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
        {
          const Particle::Manager<dim> &particle_manager = this->get_particle_manager(i);
          const auto &data_info = particle_manager.get_property_manager().get_data_info();
          if (data_info.fieldname_exists("slip_rate"))
            {
              particle_handler = &particle_manager.get_particle_handler();
              slip_rate_property_index = data_info.get_field_index_by_name("slip_rate");
            }
        }

      AssertThrow(particle_handler != nullptr,
                  ExcMessage("Time stepping model 'rsf time step' requires a particle property "
                             "with the name 'slip_rate'."));

      // Get the characteristic slip distance from the RSF rheological model
      const MaterialModel::PhaseFieldRSF<dim>* material_model =
        dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>*>(&this->get_material_model());
      AssertThrow(material_model != nullptr, 
                  ExcMessage("Time stepping model 'rsf time step' requires to get access to an object of "
                             "MaterialModel::Rheology::RateStateFriction through the material model. "
                             "Currently, 'phase field rsf' is the only material model that provides the "
                             "access."));
      characteristic_slip_distance = material_model->get_rate_state_friction_model().get_characteristic_slip_distance();
    }



    template <int dim>
    double RSFTimeStep<dim>::execute()
    {
      // Loop over the particles and find the maximum slip rate
      double local_maximum_slip_rate = 0;
      for (const auto &cell : this->get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &particle : particle_handler->particles_in_cell(cell))
            {
              const double slip_rate = particle.get_properties()[slip_rate_property_index];
              local_maximum_slip_rate = std::max(local_maximum_slip_rate, slip_rate);
            }

      const double global_maximum_slip_rate = Utilities::MPI::max(local_maximum_slip_rate, 
                                                                  this->get_mpi_communicator());

      AssertThrow(global_maximum_slip_rate > std::numeric_limits<double>::epsilon(),
                  ExcMessage("Slip rate has not been initialized when calculating the RSF time step."));

      return safety_factor * characteristic_slip_distance / global_maximum_slip_rate;
    }



    template <int dim>
    void RSFTimeStep<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time stepping");
      {
        prm.enter_subsection("RSF time step");
        {
          prm.declare_entry("Safety factor", "0.1",
                            Patterns::Double(0),
                            "A safety factor, $C_{\\delta}$, in the RSF time step limit. "
                            "The time step is computed by $\\Delta t_{RSF} = C_{\\delta}"
                            "\\frac{D_c}{V_{\\text{max}}}$, where $D_c$ and $V_{\\text{max}}$"
                            "denote the characteristic slip distance and the slip rate, "
                            "respectively.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void RSFTimeStep<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time stepping");
      {
        prm.enter_subsection("RSF time step");
        {
          safety_factor = prm.get_double("Safety factor");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace TimeStepping
  {
    ASPECT_REGISTER_TIME_STEPPING_MODEL(RSFTimeStep,
                                        "rsf time step",
                                        "This model computes the rate-state-friction time step as "
                                        "$C_{\\delta} * D_c / V_{\\text{max}}$ over all particles, "
                                        "where $D_c$ is the characteristic slip distance, $V$ is the "
                                        "slip rate, and $C_{\\delta}$ is a safety factor.")
  }
}
