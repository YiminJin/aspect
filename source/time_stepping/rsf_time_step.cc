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

              particle_property_indices.slip_rate = data_info.get_field_index_by_name("slip_rate");

              particle_property_indices.chemical_fields.clear();
              for (const unsigned int index : this->introspection().chemical_composition_field_indices())
                particle_property_indices.chemical_fields.push_back(
                  data_info.get_position_by_field_name(this->get_parameters().mapped_particle_properties.find(index)->second.first));
            }
        }

      AssertThrow(particle_handler != nullptr,
                  ExcMessage("Time stepping model 'rsf time step' requires a particle property "
                             "with the name 'slip_rate'."));

      // Get the characteristic slip distance from the RSF rheological model
      const MaterialModel::PhaseFieldRSF<dim> *material_model =
        dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>*>(&this->get_material_model());
      AssertThrow(material_model != nullptr, 
                  ExcMessage("Time stepping model 'rsf time step' requires to get access to an object of "
                             "MaterialModel::Rheology::RateStateFriction through the material model. "
                             "Currently, 'phase field rsf' is the only material model that provides the "
                             "access."));
    }



    template <int dim>
    double RSFTimeStep<dim>::execute()
    {
      const MaterialModel::Rheology::RateStateFriction<dim> &rsf_model =
        dynamic_cast<const MaterialModel::PhaseFieldRSF<dim>*>(&this->get_material_model())->get_rate_state_friction_model();

      std::vector<double> chemical_field_values(this->introspection().n_chemical_composition_fields());

      // Loop over the particles and find the minimum of a/b * D_c/V
      double local_dt_rsf = this->get_parameters().end_time - this->get_parameters().start_time;

      for (const auto &cell : this->get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &particle : particle_handler->particles_in_cell(cell))
            {
              const ArrayView<const double> particle_properties = particle.get_properties();
              const double V = particle_properties[particle_property_indices.slip_rate];
              for (unsigned int j = 0; j < chemical_field_values.size(); ++j)
                chemical_field_values[j] = particle_properties[particle_property_indices.chemical_fields[j]];
              const std::vector<double> volume_fractions = 
                MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_field_values);

              local_dt_rsf = std::min(local_dt_rsf, 
                                      rsf_model.compute_time_step(volume_fractions,
                                                                  V, CFL_number, true));
            }

      return Utilities::MPI::min(local_dt_rsf, this->get_mpi_communicator());
    }



    template <int dim>
    void RSFTimeStep<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time stepping");
      {
        prm.enter_subsection("RSF time step");
        {
          prm.declare_entry("CFL number", "0.5",
                            Patterns::Double(0),
                            "The CFL number that determines the stability of time evolution "
                            "controlled by fault slip.");
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
          CFL_number = prm.get_double("CFL number");
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
                                        "$C_{\\text{RSF}}\\min\\frac{a}{b}\\frac{D_c}{V}$ over all particles, "
                                        "where $D_c$ denotes the characteristic slip distance, $V$ is the "
                                        "slip rate, $a$ and $b$ are the direct effect and evolution effect "
                                        "parameters, respectively, and $C_{\\text{RSF}}$ serves as the CFL number.")
  }
}
