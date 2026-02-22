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

#include <aspect/material_model/phase_field_rsf.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/interpolator/interface.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/solution_evaluator.h>
#include <aspect/newton.h>

#include <random>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void PhaseFieldRSF<dim>::initialize()
    {
      AssertThrow(this->get_parameters().enable_phase_field == true,
                  ExcMessage("The phase field PSF model requires phase field to be included in "
                             "the system formulation. Please set <Formulation/Include phase field> to true."));

      AssertThrow(this->get_parameters().enable_elasticity == false,
                  ExcMessage("The phase field RSF model assumes viscoelastic rheology for intact regions, but "
                             "the elasticity is not handled by MaterialModel::Rheology::Elasticity. Please set "
                             "<Formulation/Enable elasticity> to false."));

      // Find the particle manager handling the crack driving force
      // The existance of the map from compositional index to particle property 
      // is checked by Particle::Property::PhaseFieldRSF
      for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
        {
          Particle::Manager<dim> &manager = this->get_particle_manager(i);
          if (manager.get_property_manager().get_data_info().fieldname_exists("crack_driving_force"))
            {
              particle_manager = &manager;
              break;
            }
        }
      AssertThrow(particle_manager != nullptr, ExcInternalError());

      // Initialize the particle data positions
      const auto &particle_data_info = particle_manager->get_property_manager().get_data_info();

      particle_data_positions.crack_driving_force = particle_data_info.get_position_by_field_name("crack_driving_force");
      particle_data_positions.slip_rate           = particle_data_info.get_position_by_field_name("slip_rate");
      particle_data_positions.slip_state          = particle_data_info.get_position_by_field_name("slip_state");
      particle_data_positions.normal              = particle_data_info.get_position_by_field_name("normal");
      particle_data_positions.stress              = particle_data_info.get_position_by_field_name("stress");

      // TODO
      this->get_signals().pre_assemble_stokes_system.connect(
        [&](const SimulatorAccess<dim> &)
      {
        this->perform_return_mapping();
      });
    }



    template <int dim>
    void
    PhaseFieldRSF<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      const unsigned int n_compositions = this->introspection().n_chemical_composition_fields() + 1;
      EquationOfStateOutputs<dim> eos_outputs(n_compositions);

      const std::shared_ptr<MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = out.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();

      // In principle, the values and gradients of the phase field should be calculated
      // with the particle-domain-based shape functions, which are used in the assembly
      // of the phase field system. However, those shape functions are not cached ---
      // we only have their particle-domain-wise averages, i.e. the CPDI weighting functions.
      // Since the particle-domain-based shape functions can be regardes as approximations
      // of the Q1 shape functions, we use the Q1 function values and gradients here.
      const std::shared_ptr<const MaterialModel::PhaseFieldInputs<dim>> phase_field_inputs 
        = in.template get_additional_input_object<MaterialModel::PhaseFieldInputs<dim>>();

      for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
        {
          const std::vector<double> volume_fractions = MaterialUtilities::compute_only_composition_fractions(
            in.composition[i], this->introspection().chemical_composition_field_indices());

          // Fill in the equation-of-state outputs
          equation_of_state.evaluate(in, i, eos_outputs);

          out.densities[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.densities, MaterialUtilities::arithmetic);
          out.thermal_expansion_coefficients[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.thermal_expansion_coefficients, MaterialUtilities::arithmetic);
          out.specific_heat[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.specific_heat_capacities, MaterialUtilities::arithmetic);
          out.thermal_conductivities[i] = MaterialUtilities::average_value(volume_fractions, thermal_conductivities, MaterialUtilities::arithmetic);
          out.compressibilities[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.compressibilities, MaterialUtilities::arithmetic);
          out.entropy_derivative_pressure[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.entropy_derivative_pressure, MaterialUtilities::arithmetic);
          out.entropy_derivative_temperature[i] = MaterialUtilities::average_value(volume_fractions, eos_outputs.entropy_derivative_temperature, MaterialUtilities::arithmetic);

        }
    }
  }
}

// explicit instantiation
namespace aspect
{
namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldRSF,
                                   "phase field rsf",
                                   "")
  }
}
