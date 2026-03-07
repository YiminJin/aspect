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

#ifndef _aspect_material_model_phase_field_rsf_h
#define _aspect_material_model_phase_field_rsf_h

#include <aspect/simulator_access.h>
#include <aspect/phase_field.h>
#include <aspect/solution_evaluator.h>
#include <aspect/material_model/interface.h>
#include <aspect/material_model/equation_of_state/multicomponent_incompressible.h>
#include <aspect/material_model/rheology/rate_state_friction.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldRSF : public Interface<dim>,
      public PhaseFieldModel<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        void 
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                 MaterialModel::MaterialModelOutputs<dim> &out) const override;

        std::vector<double>
        get_threshold_crack_driving_forces() const override;

        std::vector<double>
        get_critical_energy_release_rates() const override;

        bool
        is_compressible() const override;

        const Rheology::RateStateFriction<dim> &
        get_rate_state_friction_model() const;

        bool
        is_fractured(const double phase_field) const;

        double
        calculate_friction_strength(const Point<dim>          &position,
                                    const double               slip_rate,
                                    const double               slip_state,
                                    const std::vector<double> &volume_fractions) const;
        
        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        void initialize_particle_data_info();

        void perform_return_mapping();

        void update_history_states();

        double 
        calculate_creep_viscosity(const double               temperature,
                                  const std::vector<double> &volume_fractions) const;

        double
        calculate_viscoelastic_viscosity(const double creep_viscosity,
                                         const double shear_modulus) const;

        SymmetricTensor<2, dim>
        calculate_deviatoric_stress(const SymmetricTensor<2, dim> &strain_rate,
                                    const SymmetricTensor<2, dim> &old_stress,
                                    const double                   creep_viscosity,
                                    const double                   shear_modulus) const;

        double 
        calculate_friction_coefficient(const double               slip_rate,
                                       const double               slip_state,
                                       const std::vector<double> &volume_fractions) const;

        double
        calculate_friction_coefficient_derivative(const double               slip_rate,
                                                  const double               slip_state,
                                                  const std::vector<double> &volume_fractions) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::RateStateFriction<dim> rsf_rheology;

        struct ParticleDataPositions
        {
          unsigned int crack_driving_force;
          unsigned int slip_rate;
          unsigned int slip_state;
          unsigned int normal_direction;
          unsigned int slip_direction;
          unsigned int bulk_stress;
          unsigned int interface_stress;
          std::vector<unsigned int> chemical_fields;
        };

        ParticleDataPositions particle_data_positions;

        ComponentMask particle_data_mask;

        MaterialUtilities::CompositionalAveragingOperation viscosity_averaging;

        double reference_temperature;

        double maximum_viscosity;
        
        double minimum_viscosity;

        double phase_field_kinetic_threshold;

        double phase_field_geometric_threshold;

        double initial_time_step;

        std::vector<double> thermal_conductivities;

        std::vector<double> reference_viscosities;

        std::vector<double> thermal_viscosity_exponents;

        std::vector<double> elastic_shear_moduli;

        std::vector<double> cohesions;

        std::vector<double> critical_energy_release_rates;

        std::vector<double> radiation_damping_coefficients;

        Particle::Manager<dim> *particle_manager;

        std::unique_ptr<SolutionEvaluator<dim>> evaluator;

        std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags;

        unsigned int phase_field_component_index;
    };
  }
}

#endif
