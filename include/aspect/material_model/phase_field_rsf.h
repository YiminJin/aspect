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

        void update() override;

        void 
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                 MaterialModel::MaterialModelOutputs<dim> &out) const override;

        std::vector<double>
        get_critical_crack_driving_forces() const override;

        std::vector<double>
        get_critical_energy_release_rates() const override;

        bool
        is_compressible() const override;

        void
        create_additional_named_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const override;

        const Rheology::RateStateFriction<dim> &
        get_rate_state_friction_model() const;

        bool
        is_fractured(const double phase_field) const;
        
        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        void initialize_index_cache();

        void perform_return_mapping();

        void update_history_states(const SolverControl &nonlinear_solver_control);

        double 
        calculate_creep_viscosity(const double               temperature,
                                  const std::vector<double> &volume_fractions) const;

        double
        calculate_stress_relaxation_factor(const double creep_viscosity,
                                           const double shear_modulus) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::RateStateFriction<dim> rsf_rheology;

        struct IndexCache
        {
          /**
           * Structure caching the indices of the particle properties that are
           * frequently requested by this material model.
           */
          struct ParticlePropertyIndices
          {
            unsigned int crack_driving_force;
            unsigned int cohesive_force;
            unsigned int slip_rate;
            unsigned int slip_state;
            unsigned int normal_direction;
            unsigned int slip_direction;
            unsigned int ve_stress;
            std::vector<unsigned int> chemical_fields;

            /**
             * Default constructor. Initialize all the indices to 
             * <tt>numbers::invalid_unsigned_int</tt>.
             */
            ParticlePropertyIndices()
              : crack_driving_force(numbers::invalid_unsigned_int)
              , cohesive_force(numbers::invalid_unsigned_int)
              , slip_rate(numbers::invalid_unsigned_int)
              , slip_state(numbers::invalid_unsigned_int)
              , normal_direction(numbers::invalid_unsigned_int)
              , slip_direction(numbers::invalid_unsigned_int)
              , ve_stress(numbers::invalid_unsigned_int)
            {}
          };

          ParticlePropertyIndices particle_property_indices;

          /**
           * Structure caching the indices of the compositional fields that
           * are frequently requested by this material model.
           */
          struct CompositionalIndices
          {
            unsigned int slip_rate;
            unsigned int slip_state;

            /**
             * Default constructor. Initialize all the indices to
             * <tt>numbers::invalid_unsigned_int</tt>.
             */
            CompositionalIndices()
              : slip_rate(numbers::invalid_unsigned_int)
              , slip_state(numbers::invalid_unsigned_int)
            {}
          };

          CompositionalIndices compositional_indices;

          /**
           * Structure caching the indices of the variable components that
           * are frequently requested by this material model.
           */
          struct ComponentIndices
          {
            unsigned int phase_field;
            unsigned int peak_phase_field;

            /**
             * Default constructor. Initialize all the indices to
             * <tt>numbers::invalid_unsigned_int</tt>.
             */
            ComponentIndices()
              : phase_field(numbers::invalid_unsigned_int)
              , peak_phase_field(numbers::invalid_unsigned_int)
            {}
          };

          ComponentIndices component_indices;

          /**
           * Initialize the index cache.
           */
          void initialize(const Introspection<dim>     &introspection,
                          const Parameters<dim>        &parameters,
                          const PhaseFieldHandler<dim> &phase_field_handler);
        };

        IndexCache index_cache;

        MaterialUtilities::CompositionalAveragingOperation viscosity_averaging;

        double reference_temperature;

        double maximum_viscosity;
        
        double minimum_viscosity;

        double phase_field_activation_threshold;

        double phase_field_normal_lock_threshold;

        double initial_time_step;

        std::vector<double> thermal_conductivities;

        std::vector<double> reference_viscosities;

        std::vector<double> thermal_viscosity_exponents;

        std::vector<double> elastic_shear_moduli;

        std::vector<double> cohesions;

        std::vector<double> critical_energy_release_rates;

        std::vector<double> radiation_damping_coefficients;

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;

        std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags;
    };
  }
}

#endif
