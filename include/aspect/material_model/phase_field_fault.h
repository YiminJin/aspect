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

#ifndef _aspect_material_model_phase_field_fault_h
#define _aspect_material_model_phase_field_fault_h

#include <aspect/simulator_access.h>
#include <aspect/phase_field.h>
#include <aspect/solution_evaluator.h>
#include <aspect/material_model/interface.h>
#include <aspect/material_model/equation_of_state/multicomponent_incompressible.h>
#include <aspect/material_model/rheology/fault_friction.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      template <int dim>
      class PhaseFieldFaultTestAccess;
    }

    template <int dim>
    class PhaseFieldFault : public Interface<dim>,
      public PhaseFieldModel<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void 
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                 MaterialModel::MaterialModelOutputs<dim> &out) const override;

        std::vector<double>
        get_critical_crack_driving_forces() const override;

        std::vector<double>
        get_critical_energy_release_rates() const override;
        
        double
        get_phase_field_activation_threshold() const override;

        double
        get_phase_field_upper_admissibility_threshold() const override;

        bool
        is_compressible() const override;

        void
        initialize() override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        friend class internal::PhaseFieldFaultTestAccess<dim>;

        /** Time-discrete coefficients of the Maxwell law. */
        struct MaxwellCoefficients
        {
          double beta;
          double kappa;
        };

        /** Compute beta and kappa for positive viscosity and shear modulus. */
        static MaxwellCoefficients
        compute_maxwell_coefficients(const double viscosity,
                                     const double shear_modulus,
                                     const double time_step);

        /**
         * Apply the non-rotational time-discrete Maxwell law to an effective
         * bulk strain rate.
         */
        static SymmetricTensor<2,dim>
        compute_maxwell_stress(const MaxwellCoefficients &coefficients,
                               const SymmetricTensor<2,dim> &effective_bulk_strain_rate,
                               const SymmetricTensor<2,dim> &previous_stress);

        /** Recompute transient nodal I_h from the current distributed phase field. */
        void
        compute_normalization_integrals();

        /** Validate one sampled phase field/degradation pair and return h. */
        static double
        normalization_integrand(const double phase_field,
                                const double degradation,
                                const std::string &context);

        double 
        calculate_creep_viscosity(const std::vector<double> &volume_fractions,
                                  const double               temperature) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::FaultFriction<dim> fault_friction;

        MaterialUtilities::CompositionalAveragingOperation viscosity_averaging;

        double reference_temperature;

        double maximum_viscosity;
        
        double minimum_viscosity;

        double phase_field_activation_threshold;

        double phase_field_normal_lock_threshold;

        std::vector<double> thermal_conductivities;

        std::vector<double> reference_viscosities;

        std::vector<double> thermal_viscosity_exponents;

        std::vector<double> elastic_shear_moduli;

        std::vector<double> cohesions;

        std::vector<double> initial_friction_coefficients;

        std::vector<double> critical_energy_release_rates;

        std::vector<double> radiation_damping_coefficients;

        double initial_time_step;

        bool evolve_phase_field;

        double normalization_quadrature_tolerance;

        double normalization_tail_tolerance;

        unsigned int fault_composition_property_index = numbers::invalid_unsigned_int;

        std::vector<std::vector<double>> current_normalization_integrals;

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;
    };

    namespace internal
    {
      /** Narrow test seam for private PhaseFieldFault Stage B/C operations. */
      template <int dim>
      class PhaseFieldFaultTestAccess
      {
        public:
          static const std::vector<std::vector<double>> &
          compute_normalization_integrals(PhaseFieldFault<dim> &model)
          {
            model.compute_normalization_integrals();
            return model.current_normalization_integrals;
          }

          static double
          normalization_integrand(const double phase_field,
                                  const double degradation,
                                  const std::string &context = "test profile")
          {
            return PhaseFieldFault<dim>::normalization_integrand(
              phase_field, degradation, context);
          }
      };
    }
  }
}

#endif
