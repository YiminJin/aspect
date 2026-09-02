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

#include <deal.II/particles/particle_handler.h>

#include <map>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      /** Time-discrete coefficients of the Maxwell law. */
      struct MaxwellCoefficients
      {
        double beta;
        double kappa;
      };

      /** Compute beta and kappa for positive viscosity and shear modulus. */
      MaxwellCoefficients
      compute_maxwell_coefficients(const double viscosity,
                                   const double shear_modulus,
                                   const double time_step);

      /**
       * Apply the non-rotational time-discrete Maxwell law to an effective
       * bulk strain rate.
       */
      template <int dim>
      SymmetricTensor<2,dim>
      compute_maxwell_stress(const MaxwellCoefficients &coefficients,
                             const SymmetricTensor<2,dim> &effective_bulk_strain_rate,
                             const SymmetricTensor<2,dim> &previous_stress);

      /**
       * Hold a complete pending particle-stress update without changing the
       * committed particle properties. Commit validates every rank before any
       * rank writes, while rollback only discards pending values.
       */
      template <int dim>
      class MaxwellStressUpdateTransaction
      {
        public:
          void begin();

          void
          stage(const types::particle_index particle_id,
                const SymmetricTensor<2,dim> &stress);

          void
          commit(dealii::Particles::ParticleHandler<dim> &particle_handler,
                 const unsigned int property_data_position,
                 const MPI_Comm mpi_communicator);

          void rollback();

          bool is_active() const;

        private:
          bool active = false;
          bool pending_update_is_valid = true;
          std::map<types::particle_index, SymmetricTensor<2,dim>> pending_stresses;
      };
    }



    template <int dim>
    class PhaseFieldRSF : public Interface<dim>,
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
        
        std::pair<double, double>
        get_phase_field_range() const override;

        bool
        is_compressible() const override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        double 
        calculate_creep_viscosity(const std::vector<double> &volume_fractions,
                                  const double               temperature) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::RateStateFriction<dim> rsf_rheology;

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

        internal::MaxwellStressUpdateTransaction<dim> maxwell_stress_update;

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;
    };
  }
}

#endif
