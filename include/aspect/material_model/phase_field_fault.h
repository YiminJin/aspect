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

#include <functional>

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

        /**
         * Return the phase field used to evaluate degradation for I_h.
         * Negative numerical undershoots are mapped to zero. Finite values
         * above the physical upper bound are rejected without clipping.
         */
        static double
        normalization_effective_phase_field(const double raw_phase_field,
                                            const std::string &context);

        /** Validate the global minimum raw I_h sample against its tolerance. */
        static void
        validate_normalization_phase_field_minimum(const double minimum_raw_phase_field,
                                                   const std::string &context);

        /** Validate one effective phase field/degradation pair and return h. */
        static double
        normalization_integrand(const double phase_field,
                                const double degradation,
                                const std::string &context);

        /** One distributed bulk phase-field sample used by the profile integrator. */
        struct NormalizationPointSample
        {
          bool found = false;
          double phase_field = numbers::signaling_nan<double>();
          double cell_diameter = numbers::signaling_nan<double>();
        };

        /** Geometry and identity held fixed while integrating one normal profile. */
        struct NormalizationProfile
        {
          unsigned int id = numbers::invalid_unsigned_int;
          unsigned int fault_index = numbers::invalid_unsigned_int;
          unsigned int segment_index = numbers::invalid_unsigned_int;
          double xi = numbers::signaling_nan<double>();
          double surface_weight = numbers::signaling_nan<double>();
          Point<dim> origin;
          Tensor<1,dim> normal;
          std::vector<double> material_fractions;
        };

        using NormalizationPointEvaluator =
          std::function<std::vector<NormalizationPointSample>(const std::vector<Point<dim>> &)>;

        using NormalizationIntegrandEvaluator =
          std::function<double(const NormalizationProfile &,
                               unsigned int,
                               double,
                               const Point<dim> &,
                               const NormalizationPointSample &)>;

        /**
         * Integrate locally owned profiles while all ranks participate in
         * every distributed point-evaluation collective.
         */
        static std::vector<double>
        integrate_normalization_profiles(
          const std::vector<NormalizationProfile> &profiles,
          const double length_scale,
          const double quadrature_tolerance,
          const double tail_tolerance,
          const MPI_Comm communicator,
          const NormalizationPointEvaluator &evaluate_points,
          const NormalizationIntegrandEvaluator &integrand);

        /**
         * Internal empirical error-detection threshold for excessive raw
         * phase-field undershoot. This is not a physical parameter, a solver
         * tolerance, or a numerical convergence-control parameter.
         */
        static constexpr double normalization_phase_field_undershoot_tolerance = 1.e-4;

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

        double current_minimum_raw_normalization_phase_field =
          numbers::signaling_nan<double>();

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;
    };

    namespace internal
    {
      /** Narrow test seam for private PhaseFieldFault Stage B/C operations. */
      template <int dim>
      class PhaseFieldFaultTestAccess
      {
        public:
          using PointSample = typename PhaseFieldFault<dim>::NormalizationPointSample;

          static const std::vector<std::vector<double>> &
          compute_normalization_integrals(PhaseFieldFault<dim> &model)
          {
            model.compute_normalization_integrals();
            return model.current_normalization_integrals;
          }

          static double
          current_minimum_raw_normalization_phase_field(
            const PhaseFieldFault<dim> &model)
          {
            return model.current_minimum_raw_normalization_phase_field;
          }

          static double
          normalization_effective_phase_field(
            const double raw_phase_field,
            const std::string &context = "test profile")
          {
            return PhaseFieldFault<dim>::normalization_effective_phase_field(
              raw_phase_field, context);
          }

          static void
          validate_normalization_phase_field_minimum(
            const double minimum_raw_phase_field,
            const std::string &context = "test profile")
          {
            PhaseFieldFault<dim>::validate_normalization_phase_field_minimum(
              minimum_raw_phase_field, context);
          }

          static constexpr double
          normalization_phase_field_undershoot_tolerance()
          {
            return PhaseFieldFault<dim>::normalization_phase_field_undershoot_tolerance;
          }

          static double
          normalization_integrand(const double phase_field,
                                  const double degradation,
                                  const std::string &context = "test profile")
          {
            return PhaseFieldFault<dim>::normalization_integrand(
              phase_field, degradation, context);
          }

          static std::vector<double>
          integrate_normalization_profiles(
            const std::vector<Point<dim>> &origins,
            const std::vector<Tensor<1,dim>> &normals,
            const double length_scale,
            const double quadrature_tolerance,
            const double tail_tolerance,
            const MPI_Comm communicator,
            const typename PhaseFieldFault<dim>::NormalizationPointEvaluator &evaluate_points,
            const std::function<double(double)> &degradation)
          {
            AssertDimension(origins.size(), normals.size());
            std::vector<typename PhaseFieldFault<dim>::NormalizationProfile> profiles(origins.size());
            for (unsigned int i = 0; i < profiles.size(); ++i)
              {
                profiles[i].id = i;
                profiles[i].fault_index = 0;
                profiles[i].segment_index = 0;
                profiles[i].origin = origins[i];
                profiles[i].normal = normals[i];
              }

            const auto integrand =
              [&degradation](const typename PhaseFieldFault<dim>::NormalizationProfile &profile,
                             const unsigned int side,
                             const double zeta,
                             const Point<dim> &,
                             const typename PhaseFieldFault<dim>::NormalizationPointSample &sample)
              {
                const std::string context =
                  "test profile " + Utilities::int_to_string(profile.id)
                  + ", side " + Utilities::int_to_string(side)
                  + ", zeta=" + Utilities::to_string(zeta);
                const double phi = PhaseFieldFault<dim>::normalization_effective_phase_field(
                  sample.phase_field, context);
                return PhaseFieldFault<dim>::normalization_integrand(
                  phi, degradation(phi), context);
              };

            return PhaseFieldFault<dim>::integrate_normalization_profiles(
              profiles, length_scale, quadrature_tolerance, tail_tolerance,
              communicator, evaluate_points, integrand);
          }
      };
    }
  }
}

#endif
