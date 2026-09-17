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
#include <aspect/reconstructed_fault/manager.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/timer.h>

#include <functional>
#include <map>

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
        /**
         * @name Material-model interface
         * @{
         */
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
        /**
         * @}
         */

        /**
         * @name Reconstructed-fault pointwise constitutive operations
         * @{
         */
        /** Inputs for one non-committing constitutive evaluation. */
        struct ReconstructedFaultPointInputs
        {
          unsigned int fault_index = numbers::invalid_unsigned_int;
          unsigned int segment_index = numbers::invalid_unsigned_int;
          double xi = numbers::signaling_nan<double>();
          Point<dim> position;

          double slip_rate = numbers::signaling_nan<double>();
          double phase_field = numbers::signaling_nan<double>();
          double previous_phase_field = numbers::signaling_nan<double>();
          double temperature = numbers::signaling_nan<double>();
          double dynamic_pressure = numbers::signaling_nan<double>();

          std::vector<double> bulk_material_fractions;
          SymmetricTensor<2,dim> strain_rate;
          SymmetricTensor<2,dim> old_maxwell_stress;
          SymmetricTensor<2,dim> slip_tensor;
          SymmetricTensor<2,dim> normal_tensor;
        };

        /** Constitutive values required by the surface weak form. */
        struct ReconstructedFaultPointResponse
        {
          /** Evaluated traction terms, for projection-consistent weak diagnostics. */
          double shear_traction;
          /** Total normal traction and its fixed background contribution. */
          double normal_traction;
          double background_normal_traction;
          double cohesive_traction;
          double friction_traction;
          double damping_traction;
          double residual_density = numbers::signaling_nan<double>();
          double minus_derivative_wrt_slip_rate = numbers::signaling_nan<double>();
          double kappa = numbers::signaling_nan<double>();
          double localization_factor = numbers::signaling_nan<double>();
          double friction_coefficient = numbers::signaling_nan<double>();
          bool uses_adiabatic_friction_pressure = false;
        };

        /** Inputs required by the reconstructed-fault bulk weak form. */
        struct ReconstructedFaultBulkPointInputs
        {
          unsigned int fault_index = numbers::invalid_unsigned_int;
          unsigned int segment_index = numbers::invalid_unsigned_int;
          double xi = numbers::signaling_nan<double>();
          double phase_field = numbers::signaling_nan<double>();
          double previous_phase_field = numbers::signaling_nan<double>();
          double temperature = numbers::signaling_nan<double>();
          std::vector<double> bulk_material_fractions;
        };

        /** Constitutive scalars required by the fault bulk weak form. */
        struct ReconstructedFaultBulkPointResponse
        {
          double kappa = numbers::signaling_nan<double>();
          double localization_factor = numbers::signaling_nan<double>();
          double history_correction = numbers::signaling_nan<double>();
        };

        ReconstructedFaultPointResponse
        evaluate_reconstructed_fault_point(
          const ReconstructedFaultPointInputs &inputs) const;

        /** Select a manager-owned two-component Q1 field (shear, compressive
         * normal) of fixed background tractions. Invalid index disables it.
         * Configure outside mechanics and keep values frozen during the solve.
         * These offsets never enter bulk assembly or Maxwell history. */
        void set_reconstructed_fault_background_traction_property(
          unsigned int property_index,
          unsigned int shear_correction_property = numbers::invalid_unsigned_int);

        /** Fixed background at a surface coordinate. An optional three-component
         * correction (a,b,d) subtracts a(s)+b(s)/d(s), without Q1 reprojection. */
        std::pair<double,double> reconstructed_fault_background_tractions(
          unsigned int fault, unsigned int segment, double xi) const;

        /** Explicit constitutive selection, never inferred from a frozen phase. */
        bool is_mature_frictional_fault() const;

        /** Select immutable, geometry-checked outside-profile integrals for a
         * frozen mature through-boundary benchmark. Applied before the existing
         * consistent I_h projection; no outside mechanical integration is added.
         * Reattach the same file on restart, before constitutive preparation.
         */
        void set_boundary_normalization_completion_file(const std::string &path);

        ReconstructedFaultBulkPointResponse
        evaluate_reconstructed_fault_bulk_point(
          const ReconstructedFaultBulkPointInputs &inputs) const;

        /** Frozen beta*tau_old at any bulk point, independent of fault support.
         * Composition contains all compositional fields in introspection order. */
        SymmetricTensor<2,dim>
        evaluate_frozen_maxwell_stress(const double temperature,
                                      const std::vector<double> &composition,
                                      const SymmetricTensor<2,dim> &old_stress) const;

        double minimum_fault_slip_rate() const;

        /** Whether friction uses prescribed adiabatic rather than bulk pressure.
         * Only in this mode can a constant bulk-pressure shift be treated as
         * a gauge operation without changing the surface equation. */
        bool uses_adiabatic_friction_pressure() const;

        /**
         * Prepare frozen reconstructed-fault constitutive state for the
         * Stage-I mechanical solve. Missing persistent state may be
         * initialized only for a fresh timestep-zero model.
         */
        void prepare_reconstructed_fault_mechanical_solve();

        void validate_reconstructed_fault_constitutive_state() const;

        /**
         * Commit constitutive histories for an accepted mechanical state.
         * Timestep zero retains the explicitly initialized histories.
         */
        void
        commit_reconstructed_fault_mechanical_history(
          const LinearAlgebra::BlockVector &accepted_bulk_state);

        /** Return the law-specific restriction based on committed fault state. */
        double
        compute_reconstructed_fault_time_step(const double cfl_number) const;

        /**
         * @}
         */

      private:
        friend class internal::PhaseFieldFaultTestAccess<dim>;

        /**
         * @name Maxwell constitutive helpers
         * @{
         */
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
                               const SymmetricTensor<2,dim> &old_stress);
        /**
         * @}
         */

        /**
         * @name Cohesive constitutive helpers
         * @{
         */
        /** Non-committing result of the common cohesive constitutive law. */
        struct CohesiveResponse
        {
          double cohesive_traction;
          double localization_factor;
          double history_correction;
          double crack_strain_rate;
        };

        /**
         * Evaluate T_coh(V), chi, and the exact history-corrected upsilon.
         */
        static CohesiveResponse
        compute_cohesive_response(const MaxwellCoefficients &coefficients,
                                  const double current_I_h,
                                  const double previous_I_h,
                                  const double previous_cohesive_traction,
                                  const double slip_rate,
                                  const double current_h,
                                  const double previous_h,
                                  const bool mature = false);

        /** Surface mixture and cohesive-profile values shared by surface and bulk points. */
        struct LocalizationResponse
        {
          std::vector<double> surface_material_fractions;
          double surface_temperature;
          double current_I_h;
          double previous_I_h;
          double previous_cohesive_traction;
          double current_degradation;
          double current_h;
          double previous_h;
        };

        LocalizationResponse
        evaluate_reconstructed_fault_localization(
          const unsigned int fault_index,
          const unsigned int segment_index,
          const double xi,
          const double phase_field,
          const double previous_phase_field,
          const std::string &context) const;
        /**
         * @}
         */

        /**
         * @name Initial cohesive-state setup
         * @{
         */
        /** Build and commit the initial cohesive state from H and the initial phase field. */
        void
        initialize_cohesive_state_from_initial_fields();

        /** Evaluate initial q for locally owned particles associated with a fault. */
        std::map<types::particle_index, double>
        evaluate_initial_cohesive_particle_values();

        /** Validate cohesive history before entering a terminal commit. */
        void
        validate_cohesive_state_commit(
          const std::vector<std::vector<double>> &cohesive_tractions) const;

        /** Commit cohesive history after validation without allocating or throwing. */
        void
        commit_cohesive_state(
          const std::vector<std::vector<double>> &cohesive_tractions) noexcept;

        /** Sample the frozen FE temperature at the reconstructed-fault vertices. */
        void compute_fault_surface_temperatures();

        /** Exact finite-step cohesive-work candidate before irreversibility. */
        static double
        compute_crack_driving_force_candidate(
          const double time_step,
          const MaxwellCoefficients &surface_coefficients,
          const double current_degradation,
          const double previous_h,
          const double current_cohesive_traction,
          const double previous_cohesive_traction);
        /**
         * @}
         */

        /**
         * @name Adaptive normalization-profile integration
         * @{
         */
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
          const NormalizationIntegrandEvaluator &integrand,
          double *mpi_seconds = nullptr);
        /**
         * @}
         */

        /**
         * @name Normalization-integral evaluation
         * @{
         */
        /** Compute or exactly reuse transient nodal I_h for the current state. */
        void
        compute_normalization_integrals();

        /** Discard transient value and lookup data after checkpoint loading. */
        void invalidate_normalization_cache();

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

        /** Project every chemical particle property to its scalar fault property. */
        void
        project_surface_chemical_compositions();

        /** Construct the balanced rank-owned set of normal profiles. */
        std::vector<NormalizationProfile>
        build_owned_normalization_profiles(const bool all_profiles = false) const;

        /** Integrate current phase values on cached, locally owned ray intervals. */
        std::vector<double>
        integrate_cell_normalization_profiles(
          const std::vector<NormalizationProfile> &profiles,
          const NormalizationIntegrandEvaluator &integrand,
          double &sampling_seconds,
          double &mpi_seconds);

        /** Evaluate the distributed Q1 phase field and local cell size at arbitrary points. */
        std::vector<NormalizationPointSample>
        evaluate_normalization_points(const std::vector<Point<dim>> &points) const;

        /** Consistently project owned profile integrals to replicated fault vertices. */
        void
        project_normalization_integrals_to_fault(
          const std::vector<NormalizationProfile> &profiles,
          const std::vector<double> &profile_integrals);

        /**
         * Internal empirical error-detection threshold for excessive raw
         * phase-field undershoot. This is not a physical parameter, a solver
         * tolerance, or a numerical convergence-control parameter.
         */
        static constexpr double normalization_phase_field_undershoot_tolerance = 1.e-4;
        /**
         * @}
         */

        /**
         * @name Material parameters and state
         * @{
         */
        double
        compute_creep_viscosity(const std::vector<double> &volume_fractions,
                                const double               temperature) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::FaultFriction<dim> fault_friction;

        unsigned int background_traction_property = numbers::invalid_unsigned_int;
        unsigned int background_shear_correction_property = numbers::invalid_unsigned_int;
        bool mature_frictional_fault = false;

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

        struct FaultPropertyIndices
        {
          unsigned int state = numbers::invalid_unsigned_int;
          unsigned int cohesive_traction = numbers::invalid_unsigned_int;
          unsigned int previous_normalization_integral =
            numbers::invalid_unsigned_int;
          std::vector<unsigned int> chemical_compositions;
        };

        FaultPropertyIndices fault_property_indices;

        std::vector<std::vector<double>> current_normalization_integrals;
        std::string boundary_normalization_completion_file;

        /** Non-checkpointed exact value-cache key. Owned phase entries are
         * compared directly (not hashed or inferred from the timestep). Every
         * rank must match; a failed preparation cannot leave a valid entry. */
        struct NormalizationValueCache
        {
          bool valid = false;
          IndexSet owned_phase_indices;
          std::vector<double> phase_values;
          std::vector<std::uint64_t> fault_versions;
          std::vector<Point<dim>> fault_vertices;
          std::vector<double> surface_compositions;
          std::uint64_t degradation_revision = 0;
          unsigned int hits = 0;
          unsigned int integrations = 0;
          unsigned long long last_requested_points = 0;
        };
        NormalizationValueCache normalization_value_cache;
        bool restore_frozen_normalization_after_restart = false;

        /** Geometry-only lookups for the batch sequence of the last I_h solve.
         * This cache holds no values. A changed batch or invalidated mesh forces
         * a collective rebuild; unused trailing batches are discarded. */
        struct NormalizationPointLookupCache
        {
          struct Batch
          {
            std::vector<Point<dim>> points;
            // Surviving lookup requests -> original adaptive-batch indices.
            std::vector<unsigned int> request_indices;
            std::unique_ptr<Utilities::MPI::RemotePointEvaluation<dim>> lookup;
          };

          const Batch &
          get(const GridTools::Cache<dim> &grid,
              const std::vector<Point<dim>> &points);

          std::vector<Batch> batches;
          unsigned int next_batch = 0;
          unsigned int hits = 0;
          unsigned int rebuilds = 0;
          bool rejection_supported = false;
          BoundingBox<dim> search_enclosure;
        };

        mutable NormalizationPointLookupCache normalization_point_lookups;

        /** Geometry only: phase/material changes do not invalidate ray traversals. */
        struct NormalizationCellCache
        {
          struct Interval
          {
            double lower, upper, diameter;
            Point<dim> reference_origin;
            Tensor<1,dim> reference_direction;
            std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell> phase_dofs;
          };
          struct Profile
          {
            Point<dim> origin;
            Tensor<1,dim> normal;
            std::vector<Interval> intervals;
          };
          bool mesh_changed = true;
          bool supported = false;
          BoundingBox<dim> physical_box;
          std::vector<Profile> profiles;
          boost::signals2::scoped_connection change_connection, create_connection;
        };
        NormalizationCellCache normalization_cell_cache;
        bool use_cell_normalization_profiles = false;

        std::vector<std::vector<double>> current_fault_surface_temperatures;

        double current_minimum_raw_normalization_phase_field =
          numbers::signaling_nan<double>();

        bool use_adiabatic_pressure_in_fault_friction = false;

        std::vector<typename ReconstructedFaultManager<dim>::
                    ParticleScalarProjectionDiagnostics>
          initial_cohesive_projection_diagnostics;

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;

        /** Rank-local timings: no synchronization, including during exceptions. */
        std::unique_ptr<TimerOutput> performance_timer;
        /**
         * @}
         */
    };

  }
}

#endif
